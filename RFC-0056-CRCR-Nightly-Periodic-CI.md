# CRCR Support for Nightly & Periodic CI

**Authors:**
* @groenenboomj
* @jewelkm89
* @subinz1

**Status:** Draft — for CRCR Working Group discussion

**Date:** June 2026

## Summary

Extend the Cross-Repository CI Relay (CRCR) to support nightly and periodic CI schedules for downstream repositories. Currently, CRCR only dispatches on `pull_request` and `push` events from `pytorch/pytorch`. This RFC proposes adding scheduled dispatch capabilities so downstream backends can report nightly/periodic CI results to the PyTorch HUD.

## Motivation

CRCR currently dispatches downstream CI on `pull_request` and `push` events from `pytorch/pytorch`. The webhook Lambda receives these GitHub webhook events, generates a `delivery_id`, sends `repository_dispatch` to all allowlisted downstream repos, and sets `DISPATCHED` in Redis. Downstream repos run their CI, then report results back via the callback Lambda, which validates the state machine (`DISPATCHED → IN_PROGRESS → COMPLETED`) and forwards metrics to HUD.

Nightly and periodic runs have no upstream trigger. They are cron-scheduled jobs (e.g., nightly builds against `main` HEAD, weekly compatibility tests against release branches). This creates two blockers:

1. **No dispatch.** Without an upstream webhook event, there is no `repository_dispatch` to downstream repos. Downstream nightly jobs would have to self-trigger via their own `schedule: cron`.

2. **No callback path.** The state machine rejects callbacks without a prior `DISPATCHED` record (HTTP 400: "no prior dispatch"). Even if a downstream repo runs a nightly job and tries to report results, the callback is rejected.

**Impact:** Downstream backends cannot report nightly/periodic CI results to HUD. This is a gap for L3/L4 backends that need to show nightly compatibility on `hud.pytorch.org/crcr`.

## Current Architecture

### Supported Events

The webhook Lambda accepts two GitHub event types (see `_SUPPORTED_EVENTS` in `webhook/lambda_function.py`):

| Event | Source | When |
|-------|--------|------|
| `pull_request` | GitHub webhook | PR opened, reopened, synchronize, closed |
| `push` | GitHub webhook | Push to any branch in `pytorch/pytorch` |

Both follow the same dispatch path:

```
GitHub webhook (pull_request or push)
    ↓
Webhook Lambda:
    1. Verify GitHub signature (X-Hub-Signature-256)
    2. Check event type ∈ {pull_request, push}
    3. Check repo == upstream_repo (pytorch/pytorch)
    4. Generate delivery_id from X-GitHub-Delivery header
    5. For each allowlisted downstream repo:
       - Mint GitHub App installation token
       - Send repository_dispatch(event_type, client_payload)
       - Set DISPATCHED state in Redis
    ↓
Downstream repo receives repository_dispatch
    ↓
Downstream workflow runs CI, calls callback action:
    - in_progress → Callback Lambda validates state, sets IN_PROGRESS in Redis
    - completed   → Callback Lambda validates state, sets COMPLETED in Redis
    ↓
Callback Lambda forwards trusted + untrusted payloads to HUD
    ↓
HUD → DynamoDB → ClickHouse → hud.pytorch.org/crcr
```

### Existing Precedent: EventBridge in the Callback Lambda

The callback Lambda already handles EventBridge-triggered events for zombie cleanup. When `event.source == "crcr.sweeper"`, it runs the cleanup handler instead of processing a callback. This is the exact pattern Option 2 would follow — an EventBridge cron rule invoking a Lambda handler.

```python
# callback/lambda_function.py (line 18-26, existing)
if event.get("source") == "crcr.sweeper":
    config = get_config()
    result = cleanup_handler.handle(config)
    ...
```

### Downstream Workflow Structure

Downstream repos listen for `repository_dispatch` with the event types they want:

```yaml
# Existing downstream pattern
on:
  repository_dispatch:
    types: [pull_request, push]  # L1 receiver already handles both
```

The `client_payload` always contains `event_type`, `delivery_id`, and the upstream webhook payload. Downstream workflows branch on `event_type` to extract PR number, SHA, ref, etc.

## Proposed Implementation

Two options are presented for WG discussion.

### Option 1: Upstream Cron Workflow in pytorch/pytorch → Webhook Lambda

Add a GitHub Actions workflow in `pytorch/pytorch` that runs on a `schedule: cron` trigger. This workflow constructs a synthetic webhook-like payload and POSTs it to the webhook Lambda's existing `/github/webhook` endpoint. The Lambda processes it like any other webhook — validates, dispatches to downstream repos, sets `DISPATCHED` in Redis.

```
pytorch/pytorch scheduled workflow (e.g., daily 00:00 UTC)
    ↓
1. Fetch main HEAD SHA (git ls-remote or checkout)
2. Construct synthetic payload mimicking a push/nightly event
3. POST to webhook Lambda endpoint with authentication
    ↓
Webhook Lambda:
    - Verify auth (new: OIDC or shared secret — NOT GitHub webhook signature)
    - Parse synthetic payload
    - event_type = "nightly"
    - Generate delivery_id
    - Call _dispatch_to_allowlist()
    ↓
(Same as existing dispatch from here)
```

**Upstream workflow:**

```yaml
# pytorch/pytorch/.github/workflows/crcr-nightly-dispatch.yml
name: CRCR Nightly Dispatch
on:
  schedule:
    - cron: '0 0 * * *'  # midnight UTC
  workflow_dispatch: {}   # manual trigger

jobs:
  dispatch:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - name: Get main HEAD SHA
        id: sha
        run: |
          SHA=$(git ls-remote https://github.com/pytorch/pytorch HEAD | cut -f1)
          echo "sha=$SHA" >> "$GITHUB_OUTPUT"

      - name: Trigger CRCR nightly dispatch
        run: |
          TOKEN=$(curl -s \
            -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
            "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=pytorch-cross-repo-ci-relay" \
            | jq -r .value)

          curl -X POST "${{ secrets.RELAY_WEBHOOK_URL }}" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $TOKEN" \
            -H "X-GitHub-Event: nightly" \
            -d '{
              "action": "nightly",
              "repository": {"full_name": "pytorch/pytorch"},
              "ref": "refs/heads/main",
              "after": "${{ steps.sha.outputs.sha }}"
            }'
```

**Downstream workflow change** (same for both options):

```yaml
on:
  repository_dispatch:
    types: [pull_request, push, nightly]
```

#### Option 1 — Pros

| # | Advantage | Detail |
|---|-----------|--------|
| 1 | No new AWS infrastructure | No EventBridge rule, no Terraform. Uses existing Lambda endpoint. |
| 2 | Visible in pytorch/pytorch | The nightly trigger appears in the upstream repo's Actions tab. PyTorch maintainers can see run history, logs, and failures. |
| 3 | Manual trigger | `workflow_dispatch` means anyone with repo write access can manually re-trigger a nightly dispatch from the GitHub UI. |
| 4 | Familiar | It's a GitHub Actions workflow. No AWS console, no Terraform, no new infrastructure concepts for contributors. |
| 5 | Schedule flexibility | Multiple cron entries or separate workflows can define different schedules without touching the relay. |
| 6 | PR-able changes | Changing the schedule is a PR to `pytorch/pytorch`, reviewed by maintainers with standard CI. |
| 7 | Decoupled from relay deployment | The schedule lives in a workflow file, not in infrastructure. Relay re-deployments don't affect the cron schedule. |

#### Option 1 — Cons

| # | Disadvantage | Detail |
|---|-------------|--------|
| 1 | Authentication complexity | The webhook Lambda currently validates requests using GitHub webhook signatures (`X-Hub-Signature-256`). A workflow-triggered POST doesn't have this signature. Requires adding a new auth path (OIDC or shared secret). |
| 2 | Not a real GitHub webhook | The synthetic payload mimics a webhook but isn't one. The Lambda's `_verify_signature` would reject it. Needs a new code path to parse and validate this non-standard input. |
| 3 | Requires changes in pytorch/pytorch | Adding a workflow to the upstream repo requires buy-in from PyTorch maintainers. |
| 4 | GitHub cron unreliability | GitHub's `schedule:` trigger is best-effort. During high-load periods, cron runs can be delayed by minutes to hours or skipped entirely. |
| 5 | Secret management | If using a shared secret (not OIDC), the secret must be stored in `pytorch/pytorch` repo secrets and rotated in sync with the Lambda config. |
| 6 | Dependency on pytorch/pytorch | Downstream nightly scheduling depends on an upstream workflow running. If Actions runners are down or quotas exhausted, all downstream nightlies stop. |
| 7 | Event handler branching | A `nightly` event type has neither a `pull_request` nor a standard `push` payload shape, requiring new parsing logic and a synthetic payload contract. |
| 8 | `_SUPPORTED_EVENTS` gating | The Lambda's `_SUPPORTED_EVENTS = {"pull_request", "push"}` rejects unknown event types before signature verification. Adding `nightly` means changing this gate and adding the corresponding auth path. |

#### Option 1 — Implementation Effort

| Component | Work | Effort |
|-----------|------|--------|
| pytorch/pytorch workflow | New `.github/workflows/crcr-nightly-dispatch.yml` (~40 LOC) | ~0.5 day |
| Webhook Lambda auth | New auth path: OIDC verification (port `jwt_helper` from callback Lambda) or shared-secret validation | ~1.5 days |
| Webhook Lambda handler | Expand `_SUPPORTED_EVENTS`, add nightly event parsing path, generate `delivery_id` | ~1 day |
| Downstream workflow | Add `nightly` to `repository_dispatch.types` | ~0.5 day |
| HUD view | Filter/view for `event_type != pull_request` on `/crcr/nightly` | ~1 day |
| Testing | End-to-end: workflow → Lambda → downstream → callback → HUD | ~1 day |
| Org approval | PR to pytorch/pytorch — maintainer review cycle | ~1-5 days |
| **Total** | | **~5.5-9.5 days** |

---

### Option 2: EventBridge Cron → Webhook Lambda

Add an AWS EventBridge rule (cron schedule) that invokes the webhook Lambda directly. The Lambda detects the EventBridge source (no HTTP request, no GitHub signature), fetches the current `main` HEAD SHA, generates a synthetic `delivery_id`, and dispatches to downstream repos using the existing `_dispatch_to_allowlist` machinery. The downstream flow is identical to a PR or push dispatch.

```
EventBridge cron (e.g., daily 00:00 UTC)
    ↓
Webhook Lambda (new handler path: event.source == "crcr.scheduler")
    ↓
1. Fetch pytorch/pytorch main HEAD SHA via GitHub API
2. Build synthetic client_payload:
   - event_type: "nightly" (or "periodic")
   - delivery_id: "nightly-{date}-{uuid}"
   - payload: {repository: {full_name: "pytorch/pytorch"}, head_sha: "..."}
3. Call _dispatch_to_allowlist() — reuses existing dispatch logic:
   - For each allowlisted repo: repository_dispatch + DISPATCHED in Redis
    ↓
Downstream repo receives repository_dispatch with event_type: nightly
    ↓
(Same as today: build → test → callback in_progress → callback completed)
    ↓
Callback Lambda → HUD → DynamoDB → ClickHouse
```

#### Option 2 — Pros

| # | Advantage | Detail |
|---|-----------|--------|
| 1 | Full pipeline reuse | State machine, callbacks, HUD ingestion, timing metrics — all work unchanged. Zero modifications to the callback Lambda, HUD API, or ClickHouse schema. |
| 2 | Minimal downstream changes | Downstream repos add one event type string. No new workflows, no new actions, no registration steps. |
| 3 | No auth changes | EventBridge invokes the Lambda directly (not over HTTP). No signature validation, no OIDC. The Lambda receives a structured AWS event, which is inherently trusted. |
| 4 | Proven pattern | The zombie cleanup handler already uses EventBridge → Lambda (`source: crcr.sweeper`). Same architecture. |
| 5 | Central control | The relay controls who gets nightly dispatches (allowlist-gated) and when they fire. |
| 6 | State machine integrity | Every nightly run has a real `DISPATCHED` record. No special-casing or bypasses. |
| 7 | Schedule reliability | EventBridge has an SLA of 99.99%. Far more reliable than GitHub's best-effort cron. |
| 8 | Consistent SHA | All downstream repos build against the same HEAD SHA fetched once at dispatch time. |
| 9 | No cross-repo dependency | Everything lives in `pytorch/test-infra`. No PR to `pytorch/pytorch`, no maintainer approval outside the CRCR team. |

#### Option 2 — Cons

| # | Disadvantage | Detail |
|---|-------------|--------|
| 1 | New infrastructure | Requires a new EventBridge rule and a new handler path in the webhook Lambda. Terraform + Lambda code to deploy and maintain. |
| 2 | Inflexible scheduling | All downstream repos get the same cron schedule. A backend that wants every-6-hours or weekly can't customize without additional EventBridge rules. |
| 3 | Single point of failure | If the EventBridge rule or Lambda errors, no downstream repos get their nightly dispatch. Requires CloudWatch monitoring/alerting. |
| 4 | No self-service trigger | Downstream repos can't trigger an ad-hoc nightly run. They must wait for the next scheduled dispatch or ask a relay admin. |
| 5 | SHA staleness | The HEAD SHA is fetched once. If `main` advances between dispatch and the downstream build starting, the tested SHA is already stale. |
| 6 | Allowlist complexity | Needs a way to opt repos in/out of nightly dispatches. Adds a new dimension to the allowlist config. |

#### Option 2 — Implementation Effort

| Component | Work | Effort |
|-----------|------|--------|
| EventBridge rule | Terraform: `aws_cloudwatch_event_rule` + `aws_cloudwatch_event_target` pointing to webhook Lambda | ~0.5 day |
| Webhook Lambda handler | New path for `source: crcr.scheduler`: fetch HEAD SHA, build synthetic payload, call `_dispatch_to_allowlist()`. ~100-150 LOC | ~1 day |
| Allowlist | Add `nightly: true/false` per-repo flag or a separate `nightly:` section | ~0.5 day |
| Downstream workflow | Add `nightly` to `repository_dispatch.types`, branch `run-name` on `event_type` | ~0.5 day |
| HUD view | Filter/view for `event_type != pull_request` on `/crcr/nightly` | ~1 day |
| Testing | End-to-end test with `TorchedHat/pytorch-redhat-ci` | ~0.5 day |
| **Total** | | **~4 days** |

## Side-by-Side Comparison

| Criteria | Option 2: EventBridge → Lambda | Option 1: pytorch/pytorch Cron → Lambda |
|----------|-------------------------------|----------------------------------------|
| New AWS infrastructure | Yes (EventBridge rule) | No |
| Changes to pytorch/pytorch | No | Yes (new workflow file) |
| Webhook Lambda changes | New handler path (~150 LOC) | New auth path + event parser (~250 LOC) |
| Callback Lambda changes | None | None |
| Auth changes | None (internal invoke) | Significant (OIDC or shared secret) |
| Downstream repo changes | Add `nightly` to dispatch types | Add `nightly` to dispatch types |
| State machine integrity | Full | Full (with new auth path) |
| Timing metrics | Full | Full |
| Central schedule control | Yes (Terraform) | No (workflow in upstream repo) |
| Manual re-trigger | Lambda console/CLI | Built-in (`workflow_dispatch`) |
| Schedule reliability | High (EventBridge 99.99% SLA) | Medium (GitHub cron best-effort) |
| Per-repo schedule flexibility | Low (single cron) | Medium (multiple workflows) |
| Operational visibility | CloudWatch logs/metrics | GitHub Actions run history |
| Org approval needed | No (test-infra only) | Yes (pytorch/pytorch PR) |
| Security surface change | None | New auth path in webhook Lambda |
| **Implementation effort** | **~4 days** | **~5.5-9.5 days** |

## Metrics

- **Nightly dispatch success rate**: Percentage of scheduled dispatches that successfully reach all allowlisted downstream repos.
- **Nightly callback completion rate**: Percentage of dispatched nightly runs that report back `COMPLETED` via callback.
- **HUD coverage**: Number of downstream backends with nightly results visible on `hud.pytorch.org/crcr`.
- **Time-to-detection**: How quickly a nightly regression in a downstream backend is surfaced on HUD.

## Drawbacks

- Adds complexity to the dispatch pipeline (new event type, new handler path).
- Nightly failures have no upstream PR to annotate — requires a separate notification mechanism.
- Concurrent nightly and PR-triggered runs may compete for downstream CI resources.
- Increases the surface area of the relay's responsibility.

## Alternatives

**Do nothing.** Downstream repos run nightly CI independently and report results in their own dashboards. PyTorch maintainers have no visibility into nightly downstream health. This is the current state.

**Downstream self-dispatch.** Each downstream repo configures its own `schedule: cron` and bypasses the state machine. This breaks the callback path (no `DISPATCHED` record) and requires per-repo configuration with no central control.

## Prior Art

- **CRCR Zombie Cleanup (EventBridge → Lambda)**: The existing `crcr.sweeper` pattern in the callback Lambda is the direct precedent for Option 2.
- **GitHub Actions scheduled workflows**: Widely used for nightly builds across the PyTorch ecosystem (e.g., `pytorch/pytorch` nightly builds, `pytorch/vision` nightly tests).
- **RFC-0050**: The original CRCR RFC that established the dispatch → callback → HUD pipeline for `pull_request` events.
- **RFC-0054**: HUD integration RFC that defined the ClickHouse schema and dashboard views for CRCR results.

## Feasibility Assessment

**Option 1 has real advantages for visibility and self-service.** `workflow_dispatch` is genuinely useful — being able to click "Run workflow" in the GitHub UI to trigger an ad-hoc nightly dispatch without AWS console access is valuable for debugging. The schedule living in a workflow file (not Terraform) means it's visible, grep-able, and PR-able by anyone with repo access. These advantages could justify the additional effort if the WG prioritizes visibility and self-service.

**Option 2 is lower effort and lower risk.** Auth is the key differentiator — Option 2 requires zero auth changes (EventBridge invokes the Lambda directly as an internal AWS call, inherently trusted), while Option 1 requires adding a second authentication mechanism to the webhook Lambda. Option 2 follows the exact same EventBridge → Lambda pattern already used for zombie cleanup. Everything stays in `pytorch/test-infra` with no cross-repo coordination.

## Unresolved Questions

### For WG Discussion

1. **Which option does the WG prefer?** Option 2 (lower effort, central control, no auth changes) or Option 1 (upstream visibility, self-service triggers, more flexible)?

2. **Schedule ownership:** Central cron (EventBridge or upstream workflow) or per-repo opt-in schedules?

3. **Scope:** Should nightly dispatches go to all allowlisted repos, or only repos that opt in via a config flag?

4. **SHA policy:** Always `main` HEAD, or allow per-repo target branches?

5. **Failure SLA:** Is a HUD view sufficient, or do we need active notifications (Slack, issues)?

6. **Manual triggers:** How important is the ability to manually re-trigger a nightly from the GitHub UI without AWS access?

7. **Periodic vs. nightly:** Do we need both `nightly` and `periodic` event types from day one, or start with `nightly` only and add `periodic` later?

### Shared Considerations (Both Options)

**HUD Changes.** The HUD currently groups results by `pr_number`. For nightly runs there is no PR. Use `pr_number = 0` as a sentinel for non-PR runs. The `event_type` field already flows through the pipeline — it just needs to carry `nightly` or `periodic` instead of `pull_request`. Add a filter/view on `hud.pytorch.org/crcr` for non-PR results, or a dedicated `/crcr/nightly` page.

**Allowlist Scoping.** Not all L1+ repos should receive nightly dispatches. Consider:

```yaml
L2:
  - org/repo:
      oncalls: user1, user2
      nightly: true    # opt-in to nightly dispatches
```

Or gate on allowlist level (e.g., only L3+ get nightlies by default).

**SHA Selection.** For nightly runs, which SHA to test against?
- `main` HEAD at dispatch time (most common, default)
- Latest release tag (for release compatibility testing)
- A specific branch (e.g., `release/2.x`)

**Failure Notifications.** PR failures are visible on the PR. Nightly failures have no PR to annotate.
- Dedicated Slack channel for CRCR nightly failures
- Auto-create GitHub issues on consecutive failures
- HUD dashboard alert on `/crcr/nightly`
- Email to repo oncalls from the allowlist

**Deduplication.** If a nightly dispatch happens while a PR-triggered run is in progress for the same downstream repo, the state machine handles them independently (different `delivery_id`). However, concurrent builds may compete for downstream CI resources. Consider whether the downstream workflow should use `concurrency:` groups to avoid parallel nightly + PR builds.

## Resolution

TBD — pending WG discussion.

### Level of Support

TBD

### Next Steps

TBD

#### Tracking Issue

TBD
