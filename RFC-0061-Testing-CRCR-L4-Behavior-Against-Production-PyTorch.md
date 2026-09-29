## Author

- @can-gaa-hou

## Abstract

[RFC-0050](RFC-0050-Cross-Repository-CI-Relay-for-PyTorch-Out-of-Tree-Backends.md) defines four participation levels for downstream repos of the Cross-Repository CI Relay (CRCR). `L4` is the only level whose Check Runs are created for every PyTorch PR and are allowed to block merges, so it is also the level that can do the most damage if it misbehaves. 

This RFC proposes how to do that with the existing canary repo [`pytorch/crcr-test`](https://github.com/pytorch/crcr-test):

1. Promote `pytorch/crcr-test` from allowlist level **L3 → L4** in the production CRCR relay.
2. Use a new, purpose-built workflow (`crcr-l4-ci.yml`), gated by a PR **label** (`ciflow/crcr/crcr-test-L4`), to validate that at L4 a Check Run is created on the upstream PR and can block merge, with effectively zero blast radius on other people's PRs.

## Inventory of `crcr-test` Workflows

The client-side trigger conditions below are evaluated by `crcr-test` itself and do not depend on the allowlist level. What changes when the relay moves `crcr-test` from L3 to L4 is only whether the Check Runs those jobs report become visible on (and able to block) the upstream PR.

| Workflow | Job | Client-side trigger condition | Check-run visibility after L4 | `@pytorchbot merge` impact after L4 |
|---|---|---|---|---|
| `crcr-dispatch-receiver.yml` | `cancel-workflow` | PR closed, not merged, no `Merged` label | n/a | n/a |
| | `push-deleted` | `push` event with `deleted==true` | n/a | n/a |
| | `l1-critical` (push branch) | any non-deleting `push` event | n/a (light mode; HUD / relay callback deliberately skipped to limit 429 risk) | n/a |
| | `l1-critical` (PR branch) | PR open **and** labeled `ciflow/crcr/crcr-test`, **or** PR closed + merged + unlabeled | Open + labeled: unchanged (already visible). Closed + merged + unlabeled: **becomes visible**, on the already-merged PR only | Open + labeled: a failure now blocks merge |
| `crcr-l2-ci.yml` | `cancel-workflow` | PR closed, not merged, no `Merged` label | n/a | n/a |
| | `l2-critical` | Same gate as `l1-critical` (PR branch) | Open + labeled: unchanged (already visible). Closed + merged + unlabeled: **becomes visible** on the merged PR | Same as `l1-critical` (PR branch) |
| `crcr-unit-tests.yml` | `validators` | `push` / `pull_request` / `workflow_dispatch` **on `crcr-test` itself**; not a `repository_dispatch` consumer at all | **No change whatsoever.** Irrelevant to `pytorch/pytorch` and to the allowlist level | n/a |
| `crcr-l3-ci.yml` | `cancel-workflow` | PR closed, not merged, no `Merged` label | n/a | n/a |
| | `test-L3-success-…`, `test-L3-xfail-…`, `test-L3-xcancel-…`, `test-L3-xtimeout-…`, `test-L3-matrix-…` (×20 legs) | Same label / merged gate as above | Open + labeled: unchanged (already visible). Closed + merged + unlabeled: **becomes visible** on the merged PR | Open + labeled: `xfail` / `xcancel` / `xtimeout` are designed to always fail / cancel / hang, so they **block merge** while the label is present |
| `crcr-l4-ci.yml` **(new)** | `cancel-workflow` | PR closed **and** labeled `ciflow/crcr/crcr-test-L4` | n/a | n/a |
| | `test-L4-success-…`, `test-L4-xfail-…`, `test-L4-xcancel-…`, `test-L4-xtimeout-…`, `test-L4-matrix-…` (×3 legs) | PR open **and** labeled `ciflow/crcr/crcr-test-L4` | **Always visible.** This is the workflow's entire purpose (previously always invisible under L3, since these PRs have no such label by design) | Same as the `test-L3-*` jobs: a PR labeled `ciflow/crcr/crcr-test-L4` cannot be merged with `@pytorchbot merge` once `test-L4-xfail-…` has run. This is by design, not a bug, and the test PR must never be merged |

## New Workflow: `crcr-l4-ci.yml`

`crcr-l4-ci.yml` has the same structure as `crcr-l3-ci.yml`, with one change: the job-level `if:` gate checks for the `ciflow/crcr/crcr-test-L4` label instead of `ciflow/crcr/crcr-test`.

```yaml
if: >-
  ${{ github.event.client_payload.payload.action != 'closed' &&
  contains(github.event.client_payload.payload.pull_request.labels.*.name,
  'ciflow/crcr/crcr-test-L4') }}
```

Scenarios covered (each a separate job, each producing its own Check Run once L4 is active):

| Job | Mechanism | Expected Check Run conclusion |
|---|---|---|
| `test-L4-success-…` | reports `completed` / `success` | `success` |
| `test-L4-xfail-…` | a step does `exit 1`; the final callback reports `${{ job.status }}` | `failure` |
| `test-L4-xcancel-…` | `timeout-minutes: 1` + `sleep 90` cancels just this job; the final callback reports `${{ job.status }}` | `cancelled` |
| `test-L4-xtimeout-…` | reports `in_progress` and deliberately never reports `completed` | `timed_out`, set by the relay's zombie sweeper after the real `ZOMBIE_TIMEOUT_SECONDS` (default 6 h), not a faked conclusion |
| `test-L4-matrix-…` | 3-leg matrix, each leg gets a distinct `job-name` | 3 independent `success` Check Runs |

## Rollout Plan

1. Add `crcr-l4-ci.yml` to the `crcr-test` workflows.
2. Land the allowlist change (`crcr-test`: L3 → L4).
3. Create the `ciflow/crcr/crcr-test-L4` label.
4. Open one dedicated test PR against `pytorch/pytorch`, labeled with `ciflow/crcr/crcr-test-L4`. **Only** the jobs in `crcr-l4-ci.yml` should be triggered.
5. Try merging the PR with `@pytorchbot merge` and expect a failure.

## Alternatives

- **PR title prefix instead of a label.** [@KarhouTam](https://github.com/KarhouTam) suggested gating on a title prefix such as `[CRCR-TEST]`, on the grounds that nobody would name a real PR that way. A title needs no permission to set, whereas per RFC-0050 the `ciflow/crcr/<name>` label permission is already enforced by the existing pytorchbot mechanism, so a label keeps unauthorized authors from opting in.

## Reference

- [pytorch/test-infra#8312](https://github.com/pytorch/test-infra/issues/8312): CRCR sync action item to define how to test the L4 system
- [pytorch/crcr-test#16](https://github.com/pytorch/crcr-test/issues/16): original proposal this RFC was migrated from
- [RFC-0050](RFC-0050-Cross-Repository-CI-Relay-for-PyTorch-Out-of-Tree-Backends.md): CRCR design, levels and evolution path
