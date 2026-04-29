## Author(s)

- @groenenboomj
- @jewelkm89
- @subinz1
- @NayanNagabhushana-28

## Abstract

PyTorch HUD (`hud.pytorch.org`) currently only ingests test results from in-tree CI workflows. Out-of-tree (OOT) backends — custom accelerators, downstream device integrations — have no supported path to surface their CI results on the dashboard.

This RFC defines the **HUD ingestion and display layer** for OOT CI results. It builds on the cross-repository CI relay system (RFC-0050) which handles event dispatch and downstream triggering. This RFC covers:

- The **write path**: how downstream CI results flow from the relay's Result Handler into DynamoDB and ClickHouse
- The **read path**: three new HUD pages that display OOT CI health
- **Storage schemas**: DynamoDB table and ClickHouse table designs
- **DB protection**: rate limiting and payload caps
- **Security**: authentication at each hop, with a proposal for signed callback tokens

The complete pipeline is: Downstream CI → Result Handler → HUD API → DynamoDB → DynamoDB Stream → Replicator → ClickHouse.

> [!NOTE]
> - Artifact storage (logs, test reports, full results) is **owned and managed by each downstream organization**. HUD only stores URLs and links to externally-hosted artifacts.
> - This design follows the DynamoDB → ClickHouse replication pattern prescribed by PyTorch infrastructure maintainers.

## Motivation

With PyTorch's growing ecosystem of OOT backends (Intel XPU, Huawei Ascend, custom accelerators, etc.), there is increasing need for visibility into how upstream changes affect downstream projects. The relay system (RFC-0050) solves the *dispatch* problem — triggering downstream CI when a PyTorch PR is opened. But dispatch alone is not enough: results need to flow back and be displayed where maintainers and contributors can see them.

Without a standardized HUD integration:

- **Downstream maintainers** have no central place to monitor the health of their backend against PyTorch trunk.
- **PyTorch maintainers** have no visibility into whether a PR breaks OOT backends, even at the informational L2 level.
- **PR authors** cannot see OOT CI results alongside their in-tree checks without visiting each downstream repo individually.

This RFC fills that gap by defining how OOT results are ingested, stored, and displayed on HUD.

## Proposed Implementation

### Architecture Overview

The system has four components on the write path and three views on the read path.

**Write path:**

1. **Result Handler** (existing, from relay): receives callbacks from downstream CI, verifies OIDC, checks allowlist, forwards to HUD
2. **HUD API** (`/api/oot/results`): validates auth and schema, enforces payload caps, writes to DynamoDB
3. **DynamoDB** (`torchci-oot-workflow-job`): mutable storage supporting the two-callback model (`in_progress` → `completed`)
4. **ClickHouse** (`default.oot_workflow_job`): analytical storage, replicated automatically via the existing `clickhouse-replicator-dynamo` replicator

**Read path:**

1. **Global OOT Summary** (`/oot`): cross-repo health overview for CI maintainers
2. **Per-Backend Dashboard** (`/oot/[org]/[repo]`): detailed job grid for a single downstream repo
3. **PR View Integration** (`/pr/[number]`): collapsible OOT section on existing PR pages

```mermaid
flowchart TD
    subgraph trigger [Trigger]
        PR["PR opened in\npytorch/pytorch"]
        PR --> WH[webhook_handler]
        WH --> DS["Downstream CI\n(OOT backend)"]
    end

    subgraph writePath [Write Path]
        subgraph callbacks [Status Callbacks + Artifact URLs]
            DS -->|"First job starts"| CB1["POST in_progress\nto Result Handler"]
            DS -->|"Last job finishes"| CB2["POST completed\n+ artifact URLs\nto Result Handler"]
        end

        subgraph relay [Relay Server]
            CB1 --> RH[result_handler]
            CB2 --> RH
            RH -->|"Verify OIDC\nCheck allowlist\nRate limit"| FWD["Forward to\nHUD API"]
        end

        subgraph hud [HUD]
            FWD -->|"X-Hud-Internal-Bot"| API["HUD API\n/api/oot/results"]
            API -->|"Validate + write"| DDB[("DynamoDB\ntorchci-oot-workflow-job")]
        end

        subgraph replicator [Replicator]
            DDB --> STREAM["DynamoDB Stream"]
            STREAM --> REP["clickhouse-replicator-dynamo"]
            REP -->|"completed only"| CH[("ClickHouse\ndefault.oot_workflow_job")]
        end
    end

    subgraph artifacts [Artifact Storage - Downstream-Owned]
        DS --> ARTUP["Upload artifacts to\ndownstream-owned storage"]
        ARTUP --> STORE[("Downstream Storage\n(org-managed)")]
    end

    subgraph readPath [Read Path - HUD Frontend]
        CH --> DASH["Dashboard queries\n(workflow status, pass rates)"]
        STORE -.->|"External links\n(from artifact_url in CH)"| HUDFE
        DASH --> HUDFE["hud.pytorch.org\n/oot  /oot/org/repo  /pr/N"]
    end
```

### Data Flow Summary

| Phase | Source | Destination | Auth | What Happens |
|-------|--------|-------------|------|--------------|
| **Callback 1 (start)** | First downstream job | Result Handler | OIDC token | POST `in_progress` status |
| | Result Handler | HUD API | `X-Hud-Internal-Bot` | Forward `in_progress` payload |
| | HUD API | DynamoDB | Service role | PutItem with `in_progress` status (DynamoDB only, not replicated to CH) |
| **Callback 2 (end)** | Last downstream job | Result Handler | OIDC token | POST `completed` status + test counts + failures + artifact URLs |
| | Result Handler | HUD API | `X-Hud-Internal-Bot` | Forward full payload |
| | HUD API | DynamoDB | Service role | PutItem (overwrite) with completed data |
| | DynamoDB | ClickHouse | Stream + replicator | Replicated to ClickHouse (completed records only) |
| **Read** | HUD Frontend | ClickHouse | Read-only | Dashboard queries: status, pass rates, durations |
| | HUD Frontend | Downstream storage | Public URL | On-demand: external link to logs, full test results |

### Write Path

#### Two-Callback Model

Following the relay's L2 design, each downstream workflow sends two callbacks:

**Callback 1 — "In Progress"** (from the first job in the downstream workflow):
- Minimal payload: `downstream_repo`, `pr_number`, `pytorch_head_sha`, `workflow_run_id`, `status: "in_progress"`, `started_at`
- Written to DynamoDB only — used for mutable status tracking
- Not replicated to ClickHouse (no analytical value until completed)

**Callback 2 — "Completed"** (from the last job in the downstream workflow):
- Full payload: everything from Callback 1, plus `conclusion`, `completed_at`, test summary counts, failed test details (as JSON array), artifact URLs (pointing to downstream-owned storage), environment metadata
- Written to DynamoDB (overwrites the `in_progress` record), then replicated to ClickHouse
- HUD dashboards query ClickHouse for completed results

The two-callback model is why DynamoDB is the write target: the status changes from `in_progress` to `completed`, requiring an upsert. Only `completed` records are replicated to ClickHouse for dashboard queries.

#### Artifact Storage (Downstream-Owned)

Each downstream organization owns and manages its artifact storage. PyTorch infrastructure does not provision, host, or manage any storage for OOT backend artifacts.

Downstream can use any publicly accessible storage: cloud object storage, GitHub Actions native artifacts, or any other URL-accessible location.

The only requirement is that the downstream provides **publicly accessible URLs** in the "completed" callback payload. HUD stores these URLs and renders them as external links.

This ensures:
- No PyTorch infra cost for artifact storage
- No access control complexity across N downstream orgs
- Full flexibility — downstream teams choose storage that fits their existing infra

#### Hop 1: Downstream → Result Handler

| Step | Action | Failure Response |
|------|--------|------------------|
| 1 | Verify OIDC token signature against GitHub JWKS | `401 Unauthorized` |
| 2 | Check `repository` claim against cached allowlist | `403 Forbidden` |
| 3 | Verify repo is authorized at L2 or above | `403 Forbidden` |
| 4 | Per-repo rate limit check | `429 Too Many Requests` |
| 5 | Forward validated payload to HUD API | — |

The relay's Result Handler receives the callback and produces a `{trusted, untrusted}` payload:

- `trusted` — relay-generated fields: `verified_repo` (OIDC-proven identity) and `ci_metrics` (relay-measured `queue_time`, `execution_time`)
- `untrusted` — downstream-reported data: `callback_payload` (the full callback body, passed through verbatim)

HUD always prefers `trusted.verified_repo` over anything self-reported in the body.

#### Hop 2: Result Handler → HUD API

| Step | Action | Failure Response |
|------|--------|------------------|
| 1 | Validate `X-Hud-Internal-Bot` header | `401 Unauthorized` |
| 2 | Schema validation + payload caps (2MB max body) | `400 Bad Request` |
| 3 | Write to DynamoDB | `502 Bad Gateway` |

The HUD API endpoint (`torchci/pages/api/oot/results.ts`) follows the existing `webhookToDynamo` pattern:

```typescript
import type { NextApiRequest, NextApiResponse } from "next";
import { checkAuthWithApiToken } from "lib/auth/auth";
import {
  ApiError,
  validatePayloadSize,
  validateRelayPayload,
  extractDynamoRecord,
  writeToDynamo,
} from "lib/oot/ootUtils";

export const config = {
  api: {
    bodyParser: {
      sizeLimit: "2mb",
    },
  },
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const auth = await checkAuthWithApiToken(req, res);
    if (!auth.ok) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    const rawBody =
      typeof req.body === "string" ? req.body : JSON.stringify(req.body);
    validatePayloadSize(rawBody);

    const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
    const payload = validateRelayPayload(body);

    const record = extractDynamoRecord(payload);
    await writeToDynamo(record);

    return res.status(200).json({
      ok: true,
      status: record.status,
      dynamoKey: record.dynamoKey,
    });
  } catch (err: any) {
    if (err instanceof ApiError) {
      return res.status(err.statusCode).json({ error: err.message });
    }
    console.error("OOT results handler error:", err);
    return res.status(502).json({ error: "Internal error writing to DynamoDB" });
  }
}
```

#### Payload Extraction

The relay sends a `{trusted, untrusted}` payload. The HUD utility library extracts and flattens fields into a DynamoDB record:

```typescript
export interface RelayPayload {
  trusted: {
    verified_repo: string;
    ci_metrics?: {
      queue_time?: number | null;
      execution_time?: number | null;
    };
  };
  untrusted: {
    callback_payload: {
      event_type: string;
      delivery_id: string;
      payload: {
        pull_request?: { number: number; head?: { sha: string } };
        repository?: { full_name: string };
      };
      workflow: {
        status: string;
        conclusion?: string | null;
        name: string;
        url: string;
        test_results?: {
          total?: number;
          passed?: number;
          failed?: number;
          skipped?: number;
          failures?: any[];
        };
      };
    };
  };
}
```

The `extractDynamoRecord` function flattens this into a single record:

```typescript
export function extractDynamoRecord(payload: RelayPayload): OotWorkflowJobRecord {
  const { trusted, untrusted } = payload;
  const cb = untrusted.callback_payload;
  const wf = cb.workflow;
  const pr = cb.payload?.pull_request;

  const dynamoKey = `${trusted.verified_repo}/${cb.delivery_id}/${wf.name}`;
  const now = new Date().toISOString();

  const record: OotWorkflowJobRecord = {
    dynamoKey,
    status: wf.status,
    downstream_repo: trusted.verified_repo,
    upstream_repo: cb.payload?.repository?.full_name ?? "pytorch/pytorch",
    pr_number: pr?.number ?? 0,
    pytorch_head_sha: pr?.head?.sha ?? "",
    delivery_id: cb.delivery_id,
    workflow_run_url: wf.url ?? "",
    workflow_name: wf.name,
    queue_time: trusted.ci_metrics?.queue_time,
    execution_time: trusted.ci_metrics?.execution_time,
    started_at: now,
  };

  if (wf.status === "completed") {
    record.conclusion = wf.conclusion ?? undefined;
    record.completed_at = now;
    if (wf.test_results) {
      const tr = wf.test_results;
      if (typeof tr.total === "number") record.total_tests = tr.total;
      if (typeof tr.passed === "number") record.passed_tests = tr.passed;
      if (typeof tr.failed === "number") record.failed_tests = tr.failed;
      if (typeof tr.skipped === "number") record.skipped_tests = tr.skipped;
      if (tr.failures) {
        record.failed_tests_json = JSON.stringify(tr.failures);
      }
    }
  }

  return record;
}
```

#### Payload Validation

The utility library validates the incoming relay payload structure:

```typescript
export function validateRelayPayload(body: any): RelayPayload {
  if (!body?.trusted?.verified_repo) {
    throw new ApiError(400, "Missing trusted.verified_repo");
  }
  const cb = body?.untrusted?.callback_payload;
  if (!cb) {
    throw new ApiError(400, "Missing untrusted.callback_payload");
  }
  if (!cb.delivery_id) {
    throw new ApiError(400, "Missing delivery_id");
  }
  if (!cb.workflow?.status) {
    throw new ApiError(400, "Missing workflow.status");
  }
  if (cb.workflow.status !== "in_progress" && cb.workflow.status !== "completed") {
    throw new ApiError(400, `Invalid workflow.status: must be "in_progress" or "completed".`);
  }
  if (cb.workflow.status === "completed" && !cb.workflow.conclusion) {
    throw new ApiError(400, "workflow.conclusion is required when status is completed");
  }
  return body as RelayPayload;
}
```

#### Hop 3: HUD API → DynamoDB

The HUD API writes the flattened workflow job record to DynamoDB:

```
dynamoKey = `${trusted.verified_repo}/${delivery_id}/${workflow_name}`
```

- For "in_progress" callbacks: creates the record with minimal fields + `status: "in_progress"`
- For "completed" callbacks: overwrites the record with full payload including `conclusion`, test counts, failed test details (as JSON string), artifact URL, and environment metadata

#### Hop 4: DynamoDB → ClickHouse (Automatic)

This hop requires **no new code**. The existing infrastructure handles it:

1. **DynamoDB Streams** (enabled on the table with `NEW_AND_OLD_IMAGES`) captures every write
2. **`clickhouse-replicator-dynamo`** receives stream events and inserts `completed` records into ClickHouse (`in_progress` records are filtered out — they serve only as mutable state in DynamoDB)
3. **Replicator mapping** (1 new line): `"torchci-oot-workflow-job": "default.oot_workflow_job"` added to `SUPPORTED_TABLES` in `lambda_function.py`

### Storage Design

#### DynamoDB Table

Table: `torchci-oot-workflow-job`

| Field | Type | Description |
|-------|------|-------------|
| `dynamoKey` (hash key) | String | `{repo}/{delivery_id}/{workflow_name}` |
| `status` | String | `in_progress` or `completed` |
| `downstream_repo` | String | Downstream repo `org/name` (from `trusted.verified_repo`) |
| `upstream_repo` | String | Upstream repo (e.g. `pytorch/pytorch`) |
| `pr_number` | Number | PyTorch PR number |
| `pytorch_head_sha` | String | PyTorch PR commit SHA |
| `delivery_id` | String | GitHub webhook delivery ID from L1 dispatch |
| `workflow_run_url` | String | Link to downstream GHA workflow run |
| `workflow_name` | String | Downstream workflow name |
| `conclusion` | String | `success`, `failure`, `cancelled`, `timed_out` (set on "completed") |
| `queue_time` | Number | Relay-measured dispatch-to-in_progress time in seconds |
| `execution_time` | Number | Relay-measured in_progress-to-completed time in seconds |
| `started_at` | String | ISO 8601 timestamp |
| `completed_at` | String | ISO 8601 timestamp (set on "completed") |
| `total_tests` | Number | Total test count (set on "completed") |
| `passed_tests` | Number | Passed test count |
| `failed_tests` | Number | Failed test count |
| `skipped_tests` | Number | Skipped test count |
| `failed_tests_json` | String | JSON array of failed/errored test details |
| `artifact_url` | String | URL to downstream-hosted artifacts |
| `environment` | String | JSON: `{"cuda": "12.8", "device": "H100", ...}` |
| `downstream_repo_level` | String | Repo's relay level: `L2`, `L3`, `L4` |

Table configuration:
- Partition key: `dynamoKey` (String)
- Billing: on-demand (pay per request)
- Streams: enabled with `NEW_AND_OLD_IMAGES`

#### ClickHouse Table

Table: `default.oot_workflow_job`

Based on the existing `default.workflow_job` schema with OOT-specific additions:

```sql
CREATE TABLE default.oot_workflow_job
(
    `dynamoKey` String,
    `status` String,
    `downstream_repo` String COMMENT 'Downstream repo org/name, from trusted.verified_repo',
    `upstream_repo` String COMMENT 'Upstream repo, typically pytorch/pytorch',
    `pr_number` UInt64 COMMENT 'PyTorch PR number',
    `pytorch_head_sha` String COMMENT 'PyTorch PR commit SHA',
    `delivery_id` String COMMENT 'GitHub webhook delivery ID from L1 dispatch',
    `workflow_run_url` String COMMENT 'Link to downstream GHA workflow run',
    `workflow_name` String COMMENT 'Downstream workflow name',
    `conclusion` String COMMENT 'success, failure, cancelled, timed_out (set on completed)',
    `queue_time` Nullable(Float64) COMMENT 'Relay-measured dispatch-to-in_progress time in seconds',
    `execution_time` Nullable(Float64) COMMENT 'Relay-measured in_progress-to-completed time in seconds',
    `started_at` DateTime64(9) COMMENT 'ISO 8601 timestamp when record was created',
    `completed_at` DateTime64(9) COMMENT 'ISO 8601 timestamp when job completed',
    `total_tests` UInt64 DEFAULT 0,
    `passed_tests` UInt64 DEFAULT 0,
    `failed_tests` UInt64 DEFAULT 0,
    `skipped_tests` UInt64 DEFAULT 0,
    `failed_tests_json` String DEFAULT '' COMMENT 'JSON array of failed/errored test details',
    `artifact_url` String DEFAULT '' COMMENT 'URL to downstream-hosted artifacts (logs, reports)',
    `environment` String DEFAULT '' COMMENT 'JSON: {"cuda": "12.8", "device": "H100", ...}',
    `downstream_repo_level` String DEFAULT '' COMMENT 'Relay level at dispatch time: L2, L3, L4',
    `_inserted_at` DateTime MATERIALIZED now(),
    `repository_full_name` String ALIAS downstream_repo COMMENT 'Alias for consistency with workflow_job queries',
    `duration_seconds` Float64 ALIAS if(completed_at = toDateTime64(0, 9), 0, dateDiff(second, started_at, completed_at)),
    INDEX status_index status TYPE bloom_filter GRANULARITY 1,
    INDEX started_at_index started_at TYPE minmax GRANULARITY 1,
    INDEX completed_at_index completed_at TYPE minmax GRANULARITY 1,
    INDEX pr_number_index pr_number TYPE bloom_filter GRANULARITY 1,
    INDEX downstream_repo_index downstream_repo TYPE bloom_filter GRANULARITY 1
)
ENGINE = SharedReplacingMergeTree('/clickhouse/tables/{uuid}/{shard}', '{replica}')
ORDER BY (downstream_repo, delivery_id, dynamoKey)
SETTINGS index_granularity = 8192
```

Key design decisions:
- **`SharedReplacingMergeTree`** for upsert semantics: when a "completed" callback arrives, it replaces the "in_progress" row for the same `dynamoKey`
- **`repository_full_name` as ALIAS**: provides naming consistency with existing `workflow_job` queries without storing redundant data
- **`duration_seconds` as computed ALIAS**: avoids manual duration calculation
- **Bloom filter indexes** on frequently filtered columns for efficient dashboard queries

#### Data Retention

| Storage | Retention | Mechanism |
|---------|-----------|-----------|
| DynamoDB | Indefinite (low volume) | DynamoDB TTL can be added if needed |
| ClickHouse | Follows `workflow_job` table retention | Same TTL/partition dropping as in-tree |
| Downstream artifacts | Downstream org's policy | Managed by downstream org |

### DB Protection Layer

#### Rate Limiting

**At the relay (per-repo)**

```
Key:    oot:rate:{repo}
Value:  counter (atomic INCR)
TTL:    1 minute sliding window
Limit:  10 requests/minute per repo
```

Rejects at the relay before any HUD/DB traffic. First line of defense against runaway CI loops.

#### Payload Caps

| Limit | Value | Enforced at |
|-------|-------|-------------|
| Max request body | 2 MB | HUD API (`bodyParser.sizeLimit`) |
| Max `failed_tests_json` entries | 1,000 per request | HUD API schema validation |
| Max `stacktrace` length | 4 KB per test | HUD API (truncate before write) |
| Max `message` length | 1 KB per test | HUD API (truncate before write) |

### Authentication Flow

| Hop | From → To | Auth Mechanism | Credential Scope |
|-----|-----------|----------------|------------------|
| 1 | Downstream → Result Handler | OIDC token (GHA-issued, 5 min TTL) | No secrets needed by downstream |
| 2 | Result Handler → HUD API | `X-Hud-Internal-Bot` header | Same pattern as DrCI / trymerge |
| 3 | HUD API → DynamoDB | Service role | Write to `torchci-oot-workflow-job` only |
| 4 | DynamoDB → ClickHouse | Replicator service role | Existing replicator credentials |

Security properties:

| Property | How |
|----------|-----|
| Downstream never gets DB/HUD credentials | OIDC only |
| Unknown repos rejected before HUD | Allowlist at relay (cached) |
| HUD only accepts relay traffic | `X-Hud-Internal-Bot` header |
| Runaway CI caught early | Rate limit at relay |
| DB overload prevented | Payload caps |
| Artifact storage not PyTorch's burden | Each downstream org manages their own |

### Security Design

#### OIDC Authentication

The relay's callback security model is based on GitHub Actions OIDC:

1. Downstream workflow declares `permissions: id-token: write`
2. GitHub's OIDC provider mints a JWT signed with RS256 (5-minute TTL)
3. Token contains claims: `repository`, `actor`, `ref`, `run_id`
4. Result Handler verifies the signature against GitHub's public JWKS
5. Extracts `repository` claim as `verified_repo` — the only cryptographically trusted identity

**What OIDC gives us:**
- Zero secret management — downstream never handles API keys
- Caller identity — cryptographic proof of which repo is calling
- Short-lived — 5-minute TTL, auto-expires

**What OIDC does NOT give us:**
- No dispatch verification — proves *who* is calling, not *whether they were asked to call*
- No replay protection — a new OIDC token can be minted at any time
- No scope-locking — doesn't bind the callback to a specific PR or SHA

#### Trusted/Untrusted Payload Split

The relay separates the forwarded payload into two namespaces:

- **`trusted`** — relay-generated: `verified_repo` (OIDC-proven) and `ci_metrics` (relay-measured timing)
- **`untrusted`** — downstream-reported: `callback_payload` (passed through verbatim)

HUD always prefers `trusted.verified_repo` over anything self-reported.

#### Error Handling Strategy

HUD API errors are handled with the following strategy:

- **4xx errors** (validation, auth failures): propagated back to downstream so workflow authors see a red CI step and can fix their payload
- **5xx and network failures**: the Result Handler retries N times, then returns the status to downstream:
  - `"delivered"` — HUD received the data
  - `"hud_rejected"` — HUD returned 4xx (validation failure)
  - `"hud_unavailable"` — all retries failed (HUD outage)
  - `"skipped"` — no HUD URL configured

This ensures a HUD outage does not turn every downstream L2 CI red, while still giving downstream visibility into the HUD push status.

#### Proposal: Signed One-Shot Callback Token

To close gaps in dispatch provenance and replay prevention, we propose adding a signed callback token minted by L1 at dispatch time:

**How it works:**

```
L1 Dispatch:
  1. Mint JWT: sign({delivery_id, repo, pr_number, head_sha, dispatched_at}, SECRET)
  2. Store one-shot key in Redis: crcr:token:{delivery_id}:{repo} = valid (TTL 3 days)
  3. Include token in client_payload → downstream receives it

L2 Callback:
  1. Downstream echoes callback_token in the callback body
  2. Result Handler verifies JWT signature (proves L1 minted it)
  3. Checks Redis key exists (proves token hasn't been consumed)
  4. Validates delivery_id + repo in token match OIDC verified_repo
  5. On "completed" callback: delete Redis key (one-shot, consumed)
  6. Forward to HUD
```

**What the callback token adds:**

| Concern | OIDC Only | OIDC + Callback Token |
|---------|-----------|----------------------|
| Dispatch provenance | Can't verify | Signed proof |
| Replay prevention | Vulnerable | One-shot Redis key |
| Fabrication blocking | Identity check only | Token locks to specific PR/SHA |
| Orphaned job detection | No way to know | Unconsumed Redis key = orphaned |
| Queue time (trusted) | Redis lookup, can miss | Embedded in token, tamper-proof |
| Per-dispatch rate limiting | Per-repo only | Exactly 2 callbacks per dispatch |
| L3/L4 merge gating | Unsafe (self-reported) | Cryptographic proof of dispatch |

> [!NOTE]
> The callback token is not strictly required for L2 (dashboard only). However, L1 is where the token gets minted, and once L1 is deployed and downstream repos depend on the current `client_payload` shape, retrofitting it becomes a breaking change. We recommend adding it now as optional groundwork for L3/L4.

#### State Machine for Status Transitions

To prevent rollback and replay attacks at the relay level, we propose a Redis-backed state machine for status transitions:

| Transition | Action |
|------------|--------|
| No record → `in_progress` | Accepted |
| No record → `completed` | Accepted (handles missed `in_progress` callback) |
| `in_progress` → `completed` | Accepted (normal flow) |
| `completed` → `in_progress` | **Rejected** (no rollback) |
| Duplicate `completed` | **Rejected** (no replay) |

The state key includes `run_id` to support legitimate workflow reruns:

```
Key: crcr:state:{delivery_id}:{downstream_repo}:{run_id}
TTL: 3 days (matching relay Redis TTL)
```

### Read Path: HUD Pages

#### Page 1: Global OOT Summary — `/oot`

Cross-repo overview for CI maintainers. Displays a table of all OOT backend repositories sorted by pass rate (worst first), with columns for pass rate, success/failure counts, average duration, and last run time. Includes a time range selector (24h / 7d / 30d).

Clicking a row navigates to the per-backend dashboard.

ClickHouse saved query (`oot_summary`):

```sql
SELECT
    downstream_repo AS repo,
    countIf(conclusion = 'success') AS successes,
    countIf(conclusion = 'failure') AS failures,
    count() AS total,
    if(total > 0, successes / total, 0) AS pass_rate,
    avg(duration_seconds) AS avg_duration_s,
    max(started_at) AS last_run
FROM
    default.oot_workflow_job FINAL
WHERE
    started_at > now() - INTERVAL {days: UInt64} DAY
    AND status = 'completed'
GROUP BY
    repo
ORDER BY
    pass_rate ASC
```

Frontend implementation uses `useSWR` with the ClickHouse query API and renders a Material-UI table with color-coded pass rate chips (green ≥ 95%, yellow ≥ 80%, red < 80%).

#### Page 2: Per-Backend Dashboard — `/oot/[org]/[repo]`

Detailed CI health view for a single downstream backend. Displays:

- **Header**: repo name, overall health (pass rate), environment summary
- **Matrix view**: rows = PyTorch PRs, columns = OOT jobs. Color-coded status chips: green = `success`, red = `failure`, yellow = `cancelled`/`timed_out`, blue = `in_progress`
- **Failure drill-down**: click a red cell to see failed test details (parsed from `failed_tests_json`)
- **External links**: "View artifacts" → downstream-hosted logs; "View workflow run" → downstream GHA run URL

ClickHouse saved query (`oot_backend_dashboard`):

```sql
SELECT
    pr_number,
    pytorch_head_sha,
    workflow_name AS job_name,
    status,
    conclusion,
    started_at,
    completed_at,
    duration_seconds,
    total_tests,
    passed_tests,
    failed_tests,
    skipped_tests,
    workflow_run_url,
    artifact_url,
    queue_time,
    execution_time
FROM
    default.oot_workflow_job FINAL
WHERE
    downstream_repo = {repo: String}
    AND started_at > now() - INTERVAL {days: UInt64} DAY
ORDER BY
    started_at DESC
LIMIT 500
```

#### Page 3: PR View Integration — `/pr/[number]`

A collapsible "Out-of-Tree Backends" accordion added to the existing PR detail page. Only rendered when OOT results exist for the PR. Shows:

- Backend name, job name, status chip (color-coded), duration, links to run/artifacts
- Summary in the accordion header: `"3/4 passed, 1 running"`
- `in_progress` runs show a spinner-style "running" chip

ClickHouse saved query (`oot_pr_results`):

```sql
SELECT
    downstream_repo,
    workflow_name AS job_name,
    status,
    conclusion,
    duration_seconds,
    workflow_run_url,
    artifact_url,
    started_at,
    queue_time,
    execution_time
FROM
    default.oot_workflow_job FINAL
WHERE
    pr_number = {pr: UInt64}
ORDER BY
    downstream_repo, started_at DESC
```

The `OotPrSection` React component is integrated into the existing PR page (`torchci/pages/[repoOwner]/[repoName]/pull/[prNumber].tsx`) below the existing `CommitInfo` section.

### Sample Payloads

#### In-Progress Callback

```json
{
  "status": "in_progress",
  "head_sha": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
  "pr_number": 179565,
  "downstream_repo": "{org}/{repo}",
  "upstream_repo": "pytorch/pytorch",
  "workflow_run_id": 24033272679,
  "workflow_name": "{accelerator}",
  "workflow_url": "https://github.com/{org}/{repo}/actions/runs/24033272679",
  "run_id": 24033272679,
  "job_id": 67890123,
  "conclusion": null,
  "downstream_repo_level": "L2"
}
```

#### Completed — Success

```json
{
  "status": "completed",
  "conclusion": "success",
  "head_sha": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
  "pr_number": 179565,
  "downstream_repo": "{org}/{repo}",
  "upstream_repo": "pytorch/pytorch",
  "workflow_run_id": 24033272679,
  "workflow_name": "{accelerator}",
  "workflow_url": "https://github.com/{org}/{repo}/actions/runs/24033272679",
  "run_id": 24033272679,
  "job_id": 67890123,
  "total_tests": 8432,
  "passed_tests": 8432,
  "failed_tests": 0,
  "skipped_tests": 47,
  "failed_tests_json": "[]",
  "downstream_repo_level": "L2",
  "artifact_url": "https://github.com/{org}/{repo}/actions/runs/24033272679/artifacts"
}
```

#### Completed — Failure

```json
{
  "status": "completed",
  "conclusion": "failure",
  "head_sha": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
  "pr_number": 179565,
  "downstream_repo": "{org}/{repo}",
  "upstream_repo": "pytorch/pytorch",
  "workflow_run_id": 24033272679,
  "workflow_name": "{accelerator}",
  "workflow_url": "https://github.com/{org}/{repo}/actions/runs/24033272679",
  "run_id": 24033272679,
  "job_id": 67890123,
  "total_tests": 8432,
  "passed_tests": 8410,
  "failed_tests": 2,
  "skipped_tests": 20,
  "failed_tests_json": "[{\"name\":\"test_conv2d_xpu_float32\",\"classname\":\"TestConv2dXPU\",\"message\":\"AssertionError: Tensor mismatch\",\"duration_s\":1.23},{\"name\":\"test_relu_backward_xpu_float16\",\"classname\":\"TestActivationsXPU\",\"message\":\"RuntimeError: expected scalar type Float but found Half\",\"duration_s\":0.45}]",
  "downstream_repo_level": "L2",
  "artifact_url": "https://{org}-ci-artifacts.s3.amazonaws.com/pytorch/24033272679/"
}
```

### Comparison: In-Tree vs OOT Pipeline

| | In-Tree | OOT (this RFC) |
|---|---------|-----------------|
| **Hops to ClickHouse** | 6 (job → S3 → workflow_run → upload_stats → S3 → replicator → CH) | 4 (downstream → handler → HUD → DynamoDB → stream → replicator → CH) |
| **Write target** | S3 (`ossci-raw-job-status`) → `clickhouse-replicator-s3` | DynamoDB (`torchci-oot-workflow-job`) → `clickhouse-replicator-dynamo` |
| **Mutability** | Immutable (write once) | Mutable (in_progress → completed via DynamoDB upsert) |
| **Artifact storage** | Centralized storage (LF Foundation managed) | Downstream org's own storage (any public URL) |
| **Auth model** | Service roles (trusted same-org infra) | OIDC → internal token → service role |
| **Rate limiting** | None (relies on GHA concurrency) | Per-repo at relay |
| **Schema** | `default.workflow_job` | `default.oot_workflow_job` (based on `workflow_job`) |

### Files Changed (Reference Implementation)

| File | Action |
|------|--------|
| `torchci/pages/api/oot/results.ts` | New — API endpoint |
| `torchci/lib/oot/ootUtils.ts` | New — types, validation, extraction |
| `clickhouse_db_schema/default.oot_workflow_job/schema.sql` | New — ClickHouse schema |
| `aws/lambda/clickhouse-replicator-dynamo/lambda_function.py` | Edit — +1 line to `SUPPORTED_TABLES` |
| `torchci/pages/oot/index.tsx` | New — global summary page |
| `torchci/pages/oot/[org]/[repo].tsx` | New — per-backend dashboard |
| `torchci/components/oot/OotPrSection.tsx` | New — PR view OOT section |
| `torchci/pages/[repoOwner]/[repoName]/pull/[prNumber].tsx` | Edit — added OotPrSection |
| `torchci/clickhouse_queries/oot_summary/*` | New — saved query |
| `torchci/clickhouse_queries/oot_backend_dashboard/*` | New — saved query |
| `torchci/clickhouse_queries/oot_pr_results/*` | New — saved query |

## Metrics

- **Ingestion latency**: time from callback to ClickHouse availability (target: < 60s)
- **OOT pass rate**: per-backend success rate over 7d rolling window
- **HUD API error rate**: 4xx/5xx rate on `/api/oot/results`
- **Daily callback volume**: per-repo and aggregate, to validate budget thresholds
- **Queue time**: relay-measured dispatch-to-in_progress (identifies downstream infra bottlenecks)
- **Execution time**: relay-measured in_progress-to-completed (identifies test suite performance trends)

## Drawbacks

- **Additional ClickHouse table**: introduces a new `oot_workflow_job` table, increasing schema maintenance surface. Mitigated by reusing the existing `workflow_job` schema pattern.
- **DynamoDB costs**: while PAY_PER_REQUEST auto-scales, high-volume OOT backends could incur non-trivial DynamoDB costs. Mitigated by rate limiting at the relay.
- **Downstream adoption cost**: downstream repos must implement the two-callback model (or use the reusable GitHub Action). This is an incremental cost on top of the relay integration.
- **Trust boundary**: even with OIDC and allowlisting, the relay trusts downstream-reported test counts and failure details. A compromised allowlisted repo could report misleading data. Mitigated by the trusted/untrusted payload split and the proposed callback token.

## Alternatives

1. **Direct ClickHouse writes from HUD API** — eliminates the DynamoDB hop but conflicts with the prescribed DynamoDB → ClickHouse pattern. ClickHouse is append-only, making the two-callback mutable status model awkward.

2. **Separate `oot_ci` database in ClickHouse** — would isolate OOT data but prevents reuse of existing `workflow_job` query patterns and frontend components. Placing the table in the `default` database (as `oot_workflow_job`) is consistent with other tables.

3. **PyTorch-managed artifact storage** — PyTorch infra could provision storage buckets for downstream artifacts. Rejected because it introduces significant access control complexity for N downstream orgs and shifts storage costs to PyTorch.

4. **Single callback model** — one callback at workflow completion. Simpler but loses the "running" indicator on HUD, and the architecture would not match the relay's existing two-callback design.

5. **Polling-based result ingestion** — HUD periodically polls downstream repos for results. Rejected because it requires HUD to know about each downstream repo's API, creates polling overhead, and has higher latency.

## Prior Art

- **PyTorch in-tree CI → HUD pipeline**: S3-based ingestion via `clickhouse-replicator-s3`. Proven at scale but immutable (no status updates). This RFC adapts the pattern for mutable OOT data via DynamoDB.
- **`webhookToDynamo.ts`**: existing HUD pattern for writing webhook data to DynamoDB. The OOT API endpoint follows this pattern.
- **RFC-0050 (Cross-Repository CI Relay)**: defines the relay architecture, allowlist, OIDC auth, and downstream triggering that this RFC builds upon.

## How we teach this

- **Downstream developers** need to know:
  - Their CI results will appear on `hud.pytorch.org/oot/{org}/{repo}` once they reach L2
  - They should include artifact URLs in their "completed" callback for debugging links on HUD
  - Test counts and failed test details are optional but recommended for richer dashboard views

- **PyTorch CI maintainers** need to know:
  - `/oot` provides a global health overview of all OOT backends
  - Rate limits at the relay protect ClickHouse from OOT traffic spikes
  - The replicator mapping is a single line change when adding new downstream tables

- **PR authors** need to know:
  - OOT CI results appear in a collapsible section on their PR page (when results exist)
  - At L2, these are informational only and do not block merging

No documentation reorganization is needed. The `/oot` pages are self-discoverable from HUD navigation.

## Unresolved Questions

1. **Failed test detail storage**: Embedding as `failed_tests_json` String column is simpler (one table). A separate DynamoDB/ClickHouse table for test-level rows would be more queryable but adds infra. Start with embedding; add a separate table later if needed for cross-repo flaky test detection.

2. **Redis TTL for relay state**: 3-hour vs 3-day TTL. GitHub can queue jobs for up to 3 days, so 3-day TTL is recommended.

3. **Non-GHA CI support**: OIDC auth is GHA-specific. Jenkins/Buildkite would need pre-shared API keys. Deferred to a future RFC.

4. **Backfill on failure**: If DynamoDB write fails, should the handler retry? DynamoDB is highly available so failures are rare. Start without a dead-letter queue; add one if needed.

5. **Callback token adoption timeline**: the signed callback token strengthens security significantly but requires coordination with L1 deployment. Can be added incrementally.

## Implementation Plan

### Phase 1: Storage Layer

**Goal**: Set up the data stores so the write path has somewhere to land.

| Task | What | Depends on |
|------|------|------------|
| 1a | Provision `torchci-oot-workflow-job` DynamoDB table with streams enabled | — |
| 1b | Create `default.oot_workflow_job` ClickHouse table from schema file | — |
| 1c | Add `"torchci-oot-workflow-job": "default.oot_workflow_job"` to replicator `SUPPORTED_TABLES` | 1a, 1b |
| 1d | Validate: insert a test record into DynamoDB, confirm it appears in ClickHouse | 1c |

**Deliverable**: DynamoDB → Stream → Replicator → ClickHouse pipeline is live and verified.

### Phase 2: HUD API Endpoint

**Goal**: Build the ingestion endpoint that receives relay callbacks and writes to DynamoDB.

| Task | What | Depends on |
|------|------|------------|
| 2a | Create `torchci/lib/oot/ootUtils.ts` — types, payload validation, extraction logic | — |
| 2b | Create `torchci/pages/api/oot/results.ts` — POST handler with auth (`X-Hud-Internal-Bot`), schema validation, 2MB payload cap, DynamoDB write | 2a, Phase 1 |
| 2c | Unit tests for `validateRelayPayload`, `extractDynamoRecord` | 2a |
| 2d | Integration test: POST valid `{trusted, untrusted}` payload → confirm DynamoDB record created | 2b |

**Deliverable**: `/api/oot/results` endpoint accepts relay payloads and writes to DynamoDB.

### Phase 3: Relay Integration

**Goal**: Connect the relay's Result Handler to the HUD API endpoint.

| Task | What | Depends on |
|------|------|------------|
| 3a | Update Result Handler to forward validated callbacks to HUD API with `X-Hud-Internal-Bot` header | Phase 2 |
| 3b | Add per-repo rate limiting at the relay | — |
| 3c | Implement error handling: retry on 5xx, return status (`delivered` / `hud_rejected` / `hud_unavailable` / `skipped`) to downstream | 3a |
| 3d | Create reusable GitHub Action for downstream: `report-oot-status` (sends `in_progress` / `completed` callbacks) | — |

**Deliverable**: End-to-end write path works: Downstream CI → Result Handler → HUD API → DynamoDB → ClickHouse.

### Phase 4: HUD Frontend Pages

**Goal**: Build the read path — three views that query ClickHouse and display OOT results.

| Task | What | Depends on |
|------|------|------------|
| 4a | Create saved ClickHouse queries: `oot_summary`, `oot_backend_dashboard`, `oot_pr_results` | Phase 1 |
| 4b | Build `/oot` — global OOT summary page (table of repos sorted by pass rate, time range selector) | 4a |
| 4c | Build `/oot/[org]/[repo]` — per-backend dashboard (matrix view: PRs × jobs, failure drill-down, external artifact links) | 4a |
| 4d | Build `OotPrSection` component — collapsible accordion for PR pages | 4a |
| 4e | Integrate `OotPrSection` into existing PR detail page (`/pr/[number]`) | 4d |

**Deliverable**: All three HUD views render OOT data from ClickHouse.

### Phase 5: End-to-End Validation

**Goal**: Validate the full pipeline with a real downstream repo.

| Task | What | Depends on |
|------|------|------------|
| 5a | Set up a test downstream repo with the reusable GitHub Action | Phase 3 |
| 5b | Trigger a PyTorch PR → verify dispatch → downstream CI runs → callbacks sent → DynamoDB → ClickHouse → HUD pages show results | All phases |
| 5c | Test two-callback flow: verify `in_progress` appears in DynamoDB, then `completed` overwrites it and appears in ClickHouse and HUD | 5b |
| 5d | Test external artifact links render correctly on HUD pages | 5b |
| 5e | Test rate limiting: verify relay rejects requests beyond threshold | 5b |
| 5f | Test error cases: invalid auth, malformed payload, oversized payload, HUD unavailable | 5b |

**Deliverable**: Full pipeline validated end-to-end with a real downstream repo.

### Phase 6: Security Hardening (Future)

**Goal**: Add dispatch provenance and replay prevention.

| Task | What | Depends on |
|------|------|------------|
| 6a | Implement signed callback token minting at L1 dispatch | Phase 3 |
| 6b | Add token verification at Result Handler | 6a |
| 6c | Implement Redis-backed state machine for status transitions | 6a |
| 6d | Add orphaned job detection (unconsumed tokens → auto-mark as `timed_out`) | 6c |

**Deliverable**: Callback token provides dispatch provenance, replay prevention, and orphaned job detection.

## Resolution

TBD — this RFC is a work in progress.
