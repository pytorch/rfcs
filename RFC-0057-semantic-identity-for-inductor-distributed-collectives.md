# RFC-0057: Semantic Identity for Inductor Distributed Collectives

## Summary

This RFC proposes a small, private **communication semantic identity** contract for functional distributed collectives in Inductor.

Today, Inductor distributed passes can identify the transport represented by an FX node, such as `_c10d_functional.all_gather_into_tensor` or `_c10d_functional.reduce_scatter_tensor`, but they often infer the higher-level purpose of that communication from graph structure. For example, current FSDP classification uses placeholder ancestry, output reachability, and process-group transitivity. These heuristics are useful for existing FSDP graphs, but the same local graph structure can occur in tensor parallelism (TP), context parallelism (CP), or user-authored distributed code.

The proposed phase-1 contract answers one question:

> **What higher-level purpose caused this collective to be emitted?**

The first implementation is intentionally limited to FSDP communication roles. It introduces:

* a namespaced semantic `role`, such as `fsdp.param_unshard`;
* an explicit distinction between producer-provided and heuristic-inferred metadata;
* a private transport path based on existing PT2 FX annotation infrastructure;
* metadata-first FSDP classification with the current structural heuristics as fallback;
* preservation and invalidation rules for the existing FSDP bucketing and reduce-scatter deduplication transforms;
* diagnostics for coverage, mismatches, and metadata loss.

Phase 1 does **not** add a scheduling priority, hardware cost model, topology model, public annotation API, fused communication-effect schema, or strategy registry. It does not add a role-based ordering rule to `OverlapScheduler` or change its priority function.

This RFC is the first independently landable step toward a general-purpose Distributed Graph Optimizer. Follow-up work may add DTensor/TP/CP producers, physical-resource modeling, scheduling windows, mixed-parallel scheduling policy, and pass composition.

## Motivation

### Current passes know the transport but often guess the intent

Inductor already contains useful distributed optimizations:

| Area | Current behavior | Semantic limitation |
| --- | --- | --- |
| FSDP all-gather classification | Uses placeholder ancestry to identify parameter all-gathers. | A non-FSDP all-gather can have the same ancestry. |
| FSDP reduce-scatter classification | Uses conservative output reachability and process-group relationships. | The same reduce-scatter transport may represent a different parallelism purpose. |
| FSDP all-reduce bucketing | Identifies FSDP groups from all-gather/reduce-scatter evidence and process-group transitivity. | A process-group name is not itself a semantic role. |
| Reduce-scatter deduplication | Rewrites `add(wait(RS(a)), wait(RS(b)))` when algebraic and single-use conditions hold. | The existing transform is not restricted to one semantic role and may combine communications produced for different purposes. |
| Micro-pipeline TP | Recognizes `all_gather + matmul` and `matmul + reduce_scatter` patterns and replaces them with fused operators. | The original collective disappears, so any semantic identity must be transferred or deliberately dropped. |
| Overlap scheduling | Uses graph dependencies, compute domination, exposed time, process-group availability, memory, and in-flight limits. | It has no stable higher-level communication identity and no physical-resource model across process groups. |

The problem is therefore not that `OverlapScheduler` treats every collective identically. It already derives local criticality from the graph. The missing information is the semantic origin that cannot always be recovered from the lowered collective structure and that may be lost across graph transforms.

### A concrete mixed-parallel example

Consider an FSDP + TP + CP training graph in which the following communications are ready near the same region of compute:

1. an FSDP all-gather for the next parameter group;
2. a TP all-gather whose result is consumed by the current matmul;
3. a CP KV exchange that must complete before the next attention chunk.

All three may appear as functional collectives. They can use different logical process groups while still sharing the same physical NVLink or network interface.

These operations differ along several independent axes:

| Question | FSDP parameter unshard | TP activation redistribution | CP KV exchange |
| --- | --- | --- | --- |
| Why does it exist? | Materialize a sharded parameter. | Produce the layout required by a tensor-parallel compute. | Move KV state for the next attention step. |
| When is it needed? | Before the corresponding FSDP module uses the parameter. | Before the dependent matmul. | Before the dependent attention chunk. |
| Can it be issued early? | Often, within a memory-bounded prefetch window. | Sometimes, including through micro-pipelining. | Often, as part of a ring pipeline. |
| What may it contend with? | Other DP/TP/CP communication using the same physical link. | Other DP/TP/CP communication using the same physical link. | Other DP/TP/CP communication using the same physical link. |

A semantic role answers only the first row. Dependency analysis and a future scheduling-window model answer the second and third rows. A future topology and resource model answers the fourth row. Phase 1 intentionally introduces only the first piece.

### Why semantic identity is needed before scheduling policy

Without an explicit identity contract, every new distributed pass must:

* rediscover which parallelism strategy emitted each collective;
* duplicate or extend structural heuristics;
* decide independently how identity should survive bucketing, deduplication, fusion, decomposition, and lowering;
* coordinate ad hoc with all existing passes when graph nodes are replaced.

This does not scale well as more parallelism dimensions are combined. A small identity contract gives producers and consumers a common vocabulary without requiring the community to accept a new scheduler or cost model in the same change.

### Why FSDP first

FSDP is a practical first validation target because:

* Inductor already has FSDP-specific classifiers and bucketing consumers;
* current heuristics provide a backward-compatible fallback and a source of best-effort inferred metadata;
* FSDP parameter all-gather and gradient synchronization provide distinct, concrete semantic purposes;
* the existing bucketing and reduce-scatter deduplication passes exercise one-to-many and many-to-one metadata behavior;
* phase 1 can validate the transport and preservation contract before adding TP, CP, topology, or scheduling policy.

## Target Architecture

Phase 1 is not intended to be an FSDP-specific endpoint. It is the smallest independently reviewable prerequisite for a general-purpose Distributed Graph Optimizer.

```text
FSDP / DTensor / TP / CP / future distributed producers
                         |
                         v
        FX graph with identity and provenance
          phase 1: FSDP identity (this RFC)
          phase 2: DTensor / TP / CP coverage
                         |
                         v
          graph, cost, and constraint analysis
      readiness / deadline / buffer lifetime /
             topology / resource contention
                         phase 3
                         |
                         v
          distributed optimization orchestration
       +-----------------------------------------+
       | mixed-parallel scheduling policy       |  phase 4
       | bucketing / fusion / pipeline strategy |  phase 5
       | preservation and conflict contracts    |  phase 5
       +-----------------------------------------+
                         |
                         v
             lowered distributed execution plan
```

Only the phase-1 identity contract and its metadata preservation rules are proposed here. The remaining blocks provide context for the follow-up plan and are not pre-approved by accepting this RFC. The orchestration layer may run graph transforms before or after scheduling as appropriate; each transform must declare which analysis facts it preserves, combines, or invalidates.

## FSDP and Inductor Scheduling Context

This section separates FSDP runtime communication from Inductor compile-time graph scheduling. The semantic identity contract must describe the purpose of the collective without confusing these two layers.

### FSDP runtime flow

A simplified FSDP2 training flow is:

```text
forward pre-hook
  -> unshard parameters (all-gather)
  -> wait for the required parameters
  -> forward compute
  -> post-forward reshard
  -> optionally prefetch a later parameter group

backward pre-hook
  -> unshard parameters required by backward
  -> wait for the required parameters
  -> optionally prefetch a later backward parameter group
  -> backward compute
  -> post-backward gradient reduce-scatter
  -> optional additional reduction for hybrid sharding
  -> final event/buffer synchronization
```

The role of a parameter all-gather is stable: it unshards a parameter. Whether a particular instance is currently blocking compute or is being used as a prefetch is dynamic. Similarly, the role of a gradient reduce-scatter is stable, but how long it may be deferred depends on readiness, memory, buffer limits, and the optimizer deadline.

### Inductor post-grad flow

A simplified current post-grad flow is:

```text
earlier post-grad transforms, including optional micro-pipeline TP
  -> SPMD verification
  -> optional communication decomposition
  -> reduce-scatter deduplication
  -> collective bucketing
  -> stable topological sort
  -> optional OverlapScheduler
       -> optional FSDP pre-bucketing
       -> identify collective-start / wait pairs
       -> estimate compute and communication runtimes
       -> reorder using dependency, domination, memory, and in-flight limits
  -> optional low-contention collective replacement
```

Phase 1 adds identity metadata around the existing FSDP classification and transform paths. It does not replace either the FSDP runtime schedule or the current Inductor scheduling algorithm.

The current implementation points referenced by this RFC are:

* [`torch/_inductor/fx_passes/fsdp.py`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/fsdp.py) for FSDP classification, reduce-scatter deduplication, and FSDP bucketing;
* [`torch/_inductor/fx_passes/post_grad.py`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/post_grad.py) for pass ordering;
* [`torch/_inductor/fx_passes/overlap_scheduling.py`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/overlap_scheduling.py) for the existing FX-level overlap scheduler; and
* [`torch/fx/traceback.py`](https://github.com/pytorch/pytorch/blob/main/torch/fx/traceback.py) for the private PT2 annotation transport proposed below.

## Goals

* Introduce a minimal semantic identity for Inductor functional collectives.
* Use namespaced semantic roles that describe purpose rather than only the current transport operator.
* Distinguish explicit producer metadata from heuristic inference.
* Reuse existing PT2 FX annotation transport where possible instead of adding a new stable public annotation API.
* Make FSDP consumers metadata-first while retaining current heuristics as fallback.
* Define precedence, preservation, invalidation, and diagnostic rules.
* Preserve current default runtime and scheduling behavior.
* Collect evidence about annotation coverage, mismatch rate, metadata loss, and compilation overhead.
* Establish a contract that future DTensor, TP, and CP producers can extend without baking FSDP assumptions into the scheduler.

## Non-goals

Phase 1 does not:

* add or change `OverlapScheduler` priority keys;
* define `CommUrgency` or a static `role -> urgency` mapping;
* define scheduling windows or deadlines;
* model physical topology, NIC affinity, bandwidth contention, or communication runtime;
* change exposed-time accounting, memory budgeting, or in-flight limits;
* introduce a stable public `torch.distributed.comm_semantics()` API;
* require all DTensor, TP, CP, PP, or EP collectives to be annotated;
* define a general schema for fused communication-compute effects;
* introduce cross-role reduce-scatter deduplication policy;
* replace the post-grad pass sequence with a strategy registry;
* expose a public user-defined Inductor strategy API;
* require a runtime performance improvement. Phase 1 validates a compiler contract and must avoid runtime regressions.

## Proposed Contract and Implementation Direction

Code and names in this section illustrate one viable implementation. Unless a property is called out as part of the acceptance scope or a semantic invariant, the exact helper names, Python types, metadata container, and producer call sites may change during review and prototyping.

### Acceptance scope

This RFC asks the community to agree only to the following first step:

1. functional collectives may carry a private namespaced semantic identity;
2. explicit producer metadata is authoritative over structural inference;
3. current FSDP classifiers consume explicit metadata first and retain their existing heuristics as fallback;
4. inferred metadata remains visibly best-effort and is never silently upgraded to explicit truth;
5. existing transforms must preserve, conservatively combine, or invalidate the identity rather than copying one arbitrary input role;
6. the first implementation adds diagnostics and tests but no scheduling-policy change.

### Semantic identity is not scheduling priority

The phase-1 contract deliberately separates four concepts:

| Concept | Question | Phase 1 |
| --- | --- | --- |
| Semantic identity | Why was this communication emitted? | Proposed here. |
| Transport | Which collective/operator implements it? | Already available from the FX node. |
| Scheduling constraint | When may it start and when must it complete? | Future RFC. |
| Resource usage and policy | With what does it contend, and which ready operation should run first? | Future RFC. |

For example, both a blocking FSDP parameter all-gather and an early FSDP parameter prefetch have the same semantic role. Their current scheduling state is different and should be derived from dependencies, memory lifetime, and a future scheduling policy rather than stored as a fixed role property.

### Metadata transport and canonical location

PyTorch already has a private, non-backward-compatible PT2 annotation mechanism: `torch.fx.traceback.annotate()`. It transports custom metadata through Dynamo and PT2 FX tracing under `node.meta["custom"]`.

The current API documentation describes this mechanism as intended for advanced debugging, analysis, and external tooling. Using it for compiler-consumed identity is therefore a proposal, not an assumption. The first implementation must prove the required transport and obtain agreement from FX/PT2 owners; if it cannot, this RFC falls back to a dedicated private metadata path rather than silently depending on unsupported propagation.

Phase 1 proposes to prototype that mechanism rather than add a new public API. The following payload illustrates the required logical fields using primitive values, so distributed producer code need not import an Inductor-owned type:

```python
COMM_SEMANTICS_CUSTOM_KEY = "distributed.comm_semantics"

{
    "version": 1,
    "role": "fsdp.param_unshard",
    "source": "explicit",
    "producer": "fsdp",
}
```

The producer annotation should be scoped narrowly enough that unrelated operations do not inherit the collective identity. The exact call site and transport container are prototype outcomes, not public contracts of this RFC.

For an ordinary asynchronous functional collective, the canonical phase-1 location is the collective-start node. A normalization helper may read a tag transported to a wrapper or wait node, but consumers trust only the normalized tag on the start node. This avoids duplicating identity across both start and wait nodes.

An alternative top-level `node.meta["comm_semantics"]` representation is discussed below. Phase 1 prefers existing `custom` transport unless the prototype shows that it cannot survive the required PT2 paths soundly.

### Validated Inductor view

Inductor consumers should use one centralized validation path rather than read the raw dictionary ad hoc. The validated view must preserve `version`, `role`, `source`, and `producer`, and should not permit consumers to mutate the payload accidentally. Its exact Python representation is an implementation detail.

Malformed, unsupported-version, or incomplete metadata is treated as absent and made visible in debug diagnostics. It must not change default behavior.

### Phase-1 FSDP roles

Role names describe semantic purpose rather than only repeating the current collective operator:

| Role | Current transport | Meaning |
| --- | --- | --- |
| `fsdp.param_unshard` | all-gather | Materialize parameters for an FSDP parameter group. |
| `fsdp.grad_reduce_shard` | reduce-scatter | Reduce gradients and produce the sharded gradient representation. |
| `fsdp.grad_replicate_reduce` | all-reduce in applicable hybrid-sharding paths | Reduce gradients across the replicated dimension. |

Consumers must validate that role and transport are compatible. For example, `fsdp.param_unshard` on a reduce-scatter node is invalid metadata.

The names are private and experimental in phase 1. Exact naming remains open to review, but the contract should preserve the distinction between semantic purpose and transport.

### Producer and inference precedence

There are two phase-1 sources:

1. **Explicit producer metadata.** Emitted by the FSDP/SimpleFSDP path around the functional collective call. It is authoritative if valid.
2. **Inferred FSDP metadata.** Produced by current structural classifiers when explicit metadata is absent. It is visibly best-effort.

The precedence rules are:

* valid explicit metadata wins;
* inference never overwrites explicit metadata;
* if explicit metadata conflicts with a structural heuristic, the explicit role is used and the mismatch is made visible in debug diagnostics;
* if metadata is missing or malformed, current structural behavior is retained;
* inferred metadata remains `source="inferred"` through downstream transforms;
* no consumer may treat an inferred tag as if it were producer-authoritative merely because it has been stored on the node.

### FSDP consumers

Existing FSDP classifiers become metadata-first according to the following behavior:

1. valid explicit metadata with a matching role and transport classifies the collective as FSDP;
2. valid explicit metadata for a different role prevents the FSDP structural heuristic from overriding the producer;
3. missing, malformed, or inferred metadata retains the existing structural fallback; and
4. a fallback result may be cached as inferred metadata, but its source remains visibly inferred.

The same precedence applies to FSDP reduce-scatter classification and FSDP process-group identification.

To make metadata available before transforms that may erase or merge collectives, phase 1 may add an FSDP semantic-normalization pass before the existing reduce-scatter deduplication and bucketing passes. The pass does not change graph structure or scheduling.

## Metadata Preservation and Invalidation

Phase-1 metadata is a scalar identity for an ordinary collective. A transform must not copy one arbitrary input role when the output no longer has one scalar identity.

### Graph copy and stable topological sort

* `Graph.node_copy()` and equivalent one-to-one copies preserve the payload by value.
* A stable topological sort does not change the payload.
* Helpers should allocate a fresh dictionary rather than rely on mutable aliasing between node metadata dictionaries.

### Same-role bucketing

Bucketing may preserve one scalar role when all bucket members have:

* the same semantic role;
* compatible collective kind;
* compatible process group, reduction, dtype, layout, and shape constraints required by the existing transform.

For the bucketed result:

* `source="explicit"` is preserved only if every input role is explicit;
* otherwise the output is `source="inferred"`;
* `producer` identifies the bucketing transform;
* existing FX `from_node` and bucketing provenance record the original nodes.

Phase 1 does not add a second embedded list of communication effects solely for provenance.

### Reduce-scatter deduplication

The existing reduce-scatter deduplication transform is algebraic and is not currently restricted to one FSDP role. Phase 1 therefore defines metadata behavior separately from transform policy:

* if all matched reduce-scatter nodes have the same scalar role, preserve that role on the replacement;
* if roles differ, are missing, or are incompatible, the existing transform may retain its current default behavior, but the replacement receives no scalar semantic identity and debug diagnostics record the loss;
* phase 1 must not choose one input role arbitrarily;
* changing the transform to skip a mathematically valid cross-role match is a future policy decision and is not required for behavior preservation here.

This rule is intentionally conservative. A future RFC may decide that semantic identity should constrain deduplication because combining two roles can join their readiness times and change overlap opportunities even when the algebra is valid.

### Fusion and opaque communication-compute operators

Micro-pipeline TP may replace an all-gather/matmul or matmul/reduce-scatter pattern with an opaque fused operator. Phase 1 has no general fused-effect schema, and TP roles are not required.

If a phase-1 tagged collective is consumed by an opaque fusion, the transform must not copy the scalar collective role to the fused output as if it were still an ordinary schedulable collective. Existing FX provenance should be preserved, and the scalar communication identity should be removed unless the transform can prove that the replacement is still the same ordinary collective.

A later RFC may introduce a structured list of communication effects for fused or cross-role merged operators when there is a concrete consumer for that data.

### Decomposition and lowering

A decomposition or lowering may preserve the scalar identity only if the new node still represents the same ordinary communication purpose and remains a collective node understood by the consumer.

If the communication becomes an opaque region, multiple backend operations, or a fused communication-compute operator, phase 1 drops the scalar tag and relies on existing provenance. Silent arbitrary transfer is not allowed.

## DTensor Placement Transitions and Future Producers

DTensor placement information describes layout semantics, not necessarily the higher-level parallelism strategy that requested the redistribution. A future DTensor producer could expose generic placement-transition identity unless its caller supplies a more specific role. The vocabulary in the last column is illustrative and is not standardized by phase 1.

| Placement transition | Typical implementation | Generic information available to DTensor |
| --- | --- | --- |
| `Shard(d) -> Replicate` | all-gather | `dtensor.shard_to_replicate` |
| `Partial(op) -> Replicate` | all-reduce | `dtensor.partial_to_replicate` |
| `Partial(op) -> Shard(d)` | reduce-scatter | `dtensor.partial_to_shard` |
| `Shard(d0) -> Shard(d1)` | all-to-all or a composed redistribution | `dtensor.reshard` |
| `Replicate -> Shard(d)` | normally a local split | no communication role |
| unchanged placement | no-op | no communication role |

The same `Shard -> Replicate` transition can represent an FSDP parameter unshard, a TP activation redistribution, or a user-authored redistribution. DTensor must not label all such transitions as TP solely from placements.

A future high-level producer may specialize the generic identity, for example:

```text
dtensor.shard_to_replicate
  -> fsdp.param_unshard
  -> tp.activation_redistribute
```

The exact precedence between a generic DTensor tag and a high-level explicit tag will be defined with those producers. Phase 1 requires only FSDP roles.

## Validation Plan

The phase-1 contract should be validated before semantic identity affects any scheduling policy. The implementation needs evidence for:

* end-to-end transport of at least one explicit FSDP identity through the relevant Dynamo, AOTAutograd, graph-copy/retrace, and Inductor post-grad paths;
* explicit-over-inferred precedence, malformed-metadata fallback, and role/transport validation;
* the preservation and invalidation rules for copies, bucketing, reduce-scatter deduplication, and opaque replacements;
* at least one mixed FSDP + TP case in which FSDP identity is not confused with a structurally similar non-FSDP collective; and
* compatibility with existing FSDP behavior when metadata is absent or agrees with the heuristic, together with reported compile-time and metadata-memory overhead.

A valid explicit tag that conflicts with a heuristic may intentionally change FSDP classification and a downstream FSDP bucketing decision. That is the correctness purpose of producer-authoritative identity, not a new scheduling policy, and requires a targeted test.

Debug information should be sufficient to inspect explicit/inferred coverage, heuristic mismatches, malformed tags, role/transport conflicts, and metadata dropped by transforms. The number and names of counters, verifier structure, and logging format are implementation details.

Implementation may proceed incrementally or in shadow mode when useful, but the exact PR split and landing order are not part of this RFC's contract and may change based on review and prototype evidence. Runtime speedup is not a phase-1 requirement.

## Follow-up Plan

The following sequence is directional, not a commitment to exact RFC or PR boundaries. Later work may be split, reordered, or revised based on community feedback and implementation evidence; accepting phase 1 does not pre-approve later policy or public APIs.

### Phase 2: DTensor, TP, and CP producers

* add generic DTensor placement-transition provenance;
* add higher-level TP/CP producer identities where the purpose is known;
* define precedence between generic and specialized tags;
* validate backward and fused-operator propagation requirements.

### Phase 3: constraint and hardware-cost inputs

* derive or represent earliest issue point and completion deadline;
* introduce a constraint model for communication/buffer lifetimes, peak memory, in-flight limits, and correctness restrictions;
* map logical process groups to shared physical resources where possible;
* extend the existing runtime estimation behind a topology- and contention-aware hardware cost interface; and
* add bytes-per-rank and topology/resource facts without duplicating readily derivable operator data.

### Phase 4: mixed-parallel scheduling policy

* combine semantic identity, dependency-derived criticality, scheduling windows, memory limits, and physical-resource contention;
* evaluate whether semantic identity is a constraint, feature, or tie-breaker in the existing priority function;
* keep policy opt-in until correctness and performance are demonstrated;
* compare against the current domination- and exposed-time-based scheduler.

### Phase 5: pass composition and strategy framework

After semantic identity and transform invariants are validated, a future RFC may consider:

* an internal strategy registry;
* explicit analysis preservation/invalidation contracts;
* transform conflict declarations;
* graph-region strategies such as CP ring pipelining;
* whether any extension point should be public.

Together, these phases complete the pipeline shown above: producers expose intent, analysis derives constraints and costs, the scheduler applies policy, and graph strategies compose through explicit preservation and conflict rules.

## Drawbacks

1. **Additional compiler metadata.** Even a small private identity creates a maintenance obligation for producers, consumers, and graph transforms.

2. **Explicit producer coverage is incomplete.** Phase 1 starts with FSDP. Other collectives remain unannotated or inferred until later work.

3. **Incorrect explicit metadata is dangerous.** Explicit identity is authoritative, so producer code and verifier coverage must be strong.

4. **Inferred identity remains imperfect.** Storing a heuristic result does not make it true. The source distinction must remain visible to every consumer.

5. **No immediate scheduling speedup.** Phase 1 validates infrastructure and semantics. Its value is reduced misclassification risk, consistent transform behavior, and evidence for later scheduling work.

6. **Metadata may be dropped.** Without a fused-effect schema, phase 1 deliberately loses scalar identity at some opaque transforms rather than preserving misleading information.

## Alternatives

### Alternative 1: Keep extending structural heuristics

This is simpler for each individual optimization and remains the fallback in phase 1. It does not provide authoritative identity, and the number of interactions grows as parallelism strategies are combined.

### Alternative 2: Use a top-level `node.meta["comm_semantics"]` key

A dedicated top-level key is easier for consumers to discover and could hold an already-validated private representation. It also requires a new metadata propagation path through PT2 and creates a new globally recognized FX metadata field.

Phase 1 prefers existing `node.meta["custom"]` transport plus a namespaced key, subject to prototype results. If `custom` cannot provide sound propagation or is considered unsuitable for compiler-affecting metadata, the dedicated key is the fallback design.

### Alternative 3: Persist only inferred FSDP classification

Inductor could run current heuristics once and cache their results as metadata. This would unify downstream consumers but would not solve the central problem: the same heuristic can be wrong in a mixed-parallel graph. Phase 1 therefore requires at least one explicit producer path.

### Alternative 4: Include `CommUrgency` in phase 1

A static urgency field appears to connect the RFC directly to scheduling, but urgency is occurrence-dependent. A current-layer FSDP all-gather and a prefetched FSDP all-gather have the same role and different scheduling states. Adding urgency before scheduling windows and resource modeling would freeze a premature contract.

### Alternative 5: Define merged and fused effects now

A `CommEffect` list could preserve identity through cross-role merges and opaque communication-compute fusion. There is no phase-1 policy consumer for that richer schema. This RFC defers it until a concrete scheduler, cost model, or diagnostic requires it.

### Alternative 6: Propose the complete Distributed Graph Optimizer now

The complete target combines semantic modeling, topology, cost modeling, constraints, scheduler policy, and strategy composition. Proposing all components together would require resolving too many independent design and implementation questions at once.

### Alternative 7: Add a stable public annotation API now

A public API would require compatibility guarantees for roles, transport, metadata lifetime, and consumer behavior. Phase 1 uses private PT2 infrastructure and does not establish those guarantees.

## Prior Art

* **PyTorch DTensor.** Placements express layout semantics above raw collective operators. Future producer work can preserve generic transition provenance while allowing higher-level FSDP/TP/CP specialization.
* **Current Inductor FSDP passes.** Existing classifiers and bucketing passes are the first fallback and consumer set for phase 1.
* **Current FX custom annotations.** `torch.fx.traceback.annotate()` already transports private custom metadata through PT2 tracing and is the preferred phase-1 transport to prototype.
* **XLA/GSPMD.** Sharding information preserves distributed intent above low-level collectives and can feed later compiler scheduling decisions.
* **Megatron-style scheduling.** Hand-authored TP/CP/PP systems distinguish communication purpose, readiness, and pipeline deadline explicitly. This RFC introduces only the identity layer needed before comparable compiler policy can be discussed.

## How We Teach This

### For users

Phase 1 has no stable user-facing API and introduces no role-based scheduling policy. Ordinary FSDP users do not need to write annotations.

### For distributed producers

If a producer knows why it is emitting a functional collective, it may attach a private semantic identity using the selected private transport. The annotation must be narrowly scoped and must use a namespaced role.

### For Inductor developers

The rule is:

> Read validated explicit identity first; fall back to current inference when it is missing; preserve the explicit/inferred distinction; and never copy one arbitrary scalar role across a many-to-one transform.

Developer documentation should explain:

* the raw wire representation and validated Inductor view;
* explicit-vs-inferred precedence;
* canonical placement on collective-start nodes;
* same-role bucketing behavior;
* deduplication and opaque-transform invalidation;
* available validation and debugging support.

## Unresolved Questions

1. **Metadata container.** Is `node.meta["custom"]` appropriate for a private compiler identity that may later affect behavior, or should phase 1 add a dedicated top-level metadata field after the producer prototype?

2. **Exact FSDP producer locations.** Which FSDP/SimpleFSDP functional collective call sites can attach explicit metadata narrowly and propagate it through the relevant Dynamo/AOTAutograd paths?

3. **Role naming.** Are purpose-oriented names such as `fsdp.param_unshard` preferable to transport-oriented names such as `fsdp.param_allgather`?

4. **Backward metadata.** Should backward collectives be annotated directly by the producer, or may any role be copied from forward-origin metadata? The default preference is direct annotation because a backward operation's role is not generally the same as its forward origin.

5. **Deduplication policy.** When explicit roles differ but the algebraic reduce-scatter rewrite is valid, should a future opt-in policy skip the merge to preserve independent scheduling opportunities?

6. **Metadata lifetime.** At which lowering boundary should the private identity be dropped if no later consumer remains?

7. **Generic-to-specialized transition.** When future DTensor and high-level producers both provide identity, what exact precedence and validation rules should apply?

8. **Required evidence.** What producer coverage, mismatch rate, compilation overhead, and mixed-parallel examples are sufficient before a later RFC may use identity in scheduling decisions?
