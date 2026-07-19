# Device-scoped Inductor operator policies for out-of-tree backends

**Authors:**
* @peaceorwell

> **AI assistance disclosure:** Codex was used to restructure and edit this
> draft from existing design notes. This PR is intentionally being opened as a
> draft. The author will verify the technical claims and add current-`main`
> prototype evidence before marking it ready for general review.


## Summary

This RFC proposes an internal, immutable, cache-aware operator policy for
Inductor device backends. A policy lets an in-tree or out-of-tree backend:

* override or exclude AOTAutograd decompositions for a compile;
* override an Inductor lowering for a particular device type;
* request Inductor fallback for an operator on a particular device type; and
* provide a stable identity that participates in both the AOTAutograd and
  Inductor FX graph cache keys.

The first version is deliberately narrow. It supports a graph with at most one
active accelerator backend (plus CPU), immutable backend registration, and
compile-scoped decomposition selection. It does not propose process-wide
runtime mutation, complete mixed-accelerator decomposition semantics, or a
new stable public API.


## Motivation

Inductor already has useful backend extension points. A backend can register
scheduling and wrapper classes through `register_backend_for_device`, callers
can pass a compile-scoped decomposition table to `compile_fx`, and
`GraphLowering` has a `user_lowerings` path. These mechanisms avoid many
backend-specific forks, but they do not yet form a complete operator policy.

The missing part spans two compiler stages:

1. AOTAutograd decomposes ATen operators before Inductor lowering. Once an
   operator has been decomposed, a device-specific Inductor lowering cannot
   recover it.
2. Inductor lowering can depend on the device of an individual IR node, but
   the built-in lowering table is process-wide.

An out-of-tree accelerator may need to preserve an ATen operator so that it can
lower it to a vendor library call, while CUDA or CPU should continue using the
default decomposition and lowering. Today, the practical choices are to
register a separate Dynamo backend that wraps Inductor or to mutate shared
decomposition/lowering tables. The latter has three problems:

* **Isolation:** import order or backend initialization can change compilation
  behavior for other devices in the same process.
* **Cache correctness:** a cache key derived from the graph and global Inductor
  configuration does not automatically describe an out-of-tree mutation of
  decomposition or lowering state.
* **Upgrade cost:** an out-of-tree backend must patch private tables and the
  corresponding cache-key plumbing together. Missing either side can produce
  a silent wrong cache hit rather than a clear failure.

One production out-of-tree integration motivating this proposal currently
maintains device-specific sets covering 33 decomposition packets and 69
lowering packets. The exact counts are not the API requirement; they show that
this is a policy-level need rather than a one-operator exception. Before this
RFC is moved out of draft, the author will attach a current-`main` prototype
showing (a) a preserved operator reaching a device lowering without changing
another device's graph and (b) distinct cache keys for distinct policies.

The desired user experience remains the standard one:

```python
compiled = torch.compile(model, backend="inductor")
```

Loading an accelerator extension should register the policy needed by that
device. Application code should not need to select a vendor-specific wrapper
backend merely to obtain correct decompositions and lowerings.


## Proposed Implementation

### Terminology and conceptual model

This RFC uses *operator policy* to mean the immutable set of operator decisions
contributed by one device backend. The names below are illustrative rather
than a request to stabilize this exact Python API:

```python
@dataclass(frozen=True)
class DeviceOperatorPolicy:
    device_type: str
    decomposition_overrides: Mapping[OpOverload, Callable | Exclude]
    lowering_overrides: Mapping[OpOverload, Callable | Fallback]
    cache_key: str
```

`Exclude` means that an operator is removed from the decomposition table for
the current compile. `Fallback` means that, for nodes on this device, Inductor
must use its existing fallback path instead of the default lowering. A backend
may also provide a callable override.

The implementation may expose a provider or registry rather than this
dataclass directly. The required contract is:

* the policy is immutable for the lifetime of a compile;
* the registration is keyed by device type;
* the policy has a deterministic, serializable cache identity; and
* invalid combinations fail during policy construction or compile setup,
  rather than after partially mutating global state.

### Registration and lifetime

A backend registers its policy while the device extension is loaded. Possible
homes include `register_backend_for_device`, `InductorChoices`, or a dedicated
operator-policy registry. This RFC does not require one of those placements,
but it does require one authoritative resolution path.

The first version does not support adding or removing rules from a registered
policy at process scope. Model-specific tuning can later be represented as a
compile-scoped overlay, which must produce a new cache identity. Avoiding
runtime global mutation keeps registration, subprocess compilation, and cache
behavior tractable in the initial implementation.

Duplicate registration must be explicit. Registering the same provider and
identity may be idempotent; registering a different policy for an already
owned device type should fail with a diagnostic that names both registrations.

### Compile-scoped decomposition selection

At compile setup, Inductor determines the active accelerator device type from
the graph inputs and constants, resolves its registered policy, and derives a
decomposition table:

1. start with the current default decomposition table;
2. remove entries represented by `Exclude`;
3. apply callable decomposition overrides; and
4. pass the resulting table through the existing compile-scoped decomposition
   path rather than editing the global table.

The first version supports CPU-only graphs and graphs containing CPU plus one
accelerator type. If a graph contains multiple accelerator types with
incompatible decomposition policies, it should emit a clear unsupported-case
diagnostic. Silently choosing the first device would make graph order affect
semantics.

Graphs with no tensor inputs, tensor constants only, fake tensors, or symbolic
inputs need an agreed device-discovery rule. This remains an unresolved detail
for the RFC, but the rule must run before AOTAutograd cache lookup and must be
shared by cache-key construction and table selection.

This proposal does not treat a decomposition returning `NotImplemented` as a
control signal. The current decomposition protocol does not define that as
"keep the original operator". Exclusion is expressed when constructing the
compile-scoped table.

### Device-scoped lowering resolution

Lowering selection happens per IR node, where the node's device is available.
The proposed precedence is:

1. an explicit compile-scoped/user lowering;
2. a device-policy lowering or `Fallback` rule for `(operator, device_type)`;
3. the existing default Inductor lowering; and
4. the existing unsupported-operator behavior.

This preserves the ability of an explicit caller override to win while letting
two devices in the same process select different lowerings for the same ATen
operator. The default lookup and generated code are unchanged when no policy
is registered.

`Fallback` should reuse Inductor's existing fallback representation and
diagnostics. It is not a second generic fallback implementation. The important
new behavior is that choosing fallback for one device does not delete or
replace the default lowering for every device.

### Cross-stage validation

Decomposition and lowering decisions are related. If a policy excludes a
decomposition but provides neither a device lowering nor a supported fallback
for the preserved operator, policy construction should reject the rule (or
compile setup should reject it if shape/device-dependent information is
required).

Validation should report:

* the operator overload;
* the device type;
* the excluded or overridden decomposition rule; and
* which lowering/fallback requirement is missing.

This turns a late lowering failure into an actionable backend-registration
error.

### Cache identity

The effective policy identity must participate in every persistent cache whose
input or output crosses the policy boundary:

* **AOTAutograd cache:** decomposition changes alter the FX graph presented to
  Inductor, so the identity is needed before or as part of this cache lookup.
* **Inductor FX graph cache:** lowering and fallback changes can alter generated
  code even for an identical post-AOT graph.

An explicit version string supplied by the backend is the simplest initial
contract. It must be deterministic across processes for equivalent policies
and must change whenever effective rules or their implementation change.
Hashing arbitrary Python callable source is not sufficient on its own because
closures, compiled extensions, and packaging can make that identity unstable
or incomplete.

The cache-key plumbing should store the device type and policy identity, not
attempt to serialize the callable maps. A backend is responsible for bumping
its identity when its implementation changes, in the same way that a compiler
or codegen version participates in a cache key.

### Serialization, subprocesses, and packaging

Subprocess compilation and AOT packaging must resolve the same policy as the
parent compile. The initial implementation should require the device extension
to be importable before policy resolution and serialize only the stable
identity plus device type. If a policy cannot be resolved in the worker, the
compile should fail with a message identifying the missing backend extension;
it must not silently fall back to the global defaults after a cache key was
computed with a device policy.

### Compatibility and scope

When no policy is registered, this proposal must produce the same
decomposition table, lowering lookup, generated code, and cache keys as current
PyTorch. Existing registration APIs without a device argument continue to have
their current process-wide semantics.

The initial API can remain under `torch._inductor`. This RFC asks for a shared
extensibility contract used by in-tree and out-of-tree backends, not an
immediate long-term public-API stability guarantee.

The following are explicitly out of scope:

* process-wide runtime `add`/`remove` mutation;
* full decomposition semantics for graphs containing multiple accelerators;
* autotuning, Triton grid, tiling, or combo-kernel policy;
* a generic arbitrary backend property bag; and
* a guarantee that every Inductor internal extension point becomes public.

### Testing strategy

Core tests must not require vendor hardware. A test-only device backend or
OpenReg-style extension can validate:

* the same operator uses the default path on one device and a policy path on
  another;
* excluding a decomposition preserves the original operator for the target
  compile without mutating the global table;
* a device lowering and device fallback do not affect other device types;
* two policy identities produce different AOTAutograd and FX graph cache keys;
* duplicate registration and invalid exclude/lowering combinations fail with
  useful diagnostics; and
* lazy/subprocess compilation resolves the same identity or fails explicitly.

The motivating backend can run separate external CI as additional validation,
but upstream correctness must not depend on access to that hardware.

### Incremental implementation plan

The RFC is intended to land as reviewable, independently tested changes:

1. **Immutable policy model and cache identity plumbing.** Add registry and
   identity propagation to both cache layers without changing compilation
   behavior.
2. **Device lowering registry.** Add `(operator, device_type)` resolution while
   preserving the current default table.
3. **Device-scoped fallback sentinel.** Reuse the existing fallback path, scoped
   to a device policy.
4. **Compile-scoped decomposition provider.** Construct and pass the table for
   a single active accelerator without global mutation.
5. **Atomic validation and backend registration integration.** Add diagnostics,
   documentation, and a test backend example.
6. **Possible follow-up: compile-scoped policy overlays.** Address model-level
   tuning only after the immutable registration and cache contract is proven.


## Metrics

The proposal is successful if:

* an out-of-tree backend can remove its direct mutations of global
  decomposition and lowering tables and its manual cache-key patch;
* the same process can compile for the policy device and a default device
  without cross-device contamination;
* changing only the policy identity results in distinct AOTAutograd and
  Inductor FX graph cache keys;
* no-policy builds have no decomposition, lowering, generated-code, or cache
  behavior change; and
* each implementation PR includes a hardware-independent test and removes one
  corresponding out-of-tree patch or copied implementation path.

For the motivating backend, the draft will be updated with the number of
deleted monkey patches and copied lines after its current-`main` port is
complete. Historical patch counts are not proposed as an acceptance metric.


## Drawbacks

* The policy spans AOTAutograd and Inductor, increasing the number of compiler
  components that must agree on device discovery and identity.
* A registry creates lifecycle questions around import order, duplicate
  registration, process fork, and test isolation.
* Backend-supplied cache versions can be wrong. The proposal makes the contract
  explicit but cannot automatically prove that a vendor bumped its identity.
* Restricting the first version to one accelerator policy leaves a real mixed-
  accelerator use case unresolved.
* A device-specific policy can make graph transformations less uniform across
  devices, which increases the test matrix and may preserve operators that
  receive less coverage in generic compiler passes.
* Adding an internal abstraction has maintenance cost even if only one
  out-of-tree backend initially uses every part of it. The implementation
  should therefore land incrementally and reuse existing compile-scoped
  decomposition, user-lowering, fallback, and backend-registration paths.


## Alternatives

### Register a vendor-specific Dynamo backend that calls `compile_fx`

This is the closest existing solution. It can pass a custom decomposition table
without changing core PyTorch. It also requires applications and frameworks to
select a backend other than `inductor`, and the wrapper must preserve Inductor
modes, options, caching, and integrations. It does not by itself define
per-node device lowering or a shared policy identity. A current-`main`
prototype of this alternative will be included before the RFC is ready so the
remaining core gaps are measured rather than assumed.

### Use only `user_lowerings`

This is a useful implementation building block for lowering overrides.
Lowering happens after AOTAutograd decomposition, so it cannot recover an ATen
operator that has already been decomposed.

### Continue mutating global tables and patch the cache key

This has the lowest core implementation cost, but every backend must coordinate
global mutation, import order, cache hashing, subprocess behavior, and cleanup.
It cannot provide reliable isolation for multiple devices in one process.

### Represent every vendor library call as a custom operator

Custom operators are appropriate for new semantics. They do not solve the case
where existing ATen models should choose different lowering/decomposition
strategies for the same operator based on device, unless users or graph passes
rewrite those models first.

### Add only device-specific decomposition tables

This would solve the earliest compiler stage but leave lowering/fallback and
cache identity as separate global mechanisms. The main value of the policy is
that the transformation which preserves an operator and the lowering which
consumes it can be validated and versioned together.


## Prior Art

This proposal extends patterns that already exist in PyTorch rather than
introducing a parallel compiler pipeline:

* [`compile_fx(..., decompositions=...)`](https://github.com/pytorch/pytorch/blob/8034d16f97e9c2c1a6800afdec681d7b6e0b2db5/torch/_inductor/compile_fx.py#L2823-L2852)
  already provides a compile-scoped decomposition input.
* [`GraphLowering.user_lowerings`](https://github.com/pytorch/pytorch/blob/8034d16f97e9c2c1a6800afdec681d7b6e0b2db5/torch/_inductor/graph.py#L1518-L1546)
  demonstrates a higher-priority lowering path.
* [`register_backend_for_device`](https://github.com/pytorch/pytorch/blob/8034d16f97e9c2c1a6800afdec681d7b6e0b2db5/torch/_inductor/codegen/common.py#L321-L434)
  establishes device-keyed backend registration inside Inductor.
* The [Intel GPU Inductor backend RFC](https://github.com/pytorch/pytorch/issues/114856)
  established the broader goal of reusing Inductor through device backend
  registration.
* The [default backend customization RFC](https://github.com/pytorch/pytorch/issues/136118)
  describes the user-experience cost of requiring a replacement backend for
  customization that conceptually belongs under the default Inductor path.

The proposed policy connects these existing mechanisms across the
decomposition/lowering/cache boundary.


## How we teach this

The primary audience is backend implementers, not ordinary model authors.
Documentation should add one internal backend-extension example showing:

1. register an immutable policy for a test device;
2. exclude one decomposition and provide the corresponding lowering;
3. assign and version the cache identity; and
4. test that another device retains default behavior.

The term *operator policy* emphasizes that this is a coherent set of decisions
across compiler stages. The documentation should explicitly distinguish it
from `torch.library` custom operators and from a user-selected Dynamo backend.


## Unresolved questions

* Should the policy live on `register_backend_for_device`, `InductorChoices`,
  or a dedicated registry?
* How should active accelerator discovery work before AOTAutograd for fake
  tensors, constants, symbolic inputs, and graphs with no tensor inputs?
* Should the initial cache identity be a backend-supplied version string, a
  structured tuple, or a registered provider identity?
* What exact precedence should apply between explicit `user_lowerings`, device
  policy, and any future compile-scoped overlay?
* Which fallback forms can be validated at policy construction time, and which
  require node metadata during lowering?
* How should a worker import and resolve an out-of-tree policy for subprocess
  compilation and AOT packaging?
* For multiple accelerator types in one graph, should the first version reject
  only conflicting policies or reject the case unconditionally?
* Is CPU plus one accelerator the correct first-version boundary, or should
  decomposition be tied to graph partitions instead of a compile-wide active
  accelerator?


## Resolution

This section will be completed after the commenting period.

### Level of Support

5: Unclear Resolution.

### Additional Context

The RFC is currently a draft pending current-`main` prototype evidence and
maintainer feedback.

### Next Steps

1. Attach the current-`main` prototype and patch-removal inventory.
2. Resolve the registry location, device discovery rule, and cache identity
   contract during RFC review.
3. If accepted, implement the proposal in the incremental sequence above.

#### Tracking issue

To be created when the RFC is ready for triage in `pytorch/pytorch`.

#### Exceptions

Compile-scoped mutable overlays and complete mixed-accelerator decomposition
semantics are not part of the initial implementation.
