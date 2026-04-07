

<details>
<summary>Instructions - click to expand</summary>

# RFC-0051: Structured `call_hierarchy` metadata for FX nodes

**Authors:**
* @dengand-aws
* @nklshy-aws

## Summary

Add a new `node.meta["call_hierarchy"]` field to FX nodes that provides the complete, interleaved module and function call chain that produced each op during Dynamo tracing. This is provenance information that no existing metadata field captures correctly.

## Motivation

Every FX node produced by Dynamo carries metadata describing where it came from: `nn_module_stack`, `stack_trace`, and `source_fn_stack`. None of these answer a question that many downstream tools need: what is the complete, ordered sequence of modules and functions that produced this op?

The existing metadata fields each capture part of this information, but none of them capture it fully:

- `nn_module_stack` correctly captures module nesting, but only module nesting. Helper functions like `apply_rotary_pos_emb` or `rotate_half` are invisible. There is no way to tell which function within a module scope produced a given op, or to distinguish the 1st invocation of a helper from the 3rd.
- `stack_trace` is an unstructured string built by walking the tx parent chain, but it cannot be parsed to reconstruct the full hierarchy for two reasons. First, module frames are actively filtered out: `is_co_filename_from_nn_modules()` drops any frame from `torch/nn/modules/`, so if a model calls `self.linear(x)`, the `Linear.forward` frame does not appear in `stack_trace`. You could try to correlate `stack_trace` frames back to `nn_module_stack` entries, but `nn_module_stack` stores `(path, class)` tuples with no source locations, so there is no reliable key to join on. Second, `stack_trace` has no invocation counts: if a model calls `helper_fn(x)` three times, all three produce the same trace string (same filename, function name, line number), with no way to distinguish the 1st call from the 3rd.
- `source_fn_stack` captures the leaf function or module class before decomposition. 

Modules and functions exist in a single interleaved call chain at trace time, but no existing metadata field preserves that interleaving. `nn_module_stack` and `stack_trace` are built by separate code paths with different filtering decisions and no shared ordering. 

AS an example, suppose we are given a call chain like `ModuleA.forward > helper_fn > ModuleB.forward > rotate_half`, `nn_module_stack` produces `[ModuleA, ModuleB]` and `stack_trace` produces `helper_fn` and `rotate_half` as separate frames, but there is no way to determine whether `helper_fn` sits between the two modules or after both.

This provenance information matters for:

- Profiling: mapping lowered ops back to model structure so users can identify bottlenecks at the layer or helper-function level.
- Debugging: tracing a failed or incorrect op to its exact location in the model, e.g. `Attention.q_proj > apply_rotary_pos_emb > rotate_half`
- Visualization: grouping ops by their origin in the model hierarchy when displaying FX graphs or lowered IR alongside model structure.
- Compiler heuristics: backends that partition, fuse, schedule, etc. based on model structure need structured scope information

## Proposed Implementation

Each FX node gets a `call_hierarchy` field in `node.meta`: an ordered list of dicts from outermost to innermost scope. Each entry is either a module entry or a function entry:

```python
[
    {"type": "module", "class": "Qwen2Model", "attr": "model", "count": 0},
    {"type": "module", "class": "Qwen2Attention", "attr": "self_attn", "count": 0},
    {"type": "function", "name": "apply_rotary_pos_emb", "count": 0},
    {"type": "function", "name": "rotate_half", "count": 1},
    {"type": "module", "class": "Linear", "attr": "q_proj", "count": 0},
]
```

The `count` field is a 0-indexed invocation count. For modules, this tracks shared module instances called multiple times. For functions, this tracks repeated calls to the same function.

The implementation builds on existing Dynamo infrastructure. During proxy creation, `stack_trace` is already built by walking the InstructionTranslator (tx) parent chain. `call_hierarchy` is built during the same walk: at each frame, if `nn_module_stack` has new keys compared to the previous frame, the frame is a module entry; otherwise it is a function entry (filtered for torch-internal and non-meaningful frames). Because both module and function entries come from the same ordered traversal, the interleaving is correct by construction. This works regardless of how modules override `__call__` or `forward`. Module invocation counts come from `nn_module_stack`'s existing `@N` key suffix. Function invocation counts are tracked via a `function_call_counts` dict shared across the tx chain, incremented when `InliningInstructionTranslator` is created.

We can gate the feature by `torch._dynamo.config.record_call_hierarchy` (default `False`). When disabled, zero additional work is performed. Or, if overhead is low enough have it on by default. 

`call_hierarchy` is added to `_COPY_META_FIELDS` in `fx/proxy.py` so it survives graph transformations. Backward nodes receive `fwd_call_hierarchy` from their corresponding forward node, following the established `fwd_nn_module_stack` pattern in `copy_fwd_metadata_to_bw_nodes`.


## Metrics

- Compile-time instruction count overhead when enabled, measured via the existing `pr_time_benchmarks` suite on representative models (deep module nesting and deep function inlining).
- Zero overhead when disabled, verified by the same benchmarks.

## Drawbacks

- Adds a new metadata field to an already metadata-rich system. We should make sure this data isn't easily derivable from other metadata, or that it wouldn't be better to unify this with other fields. 
- Per-node memory cost when enabled: one list of small dicts per node, proportional to call depth.

## Alternatives

- Parse `stack_trace` strings at consumption time. This is what consumers do today and it is fragile, lacks invocation counts, and cannot be correctly interleaved with `nn_module_stack`.
- Extend `nn_module_stack` to include function entries. This might change the semantics of an existing field that many consumers depend on.

## Prior Art

- `nn_module_stack`
- [#87659](https://github.com/pytorch/pytorch/issues/87659): hierarchy preservation in FX graphs.

## How we teach this

The feature is opt-in and targeted at compiler backend authors and tooling developers, not end users. Documentation should cover:

- The data format with examples.
- How it relates to `nn_module_stack` and `stack_trace` (complements, does not replace).
- The `fwd_call_hierarchy` field for backward nodes in training.

## Unresolved questions

- Should the feature eventually be always-on if overhead is shown to be negligible?
- Is there any better way to encode this information?

## Resolution

TBD

### Level of Support

TBD

### Next Steps

TBD

#### Tracking issue

TBD

</details>





# [Title]

**Authors:**
* @nickname
* @nickname 


## **Summary**
A short paragraph or bullet list that quickly explains what you're trying to do.


## **Motivation**
What motivates this proposal and why is it important?
How should users and developers think about this feature, how would it impact the way PyTorch is used?
Explain impact and value of this feature


## **Proposed Implementation**
This is the bulk of the RFC. Explain the design in enough detail for somebody familiar with PyTorch to understand, and for somebody familiar with the implementation to implement. 
This should get into specifics and corner-cases, and include examples of how the feature is used, and how it will interact with other features. Any new terminology should be defined here.
Consider:
*   using examples and diagrams to help illustrate your ideas.
*   including code examples, if you're proposing an interface or system contract.
*   linking to project briefs or wireframes that are relevant.


## **Metrics **
What are the main metrics to measure the value of this feature? 


## **Drawbacks**
Are there any reasons why we should not do this? Here we aim to evaluate risk and check ourselves.

Please consider:
* is it a breaking change?
* Impact on UX
* implementation cost, both in terms of code size and complexity
* integration of this feature with other existing and planned features


## **Alternatives**
What other designs have been considered? What is the impact of not doing this?


## **Prior Art**
Discuss prior art (both good and bad) in relation to this proposal:
* Does this feature exist in other libraries? What experience has their community had?
* What lessons can be learned from other implementations of this feature?
* Published papers or great posts that discuss this


## **How we teach this**
* What names and terminology work best for these concepts and why? How is this idea best presented?
* Would the acceptance of this proposal mean the PyTorch documentation must be re-organized or altered?
* How should this feature be taught to existing PyTorch users?


## **Unresolved questions**
* What parts of the design do you expect to resolve through the RFC process before this gets merged?
* What parts of the design do you expect to resolve through the implementation of this feature before stabilization?
* What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?


## Resolution
We decided to do it. X% of the engineering team actively approved of this change.

### Level of Support
Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.


#### Additional Context
Some people were in favor of it, but some people didn’t want it for project X.


### Next Steps
Will implement it. 


#### Tracking issue
<github issue URL>


#### Exceptions
Not implementing on project X now. Will revisit the decision in 1 year.
