
# [RFC-0059: Multi-Output Autotuning in Torchinductor]

**Authors:**
* @edderstack

## **Summary**
Much of Inductor's autotuning path is currently structured around a single returned tensor output. The idea of this change would be to add support for multi-output autotune, so templates and extern kernels can share the same operator contract without additional copy operations.
This implementation would involve carrying an output spec through choice generation, benchmarking, selected IR creation and Triton template codegen. Internally, output layouts would be normalised into a flat list for allocation and benchmarking, then restored to the operator's expected return structure. Selected choices would return multiple output IR nodes, with a parent op representing the kernel call and child nodes representing each real tensor output. 

## **Motivation**
Inductor currently has an incomplete model for autotuned kernels that produce multiple tensor outputs: all Triton templates are treated as though they produce only one real output, while additional outputs are handled as preallocated input buffers that the kernel mutates. This workaround leaks implementation details into lowerings by treating outputs as inputs and also prevents multi-output Triton templates and functional ATen/extern choices from sharing the same natural output contract. In practice, an extern candidate may need wrapper-side copies into prealllocated arrays to match the current Triton mutated-buffer convention. 

This change would also give inductor a cleaner IR model: multi-output ops would no longer need to be represented as a single returned output plus mutated input buffers. Secondary outputs would be modelled as real outputs of the selected operation, improving dependency tracking for downstream consumers.
Finally, this would improve the general template support for any future Triton template with multiple outputs. 


## **Proposed Implementation**

### 1. Add a multi-output spec
First need to add a formal method for the autotuning code to describe all outputs a candidate could produce, instead of just one output layout, in a way that does not interfere with the current single-output autotuning. 
This could be achieved by adding a output spec type with helpers to normalise the outputs, so that one-output or multi-output cases would have a standardised form:
i.e. (layout,) for single outputs or (layout1, layout2,) for multioutput cases. :
OutputSpecTree = ir.Layout | tuple[ir.Layout, ...] | list[ir.Layout]

Other required helpers would be to return the primary output layout for compatibility, device checks, grid sizing etc. :
- normalize_output_spec(spec) -> return both a flattened list for internal machinery and a recipe to rebuild the original shape later
- is_multi_output_spec(spec) -> return a Boolean to differentiate single and multi-output cases
- primary_layout(spec) -> return the first real output ir.Layout

### 2. Update choice construction
Update KernelTemplateChoice to store the multi-output spec defined above, and pass that into template.choice_or_none instead of the single layout.
Alter make_ktc_generator so that it could either have layout or output_spec passed in, to maintain past compatibility, then normalise them internally so if a layout was passed in it would be converted to a single layout output_spec:

    def make_ktc_generator(
        template,
        cs,
        extra_kwargs,
        overrides,
        inputs: KernelInputs,
        *,
        layout: Layout | None = None,
        output_spec: OutputSpec | None = None,
    ):
        if (layout is None) == (output_spec is None):
            raise AssertionError("pass exactly one of layout= or output_spec=")

        if output_spec is None:
            output_spec = layout

        flat_layouts, restore = normalize_output_spec(output_spec)
        ...

So if a single layout was passed in, it would become:

    output_spec = layout
    flat_layouts = (layout,)

### 3. Change ChoiceCaller
ChoiceCaller is currently defined with a single layout; it must be extended to multiple layouts while still maintaining compatibility with the single-layout case:

    self.output_spec = normalize_output_spec(output_spec)
    self.layout = primary_layout(output_spec) 

Would also need to define a function to be overridden by TritonTemplateCaller or ExternKernelCaller (like output_node()) for multi-output choices:

    def output_nodes(self) -> tuple[TensorBox, ...]:
        return (self.output_node(),)

Old paths would still call output_node(), new multi-output paths would call output_nodes().

### 4. Update autotune benchmarking
Need to change the timing harness so that it can benchmark a client that writes or returns both single and multiple output tensors.

During autotuning, after inductor has built the list of candidate ChoiceCallers, AlgorithmSelectorCache.get_inputs is called within AlgorithmSelectorCache.benchmark_in_current_process is called to create the real tensors that will be passed to each benchmark candidate; only one output is created:
    
    out = cls.benchmark_example_value(layout, hint_override=hint_override)

That one output is stored in AutotuneArgs, which is used during benchmarking to pass every choice the same inputs, as a list of tensors, and the same output, as a single tensor.
The first step would then be to update BenchmarkTensors so that it could represent a tuple of output tensors; then have get_inputs take the output layout, normalise it so that single or multiple output layouts are treated as a tuple, then loop over these layouts to produce a tuple of output tensors:

    flat_layouts, restore = normalize_output_spec(output_spec)

    outs = tuple(
        cls.benchmark_example_value(layout, hint_override=hint_override)
        for layout in flat_layouts
    )

Would also need to update the expected output tensor (also constructed in AlgorithmSelectorCache.get_inputs) for correctness verification, replacing the single output tensor cloned into expected with a tuple of output tensors.


During benchmarking, it unpacks one output in select_algorithm.py:AlgorithmSelectorCache.benchmark_choice and zeros it:

    inputs, output = benchmark_tensors.unpack()
    output.zero_()

This would need to be replaced with a loop over output tensors to zero each individually. 


It then runs a benchmark on that output:

    result = choice.benchmark(*inputs, out=output)

In this case, a helper could be used so that single-output cases would be passed a single output tensor (instead of a single-length tuple of tensors) and new multi-output cases would be passed a tuple of tensors.

The output tensor is compared against the expected values in select_algorithm.py:AutotuneArgs.verify:

    torch.testing.assert_close(self.extern.output_tensor, self.expected, **kwargs)

Currently, verify() does not receive the output tensor from the candidate that just ran; it compares self.expected to self.extern.output_tensor, not output from the just run candidate. This works because get_inputs() constructs out_extern as an as_strided view over the same base storage as out:

    out_extern = torch.as_strided(out_base, out.size(), out.stride(), out_offset)

So triton.output_tensor and extern.output_tensor view the same backing storage, then when a triton candidate writes into out, the same data is visible through out_extern, and it is checked in verify() as self.extern.output_tensor.
This relationship would either need to be preserved for each output in the multi-output case, or the verification changed to compare the actual outputs used by the just-run candidate.

For Extern Functional benchmarking, autotune_process.py:ExternKernelBenchmarkRequest.benchmark currently only handles one functional return:

    out_new = algo(*input_tensors)
    if out is not None:
        torch._C._dynamo.guards.assert_size_stride(
            out_new, tuple(out.size()), tuple(out.stride())
        )
        out.copy_(out_new)  # for correctness checking

This would need to be modified to handle one-or-many outputs; so would need to normalise out_new and out for size/stride verification before looping through and comparing them, before copying the new results into out:

    out_new = algo(*input_tensors)
    actual_outputs = normalize_returned_outputs(returned)
    expected_outputs = normalize_benchmark_outputs(out)
    for actual, expected in zip(actual_outputs, expected_outputs):
        torch._C._dynamo.guards.assert_size_stride(
            actual,
            tuple(expected.size()),
            tuple(expected.stride()),
        )
        expected.copy_(actual)

For Triton benchmarking, autotune_process.py:make_run_fn currently only accepts a single output buffer for the output tensor out, then returns the callable run_fn with that single output tensor defined. First, make_run_fn needs to accept one-or-many output tensors:

    def make_run_fn(
        self,
        *input_tensors: torch.Tensor,
        out: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Callable[[], None]:

Then out should be normalised:

    if isinstance(out, torch.Tensor):
        output_tensors = (out,)
    else:
        output_tensors = tuple(out)

Finally, use the tuple output_tensors when calling triton:

    run_method(
        *input_tensors,
        *output_tensors,
        *extra_args,
        **warmup_arg,
        stream=stream,
        benchmark_run=True,
    )

This will allow the single output tensor cases to remain unchanged, while the multi-output cases can receive multiple output buffers to write into. The generated triton kernel signature would also need to match these output arguments.


### 5. Triton Template Codegen
Currently both benchmark/precompile codegen and final graph codegen assume one output. select_algorithm.py:TritonTemplate.generate_and_load receives a single ir.Layout, and creates one fake output buffer:

    fake_out = ir.Buffer(name="buf_out", layout=layout)
    
which is then passed into TritonTemplateKernel in make_kernel; inside TritonTemplateKernel, this becomes one output node. Helper functions within TritonTemplateKernel assume a single output node: TritonTemplateKernel.size(None) and TritonTemplateKernel.stride(None) refer to the size and stride of that output, and TritonTemplateKernel.store_output stores a single output node. Currently, extra outputs are passed as inputs and marked as mutated. 

To adapt this to multiple outputs, should normalise at the API boundary (i.e. before generate_and_load), and so pass a tuple of layouts into generate_and_load, then inside can have:

    primary_layout = output_layouts[0]
    fake_outputs = tuple(
        ir.Buffer(name=f"buf_out{i}", layout=layout)
        for i, layout in enumerate(output_layouts)
    )

Also need to update the generated cache key for multiple layouts.

TritonTemplateKernel currently only stores one output_node, need to change it to store multiple nodes while maintaining compatibility with a single node:

    self.output_nodes = output_nodes
    self.output_node = output_nodes[0]  # compatibility

TritonTemplateKernel.def_kernel only assigns names to input nodes; need to add named output nodes to add names output buffers to the generated kernel argument list (rather than having additional outputs as named inputs).
Should also add explicit output size/stride/dtype helpers to avoid ambiguity between input and output names. store_output would also need to be updated for named output storage rather than defaulting to a single output.

### 6. Benchmark metadata
The benchmark metadata defined in select_algorithm.py:TritonTemplate.generate currently gets one output tensor meta:

    output_tensor_meta=TensorMeta.from_irnodes(layout),

The benchmark request must know the dtype/shape/stride/device of every output tensor not just the first one, as it is used when autotuning runs in a subprocess, or when the benchmark request needs to recreate fake tensors. 
Also, autotune_process.py:BenchmarkRequest accepts a list of output_tensor_metas, but collapses them to the first one, which would be inappropriate for true multi-output templates. 
So, need to alter BenchmarkRequest so that it does not collapse the output metas to the first element,  but instead normalises both single and multioutput cases to a list of metas:

    if isinstance(output_tensor_meta, TensorMeta):
        self.output_tensor_meta = [output_tensor_meta]
    else:
        self.output_tensor_meta = list(output_tensor_meta)

Then could have the definition of output_tensor_meta in select_algorithm.py:TritonTemplate.generate as:

    output_tensor_meta=TensorMeta.from_irnodes(output_layouts)

So TritonBenchmarkRequest.make_run_fn can pass all benchmark output tensors to the generated kernel.
autotune_process.py:BenchmarkRequest.benchmark must also change the definition of the output tensor from the meta from

    out = self.output_tensor_meta.to_tensor()

to

    output_tensors = tuple(meta.to_tensor() for meta in self.output_tensor_meta)
    out = output_tensors[0] if len(output_tensors) == 1 else output_tensors

Also need to update the type annotation in make_run_fn to represent this change. 

### 7. Selected IR
After autotuning has picked the winning ChoiceCaller, inductor must return IR for all outputs, instead of just the first output. 
select_algorithm.py:AlgorithmSelectorCache.__call__ references a single output node with choice.output_node() at several points throughout, these need to be replaced with a reference to output_nodes() and a helper function to restore the output structure to that expected by the operator's return contract (i.e. a single TensorBox instead of a single TensorBox within a tuple for single output cases).
The IR shape must reflect the number of outputs; one child node should be created per real tensor output, then returned to the lowering. Current Triton selected IR creation in select_algorithm.py:TritonTemplateCaller.output_node currently creates one TritonTemplateBuffer with one layout; for multioutput, the parent should use MultiOutputLayout, to signal that this IR node represents an operation that produces multiple outputs, while the children for Triton should use AllocatingMultiOutput, because Triton writes into output pointers.
ExternKernelMultiOut already uses this pattern for .out() variant ops.

The single output assumptions must also be corrected in ir.py:TritonTemplateBuffer, which should take output_spec instead of a single Layout when initialising and defines the template buffer itself as the output tensor; for multi-output Triton the parent is only the operation container, the real outputs are the children. Initially, should probably disable epilogue fusion for multi-output triton templates.

select_algorithm.py:ExternKernelCaller also assumes one final result. 
There are two cases:
- If the extern choice is functional and returns a tuple, create a parent extern op with MultiOutputLayout, then create MultiOutput children.
- If the extern choice has real .out parameters, create allocating children, similar to ExternKernelMultiOut.
Tuple-return machinery already exists in FallbackKernel.create; it recursively creates MultiOutput children from structured outputs in FallbackKernel.generate_output, but ExternKernelCaller.output_node() currently wraps the result as one TensorBox, so it would need a separate multi-output path.

Also need to handle the deferred MultiTemplateBuffer path in select_algorithm.py:AlgorithmSelectorCache: it returns a MultiTemplateBuffer instead of picking a concrete choice immediately. Would either need to disable this for multi-output choices initially, or add functionality to MultiTemplateBuffer so that it could also be a multi-output parent with child output nodes.

### 8. Testing
Need to include tests that verify:
- Single-output autotuning still works
- Multi-output Triton selected path works
- Multi-output extern selected path works
- All selected return paths use the helper
- An implemented multi-output operation returns the correct values


## **Example Multi-Output Operations**

### flex_attention Forward
**Outputs:** (out, logsumexp, max_scores)

**Current behaviour:**
- Treats logsumexp and max_scores as mutated buffers, then lowering manually recombines output tuple
- No ATen ExternKernelChoice is included in autotuning

### flex_attention backward
**Outputs:** (grad_query, grad_key, grad_value)

**Current behaviour:**
- Treats grad_key and grad_value as mutated buffers, then lowering manually recombines output tuple
- No ATen ExternKernelChoice is included in autotuning

### flex_flash_attention Forward
**Outputs:** (template_output, lse)

**Current behaviour:**
- Treats lse as a mutated buffer, then lowering manually recombines output tuple
- No ATen ExternKernelChoice is included in autotuning

### flex_flash_attention Backward
**Outputs:** (grad_query, grad_key, grad_value)

**Current behaviour:**
- Treats grad_key and grad_value as mutated buffers, then lowering manually recombines output tuple
- No ATen ExternKernelChoice is included in autotuning

### flex_decoding
**Outputs:** (output, logsumexp)

**Current behaviour:**
- Template writes intermediate buffers, with only one returned as output and the others treated as mutated input buffers
- After autotuning, logsumexp and output are computed from intermediates using seperate lowerings/reductions
- No ATen ExternKernelChoice is included in autotuning

### convolution_backward
**Outputs:** (dx, dw, db)

**Current behaviour:**
- Splits operation into three independent autotune problems, one for each output
- aten.convolution_backward.out included in the autotuning with only one output requested at a time

## **Metrics**
What are the main metrics to measure the value of this feature? 
- No regressions for existing single output autotuning tests.
- Correctness tests for all outputs, instead of just the primary output.
- No extra runtime copies in the ATen/extern path
- Benchmark parity between current mutated-buffer path and new multi-output path

## **Drawbacks**
The current autotune path is single-output in many places, which would need to be altered to accomodate both the new multi-output case and the past single-output to maintain compatibility:
- torch/_inductor/select_algorithm.py:BenchmarkTensors stores one output tensor
- torch/_inductor/select_algorithm.py:AutotuneArgs.from_choice_args takes one Triton output and one extern output
- torch/_inductor/ir.py:ChoiceCaller stores one Layout and returns one output_node (defined in torch._inductor.select_algorithm.ExternKernelCaller and torch._inductor.select_algorithm.TritonTemplateCaller)
- torch/_inductor/kernel_template_choice.py:KernelTemplateChoice stores one Layout and passes it into the ChoiceCaller generator torch._inductor.select_algorithm.ExternKernelChoice.choice_or_none
- torch/_inductor/choices.py:InductorChoices.get_template_configs only takes one Layout from kernel_inputs.output_layout
- torch/_inductor/select_algorithm.py:TritonTemplate.generate_and_load creates one fake output buffer
- torch/_inductor/select_algorithm.py:TritonTemplateKernel stores one self.output_node
- torch/_inductor/select_algorithm.py:benchmark_choice zeros and passes one output to benchmark.choice
- torch/_inductor/select_algorithm.py:AlgorithmSelectorCache ultimately returns one choice.output_node() (through the thin wrapper torch/_inductor/select_algorithm.py:autotune.select_algorithm)
- torch/_inductor/autotune_process.py:ExternKernelBenchmarkRequest only handles one functional return, and collapses lists of output_tensor_meta to the first element 
- torch/_inductor/autotune_process.py:TritonBenchmarkRequest.make_run_fn only expects a single output
- torch/_inductor/select_algorithm.py:AlgorithmSelectorCache.__call__ references a single output node with choice.output_node() at several points throughout

Therefore, this patch will result in a wide API surface, which will require careful testing to validate that the multi-output state is carried consistently through all layers. Due to the size of this change, it may need to be splut over several PRs.

Some other potential drawbacks are:
- Most existing autotuned operations are single-output, but this patch changes code used by all of them; it must be carried out carefully to maintain compatibility.
- Cache key complexity, noted in "Unresolved Questions" below.
- Increased complexity in correctness verification: there may be variation in precision and dtype between the outputs
- Increased scheduler/IR complexity: adding child nodes for each output will increase the complexity of the graph for multiple output cases

## **Unresolved questions**
- Should the deferred MultiTemplateBuffer path in select_algorithm.py:AlgorithmSelectorCache be implemented for multi-output cases? It returns a MultiTemplateBuffer instead of picking a concrete choice immediately. Would either need to disable this for multi-output choices initially, or add functionality to MultiTemplateBuffer so that it could also be a multi-output parent with child output nodes.
- How should epilogue fusion be handled? Can it safely target only the primary output while the secondary output is also produced? 
- Currently parts of the selector assume that the autotune cache is based mostly on inputs, not on output layout. Multi-output templates can make the output contract part of the benchmarked behaviour, with correctness and timing affected by output count, dtype, shape, stride etc. Should the generation of cache keys include the full output spec?   

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
