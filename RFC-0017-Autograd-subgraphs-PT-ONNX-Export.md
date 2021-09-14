## Summary
Expanding the traced graph resulting from a `torch.autograd.function` in order to enable inline export from PyTorch to ONNX. Expansion of graph of such functions to happen only when the PT-ONNX exporter passes the relevant flag.

## Motivation
Currently, the PT-ONNX exporter enables the export of a `torch.autograd.function` using two different methods.

* `Static Symbolic Method` : This method involves adding a static method named symbolic to the function class. The symbolic method needs to return the utility of the function in terms of only ONNX operators.

* `PythonOp Symbolic Method` : This method involves registering the autograd function as PythonOp symbolic. The `symbolic_pythonop` function is used to register the autograd function.

Although these two current approaches the enable the export of a `torch.autograd.function` that consist of a symbolic function that is trivial enough to write in terms of other ONNX symbolic functions, certain functions consist of complicated structures such as loop blocks, if blocks or calling other modules within themselves.

A simplistic solution to this problem would be to expand the traced graph of the `torch.autograd.function` in terms of the operators that constitute the function so that there would be no requirement to write complex symbolic functions.

The goals are

* Goal: extract the subgraph from the traced graph of the `torch.autograd.function`.

* Goal: select the appropriate `traceValue` to be set in these situations.

* Goal: only affect code path of ONNX export when this feature is specifically turned on.

## User description

Users are not aware of this implementation detail.

## Implementation description

The expand_autograd_op flag is passed from the PT-ONNX exporter to the `trace` function

**ONNXTracedModule**
https://github.com/pytorch/pytorch/blob/master/torch/jit/_trace.py

```python
  class ONNXTracedModule(torch.nn.Module):
    def __init__(
        self,
        inner,
        strict=True,
        force_outplace=False,
        return_inputs=False,
        return_inputs_states=False,
        expand_autograd_op=False,
    ):
        super(ONNXTracedModule, self).__init__()
        # inner may be a Module, or it may be an arbitrary callable
        # If it's a Module, we get its parameters automatically, which lets
        # us avoid a special casing functions versus modules.
        self.inner = inner
        self.strict = strict
        self._force_outplace = force_outplace
        self._return_inputs = return_inputs
        self._return_inputs_states = return_inputs_states
        self._expand_autograd_op = expand_autograd_op
    // ...

    graph, out = torch._C._create_graph_by_tracing(
            wrapper,
            in_vars + module_state,
            _create_interpreter_name_lookup_fn(),
            self.strict,
            self._force_outplace,
            self._expand_autograd_op,
        )

```

**trace()**
https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/frontend/tracer.cpp

```c++
  std::pair<std::shared_ptr<TracingState>, Stack> trace(
    Stack inputs,
    const std::function<Stack(Stack)>& traced_fn,
    std::function<std::string(const Variable&)> var_name_lookup_fn,
    bool strict,
    bool force_outplace,
    bool expand_autograd_op,
    Module* self,
    const std::vector<std::string>& argument_names) {
  try {
  // ...

  getTracingState()->force_outplace = force_outplace;
  getTracingState()->expand_autograd_op = expand_autograd_op;
```   

**postTraceRecord**
https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/python_function.cpp

When the flag to expand the autograd function is `True`, the `traceValue` is not set.
```c++
  static void _trace_post_record(
    torch::jit::Node* node,
    PyObject* op_obj,
    const variable_list& input_vars,
    PyObject *output_objects,
    bool is_inplace,
    bool unpack_output) {
  if (!jit::tracer::isTracing()) {
    return;
  }
  // ...

  for (const auto i : c10::irange(num_outputs)) {
    PyObject* obj = PyTuple_GET_ITEM(output_objects, i);
    if (THPVariable_Check(obj) && !getTracingState()->expand_autograd_op) {
      Value* value = node->outputs()[i];
      const auto& tensor = THPVariable_Unpack(obj);
      if (tensor.defined()) {
        value->inferTypeFrom(tensor);
        jit::tracer::setValueTrace(tensor, value);
      }
    }
  }
  // ...
```

