## Summary
Extending `TracingState.push_scope` and `TracingState.pop_scope` with function overload to take module inputs, attributes, and outputs as additional arguments. Enalbing more information of the module call to be recorded when `recording_scopes` is turned on.

## Motivation
A particular `Scope` object is created for each `nn.Module` `forward` call when `recording_scopes` is turned on. Each node within the `forward` call is annotated with that `Scope` object. This enables ONNX export to group nodes based on scope of original `nn.Module`.

However, with current approach, the information of inputs, outputs, and attributes of `nn.Module` call is missing. Graph analysis can be done to figure out which `Value` should be input/output, based on the scope location of their creation and use. Yet it is no longer possible to retrieve the input/output order or input/output name. For module attributes, it is even worse as some attributes are only used in module initialization, and not visible/retrievable in traced graphs.

The goals are

* Goal: enable jit (ONNX) passes to retrieve inputs/outputs/attributes information for each `Scope`.

* Goal: only affect code path of ONNX export when this feature is specifically turned on.

## User description

Users are not aware of this implementation detail.

## Implementation description

**TracingState.**
```c++
  py::class_<TracingState, std::shared_ptr<TracingState>>(
      m, "TracingState", py::dynamic_attr())
      .def(
          "push_scope",
          [](TracingState& s,
             const std::string& scope_name,
             const py::tuple& input_tuple,
             const py::dict& input_dict,
             const py::dict& attributes) {
            // ...
          })
      .def(
          "pop_scope",
          [](TracingState& s, const IValue& output) {
            // Retrieve Value* in IR Graph, given IValue.
            auto v = s.getValue(output);
            // ...
          })
```

**model.py**
```python
    # push scope
    from typing import get_type_hints
    annotations = get_type_hints(type(self))
    base_m_annotations = get_type_hints(torch.nn.Module)
    if annotations is not None and base_m_annotations is not None:
        [annotations.pop(k, None) for k in base_m_annotations.keys()]
    else:
        annotations = {}
    attrs = {k : getattr(self, k) for k in annotations.keys()}
    tracing_state.push_scope(name, input, kwargs, attrs)

    # ...

    # pop scope
    tracing_state.pop_scope(result)
```

**recording_scopes**. Currently decided by `recording_scopes = torch.jit._trace._trace_module_map is not None`. Perhaps need to add new flag to enable the new feature.

**Information propagation**. With new API, a mapping between `Scope` of module, and `Value*` of module inputs/outputs/attributes can be established. Now what's left is to preserve and propagate this info throughout jit (onnx) passes to be used.

One option could be extending `Graph` class with below info for each `Scope`.
```c++
value_list inputs;
std::unordered_map<std::string, Value*> kwargs;
std::unordered_map<std::string, IValue> attributes;
value_list outputs;
```
`Node::destroy()`, `Graph::freeNode()`, `Value::replaceAllUsesWith` etc must be modified accordingly to ensure the values recorded above are also correctly maintained. This approach needs some work for inputs/outputs, however it should be straightforward for attributes.

