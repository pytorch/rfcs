# Motivation

Currently TorchScript supports a generic Tensor type that can be of arbitrary shape, dtype, and layout. Graphs operating on generic tensors lack important shape, dtype, layout properties for further optimizations. On the other hand, we also observe that many properties of generic tensors can be inferred from those of input tensors.

We propose to augment torch.jit.script API with an optional Input Descriptor parameter that describe tensor input properties (e.g., dtype, device, shape, rank, layout) to the model being scripted. Input descriptor (this RFC) combined with a property propagation JIT pass (not included in this RFC) allow the JIT to specialize a graph of generic tensors to a graph of tensors with more known properties without adding much burden to user-level type annotation. This ability would aid many other downstream optimizations (e.g., AMP, control-flow removal, layout optimization, kernel fusion and codegen, tensor type/shape validation) as well enable more effective integrations of external optimizers that require static graphs.

# Design: `torch.jit.script` Input Descriptor

## User interface

Input descriptors is introduced as an optional parameter to `torch.jit.script`:
```
torch.jit.script(model,
    input_descriptors: Dict[param_name: str, meta: InputDescriptor]=None, ...)
torch.jit.script(function,
    input_descriptors: Dict[param_name: str, meta: InputDescriptor]=None,...)
torch.jit.load(...)
```
where

* `input_descriptors` specifies known meta information for certain parameters of a scripted function/method
   * `param_name` is the name of a parameter of a scripted method/function
   * `InputDescriptor` can be
      * a `MetaTensorType` object (see next) for any parameter of Tensor type, or
      * a typing Python object (e.g., `int`, `List[int]`, or `List[MetaTensorType]`), or
      * any Python object that matches the type of the parameter it describes

Note that when the type of a model input parameter is specified via `InputDescriptor`, this specification takes precedence over the type annotation specified at the signature of the scripted method/function. This is because
- `InputDescriptor` can specify rich tensor properties (e.g., `dtype`, `shape`) than the `Tensor` type annotation allowed in the TorchScript langauge;
- Since input descriptors are specified at `torch.jit.script()`, in principle, there can be multiple input descriptors specified for a model if the model is scripted in multiple ways.

## `MetaTensorType` object

We use `MetaTensorType` object to describe properties for a parameter of a scripted function/method. The complete set of input properties is not yet finalized (as it depends on the propagation pass implementation). If an attribute (e.g., `dtype`) has a `None` value, it means that no information about the attribute is known.
```
class MetaTensorType:
    def __init__(dtype=None, rank=None, shape:List[int]=None, device=None, requires_grad=None, layout=None, ...):
        self.dtype = dtype
        self.rank = rank
        self.shape = shape
        self.device = device
        self.requires_grad = requires_grad
        self.layout = layout
        ...
```
Note that shape is represented as a list of integers or strings to specify the (symbolic) length of each dimension.

* The length of the list corresponds to the rank of the tensor.
* Any non-negative integer number presents actual number of elements along that dimension.
* Any string represents a symbolic length. The same string represents the same symbolic length within the scope of input descriptors for one `torch.jit.script(...)` call-site.

Here are some examples of shape representation:
```
[100, 200] # 100 by 200 2D tensor
["i", "i", 100] # 3D tensor, innermost dimension 100, outer two most dimensions the same
["i1", "i2", "i3"] # 3D tensors with 3 independent lengths named as "i1", "i2", "i3"
```

## Using input descriptor

Input descriptor combined with a tensor property propagation pass essentially is a specialization engine that generates TorchScript IR graphs with richer tensor property information. Depending on what information is provided by the input descriptor, three types of specializations may happen:

* Type specialization, i.e., input descriptor specifying types
* MetaTensorType specialization, i.e., input descriptor specifying tensor meta-properties
* Value specialization, i.e., input descriptor specifying values

The input descriptor mechanism is quite flexible. One can specify rpoperties for a subset of parameters with a subset of tensor properties. It may also specify a particular parameter with a real Python value (e.g., `True` or `100`). The specialization is fully programmable and a model can be associated with multiple input descriptors, one for each `torch.jit.script()` call sites.

When invoking a scripted method/function with input descriptors, the actual arguments must satisfy properties specified in the input descriptor. This checking is crucial to ensure the correct usage of a scripted model specializd for a given input-descriptor. Since input descriptor is needed for type checking when using a scripted model, they need to be added to serialization and de-serialization support.

Consider the following example. We use input descriptor to describe the shape and dtype properties of parameter `x`, and to specify the value of parameter `y`. The TS graph generated by `torch.jit.script` is specialized to this particular input descriptor.
```
class MyModel(torch.nn.Module):
    def __init__(self):
        ...
    def forward(x, flag):
        if flag:
            return torch.add(x, 1)
        else:
            return torch.sub(x, 1)

myFlag = ...
...
meta = MetaTensorType([100,200], dtype=float)
# Scripted model specialized to the input_descriptor
scripted = torch.jit.script(MyModel(),input_descriptor={"x":meta, "flag":myFlag})
```
When invoking the input-descriptor specialized model, inputs are checked against the input descriptor as shown below:
```
myTensor = torch.randn([100, 200], dtype=float, device="cuda")
# Type checking success: myTensor satisfy all properties of meta
scripted(myTensor, myFlag)

badTensor = torch.ones([100], dtype=float)
# Type checking error: badTensor violates shape property of meta
scripted(badTensor, myFlag)
```

## Extract input descriptors from real inputs

`MetaTensorType` can be manually provided by model users where they have an understanding of the characteristics of model inputs (e.g., the shape and dtype of input tensors). 

We can provide helper functions to automatically extract `MetaTensorType` out of real inputs. Given a set of example tensors for a particular parameter, we may infer the maximal set of common properties that are true for all the inputs.

`MetaTensorType` can be gradually widened. For instance, if a new input fails the current tensor descriptor, it can be added to the example input to relax constraints specified in the tensor descriptor. This feature would make input-descriptor easier to use or to experiment with. Note that the caveat of input-descriptor widening is that it may decrease the information of input descriptors. But since `MetaTensorType` is a regular Python object, users may manually split a `MetaTensorType` into multiple ones and script a model multiple times, each with a different input descriptor.

# Design Discussions

Here are the different dimensions that we classify the design space of specifying more refined tensor properties through user inputs:

* *Caller- vs callee-side annotations*
    * Caller-side annotation offers the flexibility of specifying different input descriptors for different call sites. It is fundamentally a specialization mechanism based on call-sites
    * Callee-side annotation specifies typing constraints for the callee. It states the inherent typing truth of a function. This can be restrictive because callee functions from frameworks (incl. PyTorch) may have inherently dynamic properties (e.g., swift-tfp)
* *Equality vs inequality constraints*
    * Some descriptor can only specify equality constraints on types and properties
    * Other descriptor is more expressive in specifying constraints (e.g., swift-tfp)

## Why not parameterize TorchScript tensor types with dtypes?

Our first instinct was indeed to introduce parameterized (e.g., by `dtype` or `shapes`) tensor types to the TorchScript langauge (such as `torch.floatTensor`). Why did we abandon this approach? We realized two significant limitations of this approach:
* Annotating parameterized tensor types at all function interfaces are cumbersome and error-prone. Most TS users assume default `Tensor` types for unannotated parameters, if they want to annotate tensors with more refined tensor properties pass, they have to do a lot more explicit annotation;
* Many functions cannot be annotated with a specific `dtype` or `shape` at its function interface level. The more refined tensor properties really depend on those of the inputs to the model. For example, how does one annotate a more refined tensor type for any of the `torch.nn.` layers? Most of them are designed to work w/ tensors of many types or shapes. So even if users are willing to annotate more tensors with refined properties, they cannot do so for functions from common library codes that are polymorphic in terms of tensor properties by nature.

As such, we decide to choose caller-side annotation, i.e., to annotate types at the call-site of a module and use JIT to propagate the input properties throughout the graph. This way we treat JIT as a specialization engine and did not touch the type system of the TS langauge. Note that the type system of TorchScript IR does support more refined tensor properties and is (to some extent) independent of that of TorchScript language.

## Why not using empty or meta tensors to specify tensor properties?

One of our earlier design is to use empty tensors or meta tensors  (i.e., tensors created with `device=meta`) to specify tensor properties in the input descriptor. Here is one such design
```
class TensorDescriptor:
    def __init__(meta: Tuple[Tensor, List[str]]=None, shape_attr: List[int]=None)
```
where

* `meta.first` is a tensor (usually an empty tensor is sufficient) where a subset of its attributes are used to describe the input tensor
* `meta.second` is a list of attribute names that specify which subset of attributes of meta are used to describe properties of the input tensor. Ifc`meta_attrs` is None, then all attributes of meta except for tensor values are used to describe input tensors
    * `dtype()`, `device_type()`, `requires_grad()`, `sizes()`, `dim()`, `strides()`
    * `shape_attr` specifies symbolic shape information that cannot be expressed by regular tensors. It is represented as a list of integers where a positive integer represents static size of that dimension, and a negative number represents a symbolic dimension. The same negative number used in different dimensions of all input_descriptors specified to one `torch.jit.script` function represent the same symbolic shape variable. For instance,
  ```
   td = TensorDescriptor([torch.empty([100,200], dtype=float, device="cuda"), ["dtype", "sizes", "device_type", "requires_grad"])
    torch.jit.script(myModel, input_descriptor={"input1":td}
  ```

This design has the following problems:
* Empty-tensor still allocates storage
* Cannot represent symbolic shape information: neither `torch.empty` nor meta-tensor allow negative numbers in shape dimension
* Having to specify a subset of attributes in empty/meta-tensors (verbose)

## How to compare input descriptor and tracing?

Both input descriptor and tracing specializes a model for a particular execution context. For input descriptor, the specialization is guided by an explicit contract that is verifiable (i.e., actual arguments to a scripted model is validated against the input-descriptor associated with the scripted model), so input descriptor is safe.

Tracing, on the other hand, is unsound because the actual arguments to a trace model may not match the properties of the original inputs to produce the model.
