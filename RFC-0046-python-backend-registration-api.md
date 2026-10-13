# A Python backend registration API for PyTorch

**Authors:**
* @qihqi

## **Summary**
A comprehensive set of APIs for defining a PyTorch backend entirely in
Python. The design goal of this APIs are:

1. Easy to write a minimal backend: backend developers should be able to implement
   a PyTorch backend with only essential work. PyTorch should infer it can infer
   based on the minimal set of required things implemented.

2. Maximaze customization with optional components: a backend developer can choose
   to implement more than neccesary for his/her backend, say, to improve performance.

3. Everything should be doable in Python for fastest experimentation. C++ APIs
   can be made available for speed with the understanding that it's optional for one to use.
   i.e. everything doable in C++ is also doable in Python.

In other words, we start by asking the question:
“What is the absolute neccesary amount of information that a backend has to tell PyTorch,
for it to create tensors and run math on that backend?”

If a backend provided those, PyTorch should infer the rest for it do its job.

In this RFC, I hope to discuss the following items with the community:
1. What is considered the "minimal set", and how to infer the rest.
2. Proposed implementation.
3. Look and feel of the registration API itself.


## **Motivation**

### Background: existing backend registration APIs

There are a few ways to register stuff to PyTorch and customize
it's behavior. Notably:

* C++ backend registration API: https://docs.pytorch.org/tutorials/advanced/privateuseone.html
  * Eager mode backend
  * You can define both data representation and operators in this API

* Python custom operators: https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html
  * You can only define operators.
  * cannot define data representation

* Dynamo custom backend: https://docs.pytorch.org/docs/stable/torch.compiler_custom_backends.html
  * a torch.compile backend: you can define how to compile a `fx graph` -> callable
  * cannot define data representation

In summary:

|   | C++  | Python  |
|---|---|---|
|  Define data representation | yes  | No  |
|  Define single operator | yes  | yes  |
|  Define graph compiler |  No |  yes |
|---|---|---|

So now, to define a full-featured backend for a device,
using torch_xla and GoogleTPU as the example here:

1. Define the `XLATensor` in C++, that contains the internal data representation
2. Define and register operators that knows how to do math on XLATensor, (can be done in both C++ or Python)
3. Define a dynamo backend, in Python.

In the above process, one also have to learn a lots of implementation
details and concepts of PyTorch, such as `DeviceGuard` and `DispatchKey`.

If we ask the question of, “What is the absolute neccesary amount of information that a backend has to tell PyTorch,
for it to create tensors and run math on that backend?”
The answer are more minimalistic:

1. We have to define a data structure that is opaque to PyTorch,
   representing our on-device data.
2. We have to implement the ops for some op set (Core ATen / Prim only?).
3. We have to tell PyTorch how to move a CPU tensor to our Backend Tensor.

Can we instead, have an API that let the backend developer tell us exactly that?

Next question is that, what are the optional things that PyTorch can infer,
but a backend can choose to tell PyTorch for improve performance and usability?

1. Tensor constructors: if absent PyTorch can create tensors on CPU and transfer via 3 above.
   if present, it makes it faster / more natural
2. non-core Aten ops: maybe regitering direct lowering for say, `einsum` can improve performance?
3. Dynamo backend: if I only have an eager backend, then PyTorch can produce a dynamo backend
   identical to `torch.compile(backend='eager')` which is already there.
   Conversely: if I only provide a dynamo backend, PyTorch should be able to generate an
   eager backend by calling my compiler with graphs with only one node.

## **Proposed API**

Pytorch provides:

```
class BackendEnvironment:

  def __init__(self, name, blob_type, cpu_to_backend, backend_to_cpu):
    ...

  def register_operator(self, aten_op_or_torch_function, callable):
    ...
```

Backend developer writes (below using Apple MLX as example):

```python
def torch_to_mlx(tensor):
  return mx.array(tensor.numpy())


def mlx_to_torch(array):
  return torch.from_numpy(np.array(array))


environment = BackendEnvironment(
  'mlx',
  mx.array,
  torch_to_mlx,
  mlx_to_torch
)

environment.register_default_decompositions()

from torch.ops import aten
def register_op(op):
  def inner(func):
    environment.register_op(op, func)
    return func
  return inner

@register_op(aten.detach)
@register_op(aten.detach.default)
@register_op(aten.clone.default)
def aten_detach_default(x, *args):
  return x

@register_op(aten.view.default)
@register_op(aten._unsafe_view.default)
def aten_view(x, shape):
  return mx.reshape(x, shape)

...
```
Let's parse the above:

This section:

```python
environment = BackendEnvironment(
  'mlx',
  mx.array,
  torch_to_mlx,
  mlx_to_torch
)
```
is saying:

1. My backend's name is 'mlx'
2. My opaque data is `mx.array`; this could be any Python class, so if a backend
   want to have a tuple of elements etc can also do that.
3. the 2 functions that maps CPU torch.Tensor to my blob and back are these 2.
   so if a user do `tensor.to('mlx')` PyTorch would knows what to call.

4. Everything else, including tensor constructors, please refer registered operators.
   If a particular tensor constructor doesn't exist, run the CPU one and move to device.

A strawman (runnable) version of this API is located here: https://github.com/qihqi/tnt/blob/main/torch_mlx.py
Although the above has used the **alternative implementation** described below.

## **Proposed Implementation**

1. Tensor creation:

When we create a backend tensor, we will first create the blob, then attach it
to an empty CPU tensor.

2. Operator wrapping / unwrapping

On call of a particular operator, say `aten.add`; PyTorch will:

1. Intercept call
2. unwrap the tensor to get the backend blob,
3. call the registered op passing down the blob.

This is down via `torch.library` registry to have a handle capturing of
each operators.

A strawman implementation of this is illustrated in this unit test:
https://github.com/pytorch/pytorch/blob/main/test/test_privateuseone_python_backend.py
using `numpy` as the backend array.

3. Dynamo integration

Currently the above implementation does not work on Dynamo, there probably
will need some minor changes in dynamo itself.

However, in the limit, having a numba backend dynamo backend for numpy should be doable.


## **Drawbacks**

There are so many ways to extend PyTorch (https://docs.google.com/presentation/d/1piuv9nBzyoqdH49D1SoE5OZUPSMpOOFqfSKOhr-ab2c/edit)
this is adding yet another way to do it. It currently utilizes existing mechanisms.


## **Alternatives**

I have tried a tensor subclass based mechanism, used in the MLX demo
above. While works, it does not work well with `torch.compile`.
The spirit of tensor subclass is [“wrapper around other Tensor that eventually desugar into ops on plain Tensors”.](https://dev-discuss.pytorch.org/t/embrace-tensor-subclass-as-a-python-device-registration-api/2771/2?u=qihqi); so we
should probably respect that convention.


## **Prior Art**
* Alban's new_device demo: https://github.com/albanD/subclass_zoo/blob/main/new_device.py -- uses torch_dispatch
* Tinygrad's torch backend: https://github.com/tinygrad/tinygrad/tree/82f10cfe2ee309fc048c4b04279e70102e84ca98/extra/torch_backend -- uses mechanism proposed by this RFC
* Torchax: https://google.github.io/torchax/ -- uses torch dispatch

## **How we teach this**
One thing to distinguish is in terminology is `backend` vs. `device`.
A backend is a way to run compute, it could be for a device or could
be for an existint device.

## **Unresolved questions**
What parts of the design do you expect to resolve through the RFC process before this gets merged?

The primary goal of this RFC is to understanding PyTorch maintainers' and the community's
opinion on this approach.

The questions to maintainers are:
* Is this a good idea in general?
* If yes is there interest in formalizing and implementing it together?
* Is the API look-and-feel OK? Any improvement on those?

The questions to community is:
* Do you see yourself using something like this, if yes what are the usecases?

## Resolution

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

### Next Steps

#### Tracking issue
<github issue URL>

#### Exceptions
