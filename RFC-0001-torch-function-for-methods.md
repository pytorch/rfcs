# RFC 0001 — `__torch_function__` for methods of the `torch.Tensor` class
## Abstract
This RFC describes changes necessary to allow `__torch_function__` to be used by methods of `torch.Tensor` in an attempt to make subclassing more accessible to the users of the class. This entails making an API for subclass views public, and a change in the signature of `__torch_function__`.

## Motivation and Scope
Quoting [[1]], [[2]] and [[3]], the potential goals of this proposal are:

1. Support subclassing `torch.Tensor` in Python
2. Preserve `Tensor` subclasses when calling `torch` functions on them
3. Preserve `Tensor` subclasses when calling `numpy` functions on them
4. Use the NumPy API with PyTorch tensors (i.e. NumPy API calls dispatch to `torch` functions)
5. Use the PyTorch API with `torch.Tensor`-like objects that are _not_ `Tensor` subclasses
6. Reuse NumPy ufunc implementations directly from PyTorch
7. Allow operations on mixed array types, e.g. `tensor + ndarray`
8. Preserve `Tensor` subclasses when calling `Tensor` methods.
9. Propagating subclass instances correctly also with operators, using views/slices/indexing/etc.
10. Preserve subclass attributes when using methods or views/slices/indexing.
11. A way to insert code that operates on both functions and methods uniformly (so we can write a single function that overrides all operators).

We propose to solve this problem with the following changes to PyTorch:

1. Make methods and operators of `torch.Tensor` go through the `__torch_function__` machinery.
2. Add a `types` argument to `__torch_function__`, to make it match NumPy's `__array_function__`.
3. Make `torch.Tensor._make_subclass` public API.
4. Make `torch.Tensor` gain a generic implementation of `__torch_function__`.

## Usage and Impact
Once this proposal is merged, users of subclasses of `torch.Tensor` will have a much more streamlined experience. Namely, the following code example will work as-is, without the need for any further modification:

```python
class SubTensor(torch.Tensor):
    a = 1

t = SubTensor([1])
s = t.sum()
isinstance(s, SubTensor)  # True
s.a  # 1
i = t[0]
isinstance(i, SubTensor)  # True
i.a  # 1

s2 = t + torch.Tensor(1)
isinstance(s2, SubTensor)  # True
s2.a  # 1

s3 = torch.Tensor(1) + t
isinstance(s3, SubTensor)  # True
s3.a  # 1
```

Additionally, it will provide subclass authors hooks to run whenever methods or operators are called, and to modify the result to their specific use-case, perform logging, or otherwise change the result or the action of the method. For example:

```python
import logging

class LoggingTensor(torch.Tensor):
    def __torch_function__(self, func, types, args, kwargs):
        logging.info(f"func: {func!r}, args: {args!r}, kwargs: {kwargs!r}")
        return super().__torch_function__(
            func,
            tuple(
                t for t in types if not issubclass(t, LoggingTensor)
            ),
            args,
            kwargs
        )
```

## Backwards Compatibility
### With PyTorch `master` as of writing
PyTorch `master` pointed to commit hash `957a07ffbd13d8a805f4d718e0282efc5d2bff85` at the time of writing. Any classes implementing `__torch_function__` based on the usage in this commit hash will break completely, due to the differing signature of the protocol. However, as a release hasn't been made with `__torch_function__` in it, this is a minor-impact issue. This brings the design of `__torch_function__` more in line with NumPy's `__array_function__`, and one familiar with NumPy's protocol could transition to PyTorch's take on it without too many surprises, with the caveat that it could also receive methods rather than functions.

### With NumPy
As we are using a different protocol compared to NumPy `__torch_function__` vs `__array_function__`, there is no difference to the usage for those using NumPy. We propose to delay the issue of allowing the usage of Torch tensors with NumPy functions to a separate RFC. 

## Detailed Description
We propose the following signature change to `__torch_function__`, to make it match NumPy: [[4]]

```python
class SubTensor(torch.Tensor):
    def __torch_tensor__(self, func, types, args, kwargs):
        # Implementation here
```

The reason for adding `types` to the signature is necessitated by the need for `super()`. If we set a requirement for `super().__array_function__` to work properly, we would need to provide an easy way for users to signal to `__array_function__` that they are calling to the next-specific implementation. The way we propose to handle this is the same as it is handled in NumPy, albeit not in the context of overriding methods, but rather, in the context of subclasses of `numpy.ndarray` or other classes that implement `__array_function__`.

The process followed during a function/method call would be equivalent to:

1. The dispatcher is called to extract the `Tensor`-likes.
2. All `Tensor`-likes are checked for `__torch_function__`. If none exist, the internal implementation is called, and the final result is returned.
3. A collection of types that implement `__torch_function__` is created, with no guaranteed order other than that subclasses come before superclasses.
4. For one argument of each type, `__torch_function__` is called. The first such function or method to return something other than `NotImplemented` will be the final result. All exceptions will be propagated upward.
5. If all `__torch_function__` implementations return `NotImplemented`, a `TypeError` is raised with an appropriate error message.

In practice, for most PyTorch functions, the list of tensor-likes is already available and the dispatcher doesn't need to be called. Additionally, while equivalent to the code above, if the `Tensor`-likes are all `Tensor`s or classes for which `__torch_function__ is torch.Tensor.__torch_function__`, the internal implementation is called immediately. This is done as a performance optimisation to avoid overhead for concrete `Tensor` objects.

To access super, one would do the following:
```python
class SubTensor(torch.Tensor):
    def __torch_tensor__(self, func, types, args, kwargs):
        # Pre-processing here
        val = super().__torch_function__(
            func,
            tuple(
                t for t in types if not issubclass(t, SubTensor)
            ),
            args,
            kwargs
        )
        # Post processing here
```

This way `__torch_function__` knows the list of types to dispatch to, and it will _not_ dispatch to `SubTensor` anymore in this example.

We will also recommend that all `Tensor` subclasses make their own methods go through `__torch_function__` via a decorator `@torch_function_dispatch`. This decorator was added and then removed for performance reasons, however it will be added back to allow external libraries to interface with the protocol. It will take a single argument: a dispatcher, i.e. a callable that returns an iterable of all the "duck-Tensors", or possible candidates for classes that may implement `__torch_function__`.

If a library forgets to add the aforementioned decorator, then the method will no longer dispatch at all to any form of `__torch_function__`. In other words, it will lose support for the protocol.

However, this will come with a disclaimer: They _must_ accept that their methods are subject to the same processing as any other `torch.Tensor` methods, namely, that all the processing _will necessarily go through `__torch_function__`, even if through superclasses first_. Specifically, processing may pass through a superclass's `__torch_function__` implementation before coming back to a subclass's internal implementation. A workaround would be to define a `__torch_function__` on the same class that directly calls the internal implementation, or follows the same pattern as the example `__torch_function__` defined on `torch.Tensor`. This will allow subclasses to behave exactly like `torch.Tensor`, while still allowing access to method overriding via the `__torch_function__` protocol.

To make this concrete, consider the following stub implementation, for which we'll consider a few concrete cases:

```python
class SubTensor(torch.Tensor):
    def __torch_function__(...):
        # Pre-processing
        ret = super().__torch_function__(func, tuple(t for t in types if not issubclass(t, SubTensor)), args, kwargs)
        # Post processing
        return ret

class SubSubTensor(SubTensor):
    @torch_function_dispatch(...)
    def __add__(self, other):
        # Pre-processing
        ret = super().__add__()
        # Post-processing
        return ret
```

In this instance, processing would follow the `__torch_function__` protocol. This means that control would end up in `SubTensor.__torch_function__`, go to `Tensor.__torch_function__` from there and and then come to `SubSubTensor.__add__`, from where it would go to `Tensor.__add__`, and then back up the stack in the reverse order. This means that great care needs to be taken when writing `SubTensor.__torch_function__` to take into account the fact that it has to handle subclass methods.

We do not propose automatic marking of functions with this decorator due to the potential backwards-compatibility break it could cause, as well as the parameters that are needed in order to allow this to happen (namely the dispatcher, which isn't in our control).

### Making `torch.Tensor._make_subclass` public API
`torch.Tensor._make_subclass` will be renamed to `torch.Tensor.make_subclass` and it will become public API. This will allow `torch.Tensor.make_subclass` correspond to `numpy.ndarray.view`, with the difference that the latter can also handle viewing as a different `dtype` and not just as a different subclass.

The reason for not choosing the name `view` is that it exists on `torch.Tensor` in an unrelated context. The semantics of this function will be the same as creating a shallow copy of the object and then changing its `__class__`, which means that the underlying data pointer will remain the same.

### Generic implementation of `__torch_function__`
`torch.Tensor` will gain a generic `__torch_function__` of the following form:

```python
class Tensor:
    def __torch_tensor__(self, func, types, args, kwargs):
        if not all(issubclass(type(self), t) for t in types):
            return NotImplemented
        
        # Defer to internal implementation
        ret = func._implementation(*args, **kwargs)
        if type(self) is not Tensor and isinstance(ret, Tensor):
            ret = Tensor.make_subclass(ret, type(self))
        return ret
```

This method matches `torch` dispatch rules, so for the most part it's possible to pretend it doesn't exist. This also has the side-effect of passing subclasses through methods, and operators (since all operators are methods).

This corresponds exactly to the implementation `numpy.ndarray` gains in [[4]], except for the fact that subclasses are passed through via another internal mechanism (namely the `__array_finalize__` protocol) there, as well as the fact that we are checking subclassing against `type(self)` instead of `Tensor`. This has the side-effect of ensuring unrelated class trees are not merged, which is an inconsistency in NumPy's own design. Specifically, consider the example of two direct subclasses of `torch.Tensor`. Both will return `NotImplemented`, and therefore, the check will fail and `TypeError` will be raised.

Since subclasses are checked before superclasses in `__torch_function__`, it is guaranteed that the subclass implementation will be called first. In this instance, since `type(self)` is a subclass of all types, the code will continue. Since `type(self)` is not `torch.Tensor`, a view into the original data is created and returned.

This also works for all operators: `__add__`, `__getitem__` and so on since in Python these operators are just dunder methods of the corresponding class.


[1]: https://github.com/pytorch/pytorch/issues/22402 "GitHub Issue 22402 on pytorch/pytorch"
[2]: https://github.com/pytorch/pytorch/issues/28361#issuecomment-544520934 "Comment on GitHub Issue 28361 on pytorch/pytorch"
[3]: https://github.com/pytorch/pytorch/issues/28361#issuecomment-557285807 "Comment on GitHub Issue 28361 on pytorch/pytorch"
[4]: https://numpy.org/neps/nep-0018-array-function-protocol.html "NEP 18 — A dispatch mechanism for NumPy’s high level array functions"