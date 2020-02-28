# Improving subclassing Tensor by propagating subclass instances
This RFC describes changes necessary to allow `__torch_function__` to be used
by methods of `torch.Tensor` in an attempt to make subclassing more accessible
to the users of the class. This entails making an API for subclass views
public, and a change in the signature of `__torch_function__`.

## Motivation and Scope
Quoting [[1]], [[2]] and [[3]], the goals of this proposal are:

1. Support subclassing `torch.Tensor` in Python
2. Preserve `torch.Tensor` subclasses when calling `torch` functions on them
3. Use the PyTorch API with `torch.Tensor`-like objects that are _not_ `torch.Tensor`
   subclasses
4. Preserve `torch.Tensor` subclasses when calling `torch.Tensor` methods.
5. Propagating subclass instances correctly also with operators, using
   views/slices/indexing/etc.
6. Preserve subclass attributes when using methods or views/slices/indexing.
7. A way to insert code that operates on both functions and methods uniformly
   (so we can write a single function that overrides all operators).
8. The ability to give external libraries a way to also define
   functions/methods that follow the `__torch_function__` protocol.

Goals 1‒6 are explicitly about subclassing, goal 7 is already partially achieved via the `__torch_function__` protocol (which we're proposing to extend to methods), and goal 8 is a by-product required to make overridden `torch.Tensor` subclass methods behave similar to `torch.Tensor` methods.

Achieving interoperability with NumPy and adopting its array protocols is out
of scope for this proposal and we propose to defer it to a later proposal.

We propose to solve this problem with the following changes to PyTorch:

1. Make methods and operators of `torch.Tensor` go through the
   `__torch_function__` machinery.
2. Add a `types` argument to `__torch_function__`, to make it match NumPy's
   `__array_function__`.
3. Make `torch.Tensor._make_subclass` public API by renaming it to `torch.Tensor.make_subclass`.
4. Make `torch.Tensor` gain a generic implementation of `__torch_function__`.

## Usage and Impact
Once this proposal is merged, users of subclasses of `torch.Tensor` will have
a much more streamlined experience. Namely, the following code example will
work as-is, without the need for any further modification:

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

Additionally, it will provide subclass authors the ability to also modify the
results of methods and operators in `__torch_function__`, along with regular
function calls, and to modify the result to their specific use-case, perform
logging, or otherwise change the result or the action of the method. For
example:

```python
import logging

class LoggingTensor(torch.Tensor):
    def __torch_function__(self, func, types, args, kwargs):
        logging.info(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
        return super().__torch_function__(
            func,
            tuple(
                t if not issubclass(t, LoggingTensor) else torch.Tensor for t in types
            ),
            args,
            kwargs
        )
```

Assuming minimum logging level is set to `logging.INFO`, the following
indicates the code run, with the logging output in the comments.

```python
t = LoggingTensor([1])

t.sum()  # sum, (LoggingTensor([1]),), {}
t[0]  # __getitem__, (LoggingTensor([1]), 0,), {}

# This is already possible
torch.sum(t)  # sum, (LoggingTensor([1]),), {}
```

To make the protocol operate only on functions rather than methods, one can
check for `func not in type(self).__dict__.values()`. To check for operators
and/or indexing, one can check `func.__name__.endswith("__")`.

### Performance
There are a few requirements for the performance of this proposal, when
implemented:

1. No deterioration for function/method calls on `torch.Tensor` objects.
2. No deterioration of current `__torch_function__` overhead
3. Sub-µs impact on the performance of subclasses not implementing
  `__torch_function__`.

Requirement 1 seems unachievable due to the structure of the code at this
point, as:

1. In methods defined in C++, `self` is excluded from the argument processing
   that gathers `Tensor`-likes in C++.
2. Similar to point 1, C++ methods that take only `self` as a `Tensor`-like don't
   pass through this processing, and they will be required to.
3. For methods defined in Python, the processing for handling `__torch_function__`
   will need to be added, similar to the original `__torch_function__` PR [[5]].

We think an overhead of sub-100 ns per method call is feasible.

## Backwards Compatibility
### With PyTorch `master` as of writing
PyTorch `master` pointed to commit hash
`957a07ffbd13d8a805f4d718e0282efc5d2bff85` at the time of writing. Any classes
implementing `__torch_function__` based on the usage in this commit hash will
break completely, due to the differing signature of the protocol. However, as a
release hasn't been made with `__torch_function__` in it, this is a minor-
impact issue. This brings the design of `__torch_function__` more in line with
NumPy's `__array_function__`, and one familiar with NumPy's protocol could
transition to PyTorch's take on it without too many surprises, with the caveat
that it could also receive methods rather than functions. The release that
`__torch_function__` will make it into PyTorch is expected to be 1.5.0.

### With NumPy
The implementation of this proposal will have no effect on how things interact with NumPy.

## Detailed Description
### Introduction
Subclasses are an important way to override functionality of classes. Given the
popularity of PyTorch, a number of subclasses have sprung up, both within and
outside PyTorch. It is important that functions operating on `torch.Tensor`, as
well as methods on it, support passing through the appropriate subclasses,
otherwise information about which type was passed into the function is lost.
The same applies equally, if not more so, to operators and indexing.

In addition, there has been interest in adding a "universal hook" that operated
on both functions and methods, perhaps modifying the control flow before
returning the result. Such a hook already exists today in the form of
`__torch_function__`, however, it only operates on functions and not on
methods, and support for subclassed `torch.Tensor` objects in this protocol is
limited.

### Proposal
We propose the following signature change to `__torch_function__`, to make it
match NumPy: [[4]]

```python
class SubTensor(torch.Tensor):
    def __torch_function__(self, func, types, args, kwargs):
        # Implementation here
```

The reason for adding `types` to the signature is necessitated by the need for
`super()`. If we set a requirement for `super().__torch_function__` to work
properly, we would need to provide an easy way for users to signal to
`__torch_function__` that they are calling to the next-specific implementation.

### Process followed during a function/method call

The process followed during a function/method call would be equivalent to:

1. The dispatcher is called to extract the `Tensor`-likes.
2. All `Tensor`-likes are checked for `__torch_function__`. If none exist, the
   internal implementation is called, and the final result is returned.
3. A collection of types that implement `__torch_function__` is created, with
   no guaranteed order other than that subclasses come before superclasses.
4. For one instance of each type in `types`, `__torch_function__` is called.
   The first such function or method to return something other than
   `NotImplemented` will be the final result. All exceptions will be propagated
   upward.
5. If all `__torch_function__` implementations return `NotImplemented`, a
   `TypeError` is raised with an appropriate error message.

In practice, for most PyTorch functions, the list of tensor-likes is already
available and the dispatcher doesn't need to be called. Additionally, while
equivalent to the code above, if the `Tensor`-likes are all `Tensor` or don't have
an `__torch_function__` implementation, the internal implementation is called
immediately. This is done as a performance optimisation to avoid overhead for
concrete `Tensor` objects.

It will be the job of the dispatcher to extract `Tensor`-like objects from the
argument list, however, arguments of type `Optional[Tensor]` will be considered
`Tensor`-like. If one gets a compound or dependent type such as `List[Tensor]`
or `Tuple[Tensor, ...]` or `Tuple[Tensor, int]`, the dispatcher will have the job
of extracting an iterable of objects that *could* be `Tensor`-like.

### Generic implementation of `__torch_function__`
`torch.Tensor` will gain a generic `__torch_function__` of the following form:

```python
class Tensor:
    def __torch_function__(self, func, types, args, kwargs):
        if not all(issubclass(type(self), t) for t in types):
            return NotImplemented
        
        # Defer to internal implementation
        ret = func._implementation(*args, **kwargs)
        if type(self) is not Tensor and isinstance(ret, Tensor):
            ret = Tensor.make_subclass(ret, type(self))
        return ret
```

This method has the effect of passing through subclasses through all
functions/methods as intended.

This corresponds exactly to the implementation `numpy.ndarray` gains in [[4]],
except for the fact that subclasses are passed through via another internal
mechanism (namely the `__array_finalize__` protocol) there, as well as the fact
that we are checking subclassing against `type(self)` instead of `Tensor`. This
has the side-effect of ensuring unrelated class trees are not merged, which is
an inconsistency in NumPy's own design. Specifically, consider the example of
two direct subclasses of `torch.Tensor`. Both will return `NotImplemented`, and
therefore, the check will fail and `TypeError` will be raised.

Since subclasses are checked before superclasses in `__torch_function__`, it is
guaranteed that the subclass implementation will be called first. In this
instance, since `type(self)` is a subclass of all types, the code will
continue. Since `type(self)` is not `torch.Tensor`, a view into the original
data is created and returned.

This also works for all operators: `__add__`, `__getitem__` and so on since in
Python these operators are just dunder methods of the corresponding class.

### The need for `super().__torch_function__`
To access super, one would do the following:
```python
class SubTensor(torch.Tensor):
    def __torch_function__(self, func, types, args, kwargs):
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

This way `__torch_function__` knows the list of types to dispatch to, and it
will _not_ dispatch to `SubTensor` anymore in this example.

To make the need for `super()` to be available concrete, let's consider the
following scenario:

```python
class SubTensor(torch.Tensor):
    def __torch_function__(...):
        # Pre-processing
        ret = super().__torch_function__(
            func,
            tuple(t if not issubclass(t, SubTensor) else torch.Tensor for t in types),
            args,
            kwargs
        )
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

If `super().__torch_function__` wasn't possible here (or if the signature
didn't include `types`), then there would be an infinite recursion in this
instance. This is because `__add__` would call `SubTensor.__torch_function__`,
`super().__torch_function__` would call `SubSubTensor.__add__` (realise, this
is a different object than `Tensor.__add__`), which, not realising that it has
been processed already, would pass control *back* to
`SubTensor.__torch_function__` and so on ad infinitum.

In this instance, with the proposed changes, processing would follow the
`__torch_function__` protocol. This means that control would end up in
`SubTensor.__torch_function__`, go to `Tensor.__torch_function__` from there
and and then come to `SubSubTensor.__add__`, from where it would go to
`Tensor.__add__`, and then back up the stack in the reverse order. This means
that great care needs to be taken when writing `SubTensor.__torch_function__`
to take into account the fact that it has to handle subclass methods.

### Protocol support for external libraries
We will also recommend that all `Tensor` subclasses make their own methods go
through `__torch_function__` via a decorator `@torch_function_dispatch`. This
decorator was added and then removed for performance reasons, however it will
be added back to allow external libraries to interface with the protocol. It
will take a single argument: a dispatcher, i.e. a callable that returns an
iterable of all the "duck-Tensors", or possible candidates for classes that may
implement `__torch_function__`.

If a library forgets to add the aforementioned decorator, then the method will
no longer dispatch at all to any form of `__torch_function__`. In other words,
it will lose support for the protocol. This can lead to confusion, as some
methods of the subclass will pass through `__torch_function__` (the ones
inherited from `torch.Tensor`), and some won't.

Note that subclasses will still be passed through due to the default
implementation of `__torch_function__`, but any `__torch_function__` defined on
the class itself (or any of its subclasses) won't have an effect on its
methods.

This is a design choice that a subclass author will have to make, whether they
prefer their own functions/methods to pass through `__torch_function__` like
PyTorch's implementations, or whether they'd like ultimately to not support the
protocol and accept having a mix of overridable and non-overridable methods.

We do not propose automatic marking of functions with this decorator due to the
potential backwards-compatibility break it could cause, as well as the
parameters that are needed in order to allow this to happen (namely the
dispatcher, which isn't in our control).

### Making `torch.Tensor._make_subclass` public API
`torch.Tensor._make_subclass` will be renamed to `torch.Tensor.make_subclass`
and it will become public API. This method will create an object that has the
same data pointer as the original object, which means that modifications to
this will be reflected in the original object. More or less, it will have the
same effect as modifying an object's `__class__` attribute in Python.

This method is already used in external libraries, and they may need it as a
way to e.g. bypass the processing of `torch.Tensor.__torch_function__`
entirely, while still creating `torch.Tensor` subclasses in their own code.

## Implementation
To implement this proposal requires three main steps:

1. Add a `types` argument to `__torch_function__` and make sure that _only_
   arguments that are instances of a type in `types` are processed.
2. Making sure that all `Tensor` methods except `__new__` and `__init__` go
   through `__torch_function__`.
3. Make `Tensor._make_subclass` and `@torch_function_dispatch` public API. 

## Proposed alternatives
One alternative that has been proposed is to automatically pass through
subclasses a-la NumPy and provide a `__torch_finalize__` method that allows for
any post-processing of the result. While this would achieve most goals, it
would miss out on the one to provide a hook for methods and operators.


[1]: https://github.com/pytorch/pytorch/issues/22402 "GitHub Issue 22402 on pytorch/pytorch"
[2]: https://github.com/pytorch/pytorch/issues/28361#issuecomment-544520934 "Comment on GitHub Issue 28361 on pytorch/pytorch"
[3]: https://github.com/pytorch/pytorch/issues/28361#issuecomment-557285807 "Comment on GitHub Issue 28361 on pytorch/pytorch"
[4]: https://numpy.org/neps/nep-0018-array-function-protocol.html "NEP 18 — A dispatch mechanism for NumPy’s high level array functions"
[5]: https://github.com/pytorch/pytorch/pull/32194 "GitHub Pull request 32194 on pytorch/pytorch"