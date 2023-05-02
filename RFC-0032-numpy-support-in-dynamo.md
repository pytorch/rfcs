# A PyTorch - NumPy compatibility layer

**Authors:**
* @ev-br
* @lezcano
* @rgommers

## Summary
This RFC describes a proposal for a translation layer from NumPy into PyTorch.
In simple terms, this accounts for implementing most of NumPy's API (`ndarray`,
the `numpy`, `numpy.linalg`, `numpy.fft`  modules, etc) using `torch.Tensor`
and PyTorch ops as backend.

The main goal is: **make TorchDynamo understand NumPy calls**.
This should enable an end user to combine code that uses the PyTorch API with
code that uses the NumPy API, in a way that allows TorchDynamo to understand
those function calls and build up an execution graph. To enable this, it is key
that there is a translation layer from NumPy to PyTorch function calls, which
TorchDynamo can use in order to build up its execution graph from PyTorch
functions/primitives only. For niche functions in NumPy that don’t have a
PyTorch equivalent, it’s okay to graph break and still call NumPy to execute
the function call.

The work is currently being done at [numpy_pytorch_interop](https://github.com/Quansight-Labs/numpy_pytorch_interop/).


## Motivation

### Introductory examples

Consider the following snippet:
```python
import numpy as np

x = np.random.randn(3, 4)
y = np.random.randn(4, 3)
z = np.dot(x, y)
w = z.sum()
```

When we trace this program with the compat layer, the semantics of the
program would stay the same, but the implementation would be equivalent to

```python
import torch
x = torch.randn(3, 4, dtype=torch.float64)
y = torch.randn(4, 3, dtype=torch.float64)
z = torch.matmul(x, y)
w = z.sum()
```

Here, we can already spot a couple differences between NumPy and PyTorch.
The most obvious one is that the default dtype in NumPy is `float64` rather than
`float32`. The less obvious is very sneakily hiding in the last line.

```python
>>> type(w)
<class 'numpy.float64'>
```

Reductions and similar operations in NumPy return the infamous NumPy scalars.
We'll discuss these and other NumPy quirks and how we dealt with them in the
[design decision section](#design-decisions).


Let's now have a look at a toy example of how this layer would be used.
```python
import torch
import numpy as np
t1 = torch.tensor([1, 3, 5])
t2 = torch.exp(t)
# Now say the user has some code lying around which uses NumPy:
def fn(x, y):
    return np.multiply(x, y).sum()

result = fn(t1, t2)
t_results = torch.empty(5, dtype=torch.float64)
t_results[0] = result  # store the result in a torch.Tensor
```

Note that this code mixing NumPy and PyTorch already works in eager mode with
CPU tensors, as `torch.Tensor` implements the `__array__` method. Now, the
compatibility layer allows us to trace through it. In order to do that, there
would be no necessary changes, other than simply ask `torch.compile` to trace
through it:

```python
@compile
def fn(x, y):
    return np.multiply(x, y).sum()
```

Then, TorchDynamo will cast `x` and `y` to our internal implementation of `ndarray`,
and will dispatch `np.multiply` and `sum` to our implementations in terms of `torch`
functions, effectively turning this function into a pure PyTorch function.

### Design decisions

The main ideas driving the design of this compatibility layer are the following:

1. The goal is to transform valid NumPy and mixed PyTorch-NumPy programs into
   their equivalent PyTorch-only execution.
2. The behavior of the layer should be as close to that of NumPy as possible
3. The layer follows the most recent NumPy release

The following design decisions follow from these:

**A superset of NumPy**. NumPy has a number of well-known edge-cases (as does
PyTorch, like spotty support for `float16` on CPU and `complex32` in general).
The decision to translate only valid NumPy programs, often allows us to
implement a superset of the functionality of NumPy with more predictable and
consistent behavior than NumPy itself has.

**Exceptions may be different**. We avoid entirely modelling the exception
system in NumPy. As seen in the implementation of PrimTorch, modelling the
error cases of a given system is terribly difficult. We avoid this altogether
and we choose not to offer any guarantee here.

**Default dtypes**. One of the most common issues that bites people when migrating their
codebases from NumPy to JAX is the default dtype changing from `float64` to
`float32`. So much so that this is noted as one of
[JAX's sharp edges](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).
Following the spirit of making everything match NumPy by default, we choose the
NumPy default dtype whenever the `dtype` was not made explicit in a factory function.
We also provide a function `set_default_dtype` that allows to change this behavior
dynamically.

**NumPy scalars**. NumPy's type system is tricky. At first sight, it looks
like PyTorch's, but with few more dtypes like `np.uint16` or `np.longdouble`.
Upon closer inspection, one finds that it also has
[NumPy scalar](https://numpy.org/doc/stable/reference/arrays.scalars.html) objects.
NumPy scalars are similar to Python scalars but with a fixed precision and
array-like methods attached. NumPy scalars are NumPy's preferred return class
for reductions and other operations that return just one element.
NumPy scalars do not play particularly well with
computations on devices like GPUs, as they live on CPU. Implementing NumPy
scalars would mean that we need to synchronize after every `sum()` call, which
would be terrible performance-wise. In this implementation, we choose to represent
NumPy scalars as 0-D arrays. This may cause small divergences in some cases. For example,
consider the following NumPy behavior:

```python
>>> np.int32(2) * [1, 2, 3]       # scalar decays to a python int
[1, 2, 3, 1, 2, 3]

>>> np.asarray(2) * [1, 2, 3]     # zero-dim array is an array-like
array([2, 4, 6])
```

We don't expect these to pose a big issue in practice. Note that in the
proposed implementation `np.int32(2)` would return the same as `np.asarray(2)`.
In general, we try to avoid unnecessary graph breaks whenever we can. For
example, we may choose to return a tensor of shape `(2, *)` rather than a list
of pairs, to avoid a graph break.

**Type promotion**. Another not-so-well-known fact of NumPy's dtype system  and casting rules
is that it is data-dependent. Python scalars can be used in pretty much any NumPy
operation, being able to call any operation that accepts a 0-D array with a
Python scalar. If you provide an operation with a Python scalar, these will be
cast to the smallest dtype they can be represented in, and only then will they
participate in type promotion. This allows for for some rather interesting behaviour
```python
>>> np.asarray([1], dtype=np.int8) + 127
array([128], dtype=int8)
>>> np.asarray([1], dtype=np.int8) + 128
array([129], dtype=int16)
```
This data-dependent type promotion will be removed in NumPy 2.0 (planned for Dec'23), and will be
replaced with [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html)
(already implemented in NumPy, it needs to be enabled via a private global switch now).
For simplicity and to be forward-looking, we chose to implement the
type promotion behaviour proposed in NEP 50, which is much closer to that of
PyTorch.

Note that the decision of going with NEP 50 complements the previous one of
returning 0-D arrays in place of NumPy scalars as, currently, 0-D arrays do not
participate in type promotion in NumPy (but will do in NumPy 2.0 under NEP 50):
```python
int64_0d_array = np.array(1, dtype=np.int64)
np.result_type(np.int8, int64_0d_array) == np.int8
```

**Versioning**. It should be clear from the previous points that NumPy has a
fair amount of questionable behavior and legacy pain points. It is for this reason that
we decided that rather than fighting these, we would declare that the compat
layer follows the behavior of NumPy's most recent release (even, in some cases,
of NumPy 2.0). Given the stability of NumPy's API and how battle-tested its
main functions are, we do not expect this to become a big maintenance burden.
If anything, it should make our lives easier, as some parts of NumPy will soon
be simplified, saving us the pain of having to implement all the pre-existing
corner-cases.

**Randomness**. PyTorch and NumPy use different random number generation methods.
In particular, NumPy recently moved to a [new API](https://numpy.org/doc/stable/reference/random/index.html)
with a `Generator` object which has sampling methods on it. The current compat
layer does not implement this new API, as the default bit generator in NumPy is
`PCG64`, while on PyTorch we use `MT19937` on CPU and `Philox` on non-CPU devices.
From this, it follows that this API will not give any reproducibility
guarantees when it comes to randomness.

**Accuracy**. For deterministic operations, we would expect to give accuracy
guarantees similar to those in PyTorch 2.0. In particular, we would expect
these decompositions to be as precise as those from NumPy when compared to an
`fp64` baseline minus perhaps a small relative error.


## The `torch_np` module

The bulk of the work went into implementing a system that allows us to
implement NumPy operations in terms of those of PyTorch. The main design goals
here were

1. Implement *most* of NumPy's API
2. Preserve NumPy semantics as much as possible

We say *most* of NumPy's API, because NumPy's API is not only massive, but also
there are parts of it which cannot be implemented in PyTorch. For example,
NumPy has support for arrays of string, datetime, structured and other dtypes.
Negative strides are another example of a feature that is not supported in PyTorch.
We put together a list of things that are out of the scope of this project in the
[following issue](https://github.com/Quansight-Labs/numpy_pytorch_interop/issues/73).

For the bulk of the functions, we started by prioritizing the most common
operations. Then, when bringing tests from the NumPy test suite, we triaged
and prioritized how important it was to fix each failure we found. Doing this
iteratively, we ended up with a small list of differences between the NumPy and
PyTorch APIs, which we prioritized by hand. That list and the prioritization
discussion can be found in [this issue](https://github.com/Quansight-Labs/numpy_pytorch_interop/issues/87).

**Visibility of the module** For simplicity, this RFC assumes that the
`torch_np` module will not be public, as the initial suggestion for it to be
made public was met with mixed opinions. This topic can be revisited in the
future if desired.
We discuss these in the section [unresolved questions](#unresolved-questions).

### Annotation-based preprocessing

NumPy accepts virtually anything that smells like an array as an input.

```python
>>> np.add(1, 3)
4
>>> np.add([1., 2., 3.], 5)
array([6., 7., 8.])
>>> np.concatenate([[1, 2, 3], [4, 5, 6]])
array([1, 2, 3, 4, 5, 6])
```

NumPy calls all these objects `array_like` objects.
To implement NumPy in terms of PyTorch, for any operation we would need to map
inputs into tensors, perform the operations, and then wrap the tensor into
a `torch_np.ndarray` (more on this class later).

To avoid all this code repetition, we implement the functions in two steps.

First, we implement functions with the NumPy signature, but assuming that in
place of NumPy-land elements (`np.array`, array-like functions, `np.dtype`s, etc)
they simply accept `torch.Tensor` and PyTorch-land objects and return
`torch.Tensor`s. For example, we would implement `np.diag` as

```python
def diag(v, k=0):
    return torch.diag(v, k)
```

In this layer, if a NumPy function is composite (calls other NumPy functions
internally), we can simply vendor its implementation, and have it call our
PyTorch-land implementations of these functions. In other words, at this level,
functions are composable, as they are simply regular PyTorch functions.
All these implementations are internal, and are not meant to be seen or used
by the end user.

The second step is then done via type annotations and a decorator. Each type
annotation has an associated function from NumPy-land into PyTorch-land. This
function converts the set of inputs accepted by NumPy for that argument into a
PyTorch-land object (think a `torch.Tensor` or a PyTorch dtype). For example,
for `np.diag` we would write

```python
@normalize
def diag(v: ArrayLike, k=0):
    return torch.diag(v, k)
```

Then, we wrap these Python-land functions with a `normalizer` decorator and
expose them in the `torch_np` module. This decorator is in charge of gathering
all the inputs at runtime and normalizing them (i.e., converting `torch_np`
objects to PyTorch counterparts) according to their annotations.

We currently have four annotations (and small variations of them):
- `ArrayLike`: The input can be a `torch_np.array`, a list of lists, a
  scalar, or anything that NumPy would accept. It returns a `torch.Tensor`.
- `DTypeLike`: Takes a `torch_np` dtype, and any other object that Numpy dtypes
  accept (strings, typecodes...) and returns a PyTorch dtype.
- `AxisLike`: Takes anything that can be accepted as an axis (e.g. a tuple or
  an `ndarray`) and returns a tuple.
- `OutArray`: Asserts that the input is a `torch_np.ndarray`. This is used
  to implement the `out` keyword.

Note that none of the code in this implementation  makes use of NumPy. We are
writing `torch_np.ndarray` above to make more explicit our intent, but there
shouldn't be any ambiguity.

**Implementing `out`**: In PyTorch, the `out` kwarg is a keyword-only argument.
It is for this reason that, in PrimTorch, we were able to implement it as [a
decorator](https://github.com/pytorch/pytorch/blob/ce4df4cc596aa10534ac6d54912f960238264dfd/torch/_prims_common/wrappers.py#L187-L282).
This is not the case in NumPy. In NumPy, `out` can be used both as a positional
and a keyword argument, and is often interleaved with other parameters. This is
the reason why we use the `OutArray` annotation to mark these. We then
implement the `out` semantics in the `@normalizer` wrapper in a generic way.

**Ufuncs and reductions**: Ufuncs (unary and binary) and reductions are two
sets of functions that are particularly regular. For these functions, we
implement support for their arguments in a generic way as a preprocessing or
postprocessing step.

**The `ndarray` class** Once we have all the free functions implemented as
functions from `torch_np.ndarray`s to `torch_np.ndarray`s, implementing the
methods from the `ndarray` class is rather simple. We simply register all the
free functions as methods or dunder methods appropriately. We also forward the
properties of `ndarray to the corresponding properties of `torch.Tensor` and we
are done. This creates a circular dependency which we break with a local
import.

### Testing

The testing of the framework was done via ~~copying~~ vendoring tests from the
NumPy test suite. Then, we would replace the NumPy imports with `torch_np`
imports. The failures on these tests were then triaged, and either fixed or marked
`xfail` depending on our assessment of the priority of implementing a fix.

In the end, to have a last check that this tool was sound, we pulled five
examples of NumPy code from different sources and ran it with this library (eager mode execution).
We were able to run the five examples successfully with close to no code changes.
You can read about these in the [README](https://github.com/Quansight-Labs/numpy_pytorch_interop).

### Limitations

A number of known limitations are tracked in the second part of the
[OP of this issue](https://github.com/Quansight-Labs/numpy_pytorch_interop/issues/73).
When landing this RFC, we will create a comprehensive document with the differences
between NumPy and `torch_np`.

### Beyond plain NumPy

**GPU**. The implementation allows for running NumPy code on GPU simply by
adding a `torch.set_default_device("cuda")` at the top of the relevant
file/code. We tested this on the code examples that we drew from the internet
and they run just fine on GPU.

**Gradients**. We have not tested gradient tracking either as we are still to
find some good examples on which to test it, but it should be a simple
corollary of all this effort. If the original tensors fed into a function
have `requires_grad=True`, the tensors will track the gradients of the internal
implementation and then the user can differentiate through their NumPy code.

### Bindings to TorchDynamo

The bindings for NumPy at the TorchDynamo level are currently being developed in
[pytorch#95849](https://github.com/pytorch/pytorch/pull/95849).


## Unresolved questions

A question was left open in the initial discussion. Should the module
`torch_np` be publicly exposed as `torch.numpy` or not?

A few arguments in favor of making it public:
* People could use it in their NumPy programs just by changing the import to
  `import torch.numpy as np`. This could be a selling point similar to JAX's
  `jax.numpy`, which could incentivize adoption.
* People would not need to use the whole `torch.compile` stack to start using
  PyTorch as a backend for their NumPy code in their codebase.
  * See [this experiment in scikit-learn](https://github.com/scikit-learn/scikit-learn/pull/25956)
    where they got a 7x speed-up on CPU on a layer just by using `torch.linalg`.
* Since the layer is rather thin and in pure Python, if there are bugs,
  external contributors could easily help fixing them or extend the supported
  functionality.

A few arguments against:
* The compat introduces a number of type conversions that may produce somewhat
  slow code when used in eager mode.
  * [Note] Keeping this in mind, we tried to use as few operators as possible,
    in the implementation, to make it reasonably fast in eager mode.
* Exposing `torch.numpy` would create a less performant secondary entry point
  to many of the functions in PyTorch. This could be a trap for new users.
