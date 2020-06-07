<!--watch-latex-md

This document is processed by watch_latex_md.py program, see

  https://github.com/Quansight/pearu-sandbox/latex_in_markdown/

You can edit this document as you wish. You can also edit the LaTeX
data in img elements, but only the content of `latex-data`:

  1. To automatically update the LaTeX rendering in img element, edit
     the file while watch_latex_md.py is running.

  2. Never change the beginning (`<img latex-data="...`) and the end
     (`...alt="latex">`) parts of the LaTeX img elements as these are
     used by the watch_latex_md.py script.

  3. Changes to other parts of the LaTeX img elements will be
     overwritten.

Enjoy LaTeXing!
-->

# Roadmap for PyTorch Sparse Tensors


|            |                 |
| ---------- | --------------- |
| Authors    | Pearu Peterson  |
| Status     | Draft           |
| Type       | Process         |
| Created    | 2020-04-23      |
| Resolution | TBD             |

## Abstract

The aim of this document is to define the future of PyTorch sparse
tensors. We propose that
> :large_blue_circle:<!--:proposal:--> a sparse tensor is a drop-in
> replacement of dense tensors for all tensor operations.

## Motivation and Scope

PyTorch implements the *Tensor* object as the basic structure of
storing data in a multi-dimensional array structure that can be
processed on various computational devices such as GPUs as well as
multicore CPUs efficiently by taking advantage of parallelism that
these devices support. PyTorch has advanged in developing
computational tools for various ML and AI algorithms that can be
applied to so-called dense or strided tensors. However, the
developement of tools applicable to sparse tensors has not passed the
experimental stage after two or more years of development, dispite the
fact that a sparse tensor would be a natural data storage format for
many ML&AI applications where the data characterizes some pairwise
relations between study subjects, or when analyzing only a small set
of features that dominate in a big system, for instance. Choosing a
sparse tensor format to store the data may lead to more performant
programs due to the efficient uses of memory as well as computational
resources.

The starting point of this proposal and the far-reaching goal is that
the PyTorch tensor represents a multi-dimensional array whereas the
used data storage format is an implementation detail that users can
select depending on their data to (indirectly) control the choise of
computational tools that would be most efficient for users'
applications. Switching a storage format of data to another data
format (dense to sparse, sparse to dense, sparse to sparse, etc)
should not require changes to the data processing algorithms written
by the end-users or other high-level software tools. This approach of
supporting data storage format agnostic algorithms allows faster
adoption of newly developed algorithms and mathematical models as
software programs.

PyTorch already implements a dispatcher system for executing programs
on a particular device depending on the data storage location. The
same dispatcher system can be used for selecting computational tools
depending on the data storage format. So, the implementation of this
proposal will be about
1. providing sparse tensor implementations and the corresponding
low-level computational tools
2. reusing PyTorch dispatcher system to make sparse tensors as drop-in
replacements of dense tensors when used as inputs to high-level
computational tools.

A prioritization of features sets is required in order to make the
stable sparse tensors support accessible for users' programs as soon
as possible.

### About the used language and notation

In PyTorch as well as in other software/hardware project
(e.g. TensorFlow, Tensor cores in NVIDIA GPU devices, etc), a
multi-dimensional array is called a tensor. However, in mathematics, a
*tensor* is a more general algebra concept than an array (see
https://en.wikipedia.org/wiki/Tensor): while a tensor can be
represented as a multi-dimensional array, not every multi-dimensional
array is a representation of a tensor.  In the following, we use the
term *tensor* only within the context of PyTorch project, and the term
*array* within mathematical context that is applicable in a more
general settings.

To represent various mathematical concepts and relations in a concise
way, we often use Python language syntax in a free way as a
replacement of mathematical typesetting. So, the reader of this
document should be familiar with the Python syntax and semantics.

We use emoji :large_blue_circle:<!--:proposal:--> to mark a proposal
statement and :large_blue_diamond:<!--:impl:--> to mark an implementation option.

## Mathematical background of arrays

One of the most common data representation in scientific and
engineering computations is an array of elements (or items or entries)
that share the same data type and size and that can be generalized to
any dimensions. The array elements are indexed using tuples of
integers that represent the coordinates in a multidimensional discrete
space of data values.  Mathematically, arrays can be interpreted as
vectors, matrices, or tensors that form algebras with various algebra
operations as well as functions on algebra elements.

When representing data as an array in computer memory, a array storage
format must be selected. The simplest and most general array storage
format is the strided array format where the data values are stored
continuously in memory so that the storage location any array element
can be easily (<img data-latex="$O(1)$" src=".images/ef0cdc8ea95d866c1fb30b9f9724a739.svg"  valign="-4.289px" width="37.737px" height="17.186px" style="display:inline;" alt="latex">) determined from its index tuple and therefore
efficient concurrent algorithms can be designed that process data in
the most efficient way.  However, when data poses certain
characteristics that would lead to arrays where majority of array
elements have the same value, choosing a strided array format may lead
to inefficient use of memory and processor resources. For such cases
various sparse array formats have been proposed that provide
considerable storage savings as well as more efficient processing
possibilities but a the cost of increased complexity in the sparse
storage format.

To be specific, in this document we consider three array storage
formats: strided dense, COO sparse, and CSR sparse array storage
formats. However, implementing other storage formats are encouraged if
these have potential to increase the performance of existing array
operations.

To relate the different array storage formats, we define for each
format a transformation to a canonical array storage format for which
we use a contiguous array storage format. We denote this
transformation as `contiguous(A)` where `A` denotes an array using any
storage format.

We can now formulate the aim of this proposal as follows: let `op` be
any array operation on arrays `A0`, `A1`, `...`, that returns
arrays `B0`, `B1`, `...` as results:
```python
B0, B1, ... = op(A0, A1, ...)
```

:large_blue_circle:<!--:proposal:--> We aim at ensuring that the following equation holds:
```python
map(contiguous, op(*map(contiguous, (A0, A1, ...)))) == map(contiguous, (B0, B1, ...))
```

Note 1: when operation `op` returns arrays with the same storage
format as the input arrays use, the equation simplifies to
```python
op(*map(contiguous, (A0, A1, ...))) == map(contiguous, (B0, B1, ...))
```
that is, the operations `op` and `contiguous` are commutative in this
particular case.

Note 2: PyTorch uses COO array storage format where the values of
elements can be (dense) arrays. As a result, its shape consists of
sparse dimensions followed by dense dimensions. If one interprets such
hybrid array as an array of scalar elements then operations that
permute array dimensions may need to return a fully dense array of
scalar elements.  As an example, transposing a 2-D array with one
sparse dimension and one dense dimension must result in a 2-D dense
array because of the (implementation specific) constraint that sparse
dimensions must preceed dense dimensions in the PyTorch COO sparse
storage format.  For arrays with a large number of elements such
densifying operations may lead to inefficient use of memory resources
that users might want to prevent, for instance, by reorganizing their
algorithms. Different strategies are possible to acknowledge users
about such events: :large_blue_diamond:<!--:impl:--> report these as
warnings or raise exceptions when a certain threshold of memory usage
has been exceeded or are about to exceed. Introducing such threshold
would be advantageous for smaller arrays when such densifying
operations would not lead to memory usage issues, and therefore, would
be allowed.

### Strided array storage format

Let `A` be a `N`-dimensional array with dimensions `A.shape = (d0, d1,
...)` and `A[I]` denotes the value of the array element with index
`I=(i0, i1, ...)`.  The strided array storage format introduces the
concept of array strides `A.strides=(s0, s1, ...)` that is a `N`-tuple
of integers. For a contiguous row-major storage of array elements the
strides can be computed as follows:
```python
A.strides[0]=1
for i in range(1, N):
    A.strides[i] = A.strides[i-1]*A.shape[i-1]
```

To constuct an array with strided array storage format, we use
the following notation:
```python
A = Strided(data, strides, shape)
```
where `data` represents a contiguous array of elements stored on some
memory storage device and encaptures in itself also the data type
information of array elements.

The memory location of the strided array element `A[I]` is given as:
```
p[I] = b + sum(A.strides[i] * I[i] for i in range(N))
```
where `b` is a reference location (e.g. the starting point of the
allocated memory).

<!--
In terms of Mathematics of Arrays (MoA,
https://www.researchgate.net/publication/308893116_A_Mathematics_of_Arrays),
the memory location of an array element can be expressed in terms of a
integer-valued *&#120574; function* such that
```python
A[I] == ravel(A)[gamma(I, A.shape, ...)]
```
where `A.shape, ...` denotes any array storage format specific parameters
that are required for determining the storage location of the array element
in memory. So, we write
```
p[I] = gamma(I, A.shape, A.strides)
```
for strided arrays.

Note that one could define `ravel(A)` for any array storage format as
`ravel(A) = ravel(contiguous(A))` but that would be impractical for
defining the `gamma` function for arrays using sparse storage
formats. However, `ravel(A)` can be left undefined for such cases and
define `gamma` according to the actual implementation of the
particular storage format: the value of array element `A[I]` has a
storage address equal to
```
p[I] = gamma(I, A)
```
.

Note: Creating a regular slice of an array is a matter of computing
new strides for the slice while using the same reference point `b` as
the base array. For creating non-regular slices a copy of base array
memory may be required.

-->

### COO sparse array storage format (PyTorch)

#### Construction

COO sparse array storage format can be described via two arrays:
`indices` and `values`. The `indices` array is a 2-D array of integers
with dimensions `(Ns, NNZ)`, and the `values` array is a 2-D array of
values with dimensions `(NNZ, Nd)` such that `Ns + Nd = N`. Here the
subscripts `s` and `d` denote the sparse and the dense parts of the
COO sparse array storage (usually `d == 1` but PyTorch generalizes COO
sparse arrays to a sparse-dense hybrid of a COO sparse arrays).

To constuct an array with COO sparse array storage format, we use
the following notation:
```python
A = COO(indices, values, shape)
```
where `indices` and `values` are strided arrays satisfying the
constraints defined above.

PyTorch supports uncoalesced sparse arrays that means there may exists
columns of `indices` that are equal, that is, the equation
```python
A.indices[:, n] == Is
```
may have multiple solutions for `n` that correspond to the same array
element `A[Is + Id]`. So, in general, the value of an array element is
defined as
```python
A[Is + Id] = sum(A.values[n] for n in range(NNZ) if A.indices[:, n] == Is)
```

On the other hand, when no `n` exists such that `A.indices[:, n] ==
Is`, then the corresponding array element is classified as unspecified
element. The unspecified elements may have many
interpretations. Often, the unspecified elements are interpreted as
elements with zero values, but other interpretations exists as
well. For instance, for sparse softmax operation the unspecified
elements are interpreted as negative infinities. In the cases of using
sparse arrays to represent graphs, the unspecified elements are
interpreted as undefined.

<!--
The memory location of a COO sparse array element `A[Is + Id]` is
determined as follows: find `n` such that
```python
A.indices[:, n] == Is
```
holds. Then
```
p[Is + Id] = gamma(Id, A.values[n])
```


The equation `A.indices[:, n] == Is` may have multiple solutions.
This is the case for uncoalesced COO sparse arrays in PyTorch and it
means that the value of the array element `A[Is + Id]` is
`sum(A.values[n] for all n such that A.indices[:, n] == Is
holds)`. Using uncoalesced COO sparse arrays is computationally
efficient for additive operations such as as adding arrays
element-wise because it has an efficient implementation that involves
concatenating the indices and values of the two sparse array. On the
other hand, non-additive operations such as element-wise
multiplication of two sparse tensor, and many other operations,
require coalescing the operands before preforming the operation. In
PyTorch, coalesce operation also sorts `indices` using lexicographic
ordering (this allows fast conversations to CSR sparse array format).

When the equation `A.indices[:, n] == Is` does not have a solution then
the corresponding array element is classified as unspecified
element. The unspecified elements can have many interpretations. In
most cases these are considered as elements with zero values but other
interpetations exists as well. For instance, for sparse softmax
operation the unspecified elements are interpreted as negative
infinities. In the cases of using sparse arrays to represent graphs,
the interpretation of unspecified elements is none.
-->

#### Unspecified elements - fill-value

Many mathematical operations map zero values to non-zero values, for
instance, <img data-latex="$\cos 0 = 1$" src=".images/49363cf83f64b9ed17d981377a98f496.svg"  width="65.992px" height="11.097px" style="display:inline;" alt="latex"> etc. With the aim of defining such
nonhomogeneous operations on sparse arrays in a memory efficient way,
we propose that > :large_blue_circle:<!--:proposal:--> Element-wise
operations on coalesced sparse arrays result > sparse arrays with the
same set of indices as the input.

For nonhomogeneous operations on sparse arrays, we'll need to
introduce fill-value property that represents the value of unspecified
elements. When the fill-value is defined, it has to have the same
properties that array elements have.

##### Example 1

PyTorch COO sparse storage format supports array elements
with dense array values. With this interpretation, the fill-value is
formally a dense array.

##### Example 2

Let us consider a 2-D PyTorch COO sparse storage format
with one sparse and one dense dimensions:

<img data-latex="
$$
A = \begin{bmatrix}
11 & ? & 13 & ? & 15\\
21 & ? & 23 & ? & 25
\end{bmatrix}
$$
" src=".images/ea550b2ce1400a64d1a3c677bc359a13.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">


where `?` represents unspecified entries, the sparse dimension is
horizontal and dense dimension is vertical.

Let's apply softmax operation
(https://en.wikipedia.org/wiki/Softmax_function) along `A` sparse
dimension while interpreting the unspecified elements as zero valued
elements. We denote the softmax normalization factors along sparse
dimension as follows:

<img data-latex="$$
\begin{align*}
d_1 &= \exp(11) + \exp(0) + \exp(13) + \exp(0) + \exp(15)\\
d_2 &= \exp(21) + \exp(0) + \exp(23) + \exp(0) + \exp(25)
\end{align*}
$$" src="to-be-generated" alt="latex">

so that

<img data-latex="
$$
\mathbf{softmax}(A) = 
\begin{bmatrix}
  \exp(11)/d_1 & 1/d_1  & \exp(13)/d_1 & 1/d_1 & \exp(15)/d_1\\
  \exp(21)/d_2 & 1/d_2 &  \exp(23)/d_2 & 1/d_2 & \exp(25)/d_2
\end{bmatrix}
$$
" src=".images/046f962195414fdd52dc1a351eebdd3c.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">


To represent this result as a sparse matrix with the same sparsity as
the input, the fill-value must be a 1-D array:

<img data-latex="
$$
\begin{bmatrix}
1/d_1\\
1/d_2
\end{bmatrix}
$$
" src=".images/f367b4731af9e0c15e4859dc22d12367.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">



##### Example 3

In graph theory, sparse arrays can be used to represent adjacency
relations between nodes while the array elements are certain
transmission coefficents between the nodes. The unspecified elements
indicate the lack of connections between the corresponding nodes.
When applying element-wise operations on such sparse arrays, the nature
of an unspecified element of being a lack of connection should not
change. Hence, the fill-value of such sparse arrays must be undefined.

In conclusion,

> :large_blue_circle:<!--:proposal:--> The fill-value of a PyTorch COO
> sparse storage format array, when defined, is either a scalar or a
> dense array with the shape of dense dimensions.

This holds independent of all possible interpretations of COO sparse
array elements.

##### :large_blue_circle:<!--:proposal:-->Fill-value API

The construction method of PyTorch COO sparse tensor is extended with
an optional fill-value argument:
```python
A = sparse_coo_tensor(indices, values, size=None, fill_value=None, dtype=None, device=None, requires_grad=False)
```

where `fill_value` can be `None` (default fill-value), or a single
scalar value, or any array-like object that can be converted to a
`Tensor` object that satisfies the requirements of a fill-value (see below).

There exists three kinds of fill-values:
1. Scalar fill-value is a `Tensor` with `shape == ()`. For example, `fill_value = 0`.
2. Undefined fill-value is a `Tensor` with `shape == (0, )`. For example, `fill_value = []`
3. Array fill-value is a `Tensor` with `shape == values.shape[1:]`.

The fill-value can be acquired via `_fill_value()` function.

The default fill-value is undefined.

For convenience, the specified fill-value data type and storage
location can be used to determine the unspecified data type and
storage location of the sparse tensor only if `values` is an empty
array-like object except `Tensor`.

The data type and storage location of a defined array fill-value must
match with the data type and storage location of `values`.
If this condition is not satisfied, a `TypeError` exception is raised.

Note 1: The proposed API is advantageous for performing element-wise
operations on sparse arrays:
```python
op(COO(indices, values, fill_value=<fill-value>)) == COO(indices, op(values), fill_value=op(<fill-value>))
```


##### :large_blue_diamond:<!--:impl:-->Fill-value implementation

Implementation-wise, it would be advantageous to store the
`A.fill_values` as a part of `A.values` (say, as the last row) because
all element-wise operations would be performed on `A.values` only.  As
a result, when a sparse array `A` has a fill-value property set then
`A.values.shape == (NNZ+1, Nd)`, otherwise `A.values.shape == (NNZ,
Nd)`.

So, a pseudo-implementation of `fill_value` property is
```python
@property
def fill_value(self):
    nnz = self.nnz
    if self._values.shape[0] == nnz + 1:
        return self._values[-1]
```

The advantage of this proposal is the simplicity for applying
element-wise operations:
```python
op(A) = COO(A.indices, op(A.values))
```
as well as one does not need to manage the storage location and

As a side-effect, the construction of `COO` must not allow using
`fill_value` when it is already specified in `values`:
```python
def __init__(self, indices, values, shape, fill_value=None):
    ...
    nnz = self.nnz
    if self.values.shape[0] == nnz + 1 and fill_value is not None:
        raise ValueError('Fill-value is already defined in values')
```


##### Implementation proposal 1 - alternative

Introduce `fill_value` property to PyTorch COO sparse storage format.

For element-wise operations we have:
```
op(A) = COO(A.indices, op(A.values), op(A.fill_value) if A.fill_value is not None else None)
```

The existing implementations need to be updated to process the
`fill_value` property for algorithm correctness.

#### Fill-value alternative is additive-value

In parallel to `A = COO(indices, values, shape, fill_value)` representation,
there exists an alternative `A' = COO'(indices, values, shape,
additive_value)` where the additive-value is the same as fill-value
for unspecified elements but for specified elements the array element
is a sum of the corresponding value and the additive-value. Compare:
```
A[I] = values[indices.index(I)] if I in indices else fill_value
A'[I] = additive_value + (values[indices.index(I)] if I in indices else 0)
```
That is, `A' = additive_value + COO(indices, values, shape, fill_value=0)`.

While `A'` is equivalent to `A` for arithmetic operations, it is not
suitable representation for sparse arrays applications in graph theory
and we shall skip discussing its possible advantages.

### CSR sparse array storage format

TBD: explain the relation between array item index and the
corresponding memory location of the item value.

## Interpretation of unspecified sparse array entries

TBD: discuss fill value vs offset vs mapping model

## Operations with PyTorch tensors

TBD: what is implemented and what is not? Comparison with SciPy,
PyData, etc sparse array projects.

### Construction

### Indexing operations

### Arithmetic operations

Let `A` and `B` be two sparse tensors and apply a element-wise
binary operation `op` (addition, multiplication, substraction, etc):
```
C = A op B
```
Clearly, `A.shape == B.shape` must hold.

### Tensor algebra operations

### Functions on tensors

#### Utility functions

* `torch.numel(input) → int`
  Returns the total number of elements in the input tensor.

  The `numel` function is often used as a predicate for non-empty
  tensor test.  The number of elements in the input tensor is
  `product(input.shape)` and its function is implementation detail
  agnostic.


## Implementation plan

TBD: prioritize

TBD: existing matmut proposal for CSR format

TBD: high-priority fill value/offset support

## Appendix: User stories

- A sparse matrix fits to GPU memory but dense matrix doesn't. https://github.com/cupy/cupy/issues/2360

- Algorithms on sparse tensors outperform the ones on dense tensors when the density of data is sufficiently low. https://github.com/pytorch/pytorch/pull/36305#issuecomment-627624391

- I'd love to be able to pass a sparse tensor in to torch.distributions... https://github.com/pytorch/rfcs/pull/4#issuecomment-619459572

- My hunch is that most users want sparse tensors to be a drop-in replacement as much as possible... https://github.com/pytorch/rfcs/pull/4#issuecomment-619459572

- Where do I need sparse tensor?... https://github.com/pytorch/pytorch/issues/10043

- The state of sparse tensors  https://github.com/pytorch/pytorch/issues/9674
