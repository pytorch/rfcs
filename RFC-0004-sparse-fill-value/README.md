# PyTorch Sparse Tensors: fill-value property

|            |                 |
| ---------- | --------------- |
| Authors    | Pearu Peterson  |
| Status     | Draft           |
| Type       | Process         |
| Created    | 2020-09-08      |
| Resolution | TBD             |

## Abstract

This proposal introduces a fill-value property to PyTorch sparse
tensors that generalizes the current interpretation of unspecified
elements from zero value to any value, including undefined value as a
future extension.

## Motivation and Scope

In general, the unspecified elements of sparse tensors have
domain-specific interpretations:
- In Linear Algebra domain, the unspecified elements are zero valued.
- In Graph domain, the unspecified elements of adjacency matrices
  represent non-edges. In this document, a non-edge corresponds to
  undefined value.
- In neural networks, sparse tensors can be inputs, say, to activation
  functions that are defined in terms of elementary functions.
  Evaluation of such functions on sparse tensors can be element-wise,
  or one-dimension-wise operations. This includes the evaluation of
  the functions on the values corresponding to unspecified
  elements. Even when the unspecified elements are initially being
  zero valued, the values of unspecified elements may be mapped to
  non-zero values. So, in the domain of Calculus, the unspecified
  elements of sparse tensors can have any defined value.

Currently, in PyTorch, the unspecified elements of sparse tensors are
zero valued, but not only. For instance, the `torch.sparse.softmax`
function assumes that the unspecified elements of a sparse tensor
input are negative infinity (-inf) valued.

In PyTorch 1.6, element-wise functions from Calculus such as `exp`,
`log`, etc, or arithmetic operations such as addition, subtraction,
etc, on sparse tensors are not supported, because the existing
functions on sparse tensors assume that the unspecified elements are
zero valued (this applies to functions in `torch` namespace, functions
in `torch.sparse` namespace may use different interpretation for
unspecified elements).

To support Calculus functions on sparse tensors, we'll propose adding a
fill-value property to PyTorch sparse tensors that will represent the
values of unspecified sparse tensor elements.  While doing so, we also
need to consider
- how the non-zero fill-value affects the result of linear algebra
  operations such as matrix multiplication, etc.
- how to define a fill-value for hybrid tensors (tensors with sparse
  and dense dimensions),
- how PyTorch autograd support needs to deal with a defined
  fill-value.


## Proposal

This proposal is about enabling sparse tensor support for Calculus
domain while preserving the existing functionality for Linear Algebra
domain, and allowing extensions to Graph domain.

In the following, we describe the sparse tensor fill-value feature
with examples to sparse tensors in COO storage format.  When a new
sparse tensor storage format is introduced to PyTorch, the same
semantics apply to the new format.

1.  We propose to extend sparse tensor constructors with a keyword
    argument `fill_value` that users can use to define the value for
    unspecified elements of the constructed sparse tensor.

    For instance, the Python signature of `sparse_coo_tensor` would be

    ```python
    torch.sparse_coo_tensor(indices, values, size=None, fill_value=None, dtype=None, device=None, requires_grad=False)
    ```

2.  The default fill-value is `0`.

    This choice is consistent with the interpretation of unspecified
    element in Linear Algebra domain.

3.  PyTorch functions that have `layout` argument, may use different
    fill-value when constructing a sparse tensor as defined by the
    following table:

    | Function              | Fill-value of returned sparse tensor |
    | :-------------------- | :----------------------------------- |
    | `torch.arange`        | 0                                    |
    | `torch.empty`         | 0                                    |
    | `torch.empty_like`    | 0                                    |
    | `torch.empty_strided` | N/A                                  |
    | `torch.eye`           | 0                                    |
    | `torch.full`          | same as `fill_value` argument        |
    | `torch.full_like`     | same as `fill_value` argument        |
    | `torch.linspace`      | 0                                    |
    | `torch.logspace`      | 0                                    |
    | `torch.ones`          | 1                                    |
    | `torch.ones_like`     | 1                                    |
    | `torch.range`         | 0                                    |
    | `torch.zeros`         | 0                                    |
    | `torch.zeros_like`    | 0                                    |

    For discussion: should we introduce the `fill_value` argument to
    functions `empty` and `empty_like`?

4.  The fill-value of a sparse tensor can be acquired via
    `fill_value()` method that returns a strided `torch.Tensor`
    instance that has the same dtype and storage location properties
    as the elements of the sparse tensor:

    - `A.fill_value().dtype == A.dtype`
    - `A.fill_value().device == A.device`

    The fill-value of a sparse non-hybrid tensor is a scalar tensor
    `torch.tensor(0, dtype=A.dtype, device=A.device)`.

5.  The fill-value of a hybrid sparse tensor has the same shape as the
    dense part of the tensor.

    For instance, for a sparse tensor `A` in COO storage format we
    have:

    - `A.fill_value().shape == A.values().shape[1:]`

    While an obvious alternative to tensor fill-value would be a
    scalar fill-value, the scalar fill-value would complicate the
    implementation of algorithms for hybrid tensors. For
    instance:

    1. Accessing a dense element of a hybrid sparse tensor, the result
       is a dense tensor. When this element corresponds to unspecified
       element, one needs to construct a new tensor and fill it with
       the scalar fill-value. A more efficient and simple way is to
       just return the output of `fill_value()` that does would not
       involve constructing a new tensor on each method call.

    2. When applying a row-wise function (such as `softmax`, for
       instance) to a hybrid tensor, each row may induce a different
       fill-value.

       For example, consider a 2D hybrid tensor with one sparse and
       one dense dimension in COO format, and apply softmax along the
       first dimension assuming the default fill-value 0:

       ```python
       >>> A = torch.sparse_coo_tensor(indices=[[0, 3]],
                                       values=[[.11, .12], [.31, .32]],
                                       size=(4, 2))
       >>> A.to_dense()
       tensor([[0.1100, 0.1200],
               [0.0000, 0.0000],
               [0.0000, 0.0000],
               [0.3100, 0.3200]])
       >>> A.to_dense().softmax(0)
       tensor([[0.2492, 0.2503],
               [0.2232, 0.2220],
               [0.2232, 0.2220],
               [0.3044, 0.3057]])
       ```

       Observe that the softmax result can be represented as a sparse
       tensor with the same sparsity structure (same indices) as the
       input only when the fill-value is a 1-D dense tensor, that is,

       ```python
       A.softmax(0) == torch.sparse_coo_tensor(indices=A.indices(),
                                               values=[[0.2492, 0.2503], [0.3044, 0.3057]],
                                               size=A.size(),
                                               fill_value=[0.2232, 0.2220])
       ```

       If fill-value would be defined as scalar, the result of
       `A.softmax(0)` would need to be a dense tensor or raise
       exception about unsupported feature.

6.  The fill-value of a sparse tensor can be changed in-place.

    For instance,

    ```python
    A.fill_value().fill_(1.2)
    ```

    resets the fill value of a non-hybrid sparse tensor `A` to `1.2`.

7.  If `A` is a sparse tensor and `f` is any calculus function that is
    applied to a tensor element-wise, then

    ```python
    f(A) == torch.sparse_coo_tensor(A.indices(), f(A.values()), fill_value=f(A.fill_value()))
    ```

    Note that if `A` would be using COO storage format then this
    relation holds only if `A` is coalesced (`A.values()` would throw
    an exception otherwise).

8.  The fill-value of an element-wise n-ary operation on sparse
    tensors with different fill-values is equal to the result of
    applying the operation to the fill-values of the sparse
    tensors.

    For instance (`*` represents unspecified element),

    ```
    A = [[1, *], [3, 4]]                  # A fill-value is 2
    B = [[*, 6], [*, 8]]                  # B fill-value is 7
    A + B = [[1 + 4, 2 + 4], [*, 6 + 8]]  # A + B fill-value is 2 + 7 = 9
    ```

9.  Existing PyTorch functions that support sparse tensor as inputs,
    need to be updated for handling the defined fill-values. This will
    be implemented in two stages:

    1. All relevant functions need to check for zero fill-value. If a
       non-zero fill-value is used, raise an exception.

       For instance:

       ```C++
       TORCH_CHECK(A.fill_value().nonzero().numel() == 0,
                   "The <function> requires a sparse tensor with zero fill-value, got ",
                   A.fill_value());
       ```

       This check ensures that the existing functions will not produce
       silently incorrect results when user passes a sparse tensor
       with non-zero fill-value as an input while the function assumes
       it to be zero.

       For discussion: Since the zero fill-value check is required in
       all functions supporting sparse inputs, we may need a more
       efficient way than `.nonzero().numel() == 0` to determine if a
       tensor contains only zeros. This efficiency issue is relevant
       only for hybrid tensors with a large dense part.

    2. Update the related functions to handle non-zero fill-values of
       input sparse tensors correctly.

       For instance, consider a matrix multiplication of two sparse
       tensors `A` and `B` with fill-values `a` and `b`, respectively,
       then the `matmul` operation can be expanded as follows:

       ```python
       matmul(A, B) = matmul(A - fA + fA, B - fB + fB)
                    = matmul(A - fA, B - fB) + fA * matmul(ones_like(A), B) + fB * matmul(A, ones_like(B))
       ```

       where the first term can be computed using existing matmul for
       sparse tensors with zero fill-value, and the last two terms can
       be replaced with a computation of a single row or column of the
       corresponding matrix products that has reduced computational
       complexity.

10. Sparse tensors with defined fill-value have intrinsic constraints
    between all the unspecified tensor elements (these are always
    equal) that must be taken into account when implementing Autograd
    backward methods for functions that receive sparse tensors as
    inputs.

    Sparse tensors with undefined fill-value don't have the intrinsic
    constraints as discussed above.


### Future extensions and existing issues

11. For Graph domain, the undefined fill-value can be specified as a
    tensor with zero dimension(s) that satisfies all the relations
    listed above except point 5. Invalidation of the point 5 will
    provide a consistent way to differentiate between defined and
    undefined fill-values.

    For instance, to reset a fill-value of a sparse tensor to undefined
    fill-value, one can use:

    ```python
    A.fill_value().resize_((0,) * len(A.values().shape[1:]))
    ```

    Note that this operation is reversible, that is, to reset a
    undefined fill-value to a defined value, say `1.2`, one can use:

    ```python
    A.fill_value().resize_(A.values().shape[1:]).fill_(1.2)
    ```

    This works only when the initial fill-value of `A` is a defined
    value with the shape constraints specified in point 5.

12. For Statistics domain, where unspecified elements are interpreted
    as "missing data". The missing data concept is different from
    undefined value in the sense that it represents a certain
    measurement flaw that cannot be foreseen to appear in a data while
    the undefined value represents a structural lack of data that is
    determined by the system under study.

    The fill-value corresponding to missing data can be specified as
    `nan` (or a tensor of `nan`-s in case of hybrid tensor). This
    choice is consistent with convention used in Python Pandas
    package.

13. The introduction of non-zero fill-value feature requires
    revisiting the existing sparse tensor API.

    1. The acronym NNZ means the "Number of Non-Zeros" in a sparse
       tensor. In PyTorch, this acronym is used in several places:

       - in the implementations of sparse tensor and related
         functionality,
       - in the `repr` output of COO sparse tensor, see
         [here](https://pytorch.org/docs/master/generated/torch.sparse_coo_tensor.html#torch.sparse_coo_tensor),
       - as a private method `_nnz()`, see
         [here](https://pytorch.org/docs/master/sparse.html?highlight=nnz#torch.sparse.FloatTensor._nnz),
       - as a optional keyword argument `check_sparse_nnz` in
         [torch.autograd.gradcheck](https://pytorch.org/docs/master/autograd.html#numerical-gradient-checking).

       The acronym NNZ is misused in PyTorch:

       - `nnz` holds the value of the "Number of Specified Elements"
         (NSE) in a sparse tensor
       - `nnz` is not always equal to the number of zeros in the
         sparse tensor, for instance, the `values` of the sparse
         tensor in COO format may contain zero values that are not
         accounted in `nnz`

       With the introduction of non-zero fill-value, the misuse of
       acronym NNZ will deepen because with non-zero fill-value the
       sparse tensor may have no zero elements, e.g. `torch.full((10,
       10), 1.0, layout=torch.sparse_coo)` for which `nnz` would be
       `0`.

       Possible actions:

       - Stop the misuse of NNZ acronym: (i) replace the usage of
         "NNZ" with "NSE", (ii) deprecate the use of `_nnz()` in favor
         or `_nse()`, (iii) remove `_nnz()` starting from PyTorch 2.0.
       - Do nothing.  This is the (undocumented) approach taken in
         Wolfram Language where [one can use "NonzeroValues" to
         determine the number of specified
         elements](https://community.wolfram.com/groups/-/m/t/1168496)
         even when the fill-value specified is non-zero.
       - Other ideas?

14. This proposal is relevant to the following PyTorch issues:

   - [Sparse tensor eliminate zeros](https://github.com/pytorch/pytorch/issues/31742)
   - [Add sparse softmax/log_softmax functionality (ignore zero entries)](https://github.com/pytorch/pytorch/issues/23651)
   - [The state of sparse tensors](https://github.com/pytorch/pytorch/issues/9674)

## Application: random sequence of rare events with non-zero mean

Let's assume one has a device that periodically outputs a
mostly-constant signal with some occasional spikes, for instance, from
the neuroscience measurements of cells membrane potential:

<a href="https://commons.wikimedia.org/wiki/File:IPSPsummation.JPG"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a1/IPSPsummation.JPG" width="50%"/></a>


For simplicity, let the corresponding signal be

```
period:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 ... 1000000
signal:  5  5  5  7  5  5  5  5  6  8  5  5  5  5  5  5  5  9  5 ... 5
```

This sequence can be stored efficiently in a sparse tensor with `5` as
fill-value:

```python
signal = torch.sparse_coo_tensor(indices=[[3, 8, 9, 17, ...]],
                                 values=  [7, 6, 8,  9, ...],
                                 size=(1000001,),
                                 fill_value=5)
```

One can conveniently convert the signal to the relevant physical unit,
say, to [membrane
potential](https://en.wikipedia.org/wiki/Membrane_potential), using a
scalar conversion factor:

```python
mV_unit = -8.0
membrane_potential = signal * mV_unit
```

that can be then used in computing other physical quantities, for instance,

```python
intracellular_concentration_of_potassium = extracellular_concentration_of_potassium * torch.exp(-a * membrane_potential)
```

where `a` is a constant defined by [Nernst
equation](https://en.wikipedia.org/wiki/Nernst_equation). Notice that
the result `intracellular_concentration_of_potassium` is a sparse
tensor with a non-zero fill-value.

Without the non-zero fill-value support, the researchers of the field
would not be able to compute these quantities as simply as above
because they would need to keep track on the constant level of the
signals at different transformation steps. For instance, using Pytorch
1.6, the same computation of intracellular concentration of potassium
would be:

```python
# for membrane_potential:
membrane_potential_base = 5 * mV_unit
membrane_potential_var_values = signal.values() * mV_unit
# for torch.exp(a * membrane_potential):
factor_base = torch.exp(-a * membrane_potential_base)
factor_var_values = torch.exp(-a * membrane_potential_var_values)
# for intracellular_concentration_of_potassium:
intracellular_concentration_of_potassium_base = extracellular_concentration_of_potassium * factor_base
intracellular_concentration_of_potassium_var_values = intracellular_concentration_of_potassium * factor_var_values

intracellular_concentration_of_potassium_var = torch.sparse_coo_tensor(
    indices = signal.indices(),
    values = intracellular_concentration_of_potassium_var_values,
    size = signal.size())
```

which is convoluted, error-prone, and does not lead to solitary data
structure as is `intracellular_concentration_of_potassium` above.

### Review of sparse array software

As a rule, software that implement sparse matrix support, do not
implement support for non-zero fill-value. The reason for this is
two-fold: either the application of the software do not require this
feature (such as applications from Linear Algebra and Graph domains),
or the feature declared not implemented yet. However, there exists
also software that implement the non-zero fill-value support for
sparse matrices.  In the following, we will review a selection of
sparse array software to learn the different approaches how the
fill-value feature (or the lack of it) is handled.

### Wolfram Language

[SparseArray](https://reference.wolfram.com/language/ref/SparseArray.html)
has a constructor variant `SparseArray[data,dims,val]` that "yields a
sparse array in which unspecified elements are taken to have value
`val`". The default fill-value is 0.

### Pandas - Python Data Analysis Library

Pandas
[SparseArray](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.arrays.SparseArray.html#pandas.arrays.SparseArray)
supports `fill_value` option to enable memory saving for most common
value in an array. The default fill-value depends on array dtype, for
instance, it is `nan` for `float` and `0` for `int`. The
interpretation of the default fill-value is "missing data".

### MathWorks

Matlab's [sparse
matrix](https://www.mathworks.com/help/matlab/ref/sparse.html) does
not support non-zero fill-value feature. Historically, the idea of
non-zero fill-value was rejected in the interest of simplicity, [see
Sec 2.5 in this
paper](https://epubs.siam.org/doi/abs/10.1137/0613024).

### Scikit-learn/Scipy.sparse

[Scikit-learn defines two
semantics](https://scikit-learn.org/stable/glossary.html#term-sparse-matrix)
for interpreting the unspecified entries in a sparse matrix:

- Matrix (here Linear Algebra) semantics where the fill-value is
  assumed to be 0,
- Graph semantics where the fill-value is assumed to be undefined
  value.

The scikit-learn algorithms of the Matrix and Graph domains use sparse
arrays from
[scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html)
which do not record the fill-value in the sparse array object
explicitly. Instead, the corresponding algorithms make an implicit
assumption about the fill-value when processing the same Scipy sparse
arrays.

See also [the feature request of default value for scipy.sparse](https://stackoverflow.com/questions/6256206/scipy-sparse-default-value).

## Final notes

In this proposal, we propose attaching the fill-value to a sparse
tensor explicitly in order to cover the third domain, Calculus, that
deals mainly with element-wise operations on tensors. The explicit
definition of the fill value is required because the corresponding
calculus algorithms cannot make any implicit assumption of the
fill-value (the fill-value can be any defined value).

In conclusion, specifying the fill-value in sparse tensors remains
optional for Matrix and Graph domains, however, for Calculus domain,
the fill-value is essential for ensuring mathematical correctness of
algorithms that consider sparse tensors as a storage-efficient
representations of a general tensor concept.

The following table summarizes the fill-value choices for different
domains of scientific research:

| Domain         | Fill-value         | Typical application
| :--------------| :----------------- | :------------------
| Linear Algebra | 0                  | zero valued elements in sparse matrices, linear algebra matrix operations and decompositions
| Calculus       | any defined value  | most common value in a sparse array, element-wise operations
| Graph          | undefined value    | a non-edge of a graph, structural lack of data
| Statistics     | `nan`              | missing data or outlier due to experimental failure
