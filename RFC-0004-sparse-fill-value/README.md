# Adding fill value property to PyTorch sparse tensors

|            |                 |
| ---------- | --------------- |
| Authors    | Pearu Peterson  |
| Status     | Draft           |
| Type       | Process         |
| Created    | 2020-09-08      |
| Resolution | TBD             |

## Abstract

This proposal introduces a `fill_value` property to PyTorch sparse
tensors that generalizes the current interpretation of unspecified
elements from zero value to any value, including an indefinite
fill value as a future extension.

## Motivation and Scope

In general, the unspecified elements of sparse tensors have
domain-specific interpretations:
- In Linear Algebra domain, the unspecified elements are zero-valued.
- In Graph domain, the unspecified elements of adjacency matrices
  represent non-edges. In this document, a non-edge corresponds to
  an unspecified fill value.
- In neural networks, sparse tensors can be inputs, say, to activation
  functions that are defined in terms of elementary functions.
  Evaluation of such functions on sparse tensors can be element-wise,
  or one-dimension-wise operations. This includes the evaluation of
  the functions on the values corresponding to unspecified
  elements. Even when the unspecified elements are initially
  zero-valued, the values of unspecified elements may be mapped to
  nonzero values (see [an example
  below](#application-random-sequence-of-rare-events-with-non-zero-mean)). So,
  in the domain of Calculus, the unspecified elements of sparse
  tensors can have any defined value.

Currently, in PyTorch, the unspecified elements of sparse tensors are
zero-valued with few exceptions. For instance, the
`torch.sparse.softmax` function assumes that the unspecified elements
of a sparse tensor input are negative infinity (-inf) valued.

In PyTorch 1.6, element-wise functions from Calculus such as `exp`,
`log`, etc, or arithmetic operations such as addition, subtraction,
etc, on sparse tensors are not supported, because the existing
functions on sparse tensors assume that the unspecified elements are
zero valued (this applies to functions in `torch` namespace, functions
in `torch.sparse` namespace may use a different interpretation for
unspecified elements).

To widen support for sparse tensors to include more functions, we
propose adding a `fill_value` property to PyTorch sparse tensors that
will represent the values of unspecified sparse tensor elements.
While doing so, we also need to consider:
- how the nonzero fill value affects the result of linear algebra
  operations such as matrix multiplication, etc.
- how to define a fill value for hybrid tensors (tensors with sparse
  and dense dimensions),
- how PyTorch autograd support needs to deal with a specified
  fill value.


## Proposal

This proposal is about enabling sparse tensor support for the Calculus
domain while preserving the existing functionality for the Linear
Algebra domain, and allowing extensions to the Graph domain.

In the following, we describe the sparse tensor `fill_value` feature
with examples involving sparse tensors in COO storage format.  When a
new sparse tensor storage format is introduced to PyTorch, the same
semantics should be applied to the new format.

0.  Used terminology

    - "Defined value" is a value for which memory is allocated and
      initialized to some value.
    - "Uninitialized value" is a value for which memory is allocated, but
      the content of the memory can be arbitrary.
    - "Indefinite value" represents a structural lack of value.
    - "Sparse tensor format" is a memory-efficient storage format for tensors
      with many equal elements.
    - "Strided tensor format" is a process-efficient storage format
      for general tensors.
    - "Dense tensor" is a tensor with many non-equal elements. Dense
      tensors can be stored in any tensor format, including a sparse
      tensor format.
    - "Dense part" is a strided tensor that is obtained by fixing the
      indices of all sparse dimensions in a hybrid tensor.

1.  We propose to extend sparse tensor constructors with an extra keyword
    argument `fill_value`, used to define the value for
    unspecified elements of the constructed sparse tensor.

    For instance, the Python signature of `sparse_coo_tensor` would be:

    ```python
    torch.sparse_coo_tensor(indices, values, size=None, fill_value=None, dtype=None, device=None, requires_grad=False)
    ```

    where `fill_value=None` indicates the constructor to use the
    default fill value.

2.  The default fill value is zero.

    This choice is consistent with the interpretation of an unspecified
    element in Linear Algebra.

    The default fill value of a sparse non-hybrid tensor is a scalar
    tensor: `torch.tensor(0, dtype=A.dtype, device=A.device)`. For the
    default fill value of a hybrid tensor, see points 5 and 6 below.

3.  PyTorch functions that have the `layout` argument may use a different
    `fill_value` when constructing a sparse tensor as defined by the
    following table:

    | Function              | `fill_value` of returned sparse tensor |
    | :-------------------- | :------------------------------------- |
    | `torch.empty`         | uninitialized value                    |
    | `torch.empty_like`    | uninitialized value                    |
    | `torch.eye`           | 0                                      |
    | `torch.full`          | same as `fill_value` argument          |
    | `torch.full_like`     | same as `fill_value` argument          |
    | `torch.ones`          | 1                                      |
    | `torch.ones_like`     | 1                                      |
    | `torch.zeros`         | 0                                      |
    | `torch.zeros_like`    | 0                                      |

    Note that this table does not include functions `torch.arange`,
    `torch.empty_strided`, `torch.linspace`, `torch.logspace` and
    `torch.range` that have the `layout` argument as well. We excluded
    these functions here because these are likely never used for
    creating sparse tensors.  See also point 14.ii below.

4.  The fill value of a sparse tensor can be acquired via the
    `fill_value()` method that returns a strided `torch.Tensor`
    instance with the same dtype and storage location properties as
    the elements of the sparse tensor:

    - `A.fill_value().dtype == A.dtype`
    - `A.fill_value().device == A.device`

    See point 6 below for how the output of `fill_value()` method is
    computed.

5.  The fill value of a hybrid sparse tensor has the same shape as the
    dense part of the tensor.

    For instance, for a sparse tensor `A` in COO storage format we
    have:

    - `A.fill_value().shape == A.values().shape[1:]`

    Using a non-scalar fill value for hybrid tensors is a natural
    extension of a scalar fill value for non-hybrid tensors:

    1. Hybrid tensors can be considered as sparse arrays with dense
       tensors as elements. The `fill_value` attribute represents an unspecified
       element of the array, hence, the fill value has all the
       properties that any array element has, including the shape and
       memory consumption.

    2. A non-scalar fill value of a hybrid tensor is more memory-efficient
       than if the fill value were a scalar.

       For instance, when applying a row-wise function, say, `softmax`
       to a hybrid tensor, each row may induce a different fill value.
       As an example, consider a 2D hybrid tensor with one sparse and
       one dense dimension in COO format, and apply softmax along the
       first dimension assuming the default fill value `0`. We have:

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
       input only when the fill value is a 1-D dense tensor, that is:

       ```python
       A.softmax(0) == torch.sparse_coo_tensor(indices=A.indices(),
                                               values=[[0.2492, 0.2503], [0.3044, 0.3057]],
                                               size=A.size(),
                                               fill_value=[0.2232, 0.2220])
       ```

       The alternative, if the fill value was defined as scalar,
       would be memory inefficient because the result of
       `A.softmax(0)` is a full sparse tensor with all elements
       specified.

6.  The fill value that is specified in the `fill_value` argument of
    the sparse tensor constructors may have different (smaller) shape
    than the fill value of the hybrid tensor (as defined by point 5).

    The specified fill value (after converting it to `torch.Tensor`
    instance) can be acquired via `_fill_value()` method.

    For example, the fill value of a (1+1)-D hybrid tensor can be
    specified as scalar:

    ```python
    A = torch.sparse_coo_tensor(indices=[[0, 3]],
                                values=[[.11, .12], [.31, .32]],
                                size=(4, 2),
                                fill_value = 1.2)
    A.fill_value() -> torch.tensor([1.2, 1.2])
    A._fill_value() -> torch.tensor(1.2)
    ```

    The output of `fill_value()` is computed as

    ```python
    A._fill_value().expand(A.values().shape[1:])
    ```

    Storing the specified fill value instead of the fill value of the
    hybrid tensor has several advantages:

    - the specified fill value may have a smaller size than the fill
      value of the hybrid tensor,
    - optimal evaluation of element-wise functions, see point 8 below,
    - and most importantly, optimal detection of zero fill value, see
      point 10 below.


7.  The fill value of a sparse tensor can be changed in place.

    For instance:

    ```python
    A._fill_value().fill_(1.2)
    ```

    resets the fill value of a sparse tensor `A` to be `1.2`.

8.  If `A` is a sparse tensor and `f` is any calculus function that is
    applied to a tensor element-wise, then:

    ```python
    f(A) == torch.sparse_coo_tensor(A.indices(), f(A.values()), fill_value=f(A._fill_value()))
    ```

    Note that if `A` would be using COO storage format then this
    relation holds only if `A` is coalesced (`A.values()` would throw
    an exception otherwise).

    From this relation follows an identity:

    ```python
    f(A).to_dense() == f(A.to_dense())
    ```

    that will be advantageous in testing the sparse tensor support of
    element-wise functions.

9.  The fill value of an element-wise n-ary operation on sparse
    tensors with different fill values is equal to the result of
    applying the operation to the fill values of the sparse
    tensors.

    For instance (`*` represents unspecified element),

    ```
    A = [[1, *],
         [3, *]]
    B = [[5, *],
         [*, 8]]
    # assume A and B fill values are 2 and 6, respectively, then
    A + B = [[6,  *],
             [9, 10]]
    # with fill value 2 + 6 = 8
    ```

10. Existing PyTorch functions that support sparse tensor as inputs,
    need to be updated for handling the defined fill values. This will
    be implemented in two stages:

    1. All relevant functions need to check for zero fill value. If a
       nonzero fill value is used, raise an exception.

       For instance:

       ```C++
       TORCH_CHECK(A._fill_value().nonzero().numel() == 0,
                   "The <function> requires a sparse tensor with zero fill value, got ",
                   A.fill_value());
       ```

       This check ensures that the existing functions will not produce
       incorrect results silently when users pass a sparse tensor
       with nonzero fill value as an input while the function assumes
       it to be zero.

    2. Update the related functions to handle nonzero fill values of
       input sparse tensors correctly.

       For instance, consider a matrix multiplication of two sparse
       tensors `A` and `B` with fill values `a` and `b`, respectively,
       then the `matmul` operation can be expanded as follows:

       ```python
       matmul(A, B) = matmul(A - fA + fA, B - fB + fB)
                    = matmul(A - fA, B - fB) + fA * matmul(ones_like(A), B) + fB * matmul(A, ones_like(B))
       ```

       where the first term can be computed using existing matmul for
       sparse tensors with zero fill value, and the last two terms can
       be replaced with a computation of a single row or column of the
       corresponding matrix products that has reduced computational
       complexity.

       In general, updating all linear algebra functions to support
       nonzero fill value will be a notable effort and it is not the
       aim of this proposal to seek for it immediately.

11. We propose to add an optional argument `fill_value` to the `to_sparse`
    method:

    ```python
    torch.Tensor.to_sparse(self, sparseDims=None, fill_value=None)
    ```

    that enables efficient construction of sparse tensors from tensors
    with repeated values equal to given `fill_value`. For example:

    ```python
    torch.ones(10).to_sparse(fill_value=1.) -> torch.sparse_coo_tensor([[]], [], (10,), fill_value=1.)
    ```

    The `fill_value` argument has the same semantics as in sparse
    tensor constructors (see point 1) and its shape must be consistent
    with the shape of the input tensor (`self`) and the value of
    `sparseDims` when specified: `self.shape[:sparseDims]` must be
    equal to `<constructed sparse tensor>.fill_value().shape`.

12. Sparse tensors with defined fill value have intrinsic constraints
    between all the unspecified tensor elements (these are always
    equal) that must be taken into account when implementing Autograd
    backward methods for functions that receive sparse tensors as
    inputs.


### Future extensions and existing issues

Introducing the fill value feature according to this proposal does not
require addressing the following extensions and issues. These are
given here as suggestions to clean up the PyTorch sparse tensor
support in general.

13. For the Graph domain, the indefinite fill value can be specified as a
    tensor with zero dimension(s) that satisfies all the relations
    listed above except point 5. Invalidation of the point 5 will
    provide a consistent way to differentiate between defined and
    indefinite fill values.

14. The introduction of a nonzero fill value feature encourages a
    revisit of the existing PyTorch tensor API.

    1. The acronym NNZ means the "Number of nonzeros" in a sparse
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

       With the introduction of nonzero fill values, the misuse of
       acronym NNZ will get worse because with nonzero fill value the
       sparse tensor may have no zero elements, e.g. `torch.full((10,
       10), 1.0, layout=torch.sparse_coo)` for which `nnz` would be
       `0`.

       Recommendation: stop the misuse of NNZ acronym via

       For the reasons above, this proposal recommends

       - renaming "NNZ" to "NSE",
       - deprecating the `_nnz()` method in favor of `_nse()` method,
       - removing the `_nnz()` method starting from PyTorch 2.0.

       Alternative: Do nothing.  This is the (undocumented) approach
       taken in Wolfram Language where [one can use "NonzeroValues" to
       determine the number of specified
       elements](https://community.wolfram.com/groups/-/m/t/1168496)
       even when the fill value specified is nonzero.

    2. The `torch` namespace functions `arange`, `range`, `linspace`,
       and `logspace` have `layout` argument that is not needed.

       Currently, PyTorch defines three layouts: `strided`,
       `sparse_coo`, and `_mkldnn`. Because the mentioned functions
       output 1-D tensors with non-equal values, one never uses
       `sparse_coo` or `mkldnn` layout in the context of these
       functions. In fact, using `torch.sparse_coo` or `torch._mkldnn`
       as the `layout` argument is currently not supported.

       We propose to remove the `layout` argument from these functions
       in two stages:

       - deprecate the explicit use of the `layout` argument in
         `arange`, `range`, `linspace`, `logspace` function calls.

       - remove the `layout` argument from `torch.arange`,
         `torch.linspace` and `torch.logspace` starting from PyTorch
         2.0.

       In addition, remove the currently deprecated function
       `torch.range` starting from PyTorch 2.0.

       If one needs the output of, say, `torch.arange` to use sparse
       storage format, one can use the `to_sparse` method:
       `torch.arange(...).to_sparse()`. Similarly, one can use
       `to_mkldnn()` method instead of specifying
       `layout=torch._mkldnn`.

    3. The method `torch.Tensor.to_sparse()` can be used for
       converting in-between different sparse storage formats.

       For that, the `layout` argument needs to be added to the
       `to_sparse()` method.

       For example, `A.to_sparse(layout=torch.sparse_gcs)` would
       convert a (strided or sparse COO) tensor `A` to a sparse tensor
       in [GCS storage
       format](https://github.com/pytorch/pytorch/pull/44190).

15. Related PyTorch issues:

   - [Sparse tensor eliminate zeros](https://github.com/pytorch/pytorch/issues/31742)
   - [Add sparse softmax/log_softmax functionality (ignore zero entries)](https://github.com/pytorch/pytorch/issues/23651)
   - [The state of sparse tensors](https://github.com/pytorch/pytorch/issues/9674)
   - [torch.softmax on sparse tensors requires the fill value feature](https://github.com/pytorch/pytorch/pull/36305#issuecomment-617622038)
   - [Compute the fill value of softmax on sparse tensor with nonzero fill value](https://github.com/pytorch/pytorch/pull/36305#issuecomment-617622038)


## Application: random sequence of rare events with nonzero mean

Let's assume one has a device that periodically outputs a mostly
constant signal with some occasional spikes, for instance, from the
neuroscience measurements of cells membrane potential:

<a href="https://commons.wikimedia.org/wiki/File:IPSPsummation.JPG"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a1/IPSPsummation.JPG" width="50%"/></a>


For simplicity, let the corresponding signal be:

```
period:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 ... 1000000
signal:  5  5  5  7  5  5  5  5  6  8  5  5  5  5  5  5  5  9  5 ... 5
```

This sequence can be stored efficiently in a sparse tensor with `5` as
fill value:

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

that can be then used in computing other physical quantities, for instance:

```python
intracellular_concentration_of_potassium = extracellular_concentration_of_potassium * torch.exp(-a * membrane_potential)
```

where `a` is a constant defined by [Nernst
equation](https://en.wikipedia.org/wiki/Nernst_equation). Notice that
the result `intracellular_concentration_of_potassium` is a sparse
tensor with a nonzero fill value.

Without the nonzero fill value support, researchers in this field
would not be able to compute these quantities as simply as above
because they would need to keep track on the constant level of the
signals at different transformation steps. For instance, using PyTorch
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

which is convoluted, error-prone, and does not result in a single data
structure as is `intracellular_concentration_of_potassium` above.

## Review of sparse array software

As a rule, software that implements sparse matrix support does not
implement support for nonzero fill value. The reason for this is
twofold: either the application of the software do not require this
feature (such as applications from Linear Algebra and Graph theory),
or the feature is declared 'not implemented yet'. However, software
also exists that implements the nonzero fill value support for sparse
matrices.  In the following, we will review a selection of sparse
array software to learn the different approaches to handling the fill
value feature (or the lack of it).

### Wolfram Language

[SparseArray](https://reference.wolfram.com/language/ref/SparseArray.html)
has a constructor variant `SparseArray[data,dims,val]` that "yields a
sparse array in which unspecified elements are taken to have value
`val`". The default fill value is 0.

### Pandas - Python Data Analysis Library

Pandas
[SparseArray](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.arrays.SparseArray.html#pandas.arrays.SparseArray)
supports `fill_value` option to enable memory saving for most common
value in an array. The default fill value depends on array dtype, for
instance, it is `nan` for `float` and `0` for `int`. The
interpretation of the default fill value is "missing data".

### MathWorks

Matlab's [sparse
matrix](https://www.mathworks.com/help/matlab/ref/sparse.html) does
not support nonzero fill value feature. Historically, the idea of
nonzero fill value was rejected in the interest of simplicity, [see
Sec 2.5 in this
paper](https://epubs.siam.org/doi/abs/10.1137/0613024).

### Scikit-Learn/SciPy.sparse

[Scikit-Learn defines two
semantics](https://scikit-learn.org/stable/glossary.html#term-sparse-matrix)
for interpreting the unspecified entries in a sparse matrix:

- Matrix (here Linear Algebra) semantics where the fill value is
  assumed to be 0,
- Graph semantics where the fill value is assumed to be indefinite
  value.

The Scikit-Learn algorithms of the Matrix and Graph domains use sparse
arrays from
[scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html)
which do not record the fill value in the sparse array object
explicitly. Instead, the corresponding algorithms make an implicit
assumption about the fill value when processing the same SciPy sparse
arrays.

The [PyData/Sparse project](https://sparse.pydata.org/en/stable/)
extends SciPy sparse arrays with `fill_value`, see
[SparseArray](https://sparse.pydata.org/en/stable/generated/sparse.SparseArray.html).

## Final notes

This proposal is to explicitly attach the fill value to a sparse
tensor in order to cover the Calculus domain, that deals mainly with
element-wise operations on tensors. The explicit definition of the
fill value is required because the corresponding calculus algorithms
cannot make any implicit assumption of the fill value (the fill value
can be any defined value).

In conclusion, specifying the fill value in sparse tensors remains
optional for Matrix and Graph domains, however, for the Calculus
domain, the fill value is essential for ensuring mathematical
correctness of algorithms that consider sparse tensors as a
storage-efficient representations of a general tensor concept.  In
addition, explicit definition of the fill value ensures that
applications with interdisciplinary nature will work correctly when
the same sparse tensor is used as input to functions that otherwise
would have different implicit interpretations about the unspecified
elements.

The following table summarizes the fill value choices for different
domains of scientific research:

| Domain         | fill value         | Typical application
| :--------------| :----------------- | :------------------
| Linear Algebra | 0                  | zero valued elements in sparse matrices, linear algebra matrix operations and decompositions
| Calculus       | any defined value  | most common value in a sparse array, element-wise operations
| Graph          | indefinite value   | a non-edge of a graph, structural lack of data
