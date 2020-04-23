# Roadmap for `torch.sparse` Matrix Product API

|            |                 |
| ---------- | --------------- |
| Authors    | Dhavide Aruliah |
| Status     | Active          |
| Type       | Process         |
| Created    | 2020-04-23      |
| Resolution | TBD             |


## Abstract

At the time of writing, the 
[`torch.sparse`](https://pytorch.org/docs/stable/sparse.html) module API is 
considered experimental. There are more than 60 unresolved GitHub 
issues involving `torch.sparse` to address&mdash;some dating back to 
2018&mdash;and a handful of which have been flagged as high priority. About
twenty of those issues deals with *multiplication* of sparse PyTorch
matrices and at least one of those is a high priority issue.

This proposal summarizes an overarching strategy for addressing a number 
`torch.sparse` issues related to the *performance of sparse matrix multiplication* in 
PyTorch. There are, of course, many other `torch.sparse` issues flagged on 
GitHub that point to other desired features & API design considerations; these 
particular matrix multiplication issues are a useful starting point for further 
API improvements for this module. That is, dealing with these issues first can
seed development of additional new `torch.sparse` features.

## Motivation and Scope

Almost one third of the current `torch.sparse` issues relate to the timings of 
matrix multiplication with sparse matrices (i.e., sparse tensors of rank 2) in 
PyTorch (e.g., issue [\#5262](https://github.com/pytorch/pytorch/issues/5262)). 
There are additional issues relating to batching sparse matrix products (e.g., 
issue [\#33430](https://github.com/pytorch/pytorch/issues/33430)) and computing 
gradients (e.g., issue 
[\#12498](https://github.com/pytorch/pytorch/issues/12498)) but these can 
likely be addressed *after* the performance issues. Note also that some issues 
that refer to "multiplication" explicitly are unrelated in that they refer to 
*elementwise* products & *broadcasting* with sparse tensors (e.g., issue 
[\#2909](https://github.com/pytorch/pytorch/issues/2909)).

For context, the internal representation of PyTorch sparse tensors uses
[*COO* (*coordinate*) format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)),
i.e., using a rank 1 `FloatTensor` of length `nnz` for the non-zero values and 
a `d x nnz` `IntTensor` to store the positions of those entries from a sparse 
tensor of rank `d`:

```{python}
>>> i = torch.LongTensor([[0, 1, 1],
                          [2, 0, 2]])
>>> v = torch.FloatTensor([3, 4, 5])
>>> torch.sparse.FloatTensor(i, v, torch.Size([2,3])).to_dense()
 0  0  3
 4  0  5
[torch.FloatTensor of size 2x3]
```
The COO storage scheme is favoured for representing high-dimensional sparse
tensors in practice because it is easy to insert new elements: simply
append a value to the end of the values array and concatenate a column of
height `d` to the right of the index matrix. It is also flexible in that
the elements do not need to be sorted in any particular order.

One unusual feature of the PyTorch COO representation of tensors is that it 
permits a *hybrid* of sparse and dense dimensions. That is, the leading
dimensions can be sparse and the remaining dimensions are dense; the methods
`sparse_dim` and `dense_dim` reveal those dimensions explicitly. Another way
to think of this is as a (sparse) tensor with (dense) tensors as entries.

It is important to observe that the `torch.sparse` API currently supports 
*uncoalesced* sparse tensors (i.e., sparse tensors in COO format with duplicate 
coordinates in the indices). Values associated with repeated indices in an
uncoalesced sparse tensor are added together to yield the value associated with
that index. Permitting uncoalesced sparse tensors improves the 
efficiency of certain operations (notably addition of sparse tensors and 
insertion of new nonzero elements into a sparse tensor).

For instance, in computing `(A + B + C) * D` for some tensors `A`, `B`, `C`,
& `D` of identical shape, the uncoalesced sum `(A + B + C)` can be computed
by concatenating the lists of indices & values from all three tensors. To
compute the elementwise product with `(A + B + C) * D`, the tensor
`(A + B + C)` must be coalesced first; this involves sorting the indices and
summing values associated with repeated indices). By contrast, forcing
sparse tensor operations to yield coalesced sparse COO tensors requires
unnecessary additional work: sorting indices to compute `(A + B)` and then
again to compute `(A + B) + C`.

The SciPy documentation for the [`csr_matrix`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html#scipy.sparse.csr_matrix)
class implies that COO format is useful mainly for assembling sparse matrices
(e.g., in the context of finite element analysis). By contrast, a number of
sparse linear algebra libraries&mdash;including
[`scipy.sparse`](https://docs.scipy.org/doc/scipy/reference/sparse.html)&mdash;make
use of the
[*CSR* (*compressed sparse row*)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
storage format to represent sparse matrices.
It resembles COO storage, but compresses the row indices, hence the name.
There is a closely related
[*CSC* (*compressed sparse column*)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS))
storage format where the roles of rows & columns are interchanged. Notice
that both CSR & CSC are mostly used with sparse tensors of rank two;
they can generalize to sparse tensors of arbitrary rank `n` (see below). Other
related formats exist (e.g.,
[GCRS](https://ieeexplore.ieee.org/document/7237032) and
[CSF](https://dl.acm.org/doi/10.1145/2833179.2833183)), but these formats are not
currently supported by most libraries for sparse linear algebra.

The CSR format replaces a dense matrix by three (rank 1) arrays, that
respectively contain the extents of rows, the column indices, and the numerical
values. That is, to represent an `M x N` sparse matrix `A` with `nnz` nonzero
entries (floating-point) in CSR format requires three separate rank 1 arrays:
`RO` (an integer array of length `M+1`), `CO` (an integer array of length
`nnz`), and `VL` (a floating-point array of length `nnz`). In the
extension of CSR format to tensors of rank greater `d>2`, the array `CO`
would be a matrix of dimensions `(d-1) x nnz`.

The entry in `RO[r]` is the cumulative number of nonzero entries in the matrix
up to and including row `r` (`r=0...M+1`). Thus, the first entry of `RO` is `0`
and the last&mdash;`RO[M+1]`&mdash;is `nnz`. The arrays `CO` and `VL` of length
`nnz`, then, store the column indices and the values of the nonzero entries of
the matrix when sorted lexicographically by the `(row, column)` index pairs.
This means that the values of the `r`th row of `A` can be sliced by extracting
elements `RO[r]` to `RO[r+1]` from the remaining arrays `CO` and `VL`. The
slice `CO[RO[r]:RO[r+1]]` give the column locations of the nonzero entries in
row `r` in increasing column order. The corresponding numerical entries would
be stored in the slice `VL[RO[r]:RO[r+1]]`. For CSC format for representing
sparse matrices, `RO` and `CO` are still tensors of rank 1 but the roles of
rows and columns are reversed from CSR format. In the extension of CSR storage
to tensors of rank `d>2`, the columns of `CO` contain the remaining `d-1`
index entries that correspond to the nonzero entries of `A` sorted in
lexicographic order.

As an illustration of CSR storage, consider this `MxN` sparse matrix `A` with
`M=4` rows, `N=6` columns and `nnz=8` nonzero entries:
```{python}
[[10, 20,  0,  0,  0,  0],
 [ 0, 30,  0, 40,  0,  0],
 [ 0,  0, 50, 60, 70,  0],
 [ 0,  0,  0,  0,  0, 80]]
```
For this matrix, the corresponding arrays are:
```{python}
RO = [0, 2, 4, 7, 8] 
CO = [0, 1, 1, 3, 2, 3, 4, 5]
VL = [10, 20, 30, 40, 50, 60, 70, 80]
```

As slicing the nonzero elements of rows of `A` is straightforward in CSR
format, this storage scheme leads to efficient algorithms for computing sparse
*matvecs* (*matrix-vector products*). Indeed, the principle reason to favour CSR
& CSC formats for storing sparse matrices is that those formats lend themselves
easily to row- or column-oriented operations on sparse matrices (e.g., BLAS
Level 2 operations like a matvec & BLAS Level 3 operations like a
matrix-matrix product).

Lower-level libraries like
[Intel's MKL](https://software.intel.com/en-us/mkl-developer-reference-fortran-blas-and-sparse-blas-routines)
provide sparse BLAS algorithms using numerous internal representations including
COO & CSR. On CUDA architecture GPUs, the sparse linear algebra libraries
[cuSPARSE](https://developer.nvidia.com/cusparse),
[CUSP](https://developer.nvidia.com/cusp), and
[MAGMA](https://ceed.exascaleproject.org/magma/#magma-sparse)
all support sparse matrix computations in both CSR & COO formats.
There are
[benchmarks](https://pdfs.semanticscholar.org/5478/de620de99b1d4e41599a8950c7d3d9ff07b5.pdf)
that suggest that sparse matvecs can be between 2 and 2.5 times faster when
using CSR format over COO storage. Of course, with uncoalesced COO format,
there are additional concerns about deterministic computations on GPUs that
arise due to nonuniqueness of the internal representation in both the sequence
and contents of the index and values arrays.

GitHub issue [\#16187](https://github.com/pytorch/pytorch/issues/16187)
presents timed comparisons of PyTorch & SciPy; this is one of several issues
that draws attention to performance deficiencies of PyTorch sparse matrix
products. This performance penalty in PyTorch sparse matrix products has numerous
potential causes:

+ converting from COO to CSR format;
+ coalescing sparse COO tensors;
+ not using optimized libraries that employ, e.g., [sparse BLAS](https://math.nist.gov/spblas/); or
+ some combination of the above.

[This Stack Overflow
discussion](https://stackoverflow.com/questions/30118673/why-cusparse-is-much-slower-than-cublas-for-sparse-matrix-multiplication)
provides some additional context; a fairly deep exploration is needed to assess
the precise cause of the slowdown.

With the considerations above, it makes sense to explore extending
`torch.sparse` to provide support for storing sparse matrices in CSR format
(in the special case of tensors of rank 2). There
are two alternative schemes to consider:

+ providing user-accessible conversion routines between COO & CSR 
representations for rank 2 tensors; and
+ using a cached internal CSR representation.

The first approach is most obvious; its primary disadvantage is that 
supporting alternative storage formats requires modifying large portions of the 
codebase (because, e.g., the algorithms for elementwise
arithmetic operations differ for CSR & COO formats). These code overheads can
be avoided using the second option instead, i.e., by caching. However, caching
a CSR representation of a sparse tensor on top of the COO representation
effectively doubles the storage required and introduces additional complications
in having to update the cached representation on modification of any individual
element. Assuming that the caching problems are worse, we'll consider the first
approach only in this proposal.

The main goals of this proposal, then, are:

1. to implement a CSR representation for `torch.sparse` tensors of 
   rank 2 with appropriate modifications to the API;
2. to implement changes to *all* associated elementwise arithmetic operations
   and similar methods that would, by necessity, need to be updated to ensure
   compatibility with the new CSR representation; and
3. to extend the PyTorch sparse array API to include the `@` infix operator
   for matrix multiplication (as supported since Python 3.5).

With regard to the third goal, the
[PEP 465](https://www.python.org/dev/peps/pep-0465/)
describes how the infix matrix multiplication operator `@` works for NumPy
arrays in various contexts. This allows, for instance, the expression

```
A @ B + C
```

to be computed meaningfully with any combination of PyTorch tensors
`A`, `B`, and `C` with appropriately compatible dimensions
(compatible in the sense of matrix multiplication and
[NumPy broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)).
Ideally, the formula can be interpreted in a manner that is agnostic as to
whether the PyTorch tensors are sparse or dense. This would obviate the use of
redundant functions like `torch.matmul`, `torch.sparse.mm`, and so on, thereby
improving readability.

## Usage and Impact

Optimizing sparse matrix multiplication in PyTorch is valuable to a broad 
spectrum of users (as evidenced by repeated independent GitHub issues raised). 
Unfortunately, addressing sparse matrix multiplication requires thinking about 
internal representations (i.e., *COO*  vs. *CSR* format).  Additional care is 
required in addressing how products are handled between dense and sparse 
tensors as opposed to sparse tensors only or dense tensors only. There is also 
the issue of ensuring that any changes made are compatible with `autograd` for 
computation of gradients through the `backward` method.

The big assumption is that faster sparse matrix products actually constitutes a 
meaningful benefit to PyTorch users&mdash;the hope is that the accuracy of this 
assumption will become clear in the early stages of prototyping.

## Backward compatibility

Ideally, none of the proposed work will break backward compatibility. The 
proposed alterations largely consist of adding an API for a CSR representation 
of sparse tensors of rank 2 to enable certain 
sparse linear algebra improvements. Unfortunately, adding a CSR representation
for sparse tensors does require modifying the implementation of sparse tensor
classes significantly. In particular, all relevant sparse tensor operations
(elementwise arithmetic operations, etc.) need to dispatch different algorithms
for the CSR and COO representations. Prior work on caching CSR representations
in PyTorch in PR [#6225](https://github.com/pytorch/pytorch/pull/6225)
illustrates some of these implementation issues.

One future feature to be mindful of is a `fill_value` attribute for sparse
tensors; this would allow a value other than zero to fill all the unspecified
entries. Enabling nonzero fill values is useful in certain sparse tensor
computations. As an example, the tensor `B` computed by `B = torch.cos(A)`
contains ones in all locations that `A==0`; if `A` is sparse with
`A.fill_value==0`, then `B` would also be sparse with `B.fill_value==1.0`.
This capability is implemented in
[`pydata/sparse`](https://github.com/pydata/sparse)
and is required for some features requested (see, e.g.,
[issue \#8853](https://github.com/pytorch/pytorch/issues/8853)). This feature
is not covered by this proposal, but it will affect matrix products when it is.

Finally, there is some additional work required to ensure that every way of
computing matrix products is tested & updated appropriately (e.g.,
`torch.matmul`, `torch.sparse.mm`, `torch.sparse.FloatTensor.mm`, etc.).
There should be a thorough pass through the existing PyTorch codebase to ensure
that any code relying on sparse tensors still functions as intended with the
changes made. A suitably robust suite of unit tests should be designed to verify
that correct behaviour is preserved with any modifications to the code. This
would include ensuring GPU support as well as compliance with `autograd`.

## Related Work

Much prior work on sparse matrices is implemented in the 
[`scipy.sparse`](https://docs.scipy.org/doc/scipy/reference/sparse.html) 
module; the [`pydata/sparse`](https://github.com/pydata/sparse) project extends 
some of those ideas to sparse arrays of generic rank `n`. As a stretch goal, it 
may make sense to have general rank `n` PyTorch tensors being supported in most 
of the same ways as `pydata/sparse` and to have sparse tensors of rank 2
supported in most of the same ways as `scipy.sparse`. 

Among projects tailored to neural networks & deep learning,
[Tensorflow](https://www.tensorflow.org/api_docs/python/tf/raw_ops/SparseMatrixSparseMatMul)
has a `SparseTensor` class that uses COO format for tensors of arbitrary
rank `n` and a `SparseMatrix` class that uses CSR format for tensors of
rank 2; explicit conversion routines between formats exist where they apply. 
There are also rank 2 `CSRNDArray` and rank 1 `RowSparseMatrix` classes 
included in
[MXNet](https://www.tensorflow.org/api_docs/python/tf/raw_ops/SparseMatrixSparseMatMul).
The project [`pytorch_sparse`](https://github.com/rusty1s/pytorch_sparse)
suggests some directions for supporting efficient sparse matrix operations
with `autograd` and also with CUDA support.

Finally, there are generalizations of sparse matrix storage formats to sparse 
tensors of arbitrary rank `n` described in various articles listed in the 
references.

## Implementation

The module `torch` already contains an attribute `torch.sparse_coo` of type
`torch.layout` with an associated constructor `torch.sparse_coo_tensor`. The
proposed CSR implementation should extend the API by adding a few attributes
to the `torch` module with similar names:

+ `torch.sparse_csr` (an attribute of type `torch.layout`);
+ `torch.sparse_csr_tensor` (an associated constructor); and
+ functions `torch.to_sparse_csr` & `torch.to_sparse_coo` for explicit
  conversion between representations (for rank 2 sparse tensors).

The `torch.sparse_csr_tensor` constructor would have a similar signature to `torch.sparse_coo_tensor`, i.e.,

```{python}
sparse_csr_tensor(indices, values, size=None, dtype=None,
                  device=None, requires_grad=False)       -> Tensor
```

where `indices` would be a list of two lists (e.g., `RO` & `CO`
from the description of CSR format above) and `values` would be a tensor
containing the corresponding (presumably) nonzero entries of the sparse
tensor. This is sufficiently general to account for tensors of rank `d>2`,
thereby enabling batches of sparse matrices. It should also be acceptable
for `values` to contain tensors (of fixed dimensions) rather than scalars.

The basic outline, then, is as follows:

1. Construct an exhaustive suite of benchmark tests to time the existing
   implementation of sparse matrix-vector & matrix-matrix products in PyTorch.

   + See [issue # 16187](https://github.com/pytorch/pytorch/issues/16187)
     for representative examples of benchmarks.
   + Include examples of matrices of fairly large dimensions (i.e., where dense
     representations are impractical or infeasible).
   + Include examples with `sparse x dense` and `sparse x sparse` matrices for
     the timings.

2. Construct and test a prototype implementation of conversion functions
   `torch.to_sparse_csr` and `torch.to_sparse_coo` to convert sparse PyTorch
   tensors of sparse rank 2 between COO and CSR formats.

   + The implementation of related functions in 
     [Tensorflow](https://www.tensorflow.org/api_docs/python/tf/raw_ops/SparseMatrixSparseMatMul)
     can serve as a useful model for these transformations.
   + Include exception handling to ensure rank 2 tensors before converting
     from COO to CSR storage.
   + Include tests to ensure that conversion functions work as intended on CUDA,
     i.e., when executed on GPU hardware.
   + See also comments in issue
     [\#16187](https://github.com/pytorch/pytorch/issues/16187).

3. Update & test the API for PyTorch sparse tensors for *all* relevant methods
   to ensure correct support for both COO format and CSR format for rank 2
   tensors.

   + This would include, e.g., the `__add__`, `__sub__`, `__mul__`, `__div__`,
     and most other methods (specifically those that do *not* rely on computing
     matrix products).
   + Include tests with all sparse COO inputs, all sparse CSR inputs, and mixed
     sparse COO & sparse CSR inputs.
   + Include tests with all sparse inputs as well as mixed sparse & dense
     inputs, e.g.,

     `A (sparse tensor) + B (dense tensor) -> (A+B) (dense tensor)`

4. Construct and test an implementation of the function `torch.sparse.mv` for
   computing sparse matvecs.

   + Link to CUDA libraries like [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html)
     or [CUSP](https://developer.nvidia.com/cusp) if possible.
   + This is explicitly requested in issue
     [\#21782](https://github.com/pytorch/pytorch/issues/21782).
   + Include test cases with sparse-matrix/dense-vector matvecs.
   + Include test cases with sparse-matrix/sparse-vector matvecs.
   + Verify against benchmarks to ensure improved performance when using GPUs.
   + Include test cases to ensure that gradients are computed as expected as
     with `backward`.

5. Construct and test an implementation of the function `torch.sparse.mm` for
   computing sparse matrix products.

   + Link to CUDA libraries like [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html)
     or [CUSP](https://developer.nvidia.com/cusp) if possible.
   + Include test cases with sparse-matrix/dense-matrix products.
   + Include test cases with sparse-matrix/sparse-matrix products.
   + Verify against benchmarks to ensure improved performance when using GPUs.
   + Include test cases to ensure that gradients are computed as expected as
     with `backward`.
   
6. Modify and test implementations of *all* functions using matrix products in
   the `torch.sparse` and `torch` namespaces to ensure all work correctly.
   
   + Include test cases to mix sparse & dense inputs.
   + Check for *all* functions in the API: `torch.addmv`, `torch.addmv_`,
     `torch.bmm`, `torch.addbmm`, etc.

7. Construct and test support for infix matrix multiplication `@` operator (as
   specified by [PEP 465](https://www.python.org/dev/peps/pep-0465/).
   
   + Modify implementations of related functions (e.g., `torch.mv`, `torch.mm`,
     `torch.matmul`, `torch.sparse.mm`, etc.) as required.
   + Consult NumPy implementation to see how this is implemented there.
   + Ensure all combinations of input dimensions are handled correctly as
     specified in [PEP 465](https://www.python.org/dev/peps/pep-0465/).
   + Include test cases with mixtures of sparse & dense inputs for robustness.

8. Investigate existing PyTorch C++ codebase to ensure that all logic involving
   sparse tensors is sound with the modifications imposed.

   + As an example, consider the code in
     [`accumulate_grad.h`](https://github.com/pytorch/pytorch/blob/5bbcddae3bfb7abaeaada29369c6ed5964d390fe/torch/csrc/autograd/functions/accumulate_grad.h#L61-L68)
     that explicitly makes reference to the `_values` and `_indices` attributes
     of sparse COO tensors. It would be necessary to include a test for COO vs.
     CSR sparse tensors and to use an appropriate algorithm for the CSR case.
   + In principle, such special cases could be caught by unit tests using
     sparse tensors as inputs.

9. Deprecate redundant portions of the `torch` and `torch.sparse` namespace to
   favour the infix operator for matrix multiplication over, say, `torch.matmul`
   (or `torch.mm`, `torch.mv`, etc.).

   + This change is optional; it depends on the volume of PyTorch code that
     exists in the wild that would break as a result of these deprecations.

Item (3) in the steps outlined above consists of constructing a table as follows
for each combination of inputs to a given operator:


|`__mul__` | dense           | COO            | CSR            |
|----------|-----------------|----------------|----------------|
| dense    | supported       | supported      | not implemented|
| COO      | supported       | supported      | not implemented|
| CSR      | not implemented | not implemented| not implemented|

The goal is to get all the "not implemented" entries to "supported" for
each combination of inputs. Ideally, there would be comprehensive unit
tests for each case to ensure correct behaviour.


The principle GitHub issues related to this work are:

-   [\#14617 error sparse matrix multiplication when calling 
coalesce](https://github.com/pytorch/pytorch/issues/14617)
-   [\#6171 Compute csr representation on torch.sparse.Tensor.coalesce for 
faster sparse matrix multiplication](https://github.com/pytorch/pytorch/issues/)
-   [\#6225 Caching the CSR representation and bugfix for 
\#6219](https://github.com/pytorch/pytorch/issues/6225)
-   [\#16187 Sparse matrix multiplication is too 
slow](https://github.com/pytorch/pytorch/issues/16187)

Some work is already in progress on some related issues that is likely relevant:

-   [\#33430 Bmm sparse dense](https://github.com/pytorch/pytorch/issues/33430)
-   [\#5672 \[feature request\] sparse x dense 
bmm](https://github.com/pytorch/pytorch/issues/5672)
-   [\#21782 add mv operator to 
SparseTensor](https://github.com/pytorch/pytorch/issues/21782)
-   [\#21266 SparseTensor multiplication with 1D vector 
enhancement](https://github.com/pytorch/pytorch/issues/21266)

## Alternatives

Some alternative solutions are possible that use other sparse array 
representations (e.g.,
[*CSC* (*compressed sparse column*)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)),
[*DOK* (*dictionary of keys*)](https://en.wikipedia.org/wiki/Sparse_matrix#Dictionary_of_keys_(DOK)), 
and blocked versions of other formats). The principal reason these are not 
seriously considered in this proposal is that, apart from the known enhancements 
attainable using CSR (or CSC) format, it is not obvious that DOK or other 
representations yield practical advantages. Again, supporting other storage 
formats requires reimplementing every method associated with the sparse tensor 
classes. At present, there is a clear demand for faster sparse matrix products
that use CSR format. Other formats may outperform CSR in certain contexts and
can, in principle, be added at a later time when it makes sense to do so.

## Discussion

-   [\#9674 The state of sparse Tensors](https://github.com/pytorch/pytorch/issues/9674)
-   [\#14617 error sparse matrix multiplication when calling coalesce](https://github.com/pytorch/pytorch/issues/14617)
-   [\#6171 Compute csr representation on torch.sparse.Tensor.coalesce for 
faster sparse matrix multiplication](https://github.com/pytorch/pytorch/issues/)
-   [\#6225 Caching the CSR representation and bugfix for \#6219](https://github.com/pytorch/pytorch/issues/6225)
-   [\#16187 Sparse matrix multiplication is too slow](https://github.com/pytorch/pytorch/issues/16187)
-   [\#33430 Bmm sparse dense](https://github.com/pytorch/pytorch/issues/33430)
-   [\#5672 \[feature request\] sparse x dense bmm](https://github.com/pytorch/pytorch/issues/5672)
-   [\#21782 add mv operator to SparseTensor](https://github.com/pytorch/pytorch/issues/21782)
-   [\#21266 SparseTensor multiplication with 1D vector enhancement](https://github.com/pytorch/pytorch/issues/21266)
-   [\#20988 How to do matrix multiplication between two 2D sparse directly
and quickly](https://github.com/pytorch/pytorch/issues/20988)
-   [\#5262 \[feature request\] efficient matmul(sparse, sparse) -> sparse](https://github.com/pytorch/pytorch/issues/5262)
-   [\#3158 \[Feature Request\] Sparse Tensor Multiplication](https://github.com/pytorch/pytorch/issues/3158)
-   [\#17825 Sparse MM not working right](https://github.com/pytorch/pytorch/issues/17825)
-   [\#19546 torch.sparse.mm is not stable because it would suddenly cause nan](https://github.com/pytorch/pytorch/issues/19546)
-   [\#26234 Does PyTorch 1.2 support autograd for multiplication of a sparse 
tensor by a dense tensor?](https://github.com/pytorch/pytorch/issues/26234)
-   [\#14489 Batch matmul with sparse matrix, dense vector](https://github.com/pytorch/pytorch/issues/14489)
-   [\#12308 nn.functional.linear() for sparse tensor](https://github.com/pytorch/pytorch/issues/12308)
-   [\#29026 RuntimeError: !t.is\_cuda() INTERNAL ASSERT FAILED at 
/pytorch/aten/src/ATen/native/sparse/SparseTensorMath.cpp:591](https://github.com/pytorch/pytorch/issues/29026)
-   [\#2886 Sparse functions (non-methods) missing from master](https://github.com/pytorch/pytorch/issues/2886)
-   [\#2389 sparse.mm(S, D) with autograd](https://github.com/pytorch/pytorch/issues/2389)
-   [\#12498 Support calculating grad for dense in sparse @ dense](https://github.com/pytorch/pytorch/issues/12498)

## References and Footnotes

-   [PEP 465: A dedicated infix operator for matrix multiplication](https://www.python.org/dev/peps/pep-0465/)
-   [Sparse matrix (Wikipedia)](https://en.wikipedia.org/wiki/Sparse_matrix)
-   [Sparse Basic Linear Algebra Subprograms (BLAS) Library](https://math.nist.gov/spblas/)
-   [Why is cuSPARSE so much slower than cuBLAS for sparse matrix multiplication?](https://stackoverflow.com/questions/30118673/why-cusparse-is-much-slower-than-cublas-for-sparse-matrix-multiplication)
-   [`torch.sparse` documentation](https://pytorch.org/docs/stable/sparse.html)
-   [`scipy.sparse` 
documentation](https://docs.scipy.org/doc/scipy/reference/sparse.html)
-   [`pydata/sparse` project](https://github.com/pydata/sparse)
-   [Sparse Matrix-Vector Multiplication with CUDA](https://medium.com/analytics-vidhya/sparse-matrix-vector-multiplication-with-cuda-42d191878e8f)
-   [A Performance Comparison of Linear Algebra Libraries for Sparse Matrix-Vector Product](https://pdfs.semanticscholar.org/5478/de620de99b1d4e41599a8950c7d3d9ff07b5.pdf)
-   [Efficient MATLAB Computations with Sparse and Factored Tensors](https://epubs.siam.org/doi/10.1137/060676489)
-   [Efficient data compression methods for multidimensional sparse array operations based on the EKMR scheme](https://ieeexplore.ieee.org/abstract/document/1252859)
-   [Efficient storage scheme for n-dimensional sparse array: GCRS/GCCS](https://ieeexplore.ieee.org/document/7237032)
-   [Tensor-matrix products with a compressed sparse tensor](https://dl.acm.org/doi/10.1145/2833179.2833183)
-   [The tensor algebra compiler](https://dl.acm.org/doi/10.1145/3133901)
-   [Sparse Tensor Algebra Optimizations with 
Workspaces](https://arxiv.org/abs/1802.10574)
-   [Format Abstraction for Sparse Tensor Algebra 
Compilers](https://arxiv.org/abs/1804.10112)
-   [A Unified Iteration Space Transformation Framework for Sparse and Dense 
Tensor Algebra](https://arxiv.org/abs/2001.00532)
-   [Automatic Generation of Efficient Sparse Tensor Format Conversion 
Routines](https://arxiv.org/abs/2001.02609)

## Copyright

This document has been placed in the public domain. Licensed under the [Open 
Publication License](https://www.opencontent.org/openpub/).
