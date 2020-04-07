# Roadmap for `torch.sparse` Matrix Product API

|            |                 |
| ---------- | --------------- |
| Author     | Dhavide Aruliah |
| Status     | Draft           |
| Type       | Process         |
| Created    | 2020-04-01      |
| Resolution | TBD             |


## Abstract

At the time of writing, the 
[`torch.sparse`](https://pytorch.org/docs/stable/sparse.html) module API is 
considered experimental according to the [PyTorch 
documentation](https://pytorch.org). There are more than 60 unresolved GitHub 
issues involving `torch.sparse` to address&mdash;some dating back to 
2018&mdash;and a handful of which have been flagged as high priority. At least 
one high priority issue deals with *matrix multiplication* of sparse PyTorch 
matrices.

This proposal summarizes an overarching strategy for addressing a number 
`torch.sparse` issues related to the *speed of sparse matrix multiplication* in 
PyTorch. There are, of course, many other `torch.sparse` issues flagged on 
GitHub that point to other desired features & API design considerations; these 
particular matrix multiplication issues are a useful starting point for further 
API improvements & additional features for this module.

## Motivation and Scope

Almost one third of the current `torch.sparse` issues relate to the speed of 
matrix multiplication with sparse matrices (i.e. sparse tensors of rank 2) in 
PyTorch (e.g., issue [\#5262](https://github.com/pytorch/pytorch/issues/5262)). 
There are additional issues relating to batching sparse matrix products (e.g., 
issue [\#33430](https://github.com/pytorch/pytorch/issues/33430)) and computing 
gradients (e.g., issue 
[\#12498](https://github.com/pytorch/pytorch/issues/12498)) but these can 
likely be addressed *after* the speed-up issues. Note also that some issues 
that refer to "multiplication" explicitly are unrelated in that they refer to 
*elementwise* products & *broadcasting* with sparse tensors (e.g., issue 
[\#2909](https://github.com/pytorch/pytorch/issues/2909)).

For context, the internal representation for PyTorch sparse tensors uses [*COO* 
(*coordinate*) 
format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) 
(i.e., using a rank 1 `FloatTensor` of length `nnz` for the non-zero values and 
a `nnz x d` `IntTensor` to store the positions of those entries in a sparse 
tensor of rank `d`):

```{python}
>>> i = torch.LongTensor([[0, 1, 1],
                          [2, 0, 2]])
>>> v = torch.FloatTensor([3, 4, 5])
>>> torch.sparse.FloatTensor(i, v, torch.Size([2,3])).to_dense()
 0  0  3
 4  0  5
[torch.FloatTensor of size 2x3]
```

One unusual feature of the PyTorch COO representation of tensors is that it 
permits a *hybrid* of sparse and dense dimensions. That is, the leading
dimensions can be sparse and the remaining dimensions are dense; the methods
`sparse_dim` and `dense_dim` reveal those dimensions explicitly. Another way
to think of this is as a sparse tensor with dense tensors as entries.

It is important to observe that the `torch.sparse` API currently supports 
*uncoalesced* sparse tensors (i.e., sparse tensors in COO format with duplicate 
coordinates in the indices). Allowing uncoalesced sparse tensors improves the 
efficiency of certain operations (notably addition of sparse tensors and 
insertion of new nonzero elements into a sparse tensor). Values associated with 
repeated indices in an uncoalesced sparse tensor are added together to yield 
the final value. Requiring coalesced COO tensors instead means that, when 
adding sparse tensors, the indices must be sorted first and values updated (as 
opposed to simply concatenating the corresponding lists of indices & values).

Issue [\#16187](https://github.com/pytorch/pytorch/issues/16187) (among others)
draws attention to deficiencies in the speed of PyTorch sparse matrix products
through comparison to timings in SciPy. A possible cause for these concerns is
the COO representation of sparse tensors in `torch.sparse`. A number of sparse
linear algebra libraries&mdash;including SciPy&mdash;permit the use of
[*CSR* (*compressed sparse row*)](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
format to represent sparse matrices. Using CSR format enables efficient
implementations of sparse *matvecs* (*matrix-vector products*) and, in turn,
sparse matrix-matrix products. Lower-level libraries like
[cuSPARSE](https://developer.nvidia.com/cusparse) and
[CUSP](https://developer.nvidia.com/cusp) (both from NVIDIA) enable efficient
sparse matrix computations on CUDA architecture GPUs with support for both CSR
& COO formats. There are
[sparse linear algebra benchmarks](https://pdfs.semanticscholar.org/5478/de620de99b1d4e41599a8950c7d3d9ff07b5.pdf)
that suggest that sparse matvecs can be between 2 and 2.5 times faster when
using CSR format over COO format. With uncoalesced COO format, there are
additional concerns about deterministic computations on GPUs that arise due to
nonuniqueness of the internal representation in both the sequence and contents
of the index and values arrays.

With the considerations above, it makes sense to extend `torch.sparse` to
provide support for storing sparse matrices in CSR format (in the special case
of tensors of rank 2 or with `sparse_dim()==2`). There are two alternatives to
consider:

+ providing user-accessible conversion routines between COO & CSR 
representations for rank 2 tensors (or tensors with `sparse_dim()==2`); and
+ using a cached internal CSR representation (effectively doubling storage but 
not requiring API modification).

The first approach is most obvious; its primary disadvantage is that 
supporting alternative storage formats requires modifying large portions of the 
code obscured by the API (because, e.g., the algorithms for elementwise
arithmetic operations differ for CSR & COO formats). The aforementioned problem
could be avoided the second option above, i.e., caching. However, caching
introduces a storage overhead and additional complications in keeping the cached
representation up-to-date. Assuming that caching introduces worse problems,
we'll consider the first approach only in this proposal.

The main goals of this proposal, then, are:

1. to implement a CSR representation for `torch.sparse` tensors of 
rank 2 or `sparse_dim()==2` with appropriate modifications to the API; and
2. to implement changes to all associated elementwise arithmetic operations
and similar methods that would, by necessity, need to be updated to ensure
compatibility with the new CSR representation.

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
of sparse tensors of rank 2 (or with `sparse_dim()==2`) to enable certain 
sparse linear algebra improvements. This would likely look something like:

+ `torch.sparse_csr2d` (layout)
+ `torch.sparse_csr2d_tensor` (constructor)
+ explicit conversions through `torch.sparse.to_sparse_csr2d` &
  `torch.sparse.to_sparse_coo`

The use of `csr2d` in identifiers emphasizes the fact that CSR format only
applies to tensors of rank 2 (there may be some caveats in operations
combining these sparse CSR tensors with scalars & vectors, i.e., tensors of
rank less than 2).

Unfortunately, adding a CSR representation for sparse tensors does requires
modifying the implementation significantly. In particular, all relevant sparse
tensor operations (addition/subtraction, elementwise multiplication/division,
etc.) need to dispatch different algorithms for the CSR and COO
representations. A suitably robust test suite should be developed to ensure
that the correct behavior is preserved. There is additional work required to
ensure that every way of computing matrix products is tested & updated
appropriately (e.g., `torch.matmul`, `torch.sparse.mm`,
`torch.sparse.FloatTensor.mm`, etc.). Again, this is largely a question of
adding unit tests to robustly verify intended code behavior.

## Related Work

Much prior work on sparse matrices is implemented in the 
[`scipy.sparse`](https://docs.scipy.org/doc/scipy/reference/sparse.html) 
module; the [`pydata/sparse`](https://github.com/pydata/sparse) project extends 
some of those ideas to general sparse arrays of rank `n`. As a stretch goal, it 
may make sense to have general rank `n` PyTorch tensors being supported in most 
of the same ways as `pydata/sparse` and to have sparse tensors of rank 2 (or 
with `sparse_dim()==2`) supported in most of the same ways as `scipy.sparse`. 
There is also prior work on caching CSR representations in PyTorch in PR 
[#6225](https://github.com/pytorch/pytorch/pull/6225). Again, there are 
additional complications in
providing GPU support as well as ensuring compliance with `autograd`.

Among projects tailored to machine learning & deep learning,
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

The basic outline is as follows:

1.  Construct a small suite of benchmark tests to time the existing 
implementation of sparse matrix-vector and matrix-matrix products with matrices 
of fairly large size (i.e., where dense representations are impractical or 
infeasible). See [issue # 
16187](https://github.com/pytorch/pytorch/issues/16187) for representative 
examples of benchmarks.
2.  Construct a prototype implementation of conversion functions/methods 
between COO and CSR formats for PyTorch sparse tensors of rank 2 (or with 
`sparse_dim()==2`). The implementation of these functions in 
[Tensorflow](https://www.tensorflow.org/api_docs/python/tf/raw_ops/SparseMatrixS
parseMatMul) can serve as a useful model. See also comments in issue 
[\#16187](https://github.com/pytorch/pytorch/issues/16187).
3. Implement sparse-dense matrix-matrix products that capitalize on CSR 
representation linking to CUDA libraries like 
[cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) where possible. 
Prepare tests to ensure that `torch.matmul` and `torch.sparse.mm` produce 
consistent results.
4. Ensure that matrix multiplication is updated correctly for `torch.sparse` in 
all relevant classes, e.g., `IntTensor.mm`, `FloatTensor.mm`, and so on. Verify 
that all sparse tensor methods or functions associated with sparse matrix 
products work correctly.
5. Update the `torch.sparse` API for all methods to ensure correct support for 
COO format and CSR format (in the case of tensors of rank 2 or 
`sparse_dim()==2`). This would include, e.g., the `__add__`, `__sub__`, 
`__mul__`, `__div__`, and most other methods.

The principle issues related to this work are:

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
seriously considered in this proposal is that, apart from the known speed-ups 
attainable using CSR (or CSC) format, it is not obvious that DOK or other 
representations yield practical advantages. Again, supporting other storage 
formats requires reimplementing every method associated with the sparse tensor 
classes. Although there is some theoretical work describing these approaches, 
it is not obvious that the development & maintenance effort is justified. By 
contrast, there is a clear demand for faster sparse matrix products that use 
CSR format.

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

-   [Sparse matrix (Wikipedia)](https://en.wikipedia.org/wiki/Sparse_matrix)
-   [`torch.sparse` documentation](https://pytorch.org/docs/stable/sparse.html)
-   [`scipy.sparse` 
documentation](https://docs.scipy.org/doc/scipy/reference/sparse.html)
-   [`pydata/sparse` project](https://github.com/pydata/sparse)
-   [Sparse Matrix-Vector Multiplication with CUDA](https://medium.com/analytics-vidhya/sparse-matrix-vector-multiplication-with-cuda-42d191878e8f)
-   [A Performance Comparison of Linear Algebra Libraries for Sparse Matrix-Vector Product](https://pdfs.semanticscholar.org/5478/de620de99b1d4e41599a8950c7d3d9ff07b5.pdf)
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
