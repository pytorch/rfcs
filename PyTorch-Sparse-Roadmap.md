
PyTorch.sparse Roadmap
======================

+------------+-----------------+
| Author     | Dhavide Aruliah |
+------------+-----------------+
| Status     | Draft           |
+------------+-----------------+
| Type       | Process         |
+------------+-----------------+
| Created    | 2020-03-24      |
+------------+-----------------+
| Resolution | TBD             |
+------------+-----------------+




Abstract
--------

At present, there are more than 60 unresolved GitHub issues involving `torch.sparse` to address. A handful have been flagged as high priority but, by and large, there hasn't been a lot of motion on those issues (some dating back two years). This proposal is largely aimed at articulating an overarching strategy for addressing a number `torch.sparse` issues related to the implementation of *matrix multiplication* inolving sparse PyTorch tensors.


Motivation and Scope
--------------------

Almost one third of the `torch.sparse` issues relate to the implementation of matrix multiplication with sparse PyTorch tensors (i.e., in the case of tensors of atmost rank 2). Most of these issues are concerned with the speed of matrix multiplication involving a sparse matrix (e.g., `#5262 efficient matmul(sparse, sparse) -> sparse <https://github.com/pytorch/pytorch/issues/5262>`_). Note that some issues that refer to "multiplication" explicitly are unrelated in that they require modification of *elementwise* products & broadcasting with sparse tensors (e.g., `#2909 Feature request: mul(Sparse, Dense) -> [Dense] <https://github.com/pytorch/pytorch/issues/2909>`_). There are also additional issues relating to batching sparse matrix products (e.g., `#33430 Bmm sparse dense <https://github.com/pytorch/pytorch/issues/33430>`_) and computing gradients (e.g., `#12498 Support calculating grad for dense in sparse @ dense <https://github.com/pytorch/pytorch/issues/12498>`_) but these can likely be addressed *after* the speed-up issues.


Usage and Impact
----------------

It seems that examining and optimizing sparse matrix multiplication in PyTorch is valuable to a broad spectrum of users (as suggested by the repetitions of this theme in issues raised). Addressing sparse matrix multiplication requires thinking about internal representations (i.e., CSR vs. COO) as well as ensuring that every way of computing matrix products is tested & updated appropriately (e.g., `torch.matmul`, `torch.sparse.mm`, `torch.sparse.FloatTensor.mm`, etc.). Some care is required in addressing how products are handled between dense and sparse tensors as opposed to sparse tensors only. There is also the issue of ensuring that any changes made are compatible with `autograd` for computation of gradients through the `backward` method.


Backward compatibility
----------------------

Ideally, there will be no external API changes that would break backward compatibility. The proposed alterations largely consist of internal representations of sparse tensors (likely caching a CSR representation) to enable faster algorithms so existing user code should not break.


Related Work
------------

The most relevant prior work is in the `scipy.sparse <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_ module as well as the `pydata/sparse <https://github.com/pydata/sparse>`_ project. As a stretch goal, it may make sense to have general rank `n` PyTorch tensors being supported in most of the same ways as `pydata/sparse` and to have tensors of rank 2 or less supported in most of the same ways as `scipy.sparse` (where it makes sense to do so). Again, there are additional complications in providing GPU support as well as ensuring `autograd` support is included.

The internal format for representing PyTorch sparse tensors internally is *COO* (*coordinate*) format (i.e., using a rank 1 FloatTensor of length `nnz` for the non-zero values and a `d x nnz` IntTensor to store the positions of those entries in a sparse tensor of rank `d`. Many of the optimized sparse linear algebra libraries for tensors of rank 2 (i.e., matrices) assume *CSR* (*compressed sparse row*) or *CSC* (*compressed sparse column*) formats. There are generalizations of these formats for sparse tensors of arbitrary rank $n$ as described in various articles:

+ `Automatic Generation of Efficient Sparse Tensor Format Conversion Routines <https://arxiv.org/abs/2001.02609>`_
+ `The tensor algebra compiler <https://dl.acm.org/doi/10.1145/3133901>`_
+ `Sparse Tensor Algebra Optimizations with Workspaces <https://arxiv.org/abs/1802.10574>`_
+ `Format Abstraction for Sparse Tensor Algebra Compilers <https://arxiv.org/abs/1804.10112>`_
+ `A Unified Iteration Space Transformation Framework for Sparse and Dense Tensor Algebra <https://arxiv.org/abs/2001.00532>`_

Implementation
--------------

As mentioned in issue `#9674 The state of sparse Tensors <https://github.com/pytorch/pytorch/issues/9674>`_, many sparse libraries use CSR format because it's really efficient for operations like matrix multiplication. One possible way to achieve this is to introduce an internal caching mechanism for CSR representations for tensors of rank 2 (or where `_sparseDims==2` which is likely to be useful, e.g., for batches of sparse matrices).

1. Construct a prototype mechanism for internal caching of CSR representation for PyTorch tensors of rank 2 (or where `_sparseDims==2`; see `#14617 <https://github.com/pytorch/pytorch/issues/14617>`_, `#6171 <https://github.com/pytorch/pytorch/issues/6171>`_, & `#6225 <https://github.com/pytorch/pytorch/issues/6225>`_).

2. Figure out how to connect this to `coalesce`, in particular, how/when to update the internal cached CSR representation efficiently when values are updated  (see, e.g., comments in `#16187 Sparse matrix multiplication is too slow <https://github.com/pytorch/pytorch/issues/16187>`_).

3. Implement sparse-dense matrix-matrix products that capitalize on CSR representation linking to CUDA libraries where possible. Prepare tests to ensure that `torch.matmul` and torch.sparse.mm` are consistent in computed results.

4. Execute benchmark tests comparing existing implementation of matrix products

5. Ensure that `torch.sparse.FloatTensor.mm`, `torch.sparse.IntTensor.mm`, and similar sparse tensor methods or functions associated with matrix products work correctly.

The principle issues related to this work are:

+ `#14617 error sparse matrix multiplication when calling coalesce <https://github.com/pytorch/pytorch/issues/14617>`_
+ `#6171 Compute csr representation on torch.sparse.Tensor.coalesce for faster sparse matrix multiplication <https://github.com/pytorch/pytorch/issues/>`_
+ `#6225 Caching the CSR representation and bugfix for #6219 <https://github.com/pytorch/pytorch/issues/6225>`_
+ `#16187 Sparse matrix multiplication is too slow <https://github.com/pytorch/pytorch/issues/16187>`_

Some work is already in progress on some related issues that should also be tied in:

+ `#33430 Bmm sparse dense <https://github.com/pytorch/pytorch/issues/33430>`_
+ `#5672 [feature request] sparse x dense bmm <https://github.com/pytorch/pytorch/issues/5672>`_
+ `#21782 add mv operator to SparseTensor <https://github.com/pytorch/pytorch/issues/21782>`_
+ `#21266 SparseTensor multiplication with 1D vector enhancement <https://github.com/pytorch/pytorch/issues/21266>`_


Alternatives
------------

Some alternative solutions are possible that use other sparse array representations (e.g., ). The principal reason these are not seriously considered in this proposal is that those would require reimplementing every method associated with the general rank `n` sparse tensor classes. Although some theoretical work describing these approaches, it is not obvious that the development & maintenance effort is justified. By contrast, there is a clear demand for faster sparse matrix products.

Discussion
----------

+ `#9674 The state of sparse Tensors <https://github.com/pytorch/pytorch/issues/9674>`_
+ `#14617 error sparse matrix multiplication when calling coalesce <https://github.com/pytorch/pytorch/issues/14617>`_
+ `#6171 Compute csr representation on torch.sparse.Tensor.coalesce for faster sparse matrix multiplication <https://github.com/pytorch/pytorch/issues/>`_
+ `#6225 Caching the CSR representation and bugfix for #6219 <https://github.com/pytorch/pytorch/issues/6225>`_
+ `#16187 Sparse matrix multiplication is too slow <https://github.com/pytorch/pytorch/issues/16187>`_
+ `#33430 Bmm sparse dense <https://github.com/pytorch/pytorch/issues/33430>`_
+ `#5672 [feature request] sparse x dense bmm <https://github.com/pytorch/pytorch/issues/5672>`_
+ `#21782 add mv operator to SparseTensor <https://github.com/pytorch/pytorch/issues/21782>`_
+ `#21266 SparseTensor multiplication with 1D vector enhancement <https://github.com/pytorch/pytorch/issues/21266>`_
+ `#16187 Sparse matrix multiplication is too slow <https://github.com/pytorch/pytorch/issues/16187>`_
+ `#20988 How to do matrix multiplication between two 2D sparse directly and quickly <https://github.com/pytorch/pytorch/issues/20988>`_
+ `#5262 [feature request] efficient matmul(sparse, sparse) -> sparse <https://github.com/pytorch/pytorch/issues/5262>`_
+ `#3158 [Feature Request] Sparse Tensor Multiplication <https://github.com/pytorch/pytorch/issues/3158>`_
+ `#17825 Sparse MM not working right <https://github.com/pytorch/pytorch/issues/17825>`_
+ `#19546 torch.sparse.mm is not stable because it would suddenly cause nan <https://github.com/pytorch/pytorch/issues/19546>`_
+ `#26234 Does PyTorch 1.2 support autograd for multiplication of a sparse tensor by a dense tensor? <https://github.com/pytorch/pytorch/issues/26234>`_
+ `#14489 Batch matmul with sparse matrix, dense vector <https://github.com/pytorch/pytorch/issues/14489>`_
+ `#12308 nn.functional.linear() for sparse tensor <https://github.com/pytorch/pytorch/issues/12308>`_
+ `#29026 RuntimeError: !t.is_cuda() INTERNAL ASSERT FAILED at /pytorch/aten/src/ATen/native/sparse/SparseTensorMath.cpp:591 <https://github.com/pytorch/pytorch/issues/29026>`_
+ `#2886 Sparse functions (non-methods) missing from master <https://github.com/pytorch/pytorch/issues/2886>`_
+ `#2389 sparse.mm(S, D) with autograd <https://github.com/pytorch/pytorch/issues/2389>`_
+ `#12498 Support calculating grad for dense in sparse @ dense <https://github.com/pytorch/pytorch/issues/12498>`_


References and Footnotes
------------------------

+ `torch.sparse documentation <https://pytorch.org/docs/stable/sparse.html>`_
+ `scipy.sparse documentation <https://docs.scipy.org/doc/scipy/reference/sparse.html>`_
+ `The tensor algebra compiler <https://dl.acm.org/doi/10.1145/3133901>`_
+ `pydata/sparse <https://github.com/pydata/sparse>`_

+ `Sparse Tensor Algebra Optimizations with Workspaces <https://arxiv.org/abs/1802.10574>`_
+ `Format Abstraction for Sparse Tensor Algebra Compilers <https://arxiv.org/abs/1804.10112>`_
+ `A Unified Iteration Space Transformation Framework for Sparse and Dense Tensor Algebra <https://arxiv.org/abs/2001.00532>`_
+ `Automatic Generation of Efficient Sparse Tensor Format Conversion Routines <https://arxiv.org/abs/2001.02609>`_


Copyright
---------

This document has been placed in the public domain. Licensed under the `Open Publication License <https://www.opencontent.org/openpub/>`_.
