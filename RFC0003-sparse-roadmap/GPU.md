<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-generate-toc again -->
**Table of Contents**

- [Introduction](#introduction)
- [Background of current PyTorch sparse matrix CUDA support](#background-of-current-pytorch-sparse-matrix-cuda-support)
    - [PyTorch CUDA sparse data structures](#pytorch-cuda-sparse-data-structures)
    - [Usage of cuSPARSE in pytorch](#usage-of-cusparse-in-pytorch)
- [Sparse tensor support for GPUs](#sparse-tensor-support-for-gpus)
    - [Overview of cuSPARSE](#overview-of-cusparse)
    - [Overview of MAGMA](#overview-of-magma)
    - [Overview of Ginkgo](#overview-of-ginkgo)
    - [Overview of ParTI](#overview-of-parti)
- [Future directions for pytorch CUDA implementation](#future-directions-for-pytorch-cuda-implementation)

<!-- markdown-toc end -->

# Introduction

Although it is clear that sparse tensors in pytorch are an
[important requirement](https://github.com/pytorch/rfcs/blob/b2d02512bb69648fc61013829205eb6dfea6a714/RFC-0004-pyTorch-sparse-matmul-roadmap.md#motivation-and-scope),
its implementation and performance on accelerators is still unclear. Most widely used
libraries like cuSPARSE and MAGMA are optimized for sparse matrix operations and do
not support rank-n tensors out of the box. This document aims to provide the reader
with a comprehensive understanding of the state of sparse tensor computation and its
implemenation specifics on GPUs. We assume that the reader is familiar with the basics
of sparse tensors such as storage formats like COO and CSR. If not, please first read
the [Dhavide's proposal](https://github.com/pytorch/rfcs/pull/4) for a background
and [Pearu's proposal](https://github.com/Quansight-Labs/rfcs/tree/pearu/rfc0005/RFC0003-sparse-roadmap)
for further details.

# Background of current PyTorch sparse matrix CUDA support

COO is also the [only supported sparse format](https://github.com/Quansight-Labs/rfcs/tree/pearu/rfc0005/RFC0003-sparse-roadmap#pytorch-implementation-of-coo-sparse-format)
for both the CPU and CUDA backend. CuSPARSE is the
only 3rd party library used for sparse operations. The element-wise operations
are implemented using
[hand-implemented](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu#L343) kernels.

## PyTorch CUDA sparse data structures

Both the CUDA and CPU variants of COO constructors utilize the same constructor
(`sparse_coo_tensor` in SparseTensor.cpp). The main data of the COO tensor
is stored in the `indices` and `values` objects.
An example can be seen in
[sparseElementwiseKernelScalar()](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/sparse/cuda/SparseCUDAApplyUtils.cuh#L76)
function where the `TensorInfo` objects representing `indices` and `values`
are passed into the CUDA kernel. Thus it can be seen that most of the functions are
heavily tailored for the COO format and new functions must be written for new
formats.

When not working at the fine-grained level of the CUDA kernel, an `indices` tensor
of type `LongTensor` and a `values` tensor of type `Tensor` can be accessed for
accessing the values and indices of COO tensors respectively. These can then
be directly interfaced with most libraries utilizing the COO tensor format
since these libraries directly accept data pointers and their length to
work with such data.

Since the sparse matrix multiplication has been shown to be most optimal on a
CSR format sparse matrix, PyTorch converts COO tensors into CSR tensors before
calling such routines.

## Usage of cuSPARSE in pytorch

The only way the CSR format differs from the COO is its storage of the row indicies. 
Since pytorch stores sparse matrices in COO by default, the pytorch implementation
of sparse matrix-matrix multiplication
[converts the row indices of the COO tensor](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu#L59)
into CSR and
[then performs the multiplication](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu#L96).
Although the cuSPARSE interface
can perform SpMM for COO tensors by specifying the appropriate macro denoting the layout,
pytorch still performs conversion to CSR before calling the SpMM function, probably for
performance reasons. Most of the cuSPARSE using functions can be found in
[SparseCUDATensorMath.cu](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/sparse/cuda/SparseCUDATensorMath.cu)
and [SparseCUDABlas.cu](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/sparse/cuda/SparseCUDABlas.cu).

# Sparse tensor support for GPUs

In this section we will have a glimpse at 3rd party libraries that support sparse
matrix or sparse tensor operations. We refer to a 'sparse matrix' as a
rank-2 tensor and 'sparse tensor' as a rank-n tensor. Sparse matrix and tensor
implementations are being actively developed in libraries
such as cuSPARSE, CUSP, MAGMA, Gingko and [ParTI!](https://github.com/hpcgarage/ParTI).

Most libraries support sparse matrix operations with ParTI! being the only library that
has been built specifically for sparse tensor operations, with extensive support for
rank-n tensors and various operations
highly optimized for GPUs. Many of the routines listed here are still in a research phase
and will probably get much better and faster over time.

### Overview of cuSPARSE

[CuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) supports operations
between sparse vectors and dense vectors, and sparse matrices
and dense vectors or a set of dense vectors. It supports dense, csr, coo and csc sparse
matrix formats and also has routines for conversion between these formats. It is
currently in use in PyTorch for sparse operations.

### Overview of MAGMA

The [MAGMA](https://icl.cs.utk.edu/magma/) library has grown out of a research project at ICL at the
University of Tennessee and is extensively used within the CUDA dense
routines of pytorch. Magma has implementations for many linear algebra routines for various sparse
types. It can be used on both multi-core CPUs and multi-GPU systems,
although it has mostly gained attention for the GPU use cases.

It supports sparse matrices of type `float` , `double`, `float complex` and `double complex`.
Sparse matrices of type `double` can be implemented using the `magma_d_matrix`
type (type definition can be found
[here](https://bitbucket.org/icl/magma/src/master/sparse/include/magmasparse_types.h)).
Magma supports sparse layouts of formats CSR, ELL, SELL-P and CSRS.

The `SpMM` operation can be highly optimized in Magma with use of the SELL-P format, which
can be seen in [Anzt2015](https://www.icl.utk.edu/files/publications/2014/icl-utk-771-2014.pdf).
The SELL-P format is an extension of CSR which pads the rows of a matrix with 0s for increasing
the resource utilization (and thereby speed) at the cost of more memory.

### Overview of Ginkgo

[Ginkgo](https://ginkgo-project.github.io/) can be thought of as an outcrop of
MAGMA specialized for sparse operations. Unlike MAGMA, research groups working
on Gingko specialize in sparse matrix computations, and therefore this
library features routines that outperform both cuSPARSE and MAGMA. For example,
[Anzt2020](https://dl.acm.org/doi/pdf/10.1145/3380930) shows how an optimized
version of MAGMA's SELL-P can further speed up SpMV.

### Overview of ParTI

[ParTI!](https://github.com/hpcgarage/ParTI) is the only library in the list that
is specifically built for sparse tensors. Several publications such as
[this one](http://fruitfly1026.github.io/static/files/sc16-ia3.pdf) show that
ParTI has support for various operations such as SpTTM (Sparse Tensor Times Matrix)
and MTTKRP (Matrcised Tensor Times Khatri Rao Product).

It supports input in the COO and [HiCOO](http://fruitfly1026.github.io/static/files/sc18-li.pdf)
(COO optimized for multi-dim data) formats.

# Future directions for pytorch CUDA implementation

The CSR format is in general faster than COO for matrix multiplication, however,
SELL-P (slightly modified CSR) has been shown to be faster than CSR for GPUs. For
rank-2 tensors, the GCSR format proposed by
[Shaikh2015](https://www.researchgate.net/publication/312167966_Efficient_storage_scheme_for_n-dimensional_sparse_array_GCRSGCCS)
can be converted into other formats required by these libraries fairly easily.
Which format to use can be determined by empirical testing and checking for
the fastest implementation of a given routine.

