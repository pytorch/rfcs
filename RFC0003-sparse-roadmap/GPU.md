<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-generate-toc again -->
**Table of Contents**

- [Introduction](#introduction)
- [Sparse tensor support for GPUs](#sparse-tensor-support-for-gpus)
    - [Library overview](#library-overview)
        - [Overview of cuSPARSE](#overview-of-cusparse)
        - [Overview of MAGMA](#overview-of-magma)
        - [Overview of Ginkgo](#overview-of-ginkgo)
        - [Overview of ParTI](#overview-of-parti)
    - [Matrix-matrix multiplication performance](#matrix-matrix-multiplication-performance)
    - [Support for stride-n tensor operations](#support-for-stride-n-tensor-operations)
- [Background of PyTorch sparse CUDA support](#background-of-pytorch-sparse-cuda-support)
    - [Usage of cuSPARSE in pytorch](#usage-of-cusparse-in-pytorch)
    - [Non-BLAS math operations on sparse tensors](#non-blas-math-operations-on-sparse-tensors)
- [Future directions for CUDA implementation](#future-directions-for-cuda-implementation)

<!-- markdown-toc end -->

# Introduction

Although it is clear that sparse tensors in pytorch are an [important requirement](https://github.com/pytorch/rfcs/blob/b2d02512bb69648fc61013829205eb6dfea6a714/RFC-0004-pyTorch-sparse-matmul-roadmap.md#motivation-and-scope),
its implementation and performance on accelerators is still unclear. Most widely used
libraries like cuSPARSE and MAGMA are optimized for sparse matrix operations and do
not support rank-n tensors out of the box. This document aims to provide the reader
with a comprehensive understanding of the state of sparse tensor computation and its
implemenation specifics on GPUs. The intention is to make informed decisions about the
future direction of sparse tensor computation for PyTorch.

# Sparse tensor support for GPUs

Sparse tensor implementations are being actively developed in libraries such as cuSPARSE, CUSP,
MAGMA, Gingko, [ParTI](https://github.com/hpcgarage/ParTI).
This section will show the sparse
matrix and tensor support that each library possesses and how it fares with the others.

Although most of the libraries support sparse matrix operations (rank-2 tensors), few
have direct support for tensor operations (rank-n).

## Library overview

### Overview of cuSPARSE

[CuSPARSE]() supports operations between sparse vectors and dense vectors, and sparse matrices
and dense vectors or a set of dense vectors. It supports dense, csr, coo and csc sparse matrix formats
and conversion between them too.

Overview of matrix-vector and SpMM routines.

Support for rank-n sparse arrays.

### Overview of MAGMA

The [MAGMA](URL) library has grown out of a research project at ICL at the
University of Tennessee and is extensively used within the CUDA dense
routines of pytorch. Magma has implementations for many linear algebra routines for various sparse
types. It can be used on both multi-core CPUs and multi-GPU systems,
although it has mostly gained attention for the GPU use cases.

It supports sparse matrices of type `float` , `double`, `float complex` and `double complex`.
Sparse matrices of type `double` can be implemented using the `magma_d_matrix` type (type definition
can be found [here](https://bitbucket.org/icl/magma/src/master/sparse/include/magmasparse_types.h)).
Magma supports sparse layouts of formats CSR, ELL, SELL-P and CSRS.

The `SpMM` operation can be highly optimized in Magma with use of the SELL-P format, which
can be seen in [Anzt2015](https://www.icl.utk.edu/files/publications/2014/icl-utk-771-2014.pdf).
The SELL-P format is an extension of CSR which pads the rows of a matrix with 0s for increasing
the data utilization (and thereby speed) at the cost of more memory.

### Overview of Ginkgo

Ginkgo 

### Overview of ParTI

ParTI! is a library built specifically for tensor operations.

## Matrix-matrix multiplication performance

In this section we will see the performance of matrix-matrix multiplication for various
libraries using the same data in different storage formats.

## Support for stride-n tensor operations

Although we have seen operations on 2 dimensional tensors so far, efficient operations
on tensors requires extra functionality like optimization of execution over multiple
dimensions and support for modified sparse tensor formats like the GCSR format proposed
in [Shaikh2015](https://www.researchgate.net/publication/312167966_Efficient_storage_scheme_for_n-dimensional_sparse_array_GCRSGCCS).

# Background of PyTorch sparse CUDA support

## Usage of cuSPARSE in pytorch

The only way the CSR format differs from the COO is its storage of the row indicies. 
Since pytorch stores sparse matrices in COO by default, the pytorch implementation
of sparse matrix-matrix multiplication [converts the row indices of the COO tensor
into CSR]() and [then performs the multiplication](). Although the cuSPARSE interface
can perform SpMM for COO tensors by specifying the appropriate macro denoting the layout,
pytorch still performs conversion to CSR before calling the SpMM function, probably for
performance reasons.

## Non-BLAS math operations on sparse tensors

Operations that do not qualify as BLAS operations (matrix-matrix multiplication for example)
are termed as non-BLAS math operations.

# Future directions for CUDA implementation

Although the GPU-enabled sparse linear algebra libaries cuSPARSE, Ginkgo
and MAGMA as noted above have fast optimized sparse routines that almost
reach peak performance, none of them support generalized rank-n tensors.
All the functions are optimized for rank-2 tensors (sparse matrices) and
support a variety of sparse matrix formats, some of them even inventing
their own formats for speeding up certain operations. None of the above
support stride-N access for sparse tensors.




Things to find out:
* CSR implementations in cuSPARSE, MAGMA and Gingko.

Publications on fast CUDA implementations:

* Anzt2020 - https://dl.acm.org/doi/10.1145/3380930
* High performance tensor contractions on GPUs - https://reader.elsevier.com/reader/sd/pii/S1877050916306536?token=7AE871C35B8EA2763680BBCD18AE8154773F5D25AFDA75CED2315BE7D8BEAFF446E541D0CDAFDA14DE1ED5C0AB288A78

Libraries:

* cuSPARSE
* Gingko - https://ginkgo-project.github.io/
* MAGMA 
