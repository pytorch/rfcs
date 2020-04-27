# Roadmap for PyTorch Sparse Tensors

|            |                 |
| ---------- | --------------- |
| Authors    | Pearu Peterson  |
| Status     | Draft           |
| Type       | Process         |
| Created    | 2020-04-23      |
| Resolution | TBD             |


## Abstract

The aim of this document is to define a direction of PyTorch sparse
tensor features: enable a sparse tensor to be a drop-in replacement of
dense tensors.


## Motivation and Scope

PyTorch implements the Tensor object as the basic structure of storing
data in a multi-dimensional array structure that can be processed on
various computational devices such as GPUs as well as multicore CPUs
efficiently by taking advantage of parallelism that these devices
support. PyTorch has advanged in developing computational tools for
various ML and AI algorithms that can be applied to so-called dense or
strided tensors. However, the developement of tools applicable to
sparse tensors has not passed the experimental stage after two or more
years of development, dispite the fact that a sparse tensor would be a
natural data storage format for many ML&AI applications where the data
characterizes some pairwise relations between study subjects or when
the aim is to analyze only a small set of features that dominate in a
big system, for instance. Choosing a sparse tensor format may lead to
more performant programs due to the efficient uses of memory as well
as computational resources.

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
proposal becomes about (i) providing sparse tensor implementations and
the corresponding low-level computational tools, and (ii) reusing
PyTorch dispatcher system to make sparse tensors as drop-in
replacements to existing dense tensors when used as inputs to
high-level computational tools. A prioritization of features sets is
required in order to make the stable sparse tensors support accessible
for users' programs as soon as possible.


## Mathematical background of arrays

One of the most common data representation in scientific and
engineering computations is an array of items that share the same data
type and size and that can be generalized to any dimensions. The array
items are naturally indexed using tuples of integers that represent
the coordinates in a multidimensional discrete space of data values.
Mathematically, arrays can be interpreted as vectors, matrices, or
tensors that form algebras with various algebra operations as well as
functions on algebra elements.

When representing data as an array in computer memory, a array storage
format must be selected. The simplest and most general array storage
format is the strided array format where the data values are stored
continuously in memory so that the storage location any array item can
be easily determined from its index tuple and therefore efficient
concurrent algorithms can be designed that process data in the most
efficient way.  However, when data poses certain characteristics that
would lead to arrays where majority of array items have the same
value, choosing a strided array format may lead to inefficient use of
memory and processor resources. For such cases various sparse array
formats have been proposed that provide considerable memory storage
savings as well as more efficient processing possibilities but a the
cost of increased complexity in the sparse storage format.

To be specific, in this document we consider three array storage
formats: strided dense, COO sparse, and CSR sparse array storage
formats. However, implementing other storage formats are encouraged if
these have potential to increase the performance of existing array
operations.


### Strided array storage format

TBD: explain the relation between array item index and the
corresponding memory location of the item value using offset and
strides.


### COO sparse array storage format

TBD: explain the relation between array item index and the
corresponding memory location of the item value.

TBD: pytorch specialities: hybrid and coalesce tensor

### CSR sparse array storage format

TBD: explain the relation between array item index and the
corresponding memory location of the item value.

## Interpretation of unspecified sparse array entries

TBD: discuss fill value vs offset

## Operations with PyTorch tensors

TBD: what is implemented and what is not? Comparison with SciPy,
PyData, etc sparse array projects.

### Construction

### Indexing operations

### Arithmetic operations

### Tensor algebra operations

### Functions on tensors

## Implementation plan

TBD: prioritize

TBD: existing matmut proposal for CSR format

TBD: high-priority fill value/offset support
