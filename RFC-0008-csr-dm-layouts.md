# CSR and DM storage formats proposal

|            |                 |
| ---------- | --------------- |
| Authors    | Pearu Peterson  |
| Created    | 2021-01-15      |

## Problem statement

Currently, PyTorch implements the following multi-dimensional array
storage formats:

- Strided: an array is defined by shape, strides, and 1D values
  (PyTorch internal buffer).
- COO: an array is defined by shape, indices (PyTorch 2D Strided
  tensor), and values (PyTorch 1D Strided tensor).

Here we do not consider hybrid COO tensors for simplicity.

For matrix multiplication of sparse tensors, the COO storage format is
suboptimal and more memory/processor efficient storage formats are
needed, especially those that allow parallel processing of array data
with minimal cache misses and to execute in streaming fashion.

## Proposal

We propose implementing the following new storage formats in PyTorch:

- CSR: a 2D array is defined by shape, compressed row indices (PyTorch
  1D tensor), column indices (PyTorch 1D tensor), and values (PyTorch
  1D tensor).

- DM: an array is defined by shape, dimensions map, and storage array
  (an arbitrary PyTorch tensor).

The DM (Dimensions Mapping) storage format is a new concept that
constitutes of dimensionality reduction and promotion of the storage
array as defined by dimensions map parameters (see the DM storage
specification below). The DM storage array is a PyTorch tensor that
can use any of the existing storage formats (Strided, COO) or any of
the proposed new storage formats: CSR and DM.

## DM storage format

The main feature of the DM storage format is to provide an one-to-one
correspondence between the elements of the *"wrapper"* array and of
the *"storage"* array when the dimensionalities of the arrays are
different.  Such bijective mapping of the array elements is realized
in between the indices sets of the wrapper and storage arrays and can
be described in multiple ways. Our description of the mapping is as
follows:

- `shape` parameter is a list of wrapper array dimensions. For
  example, if `shape=(3, 4, 5)` then the wrapper array is a 3D array
  with the first dimension having size `3`, the second `4`, and the
  last `5`.

- `dimensions` parameter is a permutation list of wrapper array
  dimensions. For example, if `dimensions=(2, 1, 0)` then the first
  and the last dimensions are considered swapped when applying the
  dimensions partitioning as decribed below.

- `partitioning` parameter is a list of `dimensions` indices defining
  the partition of the wrapper array dimensions (after the
  permutation) to the dimensions of the storage array. The
  dimensionality of the storage array must be equal to
  `len(partitioning) + 1`.  For example, if `partitioning=(1,)` and
  `dimensions=(2, 1, 0)` then the dimensionality of the storage array
  is `2` and the last dimension (with size `5`) of the wrapper array
  is mapped to the first dimension and the rest dimensions (with sizes
  `4` and `3`) to the second dimension of the storage array,
  respectively. As a result, the shape of the storage array will be
  `(5, 12)`.

While the `dimensions` and `partitioning` parameters define the
bijectitve mapping between the elements of the wrapper and storage
array, internally, the DM storage format uses a different set of the
DM storage format parameters: `rdimensions`, `rstrides`, and `roffset`
which can be computed from `dimensions` and `partitioning` parameters
but are more convinient to describe the slicing of DM formatted
arrays.  The exact definition of these parameters is implemented in
the [source code of the DM format
prototype](https://github.com/pearu/gcs/blob/main/gcs/storage.py#L953).
Obviously, the mapping between the elements of sliced DM array and the
storage array will be an injective mapping.

## Benefits and perspectives

While the CSR format supports more performant Linear Algebra operations
with 2D sparse arrays (see the benchmark results in [PyTorch PR
44190](https://github.com/pytorch/pytorch/pull/44190) description),
the DM format provides a method of generalizing 2D CSR formatted
arrays to multidimensional arrays. In addition, the DM format allows
several new perspectives to multidimensional array representation:

- no-copy slicing and axis swapping for COO and CSR/CSS tensors.

- hierarchical storage format: DM tensor can use another DM tensor as
  an storage array.

- extending storage formats with restricted dimensioniality (e.g. the
  CSR format is essentially for 2D arrays only) to arbitrary
  dimensionality. The dimensionality of the storage format can be
  greater than 2 but not greater than the dimensionality of the
  wrapper array.

In the above, we have implicitly assumed that the dimensionality of
the storage array is not be greater than the dimensionality of the DM
array. Removing this restriction would allow defining iterators over
multi-dimensional arrays with various orderings. For instance, if the
wrapper array is one-dimensional and the storage array is
two-dimensional then the DM format can be used to define a row-wise
iterator, or a column-wise iterator, or a row-wise reversed iterator,
etc as long as the bijective relation between the wrapper and storage
array elements exists.


## Additional materials

A pure Python prototype of DM storage format is provided in [GCS
package](https://github.com/pearu/gcs/tree/main/gcs) where the
dimensions mapping and slicing of DM arrays are implemented and
demonstrated on wrapping of storage arrays using various storage
formats like Strided, COO, FlatCOO, and CSR. See also [Gentle
Introduction To
GCS](https://github.com/pearu/gcs/blob/main/GentleIntroductionToGCS.md)
that exemplifies the dimensions mapping concepts for CSR storage array
in more detail.

## Work plan

- Implement CSR storage format, the implementation exists in the form
  of [PyTorch PR 44190](https://github.com/pytorch/pytorch/pull/44190)
  but it requires some clean up.
- Implement DM storage format in storage array format agnostic way. As
  a first instance, provide it as pure Python prototype.

The [PyTorch PR 44190](https://github.com/pytorch/pytorch/pull/44190)
is kept for reference but will be retired eventually.
