# GCS index interconversion

The GCS format allows expression of a sparse tensor of arbitrary shape in a form that
that collapses to a simple CSR format when sparsifying a 2D tensor. For the purpose
of this file, we will refer to the real co-ordinates of an elements within a GCS
tensor as 'dense' co-ordinates in order to easily understand conversion between
these types of co-ordinates.

In this file we will see how interconversions of dense co-ordinates and GCS co-ordinates
can be done. This discussion comes from mathemtical fundamentals of dimension reduction
laid down in [ArrayFundamentals.md](https://github.com/Quansight-Labs/rfcs/blob/pearu/rfc0005/RFC0003-sparse-roadmap/ArrayFundamentals.md#dimension-reduction-and-promotion-of-arrays).

In order to demonstrate with a concrete example, lets say we have a tensor `torch.ones(5,5,5,5,5)`.
We want to map the first two dimensions to the first GCS dimension and the rest of the dimensions to
the second GCS dimension. Using this dimension splitting, we get strides of `(5, 1)` for the first
dimension and `(25, 5, 1)` for the second dimension. Therefore, if we want to convert a dense
co-ordiate that is say `(4,1,0,3,1)` into the GCS co-ordinates, it will basically be done as follows:
```
4 * 5 + 1 * 1 = 21
0 * 5 * 5 + 3 * 5 + 1 * 1 = 16
```

This gives the GCS co-ordinates `(21, 16)`. Since we have split our dimensions along the second dimension
in this case, we can assume that our dimensions are split as `(a1, a0)` and `(b2, b1, b0)`.

If we want to convert these co-ordiates back into dense co-oridinates, that can be done using the following steps:
```
a1 = (21 // 5) mod 5 = 4
a0 = (21 // 1) mod 5 = 1
b2 = (16 // 25) mod 5 = 0
b1 = (16 // 5) mod 5 = 3
b0 = (16 // 1) mod 5 = 1
```

The above conversion from the GCS co-ordinates to the dense co-ordinates can be performed using the following
Python script:
``` python
def gcs_to_dense_convert(coords, strides0, strides1, dims0, dims1):
    strides = [strides0, strides1]
    dims = [dims0, dims1]
    dense = []
    for i, p in enumerate(coords):
        dim = dims[i]
        stride = strides[i]

        if len(dim) != len(stride):
            raise Exception("dim and stride must be same for coord i=", i)

        for r in range(len(dim)):
            dense.append((p // stride[r]) % dim[r])

    return dense

gcs_to_dense_coords((21, 16), (5, 1), (25, 5, 1), (5, 5), (5, 5, 5))
```

Conversely conversion from dense to GCS co-ordinates can be done using the following:

``` python
def dense_to_gcs_convert(coords, shape, l):
    gcs = []
    shape0 = shape[0:l]
    shape1 = shape[l-1:-1]
    ndims = len(shape)
    
    dims = [list(range(l)), list(range(l,ndims))]
    strides = [make_strides(shape0), make_strides(shape1)]

    for i in range(2):
        stride = strides[i]
        dim = dims[i]
        gcs.append(sum([stride[r] * coords[dim[r]] for r in range(len(dim))]))

    return gcs

print(dense_to_gcs_convert((4,1,0,3,1), (5,5,5,5,5), 2))
```
