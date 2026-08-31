### [RFC] Masked reductions and normalizations


#### Purpose of this document

Discuss semantics and implementation details of masked reduction and normalization operators. 


#### State of the world

We currently have a few operations in PyTorch that take masks as inputs. nn.Transformer and related operators and operators masked_scatter, masked_select and masked_fill.


#### Project goals

Extend reduction and normalization semantics to apply only to a subset of values as determined by a separate masked input. This aims to close multiple issues and the related user use cases and feature requests.

**Motivation**



* There is some explicit user demand for reductions with masked semantics for sparse Tensors
    * [[feature request] Sparse (hybrid sparse-dense) output option for topk, min, max ](https://github.com/pytorch/pytorch/issues/34951)
    * [Feature request: sparse matrix max(axis)](https://github.com/pytorch/pytorch/issues/4247) 
    * [Implemented torch.sparse.max(sparse_tensor, dim)](https://github.com/pytorch/pytorch/pull/59340)
* [RFC-0004: Adding fill value property to PyTorch sparse tensors](https://github.com/pytorch/rfcs/pull/8) is blocked on discussions around masked semantics.
* There are also demands for masked normalizations for dense Tensors and some prior art that shows performance improvements.
    * [Add masked_softmax to speed up masking in multihead attention](https://github.com/pytorch/pytorch/pull/48441)
    * [Fuse softmax and masking in MultiheadAttention](https://github.com/pytorch/pytorch/issues/44945)
    * [[FR] Safe softmax](https://github.com/pytorch/pytorch/issues/55056)   

**Existing reduction operators**

Reductions are operators that remove a dimension from a Tensor. They apply a binary operation across a list of Tensor slices and aggregate the result. This is different from normalizations that return the same shape with all dimensions intact.

Dense ([docs](https://pytorch.org/docs/master/torch.html#reduction-ops))



* argmax
* argmin
* amax
* amin
* aminmax
* all
* any
* max
* min
* dist
* logsumexp
* mean
* median
* nanmedian
* mode
* norm
* nansum
* prod
* quantile
* nanquantile
* std
* std_mean
* sum
* unique
* unique_consecutive
* var
* var_mean
* count_nonzero

Sparse COO ([docs](https://pytorch.org/docs/master/sparse.html#torch-functions-specific-to-sparse-tensors))



* sum

Sparse CSR: None

**Existing normalization operators**

These are a bit more difficult to list exhaustively, because we don't have an explicit category for them. A normalization operator applies a reduction to a Tensor slice and uses the result to normalize the slice.

Dense ([functionals](https://pytorch.org/docs/master/nn.functional.html))



* softmax
* log_softmax
* batch_norm
* group_norm
* layer_norm
* instance_norm
* local_response_norm
* normalize
* cross_entropy
* nll_loss

Sparse COO ([docs](https://pytorch.org/docs/master/sparse.html#torch-functions-specific-to-sparse-tensors))



* softmax
* log_softmax
* norm / native_norm

Sparse CSR: None

**Operator constraints and general signature - Reductions**

**Input types:** A mask is a boolean tensor. It accompanies a dense Tensor of the same shape. If an entry is True the corresponding element at the same index in the paired dense Tensor is a "valid" value. If it is False it is not. "valid" here means that this value is meant to be included in the computation and otherwise is meant to be ignored. This matches the semantics of masked_scatter, masked_select and masked_fill. 

**Fully masked rows:** If a slice (e.g. row) is fully masked out there is no guarantee the corresponding return values are filled with any specific value such as the operation's identity value. However, given a sparse input with a row entirely zero and masked out the result is likely to be zero to maximize memory savings.

**Operator coverage:** As a first step we should implement the basics: sum, prod, mean, amin, amax.

**Semantics:** Indeed the best way to describe the behavior is to implement it. Please note that this is only meant to describe semantics and is not an actual implementation. It also implies the gradients for these operations. Whether an input or mask is backed by a dense or compressed storage layout has no influence on the semantics of these operations. More concretely, unspecified values for sparse layouts are always zero in this context. If the mask is None the behavior of the operator is equivalent to the unmasked variant.

**Input layouts:** If an input is passed a sparse Tensor, it will return a sparse output. At first the layout of the mask and input must match. If the input is a sparse Tensor, the layout of the mask must be too. This is to ease kernel development and might quickly change, if there is user demand. However, the user can always convert a mask to sparse or dense before invocation.


```
def masked_sum(input, dim, keepdim, dtype, mask):
    return torch.sum(input * mask, dim, keepdim, dtype=dtype)

def masked_prod(input, dim, keepdim, dtype, mask):
    return torch.prod(input * mask + ~mask, dim, keepdim, dtype=dtype)

def masked_mean(input, dim, keepdim, dtype, mask):
    return (torch.sum(input * mask, dim, keepdim, dtype=dtype) /                 
            torch.sum(mask, dim, keepdim, dtype=dtype))

def masked_amin(input, dim, keepdim, mask):
    return torch.amin(input.masked_fill(~mask, float('+inf')), dim, keepdim)

def masked_amax(input, dim, keepdim, mask):
    return torch.amax(input.masked_fill(~mask, float('-inf')), dim, keepdim)
```


NOTE: This doesn't work when the mask and input don't match in dimension. In full these operations are meant to respect standard broadcasting rules.

Now if the user wants to update the mask after the reduction they can use torch.any. We might decide to provide a flag later on if the user wants to get the updated mask together with the reduced input.


```
new_mask = torch.any(mask, dim, keepdim)
```


**Example invocations**


```
>>> input
tensor([[-3., -2., -1.],
        [ 0.,  1.,  2.]])

>>> mask
tensor([[ True, False,  True],
        [False, False, False]])

>>> masked_sum(input, 1, False, torch.float, mask)
tensor([-4.,  0.])

>>> masked_prod(input, 1, False, torch.float, mask)
tensor([3., 1.])

>>> masked_mean(input, 1, False, torch.float, mask)
tensor([-2., nan])

>>> masked_amin(input, 1, False, mask)
tensor([-3., inf])

>>> masked_amax(input, 1, False, mask)
tensor([-1., -inf])

>>> torch.any(mask, 1, False)
tensor([ True, False])
```


**Operator constraints and general signature - Normalizations**

**Input types:** Input type contracts are the same as for reductions.

**Masked elements:** A masked out element may change in value and is not guaranteed to have any specific value after the normalization is applied. The value is likely to be chosen to ease kernel development for example to recycle the indices of sparse inputs.

**Operator coverage:** softmax and log_softmax are obvious immediate candidates since there are already sparse implementations that can be recycled and demand for dense variants.

**Semantics:** Similar to reductions the behavior is best described by below code. If the mask is None the behavior of the operator is equivalent to the unmasked variant.

**Input layouts:** Input layout contracts are the same as for reductions.


```
def masked_softmax(input, dim, _stacklevel=3, dtype=None, mask=None):
    assert mask is not None
    return torch.nn.functional.softmax(input.masked_fill(~mask, float('-inf')),                  
                                       dim, _stacklevel, dtype)

def masked_log_softmax(input, dim, _stacklevel=3, dtype=None, mask=None):
    assert mask is not None
    return torch.nn.functional.log_softmax(input.masked_fill(~mask, float('-inf')), 
                                           dim, _stacklevel, dtype)

def masked_normalize(input, p, dim, eps=1e-12, out=None, mask=None):
    assert mask is not None
    return torch.nn.functional.normalize(input.masked_fill(~mask, 0),
                                         p, dim, eps, out)
```


NOTE: This doesn't work when the mask and input don't match in dimension. In full these operations are meant to respect standard broadcasting rules.

**Example invocations**


```
>>> input
tensor([[-3., -2., -1.],
        [ 0.,  1.,  2.]])

>>> mask
tensor([[ True, False,  True],
        [False, False, False]])

>>> masked_softmax(input, 1, dtype=torch.double, mask=mask)
tensor([[0.1192, 0.0000, 0.8808],
        [   nan,    nan,    nan]], dtype=torch.float64)

>>> masked_log_softmax(input, 1, dtype=torch.double, mask=mask)
tensor([[-2.1269,    -inf, -0.1269],
        [    nan,     nan,     nan]], dtype=torch.float64)

>>> masked_normalize(input, 2.0, 1, mask=mask)
tensor([[-0.9487,  0.0000, -0.3162],
        [ 0.0000,  0.0000,  0.0000]])

>>> masked_normalize(input, 3.0, 1, mask=mask)
tensor([[-0.9880,  0.0000, -0.3293],
        [ 0.0000,  0.0000,  0.0000]])
```


**Additional constructors and conversions**

For best performance power users will likely want to provide a mask that matches the indices of the sparse input Tensor, if it also aligns with the intended semantics. For each COO and CSR we can provide helper functions that can construct these masks, which can be as simple as calling into the corresponding constructor with binary values and given indices.


#### Potential future work

**Masked Tensor**

Introduce a new Tensor type that has the mask baked into its representation. This is similar to NumPy's [masked array](https://numpy.org/doc/stable/reference/maskedarray.generic.html). We may also end up implementing something like this, but there is plenty of discussion about the exact semantics of unary, binary and reduction operators and how they relate to each other. It will take longer to determine this and masked reductions can be used as a step towards it while also being reusable.


#### Open questions

**Namespace**

torch.sparse, torch.mask or toplevel torch? Should be a prototype at first.
