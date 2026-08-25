# Adding a `torch.bcomplex32` dtype

**Authors:**
* @hameerabbasi

## **Summary**
This RFC proposes the addition of a `torch.bcomplex32` dtype, a dtype with a real/imaginary components having the dtype `torch.bfloat16`.

## **Motivation**
PyTorch's floating-point support for operations is wide, even when such floating-point datatypes are not directly supported by the target hardware. Examples of this include `torch.float16` and `torch.bfloat16`.

However, the corresponding complex-valued datatypes aren't currently fully supported (in the case of `torch.complex32`) or even present (in the case of `torch.bcomplex32`). This poses challenges for users who would like to perform operations on tensors with these datatypes (namely, those tensors where the real and imaginary component is `torch.float16` or `torch.bfloat16`).

Historically, the reason for this missing support was the lack of demand, coupled with the additional binary size associated with supporting such operations in eager mode. However, the demand is rapidly changing with the rise of world models (a term used to mean a neural network that's aware of physics laws).

## **Proposed Implementation**
With WIP the support for the compilation of complex-valued tensors (see [#16982](https://github.com/pytorch/pytorch/pull/169832), and [#16721](https://github.com/pytorch/pytorch/pull/167621)), the binary size becomes less of a concern, as such kernels can be JIT compiled on-demand by decomposing them into operations on purely real-valued tensors.

This lifts the barrier for binary size for such datatypes. However, the datatypes themselves still need to exist; perhaps as shell dtypes and support for some basic operations. [#173783](https://github.com/pytorch/pytorch/pull/173783) proposes the addition of `torch.bcomplex32` as a shell dtype, only supporting some very basic operations. Examples of such operations are:

* `torch.view_as_real`/`torch.view_as_complex`
* `torch.real`, `torch.imag`
* `Tensor.item()`, `Tensor.data()`
* `torch.conj`, `torch.neg`

## **Metrics**
*WIP*

## **Drawbacks**
However, this change comes with some backwards-incompatible behaviour, which is best illustrated by a code example:
```python
re = torch.randn((5,), dtype=torch.bfloat16)
im = torch.randn((5,), dtype=torch.bfloat16)

c = torch.complex(re, im)

# `float64` before #173783, but `bcomplex32` after
print(c.dtype)
```
Such backwards-incompatible behaviour is unavoidable due to the need to construct `bcomplex32` tensors. However, there is another related concern:
```python
# Now errors due to `bcomplex32`
torch.tanh(c)
```
However, the workaround is also simple:
```python
torch.tanh(c) # Raises an error asking users to cast to `complex64`
c = c.to(torch.complex64)
torch.tanh(c) # Now works
```

Overall, the authors believe that the backwards-incompatible behaviour is worth the internal consistency.

## **Alternatives**
The alternative is making the `torch._subclasses.complex_tensor.ComplexTensor` subclass public, and changing its internal design so it stores the dtype of the components rather than the composite (e.g. `float64` instead of `complex128`). However, this would be a much more invasive change, as the dispatching of many complex-valued operations (such as `torch.real`, `torch.conj`, ...) is special-cased for tensors (and subclasses) with real-valued dtypes.

## **Prior Art**
A number of libraries have support for the `bcomplex32` dtype
* CUDA and HIP both have the `CUDA_C_BF16` and `HIP_C_BF16` datatypes.
* Most Python array libraries don't have the equivalent of `torch.complex32` or `torch.bcomplex32`.

## **How we teach this**
The documentation will add additional references to `torch.bcomplex32`, and on the page for `torch.complex` there will be a backwards compatibility note.

## Resolution
TBD

### Level of Support
TBD

#### Additional Context
TBD

### Next Steps
TBD

#### Tracking issue
WIP

#### Exceptions
TBD
