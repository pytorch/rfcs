

<details>
<summary>Instructions - click to expand</summary>

- Fork the rfcs repo: https://github.com/pytorch/rfcs
- Copy `RFC-0000-template.md` to `RFC-00xx-my-feature.md`, or write your own open-ended proposal. Put care into the details.
- Submit a pull request titled `RFC-00xx-my-feature`. 
    - Assign the `draft` label while composing the RFC. You may find it easier to use a WYSIWYG editor (like Google Docs) when working with a few close collaborators; feel free to use whatever platform you like. Ideally this document is publicly visible and is linked to from the PR.
    - When opening the RFC for general discussion, copy your document into the `RFC-00xx-my-feature.md` file on the PR and assign the `commenting` label.
- Build consensus for your proposal, integrate feedback and revise it as needed, and summarize the outcome of the discussion via a [resolution template](https://github.com/pytorch/rfcs/blob/rfc-process/RFC-0000-template.md#resolution).
    - If the RFC is idle here (no activity for 2 weeks), assign the label `stalled` to the PR.
- Once the discussion has settled, assign a new label based on the level of support:
    - `accepted` if a decision has been made in the RFC
    - `draft` if the author needs to rework the RFC’s proposal
    - `shelved` if there are no plans to move ahead with the current RFC’s proposal. We want neither to think about evaluating the proposal
nor about implementing the described feature until some time in the future.
- A state of `accepted` means that the core team has agreed in principle to the proposal, and it is ready for implementation. 
- The author (or any interested developer) should next open a tracking issue on Github corresponding to the RFC.
    - This tracking issue should contain the implementation next steps. Link to this tracking issue on the RFC (in the Resolution > Next Steps section)
- Once all relevant PRs are merged, the RFC’s status label can be finally updated to `closed`.

</details>


# Batch Kronecker Product

**Authors:**
* @dannygoodacre


## **Summary**
Expand the PyTorch Kronecker product to allow for batching the product along a specified dimension.


## **Motivation**

This feature is motivated by a need for such a function in a project I am working on, which simulates the evolution of quantum spin systems. Since this feature can be generalised from matrices to tensors --- as the Kronecker product has already been in PyTorch --- I feel that it belongs within PyTorch itself.


## **Proposed Implementation**
In the following, `a` and `b` are tensors of the specified sizes, and `batch_kron(a, b, batch_dim)` the proposed feature, where `batch_dim = 0` refers to the outermost dimension.

### In use
For a given dimension along which to batch, the requirement is that all dimensions agree up to and including the batch dimension. For example,
```
a.shape = (2,3,4)
b.shape = (2,3,5)
```
could be batched along dimension 0 and 1, whereas
```
a.shape = (3,7,2)
b.shape = (3,5,3)
```
could only be batched along dimension 0.

Some more examples:

```
a.shape = (2,5,4)
b.shape = (2,5,7)
batch_kron(a, b, 0).shape = (2,25,28)
batch_kron(a, b, 1).shape = (2,5,28)

a.shape = (5,4,3)
b.shape = (5,3,9)
batch_kron(a, b, 0).shape = (5, 12, 27)
batch_kron(a, b, 1) 'Error: required dimensions do not match.'
```
For all dimensions higher than the batch dimension, the regular Kronecker product is applied to each pair of tensors. The result is a tensor of the required shape, as seen in the next subsection.

### How it works internally
The method is similar to how `kron` currently works within LibTorch, although extra care has to be taken when handling the batch dimensions. 

The idea is to reshape the two tensors in such a way that there are dimensions of size one inserted at alternating points in the shape of each tensors. This means that when the vectors are Kroneckered, the dimensions align accordingly and the result can be reshaped as needed.

For example, consider 
```
a.shape = (2,3,5)
b.shape = (2,3,8)
```
For the regular Kronecker product, the reshaping looks like the following:
```
a_reshape = (2,1,3,1,5,1)
b_reshape = (1,2,1,3,1,8)
result_reshape = (4,9,40)
```
We see that the resultant shape is merely an element-wise product of the initial shapes.

For the batch Kronecker product (in dimension zero), we have the following:

```
a_reshape = (2,3,1,5,1)
b_reshape = (2,1,3,1,8)
result_reshape = (2, 9, 40)
```

And in dimension one:

```
a_reshape = (2,3,5,1)
b_reshape = (2,3,1,8)
result_reshape = (2,3,40)
```

We can see that for the dimensions up to and including the batch dimension, there is no change; and that the dimensions after the batch dimension are alternated in the same manner as the regular Kronecker product. 

The element-wise multiplying of the dimension sizes in the resultant shape also follows this parallel to the regular Kronecker product.


### A further generalisation

This functionality is beyond what I need, but I feel that this generalisation would be suited to PyTorch.

So far, all of my uses have been with tensors that have an equal number of dimensions. But there is nothing to say this must always be the case, since the Kronecker product will work will tensors of any shape.

For example,
```
a.shape = (2,5,2,3)
b.shape = (2,5,6)
batch_kron(a, b, 0).shape = (2,5,10,18)
batch_kron(a, b, 1).shape = (2,5,2,18)
```
We see that after the batch dimensions, the rest of the dimensions are right-aligned and Kroneckered as usual.


## **Drawbacks**
This is not a breaking change.

We *may* be able to utilise some of the existing code for the Kronecker product in this implementation, since the approaches are similar, but I have not yet found a way to do this without writing a separate C++ function.

Accomodating the generalisation detailed above may be quite slow, and this will be an issue for cases when the user doesn't want to do this. We could introduce a faster version for the simpler case, and internally determine which to use based on the lengths of the tensor shapes.


## **Prior Art**
A version been implemented in [pylabk](https://github.com/yulkang/pylabyk/blob/master/numpytorch.py) for only 3D tensors, batching the product in the first dimension only. The approach used in the same as how `kron` is currently implemented, albeit using PyTorch functions instead of being written in C++ for LibTorch.

This feature had already been proposed in an [issue on PyTorch](https://github.com/pytorch/pytorch/issues/77537), but this was not fleshed out at all. I felt that raising this as an RFC was necessary in order to clear up the concerns I currently have.


## **How we teach this**

The term `batch_dim` refers to the dimension along which we will distribute the Kronecker product. Otherwise, the terminology is identical to that which is currently used in PyTorch documentation.

Depending on unresolved question 1, we may need to re-write the `kron` documentation.

Given an understanding of the Kronecker product, this feature is intuitive to understand.


## **Unresolved questions**
1. Should we overload the current `kron` function with the `batch_dim` parameter --- `kron(a, b, batch_dim=None)`, for example --- or introduce a new function?
2. Do we want the further generalisation, or just the implementation for a matching number of dimensions?
3. Should this be implemented at the PyTorch or LibTorch level?


## Resolution

### Level of Support


#### Additional Context


### Next Steps


#### Tracking issue


#### Exceptions
