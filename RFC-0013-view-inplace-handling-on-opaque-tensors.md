## Summary

In the ideal end state, dispatcher should be able to provide an opt-in
kernel to functionalize view & inplace ops. Currently we have a similar
infrastructure in pytorch/xla, but that’s only available on the lazy
traced graph. To make this elimination pass available to FX and Mobile
GPU like other backends that only support opaque tensor storage, this
pass should be lifted up to dispatcher level.

## Problem Statement

Historically PyTorch has view/inplace ops that reuse the same storage
for different tensors, this allows us to do fast and memory efficient
reshaping, slicing and element-wise operations. But by sharing storage
it also affect user-facing APIs in both forward and backward pass.

```
x = y[0]
x.fill_(2)
# y is modified as well!
```

And these behaviors are hard for backends without storage to follow,
some use cases are:

* XLA Tensor doesn’t have storage, so they developed a small infra to
  eliminate view and inplace ops.
* Vulkan cannot support aliases, as they represent tensors as textures
  and thus cannot have pointers into internal contents of the tensors.
* FX/Torchscript IR may want IR without view/inplace as well for ease of
  mutation.
* vmap cannot handle certain in-place operations. A way to replace an
  in-place operation with an out-of-place operation would make it so
  that vmap can work over more code

**Since it’s a common request from multiple backends(not only lazy
backends), the best way to solve the issue is actually lift/reimplement
what pytorch/xla’s inplace & view elimination so that it’s available in
dispatcher, backends can choose to opt in this pass or not depending on
their own need.**

This work doesn’t conflict with what we are doing to [upstream
pytorch/xla](https://github.com/pytorch/rfcs/pull/18) and can be done in
parallel, but we think it’s a better end state so that everyone get the
benefit of view/inplace elimination.

## How XLA view removal works, briefly

Code pointers:

* https://github.com/pytorch/xla/blob/master/torch_xla/csrc/view.h
* https://github.com/pytorch/xla/blob/master/torch_xla/csrc/view.cpp

Primary data structures:

* ViewInfo - reified representation of a view operation (so we can replay it later)
* Alias - every set of tensors which may alias each other have a single Alias struct that represents all the aliases. Contains:
    * ir_value_ - the current value of the base tensor
    * updates_ - list of updates in chronological order which were made to the base tensor. Note that the updates themselves may be putbacks of modifications of a view of the tensor, so they have to record both the ir_value that was put into the location, as well as the set of views that narrows the base tensor to the place where the putback occurs
    * generation_ - incremented every time the view is updated; if this is bumped, the views also need to get updated
* View - represents a view of a tensor. Contains:
    * view_infos_ - a reified list of view operations that constructed this view from the base tensor. This is used to retrieve what the updated view looks like, after any deferred updates have been applied
    * alias_ - the Alias info corresponding to this base of this view
    * ir_value_ - the actual (lazy) current value of this view
    * generation_ - the generation of the base at the time this view was created. If this and `alias_->generation_` don’t agree you have to do an update to this View to apply deferred operations

Example:

```
x = torch.zeros(2)
x1 = x.view(1, 2)
x[1].fill_(2)
y = x1 + 3
```

1. x is the base tensor. It (lazily) gets an Alias structure allocated for it. In the beginning, generation=0 and has no updates. Contents = [0, 0]
2. x.view(1, 2) is a view. Aliases with x, view info is `view(1, 2)`, contents = [[0, 0]], generation = 0. 
3. x[1] is a view. Aliases with x, view info is `index(1)`, contents = 0, generation = 0
4. x[1].fill_(2) performs an inplace update. Updated contents = 2, generation = 1. It also updates the Alias lazily, registering that the contents of the view (2) have been updated at `index(1)` in the alias (updates_ is no empty)
    1. NB: The alias update is done lazily, because if the base tensor is never used again in the future, updating it is a waste
5. x1 + 3 makes use of view. To use a view, we check if the generation matches; it does not. This means we have to play forward the updates on the alias (alias contents is now [[0, 2]]), and then redo the view operations (`view(1, 2)`) to reconstruct the updated view
6. FIN

Memory management properties:

* The main source of leaks in the direct XLA implementation is the
  updates array: it retains a reference to updated view contents (so it
  knows to how to replay them when a sync happens). This is a time-space
  tradeoff; by delaying applying the update to base, we save when we
  didn’t actually need to apply the update at all, at the cost of having
  to retain all of the intermediate update infos until all aliases go
  dead.
* A “miniature” lazy tensor implementation must be maintained for views,
  so that we can replay the views after updates have been made to the
  base. PyTorch already has native support for this functionality.
  Additionally, if you have support for arbitrary stride updates (XLA
  does not, but maybe you can write this kernel), only strides need to
  be maintained to handle view information.
