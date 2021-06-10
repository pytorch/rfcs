## Summary

`InferenceMode` is a new context manager / RAII guard analogous to
`NoGradMode` to be used when you are certain your operations will have no
interactions with autograd (e.g., model training).  Code run under this
mode gets better performance by disabling view tracking and version
counter bumps.

## Motivation

In production use of PyTorch for inference, we have seen a proliferation
of uses of the C++ guard `AutoNonVariableTypeMode`, which disables
autograd, view tracking and version counter bumps.  Unfortunately,
current colloquial use of this guard is unsafe: it is possible to use
`AutoNonVariableTypeMode` to bypass PyTorch's safety checks for, e.g.,
ensuring tensors saved for backwards are not subsequently mutated.

`InferenceMode` offers a drop in replacement for
`AutoNonVariableTypeMode` which:

1. Preserves the performance characteristics of
   `AutoNonVariableTypeMode` (Autograd, view tracking and version
   counter bumps are skipped for all tensors allocated within the
   inference mode region), but

2. Is safe, in the sense that it is not possible to bypass version
   counter updates on tensors which may alias with tensors which
   have been saved for backwards.

For now, this guard is to only be made available inside C++, although
we could also introduce a Python guard `torch.inference_mode` as well.

Some goals and non-goals:

* Goal: `InferenceMode` is semantically equivalent to `NoGradMode`,
  except some operations may not be supported.  (In other words, this is
  a partial equivalence: *if* inference mode does not throw an error,
  then it behaves the same way as no grad mode).  Caveat: this
  equivalence does not extend to methods that expose private
  implementation details; esp., `Tensor._is_view` and `Tensor._base`.

* Goal: It should be possible to run code that allocates parameters
  (tensors with `requires_grad=True`) unchanged inside of an inference
  mode block.

* Goal: Don't be a global or compile time flag.  This makes
  `InferenceMode` widely applicable as it can still be used in processes
  where there may be training going on in another thread (e.g.,
  federated learning on mobile).

* Non-goal: `InferenceMode` doesn't affect computation beyond its scope.
  Indeed, the capacity for tensors allocated in `InferenceMode` (so
  called "inference tensors") to behave differently even outside of
  `InferenceMode` is one of the key implementation tools to ensuring
  that `InferenceMode` is safe.

* Non-goal: Make operations on inference tensors fast outside of
  `InferenceMode`; nor, be maximally expressive with inference
  tensor outside of `InferenceMode`.

* Non-goal: Avoid performance slowdown for view/inplace operations
  outside of `InferenceMode`.  Benchmarking on popular models reveal
  that a slight slowdown on these operations is acceptable; in our
  case, this slowdown will be due to an extra redispatch in these cases.

## User description

`InferenceMode` is an RAII guard which can be enabled for a given block
of code.  Inside inference mode, all newly allocated (non-view) tensors
are marked as **inference tensors**; these tensors are guaranteed not to
alias with tensors that may have been saved for backwards (or are
otherwise making use of version counters--perhaps more accurately,
you could call these "non version counter tracked tensors").  Inference
tensors:

* Do not have a version counter.
* Raise an error if you try to read their version (e.g., because you
  saved this tensor for backwards.)
* Raise an error if you try to mutate them into requiring gradients
  (e.g., directly set `requires_grad=True` or mutate them with a tensor
  that `requires_grad=True`.)

A non-view tensor is an inference tensor if and only if it was
allocated during inference mode.  A view tensor is an inference
tensor if and only if the tensor it is a view of is an inference tensor.

Inside an `InferenceMode` block, we make the following performance
guarantees:

* All operations do not record `grad_fn`, even if their `requires_grad=True`
  (like `NoGradMode`).  This applies for both inference tensors and
  normal tensors (also like `NoGradMode`).
* View operations on inference tensors do not do view tracking; views
  and base inference tensors are indistinguishable.
* Inplace operations on inference tensors are guaranteed not to do
  a version counter bump (which is equivalent to an atomic increment).
  Inplace operations on normal tensors still do version counter bumps.

## Implementation description

**Dispatcher.**  The dispatcher decides what implementation of a kernel
to call when an operator is invoked.  The set of possible options is
controlled by several sources:

* Tensor inputs (keys are unioned from all inputs)
* TLS included set
* TLS excluded set (which removes keys from the above two sources)

**Autograd.**  This is a preexisting dispatch key which is responsible
for recording `grad_fn` on output tensors when any of their inputs
`require_grad`.

Autograd dispatch key is associated with tensors.  Prior to this
proposal, all tensors unconditionally have an autograd key.
(Technically, the autograd dispatch key is not a single key,
but a set of keys per backend; for the purposes of this proposal,
this doesn't matter.)

**InplaceOrView.**  This is a new dispatch key which is responsible for
doing version counter bumps on inplace operations, and view metadata
tracking for view ops.  Previously, this functionality was also done
as part of the Autograd kernel.  For all other operators, it is a fallthrough
kernel.  Here is an example kernel for an inplace op and a view op prior
to this proposal:

```
Tensor & add__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other, Scalar alpha) {
  {
    at::AutoDispatchBelowInplaceOrView guard;
    at::redispatch::add_(ks & c10::after_InplaceOrView_keyset, self, other, alpha);
  }
  increment_version(self);
  return self;
}

Tensor expand(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef size, bool implicit) {
  auto _tmp = ([&]() {
    at::AutoDispatchBelowInplaceOrView guard;
    return at::redispatch::expand(ks & c10::after_InplaceOrView_keyset, self, size, implicit);
  })();
  std::function<at::Tensor(const at::Tensor&)> func=nullptr;
  if (false || !self.unsafeGetTensorImpl()->support_as_strided()) {
    auto size_vec = size.vec();
    func = [=](const at::Tensor& input_base) {
      return input_base.expand(size_vec, implicit);
    };
  }
  auto result = as_view(
   /* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true,
   /* is_fw_differentiable */ true, /* view_func */ func,
   /* creation_meta */ at::GradMode::is_enabled() ? CreationMeta::DEFAULT : CreationMeta::NO_GRAD_MODE
  return result;
}
```

InplaceOrView is considered part of the default TLS included set; i.e.,
it is always run.  It is also associated with normal tensors (like Autograd),
so that these kernels get run even if InplaceOrView is not in the
default TLS included set.

**The algorithm.**  At a high level, we would like to skip both the
Autograd and InplaceOrView kernels while in inference mode, whenever
it is safe to do so.  Whether or not this is safe is maintained by
a pair of invariants:

  **The no-aliasing invariant:** Inference tensors are guaranteed
  not to alias with any tensor which is saved for backwards (or
  otherwise depends on accurate version counter tracking)

  **The immutable invariant:** Inference tensors are immutable outside of
  inference mode.

The no-aliasing invariant guarantees it is safe to skip version counter
bumps when mutating inference tensors, as the set of tensors affected by
mutation is precisely the set of aliases to that tensor.  The immutable
invariant guarantees it is safe to skip view metadata, as view metadata
is only used to enable inplace updates on tensors that require
gradients.

**Inference mode** is defined to be the state when:

* Autograd is added to the TLS excluded set
* InplaceOrView is removed from the TLS included set (recall that by
  default, InplaceOrView is part of the TLS included set)
* If view metadata is recorded (e.g., because a tensor has InplaceOrView
  directly recorded on it), the creation metadata of the view is
  set to forbid subsequent inplace modification with
  `requires_grad=True` tensors (`CreationMeta::NO_GRAD_MODE`)

It is legal for only Autograd to be excluded (this happens during normal
processing of Autograd kernels), but it is illegal for InplaceOrView to
be removed from the TLS included set if Autograd is not also excluded.

An **inference tensor** is a tensor that does not have the Autograd or
InplaceOrView dispatch keys and has no version counter.  Whether or not
the result of a functional/view operation is an inference tensor (e.g.,
that omit these keys) is the result of the following rules:

* If a functional operation, the output tensor is an inference
  tensor if and only if we are running in inference mode.  In practice,
  this is implemented by only adding the Autograd+InplaceOrView keys
  in the TensorImpl constructor if inference mode is off.
* If a view operation, the output tensor is an inference tensor
  if and only if the input tensor is an inference tensor.  In practice,
  this is implemented by propagating the dispatch key set from the
  base tensor to the view tensor.

These rules guarantee half of the no-aliasing invariant: functional
operations are guaranteed to have non-aliasing outputs and are safe to
mark as inference tensors; view operations introducing aliasing
relationships, and it is only safe for inference tensors to alias other
inference tensors.

Furthermore, the following operations on inference tensors are disabled:

* Inplace modifications on inference tensors outside of inference mode
  (tested at the point we do version counter increments; this code is
  guaranteed to run outside of inference mode because InplaceOrView is
  part of default included TLS).  This guarantees the immutability
  invariant.  (TODO: Also need to prevent `requires_grad` from being
  explicitly toggled)
* Saving an inference tensor for backwards (tested in the constructor
  of SavedVariable).  This guarantees the other half of the no-aliasing
  invariant.

**Examples.**  Given the rules above, we can describe the behavior
for each combination of possibilities:

* In inference mode...
  * Inplace operation...
    * On a normal tensor - version counter will increment (due
      to InplaceOrView key on the normal tensor)
    * On an inference tensor - no increment
  * View operation...
    * On a normal tensor - view metadata is recorded, creation
      meta is set to `INFERENCE_MODE`, version counter is propagated,
      result is a normal tensor
    * On an inference tensor - view metadata is not recorded,
      result is an inference tensor
  * Functional operation...
    * On a normal tensor - produces an inference tensor
    * On an inference tensor - produces an inference tensor
* Outside of inference mode...
  * Inplace operation...
    * On an inference tensor - forbidden
  * View operation...
    * On an inference tensor - allowed, view metadata is not
      recorded, result is an inference tensor

**Edge case: explicit `requires_grad` setting.**  One might expect that in
no grad mode that it is impossible to allocate a tensor with
`requires_grad=True`.  However, this is not true: any tensor that
is explicitly allocated with `requires_grad=True` preserves this
property outside of no grad mode:

```
>>> with torch.no_grad():
...   x = torch.empty(2, requires_grad=True)
...
>>> x
tensor([-1.3667e-17,  4.5801e-41], requires_grad=True)
```

This can also be achieved by explicitly setting
`x.requires_grad = True`.  Furthermore, in no grad mode, this requires
grad setting propagates to views

```
>>> with torch.no_grad():
...   x = torch.empty(2)
...   y = x.view(2)
...   x.requires_grad = True
...
>>> y.requires_grad
True
```

This poses a problem for inference mode, which doesn't track view
metadata and cannot implement this propagation.  Our proposed solution
is to forbid setting `requires_grad` (but permit tensors to be directly
constructed with `requires_grad=True`).  This cannot be easily
implemented today as internally `requires_grad=True` factory is
implemented by first constructing a tensor, and then setting its
`requires_grad=True`.

## Future work: skipping Autograd kernels when `requires_grad=False`

As view and inplace handling has been moved out of Autograd kernels, a
tantalizing possibility is to remove the Autograd dispatch keys from
tensors with `requires_grad=False`, thus skipping this kernel entirely.

But this work is currently blocked for the following reason:

- If `requires_grad=False` skips Autograd kernel, functional ops won't
  be able to go through `AutoDispatchBelowInplaceOrView` guard which
  suppresses both autograd and InplaceOrView keys in TLS excluded. Not
  suppressing InplaceOrView key means unnecessary calls to
  `as_view/increment_version` if any view/inplace ops are used in the
  kernel implementation which adds a lot of overhead. To avoid overhead,
  instead of fallthrough kerenl being backend fallback, we'll want to
  use a real kernel that suppresses InplaceOrView key. But compared to
  the current implementation which only adds an extra dispatch for
  view/inplace ops, it forces all functional ops to have an extra
  dispatch as well. That's why it's blocked.
- To unblock it requires some fixes like identifying `at::` callsites in
  backend-specific kernels (static analysis? ) , replacing these with
  `at::native::` should unblock us from linking `requires_grad` with
  VariableType kernel.  Alternately, do
  https://github.com/pytorch/pytorch/issues/54614
