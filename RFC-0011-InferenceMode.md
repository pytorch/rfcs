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
  then it behaves the same way as no grad mode).

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
  `InferenceMode`

* Non-goal: Avoid performance slowdown for view/inplace operations
  outside of `InferenceMode`.  Benchmarking on popular models reveal
  that a slight slowdown on these operations is acceptable; in our
  case, this slowdown will be due to an extra redispatch in these cases.

# User description

`InferenceMode` is an RAII guard which can be enabled for a given block
of code.  Inside inference mode, all newly allocated (non-view) tensors
are marked as **inference tensors**; these tensors are guaranteed not to
alias with tensors that may have been saved for backwards (or are
otherwise making use of version counters--perhaps more accurately,
you could call these "non version counter tracked tensors").  Inference
tensors do not have a version counter, and raise an error if you try
to read their version (e.g., because you saved this tensor for
backwards.)

A non-view tensor is an inference tensor if and only if it was
allocated during inference mode.  A view tensor is an inference
tensor if and only if the tensor it is a view of is an inference tensor.

Inside an `InferenceMode` block, we make the following performance
guarantees:

* All operations do not record `grad_fn`, even if their `requires_grad=True`
  (like `NoGradMode`).  This applies for both inference tensors and
  normal tensors (also like `NoGradMode`).
* View operations do not do view tracking (like `NoGradMode` after
  PyTorch 1.9).  This applies for both inference tensors and normal
  tensors.
* Inplace operations on inference tensors are guaranteed not to do
  a version counter bump (which is equivalent to an atomic increment).
  Inplace operations on normal tensors still do version counter bumps.

# Implementation description

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
the invariant:

  **The invariant:** Any inference tensor (tensor whose dispatch key
  set does not include Autograd nor InplaceOrView) is guaranteed
  not to alias with any tensor which is saved for backwards (or
  otherwise depends on accurate version counter tracking).

This invariant guarantees that it safe skip version counter bumps on
inference tensors.

**Inference mode** is defined to be the state when:

* Autograd is added to the TLS excluded set
* InplaceOrView is removed from the TLS included set (recall that by
  default, InplaceOrView is part of the TLS included set)
* View metadata is not recorded (similar to NoGradMode)

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

These rules satisfy the invariant: functional operations are guaranteed
to have non-aliasing outputs and are safe to mark as inference tensors;
view operations introducing aliasing relationships, and it is only
safe for inference tensors to alias other inference tensors.

Finally, we must ensure that inference tensors are not saved for
backwards.  **TODO FINISH ME**

**Examples.**  Given the rules above, we can describe the behavior
for each combination of possibilities:

* In inference mode...
  * Inplace operation...
    * On a normal tensor - version counter will increment (due
      to InplaceOrView key on the normal tensor)
    * On an inference tensor - no increment
  * View operation...
    * On a normal tensor - view metadata is not recorded,
      version counter is propagated, result is a normal tensor
    * On an inference tensor - view metadata is not recorded,
      result is an inference tensor
  * Functional operation...
    * On a normal tensor - produces an inference tensor
    * On an inference tensor - produces an inference tensor
* Outside of inference mode...
  * Inplace operation...
    * On an inference tensor - allowed, no increment
  * View operation...
    * On an inference tensor - allowed, view metadata 


**TODO EVERYTHING ELSE**

     - Inplace operations on both normal/inference tensors are OK
        - Inplace operation on inference tensor is guaranteed not to VC bump
        - NB: if you do an inplace operation on a normal tensor, you WILL get a version counter bump
     - View operations on both normal/inference tensors are OK
        -  View operation on inference tensor is guaranteed not to allocate view metadata
        -  View operation on normal tensor produces a normal tensor(NO_GRAD_FN), behavior is the same as creating a view inside NoGrad mode. 

* **Inference tensor** are tensors that are constructed if and only if inference mode is enabled, with the exception of views on normal tensors. Non-inference tensors are called **normal tensors**. 
    * Q: Why not views on normal tensors? A: Because we guarantee performance on inference tensors, but views on normal tensors require additional safety checks (e.g. normal tensor ----(view)---> ----(inplace)----> this should properly bump version on base which requires view produce a normal tensor).
    * NB: Inference tensors and bad normal tensors are leaf tensors.
    * Outside of inference mode, the following operations on inference tensors is forbidden:
        * Inplace/view operations (functional operations produce normal tensors), if at least one input tensor is inference mode.
            * Why? In principle, these are safe if they produce inference tensors, but we are trying to maintain the invariant that inference tensors are ONLY created in inference mode.
            * Impl: Functional on normal tensors is allowed because we cannot conveniently ban it (VariableType/InplaceOrView kernel are all skipped)
        * Mixing inference and normal tensors, even for functional operations, is forbidden.
            * Why? For simplicity of implementation. In particular, if you save the inference tensor in backwards, you’re likely to hit an error in a weird place (better to error early). By forbidding mixed operations, it is impossible for this situation to occur.
    * Impl: inference tensors are guaranteed to have is_leaf=True. 


 - **Normal tensor** has both Autograd & InplaceOrView keys. This includes both `requires_grad=true` and `requires_grad=false` tensors. (see [Ideal end state] section for more details).
 - Additional notes:
   - All Inference tensors are created in inference mode, but not all of the tensors created in inference mode are inference tensors. For example, a view of normal tensor created in inference mode is still a normal tensor (but with special `creation_meta=NO_GRAD_FN`!).
   - (Autograd & !InplaceOrView) and (!Autogad & InplaceOrView) are invalid states, we don't have such tensors.



## Alternative implementations we've considered and why they don't work:
1. For NormalMode + All inference tensors + functional op, an alternative behavior we prefer but didn't implement is throwing an error by forcing this op go through VariableType kernel and hit the assert_no_inference_tensor check. But to do that we'll have to add c10::autograd_dispatch_keyset to the globally enabled set, but doing that might accidentally call autograd kernel from a backend that doesn't match tensor input. Thus we allow functional ops run without throwing an error.
2. 
```
    // 1. When InferenceMode is enabled, Autograd dispatch keys are excluded
    //    but not InplaceOrView key.
    //
    //    For example:
    //    torch::Tensor a = torch::ones({1, 2, 3}).set_requires_grad(true);
    //    torch::Tensor k = a + 2;
    //    {
    //      c10::InferenceMode guard(true);
    //      k.add_(2);
    //    }
    //    `k.add_(2)` still need to go through InplaceOrView kernel so that it's
    //    prepared for future autograd.
    //  2. When InferenceMode is disabled, InplaceOrView must be added
    //     to included set.
    //
    //     For example:
    //     torch::Tensor a;
    //     {
    //       c10::InferenceMode guard(true);
    //       torch::Tensor in = torch::ones({2, 2});
    //       a = in.view({1, 4});
    //     }
    //     torch::Tensor c = a.view({4, 1}); // (*)
    //     If we don't add InplaceOrView to included set, (*) will skip its as_view
    //     setup entirely, `c` will be a Tensor that is not from Inference mode
    //     but has potentially wrong view metadata which should be forbidden..
    //     By going through InplaceOrView kernel, we can throw an error since it
    //     broke our invariant: "Autograd keys must be in excluded set before
    //     reaching InplaceOrView kernel".
```

# Ideal end state
Ideal end state is that we can link skip VariableType kernel when requires_grad=False which means we don't always go through VariableType kernel in normal mode.
But this work is currently blocked for the following reason:
- If requires_grad=False skips VariableType kernel, functional ops won't be able to go through `AutoDispatchBelowInplaceOrView` guard which suppresses both autograd and InplaceOrView keys in TLS excluded. Not suppressing InplaceOrView key means unnecessary calls to `as_view/increment_version` if any view/inplace ops are used in the kernel implementation which adds a lot of overhead. To avoid overhead, instead of fallthrough kerenl being backend fallback, we'll want to use a real kernel that suppresses InplaceOrView key. But compared to the current implementation which only adds an extra dispatch for view/inplace ops, it forces all functional ops to have an extra dispatch as well. That's why it's blocked.
- To unblock it requires some fixes like identifying at:: callsites in backend-specific kernels (static analysis? ) , replacing these with at::native:: should unblock us from linking requires_grad with VariableType kernel.
