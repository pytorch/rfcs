Note: a large part of this RFC will become "InferenceMode" documentation once it's finalized.

## Goals:
- Provide a RAII in C++ and a context manager in Python frontend to switch between inference mode and normal mode, with the following constraints:
  - correctness is always guaranteed. (compared to `AutoNonVariableType` which has risks producing silent wrong result.)
  - performance of inference mode should match current existing `AutoNonVariableTypeMode` which is widely used in prod.
  - switching between normal mode and inference mode should be really easy with minimal code change.
- Make `AutoNonVariableTypeMode` an internal only API, replace all callsites of `AutoNonVariableTypeMode` outside pytorch codebase with the new `InferenceMode`.

## Non-goals:
- Match the theoretical best inference performance which can be achieved by stripping all autograd related stuff at build time (not flexible).
- Allowing the most flexible interaction between normal mode and inference mode. Current main use case for inference mode is "either inference or normal" without mixing, so we ban a lot of interactions between two modes to keep the implementation simple.

# Different levels of control over autograd (copied from @Alban)
The following modes are ranked from slowest to fastest in speed, and from the most flexible to the most restrictive in what users can do.

* Normal Mode: we create the graph for all Tensors that require gradients, always track view and inplace even they don't require gradients.
* GradMode disabled: we never create the graph, still track all views and inplace. User code always succeeds to properly track gradients.
* InferenceMode: we never create the graph, only track view and inplace if that could lead to silent error, skip that logic otherwise (we can potentially skip the allocation of the version counter for these tensors). Raise errors if users try to mix inference mode and autograd. (this one will have the same perf as AutoNonVariableTypeMode used today, but it will be safe!).
* (Not available yet) Compile time no grad: all autograd related code is completely removed for the best perf. This requires the users to change their code to make sure they don't use any autograd construct or they will see errors.

# New concepts

In this RFC we introduces the following new concepts:
- **InplaceOrView** is a new dispatch key in dispatcher. It's fallthrough kernel by default, but it does `increment_version` for inplace ops and `as_view` setup for view ops. Here's some generated InplaceOrView kernels:
```
   Tensor & add__Tensor(c10::DispatchKeySet ks, Tensor & self, const Tensor & other, Scalar alpha) {

     TORCH_CHECK(c10::impl::is_all_dispatch_keyset_excluded(c10::autograd_dispatch_keyset),
       "Calling inplace/view ops on inference tensor outside InferenceMode is not allowed, ",
       "consider making a clone first. ",
       "If you have a valid use case, please make a feature request to PyTorch.");
     {
       at::AutoDispatchBelowInplaceOrView guard;
       at::redispatch::add_(ks & c10::after_InplaceOrView_keyset, self, other, alpha);
     }
     increment_version(self);
     return self;
   }


   Tensor expand(c10::DispatchKeySet ks, const Tensor & self, IntArrayRef size, bool implicit) {

     TORCH_CHECK(c10::impl::is_all_dispatch_keyset_excluded(c10::autograd_dispatch_keyset),
       "Calling inplace/view ops on inference tensor outside InferenceMode is not allowed, ",
       "consider making a clone first. ",
       "If you have a valid use case, please make a feature request to PyTorch.");
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
     auto result = as_view(/* base */ self, /* output */ _tmp, /* is_bw_differentiable */ true, /* is_fw_differentiable */ true, /* view_func */ func, /* creatio
     return result;
   }
 ```
 - **Inference mode** a thread local state that can be turned on via RAII guard/context manager. (Either you are in inference mode, or you are not.) Intuitively, inference mode lets you do inference only operation with better performance than normal mode.
   - All operations do not create autograd graph, even if the inputs require_grad=True
   - Setting requires_grad in inference mode will update requires_grad field on tensors, but it doesn't affect any behavior inside InferenceMode.
   - Things that continue to work:
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
