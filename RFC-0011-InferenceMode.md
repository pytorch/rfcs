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
 - **Inference mode** can be turned on when you are sure you don't need any autograd computation. This saves the cost of creating autograd graph and `as_view` / `version_counter` setup compared to the normal mode.
 - **Inference tensor** is defined as a tensor without Autograd **and** InplaceOrView keys on it.
 - **Normal tensor** has both Autograd & InplaceOrView keys. This includes both `requires_grad=true` and `requires_grad=false` tensors. (see [Ideal end state] section for more details).
 - Additional notes:
   - All Inference tensors are created in inference mode, but not all of the tensors created in inference mode are inference tensors. For example, a view of normal tensor created in inference mode is still a normal tensor (but with special `creation_meta`!).
   - (Autograd & !InplaceOrView) and (!Autogad & InplaceOrView) are invalid states, we don't have such tensors.

# Expected Behavior
## Implementation:
1. Inference Mode: InplaceOrView not in included, Autograd in excluded
2. Normal Mode: InplaceOrView in included, Autograd not in excluded
3. In VariableType kernel, throw an error if input is inference tensor.
4. In InplaceOrView kernel, throw an error if Autograd keyset is not in excluded set already.
5. In VariableType kernel, throw an error if input is a view with `NO_VARIABLE_TYPE_VIEW` creation_meta.
## Behavior
| Mode          | Input                                    | Op         | Go through Kernels                        | Produced Output                                            |   |   |
|---------------|------------------------------------------|------------|-------------------------------------------|------------------------------------------------------------|---|---|
| InferenceMode | All inference tensors                    | functional | CPU                                       | inference tensor                                           |   |   |
| InferenceMode | All inference tensors                    | view       | CPU                                       | inference tensor                                           |   |   |
| InferenceMode | All inference tensors                    | inplace    | CPU                                       | inference tensor                                           |   |   |
| InferenceMode | Contains normal tensor                   | functional | InplaceOrView(fallthrough), CPU           | inference tensor                                           |   |   |
| InferenceMode | Contains normal tensor                   | view       | InplaceOrView, CPU                        | normal tensor (with creation_meta=NO_VARIABLE_TYPE_VIEW)   |   |   |
| InferenceMode | Contains normal tensor                   | inplace    | InplaceOrView, CPU                        | normal tensor (which is input itself with updated version) |   |   |
| NormalMode    | All inference tensors                    | functional | InplaceOrView(fallthrough), CPU           | normal tensor (see note*)                                  |   |   |
| NormalMode    | All inference tensors                    | view       | InplaceOrView(ERROR4!), CPU               |                                                            |   |   |
| NormalMode    | All inference tensors                    | inplace    | InplaceOrView(ERROR4!), CPU               |                                                            |   |   |
| NormalMode    | Mixed normal tensor and inference tensor | functional | VariableType(ERROR3!), InplaceOrView, CPU |                                                            |   |   |
| NormalMode    | Mixed normal tensor and inference tensor | view       | VariableType(ERROR3!), InplaceOrView, CPU |                                                            |   |   |
| NormalMode    | Mixed normal tensor and inference tensor | inplace    | VariableType(ERROR3!), InplaceOrView, CPU |                                                            |   |   |
|               |                                          |            |                                           |                                                            |   |   |
|               |                                          |            |                                           |                                                            |   |   |
## additional notes:
1. ERROR3 means it hits (3) described in implementation section and ERROR4 means it hits (4) in implementation section.
2. Functional ops on inference tensors might run slower outside InferenceMode than inside.
   But it's fine that we don't care about perf of this case that much.

## Alternative implementations we've considered and why they don't work:
1. For NormalMode + All inference tensors + functional op, an alternative behavior we perfer but didn't implement is throwing an error by forcing this op go through VariableType kernel and hit the assert_no_inference_tensor check. But to do that we'll have to add c10::autograd_dispatch_keyset to the globally enabled set, but doing that might accidentally call autograd kernel from a backend that doesn't match tensor input. Thus we allow functional ops run without throwing an error.
2. Why implementation (1) and (2)?
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
3.
# Ideal end state
Ideal end state is that we can link skip VariableType kernel when requires_grad=False which means we don't always go through VariableType kernel in normal mode.
But this work is currently blocked for the following reason:
- If requires_grad=False skips VariableType kernel, functional ops won't be able to go through `AutoDispatchBelowInplaceOrView` guard which suppresses both autograd and InplaceOrView keys in TLS excluded. Not suppressing InplaceOrView key means unnecessary calls to `as_view/increment_version` if any view/inplace ops are used in the kernel implementation which adds a lot of overhead. To avoid overhead, instead of fallthrough kerenl being backend fallback, we'll want to use a real kernel that suppresses InplaceOrView key. But compared to the current implementation which only adds an extra dispatch for view/inplace ops, it forces all functional ops to have an extra dispatch as well. That's why it's blocked.
- To unblock it requires some fixes like identifying at:: callsites in backend-specific kernels (static analysis? ) , replacing these with at::native:: should unblock us from linking requires_grad with VariableType kernel.
