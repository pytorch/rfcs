# Structured kernel definitions in PyTorch

This is a proposal for a new code generation facility for writing
kernels in PyTorch, where we will automatically generate
easy-to-get-wrong boilerplate for functional (`add`), inplace (`add_`) and
out (`add_out`) variants of functions, as well as common code (device
guards, version counter tracking). The net result is you only need to
write a shape checking function and an out-kernel when writing a
function.

**NOW HAS A PROTOTYPE** https://github.com/pytorch/pytorch/pull/45277 

## Goals

* **Be an opt-in mechanism**: always OK to continue to write code as it
  is today. This ensures that third-party backend extenders also can
  make things work without making use of codegen

* **Reduce shape checking boilerplate**: make it easier to share common
  shape checking code between CPU and CUDA implementations, as well as
  with out-of-tree backend implementations

* **Reduce functional/inplace/out boilerplate**: avoid having to write
  foo, foo_, foo_out variants for every function; make it harder to
  forget to add foo_out variant when it is appropriate

* **Be able to run forward shape computation without kernel**: so
  compilers and static runtimes can easily compute sizes of all elements
  in the framework without having to actually run the computation

* **Provide entry point for static runtime**: provide public API for
  accessing operators directly, bypassing output allocation, shape
  checking, device guards.

* **Bypass dispatcher overhead**: in high performance cases, e.g., core
  framework operators, reduce the number of redispatches to improve
  performance

* **Unify with version counter bumps and view metadata tracking**: these
  are currently done in autograd generated code but must be performed
  unconditionally even when autograd is disabled. This logic to be
  incorporated with logic here

* **Work with mobile selective build**: mobile selective build is
  implemented in codegen, and implementation strategy must be compatible
  with design constraints in mobile

## Non-goals

* **Directly handle TensorIterator**: TensorIterator is a uniquely
  complicated mechanism, and we think it should be designed for on its
  own.  However, we think that this is an important case and should
  be handled sooner rather than later.

* **Be zero runtime cost all the time**: We are willing to give up some
  runtime efficiency for cleaner code. We are willing to give up
  efficiency *by default* for out of tree implementers, unless they opt
  in to higher performance. There is always an escape hatch to be high
  performance if absolutely necessary

* **No codegen**: As long as it is possible to implement things out of
  tree (at the cost of human understandable boilerplate), codegen is a
  reasonable strategy for implementing features we need

* **Major format change to native_functions.yaml**: We could do this. We
  are choosing not to in order to reduce the degrees of freedom in what
  changes we make to `native_functions.yaml`.

## Summary

* Core

    * Define a new API for shape checking functions. Shape checking
      functions are called from generated code that implements
      functional/inplace/out variants of functions.

    * Define a new API for kernels. Static kernel API does not do shape
      checking, output is always preallocated at correct size. This API
      is public and a suitable entrypoint for static runtime. Static
      kernels are called from generated code like above.

    * Code generate meta functions from shape checking functions to
      provide public API for running shape computations without any
      kernel.

    * Generated code is augmented to do version counter bumps and view
      tracking

* Extensions

    * Add a new dispatch key (name tbd) which contains shape checking,
      device guard, version counter bump. External backend
      implementations of PyTorch operators which do not explicitly opt
      out of this dispatch key will get this logic applied to them. This
      key is skipped for core operators, as this checking logic is fused
      directly into the backend kernel. External backends can also opt
      to fuse this logic in.

## Details

Let’s suppose you want to write a new operator from scratch. The first
thing you will do is add a native_functions.yaml declaration for it.
Let’s take `upsample_nearest1d` as the example for this post.  You will
write an entry like this, describing the *functional* version of this
operator:

```
- func: upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor
  dispatch:
    CPU: upsample_nearest1d_cpu
    CUDA: upsample_nearest1d_cuda
```

Because you are a conscientious implementor, you also want to provide an
out= variant of this function. This version gets a separate entry:

```
- func: upsample_nearest1d.out(Tensor self, int[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU: upsample_nearest1d_out_cpu
    CUDA: upsample_nearest1d_out_cuda
```

Ordinarily, these two declarations would cause function signatures for
`upsample_nearest1d_cpu`, etc., to be generated into NativeFunctions.h,
and then you would go ahead and implement them.  However, we propose a
new *structured* format for writing kernels. We’ll do this by marking
the out version of this operator as structured and deleting dispatch
entries from the functional version (the functional operator is
*implicitly* associated with the out-of-place version in the same way
derivatives.yaml schemas are matched with their functional/inplace/out
variants):

```
- func: upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor
  # [NEW] dispatches for this function are omitted

- func: upsample_nearest1d.out(Tensor self, int[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
  structured: True  # [NEW]
  dispatch:
    CPU: upsample_nearest1d_structured_cpu
    CUDA: upsample_nearest1d_structured_cuda
```

Structured definitions require a different set of functions to be
written to implement the operator.  In this particular case, the
functions you have to implement are:

```
namespace meta {
 TensorMeta upsample_nearest1d(
 const Tensor& self, IntArrayRef output_size, optional<double> scales);
 }
 
 namespace native {
 // Precondition: out is an allocated and appropriately sized tensor;
 // all shape checks have passed, device guards have been set,
// version counter bumps are all handled, etc...
 void upsample_nearest1d_structured_cpu(
const TensorMeta& out_meta, const Tensor& out, const Tensor& self, IntArrayRef output_size, optional<double> scales);
 void upsample_nearest1d_structured_cuda(
const TensorMeta& out_meta,  const Tensor& out, const Tensor& self, IntArrayRef output_size, optional<double> scales);
 }
```

The code generator then generates the boilerplate code to put these
functions together:

```
namespace native {

Tensor upsample_nearest1d_cuda(const Tensor& self, IntArrayRef output_size, optional<double> scales) {
  CUDADeviceGuard g(self.device());
  TensorMeta expected = meta::upsample_nearest1d(self, output_size, scales);
  Tensor result = native::empty(expected.sizes(), expected.options());
  upsample_nearest1d_structured_cuda(expected, result, self, output_size, scales);
  return result;
}

Tensor& upsample_nearest1d_out_cuda(Tensor& result, const Tensor& self, IntArrayRef output_size, optional<double> scales) {
  // In event of multiple tensor arguments, code generation should
  // be responsible for making sure all devices are consistent
  CUDADeviceGuard g(self.device());
  TensorMeta expected = meta::upsample_nearest1d(self, output_size, scales);
  TORCH_CHECK(expected.options == result.options());
  result.resize_(expected.size());
  upsample_nearest1d_structured_cuda(expected, result, self, output_size, scales);
  increment_version(result);
  return result;
}

// CPU follows similarly

}

```

Notice that the generated boilerplate here handles several concerns:

* It properly sets up device guards for the kernel

* For the `_out` variant, it tests that the inferred tensor options are
  consistent with the passed in tensor

* It centralizes the PyTorch convention that improperly sized out=
  arguments get resized to fit the output

* It handles version counter bumping for mutations

* It is carefully written so that there are no extra redispatches; all
  calls are direct non-dispatched function calls

It’s also worth noting what things the boilerplate here does not handle:

* It forces the allocation of a TensorMeta struct per output tensor,
  before the final allocation of the output tensor. You could save a few
  instructions by eliminating this struct entirely. However, I could not
  think of an elegant API for doing so.

In absolute terms, the boilerplate code saved here is not all that
large; only several lines. However, when multiplied over the number of
kernels in PyTorch, and the subtlety of remembering to handle all of
these issues, and we think the use of code generation to automatically
generate this boilerplate is worth it.

The functionality code generated here is not made available to other
backends. This is problematic, so we also generate a separate dispatch
key handler which will automatically handle shape checking and other
concerns for backends that are not concerned with performance:

```
// Name TBD; for this example we will call it DispatchKey::Common, as the
// functionality here is common to all backends. This is an alias key
// that resolves CommonXLA/CommonMSNPU/... in the same way as Autograd.

Tensor upsample_nearest1d_common(const Tensor& self, IntArrayRef output_size, optional<double> scales) {
  // TODO: RecordFunction could be added here, if desired
  DeviceGuard g(self.device());
  TensorMeta expected = meta::upsample_nearest1d(self, output_size, scales);
  Tensor result = at::empty(expected.sizes(), expected.options());
  ExcludeDispatchKeyGuard g2(DispatchKey::Common);
  // Notice that upsample_nearest1d is ignored here. It may be a good idea
  // to skip this implementation, or use a slightly different variant, if
  // a backend explicitly registered upsample_nearest1d
  at::upsample_nearest1d_out(expected, result, self, output_size, scales);
  return result;
}

TORCH_LIBRARY_IMPL(aten, Common, m) {
  m.impl("upsample_nearest1d", upsample_nearest1d_common);
}
```

A backend that would like to fuse this boilerplate into their kernel for
performance reasons can simply override the Common entry with
fallthrough:

```
TORCH_LIBRARY_IMPL(aten, CommonXLA, m) {
  m.impl("upsample_nearest1d", CppFunction::makeFallthrough);
}
```

Finally, structured kernels also implicitly register an implementation
for the Meta key, which is the API for dry running shape/dtype
calculations without any kernels involved:

```
Tensor upsample_nearest1d_meta(const Tensor& self, IntArrayRef output_size, optional<double> scales) {
  TensorMeta expected = meta::upsample_nearest1d(self, output_size, scales);
  return native::empty_meta(expected.sizes(), expected.options());
}

TORCH_LIBRARY_IMPL(aten, Meta, m) {
  m.impl("upsample_nearest1d", upsample_nearest1d_meta);
}
```

## Discussion

* Why not just use the Common dispatch key for everything?

    * Performance reasons. Introducing the common key would induce an
      extra redispatch, which at time of writing would give up quite a
      bit of performance due to dispatch overhead, for no particularly
      good reason.

* Will this increase binary size?

    * Not necessarily. By-in-large, all of the boilerplate code
      generated here is already duplicated at the source level today
      (for example, many operators have shape checks factored into
      header files that are then shared between two operators).

* I don’t like codegen.

    * An earlier version of this proposal had the boilerplate
      generated using C++ templates rather than codegen. However, we
      think the formulation in this proposal is superior under the
      constraint that mobile selective build must keep working, as we
      cannot directly write registrations in source files, and so we
      must intermediate between the structured and non-structured
      variants.

* What about TensorIterator?

    * TensorIterator is challenging in several respects. It doesn’t fit
      nicely into the proposed structure here, as the TensorIterator
      structure must be passed to the actual kernel to do the actual
      operation which means that the internal calling convention is not
      the same as the external one. Even if TensorIterator were
      refactored to split shape checking and everything else, there are
      internal computations (such as what the computation dtype should
      be)

    * More broadly speaking, the meta-issue is that TensorIterator is
      very complicated and very widely used, and so it takes a lot of
      time to make major changes to its API.

    * Meta functions can be added manually for TensorIterator, so you
      can still get meta functions for important operators. I prototyped
      an example of this at
      https://github.com/ezyang/pytorch/commit/fa1d87032dbb8e15d7d6e8f3aabb1b659487658c
