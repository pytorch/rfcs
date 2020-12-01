# Structured kernel definitions in PyTorch

This is a proposal for a new code generation facility for writing
kernels in PyTorch, where we will automatically generate
easy-to-get-wrong boilerplate for functional (`add`), inplace (`add_`)
and out (`add_out`) variants of functions, as well as common code
(device guards). The net result is you only need to write a shape
checking function and an out-kernel when writing a function.

## Goals

* **Be an opt-in mechanism**: always OK to continue to write code as it
  is today. This ensures that third-party backend extenders also can
  make things work without making use of codegen.

* **Reduce shape checking boilerplate**: make it easier to share common
  shape checking code between CPU and CUDA implementations, as well as
  with out-of-tree backend implementations.

* **Reduce functional/inplace/out boilerplate**: avoid having to write
  `foo`, `foo_`, `foo_out` variants for every function; make it harder to
  forget to add `foo_out` variant when it is appropriate.

* **Be able to run forward shape computation without kernel**: so
  compilers and static runtimes can easily compute sizes of all elements
  in the framework without having to actually run the computation.

* **Provide entry point for static runtime**: provide public API for
  accessing operators directly, bypassing output allocation, shape
  checking, device guards.

* **Bypass dispatcher overhead**: in high performance cases, e.g., core
  framework operators, reduce the number of redispatches to improve
  performance.

* **Unify with version counter bumps and view metadata tracking**: these
  are currently done in autograd generated code but must be performed
  unconditionally even when autograd is disabled. This logic to be
  incorporated with logic here.

* **Work with mobile selective build**: mobile selective build is
  implemented in codegen, and implementation strategy must be compatible
  with design constraints in mobile.

* **Fix some operator UX paper cuts**: have C++ compiler tell you when
  you've specified the signature of a native function incorrectly,
  rather than get a linker error.

* **Handle TensorIterator operators without major refactoring**: TensorIterator
  operators cover a third of all operators in PyTorch, and structured
  kernels must be able to account for them, without requiring major
  changes to how TensorIterator is implemented.

## Non-goals

* **Be zero runtime cost all the time**: We are willing to give up some
  runtime efficiency for cleaner code. We are willing to give up
  efficiency *by default* for out of tree implementers, unless they opt
  in to higher performance. There is always an escape hatch to be high
  performance if absolutely necessary.

* **No codegen**: As long as it is possible to implement things out of
  tree (at the cost of human understandable boilerplate), codegen is a
  reasonable strategy for implementing features we need.

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

    * Generated code is augmented with extra boilerplate like device
      guards that all kernels typically need to do.

* Extensions

    * Add a new dispatch key `Common` which contains shape checking,
      device guard. External backend implementations of PyTorch
      operators which do not explicitly opt out of this dispatch key
      will get this logic applied to them. This key is skipped for core
      operators, as this checking logic is fused directly into the
      backend kernel. External backends can also opt to fuse this logic
      in.

## `native_functions.yaml` syntax

Let’s suppose you want to write a new operator from scratch. The first
thing you will do is add a `native_functions.yaml` declaration for it.
Let’s take `upsample_nearest1d` as the example for this post.

In the classic system, you will write an entry like this, describing the
*functional* version of this operator:

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
and then you would go ahead and implement them.

### Structured keyword proposal

We propose a new *structured* format for writing kernels. In this
proposal variant, we’ll do this by marking the out version of this
operator as structured and deleting dispatch entries from the functional
version, instead delegating its implementation to the out version:

```
- func: upsample_nearest1d(Tensor self, int[1] output_size, float? scales=None) -> Tensor
  structured_delegate: upsample_nearest1d.out  # [NEW], replacing dispatch

- func: upsample_nearest1d.out(Tensor self, int[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
  structured: True  # [NEW]
  dispatch:
    CPU: upsample_nearest1d_structured_cpu
    CUDA: upsample_nearest1d_structured_cuda
```

## Functions to define

Structured definitions require a different set of functions to be
written to implement the operator.  In this particular case, the
functions you have to implement are:

```
namespace meta {
  /* macro expands to: upsample_nearest1d::upsample_nearest1d( */
  TORCH_META_FUNC(upsample_nearest1d) (
    const Tensor& self, IntArrayRef output_size, optional<double> scales
  ) {
    ... compute sizes and options, check shapes ...
    set_output(sizes, options);
  }
}

namespace native {
  // Precondition: out is an allocated and appropriately sized tensor;
  // all shape checks have passed, device guards have been set,
  // version counter bumps are all handled, etc...
  /* macro expands to: void upsample_nearest1d_structured_cpu::impl( */
  TORCH_IMPL_FUNC(upsample_nearest1d_structured_cpu) (
    const Tensor& out, const Tensor& self, IntArrayRef output_size, optional<double> scales
  );
  /* macro expands to: void upsample_nearest1d_structured_cpu::impl( */
  TORCH_IMPL_FUNC(upsample_nearest1d_structured_cuda) (
    const Tensor& out, const Tensor& self, IntArrayRef output_size, optional<double> scales
  );
}
```

The code generator then generates the boilerplate code to put these
functions together.  The boilerplate is somewhat involved, and we will
explain it below.

```
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Not code generated; common code
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

#define TORCH_META_FUNC(name) name::name
#define TORCH_IMPL_FUNC(name) void name::impl

// Parent class for all code-generated meta:: classes
struct MetaBase {
  // TODO: Maybe some of these should be optional, but in many cases they
  // be implicitly made optional by passing an empty list
  virtual void set_output(int64_t output_idx, IntArrayRef size, IntArrayRef strides, TensorOptions options, DimnameList names) = 0;

  // Convenience helpers
  void set_output(IntArrayRef size, TensorOptions options) {
    set_output(0, size, {}, options, {});
  }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Code generated per operator
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

namespace meta {

struct upsample_nearest1d : public MetaBase {
  void meta(const Tensor& self, IntArrayRef output_size, optional<double> scales); // user defined
};

} // namespace meta

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Code generated per dispatch table entry for operator
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

namespace native {

struct upsample_nearest1d_cuda : public meta::upsample_nearest1d {
  void impl(const Tensor& self, IntArrayRef output_size, optional<double> scales); // user defined
};

// functional implementation

// NB: set_output could be devirtualized with CRTP, but for now we don't do this
struct upsample_nearest1d_cuda_functional final : public upsample_nearest1d_cuda {
  void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options, DimNameList names) override {
    outputs_[output_idx] = at::native::empty_strided(sizes, strides, options);
    if (!names.empty()) namedinference::propagate_names(outputs_[output_idx], names);
  }
  std::array<Tensor, 1> outputs_;
};

Tensor upsample_nearest1d_cuda(const Tensor& self, IntArrayRef output_size, optional<double> scales) {
  CUDADeviceGuard g(self.device());
  upsample_nearest1d_cuda_functional op;
  op.meta(self, output_size, scales);
  op.impl(op.outputs_[0], self, output_size, scales);
  return std::move(op.output_[0]);
}

// out-place implementation

struct upsample_nearest1d_cuda_out final : public upsample_nearest1d_cuda {
  upsample_nearest1d_cuda_out(const Tensor& out)
    : outputs_{std::ref(out)} {}
  }
  void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
    TORCH_CHECK(outputs_[output_idx].options() == options);
    at::native::cuda::resize_(outputs_[output_idx], sizes);
    at::native::cuda::as_strided_(outputs_[output_idx], strides);
    if (!names.empty()) namedinference::propagate_names(outputs_[output_idx], names);
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

Tensor& upsample_nearest1d_out_cuda(Tensor& result, const Tensor& self, IntArrayRef output_size, optional<double> scales) {
  // In event of multiple tensor arguments, code generation should
  // be responsible for making sure all devices are consistent
  CUDADeviceGuard g(self.device());
  upsample_nearest1d_cuda_out op(result);
  op.meta(self, output_size, scales);
  op.impl(result, self, output_size, scales);
  // Add this if version bumping happens here
  // increment_version(result);
  return result;
}

// CPU follows similarly

} // namespace native
```

The key idea is we use object oriented programming to factor the
boilerplate into several parts (the user-provided `meta` and `impl` definitions,
as well as the framework-provided `set_output` helper) which we then specialize
for each variation (functional/out/inplace) of the kernel that
we need to generate.  Here is the step-by-step:

1. At the top of the inheritance hierarchy is `MetaBase`, which defines
   `set_output` virtual method that will be varied depending on which
   variant of the operator we're defining.  In a functional kernel, this
   method is overridden to actually allocate the output tensor; in an
   out-of-place kernel, this kernel only resizes the pre-existing
   output.

2. `meta::upsample_nearest1d` inherits from `MetaBase`; there is one
   per structured function group.  The user defines the `meta` method on
   this class. This method does general shape checking work and
   eventually makes a call to the virtual `set_output` which specifies
   what the output shape should be (still unspecified!)

3. For each device type to be implemented, we extend the meta class
   into a class with a user-defined `impl` method that says how to
   actually do the kernel computation for that device.  This method is
   assumed to take an `out` tensor that has been appropriately sized and
   placed on the correct device.  Because `impl` is a method, users
   of the scaffolding get nice error messages when they write a
   method implementation whose C++ type doesn't match the generated
   header (this is in contrast to the previous approach of generating
   function prototypes, which result in a linker error if you get
   the signature wrong).

4. Finally, for each variant of the function we need (functional,
   out-of-place, inplace), we extend one last time to provide the
   correct override implementation of `set_output`.

5. In the final kernel function we register for operators, we construct
   one of these classes, call its meta and impl methods, and then
   finally return the output tensor in an appropriate way.  These
   functions also take care of other boilerplate operation, such as
   setting up device guards, version counter bumps, etc.

The boilerplate here is written very carefully for performance:

* In all of the places where we call other operators, we bypass
  dispatcher, instead directly calling to their native implementations.
  This is feasible because we generate code for CPU/CUDA implementations
  separately, so we can directly code in the correct location.

* The use of `set_output` as a method means we can avoid allocating an
  owning vector to store sizes; instead, initializer lists can be used
  whenever the size is statically known.

* `set_output` is virtual.  This is a tradeoff between code size
  and devirtualization: by making `set_output` virtual, we can
  reuse the generated code for a single meta function across all
  variations of a function; it is not too difficult to devirtualize
  with CRTP, but we don't expect there to be many benefits to
  inlining `set_output`, as `at::native::empty` won't inline.

* The `outputs_` field is a statically-sized array, rather than a
  vector, because we generate the code per operator and thus
  know at compile time how many outputs there are.

* `set_output` doesn't return a reference to the `Tensor` that was
  freshly allocated.  This is to avoid users from doing an antipattern
  where they first allocate a tensor, and then restride it in a
  subsequent call.  The intention is that (eventually) these functions
  can do all of the allocation and striding in one go, without
  redundancy.  (This also applies to setting names!)  This means
  that we may provide multiple `set_output` overloads for handling
  various situations.

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

struct upsample_nearest1d_common final : public meta::upsample_nearest1d {
  void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options, DimnameList names) override {
    outputs_[output_idx] = at::empty_strided(sizes, strides, options, names);  // go via dispatcher
    if (!names.empty()) namedinference::propagate_names(outputs_[output_idx], names);
  }
  std::array<Tensor, 1> outputs_;
};

Tensor upsample_nearest1d_common(const Tensor& self, IntArrayRef output_size, optional<double> scales) {
  // TODO: RecordFunction could be added here, if desired
  DeviceGuard g(self.device());
  upsample_nearest1d_common op;
  op.meta(self, output_size, scales);
  ExcludeDispatchKeyGuard g2(DispatchKey::Common);
  // Notice that upsample_nearest1d is ignored here. It may be a good idea
  // to skip this implementation, or use a slightly different variant, if
  // a backend explicitly registered upsample_nearest1d
  at::upsample_nearest1d_out(op.outputs_[0], self, output_size, scales);
  return std::move(op.tensor_);
}

TORCH_LIBRARY_IMPL(aten, Common, m) {
  m.impl("upsample_nearest1d", upsample_nearest1d_common);
}

// out variant proceeds similarly; with a dispatched resize_ call in set_output
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
struct upsample_nearest1d_meta final : public meta::upsample_nearest1d {
  void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options, DimnameList names) override {
    outputs_[output_idx] = at::native::empty_strided_meta(sizes, strides, options);
    if (!names.empty()) namedinference::propagate_names(outputs_[output_idx], names);
  }
  std:array<Tensor, 1> outputs_;
};

Tensor upsample_nearest1d_meta(const Tensor& self, IntArrayRef output_size, optional<double> scales) {
  upsample_nearest1d_meta op;
  op.meta(self, output_size, scales);
  return std::move(op.outputs_[0]);
}

TORCH_LIBRARY_IMPL(aten, Meta, m) {
  m.impl("upsample_nearest1d", upsample_nearest1d_meta);
}

// out variant proceeds similarly; with a meta::resize_ call in set_output
```

Discussion:

* Why not just use the Common dispatch key for everything?

    * Performance reasons. Introducing the common key would induce an
      extra redispatch, which at time of writing would give up quite a
      bit of performance due to dispatch overhead, for no particularly
      good reason.

* I don’t like codegen.

    * An earlier version of this proposal had the boilerplate
      generated using C++ templates rather than codegen. However, we
      think the formulation in this proposal is superior under the
      constraint that mobile selective build must keep working, as we
      cannot directly write registrations in source files, and so we
      must intermediate between the structured and non-structured
      variants.

## Handling TensorIterator

TensorIterator accounts for a third of operators in PyTorch,
and is characterized by a class which computes a lot of metadata
and carries out allocation.  Previous iterations of the structured
kernel design struggled to account for this style of kernel writing in a
clean way, without requiring major rewrite TensorIterator.

This class based design permits TensorIterator to work, by
making TensorIterator itself a subclass of MetaBase.  The modified
class hierarchy now looks like this:

```
struct TensorIteratorBase : public MetaBase;
struct TensorIterator : public TensorIteratorBase;

namespace meta {
  struct add : public TensorIteratorBase;
}
```

TensorIterator itself remains an implementation of the old-style API for
kernels that are not yet ported to structured kernels.
TensorIteratorBase contains the bulk of the implementation, but all
places that previously allocated tensors now call `set_output`:

```
  // allocate memory for output, memory format depends on setup_type
  switch (setup_type) {
    case FastSetupType::CONTIGUOUS:
      {
        for (int i = 0; i < num_outputs_; i++){
          auto& op = operands_[i];
          /* BEFORE:
          if (!op.tensor.defined()) {
            TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
            op.tensor = at::empty(shape_, op.options(), MemoryFormat::Contiguous);
            op.current_dtype = op.target_dtype;
          } else if (op.will_resize) {
            at::native::resize_output(op.tensor, shape_);
          }
          */
          // AFTER:
          set_output(i, shape_, {}, op.options(), names_);
        }
        break;
      }
```

TensorIterator defines an override of `set_output` that recovers the old
behavior, while structured kernel subclasses override `set_output` in
the same way as before.

The code generation only requires a very modest extension: an `inherits`
field that lets you replace `MetaBase` with your own custom base
implementation class:

```
- func: add.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  inherits: TensorIteratorBase  # [NEW]
  dispatch:
    CPU: upsample_nearest1d_structured_cpu
    CUDA: upsample_nearest1d_structured_cuda
```

Now you can simply construct it appropriately in your function
definitions:

```
TORCH_META_FN(add) (
  const Tensor& self, const Tensor& other)
) {
  // Call method on TensorIteratorBase to actually build the struct
  build(...config..., {self, other});
  // TensorIteratorBase itself will call back to set_output when
  // it tries to do output allocation.  It can also save a pointer
  // to the output for itself
}

namespace native {
  TORCH_IMPL_FUNC(add_cpu) (
    const Tensor& out, const Tensor& self, Tensor& other
  ) {
    add_stub(device_type(), *this);
  }
}
