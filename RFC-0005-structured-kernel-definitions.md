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

* (Under question) **Unify with version counter bumps and view metadata tracking**: these
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

```cpp
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
    const Tensor& self, IntArrayRef output_size, optional<double> scales, const Tensor& out
  );
  /* macro expands to: void upsample_nearest1d_structured_cuda::impl( */
  TORCH_IMPL_FUNC(upsample_nearest1d_structured_cuda) (
    const Tensor& self, IntArrayRef output_size, optional<double> scales, const Tensor& out
  );
}
```

The code generator then generates the boilerplate code to put these
functions together.  The boilerplate is somewhat involved, and we will
explain it below.

```cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Common code
// Abridged from aten/src/ATen/TensorMeta.h
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

#define TORCH_META_FUNC(name) void structured_##name::meta
#define TORCH_META_FUNC2(name, overload) void structured_##name##_##overload::meta
#define TORCH_IMPL_FUNC(name) void structured_##name::impl

// Parent class for all code-generated meta:: classes
struct MetaBase {
  // TODO: Maybe some of these should be optional, but in many cases they
  // be implicitly made optional by passing an empty list
  virtual void set_output(int64_t output_idx, IntArrayRef size, IntArrayRef strides, TensorOptions options, DimnameList names) = 0;
  // Returns a reference to an undefined tensor if no output is
  // available
  virtual const Tensor& maybe_get_output(int64_t output_idx) = 0;

  // Convenience helpers
  void set_output(IntArrayRef size, TensorOptions options) { set_output(0, size, {}, options, {}); }
  const Tensor& maybe_get_output() { return maybe_get_output(0); }
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Code generated per operator
// Generated to build/aten/src/ATen/MetaFunctions.h
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

namespace meta {

struct structured_upsample_nearest1d : public MetaBase {
  void meta(const Tensor& self, IntArrayRef output_size, optional<double> scales); // user defined
};

} // namespace meta

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
// Code generated per dispatch table entry for operator
// Generated to, e.g., build/aten/src/ATen/RegisterCUDA.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

namespace native {

struct structured_upsample_nearest1d_cuda : public meta::upsample_nearest1d {
  void impl(const Tensor& self, IntArrayRef output_size, optional<double> scales); // user defined
};

// functional implementation

// NB: set_output could be devirtualized with CRTP, but for now we don't do this
struct structured_upsample_nearest1d_cuda_functional final : public structured_upsample_nearest1d_cuda {
  void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options, DimNameList names) override {
    outputs_[output_idx] = at::native::empty_strided(sizes, strides, options);
    if (!names.empty()) namedinference::propagate_names(outputs_[output_idx], names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx];
  }
  std::array<Tensor, 1> outputs_;
};

Tensor structured_upsample_nearest1d_cuda(const Tensor& self, IntArrayRef output_size, optional<double> scales) {
  CUDADeviceGuard g(self.device());
  upsample_nearest1d_cuda_functional op;
  op.meta(self, output_size, scales);
  op.impl(self, output_size, scales, op.outputs_[0]);
  return std::move(op.outputs_[0]);
}

// out-place implementation

struct structured_upsample_nearest1d_cuda_out final : public structured_upsample_nearest1d_cuda {
  upsample_nearest1d_cuda_out(const Tensor& out) : outputs_{std::ref(out)} {}
  void set_output(int64_t output_idx, IntArrayRef sizes, IntArrayRef strides, TensorOptions options) override {
    at::native::resize_output(outputs_[output_idx], sizes);
    if (!strides.empty()) {
        TORCH_INTERNAL_ASSERT(!options.memory_format_opt().has_value());
        outputs_[output_idx].get().as_strided_(sizes, strides);
    } else if (options.memory_format_opt().has_value()) {
        outputs_[output_idx].get().unsafeGetTensorImpl()->empty_tensor_restride(*options.memory_format_opt());
    }
    if (!names.empty()) namedinference::propagate_names(outputs_[output_idx], names);
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
};

Tensor& structured_upsample_nearest1d_out_cuda(Tensor& out, const Tensor& self, IntArrayRef output_size, optional<double> scales) {
  // In event of multiple tensor arguments, code generation should
  // be responsible for making sure all devices are consistent
  CUDADeviceGuard g(self.device());
  upsample_nearest1d_cuda_out op(out);
  op.meta(self, output_size, scales);
  op.impl(out, self, output_size, scales);
  return out;
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

* Because we generate code separately for CPU/CUDA, we can bypass
  the dispatcher entirely. In the current implementation, we don't
  do this, but the optimization opportunity is available.

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

```cpp
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

```cpp
TORCH_LIBRARY_IMPL(aten, CommonXLA, m) {
  m.impl("upsample_nearest1d", CppFunction::makeFallthrough);
}
```

Finally, structured kernels also implicitly register an implementation
for the Meta key, which is the API for dry running shape/dtype
calculations without any kernels involved:

```cpp
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

```cpp
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

```cpp
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

The code generation only requires a very modest extension: an `structured_inherits`
field that lets you replace `MetaBase` with your own custom base
implementation class:

```
- func: add.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase  # [NEW]
  dispatch:
    CPU: upsample_nearest1d_structured_cpu
    CUDA: upsample_nearest1d_structured_cuda
```

Now you can simply construct it appropriately in your function
definitions:

```cpp
TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, Scalar alpha
) {
  build_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

TORCH_IMPL_FUNC(add_out) (
  Tensor& result, const Tensor& self, const Tensor& other, Scalar alpha
) {
  add_stub(device_type(), *this, alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == output().dtype());
}
```

In the case of TensorIterator, all of the arguments are stored in the
struct at construction time, so the impl func doesn't need to make
use of any of the tensors.  Additionally, when `set_output` is invoked
in the meta function, after doing all appropriate allocations, it will
delegate to the underlying `set_output` in TensorIterator, letting it
query with `maybe_get_output()` to register any outputs necessary.

## Common error checking code

Some operators do not have out variants, and thus do not make sense
as structured kernels per se (i.e., cannot do boilerplate reduction
between out and functional implementations).  However, these operators
still require meta implementations, and can still usefully have
error checking code that is shared.

TODO WRITE MORE

## Piggybacking other improvements

Ports to structured kernels must be done individually by hand.  Because
the porting process is already labor intensive, we should do other
improvements "while the patient is open".  Here are some candidate
improvements which should consider applying

### Removing mutable references from out arguments

Historically, out and inplace variants of functions take a `Tensor&`
rather than a `const Tensor&`.  This convention has lead to no end
of confusion for kernel writers, who incorrectly surmise that given
a mutable reference `Tensor& out`, one may assign the output by
writing `out = ... some expression ...`  (this doesn't work).

The absence of the const modifier is currently relied upon by
template metaprogramming machinery (to detect if arguments are out
tensors or not); however, because the implementations of structured
kernels are a layer below the operator registration layer, the
const modifier can be eliminated from the `TORCH_IMPL_FUNC` API
without requiring the rest of the system to be updated.

One implication of this change is that the out parameter cannot be
easily passed to existing public API that requires a mutable reference.
This can be easily remedied by updating the existing APIs to accept
const references and not only mutable references, or, if truly
necessary, using a `const_cast` to get out of jail free.

**Mutable reference removal has landed.**

### Type refinement

Once we have dispatched to a CPU kernel, we know that the tensor in
question is in fact a CPU tensor, and not (for example) a CUDA tensor.
However, this information is not retained inside of the body of the
kernel, and so if a user makes a method call or regular `at::` namespace
function call, the dispatcher still must inspect the type tag to
rediscover, yes, indeed, we still have a CPU tensor.

One promising approach to solving this problem is to refine the type of
a tensor from `Tensor` to `CPUTensor`, where a `CPUTensor` represents a
tensor that is statically known to be a CPU Tensor.  Operations
(functions and methods) on `CPUTensor` bypass dispatching and go
directly to the CPU implementations in question.  `const CPUTensor&` can
be defined to implicitly convert into `const Tensor&`, which means
existing APIs that don't know how to short circuit can continue to do
pre-existing behavior.

The primary consequence of making this change immediately is we must
immediately create a CPUTensor class with enough methods to cover the
usual surface area (even if those methods don't apply any performance
optimization).  With code generation this should not be too much code.
This would also require the creation of a CPUTensorRef class to ensure
that CPUTensors can be created from `const Tensor&` without incurring
a reference count bump).

One question is whether or not the existence of CPUTensor means we should
eliminate the `at::cpu::` namespace (as they serve near equivalent purposes;
if you have functions which support CPUTensor, simply (unsafely) cast
your Tensor to a CPUTensor and then utilize the regular API).  One
possible argument for retaining the `at::cpu::` namespace is that these
functions are guaranteed to bypass dispatching, whereas other functions
may implicitly downcast to `Tensor` and do an optimized call.

**Type refinement HAS NOT landed.**

## Long term status of unstructured kernels

Structured operators are currently strictly defined in terms of an out
operation.  However, there are some operators in PyTorch which do not
have out variants, because they typically don't make sense.  Some of the
most notable operator types of this form:

* View operations
* Factory functions
* `copy_`

Since non-structured kernels simply operate at a lower level of
abstraction, in principle, it is not a big deal if some operators
never become structured; to make an analogy, sometimes you have to
write assembly, and as long as it is not too frequent, there is not
too much to be gained from trying to extend the functionality of your
system to expunge these entirely.

However, there is one practical problem with doing this: we continue
to have separate code generation paths for structured and unstructured
kernels, and in some cases, there are improvements that could be
profitably applied to both structured and unstructured kernels (for
example, elimination of mutable `Tensor&` references).  For some
improvements, the correct answer to "I want my operator to have this
improvement" is "port it to structured kernels".  However, in the cases
where this is not possible, there must be some alternate recourse.

Here are the list of planned improvements to structured kernels which
also should be equivalently applied to unstructured kernels:

* Generation of `at::cpu::` stubs for static runtime.  Suggested
  resolution: implement for unstructured as well.  **This has
  LANDED.**

* Removal of mutable references. Suggested resolution: don't bother
  fixing in the unstructured case (until someone decides to purge
  mutable references from the public API.  Which, let's be honest,
  probably isn't going to happen).

## How to get involved with structured kernels

There's still a lot of work to be done in structured kernels!  Here
are some of the things to be done.

Folks who have expressed interest in helping: @ailzhang, @bdhirsh,
@hameerabbasi

### Port kernels to be structured

Kernels vary wildly in difficulty with regards to how difficult they are
to port, but at least a substantial chunk of operators should be
possible to port to structured without too much difficulty.

Ailing has graciously attempted to port a kernel, and made some
observations about things you might have to do when porting:

1. Every kernel in the operator must be c10 full.  If something is
   using hacky wrapper, port it to stop using hacky wrapper (usually
   by reordering arguments) first.

2. Don't accidentally remove the dispatch table for the out kernel,
   you still need that one!

3. Change all `Tensor&` arguments to `const Tensor&` (as structured
   kernels do not take mutable references).  If possible, change all the
   helper functions the kernel uses to also use `const Tensor&` (most
   commonly, you will need to change `DispatchStub` signatures).  If you
   can't conveniently change an API because it would have large knock
   on effects, use a `const_cast<Tensor&>` and mark it with a TODO.

There are some kernels which should be easier to port to structured.
[This post](http://blog.ezyang.com/2020/05/a-brief-taxonomy-of-pytorch-operators-by-shape-behavior/)
taxonomizes operators; we have working examples of non-reduction
TensorIterator kernels and Fixed kernels, and those should work out
without too many hiccups.  For example, try finishing the rest of the
upsample kernels.

Things that are known not to work:

* Operators that call a lot of other operators (even if they are not
  technically composite).  In these situations, there may not be any
  place where shape computation is actually done in the old kernel, so
  you would have to reconstruct this logic from scratch.  Many linear
  algebra kernels fall in this bucket.

* Reductions.  These use the TensorIterator API in a way that we have
  trouble supporting today (they allocate output outside of
  TensorIterator and then "force" TensorIterator to not resize in
  shape computation).

* Kernels that directly overwrite `Tensor&` argument (reductions are
  known to do this!)  Then again, these kernels are just WRONG and
  should be fixed.

What operators to prioritize?  Consider picking some important model
(James Reed and Yinghai Lu may have some suggestions) and getting to
100% coverage there.

### Get tracing with meta tensors working

Right now, `torch.jit.trace` requires you to provide real tensors,
because it's the only way to get accurate shape tracking.  Meta tensors,
which take advantage of structured kernels, allow for JAX style tracing
where you can feed in tracers that have shape but no data, and do fast
tracing on the fly.

Most of the pieces to make this work should exist, it's just a matter of
putting it all together.
