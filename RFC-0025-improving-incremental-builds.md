# Improving incremental builds
**Authors:**
@peterbell10

## **Summary**

This RFC proposes changes to ATen that will allow more granular header
dependencies and tools to enforce their usage across the codebase.
This should greatly improve incremental and cached build performance.

## **Motivation**

In ATen, certain generated headers files include the definitions of
every operator in PyTorch. Consequently any new operator, or update
to any of ATen's over 1000 existing operators in
`native_functions.yaml`, requires a full rebuild of ATen and torch.
These problematic headers include `Tensor.h`, `Functions.h`,
`NativeFunctions.h`, which are all widely used and currently
unavoidable.

For example, adding a new operator that isn't referenced anywhere
results in 1000+ files being rebuilt over many hours of CPU time
(single threaded). After the proposed changes you can expect at most a
small handful of files to be rebuilt, most of which should contain
code generated specifically for that new operator.

## **Proposed Implementation**

This RFC proposes two changes to allow breaking up these header
dependencies, and avoid unnecessary header dependencies in ATen.

1. Add a base class to `Tensor` called `TensorBase` which has no
   method operators but is otherwise fully interoperable. This is
   particularly useful for kernel code which often doesn't call any
   operators but still needs to pass `Tensor`s around.
2. Split code generated header files up on a per-operator basis, so
   only necessary operators need to be included. These headers still
   depend on `Tensor.h` so will depend on method operators.

Additionally, two macros are introduced which can be defined to
enforce strict adoption of these more granular headers:

- `TORCH_ASSERT_NO_OPERATORS` when defined, will make inclusion of any
  code generated from the operator list into a compiler error. In
  practice this limits you to only use `TensorBase` and `TensorIterator`.
- `TORCH_ASSERT_ONLY_METHOD_OPERATORS` which allows `Tensor.h` to be
  included, but not the other umbrella headers. Instead specific
  operators must be included directly as they are used.

These will be implemented by inserting preprocessor guards into the
relevant header files, with `ATen/core/TensorBody.h` as an example
```cpp
#ifdef TORCH_ASSERT_NO_OPERATORS
#error This change adds a dependency on native_functions.yaml,            \
  meaning the file will need to be re-compiled every time an operator     \
  is changed or added. Consider if your change would be better placed in  \
  another file, or if a more specific header might achieve the same goal. \
  See NOTE: [Tensor vs. TensorBase]
#endif
```

This assists in first detecting problematic includes, even when
buried in layers of transitive includes. Then further with
maintaining the header hygiene into the future, as developers get
immediate feedback when undesirable headers are added in the future.

### TensorBase
`Tensor` serves two related but separate functions. It is both a
user-friendly handle to the underlying `TensorImpl` data structure,
and an interface to call operators with the `variant: method`
property. `TensorBase` acts only as a handle without any of the
code-generated methods and lives in it's own header
(`ATen/core/TensorBase.h`).

`Tensor` inherits publicly from `TensorBase` to make `TensorBase&`
arguments implicitly callable with `Tensor` without touching the
reference count. Similarly, an implicit move constructor
`Tensor(TensorBase &&)` allows `Tensor` to be implicitly created from
`TensorBase` so long as there is no reference count change. This way
the two types cleanly inter-operate without silently degrading
performance with reference count increments.

The primary use case for `TensorBase` is kernel code which is often
the most expensive to compile yet often don't reference operators.
Many kernels simply hand tensors off to `TensorIterator` or call
`data_ptr<>` and access the data directly. To enable this use case,
`TensorIterator.h` and the `cpu/Loops.h` and `cuda/Loops.cuh` headers
must be kept clean of any `Tensor` includes (forward declarations are
fine though).


### Per-Operator Headers

The contents of code-generated "umbrella" headers will be split up
into separate headers so all they do is include from one header per
operator. The actual generated code will not be changed, only moved to
a new header. These new headers will live in the `ATen/ops` folder.
`at::add` for example will generate 8 new headers,

```cpp
#include <ATen/ops/add.h>        // Split from Functions.h
#include <ATen/ops/add_meta.h>   // Split from NativeMetaFunctions.h
#include <ATen/ops/add_native.h> // Split from NativeFunctions.h
#include <ATen/ops/add_ops.h>    // Split from Operators.h

// Plus static dispatch functions split from {DispatchKey}Functions.h
#include <ATen/ops/add_compositeexplicitautograd_dispatch.h>
#include <ATen/ops/add_cpu_dispatch.h>
#include <ATen/ops/add_cuda_dispatch.h>
#include <ATen/ops/add_meta_dispatch.h>
```

Note that `add` has 4 overloads which are included in the same header
files. This way the caller doesn't need to know which overload is
relevant to their use case, only the basic operator name.

Unfortunately, per-operator headers aren't compatible with Meta's
internal `buck` builds because the codegen output list depends
dynamically on the contents of an input file (namely
`native_functions.yaml`). So these headers can only be included when
`AT_PER_OPERATOR_HEADERS` is defined which is the default for CMake
and bazel builds. So, they must be included behind an `#ifdef` guard,
e.g.

```cpp
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/add.h>
#include <ATen/ops/sum.h>
#endif
```

### Enforcing adoption
These tools can only improve incremental builds if they are used
widely throughout the codebase and continue to be enforced as the code
develops. So, adoption will be enforced through the
`TORCH_ASSERT_NO_OPERATORS` and `TORCH_ASSERT_ONLY_METHOD_OPERATORS`
macros.

First, `ATen` and `torch` will be migrated to using the per-operator
headers, defining `TORCH_ASSERT_ONLY_METHOD_OPERATORS` in each file as
they are updated. Then, once most of the codebase is compliant, the
build system will be changed to define
`TORCH_ASSERT_ONLY_METHOD_OPERATORS` by default in all new files. In
the rare occasion where umbrella headers are strictly necessary, these
files can opt-out by explicitly `#undef`ing the macro making it
`grep`-able and clearly visible in code-review. One such example is
`autograd/generated/Functions.cpp` which includes verbatim code from
`derivatives.yaml` so doesn't know which operators are being called.
This should be extremely rare though.

The initial migration can be assisted by
[`include-what-you-use`](https://include-what-you-use.org/) which is a
clang-based tool to find the necessary headers based on what symbols
are actually used in a file. For most cpp files, it is as simple as
copying the `<ATen/ops/*>` headers from the tool output.

`TORCH_ASSERT_NO_OPERATORS` cannot be enforced as strictly because
compliance often requires significantly restructuring the code.
However, where the cost-benefit is favorable, this macro should be
defined manually in the source file. This will include files that
already use `TensorIterator` exclusively and just need their includes
pruned; and more costly to compile files where taking time to
restructure them is worthwhile.

## **Metrics**

Some key metrics that should be improved by these changes:
- Time to rebuild after adding editing a functional operator
- Time to rebuild after adding editing a method operator
- `sccache` miss rate in open source CI

For example, we can compare against a branch with the
`ATen/native/cpu` and `ATen/native/cuda` folders migrated to
per-operator headers, and some key files edited to use `TensorBase`.
Despite only partial migration, the table below shows ~2x speedup in
incremental build for functional operators and ~1.4x for method
operators.

| Variant  | Jobs (before) | Jobs (after) | Time (before) | Time (after) |
|----------|:-------------:|--------------|---------------|--------------|
| Function |      1403     |      775     |   10 m 00 s   |    4 m 50 s  |
| Method   |      1403     |     1184     |   10 m 00 s   |    7 m 15 s  |

We can expect even further improvements once all of ATen and torch
have adopted per-operator headers.

## **Drawbacks**
There is no denying that it's more convenient to just include
`<ATen/ATen.h>` at the top of the file and not have to think about it.
This will require additional effort on the part of developers to
maintain the list of operator includes.

C++ users who include internal header files without including
`ATen/ATen.h` may also find their compilation failing as unnecessary
includes are trimmed from internal headers.

## **Alternatives**
[pytorch/pytorch#61915](https://github.com/pytorch/pytorch/pull/61915)
proposed generating new operators into separate headers called
`*_Delta.h` (e.g. `Functions_Delta.h`) during local development. This
improves incremental builds locally with comparatively fewer changes
to the ATen codebase. However,

1. It only helps when adding operators, not changing existing ones
2. It does nothing for clean builds using a cache (like in CI)
3. Incremental builds no long reproduce a from-scratch build

[pytorch/pytorch#62258](https://github.com/pytorch/pytorch/issues/62258)
proposed removing operator header dependencies from CUDA kernels
specifically. This is aligned with the aims of `TensorBase`, but
leaves out several hundred non-kernel source files. Individually these
files take around 10 seconds each to compile, but together account for
2-3 hours of (serial) CPU time.


## **Prior Art**
Not aware of any relevant prior art.

## **How we teach this**
This change doesn't effect end-users.

For PyTorch developers, a developer wiki entry should outline the
motivation from this RFC and describe how to write code in compliance
with the two enforcement macros:
- How to include the new per-operator headers
- Using the include-what-you-use tool
- Structuring headers to minimize unnecessary dependencies
- Modifying kernels to be `TORCH_ASSERT_NO_OPERATORS` compliant

This wiki entry should also be linked from the enforcement macro error
messages to increase visibility.

## **Unresolved questions**
- Is it possible/desirable to enforce `TORCH_ASSERT_NO_OPERATORS` automatically?
  e.g. for all `.cu` files, or all files over a certain compile time.
- Can `include-what-you-use` completely automate operator includes?
  Tools exist for strictly managing _all_ includes, but that would be
  a significant change from existing include style.
- Is it worth adopting `TensorBase` more widely than just kernels?
  e.g. if the operators themselves used `TensorBase` then you no
  longer have to include hundreds of method operators.
