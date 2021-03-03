# TensorRef abstraction in C++

This proposal introduces a new class `TensorRef`, to replace all
places in our codebase where we currently use `const Tensor&`.  The
distinguishing characteristics of this class are:

1. It is non-owning
2. It is as safe as other by-value reference types (like `c10::ArrayRef` or
   `std::string_view`)
3. It is implicitly convertible (with some exceptions) to `const Tensor&`
   (i.e., it can be introduced incrementally)

## Motivation

* In order to call a function that accepts `const Tensor&`, you must
  somehow have a `Tensor` at hand.  However, `Tensor` is always an
  owning value type, and sometimes you only have a non-owning reference
  to a Tensor, but not a reference to Tensor itself (e.g.,
  `TensorImpl*`).  In these cases, you cannot conveniently call the
  function without first inducing a refcount bump (to establish an
  owning `Tensor` object, to pass by reference).

* For performance reasons, IValue `toTensor` returns a `const Tensor&`.
  For this to be possible, IValue must internally store a `Tensor`
  object (otherwise, there is nothing to point the reference to.)  This
  induces a dependency from IValue header to Tensor header, which can
  cause problems in some situations (as Tensor class has a large API and
  thus has many dependencies on many other classes.)

* `const Tensor&` implies a double indirection, as `Tensor` is itself
  already a pointer.  A TensorRef type implemented internally as a
  `TensorImpl*` would eliminate this double indirection.  (NB: The
  proposal below doesn't actually solve this problem, due to the definition
  of Itanium ABI.)

## Proposal

Add the following class to PyTorch:

```
struct TensorRef {
    Tensor base_;
    TensorRef(TensorImpl* impl) : base_(TensorPtr::reclaim(impl)) {}
    ~TensorRef() {
        base_.impl_.release();
    }
    operator const Tensor& () & {
        return base_;
    }
    // Generated copy of all Tensor methods, forwarding to base_
    // (with a hypothetical C++ language extension for operator dot,
    // would be possible to do this without codegen)
};
```

Main points:

* This class is implemented as containing an actual Tensor object, but
  we unsafely use reclaim/release to ensure that no refcount bump occurs
  on construction/destruction of the object.  We verified with Godbolt
  that the compiler is able to eliminate the `base_` destructor.  This
  makes it possible to take out a `const Tensor&`, which makes it easier
  to incrementally convert code to use TensorRef (given a TensorRef, you
  can always convert it back into `const Tensor&` if you need to call a
  function that takes the legacy type).

  The downside to this implementation strategy is that TensorRef is
  not a trivial object (even though, after optimizations, the
  constructor and destructor are in fact trivial), so that
  per the Itanium C++ ABI, we must still push this object to the stack
  and pass it by reference.  So we don't actually manage to eliminate
  a double indirection here; in fact, if you take a TensorRef and pass
  it on by value, the generated code will be worse (as each copy
  construction will push another copy of TensorRef onto the stack).
  This problem can be fixed by reimplementing TensorRef as directly
  holding onto a `TensorImpl*`, but then the `const Tensor&` conversion
  is unimplementable.  Note that if TensorRef is trivial, methods
  must be implemented carefully to inline away all class calls (e.g.,
  by delegating onto a out-of-line non-method function), otherwise
  the same Itanium ABI requirement will force us to put the TensorRef
  onto the stack.

* Implicit conversion to `const Tensor&` on temporaries is unsafe,
  because const reference lifetime extension doesn't apply.  Thus, the
  ref qualifier `&` ensures that this conversion is not permitted on
  temporaries.  In standard usage, `TensorRef` will be an `lvalue`
  (e.g., passed in as an argument) and the implicit conversion should
  always be valid.

* Due to the above, if IValue is adjusted to no longer hold onto a
  Tensor, then IValue `toTensor` is obligated to return a `TensorRef`
  by value (because there is no non-temporary `TensorRef` and therefore
  we cannot manufacture a `const Tensor&`.)  This means that some code
  that was previously written assuming a `const Tensor&` return may
  no longer work (e.g., because the ref qualifier rules out the implicit
  conversion).  However, the author of this RFC does not believe it
  is possible to introduce unsafety this way.

## Alternatives

* Presently, IValue produces a `const Tensor&` in its interface by
  directly storing a `Tensor`.  It may be possible to solve include
  cycles involving IValue in other ways; in particular, by avoiding
  use of IValue in Tensor APIs, or by avoiding use of IValue in
  arguments that are defaulted (defaulting can always be reimplemented
  with N overloads, one per default argument.  This is not a huge burden
  in the presence of code generation).

## See also

* FB Only: [We shouldn't feel bad about passing Tensor by reference](https://fb.workplace.com/groups/pytorch.dev/permalink/801504910427991)
