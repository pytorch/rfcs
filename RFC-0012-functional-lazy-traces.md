# Functional lazy traces (from XLA) to PyTorch

**tl;dr** XLA contains a non-XLA specific TorchScript-like IR
specifically designed for lazy evaluation. Nodes in this IR correspond
to ATen operations but with mutation complete eliminated at tracing time
(compared to the incomplete remove mutation pass) and with a distinct
class per operator node (compared to TorchScript IR which is dynamically
typed). Alex Suhan makes the case that splitting out this IR from XLA
and publishing it separately is the fastest way to bring accelerators
which need lazy tensors online; Edward Yang asks PyTorch composability
and JIT teams to think about what the best path for this IR to core
PyTorch is.

## Background

PyTorch XLA allows PyTorch programs to be run by XLA (and most
importantly, on TPUs) by way of a lazy tensor abstraction. PyTorch
models executing on the XLA device lazily trace execution until XLA
requires materialization, at which point the set of operations that were
run are lowered into XLA HLO IR and then run by XLA itself. Although the
integration with XLA is imperfect (based on internal work running some
models on TPU), it remains the best and most comprehensive
implementation of lazy tensors to date on PyTorch.

Alex Suhan has identified that much of the infrastructure in torch_XLA
is actually not specific to XLA. In particular, `torch_xla` defines an IR
*in terms of ATen operators* (not XLA operators; see
https://github.com/pytorch/xla/tree/master/torch_xla/csrc/ops ), and
first constructs this IR before lowering into HLO IR. This puts the IR
in the same conceptual space as TorchScript IR, except:

* It is purely functional (no support for mutation or control flow).
  XLA’s tracing process keeps tracks of aliases and views so that it can
  extend the IR at all points where a mutation is visible. This is
  contrast to the JIT remove mutation pass, which may refuse to remove
  mutation when it cannot prove it is sound
  (https://github.com/pytorch/pytorch/issues/39362), and the ONNX
  functionalization option for tracing, which simply hopes that
  mutations don’t being visible at other relevant aliases (not always
  true, see https://github.com/pytorch/pytorch/issues/34538)

* It is specifically designed to be traced every iteration your model is
  run, so the IR is designed to be more efficient for this use case:
  e.g., it has separate classes for every ATen operator, rather than a
  dynamically typed dictionary of attributes; it maintains a hash of an
  IR node [as it is being constructed](https://github.com/pytorch/xla/blob/08ae1044c2a7e314895f9946104cbe399e096515/torch_xla/csrc/ir.cpp#L149).

* The IR is modestly normalized; for example, optionality is removed in
  some cases (an optional min value is replaced with an explicit number
  computed via dtype); some operators are decomposed into a more
  elementary form (addcmul). The IR also serves as a layer of insulation
  between PyTorch’s operator bindings (which change frequently with no
  BC guarantees) and XLA’s lowering (e.g., in
  https://github.com/pytorch/xla/pull/2692 a change in operator
  signature resulted in a change in the PyTorch-XLA bridge, but not the
  IR itself).

Alex has been working on a [refactor of torch_xla](https://github.com/fairinternal/nnc_eager/tree/asuhan/lazy_core_only)
which splits out this core functionality into its own library that has
no dependency on TensorFlow/XLA, with the intention of publishing it as
a library that accelerator vendors can use to bootstrap a lazy tensor
that records a functional IR which they can then feed to their personal
IR lowerings. He is also interested in eventually merging this code into
PyTorch proper.

## How to put this in core?

Although this lazy core functionality is smaller than `torch_xla`, it is
still not small in an absolute sense (30k lines of C++ code; though
possibly reducible further), much of it handwritten code for each
operator XLA supports, but some of it also utility functions that
arguably should already be available in PyTorch. Furthermore, it has
some overlap with TorchScript JIT IR (though, as described above, there
are important differences in goals between XLA’s IR and TorchScript’s
IR.) This leads to a number of challenges for putting the generic parts
of XLA in core.

* **How quickly is this needed?** Alex has stated that he would like to
  ship this API to accelerator partners *within H1 2021*; it is
  difficult to imagine being able to deliver anything besides a direct
  drop of this code in that period of time. (Fortunately, it is also OK
  for this code to live out of repository, for the time being.) How to
  *move fast* in getting this code into core? (Is a rewrite necessary?
  Can most code be reused as is?)

* **How to ensure maintainability as new operators are added?** Taken as
  is, `torch_xla` posits that when a new operator is to be added to the
  lazy IR, we must define an entirely new C++ class to represent it and
  write bindings to construct this IR. This is a burden on operator
  writers; so in practice, operators only get added to the IR when XLA
  also adds support for the lowering. Should the IR classes be code
  generated instead? One benefit of the current scheme is that the
  classes can evolve with a higher degree of backwards compatibility
  than PyTorch operator definitions usually abide by.

* **How to resolve tech debt from utility classes/functions in XLA that
  are duplicated in PyTorch?** Did you know torch_xla defines its own
  C++ type to represent devices? (This is because the “TPU” device is
  not representable in PyTorch.) There are probably many examples of
  places in torch_xla where code in PyTorch core could have been
  adjusted to better accommodate torch_xla’s use cases, but instead the
  classes/functions were simply reimplemented in XLA to avoid having to
  get some code into PyTorch core. Should all of this debt be resolved
  before importing XLA into core? Does it matter?

    * ~~absl dependency~~
    * lazy_tensors/
        * Shape (compare with TensorOptions)
        * PrimitiveType (compare with ScalarType)
        * Literal (compare with Scalar)
        * permutation_util.h: utilities for working with permutations
    * lazy_tensor_core/
        * data_ops.h: squeeze/unsqueeze shape computation
        * device.h: reimplementation of device with TPU
        * helpers.h: type promotion logic, dimension manipulation
        * shape_builder.h: shape building TensorIterator style
        * tensor_util.h: stride calculation
        * torch_util.h: deepcopy, numeric tensor utilities

* **How to ensure maintainability of lazy IR (nb: in memory only, no
  serialization, no BC) in parallel with TorchScript IR?** One of the
  biggest maintainability tensions going forward in PyTorch today is
  juggling the intersection of TorchScript, FX and NFC (all of which
  have large similarities, but with different constraints.) Lazy IR
  represents yet another point in the design space where there are a lot
  of similarities to other existing IRs, but where no other existing
  solution is easily usable. Can we support adding a fourth blessed IR?
  How to avoid requiring cross cutting changes making people have to
  modify four distinct subsystems?

Edward’s opinion:

* Directly shipping XLA’s code as is, is the only plausible route to
  shipping in the next three months. (It might be possible to prototype
  within the half a reimplementation of the system from core PyTorch,
  but it seems exceedingly unlikely that you’d be able to achieve
  feature parity in this timeframe).

* We need to take a serious look at the backend backwards compatibility
  problem; it comes up with all of our backend implementors and we need
  some solution that doesn’t require backend implementors to make
  changes whenever we do a client side BC change. Theoretically this
  should be possible (newly added capabilities should simply gracefully
  fail when sent to a backend that doesn’t support them) but this is
  difficult to do directly in C++ (code generation would work).

* Personally, I think that people should try to write their lowerings in
  “finally tagless” style (i.e., bypassing the construction of an
  intermediate PyTorch IR); however, seeing that preexisting XLA still
  found it useful to construct an IR corresponding to PyTorch directly,
  I am not too sure how likely this will actually happen.

* There should be room for more than one IR representation in PyTorch as
  a whole (think micropass compiler, or bytecode versus graph IR); if we
  can avoid having to maintain a large apparatus of code for each IR it
  will make having multiple more tolerable (XLA’s IR is relatively
  simple, as far as things go, in part because it doesn’t really support
  mutations on it). So IMO the main problem is socializing this idea
  with the JIT team (and making sure that the operator story is
  reasonable).

* I don’t know if a rewrite is right or not, but it should be on the
  table, particularly for code generating the IR classes. Utilities can
  probably be refactored incrementally (aka never).

## Appendix: Things that are BC-compat frontend but not backend

One benefit of the hand written class IR is it is more obvious when you
are breaking BC. In native_functions.yaml most authors think about what
it means to break client BC, but not backend BC. Note that many of these
changes are not *morally* BC breaking, but break backends because our
implementation was bad.

* Renaming overloads
* Adding a new optional argument
* Reorganizing which functions implement an overload (e.g., f(int, bool), f(int, int = 0) ~> f(int, bool = false), f(int, int))
* Changing the dispatcher type translation
