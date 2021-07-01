## Overview

Missing/incorrect type annotations is a common reason for failed model scripting. For a long time, the process of mitigating missing/incorrect type annotations has been through trial and error, ie. by repeatedly hitting the wall of various type-checking errors generated from torch.jit.script and fixing them one by one. This is both inefficient and frustrating.

Most importantly this painful process is not necessary at all because TorchScript can simply run the unscripted program and observe the types. Through a set of preliminary experiments (https://fb.workplace.com/notes/yanan-cao/a-study-of-monkeytype-for-torchscript/258930972443146/), we find that by leveraging this technique with the help of existing tools like MonkeyType, torch.jit.script needs much fewer or even no type annotations.

## Background and Motivation

`torch.jit.script`

`torch.jit.script` compiles a PyTorch program by inspecting its Python source code and constructs TorchScript IR that preserves semantics of it. TorchScript IR can then be transformed, packaged and eventually deployed.

TorchScript IR is statically-typed. All values must have a clear type to be meaningfully usable. However, Python language isn’t statically-typed, thus Python source code usually does not contain complete or accurate type annotations simply because they are not necessary for Python runtime.

`torch.jit.script` mostly relies on explicit type annotations like following to figure out types of functions

```def fn(t*: torch.Tensor*, scale*: float*, add_bias*: bool*) *-> torch.**Tensor*:
    ......
```

In absence of type annotations, TorchScript uses a very simple type inference algorithm: everything is torch.Tensor. This is problematic since many common basic types, even int, bool, float, would not be inferred by TorchScript, leading to compilation error. Though we can devise a more sophisticated type inference algorithm, its effect/cost ratio is low given the accompany complexity and risk, more details can be found in https://fb.quip.com/Po95A4UphzUt#aXQACA0jRFs.

Combining all aforementioned factors, TorchScript effectively has a hard requirement for source code to have type annotations in order for compilation to be successful. This requirement, however, poses challenges to users.

## Challenges of Type Annotations

Adding type annotations is challenging for a few reasons :

   * Costly Effort: Users would need to invest plenty of effort into going through the code base and add missing annotations. For large models or models with meta-programming, the effort is especially costly.
   * Knowledge Gap: Users may not know the code base well to tell the correct types. This is especially common for models downloaded from model zoo or a Github repo. In such cases, it is borderline impossible for users to annotate needed types.
   * Third-party Library Dependencies: Users can not add type annotations to third-party libraries that their model depend on, because these libraries are usually complex, shared by multiple programs and managed by a package manager. This literally makes it impossible for users to TorchScript their models.

As a result, users would just give up using TorchScript or even PyTorch entirely.

## MonkeyType

MonkeyType (https://github.com/Instagram/MonkeyType) is a Python tool that is designed in-house by FB/Instagram in order to add type annotations to vast (millions of lines of) legacy Python code base.

MonkeyType general workflow:

* MonkeyType registers Python profiling hook via sys.setprofile (https://docs.python.org/3/library/sys.html#sys.setprofile), the hook intercepts function/method calls and records values going into and out of function/method calls.
* Runs a Python function/method with example input provided by user under registered profiling hooks.
* The registered hooks generates a series of CallTrace (https://github.com/Instagram/MonkeyType/blob/08ee7d2aa268d4dd00f071535522c70f794fd05b/monkeytype/tracing.py#L42), which contain a mapping from function objects to types of values related to each function.
* Collected types are coalesced (via hybrid types like Union or Optional types) to form a single signature for each function.
* Computed function signatures are dumped into a data store (defaults to a sqlite database saved on disk)
* MonkeyType then applies computed function signatures back to source code in one of two ways:
    * Directly modifies source code by leveraging libcst (https://pypi.python.org/pypi/libcst)
    * Generates stub files (https://mypy.readthedocs.io/en/latest/getting_started.html#library-stubs-and-typeshed) that are honored by Python type checkers.

Though MonkeyType is most often used as an independent tool, it provides a relative stable API that allows using it as a Python library as well as customizing its behaviors, including:

* Defining custom data store for profiling data
* Defining callable to rewrite inferred types to something else
* Defining a code filter that teaches MonkeyType to selectively ignore certain Python modules in tracing

MonkeyType is battle-tested within and outside of Instagram.

## Proposal

We think the painful process of adding/fixing type annotations is not necessary because torch.jit.script doesn’t currently fully take advantage of a very accurate and coherent reference to understand types of variable: actual runtime values used in execution. By profiling compilation target and and extract types of runtime values, TorchScript can make educated guesses about arguments and return types of all involved functions and methods. These type information can greatly help torch.jit.script compilation and as a result eliminate painful process of forcing user to annotate types.

10000-Foot View

We propose that in each torch.jit.script call, TorchScript takes following steps:

* At beginning of torch.jit.script, TorchScript runs compilation target (a function or nn.Module) in eager mode under profiling of MonkeyType
* During execution, MonkeyType profiles types of all values passed into and returned from functions and methods.
* From profiling data, TorchScript can infer a set of coherent types that each function/method can accept, which we call “profiled types”, which is ephemeral and stored in a custom memory structure.
* TorchScript then combines profiled types and explicitly annotated types to form a set of comprehensive function signatures to assist compilation.
* TorchScript compiles target with assistance of combined set of function signatures.


Following sections describes in detail how to implement proposed functionality.

## API Changes

```torch.jit.script(
    obj, optimize=None, _frames_up=0, _rcb=None,
    *example_inputs**: Optional[List[Tuple]] =* *None*
    )
```

In addition to a function or nn.Module to compile, torch.jit.script should take an argument example_inputs of type Optional[List[Tuple]].

When specified by user, example_inputs should contain one or more tuples, each is a set of valid inputs to the callable obj.

After modification, this API can be invoked like following:

```
# Creating nn.Module
m = SomeModule()

# Scripting module with inferenced types from profiling
# execution with example_inputs
scripted_m = *torch**.jit.script*(m, example_inputs= [
    (torch.rand(2,3), True, 3),
    (torch.rand(2,3), False, 6),
    (torch.rand(2,3), False, "Some Input Text")
]

# Run scripted module with real inputs for profit
for i in range(10):
    scripted_m(real_inputs[i])
```

Note that the example inputs do not have to be similarly-typed. In fact, they should ideally be as different as possible in order to cover as many execution paths in compilation target as possible.

## Adding MonkeyType to PyTorch

In order to make MonkeyType usable, PyTorch users need to run a manual step to install MonkeyType from pip. This means that we can not always assume MonkeyType is available, thus our implementation must perform import checks on MonkeyType. In absence of MonkeyType, errors should be given to users who try to use this approach.

The alternative approach is to add MonkeyType as a submodule, but it has multiple downsides:

* PyTorch currently only allows critical build dependencies in submodules, MonkeyType doesn’t qualify
* Needs to modify PyTorch build system

## CallTrace Store Data Structure*

We should create a data structure JitTypeTraceStore that conforms to CallTraceStore (https://github.com/Instagram/MonkeyType/blob/f680c783c3aec6b0f613c4ea0268032cab23e788/monkeytype/db/base.py#L29), so that it can hold traced function signatures generated by MonkeyType and can provide a convenient interface for TorchScript to query later.

```
class JitTypeTraceStore(CallTraceStore):
    def __init__(self):
        super().__init__()
 *# A dictionary keeping all collected CallTrace
        # key is fully qualified name of called function
        # value is list of all CallTrace
        self.trace_records: Dict[string, List[CallTrace]] = {}*

    def add(self, traces: Iterable[CallTrace]):
        for t in traces:
            qualified_name = get_qualified_name(t.func)
            self.trace_records[qualified_name].append(t)

    # ... other boiler plate methods ...
```
`JitTypeTraceStore` holds traced function signatures for fast look up by qualified name later. CallTrace contains a pointer to the raw callable as well so that it is always possible to disambiguate.

## MonkeyType Config For JIT

MonkeyType provides a configurable tracing API that allows deep customization: MonkeyType Configuration (https://monkeytype.readthedocs.io/en/latest/configuration.html). We can directly invoke it as part of torch.jit.script to collect function/method signatures.

Customizing MonkeyType tracing behavior is done via subclassing monkeytype.config.Config (https://monkeytype.readthedocs.io/en/latest/configuration.html#monkeytype.config.Config) and overriding its methods.

```
class JitTypeTraceConfig(monkeytype.config.Config):
    def __init__(self, s: JitTypeTraceStore):
        super().__init__()
        self.s = s

    def trace_store(self) -> CallTraceStore:
        return s

    def code_filter(self) -> Optional[CodeFilter]:
        return default_code_filter
```

This config effectively tells MonkeyType to use our customized JitTypeTraceStore to hold recorded function signatures and to use default_code_filter (https://github.com/Instagram/MonkeyType/blob/f680c783c3aec6b0f613c4ea0268032cab23e788/monkeytype/config.py#L111) which avoids excessive trace records from calling python builtin methods and other third party libraries.

## Trigger MonkeyType Tracing

Inside torch.jit.script, right after performing basic script eligibility checks, we should perform MonkeyType type tracing by invoking following code

```
from monkeytype import trace as monkeytype_trace

s = JitTypeTraceStore()
monkeytype_config = JitTypeTraceConfig(s)
with monkeytype_trace(monkeytype_config):
    for example_input : example_inputs:
        obj(*example_input)
```

*PostProcess MonkeyType Trace Records*

*Type Rewriting*
TorchScript does not support all types and their annotations that are otherwise allowed in Python. Therefore, we need to post-process the trace records from MonkeyType so that they are ignored or removed.

One example of such types are *nn.Module subclasses*. MonkeyType would blindly record a user-defined nn.Module subclass as a viable type while TorchScript would fail parsing it. In this case, nn.Module subclass types inferred by MonkeyType should simply be dropped.

Another example is *all function return types* inferred by MonkeyType. Because TorchScript has a sophisticated return type inference algorithm, TorchScript can deduct more accurate return type for all functions and methods than what MonkeyType can observe. Utilizing return types inferred by MonkeyType is more likely to cause unnecessary type conflicts than being helpful.

## Aggregation
MonkeyType tracing records function signature for every single function invocation. In other words, if a function is invoked multiple times with different argument/return types.

For example:

```
def fn(cond, v):
    if cond:
        return v
    else:
        return v + 1
```

MonkeyType may yield following trace records for fn:

```
TraceRecord1: Arguments: {cond: Bool, b: Int}, Return: Int
TraceRecord2: Arguments: {cond: Bool, b: Float}, Return: Float
TraceRecord3: Arguments: {cond: Bool, b: Float}, Return: Float
```

This essentially indicates that types for argument b and return value are dynamic and aren’t expressible with simple types. Luckily, we can still use Any (or Union after it is available) type to express type of fn as

```
fn: Arguments: {cond: Bool, b: Any}, Return: Any
```

or use Union for more accurate typing

```
fn: Arguments: {cond: Bool, b: Union[Int, Float]}, Return: Union[Int, Float]
```

In order to implement this analysis, I think we can add an additional method to JitTypeTraceStore called analyze to consolidate collected trace records.

```
class JitTypeTraceStore(CallTraceStore):
    # ... other methods ...
    def analyze(self):
        self.consolidated_types = {}
        # Perform analysis described above

    def query(self, qualified_name):
        # returns types from consolidated_types
```

## Compiling With Observed Types

TorchScript currently relies on annotations.get_signature() (https://github.com/pytorch/pytorch/blob/758fb94fcb8e20d41df6b055e80725e37ddb4854/torch/jit/annotations.py#L62) to get signature of functions. It does so by:

* Using inspect.signature(fn) to get Python3-style type annotations, or
* Parsing source code of fn declaration to find Python2-style type comment

Withe the information we collected from MonkeyType, annotations.get_signture() can be enhanced to look up types from JitTypeTraceStore instance that MonkeyType stores trace records. Then use types provided by MonkeyType for arguments that are not manually annotated by users in source code.

For sake of user-friendliness, we need to clearly indicate a type is coming from profiling-based inference. Concretely, every type defined in JIT type system should have an inferred knob so that error messages can be more informative and actionable by letting user know that the offending type is from profiling.

## Backward Compatibility

This feature is fully backward compatible for a few reasons:

* Feature is only enabled when user provides additional argument to torch.jit.script call. It should have no impact for execution of legacy code.
* When feature is enabled, it honors manual type annotations from users and only provides additional typing information for arguments that were previously unannotated. Without assistance from MonkeyType, these arguments would cause compilation failure anyway. This feature should never negatively impact compilation.

## Limitations

This approach has some limitations by design:

* Type inference is only as good as input examples provided. For code paths and types that never show up in executing model with provided example input, there is no way for TorchScript to infer their types correctly.
* The need of instrumenting Python and running compilation target adds overhead to compilation, causing longer compilation times.
    * In practice, this shouldn't be a problem because `jit.trace` has similar (if not higher) overhead.
* Currently `torch.jit.script` also compiles custom Python classes, which have multiple entry points of invocation. Thus it doesn’t immediately fit into this design. However, the way of compiling a class is about to change soon, so hopefully this won’t hurt much. Otherwise, the design needs to be revised.
* MonkeyType only observes types of arguments and return value of function/method calls. For other statements, like variable assignment, MonkeyType can not trace them. Therefore, this design doesn’t address issues like incorrect type inference for a = {}.
* MonkeyType does not work for `torch.jit.interface` classes because their methods are never actually invoked during execution, thus MonkeyType can not infer their type annotations. I believe this limitation won’t translate to real problem in practice for following reasons:
    * torch.jit.interface is designed as a way to allow users to specify a module with required function/signatures, it is hard to imagine a case where user explicitly creates a `torch.jit.interface` class without knowing exact signatures for all required methods.
    * `torch.jit.interface` is a secret feature that is never publicized, it is overall rarely used.

## Alternatives Considered

## Profiling in FX-based Interpreter

Instead of using MonkeyType, we could create an FX-based interpreter (https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern) that we can easily customize to observe types of values that pass through interpretation. After chatting with https://fb.quip.com/DbWAEAPPuEy and https://fb.quip.com/LdZAEArCq7c, I found that there are two main disadvantages of this approach when compared against MonkeyType:

* FX-traced modules do not capture control flow, therefore it doesn’t work on all target use cases of jit.script
* FX doesn’t trace dispatches into underlying dunder methods, which could be customized by users and contain functions/methods that need typing.
* (From Ansley) The FX Interpreter is meant to execute a GraphModule (an nn.Module subclass created by symbolic tracing) stepwise, allowing the user to perform program introspection or on-the-fly changes. In other words, it works with something that has already been symbolically traced. Because symbolic tracing transforms Modules into their functional equivalents, type information can be lost. I think that it would be possible to use the FX Interpreter, but it seems to me that we’d be introducing an additional layer of slowness/complexity. We’d still have to do the same thing we’d be doing with MonkeyType (feed sample inputs through the program in order to get the correct type), but we’d have to do it by hand.

## PyAnnotate Profiler

PyAnnotate (https://github.com/dropbox/pyannotate) is another open source tool that has similar functionality with MonkeyType. It also observes types of runtime values in Python execution and dumps function signatures. However, when compared to MonkeyType, it has following disadvantages:

* PyAnnotate only generates Python2-style type lines, which requires additional parsing and may not be expressive enough in the future when Python evolves.
* Unless we fork and modify their source code, PyAnnotate is less customizable. For example, it doesn’t support filtering functions/methods from other third party modules.
* PyAnnotate is not as well-maintained, evident by the fact that it only supports Python2-style type line generation and that last meaningful commit into its repo is from April of 2020, while MonkeyType has commits almost every month.
* PyAnnotate is owned by DropBox, while MonkeyType is maintained by Facebook/Instagram, therefore we are likely going to get better support.

## Bidirectional Type Inference Algorithm

This is another approach of solving the general problem of missing/incorrect type annotations. Basic idea is to devise an algorithm that infers type of an argument based on how it is used in function body. This approach carries a lot higher complexity and may not be a good fit for TorchScript language specifically.

Stand-alone Tool with Human Intervention to Permanently Annotate Source Code

Another option we thought about is to create a stand-alone tool that is based on MonkeyType to help users permanently annotate their source code, instead of only generating one-time type annotations on the fly as we proposed in this approach.

The additional benefit is that stand-alone tool could optionally ask for user’s opinion on whether annotated types match their expectation and make revision based on user input.

However, using a stand-alone tool has a few down-sides:

* MonkeyType is unaware of TorchScript limitation of expressing nn.Module types and would blindly apply nn.Module and its subclasses as type annotation. Though legal in Python, they are not accepted in TorchScript. We need to configure MonkeyType to ensure it never annotates module types.
* TorchScript has sophisticated type inference for return values of functions/methods, empirically it is more than enough to infer all return types of functions. Having MonkeyType annotate return types is more likely to cause type conflicts rather than helping with compilation.
* User code may reference third-party libraries that user installed in system-wide locations. If we have a stand-alone tool that profiles types and applies derived type annotations to them, then user environment is polluted. If type annotations are not applied to them, then TorchScript would fail.
* Using a stand-alone tool is a less stream-lined experience compared to type inference embedded in torch.jit.script calls because users need to create another program that drives compilation target function.
* Intuitively I feel creating such stand-alone tool belongs and quip it with human intervention possibility is more like an enhancement to MonkeyType rather than being a part of PyTorch.
