# Profile Directed Typing (PDT)

## Overview

A common reason for failed model scripting is missing type annotations. For a long time, the process of fixing incorrect type annotations has been through trial and error—that is, by slowly going through the various type-checking errors generated from `torch.jit.script` and fixing them one by one. This workflow is both inefficient and frustrating.

This process is not necessary because TorchScript can observe the types by running the unscripted program with a set of example inputs. In our preliminary experiments, we were able to leverage existing tools like MonkeyType to greatly reduce—or even get rid of—the need to annotate types by hand. We call this process **Profile Directed Typing (PDT)**.

## Background and Motivation

`torch.jit.script` is a decorator or function call that compiles a PyTorch program. It first inspects the Python source code, then it constructs a semantics-preserving intermediate representation (IR). The TorchScript IR can then be transformed, packaged and eventually deployed.

The TorchScript IR is statically typed, which means that all values must have a known type at runtime. Python, on the other hand, is not statically typed. This means that Python source code may not contain complete or accurate type annotations, as they aren't necessary for the Python interpreter.

`torch.jit.script` relies on explicit type annotations to determine the types of functions. In the absence of type annotations, TorchScript uses a simple type inference algorithm: everything is `torch.Tensor`. Unfortunately, even basic types like `int`, `bool`, and `float` are not inferred by TorchScript, which leads to easily-avoidable compilation errors. The tradeoffs of moving to a more sophisticated bidirectional type inference algorithm have been discussed by the team, and we have ultimately concluded that the benefits of a bidirectional type inference algorithm would not outweigh the added complexity.

Although TorchScript often needs annotated source code to successfully compile, this requirement poses a challenge to users.

## Challenges of Type Annotations

Adding type annotations is challenging for a few reasons:

* **Effort:** There is a high developer cost in adding missing annotations to an entire codebase.
* **Knowledge Gap**: Users may not know the code base well enough to determine the correct types. This is especially common for models downloaded from ModelZoo or a Github repo.
* **Third-party Library Dependencies**: Users may not have the access rights necessary to add type annotations to a third-party library.

Because of these challenges, many users give up on using TorchScript or even PyTorch entirely.

## MonkeyType

[MonkeyType](https://github.com/Instagram/MonkeyType) is a Python tool that was designed by Instagram to add type annotations to millions of lines of legacy Python code.

The general workflow of MonkeyType is as follows:

* A Python profiling hook is registered via [sys.setprofile](https://docs.python.org/3/library/sys.html#sys.setprofile). The hook intercepts function calls and records any values that enter or exit the intercepted calls.
* A Python function is run with example input provided by the user under registered profiling hooks.
* The registered hooks generates a series of [CallTrace](https://github.com/Instagram/MonkeyType/blob/08ee7d2aa268d4dd00f071535522c70f794fd05b/monkeytype/tracing.py#L42)s, which contain a mapping from function objects to their argument and return types.
* The collected types are coalesced into a single type (namely, a `Union`). This allows MonkeyType to form a single, distinct signature for each function.
* The computed function signatures are dumped into a data store. (The data store defaults to a sqlite database saved on disk.)
* The computed function signatures are applied back to the source code in one of two ways:
    * Source code is directly modified by leveraging [libcst](https://pypi.python.org/pypi/libcst)
    * [Stub files](https://mypy.readthedocs.io/en/latest/getting_started.html#library-stubs-and-typeshed) (that are honored by Python type checkers!) are generated


Though MonkeyType is most often used as an independent tool, it provides a relatively stable API that allows it to be used as a Python library. Many of its behaviors can be customized, including:

* Defining a custom data store for profiling data
* Defining a Callable to rewrite inferred types to something else
* Defining a code filter that teaches MonkeyType to selectively ignore certain Python modules in tracing

## Proposal

The painful process of adding and fixing type annotations should not be necessary when we have access to the values that will be used at runtime. By using MonkeyType, we can profile the compilation target and extract the arguments and return types of any involved functions.

In every call to `torch.jit.script`, TorchScript should take the following steps:

* Run the compilation target (a function or `nn.Module`) in eager mode. During execution, MonkeyType will profile the types of all values passed into and returned from functions and methods.
* Infer a set of coherent types that each function/method can accept, which we call “profiled types”. The profiled types are ephemeral and stored in a custom memory structure.
* Combine the profiled types and explicitly annotated types to form a set of comprehensive function signatures.
* Compile the target with assistance of combined set of function signatures.

The following sections will describe in detail how to implement the proposed functionality.

### API Changes

In addition to the function or nn.Module to compile, `torch.jit.script` accepts an `Optional[List[Tuple]]` `example_inputs`. Each tuple in the list represents a set of valid inputs to the Callable.

The new version of `torch.jit.script` can be invoked like following:

```
m = SomeModule()

example_inputs= [
    (torch.rand(**2**,**3**), **True**, **3**),
    (torch.rand(**2**,**3**), **False**, **6**),
    (torch.rand(**2**,**3**), **False**, "Some Input Text")
]

# Script module with PDT turned on. `example_inputs` is used to infer
# the arg/return types of any functions touched by `SomeModule::forward`
scripted_m = torch.jit.script(m, example_inputs)

# Run the newly scripted and annotated model with real inputs
for i in range(**10**):
    scripted_m(real_inputs[i])
```

Note that the example inputs do not have to be similarly typed. In fact, they should be as different as possible to cover a greater number of execution paths in the compilation target.

### MonkeyType Dependency

We now require PyTorch users to manually install MonkeyType if they want to use PDT.

An alternative approach would be to add MonkeyType as a submodule, but doing so would require us to modify the PyTorch build system. Furthermore, PyTorch only allows critical build dependencies in submodules, and MonkeyType doesn’t qualify as a “critical build dependency”.

### Customized `MonkeyType::CallTraceStore`

We implemented a custom [CallTraceStore](https://github.com/Instagram/MonkeyType/blob/f680c783c3aec6b0f613c4ea0268032cab23e788/monkeytype/db/base.py#L29), which we call `JitTypeTraceStore`. This data structure holds the traced function signatures generated by MonkeyType and provides an interface for TorchScript to query later.

```
class JitTypeTraceStore(CallTraceStore):
    def __init__(self):
        super().__init__()
        # key - the fully-qualified name of the function
        # value - a list of all the corresponding CallTraces
        self.trace_records: Dict[string, List[CallTrace]] = {}

    def add(self, traces: Iterable[CallTrace]):
        for t in traces:
            qualified_name = get_qualified_name(t.func)
            self.trace_records[qualified_name].append(t)

    # ... other boilerplate methods ...
```

`JitTypeTraceStore` holds the traced function signatures for fast lookup by qualified name. `CallTrace` then contains a pointer to the raw Callable, which means that it's always possible to disambiguate different functions.

### Customized `MonkeyType::Config`

To customizing our tracing behavior, we subclass the MonkeyType configurable tracing API [Config](https://monkeytype.readthedocs.io/en/latest/configuration.html) and override certain key methods.

```
class JitTypeTraceConfig(monkeytype.config.Config):
    def **init**(self, s: JitTypeTraceStore):
        super().**init**()
        self.s = s

    def trace_store(self) -> CallTraceStore:
        return s

    def code_filter(self) -> Optional[CodeFilter]:
        return default_code_filter
```

This config effectively tells MonkeyType to use our customized `JitTypeTraceStore` to hold recorded function signatures. We use the [default code filter](https://github.com/Instagram/MonkeyType/blob/f680c783c3aec6b0f613c4ea0268032cab23e788/monkeytype/config.py#L111) (`default_code_filter`) and avoid the excessive trace records that would result from recording Python builtins and third-party libraries.

### Tracer Invocation

After performing basic script eligibility checks in `torch.jit.script`, the following code begins the tracing process:

```
from monkeytype import trace as monkeytype_trace

s = JitTypeTraceStore()
monkeytype_config = JitTypeTraceConfig(s)
with monkeytype_trace(monkeytype_config):
    for example_input : example_inputs:
        obj(*example_input)
```

### Type Rewriting

TorchScript only supports a subset of Python types, so there’s the danger that MonkeyType could gather an unscriptable type. To prevent this from happening, we scan through the trace records and remove any types that are invalid in TorchScript.

There is also one interesting situation in which TorchScript’s default inference algorithm is more sophisticated than PDT: function return. TorchScript can deduce more accurate return types than MonkeyType can observe, so, in this case, we simply discard the types gathered by MonkeyType.

### Aggregation

MonkeyType tracing records a function signature for every function invocation. For example, given the following function and a set of sample inputs:

```
def fn(cond, x):
    if cond:
        return x
    else:
        return x + 1
```

MonkeyType may yield the following TraceRecords:

```
TraceRecord1: Arguments: {cond: Bool, x: Int}, Return: Int
TraceRecord2: Arguments: {cond: Bool, x: Float}, Return: Float
TraceRecord3: Arguments: {cond: Bool, x: Float}, Return: Float
```

In other words, the types for the argument `x` is dynamic and can’t be expressed with a single type. To account for this, we can simply use `Union` to express the function signature as:

```
fn: Arguments: {cond: Bool, b: Union[Int, Float]}, Return: Union[Int, Float]
```

In order to aggregate types in this way, we will add an additional method `analyze` to `JitTypeTraceStore` to consolidate the collected trace records.

```
class JitTypeTraceStore(CallTraceStore):
    # ... other methods ...
    def analyze(self):
        self.consolidated_types = {}
        # Perform analysis described above

    def query(self, qualified_name):
        # Return types from `consolidated_types`
```

### Compiling With Observed Types

TorchScript currently relies on `[annotations::get_signature](https://github.com/pytorch/pytorch/blob/758fb94fcb8e20d41df6b055e80725e37ddb4854/torch/jit/annotations.py#L62)` to get the signature of functions. `get_signature` works by either using `inspect::signature` for Python3-style type annotations or parsing the source code to find any Python2-style type comments.

We enhance `get_signature` to look up types from `JitTypeTraceStore` and to use those types for arguments that are not manually annotated by users in source code.

We will clearly indicate that a type is coming from profiling-based inference for easier user-side debugging. Concretely, this will be implemented by adding a flag to every type in the JIT type system to denote whether or not a given instance of that type was inferred or not. This flag allows us to have more specific and actionable error messages.

## Backward Compatibility

This feature is fully backward compatible:

* The feature is only enabled when the user provides an additional argument to `torch.jit.script`. It should have no impact on the execution of legacy code.
* When the feature is enabled, it honors manual type annotations from users and only provides additional typing information for arguments that were previously unannotated. Without assistance from MonkeyType, these arguments would have caused compilation failure anyway.

## Limitations

This approach has some limitations by design:

* Type inference is only as good as the input examples provided. There is no way for TorchScript to infer the types for code paths that are not hit with the provided input.
* Fully running the compilation target in eager mode adds overhead to the compilation and causes longer overall compile times.
    * Counterpoint: This shouldn't be a problem because `jit.trace` has similar—or higher--overhead.)
* Currently, `torch.jit.script` also compiles custom Python classes, which may be invoked in multiple ways. We plan to change our method of class compilation soon, but, if this doesn't happen, we'll need to revisit the design of PDT.
* MonkeyType only observes the types of arguments and return values. MonkeyType cannot infer the types in other situations, e.g. variable type assignment.
* MonkeyType does not work for `torch.jit.interface` classes because their methods are never actually invoked during execution, thus MonkeyType cannot infer their type annotations.
    * Counterpoint: `torch.jit.interface` is a way to allow users to specify a module that contains certain functions. It’s hard to imagine that a user would create a `torch.jit.interface` class without knowing the exact signatures for all required methods.
    * Counterpoint: `torch.jit.interface` is never publicized and rarely used.

## Alternatives Considered

### Profiling in FX-based Interpreter

Instead of using MonkeyType, we discussed creating an [FX-based interpreter](https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern) that could be customized to observe the types of the values that pass through the program during interpretation. However, there are several reasons why this approach is suboptimal:

* FX-traced modules do not capture control flow, which means that this implementation wouldn’t for all use cases of `torch.jit.script`.
* FX doesn’t trace into the underlying dunder methods, which could be customized by users and contain functions/methods that need typing.

### PyAnnotate Profiler

[PyAnnotate](https://github.com/dropbox/pyannotate) is another open source tool that has similar functionality to MonkeyType—that is, it runs a preliminary version of the user’s code to observe the types, dumps the function signatures into an ephemeral file, and finally applies the function signatures back to the original program. However, when compared to MonkeyType, PyAnnotate has following disadvantages:

* PyAnnotate only generates Python2-style type comments, which requires additional parsing and may not be expressive enough for later versions of Python.
* PyAnnotate is less customizable—for example, it doesn’t support filtering functions/methods from third-party modules. It would be possible to fork and modify their source code, but this would represent a significant time cost.
* PyAnnotate is not as well-maintained, while MonkeyType has commits almost every month.
* PyAnnotate is owned by Dropbox, while MonkeyType is maintained by Facebook. Using an in-house tool will likely lead to better support.

### Bidirectional Type Inference Algorithm

Another approach would be to devise an algorithm that infers the type of an argument based on how it is used in the function body. However, this solution carries a higher complexity and may not be a good fit for TorchScript language.

### Human-in-the-Loop Tool

We thought about creating a tool that, like PDT, would be based on MonkeyType; the difference would be that the type annotations would be suggested and it would be up to the user to actually make the proposed changes.  However, a standalone tool like this would have some major drawbacks:

* Both MonkeyType and our user could be unaware of which Python types are not valid TorchScript. It’s likely that a more inexperienced user would be forced to make several passes through their code.
* User code may reference a third-party library that the user has installed in a system-wide location. If we blindly apply type annotations to all code touched by the user’s program, we could pollute the user’s global environment.
* Using a standalone tool is less streamlined and degrades our user experience.
