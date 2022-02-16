# Context
With recent developments in PyTorch Edge applications on embedded systems and resource limited devices, the issue of op registration/dispatching runtime overhead and build-time complexity has risen to the fore.

We thought about possible solutions:
* One option is to keep using the torch-library C++ API to register/dispatch ops, but this will necessitate careful cost-cutting for these use cases with more and more intrusive “#ifdef” customizations in the core framework.
* The other option (which is what we propose here) is to utilize the function schema yaml file to "declare" ops and then use the codegen framework to generate lightweight code for runtime execution.

The essential point of this proposal is that the function schema DSL (which we use to declare the standard ATen ops) combined with the codegen framework (which we use to generate the registrations, dispatch stubs and other “glue” code for ATen ops) is the bare minimum set of reusable tools for building custom extensions that are compatible with the PyTorch ecosystem.
# Motivation
* **Performance**
  * For recent use cases of Edge interpreter, we need to satisfy more and more strict initialization latency requirements, where analysis shows op registration contributes to a large portion of it.
  * With existing meta-programming based unboxing logic shared between mobile and server, it’s relatively inflexible to introduce optimizations.
  * Also with static dispatch, we don’t have to register all of the ops into the JIT op registry, which saves runtime memory usage and further reduces static initialization time.
  * It is possible to avoid dispatching at runtime.
* **Modularity and binary size**
  * Currently the mobile runtime consists of both JIT op registry and c10 dispatcher. This project will make it possible to not depend on the c10 dispatcher, delivering a cleaner runtime library.
  * This project creates an opportunity to reduce binary size by getting rid of the dispatcher and enables further size optimization on unboxing wrappers.
* **Ability to incorporate custom implementation of ATen ops**
  * For some of the edge use cases, we need to support custom implementations of ATen ops. With an extra op registration path such as codegen unboxing it is easier to hookup ops with custom native functions.

# Overview
  
Currently the lite interpreter (or Edge runtime) registers all ATen ops into the dispatcher and some other ops into the JIT op registry. At model inference time the interpreter will look for the operator name in the JIT op registry first, if not found then it will look into the dispatcher. This proposal **adds a build flavor that moves these ATen ops from dispatcher to JIT op registry** so that it’s easier to optimize (e.g., avoid schema parsing) and can also reduce dependencies. 

The interpreter is looking for a boxed function but our native implementation is unboxed. We need “glue code” to hook up these two. This proposal **extends the capabilities of codegen to generate the unboxing wrappers for operators**, as well as the code to register them into the JIT op registry. The interpreter will call generated unboxing wrappers, inside these wrappers we pop out values from the stack, and delegate to the unboxed API.

To avoid hitting the dispatcher from the unboxed API, we will choose static dispatch so that we hit native functions from the unboxed API directly. To make sure we have feature parity as the default build, this proposal **adds support for multiple backends in static dispatch**.

In addition to that, this proposal also supports features critical to Edge, such as **tracing based selective build** and **runtime modularization** work.

# Step by step walkthrough

How will our new codegen unboxing wrapper fit into the picture of op registration and dispatching? For these use cases, we only need per-op codegen unboxing (red box on the left) as well as static dispatch. This way we can avoid all dependencies on c10::Dispatcher. 

We are going to break the project down into three parts, for **step 1 we are going to implement the codegen logic** and generate code based on [native_functions.yaml](https://fburl.com/code/2wkgwyoq), then we are going to verify the flow that we are able to find jit op in the registry and eventually call codegen unboxing wrapper (the red flow on the left). **Step 2 will focus on how to make sure we have feature parity** with the original op registration and dispatch system, with tasks like supporting multiple backends in static dispatch, supporting custom ops as well as custom kernels for ATen ops.For **step 3 we are going to integrate with some target hardware platforms**. These are the problems we need to address in step 3 including: avoiding schema parsing at library init time, supporting tracing based selective build. The goal of step 3 is to make sure per-op codegen unboxing works for our target hardware platforms and is ready to ship to production.


### Step 1

Bring back the unboxing kernel codegen using the new codegen framework. And make the registration no-op when we turn on the static root-op dispatch for lightweight dispatch use cases. All tasks in step 1 are based on the server version of PyTorch interpreter.


#### Codegen core logic

These tasks will generate C++ code that pops ivalues out from a stack and casts them to their corresponding C++ types. This core logic should be shared across two types of codegens so that it can be covered by all the existing tests on server side.



* **JIT type -> C++ type**. This is necessary for some of the optional C++ types, e.g., we need to map `int` to `int64_t` for the last argument in the example.
    * This is already done in [types.py](https://github.com/pytorch/pytorch/blob/master/tools/codegen/api/types.py), and we need to integrate it into our new codegen.
* **JIT type -> IValue to basic type conversion C++ code.** E.g., the first argument of this operator: `Tensor(a) self` needs to be translated to: `(std::move(peek(stack, 0, 4))).toTensor()`
    * IValue provides APIs to directly convert an ivalue to these basic types. See [ivalue_inl.h](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/ivalue_inl.h#L1453-L1493)
    * Here’s a [list](#bookmark=id.deyvpbsb5yel) of all the JIT types appearing in native_functions.yaml, most of them can be converted using ivalue’s API.
    * Add a binding function between a JIT type to a piece of C++ code that converts IValue to a specific C++ type.
* **JIT type -> IValue to ArrayRef type conversion C++ code. **IValue doesn’t provide explicit APIs for these ArrayRef types, but they are widely used in native_functions.yaml. 
    * We can use the meta programming logic ([make_boxed_from_unboxed_functor.h](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h#L354)) as reference, convert the ivalue to vector then to ArrayRef.
* **JIT type -> IValue to TensorOptions type conversion C++ code.**
    * Handle TensorOptions (that is not 1-1 mapping across two types of arguments), we can refer to [python.py](https://github.com/pytorch/pytorch/blob/master/tools/codegen/api/python.py#L999-L1068), maybe follow the logic over there.
* **JIT schema -> unboxed function**. With all the arguments being translated, generate the C++ code to call the correct unboxed function and return the result (push it back to stack).
    * Figure out how to map schema to unboxed C++ function. Reference [python.py](https://github.com/pytorch/pytorch/blob/master/tools/codegen/api/python.py#L955)
    * Deal with method and function separately, also handle the `out` cases.


#### Codegen source file details

With the logic from the previous section, we should be able to wrap the code into a function pointer and register it into [torch::jit::OperatorRegistry](https://fburl.com/code/bxu4rfem). 



* Wrap generated C++ code in [OperatorGenerator](https://fburl.com/code/rdg601q8) so that it gets registered into the registry. Generate code for all functions in [native_functions.yaml](https://fburl.com/code/5hy194vj). Code snippet as an example:
```cpp
CodegenUnboxingWrappers.cpp
===================
RegisterOperators reg({
   OperatorGenerator(
       TORCH_SELECTIVE_SCHEMA("aten::get_device(Tensor self) -> int"),
       [](Stack & stack) {
         RECORD_FUNCTION("get_device", std::vector<c10::IValue>());
         at::unboxing::get_device(stack);
       },
       aliasAnalysisFromSchema()
   ),
...
})
CodegenFunctions.h
===================
namespace at {
namespace unboxing {

TORCH_API at::Tensor get_device(Stack & stack);

} // namespace unboxing
} // namespace at

CodegenFunctions.cpp
=====================
namespace at {
namespace unboxing {

TORCH_API at::Tensor get_device(Stack & stack) {
    auto result = at::get_device(
        (std::move(peek(stack, 0, 1))).toTensor()
    );
    drop(stack, 1);
    pack(stack, std::move(result));
}

} // namespace unboxing
} // namespace at
```




* Generate separate header/cpp for codegen unboxing wrapper. We should put the codegen unboxing body into a separate function with dedicated namespace so that it is on par with the other codegen (Functions.h).
* Compile generated code with current runtime and make sure the calls to ATen ops are getting dispatched to our codegen’d unboxing wrapper.
    * The easiest way to test is to generate a wrapper that prints out/throws an exception. Then we can execute a scripted module to trigger the dispatch.
#### Server & OSS integration

**Bringing codegen unboxing to server build is out of the scope of this project.** We evaluated the option of replacing JIT op registration hook (`register_c10_ops.cpp`) with codegen unboxing wrappers, but realized that effort needs proper design and a lot of effort and only brings in small value:



* Having two op registration mechanisms brings more confusion.
* For the scenario of adding a new operator (not to `native_functions.yaml`), we need to provide clear guidance to add it to the JIT op registry as well, otherwise JIT execution will break.
* We can add tests on the mobile build for the sake of coverage.

For OSS mobile integration, we will need to have a new build flavor to switch between c10 dispatcher vs jit op registry. This new flavor will include codegen source files (`CodegenFunctions.h, CodegenFunctions.cpp, CodegenUnboxingWrappers.cpp`) instead of existing dispatcher related source files: `Operators.cpp`, `RegisterSchema.cpp `etc, similar to the internal build configuration.



### Step 2

With step 1 we already have a working codegen unboxing + static dispatch system working but it only works for the `CPU` backend. Nowadays most models being deployed on edge devices are quantized models so we will need to support both `CPU` and `QuantizedCPU` backend. In addition to that, a lot of our models feature custom ops, however we can’t register custom ops through the old dispatcher  (`TORCH_LIBRARY`) APIs any more. Here I’m proposing a solution that exposes the `native_function.yaml` syntax to the internal developers targeting this runtime mode: allow them to use the yaml file format to declare their custom ops and/or custom kernels.

 


#### Support multiple backends in static dispatch

**NOTE: this may be optional if we enabled backend tracing for ops.** For the vast majority of models, we will only have 1 backend per operator, meaning that if we can pass the backend info into codegen, we don’t have to do dispatch based on dispatch key.

In the scenario that a model contains both floating point ops and quantized ops, our codegen should be able to statically dispatch to the correct backend. The following diagram shows what will be generated and included in the build and demonstrates the dependency relationship.

Let’s take `acosh` as an example:
```yaml
native_functions.yaml
=====================
- func: acosh(Tensor self) -> Tensor
  variants: function, method
  structured_delegate: acosh.out

- func: acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
  structured: True
  structured_inherits: TensorIteratorBase
  dispatch:
    CPU, CUDA: acosh_out
```

And if we pass the backends we want to codegen for both `CPU` and `QuantizedCPU` backends, our `Functions.h` will be something like this (borrowing from Jiakai’s [PR](https://github.com/pytorch/pytorch/pull/51554/commits/0ba3d4cc42187f69f17e0c382f0ab51e071a4a44)):

```cpp
Functions.h
===========
// aten::acosh(Tensor self) -> Tensor
TORCH_API inline at::Tensor acosh(const at::Tensor & self) {
    DispatchKeySet _dk_set = c10::detail::multi_dispatch_key_set(tensor);
    DispatchKey _dk = _dk_set.highestPriorityBackendTypeId();
    switch (_dk) {
    case DispatchKey::CPU:
        return at::cpu::acosh(self);
    case DispatchKey::QuantizedCPU:
    default:
        TORCH_CHECK(false, "Unsupported static dispatch", _dk);
    }
}
```
Also we will generate these files:



* `CPUFunctions_inl.h` (does not contain `acosh` declaration)
* `CPUFunctions.h`
* `RegisterCPU.cpp` (without `TORCH_LIBRARY` calls)
* `QuantizedFunctions_inl.h` (contains acosh declaration)
* `QuantizedFunctions.h`
* `RegisterQuantizedCPU.cpp` (without `TORCH_LIBRARY` calls, contains `acosh` definition)



### Step 3

With step 2 finished we should have feature parity as the existing op registration & dispatch system. Now we need to consider the problems specific to edge devices. How do we support custom kernels for different edge devices? How do we make sure the performance is improved as expected? How do we make the binary size as small as possible? This step is aiming to tackle these problems and the end goal is to ship this codegen unboxing + static dispatch approach.


#### Bring codegen to target platform



* Consider adding ops to our new `custom_ops.yaml` created in step 2 (maybe also rename), let the codegen read from the new yaml. The benefit of doing this is that we can easily support ATen ops with custom kernels (not to be confused with custom ops) and we only have a single source of truth.
    * There are two options, either we figure out all the dependencies for all the ops required, or we leverage tracing based selective build.
* Bring everything to our target hardware platform to make sure it builds and runs.
* Disable current Dispatcher. Avoid linking any `TORCH_LIBRARY` API calls in the build, we can only profile the performance this way.
    * With codegen unboxing + static dispatch, we hope that we can reach a much smaller percentage of cycle count for the op registration step.



#### Avoid schema parsing at runtime

As mentioned in step 1, we are registering an operator into the registry along with a schema string. We realized at the library initialization time we need to spend a lot of resources on schema parsing, according to the [profiling results](https://fburl.com/qdud9ni0) based on our prototype. We also noticed that the required information to instantiate a schema object are all available at codegen time, we can pass these data to the registry directly so that we can save time at runtime. For example:


```
CodegenUnboxing.cpp
===================
RegisterOperators reg({
   OperatorGenerator(
       "aten::get_device", // name
       "", // overload_name
       arguments, // a vector of arguments
       returns, // a vector of returns
       [](Stack & stack) {
         RECORD_FUNCTION("get_device", std::vector<c10::IValue>());
         at::unboxing::get_device(stack);
       },
       aliasAnalysisFromSchema()
   ),
...
})
```


This way we can directly instantiate `FunctionSchema` objects without parsing at runtime. Of course we need to change APIs in `operator.h` to make this happen.

Q: Can we completely get rid of `FunctionSchema` and only register name/overload_name?

A: No, because we should have feature parity to the current system and backward compatibility for mobile models is a feature we need to support for the lightweight dispatch system. Currently we rely on the number of arguments to let the new runtime be able to run the old model.


#### Support tracing based selective build

* In [gen.py](https://github.com/pytorch/pytorch/blob/master/tools/codegen/gen.py) the files we generate will go through the selector similar to what we are doing to `RegisterSchema.cpp` right now.
* We need to make sure the binary size is on-par with or even better than existing tracing based selective build.

## Risks

There are 3 risks:



1. Performance gain of using JIT op registry is insignificant or even worse than dispatcher.
    1. De-risked: from the prototype running on a target platform it is proved to save latency on initial load.
2. Binary size regression. Need to make sure selective build works.
3. Mobile use case requires features only available on dispatcher.
    1. E.g., boxed fallback mechanism for [conj](https://fburl.com/ynkc32k2) operator.

## Testing & Tooling Plan

Expand existing tests on lite interpreter to cover codegen logic. Let the cpp test target depending on codegen unboxing library, test if a module forward result from JIT execution equals to the lite interpreter execution. Since JIT execution goes through metaprogramming unboxing and lite interpreter execution goes through codegen unboxing, we can make sure the correctness of codegen unboxing. Example:

```cpp
TEST(LiteInterpreterTest, UpsampleNearest2d) {
 Module m("m");
 m.define(R"(
   def forward(self, input: Tensor, scale:float):
     return torch.upsample_nearest2d(input, [1, 1], float(scale), float(scale))
 )");

 std::vector<IValue> inputs;
 inputs.emplace_back(torch::rand({1, 3, 128, 128}));
 inputs.emplace_back(at::Scalar(2.0));
 auto ref = m.forward(inputs);

 std::stringstream ss;
 m._save_for_mobile(ss);
 mobile::Module bc = _load_for_mobile(ss);
 IValue res;
 res = bc.forward(inputs);

 auto resd = res.toTensor();
 auto refd = ref.toTensor();
 ASSERT_TRUE(resd.equal(refd));
}
```

## Appendix

See design doc: [Lightweight operator registration & dispatching](https://docs.google.com/document/d/1XgJDhm0crrBNMRAwm5XXVnGvB7Z73qgsAzJTkt4FjAg/edit)