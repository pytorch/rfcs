# PyTorch Operator Versioning


PyTorch’s operators sometimes require changes to maintain the high quality user experience (UX) that PyTorch is known for. These changes can be BC-breaking, where older programs will no longer run as expected on the latest version of PyTorch (an old writer / new reader problem) or FC-breaking, where new programs will not run on older versions of PyTorch (a new writer / old reader problem). BC and FC breaking changes have been challenging to coordinate across PyTorch because there are multiple consumers of PyTorch’s op set and we promise to keep models running in production working as expected.

This document proposes a new BC and FC policy based on operator versioning. 


## History


### Backwards Compatibility

Backwards compatibility (BC), the ability for PyTorch to continue running programs from older versions, is important so programs don’t need to be rewritten.

PyTorch makes the following BC promises today:



* **OSS** — “stable” features will be deprecated for one release before a BC-breaking change is made. [PyTorch OSS BC-breaking policy](https://pytorch.org/docs/master/) 
* **FB Internal** — we will not break a serialized torchscript program running in production at Facebook

BC-breaking operator changes were previously governed by the [Backward-compatibility Breaking Change Review Process](https://fb.quip.com/gydOArylrcKd), but this only covered torchscript and eager.


### Forwards Compatibility

Forwards compatibility (BC), the ability for older versions of PyTorch to run programs from newer versions, is important so users don’t need to update PyTorch.

PyTorch makes the following FC promises today:



* **OSS** — no promise
* **FB Internal** — PyTorch commits can run existing PyTorch eager, package/deploy, and serialized torchscript programs for at least two weeks
    * The addition of a new kwarg-only argument at the end of an op’s parameter list (but before out=, if present) with a default value is FC-compatible for serialized [torchscript](https://fb.workplace.com/groups/pytorch.dev/permalink/909079013003913/) and [edge](https://fb.workplace.com/groups/pytorch.dev/permalink/912379562673858/).


## Goals



* To support backward and (some) forward compatibility for an arbitrary BC or FC break update (schema, functional, etc) on an operator by [versioning](https://docs.google.com/document/d/1nyXmss2O003ZgKrhDmd-kyLNjjMqEXww_2skOXqkks4/edit). 
* It’s a systematic flow to prevent BC/FC breakage on both deploy and runtime stages.
* To provide testing that accurately detects dangerous BC and FC-breaking changes.
* To support both server and edge use cases, including TorchScript, package/deploy and Edge.


## Non-goals



* It does not mean that models with old operator schema can **always** run successfully on new runtime and vice versa. 
* It’s not for the “automatic” BC/FC support that can be done without any developer’s manual work (for example, the number of arguments mentioned in the Context). To apply versioning on the updated operator, the author needs to manually add a line in the version table and provide the resolution function for BC. This proposal is for BC/FC breakages that the automatic supports don’t apply. 
* It does not include the BC/FC for package/deploy itself. The Python-only operators are transparent to TS and Edge clients, with the TS compilation. 


# Proposal



* **Eager changes**
    * `operator_versions.yaml` and `operator_upgraders.py` are added to register operator upgrades that are BC/FC breaking. 
        * Note: this will not cover functional operators
        * The default value is zero
        * A version bump is also required for FC break only. It's is good for compatibility analysis: if the client is running old runtime, we don't deliver the new model with the un-compatible operator to avoid unexpected crash.
    * **Use a single operator version number for all operators**
        * This number may be shared by the deploy version, but separate from other file format versions
    * **Older versions of operators registration are kept but they should only be matchable with special circumstances**
    * **Newer version of the operator registry must specify an upgrader** that conforms to the older version of the operator schema. Its body is a TorchScript-able function that uses the newer version of operator to implement old semantics.
        * One upgrader per historic signature. The registry specifies the symbol and the file formats those upgraders are applied to.
    * [Improved BC testing] Tests that the old serialized version of the operator can still be loaded on the new runtime and run as expected need to be easy to add
        * This seems straightforward for Torchscript and Edge testing but I’m not sure how it would work for deploy/package
    * [Improved FC testing] Tests that the new version of the operator can still be loaded on old runtimes and run as expected need to be easy to add 
        * This might require a new test job, which could be tricky to setup. We have no plans to support this.
* **Torchscript changes**
    * TorchScript Runtime build contains a table of operators and their corresponding versions
    * Each serialized model contains a table of operators and their corresponding version according to the TorchScript compiler that generated the model
        * Q: will this require a new serialization format?
        * No. `kProducedFileFormatVersion` would be used as the global operator version number.
    * During loading into the TorchScript compiler, TorchScript needs to match operator schema according to the table of operator versions stored in the package. This would generate IR that conforms to older schema. 
    * Then a TorchScript pass takes IR of older schema and replaces older versions of operator invocations with bodies of upgraders.
    * Out-of-support operator versions (ie. those no longer defined in `native_functions.yaml` with a valid upgrader) need to throw an error
* **Edge runtime and mobile delivery service changes**
    * Delivery compatibility: communicating operator version table deployed on device and deliver models appropriately
    * Runtime: load upgraders at model loading time, to ensure older models always work after updating runtime
    * Unknown operator versions need to throw an error
    * The operator version and upgraders are built into the runtime for BC.
    * Allow for the addition of optional keyword-only arguments without a version bump or FC concern
    * Since additional operators can be introduced in upgraders, tracing based selective build should also cover upgraders: easier for BC because the new runtimes goes with the upgraders.
    * We should also consider the timeline for mobile to no longer use upgraders by requiring models that are too old update themselves before deployment (SLA time window).
* **torch.package changes**
    * Each torch.package package contains a table of operators and corresponding version according to PyTorch build used to package the model
        * Q: How does the torch.package scenario for mapping old versions to current PyTorch operators work?
        * A: Operator versioning, by design, can’t cover all torch.package use cases. So this should be out of scope.
* **New documentation required**
    * e2e BC-breaking guide
        * To make a BC-breaking change update the version and write a torchscript adaptor and a mobile adaptor
    * e2e FC-breaking guide
        * It’s OK to add new optional keyword-only arguments as long as their default semantic preserve the operator’s current semantics

Note that the proposal does not introduce an explicit version to _all_ PyTorch operators. Instead code changes are only required for updated operators with BC/FC breakage, that cannot be handled by automatic BC/FC methods. For other operators, the implicit version is v0.

As an example, there’s a BC/FC breaking update on operator foo.

Before:
```
foo(Tensor self, Scaler alpha=1, Tensor b, *, Tensor(a!) out) -> Tensor(a!)
```
After:
```
foo(Tensor self, Tensor c, Scaler alpha=1, Tensor b, *, Tensor(a!) out) -> Tensor(a!)
```
In schema, a Tensor argument, c is added. Note that it’s not added as a “tailing default argument”, so that BC/FC cannot be handled automatically.

Accordingly, in the kernel of foo, the implementation is updated based on the new argument c. The pseudo code (it’s in Python format, but can be written in C++ as well) looks like:


```python
def foo(Tensor self, Tensor c, Scaler alpha=1, Tensor b, *, Tensor(a!) out) -> Tensor(a!):
  # The original kernel implementation
  ...
  if not c.empty():
    a.add_(c)
```



## Code changes (minimize the work of developer)

If there is a BC/FC break with schema change in a PR, a lint error can be automatically generated, with instructions below to update the PR. 


### Version bump

Update version field in _operator_versions.yaml_.
The current table that torchscript uses should be migrated to _operator_versions.yaml_.

### BC updates

The developer needs to implement a BC “upgrader” in Python. The upgrader code is put in a centralized python file, _operator_upgraders.py_, in TorchScript format.

_operator_version.yaml_
```yaml
- func: foo
 version: 10
 upgrader: foo_upgrader_0_9
 version: 25
 upgrader: foo_upgrader_10_24
```
_operator_upgraders.py_
```python
def foo_upgrader_0_9(Tensor self, Tensor c, Scaler alpha=1, Tensor b, *, Tensor(a!) out):
  c = at.empty()
  foo(self, c, alpha, b, out = self) 

def foo_upgrader_10_24(...):
  ...
```

* Different from the upgraders defined[ here](https://github.com/pytorch/pytorch/blob/8a094e3270d2fbec6060099b7059898f4a1c104a/torch/csrc/jit/frontend/builtin_functions.cpp#L98), 
* For some operator updates, it’s not possible to have a BC adapter. If it's FC break, an upgrader is not needed. In such a case, the operator number could still help to check compatibility and to quickly detect the source of failure with meaningful error. 
* For most of the operator changes, the upgrader code is not expected to be heavy. However, the performance overhead should be observed, especially for edge use cases.
* If there are multiple upgraders (for example, v20 runtime loading both v0 and v10 models). 
* If the breakage is hard (there is no upgrader), throw “version not supported” error and suggest to regenerate the model.


### FC updates

* Except for version bump, no FC update is needed for
    * The BC/FC break op is not in the model (using [Dynamic versioning](https://github.com/pytorch/pytorch/pull/40279/files))
    * Server with 2-week FC window
    * Mobile delivering systems use [compatibility analysis](https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/mobile/model_compatibility.cpp) to guard FC break models to be delivered to clients. 
    * For internal refactors that an operator may call different transient operators, re-tracing is required for mobile to make sure the inputs of tracing based selective build don't have missing operators. Currently, it's guarded by a CI lint and a command line to retrace.
      
If needed as a temporary hotfix, an optional downgrader can be used in the backport function shown in the diagram below. It's not planed in this RFC and the discussion is left in the "Open Questions" session.


## How does it handle BC and FC?

Aligned with the “Compatibility scenarios” in the [doc of versioning in general](https://docs.google.com/document/d/1nyXmss2O003ZgKrhDmd-kyLNjjMqEXww_2skOXqkks4/edit), we do BC and FC validation at both deploy time and runtime.

### BC

Deploying a new runtime that needs to execute an existing on-device model. 



* Deploy time
    * The upgraders must be delivered together with the new runtime. 
* Runtime
    * The new runtime load the upgraders lazily: only loads the corresponding upgraders when the version matches. 
    * The new runtime should always be able to run the old model, unless the old model is “retired”. Error out when the runtime’s op min version > the model's op version.


### FC

Deploying a new model to an existing runtime. 



* Deploy time. Assuming we **keep the op version table of old runtime [query runtime]**,
    * For each operator in the “new” model, the version number is in the range of version in the “old” runtime. 
    * Otherwise, if possible, backport the model with old operator schema
* Runtime
    * The “old” runtime can run the “new” model, and errors out at load time:
        * When an unknown op appears.
        * When reaching an op whose minimum runtime version >= current
    
# Open Questions

## Use deprecation window to handle backward compatibility
One future option is to keep both old and new operators, but set a certain deprecation window for old operators. Deprecate the old operator when the window expires. There are some open questions on this option:
* What would be the window length? Would it be different for different situations (internal vs. external, server vs. mobile, etc.)
* From user's point of view, the number of operators my bloat, depending on the length of deprecation window. 

## Downgraders for FC
Dual to upgreaders for BC on client, downgraders can be used for FC on server. There are several options:
* We set a 2-week (maybe 3 week) FC window. The FC break update is split into two PRs. The first PR with new operator readers is rolled out. After the FC window (supposing all client runtime are updated to be able to read the new operator), the producer of the new operators are turned on to generate models with new operator schema.
* The 2-week window may not be sufficient for mobile, where the runtime updated cannot be fully controlled. In such a case, we need to "backport" the new model to an old format. 
    * We save the historical models that were exported in earlier code. If a compatible older model can be found, the older model is delivered to the old runtime.
    * We could apply a downgrader at the server side: to rewrite the new model in the old format. Since the downgrader happens at model delivery time, it can be done out of the major export flow. 
* Keep old PyTorch release binaries (for example, PyTorch 1.10). Mobile can backport some operators to PyTorch 1.10 opportunistically. It's more challenging to maintain historical binaries than historical models. 
