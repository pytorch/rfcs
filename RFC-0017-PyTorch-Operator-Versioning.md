
<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 1; WARNINGs: 2; ALERTS: 3.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>
<a href="#gdcalert2">alert2</a>
<a href="#gdcalert3">alert3</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>



# PyTorch Operator Versioning


# Context

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
    * operator_versions.yaml** and **operator_upgraders.py** are added to register operator upgrades that are BC/FC breaking. **

<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: undefined internal link (link text: "Example changes"). Did you generate a TOC? </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

[Example changes](#heading=h.enajdjh00ukn)
        * Note: this will not cover functional operators
        * The default value is zero
    * **Use a single operator version number for all operators**
        * This number may be shared by the deploy version, but separate from other file format versions
    * **Older versions of operators registration are kept but they should only be matchable with special circumstances**
    * **Newer version of the operator registry must specify an upgrader** that conforms to the older version of the operator schema. Its body is a TorchScript-able function that uses the newer version of operator to implement old semantics.
    * [Improved BC testing] Tests that the old serialized version of the operator can still be loaded on the new runtime and run as expected need to be easy to add
        * This seems straightforward for Torchscript and Edge testing but I’m not sure how it would work for deploy/package
    * [Improved FC testing] Tests that the new version of the operator can still be loaded on old runtimes and run as expected need to be easy to add 
        * This might require a new test job, which could be tricky to setup
* **Torchscript changes**
    * TorchScript Runtime build contains a table of operators and their corresponding versions
    * Each serialized model contains a table of operators and their corresponding version according to the TorchScript compiler that generated the model
        * Q: will this require a new serialization format?
        * Yes, but it is easy to make it BC and FC
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

foo(Tensor self, Scaler alpha=1, Tensor b, *, Tensor(a!) out) -> Tensor(a!)

After:

foo(Tensor self, **Tensor c**, Scaler alpha=1, Tensor b, *, Tensor(a!) out) -> Tensor(a!)

In schema, a Tensor argument, c is added. Note that it’s not added as a “tailing default argument”, so that BC/FC cannot be handled automatically.

Accordingly, in the kernel of foo, the implementation is updated based on the new argument c. The pseudo code (it’s in Python format, but can be written in C++ as well) looks like:


```
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


### BC updates

The developer needs to implement a BC “upgrader” in Python. The upgrader code is put in a centralized python file, _operator_upgraders.py_. 



<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline drawings not supported directly from Docs. You may want to copy the inline drawing to a standalone drawing and export by reference. See <a href="https://github.com/evbacher/gd2md-html/wiki/Google-Drawings-by-reference">Google Drawings by reference</a> for details. The img URL below is a placeholder. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![drawing](https://docs.google.com/drawings/d/12345/export/png)



* Different from the upgraders defined[ here](https://github.com/pytorch/pytorch/blob/8a094e3270d2fbec6060099b7059898f4a1c104a/torch/csrc/jit/frontend/builtin_functions.cpp#L98), 
* For some operator updates, it’s not possible to have a BC adapter. In such a case, the operator number could still help to check compatibility and to quickly detect the source of failure with meaningful error. 
* For most of the operator changes, the upgrader code is not expected to be heavy. However, the performance overhead should be observed, especially for edge use cases.
* If there are multiple upgraders (for example, v2 runtime loading v0 model), the upgraders must be chained: upgrader_0_1, then upgrader_1_2. 
* If the breakage is hard (there is no upgrader), throw “version not supported” error and suggest to regenerate the model.


### FC updates



* No FC update is needed for
    * The BC/FC break op is not in the model (using [Dynamic versioning](https://github.com/pytorch/pytorch/pull/40279/files))
    * Server with 2-week FC window
    * Mobile delivering systems use compatibility analysis to guard FC break models to be delivered to clients. If needed as a temporary hotfix, the downgrader can be used in the backport function shown in the diagram below.


# How does it handle BC and FC?

Aligned with the “Compatibility scenarios” in the [doc of versioning in general](https://docs.google.com/document/d/1nyXmss2O003ZgKrhDmd-kyLNjjMqEXww_2skOXqkks4/edit), we do BC and FC validation at both deploy time and runtime.

 

<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline drawings not supported directly from Docs. You may want to copy the inline drawing to a standalone drawing and export by reference. See <a href="https://github.com/evbacher/gd2md-html/wiki/Google-Drawings-by-reference">Google Drawings by reference</a> for details. The img URL below is a placeholder. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![drawing](https://docs.google.com/drawings/d/12345/export/png)

The diagram above is generic for both Torchscript and bytecode for Edge devices. Limitations of edge runtime will be listed below.


## BC

Deploying a new runtime that needs to execute an existing on-device model. 



* Deploy time
    * The upgraders must be delivered together with the new runtime. 
* Runtime
    * The new runtime load the upgraders lazily: only loads the corresponding upgraders when the version matches. 
    * The new runtime should always be able to run the old model, unless the old model is “retired”. Error out when the runtime’s op min version > the model's op version.


## FC

Deploying a new model to an existing runtime. 



* Deploy time. Assuming we **keep the op version table of old runtime [query runtime]**,
    * For each operator in the “new” model, the version number is in the range of version in the “old” runtime. 
    * Otherwise, if possible, backport the model with old operator schema
* Runtime
    * The “old” runtime can run the “new” model, and errors out at load time:
        * When an unknown op appears.
        * When reaching an op whose minimum runtime version >= current