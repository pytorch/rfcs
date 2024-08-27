
# [RFC] Generalization of PyTorch framework UT for non-cuda device execution

**Authors:**
* @ankurneog


## **Summary**
Modify PyTorch framework UTs so that non-cuda devices such as intel Gaudi and intel XPU is able to harness the content and improve quality.


## **Motivation**
The Pytorch framework UTs are good indicator for device stack health, however these are mostly written for cpu and cuda devices, which restricts its use for non-cuda devices.

We propose to modify the content wherever possible to make it available for non-cuda device execution

This will also ensure greater participation for content enhancement.

### **Examples**

*  The execution is blocked for non-native devices using decorators such as ```onlyNativeDevices```
*  The execution is blocked for cuda only using decorators such as ```onlyNCCL``` or ```onlyCUDA```
*  Need scalable mechanism to select Dtypes per op described in OpInfo or ModuleInfo instead of using separate variable similar to ```dtypesIfCUDA```
*  Need scalable mechanism to skip for different devices instead of using specific decorator ```skipIfCUDA```
*  The dynamo content should be refactored to allow tweaking per platform/device for eg. addition of custom backends or skipping in case of unsupported backends
*  Distributed content assumes most execution is done for nccl and gloo, with almost entire non-cpu content hard coded for nccl.

## **Proposed Implementation**
Since the content is huge, we propose a staggered approach for the implementation
Steps:
*   Remove restriction imposed through @onlyNativeDevices in core content, replace these with hooks so that supported devices can enable their content selectively.
These should be flexible enough to support both in-tree and out-of-tree devices.
*   Dtypes for a device should be dynamically loaded per op based on a common dictionary, instead of using different variables per device , eg: dtypesIfCuda
*   Miscelleneous decorators such as @skipIfCuda should be generalized @skipIfDevice
*   Extend use of instantiate_device_type for all content, so that developers are forced to use generalized device code rather than using "cuda" or "cpu"
*   Generalize common distributed content , so that it can be extended for non nccl backends such as intel's hccl and ccl
*   Generalize the dynamo content for specific backends which other devices might want to verify with existing content, the backends should always be extracted from
    a list that is abstracted out and the list can be appended per device per TC.



#### Metrics
Other devices can track the pass-percentage and be part of the CI if the coverage and pass percentage is good.

#### Additional Context
Towards adding support for Intel Gaudi devices we have already done couple of changes in this regard.
* Removing onlyNativeDevice : https://github.com/pytorch/pytorch/pull/128584

* Changing Dynamo Content : https://github.com/pytorch/pytorch/pull/130714

* Generalizing Distributed Content : https://github.com/pytorch/pytorch/pull/131758

* Generalizing FSDP Content : https://github.com/pytorch/pytorch/pull/133209

More to follow


### Next Steps
As part of introducing support for intel Gaudi which is an out-of-tree device, we are already introduces changes to support it in a manner that can be used by other devices as well.



