# PyTorch Accelerator Integration Enhancements

**Authors:**

* @FFFrog
* @zeshengzong

## Summary

We propose to improve the community guidance for Accelerator Integration by writing comprehensive documentation and using [torch_openreg][TORCH OPENREG] as a code example to provide reference implementations for third-party device integration. This proposal aims to clarify the integration process and offer a step-by-step example for new accelerator developers.

Key objectives include:

* **Documentation**: Provide modular and comprehensive documentation covering core components, ensuring alignment with the realy code implementation to lower the barrier for Integration getting started.
* **Integration Example**: Use [torch_openreg][TORCH OPENREG] as a example for Accelerator Integration, offering a clear and concise reference implementation for new accelerator developers.
* **Test Backend**: Enhance [torch_openreg][TORCH OPENREG] as an in-tree test backend for Accelerator Integration, add C++ implementation consistent with real accelerator bebaviors to continuously validate the correctness and stability of the integration mechanism.

## Motivation

Since the release of PyTorch 2.1, the community has made significant efforts to address the challenges of new device integration. Including improve [PrivateUse1][PRIVATEUSE1] mechanism, providing and optimizing registration for all modules (distributed, compiler, etc.), and device-agnostic refactoring (e.g., `torch.accelerator`, memory management).

These improvements enable accelerator developers to integrate new devices without modifying PyTorch core code. This empowers vendors to independently manage integration timelines and quickly iterate on product features. Several accelerators have already been integrated or in the process of integrating with PyTorch using the Accelerator Integration mechanism. Examples include:

* [Torch MAIA](https://github.com/pytorch/pytorch/issues/155864)
* [Torch NPU](https://github.com/ascend/pytorch)
* [Torch MUSA](https://github.com/MooreThreads/torch_musa)
* [Torch MLU](https://github.com/Cambricon/torch_mlu)

However, new accelerator integration still faces several major challenges:

* The functionality is spread across many modules, with fragmented interfaces and a lack of integration guides or reference implementations, developers have to explore on their own.
* Frequent update modules make it hard for new accelerators to keep up with changes.
* There is no ongoing quality validation system for Accelerator Integration. The absence of a complete in-tree test backend guard a risk of regressions, potentially affecting many downstream accelerators.

Related issues:

* [https://github.com/pytorch/pytorch/issues/155864](https://github.com/pytorch/pytorch/issues/155864)
* [https://github.com/pytorch/pytorch/issues/144955](https://github.com/pytorch/pytorch/issues/144955)
* [https://github.com/pytorch/pytorch/issues/144845](https://github.com/pytorch/pytorch/issues/144845)

Therefore, we propose addressing these critical gaps by providing thorough documentation and reference implementations to ensure the third-party ecosystem is truly **usable, user-friendly, and sustainable** in the long term.

## Proposed Implementation

To lower the entry barrier for new accelerators, we will provide structured, modular documentation aligned with [torch_openreg][TORCH OPENREG] as an example:

* **Modular**: Documents organized by PyTorch core components (e.g., device management, operator registration, custom tensor, autograd, AMP, memory, distributed, compiler, etc.).
* **Code Examples**: The documentation will align directly with the `torch_openreg` codebase and include matching code snippets for immediate reference.
* **Step-by-Step Guides**: Start with the minimal runnable operator and gradually guide the developer through the entire integration process, enabling phased backend development.
* **Troubleshooting Guide**: Include tips for common issues and debugging tools (e.g., `Dispatch Key Debug`, `TORCH_SHOW_DISPATCH_TRACE`) to improve development efficiency.

To make this guidance, `torch_openreg` need to be refactored and optimized. Specific tasks are outlined here:

* [https://github.com/pytorch/pytorch/issues/158917](https://github.com/pytorch/pytorch/issues/158917)

---

[TORCH OPENREG]: https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg "OpenReg URL"
[PRIVATEUSE1]: https://docs.pytorch.org/tutorials/advanced/privateuseone.html "PrivateUse1"
