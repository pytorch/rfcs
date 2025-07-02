# Supporting RISC-V for PyTorch on CPU

**Authors:**
* @luhenry


## **Summary**

PyTorch currently provides CPU builds for Linux on x86_64 and AArch64, available directly from `https://download.pytorch.org/whl/cpu`. Additionally, it is cross-compiled for s390x on GitHub Actions. RISC-V is an alternative, open-standard hardware architecture that is gaining significant traction across the industry. To support this growing ecosystem, we propose adding out-of-the-box support for PyTorch on RISC-V. The north star is to enable users on a RISC-V machine to simply run the following command to install PyTorch:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## **Motivation**

* **Meeting Community Demand:** The RISC-V community has a strong interest in AI/ML workloads, and PyTorch is a leading framework in this domain. Providing native support addresses this need directly.
* **Alignment of Values:** Both the PyTorch and RISC-V communities are deeply committed to open-source principles and broad accessibility. Supporting RISC-V aligns with these shared values.
* **Expanding PyTorch's Reach:** RISC-V's single ISA targets a wide range of hardware verticals (embedded, automotive, server, desktop, mobile, etc.). Adding support allows PyTorch to be used transparently across these diverse markets, significantly broadening its potential user base.
* **Industry Trend:** The increasing industry traction of RISC-V, combined with the growing importance of AI/ML workloads, makes native PyTorch support a timely and strategic move.
* **Enabling New Hardware and Specialized Applications:** RISC-V's flexibility and open nature facilitate the development of custom hardware and specialized accelerators. Native PyTorch support on RISC-V will enable researchers and developers to deploy AI/ML models on these novel platforms, fostering innovation in areas like edge computing, embedded AI, and domain-specific architectures.
* **Major Ecosystem Support:** NVIDIA has announced CUDA availability on RISC-V, further validating the architecture's relevance for AI/ML workloads ([announcement](https://x.com/risc_v/status/1946251939823370697)).

## **Proposed Implementation**

### Dependencies

To actively help enable the Python ecosystem on RISC-V, the [RISE project](https://riseproject.dev/), a consortium of companies interested in the success of RISC-V, has been openly building, testing, and distributing many Python packages on RISC-V, and made them freely available at https://riseproject.gitlab.io/python/wheel_builder/. It is meant as a stop-gap solution, as the Python ecosystem gradually improves its support for RISC-V. See [pypa/manylinux#1426](https://github.com/pypa/manylinux/discussions/1426) for more details.

* **RISC-V support in PyPA tooling:** Work is ongoing to add RISC-V support to the Python packaging ecosystem, with [manylinux_2_39_riscv64](https://quay.io/repository/pypa/manylinux_2_39_riscv64) recently introduced. The discussion is tracked in [pypa/manylinux#1426](https://github.com/pypa/manylinux/discussions/1426).
* **PyTorch dependencies on RISC-V:** RISE is actively working to provide PyTorch dependencies for RISC-V, as demonstrated by their [wheel builder](https://riseproject.gitlab.io/python/wheel_builder/). While this proves that packaging is feasible, it remains a temporary solution as upstream support for RISC-V is being integrated into these dependencies.
* **CI/CD hardware availability:** Access to RISC-V hardware in CI/CD environments is currently limited. RISE is addressing this by making available free RISC-V runners for GitHub and GitLab projects, and is committed to sponsoring hardware resources for key projects like PyTorch to ensure adequate testing and build capacity.

### Work to be done

> **Note:** This section needs to be largely modified to go into many more details. I would highly value having feedback from existing maintainers to help me scope it properly.

PyTorch can already be built on RISC-V, and this is currently being done internally. Efforts are underway at RISE to add automated building, testing, and packaging for RISC-V. However, this approach is intended as a temporary measure; the long-term goal is for upstream PyTorch to provide official RISC-V distributions without relying on intermediaries like RISE.

At present, PyTorch builds successfully on RISC-V without code modifications. A step-by-step guide is available at https://gist.github.com/luhenry/ffa8158ab6b8a4c56f354b3f0f5a61a7.

A key challenge remains around testing and the hardware resources required to support comprehensive CI for PyTorch, given the scale of contributions. This topic has been discussed with members of the PyTorch Foundation and is recognized as the primary area needing work: determining the necessary machine capacity and identifying suitable providers.

## **Metrics**

The primary goal is to enable users to run `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` directly on a RISC-V machine, with all required dependencies readily available.

## **Drawbacks**

While this change does not introduce complexity to the PyTorch codebase itself, it does require additional infrastructure to build and test on RISC-V for all contributions. RISC-V hardware is less accessible than other platforms, with limited availability on cloud providers and the need for specialized developer boards or emulation via QEMU. This makes it more challenging for contributors to debug and test issues locally. However, since PyTorch already builds and runs on RISC-V without code modifications, this has not been a significant problem so far.

There is also a dependency on external communities to provide RISC-V support for packages outside of PyTorch’s direct control (e.g., NumPy). Ongoing efforts by these communities, as well as by RISC-V advocates, are addressing these gaps.

## **Alternatives**

Without upstream support, RISE and various hardware vendors may distribute their own versions of PyTorch for RISC-V, which could lead to fragmentation and inconsistent support across the ecosystem.

## **Prior Art**

RISC-V adoption is accelerating, with many major projects now supporting the architecture. For example, Llama.cpp has been available on RISC-V since [this release](https://github.com/ggml-org/llama.cpp/releases/tag/b4969). Other notable projects with RISC-V support include the Linux kernel, GNU toolchain, LLVM, OpenJDK, Go, Ubuntu, and Red Hat among many more.

## **How we teach this**

### Would the acceptance of this proposal mean the PyTorch documentation must be re-organized or altered?

The PyTorch documentation would require updates to list RISC-V as a supported platform.

### How should this feature be taught to existing PyTorch users?

No significant changes are needed for existing users, aside from communicating the new availability of PyTorch on RISC-V systems.

## **Unresolved questions**

### What parts of the design do you expect to resolve through the RFC process before this gets merged?

Key questions include determining the required pool of RISC-V machines for the PyTorch project, estimating necessary machine capacity, identifying infrastructure changes, supporting the Infrastructure team, and integrating RISC-V support into the developer workflow with minimal disruption.

### What parts of the design do you expect to resolve through the implementation of this feature before stabilization?

To be determined.

### What related issues do you consider out of scope for this RFC that could be addressed in the future independently of the solution that comes out of this RFC?

Support for accelerators such as CUDA, HIP, and SYCL is considered out of scope for this RFC.

## Resolution
We decided to do it. X% of the engineering team actively approved of this change.

### Level of Support
Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.


#### Additional Context
Some people were in favor of it, but some people didn’t want it for project X.


### Next Steps
Will implement it. 


#### Tracking issue
<github issue URL>


#### Exceptions
Not implementing on project X now. Will revisit the decision in 1 year.
