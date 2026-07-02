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
* **PyTorch dependencies on RISC-V:** A growing number of PyTorch's Python and native dependencies are being enabled natively on `riscv64` upstream. In the meantime, RISE maintains a [wheel builder](https://riseproject.gitlab.io/python/wheel_builder/) and PyPI index (`https://gitlab.com/api/v4/projects/riseproject%2Fpython%2Fwheel_builder/packages/pypi/simple`) that builds, tests, and distributes the remaining packages on `riscv64`. This index is a stop-gap and will be retired as upstream support lands.
* **CI/CD hardware availability:** RISE now operates the [RISE RISC-V Runners](https://riseproject-dev.github.io/riscv-runner/), a free managed GitHub Actions service that runs jobs on real RISC-V hardware (Scaleway EM-RV1 bare-metal nodes) via the [`rise-risc-v-runners`](https://github.com/apps/rise-risc-v-runners) GitHub App. Workflows opt in by setting `runs-on: ubuntu-24.04-riscv`, and Docker-in-Docker is supported out of the box. See the [announcement from 2026-03-24](https://riseproject.dev/2026/03/24/announcing-the-rise-risc-v-runners-free-native-risc-v-ci-on-github/) for background. Each node runs at most one job at a time, so overall capacity remains the primary gating factor for scaling PyTorch CI onto this fleet — see *Unresolved questions* for sizing discussion.

### Work to be done

PyTorch already builds on RISC-V without code modifications, and a prototype of PyTorch's upstream CI running on the RISE RISC-V Runners is maintained at [`riseproject-dev/pytorch@manywheel-riscv64-1`](https://github.com/riseproject-dev/pytorch/tree/manywheel-riscv64-1). On that branch, the `docker-build` and `linux-riscv64` jobs build successfully end-to-end; test execution is not yet wired up but is close. A manual step-by-step build guide (useful for local reproduction) is available at https://gist.github.com/luhenry/ffa8158ab6b8a4c56f354b3f0f5a61a7.

The current status was presented at PyTorch Conference EU 2026 in the lightning talk "ExecuTorch on Microcontrollers: Deploying PyTorch to the smallest edge" ([session page](https://pytorchconferenceeu2026.sched.com/event/2HinF/lightning-talk-executorch-on-microcontrollers-deploying-pytorch-to-the-smallest-edge-rj-ascani-matthias-cremon-meta), [slides (PDF)](https://hosted-files.sched.co/pytorchconferenceeu2026/0b/PyTorch%20Conf%20EU%20-%202026%20-%20PyTorch%20on%20RISC-V.pdf)).

Automated building, testing, and packaging on RISC-V is intended as a bring-up step; the long-term goal is for upstream PyTorch to produce official RISC-V distributions directly, without relying on RISE as an intermediary. To that end, RISE is engaging with the PyTorch Multi-Cloud Working Group to scope what resources (hardware, runners, build infrastructure) it can contribute and how they integrate with the existing PyTorch CI infrastructure.

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

The main open items are already in flight rather than fully open:

* **Runner capacity and performance:** faster RISC-V runners are expected in the coming weeks/months via RISE, and interim stop-gaps are possible depending on the machine count PyTorch CI actually requires. Sizing and integration are under discussion with the PyTorch Multi-Cloud Working Group.
* **Dependencies:** continue pushing upstream native `riscv64` support in PyTorch's Python and native dependencies so that the RISE wheel-builder index can eventually be retired. PyPA's [manylinux_2_39_riscv64](https://quay.io/repository/pypa/manylinux_2_39_riscv64) is the forward path for wheel distribution.
* **Conda:** engaging with conda-forge to add a `linux-riscv64` platform so that PyTorch's conda channel has a viable path on RISC-V, tracked in [conda-forge/conda-forge.github.io#1744](https://github.com/conda-forge/conda-forge.github.io/issues/1744).

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
