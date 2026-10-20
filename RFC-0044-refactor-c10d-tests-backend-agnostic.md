# [RFC] Refactor test_c10d_nccl.py to Support Backend-Agnostic Testing for PyTorch

**Authors:**
* @harikodali

## **Summary**
Propose splitting the `test_c10d_nccl.py` file into two separate files: `test_c10d_nccl.py` for NCCL-specific tests and `test_c10d_backend.py` for backend-agnostic tests that can run on multiple device backends (e.g., NCCL, HCCL, XCCL) by dynamically detecting the device type and corresponding backend at runtime. This will enhance test coverage for non-CUDA accelerators like Intel Gaudi (HPU) and Intel XPU in the official PyTorch repository.

## **Motivation**
The `test_c10d_nccl.py` file contains tests for collectives, process group initialization, Distributed Data Parallel (DDP), and other NCCL-specific functionality. While these tests are critical for validating CUDA-based NCCL backends, they are not reusable for other backends like Intel's HCCL for Gaudi devices and Intel's XCCL or other non-CUDA accelerators.
To address this, we propose refactoring `test_c10d_nccl.py` to extract common, backend-agnostic tests into a new `test_c10d_backend.py` file. This file will dynamically detect the device type and backend at runtime, allowing tests to run on any supported accelerator (e.g., CUDA, HPU, XPU) without hardcoding backend-specific logic. This approach aligns with prior efforts to generalize PyTorch tests for non-CUDA devices, enabling broader test coverage and native support for accelerators like HPU/XPU in the official PyTorch repository.

### **Examples**
The following test classes in `test_c10d_nccl.py` can be generalized to run on multiple backends:
* `RendezvousEnvTest`, `TimeoutTest`, `ProcessGroupNCCLInitTest`, `DistributedDataParallelTest`, `WorkHookTest`, `CommTest`, `NcclProcessGroupWithDispatchedCollectivesTests`, `LargeCommTest`, `SparseCollective`: These tests can be refactored to use device-agnostic APIs.

Currently, these tests are hardcoded for NCCL and CUDA, using decorators like `@requires_nccl()` or explicit `"nccl"` backend references, which block execution on non-CUDA devices.

## **Proposed Implementation**
We propose a staggered approach to refactor `test_c10d_nccl.py` and introduce `test_c10d_backend.py`:

1. **Extract Backend-Agnostic Tests**:
   - Move the identified test classes (`RendezvousEnvTest`, `TimeoutTest`, etc.) from `test_c10d_nccl.py` to `test_c10d_backend.py`.
   - Replace NCCL-specific decorators (e.g., `@requires_nccl()`) with a generalized `@requires_accelerator_dist_backend` that checks for backend availability dynamically.

2. **Dynamic Device and Backend Detection**:
   - Use `torch.accelerator.current_device()` and runtime backend detection (e.g., `torch.distributed.get_default_backend_for_device()`) to substitute hardcoded `"cuda"` and `"nccl"` references with the current device type and corresponding backend.

3. **Generalize Test Logic**:
   - Refactor test logic to use abstract backend APIs instead of NCCL-specific calls.


### **Metrics**
- **Adoption**: Enable other non-CUDA devices to use `test_c10d_backend.py` in their CI pipelines, increasing participation in PyTorch’s test ecosystem.

### **Additional Context**
This proposal builds on prior efforts to generalize PyTorch tests for non-CUDA devices:
- Removed `@onlyNativeDevice` restrictions: [PR #128584](https://github.com/pytorch/pytorch/pull/128584)
- Generalized Dynamo content: [PR #130714](https://github.com/pytorch/pytorch/pull/130714)
- Generalized Distributed content: [PR #131758](https://github.com/pytorch/pytorch/pull/131758)
- Generalized FSDP content: [PR #133209](https://github.com/pytorch/pytorch/pull/133209)

These efforts enabled native test support for Intel Gaudi (HPU) by dynamically substituting device and backend types, allowing tests to run seamlessly on other devices. This RFC extends the same philosophy to distributed backend tests, ensuring broader backend support.