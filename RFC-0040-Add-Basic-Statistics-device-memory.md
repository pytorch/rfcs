**Authors:**
* @lichuyang 


## **Summary**
This RFC proposes adding basic memory statistics tracking to OpenRegDeviceAllocator by migrating from `at::Allocator` to `c10::DeviceAllocator` interface, enabling essential observability features.

## **Motivation**
The current Device Allocator implementation in OpenReg provides only the most basic memory allocation functionality. It directly calls orMalloc() on every allocation request and orFreeHost() on every deallocation. While this approach demonstrates the minimal integration path for PrivateUse1 backends, it lacks essential observability features that are fundamental to any other memory allocator.

Currently, when users work with OpenReg devices, they have zero visibility into memory consumption. This "black box" behavior makes it difficult for users to debug out-of-memory (OOM) errors, optimize memory usage and monitor long-running applications, etc. Moreover, all major PyTorch backends (CUDA, XPU) provide memory statistics APIs. For backend developers, the current implementation doesn't demonstrate the minimum viable statistics implementation that real backends should provide.

The implementation of basic statistics will promote the ecosystem compatibility and enhance debugging for further development.

## **Proposed Implementation**
This proposal enhances the OpenRegDeviceAllocator with memory statistics tracking and cache management capabilities, following PyTorch's established patterns used by CUDA and XPU backends. The implementation will leverage existing PyTorch infrastructure (c10::CachingDeviceAllocator::DeviceStats).

Here is the minimal implementation:
```c++
namespace c10::openreg {

class OpenRegDeviceAllocator final : public c10::DeviceAllocator {
 private:
  c10::CachingDeviceAllocator::DeviceStats stats_;
  bool initialized_ = false;

 public:
  OpenRegDeviceAllocator() = default;

  at::DataPtr allocate(size_t nbytes) override {
    //...
    
    void* data = nullptr;
    if (nbytes > 0) {
      orMalloc(&data, nbytes);
      TORCH_CHECK(data, "Failed to allocate ", nbytes, " bytes on openreg device.");
      
      stats_.allocated_bytes[0].increase(nbytes);
      stats_.reserved_bytes[0].increase(nbytes);
    }
    
    auto curr_device = c10::Device(c10::DeviceType::PrivateUse1, current_device_index);
    return {data, data, &ReportAndDelete, curr_device};
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }

  void copy_data(void* dest, const void* src, std::size_t count) const final {
    orMemcpy(dest, src, count, orMemcpyDeviceToDevice);
  }

  bool initialized() override {
    return initialized_;
  }

  void emptyCache(MempoolId_t mempool_id = {0, 0}) override {
    // TODO: for further development of memory caching
  }

  void recordStream(const DataPtr& ptr, c10::Stream stream) override {
    // TODO: for further development of memory caching
  }

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex device) override {
    return stats_;
  }

  void resetAccumulatedStats(c10::DeviceIndex device) override {
    for (size_t i = 0; i < stats_.allocated_bytes.size(); ++i) {
      stats_.allocated_bytes[i].reset_accumulated();
      stats_.reserved_bytes[i].reset_accumulated();
    }
    stats_.num_alloc_retries = 0;
  }

  void resetPeakStats(c10::DeviceIndex device) override {
    for (size_t i = 0; i < stats_.allocated_bytes.size(); ++i) {
      stats_.allocated_bytes[i].reset_peak();
      stats_.reserved_bytes[i].reset_peak();
    }
  }

 private:
  static void ReportAndDelete(void* ptr) {
    if (!ptr) return;
    orFreeHost(ptr);
  }
};

// ...

}
```


### Next Steps
Will implement it.

#### Tracking issue
<https://github.com/pytorch/pytorch/issues/158917>