# DirectML Backend for PyTorch on Windows and Windows Subsystem for Linux

**Authors:**
* @nickname
* @nickname 


## Background
Direct Machine Learning (DirectML) is a low-level API for machine learning. It has a familiar (native C++, nano-COM) programming interface and workflow in the style of DirectX 12. DirectML allows the integration of machine learning workloads into games, engines, middleware, backends, or other applications. DirectML is supported by all DirectX 12-compatible hardware and is layered on top of the Compute Driver Model which is the common abstraction layer to provide consistent, reliable, and secure execution across all Windows hardware devices and vendors.
 
Many companies currently use DirectML for inferencing ML models in Windows applications. Some examples are Adobe, GE Healthcare, Logitech, and Corel. Check out this blog post to learn more. DirectML sits within a tightly integrated stack of technologies built into Windows called the Windows AI Platform.  Each layer has a specific purpose, and together they give developers the combined promise of high performance and ease of use with large scale hardware reach. The Windows AI Platform aims to make the job of training and inferencing an ML model on Windows and Windows Subsystem for Linux easier by providing one platform that works across the breadth of Windows devices. Following the launch of the DirectML backend for training TensorFlow models, one of the most common requests is to provide similar support for PyTorch.

## Motivation
Students and machine learning enthusiasts have expressed a desire to accelerate machine learning training workloads on their existing hardware and not be tied to a hardware specific library. Many of these devices are running Windows and bringing a DirectML backend to PyTorch unlocks accelerated training and inference on any DirectX12 compatible GPU. With a DirectML backend for PyTorch, DirectML can now support training ML models for new and existing customers. This bridges the gap in the ML workflow from training to inferencing all while using DirectML’s hardware acceleration capabilities.
As the DirectML team builds up support for a PyTorch backend, the focus is targeting student workflows to enable PyTorch beginners to utilize their existing Windows machines. By also integrating with Windows Subsystems for Linux, students and enthusiasts can use the Linux tools they are already familiar with and that are common in the machine learning community. Then, the focus will shift to enabling professional data scientist workflows in order to support new and existing customers of DirectML. 
Since the initial preview release of a DirectML backend for PyTorch there have been over 6,400 downloads on PyPI, and an extremely positive response on multiple online forums. The diagram below shows an updated version of the DirectML entry points to include the PyTorch backend:
 
## Proposal
We propose to add a DirectML backend as an out-of-tree plugin extension for PyTorch.
The DirectML backend will provide hardware acceleration across a broad spectrum of hardware, via the comprehensive implementation of ATen functions. 
This will enable training scenarios across DirectX12 supported hardware on Windows 10, Windows 11, or Ubuntu-20.04 through the Windows Subsystems for Linux. 
We plan to achieve maximal code reuse between existing PyTorch scripts that target CPU or other backend accelerators. For example,
import torch
import torch_directml as dml
a = torch.ones((4,4), device=dml.device())
print(a.device)
print(a)

### Architecture 
The PyTorch dispatcher needs to be enlightened to recognize DirectML as a backend.
Per Extending Dispatcher for a new backend in C++ article, DirectML will extend the PyTorch dispatcher, but will not use the prototyping PrivateUse1 dispatch keys. Instead, DirectML will appear as a named backend in the PyTorch repository. The implementation of the DirectML backend would follow the out-of-tree backend architecture like that of the XLA and ORT backends.
The changes needed for an out-of-tree backend are summarized in this ORT Backend PR https://github.com/pytorch/pytorch/pull/58248. This mechanism has been validated in a Microsoft internal development fork of Torch for DirectML backend development.

### Operator Registration and Support
The PyTorch dispatcher routes ATen functions to registered DirectML kernels when the implementations have been registered.
The registration for DirectML supported operators is performed using the torchgen/gen_backend_stubs.py utility.
The resulting generated registrations are built into the backend using the torch.utils.cppextension (https://pytorch.org/tutorials/advanced/cpp_extension.html).
Currently 330 operators have been implemented in our pre-releases.


### Device Enumeration
Since DirectML works on a wide spectrum of hardware, we will also offer enumeration APIs to help users pick the appropriate device for training. 
import torch
import torch_directml as dml
print(dml.device_count()) # returns: 2
print(dml.device_name(0)) # returns: 'NVIDIA GeForce GTX 1060 6GB'
print(dml.device_name(1)) # returns: 'Intel(R) UHD Graphics 630'
print(dml.device_name(2)) # returns: '' <-- error: invalid argument
'#' Get the default DirectML device
'#' returns: device(type=’dml’, index=0)
a = torch.ones((4,4), device=dml.device()) 
'#' Get specific DirectML device
'#' returns: device(type=’dml’, index=1)
a = torch.ones((4,4), device=dml.device(1))
'#' Get invalid DirectML device
'#' returns: ‘Exception: Invalid device_id argument supplied 2. device_id must be in range [0, 2).’
a = torch.ones((4,4), device=dml.device(2))

## Telemetry
We will collect the following data, ensuring that we fully inform the user and give the option to turn off data collection:
- Windows version
- Hardware device type
- Monthly active devices 
Telemetry is also controlled at an OS level, so if the user has opted out of telemetry altogether then no data will be collected.

## Testing and Support
We validate DirectML backend against the following test matrix of parameters.
Platform: Windows10, Windows11, Ubuntu-20.04
Python Versions: 3.7, 3.8, 3.9, 3.10
PyTorch Versions: main
Hardware (Support any DX12 capable device, but test the following IHV configurations):
- Intel UHD Graphics 630 (Coffee Lake) 
- AMD Radeon RX 5700 XT (RDNA1)
- AMD Radeon RX 580
- AMD Radeon 6900XT  
- NVIDIA GeForce GTX 1070 (Pascal)
- NVIDIA GeForce RTX 3070 (Ampere)
- NVIDIA GeForce RTX 2080 SUPER (Turing) 
- Qualcomm Adreno 680

## Documentation
Currently, we have documentation pages for Windows and WSL here (https://docs.microsoft.com/en-us/windows/ai/directml/gpu-pytorch-windows). We would like to have a documentation page on pytorch.org similar to the XLA page. 

## Release Plan
The PyTorch+DirectML backend will be distributed as torch_directml, a single python package on PyPI.
NOTE: Currently we distribute the pytorch-directml package on PyPI. This package provides the full functionality of Torch and is built from a Microsoft internal fork of the Torch project that has been enlightened to support a DirectML backend in-tree. This package will be deprecated in favor of the plugin model in the next release of DirectML.

### Release Cadence
The DirectML team plans to release three times per year and will support the latest version of PyTorch and new PyTorch features within 6 months. PyTorch-DirectML plans to ship with new features and support with every official PyTorch release.

### Issue Tracking
We plan to open source the DirectML backend in a GitHub repository. We will use the GitHub repository to manage and track issues submitted by users. If an issue is created on the main PyTorch repository, we will respond and track accordingly.
