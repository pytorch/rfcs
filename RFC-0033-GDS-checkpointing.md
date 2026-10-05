# Checkpointing utilizing GDS

**Authors:**

* @antferdom


## **Summary**
The effectiveness of scaling laws and [emergent capabilities ](https://arxiv.org/abs/2206.07682)found in LLMs implies that scaling model parameters up to regimes of billions (*e.g.* [PaLM](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)) or even trillions ([ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning](https://arxiv.org/pdf/2104.07857.pdf)) is a relevant engineering problem that leads to better model generalization. Checkpointing becomes the memory-bandwidth bottleneck in these regimes due to significant temporary tensor allocations in system memory before allocation on the associated device (*e.g.* CUDA). This complexity is present in both training and inference scenarios. Therefore, bypassing the CPU can reduce system memory pressure and lead to the device directly communicating with the disk.

![](https://d29g4g2dyqv443.cloudfront.net/sites/default/files/akamai/GPUDirect/cuda-gpu-direct-blog-refresh_diagram_1.png)

In this RFC, we strictly target NVIDIA GPUs and its new storage technology, which enables a direct data path for direct memory access (DMA) transfers between GPU memory and storage, called [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html#abstract) (GDS). Therefore, practical zero-copy checkpoint loading becomes feasible since we eliminate the canonical path `torch.load` (Storage->CPU->GPU).


## **Motivation**
We aim to leverage a direct path between the disk and GPU for efficiently reading and writing the model parameters. For the CUDA ecosystem, we explore [cuFile](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html), which enables [GPUDirect Storage (GDS)](https://developer.nvidia.com/blog/gpudirect-storage/). Removing the intermediate CPU tensor allocations and memory-bandwidth interchange between disk storage technologies and CPU previous to device memory allocation would bring new research opportunities and increase the performance of existing ones.

The CPU would not be primarily busy by just moving data and acting as a mere connection memory node between disk and device. Therefore, we can theoretically speed up checkpoint serialization and deserialization and off-load some portion of the given job computation to the CPU in a more efficient paradigm.

This feature can open discussion about extending the underlying transfer type used in `torch.load` and `torch.save`, in this case, GPUDirect.


## **Proposed Implementation**
This feature would be added as an additional **transfer type** argument for both `torch.load` and `torch.save` function calls. The following list of the available transfer type alternatives:

- Storage->GPU (GDS)
- Storage->CPU
- Storage->CPU->GPU
- Storage->CPU->GPU_ASYNC
- Storage->PAGE_CACHE->CPU->GPU
- Storage->GPU_ASYNC
- Storage->GPU_BATCH

We recommend using [KvikIO](https://github.com/rapidsai/kvikio), a Python and C++ library for high-performance file IO. It provides C++ and Python bindings to cuFile.

The following example is a general overview of interacting with cuFile from Python using KvikIO.

Filename: **cutorch.py**

```python
# %%
import kvikio
import kvikio.defaults
from kvikio.defaults import (
    get_num_threads,
    set_num_threads,
)
import cupy as cp
import torch

import logging
import time
import os

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


TENSOR_DIMS = (50_000, 50_000)
before = get_num_threads()

print(f"kvikio number of threads: {before}")
# %%
"""
PyTorch also supports __cuda_array_interface__
zero-copy data exchange between CuPy and PyTorch can be achieved at no cost. 
"""
a = torch.rand((4,4), device="cuda")
b = cp.asarray(a)
logger.info(f"Tensor a: {a}")
logger.info(f"Tensor b: {b}")
# check the underlying memory pointer is the same
assert a.__cuda_array_interface__['data'][0] == b.__cuda_array_interface__['data'][0]
# %%
# convert a cupy array to torch tensor
a = cp.arange(10)
b = torch.as_tensor(a, device="cuda")
b += 1
print(f"Tensor b: {b}")
print(f"Tensor a: {a}")
# %%

# %%
# torch tensor -> numpy ndarray -> cupy array -> torch tensor
import numpy as np

from kvikio.numpy import fromfile, tofile
from numpy.typing import ArrayLike, DTypeLike

x = torch.empty(10_000, 10_000)
print(x)
print(x.shape)
x_np = x.detach().cpu().resolve_conj().resolve_neg().numpy()
x_np.tofile("tensor_program")
x_cu = fromfile(file="tensor_program", dtype=float, like=cp.empty(()))
print(x_cu)
print(f"Array type: {type(x_cu)}")
print(f"Device: {x_cu.device}")
# convert a cupy array to torch tensor
x_cutorch = torch.as_tensor(x_cu, device="cuda")
print(x_cutorch)
print(f"Device: {x_cutorch.device}")
# %%
# torch tensor -> cupy array -> torch tensor
import kvikio

st = time.perf_counter_ns()
x = torch.empty(*TENSOR_DIMS, device="cuda")
x_cu = cp.asarray(x)
# Write whole array to file
f = kvikio.CuFile("tensor_program", "w")
f.write(x_cu)
f.close()
et = time.perf_counter_ns() - st
print(f"cuFile serilization elapsed time: {et*1e-9:.2f} s")
del x, x_cu
torch.cuda.empty_cache()
# %%
# torch native serialization
st = time.perf_counter_ns()
x = torch.empty(*TENSOR_DIMS, device="cuda")
torch.save(x, "native_tensor_.pt")
et = time.perf_counter_ns() - st
print(f"Elapsed time: {et*1e-9:.2f} s")
# %%
# deserialize a torch tensor from a CuFile
import cupy
# import cunumeric as num

tensor_size = os.path.getsize("tensor_program")
logger.info(f"Tensor size: {tensor_size / 2**30:.2f} GB")
x_cu = cp.asarray(torch.empty(*TENSOR_DIMS, device="cuda"))
# x_cu = cp.empty(shape=(50_000, 50_000))
with kvikio.defaults.set_num_threads(32):
    assert get_num_threads() == 32
    st = time.perf_counter_ns()
    f = kvikio.CuFile("tensor_program", "r")
    f.read(x_cu)
    x_cutorch = torch.as_tensor(x_cu, device="cuda")
    et = time.perf_counter_ns() - st
    logger.info(f"Tensor loading time (cuarray -> torch.Tensor): {et*1e-9:.2f} s")
print(x_cutorch)
print(f"Device: {x_cutorch.device}")
# %%
del x_cutorch, x_cu
torch.cuda.empty_cache()
# %%
# torch native deserialization
tensor_size = os.path.getsize("native_tensor_.pt")
logger.info(f"Tensor size: {tensor_size / 2**30:.2f} GB")
st = time.perf_counter_ns()
x = torch.load("native_tensor_.pt")
logger.info(f"Elapsed time: {(time.perf_counter_ns() - st)*1e-9:.2f} s")
# %%
```



## **Metrics **

The main metric for this feature is the performance boost regarding IO speed; thus, we try to follow measuring schemas present in other standard Linux IO tester utilities like [fio](https://github.com/axboe/fio).

From the [fio HOWTO](https://github.com/axboe/fio/blob/a142e0df6c1483a76d92ff7f9d8c07242af9910e/HOWTO.rst#L2643)
**bw**

> Aggregate bandwidth of threads in this group followed by the minimum and maximum bandwidth of all the threads in this group. Values outside of brackets are power-of-2 format and those within are the equivalent value in a power-of-10 format.

_e.g._ **fio Output:**
```bash
Run status group 0 (all jobs):
   READ: bw=2613MiB/s (2740MB/s), 163MiB/s-181MiB/s (171MB/s-190MB/s), io=64.0GiB (68.7GB), run=22596-25078msec

Disk stats (read/write):
  vda: ios=86014/479, sectors=134213632/5184, merge=0/171, ticks=492865/5535, in_queue=297040, util=97.62%
```

NVIDIA also provides a IO micro-benchmarking utility for the specific use case of GPUDirect. [gdsio](https://docs.nvidia.com/gpudirect-storage/configuration-guide/index.html) It supports a series of command line arguments to specify the target files, file sizes, IO sizes, number of IO threads, etc. The following is a Python wrapper of such utility:

```python
import subprocess
import os
import pathlib
from typing import Optional, Union, List, Tuple, Dict

import pandas as pd
import seaborn as sns

# Globals
GDSIO_PATH: str = "/usr/local/cuda-11.8/gds/tools/gdsio"
WD = os.path.dirname(os.path.abspath(__file__))
GDSIO_DIR = os.path.join(WD, "gds_benchmarks/")
DEVICE = 0
NUMA_NODE = 0
LOAD = "randread"

LOAD_TYPE = {"read": 0, "write": 1, "randread": 2, "randwrite": 3}
"""
-x <xfer_type> [0(GPU_DIRECT), 1(CPU_ONLY), 2(CPU_GPU), 3(CPU_ASYNC_GPU), 4(CPU_CACHED_GPU), 5(GPU_DIRECT_ASYNC), 6(GPU_BATCH)]
xfer_type:
    0 - Storage->GPU (GDS)
    1 - Storage->CPU
    2 - Storage->CPU->GPU
    3 - Storage->CPU->GPU_ASYNC
    4 - Storage->PAGE_CACHE->CPU->GPU
    5 - Storage->GPU_ASYNC
    6 - Storage->GPU_BATCH
"""
transfer_type = {"GDS": 0, "CPU_ONLY": 1, "CPU_GPU": 2, "CPU_ASYNC_GPU": 3, "CPU_CACHED_GPU": 4, "GPU_ASYNC": 5, "GPU_BATCH": 6}


def init_gds_files(gdsio_path: Union[pathlib.Path, str],
                   output_dir: Union[pathlib.Path, str],
                   file_size: str,
                   device: int,
                   workers: int) -> None:
    command = f"{gdsio_path} -D {output_dir} -d {device} -T 1 -s {file_size} -w {workers} -I 3"
    subprocess.run(command.split())

def main(gdsio_path: Union[pathlib.Path, str],
         output_dir: Union[pathlib.Path, str],
         device: int,
         numa_node: int,
         load_type: str) -> None:
    file_size = "30G"
    io_sizes = ["128K", "256K", "512K", "1M", "4M", "16M", "64M", "128M"]
    threads = [1, 4, 16, 32]
    runtime = "30"

    os.makedirs(output_dir, exist_ok=True)
    # if benchmark files do not exist, create them
    if not os.path.isfile(os.path.join(output_dir, f'gdsio.{max(threads) - 1}')):
        init_gds_files(gdsio_path, output_dir, file_size, device, max(threads))


    stats = {"Transfer Type": [], 
             "Threads": [],
             "DataSetSize": [],
             "IOSize": [],
             "Throughput": [], 
             "Avg_Latency:": [],
             "total_time": [], 
             }

    command_global = f"{gdsio_path} -D {output_dir} -d {device} -n {numa_node} -T {runtime} -s {file_size}"
    for io_size in io_sizes:
        for thread in threads:
            for transfer_name, x in transfer_type.items():
                command_job = command_global + f" -i {io_size} -w {thread} -x {x} -I {LOAD_TYPE[load_type]}"
                print(f"Running {command_job}")
                res = subprocess.run(command_job.split(), capture_output=True).stdout
                print(res)
                res = str(res).split(" ")
                latency = float(res[res.index("Avg_Latency:") + 1])
                throughput = float(res[res.index("Throughput:") + 1])
                data_set_size: str = res[res.index("DataSetSize:") + 1]

                stats["Transfer Type"].append(transfer_name)
                stats["Threads"].append(thread)
                stats["DataSetSize"].append(data_set_size)
                stats["IOSize"].append(io_size)
                stats["Throughput"].append(f"{throughput*1e-9:.2f}")
                stats["Avg_Latency:"].append(f"{latency*1e-6:.2f}")


    df = pd.DataFrame.from_dict(stats)
    df.to_csv(f'gds_bench_save_device_{device}_numa_{numa_node}_{load_type}.csv')

if __name__ == '__main__':
    main(GDSIO_PATH, GDSIO_DIR, DEVICE, NUMA_NODE, LOAD)
```



## **Drawbacks**

This type of storage technologies are highly dependent on the hardware vendors, NVIDIA in this case, and PyTorch might want to lead the development of such integrations to them entirely.
