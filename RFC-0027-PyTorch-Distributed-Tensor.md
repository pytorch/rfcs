# RFC: PyTorch DistributedTensor

We propose distributed tensor primitives to allow easier distributed computation authoring in SPMD(Single Program Multiple Devices) paradigm. The primitives are simple but powerful when used to express tensor distributions with both sharding and replication parallelism strategies. This could empower native Tensor parallelism among other advanced parallelism explorations. For example, to shard a big tensor across devices with 3 lines of code:

```python
import torch  
from torch.distributed import DeviceMesh, Shard, distribute_tensor  
  
# Create a mesh topology with the available devices.
mesh = DeviceMesh("cuda", list(range(world_size)))  
big_tensor = torch.randn(100000, 88)  
# Shard this tensor over the mesh by sharding `big_tensor`'s 0th dimension over the 0th dimension of `mesh`.
my_dtensor = distribute_tensor(big_tensor, mesh, [Shard(dim=0)])
```

To see a complete design doc for this proposal, please refer to this [doc](https://docs.google.com/document/d/1nFeJ8NSFNhNlCkNgWK31ZGRqm1L9rd0i_XN_RprphaI/edit#heading=h.6sovjqv9jiqn)

## Motivation

Today there are mainly three ways to scale up distributed training: Data Parallel, Tensor Parallel and Pipeline Parallel. Each of them works on a separate dimension where solutions have been built independently (i.e. PyTorch DDP, FSDP, ShardedTensor, PiPPy, etc.). When training really large models, users would like to use these technologies together (i.e. 3-D Parallelism), while the interoperability of the existing solutions are not great and often hard to use (i.e. users might want arbitrary combinations of the data parallel, tensor parallel and pipeline parallel). This is becoming an issue for users and one of the biggest reasons is that there’s no common abstractions that build the bridge between different parallelism strategies.

An ideal scenario is that users could just build their models like in a single node/device, without worrying about how to do distributed training in a cluster, and our solutions could help them run distributed training in an efficient manner. For example, researchers just need to build their big transformer model, and PyTorch Distributed automatically figures out how to split the model and run pipeline parallel across different nodes, how to run data parallel and tensor parallel within each node. In order to achieve this, we need some common abstractions to represent data distribution and run the distributed computation.
  
There're many recent works that working on tensor level parallelism to provide common abstractions, see the `Related Works` in the last section for more details. Inspired by [GSPMD](https://arxiv.org/pdf/2105.04663.pdf), [Oneflow](https://arxiv.org/pdf/2110.15032.pdf) and [TF’s DTensor](https://www.tensorflow.org/guide/dtensor_overview), we introduce a DistributedTensor concept to represent generic data distributions across hosts. DistributedTensor is the next evolution of ShardedTensor and provides basic abstractions to distribute storage and compute. It serves as one of the basic building blocks for distributed program translations and describes the layout of a distributed training program. With the DistributedTensor abstraction, we can seamlessly build parallelism strategies such as tensor parallelism, DDP and FSDP.

## Value Propsition

DistributedTensor primarily:
-   Offers a uniform way to save/load state dict during checkpointing, even when there’re complex data distribution strategies such as combining tensor parallelism with parameter sharding in FSDP.
-   Could natively offer Tensor Parallelism solution in eager mode, just like our current ShardedTensor solution. Moreover, it gives additional flexibility for advanced users who want to mix sharding and replication.
-   Could be the entry point of a SPMD programming model for ML System Engineers, providing good UX to mix up different types of parallelism, and could be used as a fundamental building block of a compiler based distributed training.

## PyTorch DistributedTensor

### DistributedTensor API

We offer both a lower level DistributedTensor API and a module level API to create a `nn.Module` with “distributed” parameters.

#### Basic DistributedTensor API Examples

Here are some basic DistributedTensor API examples that showcase: 
1. How to construct a DistributedTensor directly, to represent different types of sharding, replication, sharding + replication strategies.
2. How to create DistributedTensor from a local `torch.Tensor`.
3. How to “reshard” an existing DistributedTensor to a different DistributedTensor with modified placement strategy or world size.

```python
import torch  
import torch.distributed as distributed  
from torch.distributed import DTensor, DeviceMesh, Shard, Replicate, distribute_module  

# initialize a nccl process group on each rank
distributed.init_process_group(backend="nccl")
  
# construct a device mesh with available devices (multi-host or single host)  
device_mesh = DeviceMesh(device_type="cuda", [0, 1, 2, 3])  
# if we want to do row-wise sharding  
rowwise_placement=[Shard(0)]  
# if we want to do col-wise sharding  
colwise_placement=[Shard(1)]  
# distributed tensor returned will be sharded across the dimension specified in placements  
distributed.empty((8, 12), device_mesh=device_mesh, placements=rowwise_placement)  
  
# if we want to do replication across a certain device list  
replica_placement = [Replicate()]  
# distributed tensor will be replicated to all four GPUs.  
distributed.empty((8, 12), device_mesh=device_mesh, placements=replica_placement)  
  
# if we want to distributed a tensor with both replication and sharding  
device_mesh = DeviceMesh(device_type="cuda", [[0, 1], [2, 3]])  
# replicate across the first dimension of device mesh, then sharding on the second dimension of device mesh  
spec=[Replicate(), Shard(0)]  
distributed.empty((8, 8), device_mesh=device_mesh, placements=spec)  
  
# create a DistributedTensor that shards on dim 0, from a local torch.Tensor  
local_tensor = torch.randn((8, 8), requires_grad=True)  
rowwise_tensor = DTensor.from_local(local_tensor, device_mesh, rowwise_placement)  
  
# reshard the current rowise tensor to a colwise tensor or replicate tensor  
colwise_tensor = rowwise_tensor.redistribute(device_mesh, colwise_placement)  
replica_tensor = colwise_tensor.redistribute(device_mesh, replica_placement)

```

#### High level User Facing APIs

Users can use DistributedTensor tensor constructors directly to create a distributed tensor (i.e. `distributed.ones/empty`), but for existing modules like nn.Linear that are already having torch.Tensor as parameters, how to make them distributed parameters? We offer a way to directly distribute a torch.Tensor and a module level APIs to directly distribute the module parameters. Below is the high level API we introduce:

```python
def distribute_tensor(tensor: torch.Tensor, device_mesh: DeviceMesh=None, placements: List[Placement]=None):  
    '''
    distribute the tensor according to device_mesh and placements, `tensor` could be a "meta" tensor.  
    '''  
  
def distribute_module(  
    module: nn.Module,  
    device_mesh: DeviceMesh=None,  
    partition_fn: Callable[[str, nn.Module], ...]=None,
    input_fn: Callable[torch.Tensor]=None,  
    output_fn: Callable[torch.Tensor]=None,  
):  
    '''  
    This function converts all module parameters to distributed tensor parameters according to the `partition_fn` specified.  
    It could also control the input/output of the module by specifying the `input_fn` and `output_fn`. 
    '''
```

#### High level API examples:

```python 
def MyModule(nn.Module):  
    def __init__(self):  
        super.__init__()  
        self.fc1 = nn.Linear(8, 8)  
        self.fc2 = nn.Linear(8, 8)  
        self.relu = nn.ReLU()  
     
    def forward(self, input):  
        return self.relu(self.fc1(input) + self.fc2(input))  
  
mesh = DeviceMesh(device_type="cuda", [[0, 1], [2, 3]])

def shard_params(mod_name, mod):  
    rowwise_placement = [Shard(0)]
    def to_dist_tensor(t): return distribute_tensor(t, mesh, rowwise_placement)  
    mod._apply(to_dist_tensor)  

sharded_module = distribute_module(model, device_mesh, partition_fn=shard_params)  
  
def shard_fc(mod_name, mod):  
    rowwise_placement = [Shard(0)]
    if mod_name == "fc1":  
        mod.weight = torch.nn.Parameter(distribute_tensor(mod.weight, mesh, rowwise_placement))

sharded_module = distribute_module(model, device_mesh, partition_fn=shard_fc)
```

## Compiler and DistributedTensor

DistributedTensor provides efficient solutions for cases like Tensor Parallelism. But when using the replication more often with DistributedTensor, it might become observably slow compared to our existing solutions like DDP/FSDP. This is mainly because replication in eager mode is part of data parallel, and existing solutions like DDP/FSDP could have the global view of entire model architecture, thus could specifically optimize for data parallel, i.e. using collective fusion and computation overlap, etc. DistributedTensor itself is only a Tensor-like object and only knows its local computation operation, it does not know the subsequent operations that happened afterwards.

In order to recover the performance when using DistributedTensor directly to do training (i.e. Users might want to use DistributedTensor to do DDP-like training), DistributedTensor also needs the global view to do things like communication optimization. We are exploring a compiler based solution accompanied with DistributedTensor so that we could run optimizations on top of it, which will be shared later.

## Related Works

This work is mainly inspired by [GSPMD](https://arxiv.org/pdf/2105.04663.pdf), [Oneflow](https://arxiv.org/pdf/2110.15032.pdf) and [TF’s DTensor](https://www.tensorflow.org/guide/dtensor_overview). All of these three works use a single “distributed tensor” concept for both replication and sharding, and the solutions could enable users to build up their distributed training program in a uniform SPMD programming model. Specifically:

GSPMD: 
-   GSPMD is now the fundamental component of JAX/TensorFlow distributed training and enables various optimizations with the XLA compiler to allow users to train their models efficiently in a large scale setting. 
-   Fundamentally, GSPMD have three types of sharding strategies within a tensor: “tiled”, “replicated”, “partially tiled” to represent sharding and replication.
-   At the core of GSPMD Partitioner, it utilizes the XLA compiler to do advanced optimizations, i.e. sharding propagation and compiler based fusion. 
-   XLA mark_sharding API: PyTorch XLA’s [mark_sharding](https://github.com/pytorch/xla/pull/3476) API uses [XLAShardedTensor](https://github.com/pytorch/xla/issues/3871) abstraction (i.e. sharding specs) in PyTorch/XLA. Under the hood XLAShardedTensor is utilizing the GPSMD partitioner to enable SPMD style training on TPU.

OneFlow GlobalTensor: 

-  OneFlow is building up their own solution of the “GlobalTensor” concept, which is a variant form of GSPMD sharding, allowing users to explore different parallel strategies with GlobalTensor. 
-  OneFlow also has three types of tensor, but they are slightly different from GSPMD: “split”, “broadcast”, and “partial sum”. They don’t use partially tiled and instead have a concept of partial sum to partition the values.

TensorFlow DTensor:
-   [DTensor Concepts](https://www.tensorflow.org/guide/dtensor_overview) is an extension of TensorFlow synchronous distributed training. its sharding API, supported features and its compilation passes with MLIR.
-   DTensor also allows sharding and replication on an n-d mesh like device network.
-   DTensor implements MLIR passes to do propagation and operator implementations.

There are also several cutting edge research fields that embeds tensor sharding as part of the system, i.e. [Megatron-LM](https://arxiv.org/pdf/1909.08053.pdf) for tensor parallelism on Transformer based models. [DeepSpeed](https://github.com/microsoft/DeepSpeed) for training large scale models with different optimization techniques on top of tensor sharding.

In PyTorch, we have existing [ShardedTensor](https://docs.google.com/document/d/1WEjwKYv022rc1lSrYcNWh3Xjx9fJC7zsrTKTg0wbPj0/edit?usp=sharing) work in the prototype stage, which introduces basic PyTorch sharding primitives as our Tensor Parallelism solution. But ShardedTensor only has tensor sharding support, which makes it hard to be used by users to describe other data distributions strategies like replication or replication + sharding. As a distributed system developer who wants to explore more parallelism patterns, it’s crucial to have a basic building block that describes the data distribution in a uniform way. This DistributedTensor RFC aims at solving this and provide a fundamental abstraction for distributed training.
