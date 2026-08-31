# RFC: PyTorch Tensor Parallelism User API

## Background

Provide a detailed API design for high-level PyTorch Tensor Parallelism API design. This is an evolvement of PyTorch Sharding introduced in https://github.com/pytorch/pytorch/issues/72138 and is directly built on top of DTensor proposed in https://github.com/pytorch/pytorch/issues/88838. We want users to only focus on how their modules to be distributed and hide all other details. (Caveat, for now, we only support linear/transformer based models). 

## Motivation

To scale the large model training, especially transformer based model training, multiple parallelism paradigms are proposed and considered. Among them, model parallelism like Megatron-LM is getting popular together with 3D parallelism. We have already proposed a standardized sharding api in the past (https://github.com/pytorch/pytorch/issues/72138). Now to enable more generic data distributions more than sharding across hosts, we have proposed a new design of Distributed Tensor (DTensor) in https://github.com/pytorch/pytorch/issues/88838 and we want to not only provide similar functionality of model parallelism as Megatron on top of DTensor, but also provide better usability so that users don't need to change their model to use tensor parallelism directly.

## Design
We are proposing APIs which cover three different use cases during module annotation. These APIs not only include the TP-only case, it also covers 2D parallel and 3D parallel down the road.

- One base Parallel Style Class and three children in-house parallel style. This is extendible so that users can create their own parallel styles if the in-house ones do not meet their requirements.

```python
class ParallelStyle(ABC):    
"""    
The parallel style user wants the module or submodule to be parallelized. 
We can add more in future, but this seems sufficient for immediate needs. 
Users can extend this class to build their own parallel style with customized input/output preparations.    
"""    
_prepare_input: Callable[[Union[Tensor, DTensor], Optional[DeviceMesh], Optional[Int]], DTensor]    
_prepare_output: Callable[[DTensor, Optional[DeviceMesh], Optional[Int]], Union[Tensor, DTensor]]     


class RowwiseParallel(ParallelStyle):    
"""    
Partitioning the row of a module. 
We assume the input to be a sharded DTensor and output to be a replicated DTensor.    
"""    
    def __init__(self):        
         super().__init__(MakeInputShard, MakeOutputReplicated)


Class ColwiseParallel(ParallelStyle):    
"""   
Partitioning the column of a tensor or module. 
We assume the input to be a Replicated DTensor and output to be a sharded DTensor. 
"""    
     def __init__(self):
         super().__init__(MakeInputReplicated, MakeOutputReplicated)


class PairwiseParallel(ParallelStyle):    
"""    
We concatenate colwise and rowwise styles as a fixed pair like what Megatron-LM(https://arxiv.org/abs/1909.08053) is doing. 
We assume both input and output to a Replicated DTensor. 
We now only support Multihead Attention, MLP and transformer for this style.    
We also need to assume the input is a nn.Multihead Attention, nn.Transformer or even-number layers of nn.Linear for now.    """
    def __init__(self):
        super().__init__(MakeInputReplicated(), MakeOutputReplicated())
```

- One API for module level parallel and the user needs to specify what parallel style they want to apply to the whole module or users can specify parallel style per the module path. For PairwiseParallelStyle, we only support it for MHA, MLP and transformer models for now.

```python
def parallelize_module(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallelize_plan: Union[ParallelStyle, Dict[str, ParallelStyle]],
    tp_mesh_dim: int=0,
) -> None:
    '''
    This function converts all module parameters to distributed tensor parameters according to the `parallelize_plan` specified.
    Users can always use FSDP or DDP as a fallback if the model does not fall into the type we support here.
    Args:
        module (nn.Module): user module to be partitioned.
        parallel_plan (ParallelPlan): the parallel plan which the user wants.
        device_mesh (DeviceMesh): the device mesh to place the module.
        tp_mesh_dim (int): the dimension of TP in the device mesh.
    '''

# Code example is shown as following
import torch
import torch.distributed.tensor_parallel as tp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed import DeviceMesh

# initialize a new device mesh for TP for the given tp world size
device_mesh = DeviceMesh("cuda", torch.arange(world_size))
# colwise parallel of a Linear module
layer_one = torch.nn.Linear(8,16)
tp.parallelize_module(layer_one, tp.ColwiseParallel(), device_mesh)

# rowwise parallel of a Linear module
layer_two = torch.nn.Linear(16,8)
tp.parallelize_module(layer_two, tp.RowwiseParallel(), device_mesh)

# Megatron-LM style pairwise parallel for a transformer model
# Users do not need to specify col/row wise parallel for each module or parameter. 
transformer_model = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
pairwise_style = tp.PairwiseParallelStyle()
tp.parallelize_module(transformer_model, pairwise_style, device_mesh)

# Customized module
class DemoModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.attn = AttentionModule(...) # Defined by user.
        self.layer_norm = LayerNorm(...)
        self.mlp = CustomizedMLP(...) # Defined by user.
    def forward(self, x):
        return self.mlp(self.layer_norm(self.attn(x)))

customized_model = DemoModel(...)
tp.parallelize_module(customized_model, {“attn”: pairwise_style, “mlp”: pairwise_style}, device_mesh)
```

- For 2D parallel, the code is similar. To recap how we do 2D parallelism with FSDP. We will first parallelize modules within 8 GPUs on each host and then wrap the module with FSDP. Basically TP first shards the weight of a module and then FSDP shards the local tensor of TP-sharded weights. And another common practice of 2D parallel is to perform it on each layer of a transformer encoder or decoder rather than directly applying it to the whole model directly.

```python
# Below is another example showing 2D parallel with FSDP.
# initialize a new device mesh for 2D parallel for the given world size
device_mesh_2D = DeviceMesh("cuda", torch.arange(world_size).reshape(dp_size, tp_size))
# Pairwise parallelize a transformer model
transformer_model = torch.nn.Transformer(nhead=16, num_encoder_layers=12)
parallelize_module(transformer_model, tp_style, device_mesh_2D, tp_mesh_dim=1)
# Wrap the transformer with FSDP
dp_pg = device_mesh_2D.get_dim_groups()[0]
transformer_model = FSDP(transformer_model, pg=dp_pg)
```


### Low-level API for TP:
We also want to build some low-level APIs to provide more flexibility and usability for users as we continue to build more high-level TP features.

```python
def _parallelize_mlp(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallel_style: ParallelStyle=PairwiseParallelStyle(),
    tp_mesh_dim: int=0,
) -> None:
    '''
    This function assumes the input module is a sequence of nn.Linear and we parallelize the module based on the given parallel style.
    Args:
        module (nn.Module): user module to be partitioned.
        device_mesh (DeviceMesh): the device mesh to place the module.
        parallel_style (ParallelStyle): Parallel style with input/output preparation.
        tp_mesh_dim (int): the dimension of TP in the device mesh.
    '''


def _parallelize_multihead_attn(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallel_style: ParallelStyle=PairwiseParallelStyle(),
    tp_mesh_dim: int=0,
) -> None:
    '''
    This function assumes the input module is a class of nn.MultiheadAttention or a customized multihead attention. We will replace it with our own version of the multihead attention module.
    We directly assume the input module will be a nn.MultiheadAttention or module which has a similar structure.
    Args:
        module (nn.Module): user module to be partitioned.
        device_mesh (DeviceMesh): the device mesh to place the module.
        parallel_style (ParallelStyle): Parallel style with input/output preparation.
        tp_mesh_dim (int): the dimension of TP in the device mesh.
    '''

def _parallelize_linear(
    module: nn.Module,
    device_mesh: DeviceMesh,
    parallel_style: ParallelStyle=ColwiseParallel(),
    tp_mesh_dim: int=0,
) -> None:
    '''
    This function assumes the input module is a class of nn.Linear.
    We directly assume the input module will be a nn.Linear.
    Args:
        module (nn.Module): user module to be partitioned.
        device_mesh (DeviceMesh): the device mesh to place the module.
        parallel_style (ParallelStyle): Parallel style with input/output preparation.
        tp_mesh_dim (int): the dimension of TP in the device mesh.
    '''
