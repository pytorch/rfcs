# [RFC] Model Partitioning in Pipeline Parallelism

Credit for writing this RFC goes to @WanchaoL.

## Background

We introduced Pipeline Parallelism API in PyTorch 1.8. The current Pipeline parallelism provides an intuitive API to use. But currently we are passing the obligation of model partitioning to the user, where user need to explicitly convert different parts to different devices before passing to `sync.Pipe`. This might be OK for simple models, but as models scale in complexity, it becomes very hard to estimate and partition the model properly by hand. The user might need to do a lot of experiments before actually running the model in Pipeline Parallelism for efficiency purposes. Therefore, it’s crucial that we provide a way to automatically partition the model so that we could reduce the burden for users, this might also give some insights when we do other model parallelism work. This doc did some explorations on the existing works from different libraries, and proposing a way for PyTorch’s pipeline parallelism to do automatic partitioning.


## Relevant works

How is the industry dealing with model partitioning for pipeline parallelism so far? There’re plenty exploration works going on from different pipeline parallelism implementations, and some of them produces pretty nice performance improvements. 

***FairScale/torchgpipe:***
FairScale and torchgpipe use similar/same approaches when partitioning the model, their partition APIs:

*  `Pipe(..., balance: List[int],...)` use `balance` to partition the model during construction
* two utils: `balance_by_time` and `balance_by_sizes` to generate the `balance` list automatically based on execution time or memory sizes (parameter + optim states)

Partition Algorithm: use “Block Partition of Sequences” algorithm to find a balanced partition base on the costs. 
Note they only allow to partition on the top-level modules of the Sequential list based on the costs (memory/time)

***SageMaker***
Major logic is under `model_partition.py` and the basic ideas:

* `ModuleParitioner` to drive the model partition progress base on trace results 
* `ModulePartitioner.partition` create a tree of `ModuleNodes`, populate the costs (execution time + memory cost) and run the partition algorithm, return a `Dict[Module, device_id]`

Partition Algorithm: SageMaker provided an automated model parallelism by their own proposed model partition algorithm based on BFS + DP-based method for device allocations, plus reallocation using d'Hondt method

***DeepSpeed:***
Provide several mechanisms for partition the model across GPUs with `partition_method`:

* `partition_method="parameters"` (default): balances the number of trainable parameters on each pipeline stage.
* `partition_method="type:[regex]"`: balances layers whose class names match [regex].
* `partition_method="uniform"`: balances the number of layers per stage.

Partition Algorithm: For partition_method = “parameters”, DeepSpeed use the mechanism of counting the layer’s parameters size, and do a binary search (find the smallest weight of the heaviest partition) to find a balanced partition.

***PipeDream***
More on asynchronous training pipeline planning, like parameter server approaches, the algorithm itself also uses Dynamic Programming, but the states is highly correlated with the communication and synchronization costs, which is not a good choice for us (unless we consider async training in the future). 

***DAPPLE***

DAPPLE combines DDP with pipeline parallelism to form a bigger search space and use a more efficient schedule if possible. Their partition approach is a DP-based algorithm, it first tries to find the “pivotal” stage, then optimize the overall latency,  the “overall latency” optimization here tries to reduce the bubbles in the pipeline as small as possible.



## Proposing: Automated Model Partitioning in PyTorch Pipeline Parallelism

Given the existing work and their limitations, we introduce a set of APIs that’s flexible enough for future improvements and intuitive to use. We expose a single API `create_balanced_partition`, to take the model and do the partition under the hood.

```
def create_balanced_partition(model: nn.Module,
                              devices: List[RemoteDevice],           
                              *sample_input) -> List[RemoteModule]:
 #device = torch.device("cuda")):
 # if model itself if larger than a single cuda device memory, 
 # should we allow the user to profile the model on cpu?
 # Decision: probably no, as the characteristics are different

 # Step 1
 # fx profiler to collect statistics based on a single run of
 # the sample_input
 interp = ProfilingInterpreter(model)
 interp.run(sample_input)
 
 # Step2
 # Partition algorithm to calculate the least 
 # cost partition assignment
 num_partitions = len(devices)
 partitions = _partition_by_cost(interp, num_partitions)

 # Step3
 # Use the results from Step 2 to assign each part of submodule to the device
 # returned a fx-based model with partition results applied (assigned to devices already)
 return partitioned_model
```


Note that this will be the only API we expose to the user, it returns a partitioned model which already been transferred to the corresponding devices. This allow us to iterate on the underlying implementation and experiment more efficient partition algorithms in the future.

## FX compatibility

Pipeline parallelism auto partition capability needs more advanced knowledge in order to accurately divide a model into a balanced partition set.  In order to achieve auto partitioning with the most balanced approach, we need to get the model execution graph and try to split the model base on the graph. Using torch.fx can give us a helpful graph representation in order for us to do more advanced partition with extensive analysis with tracing and profiling. 

The model that passed in should be fx compatible in order to generate the partitions with the most accurate estimation. What models does not have fx compatibility currently:

1. models that contain input dependent control flow (no way to fix as far as I know
2. models that have tensor constructors (this could potentially be fixed)
3. models that contains builtin-op with non tensor inputs https://github.com/pytorch/pytorch/issues/53937 (this could potentially be fixed)

For 2 and 3, I think it could be fixed, for 1, a fundamental limitation is there. 
Should we make this assumption? 

We should try to symbolically trace the model first, and if that fails with tracing exceptions (i.e. detecting data dependent control flows, we could detect that during tracing), we should fall back to a legacy partition algorithm with python available only (i.e. without extensive graph analysis, we can simply try to balance the model with the top-level submodules like nn.Sequentials)


@jamesr66a: We could also try doing unintrusive tracing, similar to SageMaker or the upcoming define-by-run quantization API. `torch.fx` symbolic tracing supposes you want to extract a freestanding representation of the program, so is rather strict in the error conditions for which it fails. OTOH, we can do a "best effort" program recording using `__torch_function__`/`__torch_dispatch__` and record information about the structure of the program, but not necessarily require that it fully represent the whole program.

## Profiling using torch.fx

We will use torch.fx to do a profiling run to collect the statistics (i.e. execution_time, parameter_size, execution_order, etc.). The profiler still needs user to pass in a sample input, and we do a full run on the model base on this sample input. We can adjust the statistics to collect base on the partition algorithm we choose (i.e. module-level statistics or op-level statistics)

```
from torch.fx import Interpreter

class ProfilingInterpreter(Interpreter):

    def __init__(self, mod : torch.nn.Module):
        gm = torch.fx.symbolic_trace(mod)
        super().__init__(gm)

        # We are going to store away three things here:
        #
        # 1. execution time of each module/node
        # 2. parameter_sizes of each module/node
        # 3. activation_sizes of each module
        self.execution_times_sec : Dict[str, float] = []
        self.parameter_sizes : Dict[str, float] = {}
        self.activation_sizes: Dict[str, float] = {}

    def run_node(self, n : torch.fx.Node) -> Any:
        # Record the time we started running the op
        t_start = time.time()
        # Run the op
        return_val = super().run_node(n)
        # Record the time we finished running the op
        t_end = time.time()
        self.execution_times_sec[module_qual_name] = t_end - t_start

        # also update the parameter_size and activation_size for
        # module nodes 

        return return_val
```





## Partition Algorithm

There’re several partition algorithms as we mentioned in the relevant works section, each of them have its own pros/cons. For example, torchgpipe/fairscale only allows partitioning the model on the top level, SageMaker only partition on the module level instead of operation level. So there might be some unbalanced cases for those approaches.  DAPPLE explores a bigger search space by controlling the stages/schedules, but the algorithm complexity is too high. 

Our approach here starts with an approach similar to fairscale, which only do top-level partition, but it could be improved further by exploring more partition approaches, i.e. with operation level partitioning by torch.fx, we could partition the model in a more even manner, which further improves the efficiency. Since we only expose a simple API and return the partitioned model with stages assigned to the devices, the internal implementation can be improved as we explore more ideas. The plan for partition algorithm exploration:

* Use fairscale/gpipe's [Block Partition of Sequences](https://arxiv.org/pdf/1308.2452.pdf) as our first step
* See if we can apply [Fiduccia-Mattheyses Heuristic](https://en.wikipedia.org/wiki/Fiduccia%E2%80%93Mattheyses_algorithm) to partition the graph and how the performance compare with the default one
* See if we can do operation-level tracing, and partition the model into several balanced `torch.fx.GraphModule` instead of using the original module architecture

### Approach that enables more possibility of balanced partitioning

* fx trace the graph, collect the statistics of each node
* partition the graph using a bottom up approach: starting from treating each node as a partition, and merge them if it lowers the cost, use BFS to scan the merge order
* The result will be an `fx.GraphModule` for each partition, the original module architecture is not preserved
    * qual names of param might change

Partition:
Dict[fx.graphmodule, remotedevice]

in order to apply the algorithm, we need to assign each top-level submodule with a proper cost, we calculate the cost based on memory and execution costs.


### Cost calculation

The cost of a submodule is defined as the following:

```
cost(m) = time_weight * execution_time(m) 
            + memory_weight * memory_cost(m)

memory_cost(m) = parameter cost * optimizer_multiplier 
                    + activation_cost
```

where `memory_weight + time_weight = 1`. How to weigh between time and memory cost? Undecided, a heuristic number `time_weight=0.6`

* @mrshenli: regarding balancing time between memory, I kind of feel we should prioritize time over memory. Because people usually use Pipeline parallel to accelerate training, and the training speed the ultimate goal. Balanced execution time has a directly impact on total pipeline makespan (suppose one phase is slower than others, then all other devices will need to wait for that phase). If the above assumption is correct, it looks like we should first try to balance time, and only try to balance memory when a shard of the optimally time-balanced model cannot fit in some devices. 

Time > memory (memory constraint only, need to count in some buffer)

How do we decide the `optimizer_multiplier` since different optimizers have different states maintained?

We can use the param_scale mechanism which specify the scale for each parameters like in fairscale or torchgpipe. One disadvantage is that this `param_scale` is required from the user as a parameter, and user need to know the scale base on our documentation? Still seems a bad UX experience, but if we provide a default one, it’s likely not accurate.


```
    =========  =============  =========================================
    Optimizer  `param_scale`  Internal State
    =========  =============  =========================================
    SGD        2--3           (momentum_buffer)
    Adam       4--5           exp_avg, exp_avg_sq, (max_exp_avg_sq)
    Adadelta   4              square_avg, acc_delta
    Adagrad    3              sum
    RMSprop    3--5           square_avg, (momentum_buffer), (grad_avg)
    =========  =============  =========================================
```


Since we assigned the cost to each top-level module, we can do the partition, basic partition function will be like the below:

```
def _partition_by_cost(interp: torch.fx.Interpreter,
                       num_partitions):
    submodule_costs = interp.get_costs() 
    # Dict[module_qual_name, cost], execution order
    
    # use Block Partition of Sequences to minimize the variance
    # of the submodules' cost list 
    partitions = 
        block_partition.solve(submodule_costs, num_partitions)
    return partitions
```


Q: How do we collect the communication costs?

* activation_size and gradient size, could potentially contribute to part of the communication cost
* What about the device info? how do we know each pairs are NVLink or PCIe?



### Further exploration of partition algorithm

Since we hide the underlying partition algorithm, we can do further partition algorithm explorations, some potential explorations we can do:

* exploration like in DAPPLE when using non-simple schedule (i.e. we reduce the bubble by do the backward as early as possible). 
* Using torch.fx, we can trace the module architecture in module level, and possibly split the entire traced graph into several submodules (this might not fully resemble the existing module architecture). 
* DP-based algorithm like SageMaker, but could be on operation-level with more granularity 



## Potential issues

**What if a model with a big memory requirement that couldn’t fit into a single GPU?**
A: we construct the model on CPU and do fx tracing on CPU, after the partition we move it to the corresponding device

**What if a “module” (submodule) with a big memory requirement that couldn’t fit into a single GPU?**
Can we use ShardedTensor to shard the module parameters? or we could do operation level tracing, partition this module into 2 submodules, then assign to different devices. 

**What if constructing the model on CPU itself is hard?** 
User created the model first on meta device, we use the model that’s not materializing to do symbolic tracing, but we couldn’t do profiling since it’s not materialized yet, after we do symbolic tracing, we should use a simple partition algorithm (i.e. only count the param sizes) to do the partition, then materialize afterwards.



## Reference


1. SageMaker https://arxiv.org/abs/2111.05972
2. Deepspeed https://www.deepspeed.ai/tutorials/pipeline/#load-balancing-pipeline-modules
3. Torchgpipe [https://github.com/kakaobrain/torchgpipe](https://github.com/kakaobrain/torchgpipe/tree/master/torchgpipe)
4. Fairscale https://github.com/facebookresearch/fairscale
5. FX_IR partitioner for accelerators 
https://fb.workplace.com/notes/wang-xu/fx-ir-partitioner/165617378604863
