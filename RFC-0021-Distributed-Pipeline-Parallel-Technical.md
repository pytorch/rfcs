# [RFC] Distributed Pipeline Parallel Training Technical Approach

## Background - PyTorch Training Loop

PyTorch does not vend a standard training loop abstraction. As a result, the training process for a PyTorch model consists of free-form Python code. An example of a standard training loop in PyTorch might look something like this (borrowed from the PyTorch transfer learning [tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)):

```
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
```

As you can see, the code is written in a way that is very free-form and configurable. There is some similarity between training loops, such as the common appearance of various constructs:


* A DataLoader object that yields input and target data for the training process, potentially after being subject to data augmentation techniques
* An Optimizer object that manages the behavior of updating parameter values given gradients subject to a specific update policy
* A model to be run in forward propagation as well as back-propagation
* A Loss function that takes the model output, targets, and yields a differentiable expression that reduces the divergence between the two down to a scalar value to be minimized
* Calls to `backward()` for backpropagation and `optimizer.step()` to apply parameter updates after gradient computation
* A learning rate scheduler that decays the learning rate according to a policy

However, these things may not all be present during a training process and may be used in unconventional ways within the free-form Python code.

## Background - Pipeline Parallel Training

Pipeline parallel deep learning training is a technique to distribute the process of training (or inferencing) a deep learning model over a series of machines. As a technique for splitting up a large model, pipeline parallel training often requires less communication bandwidth between machines than other techniques such as tensor splitting. Pipeline parallel training can also result in high overlap between numerical computation and cross-stage communication operations. A graphical example of pipeline parallelism from [1] can be seen below:

![An example pipeline-parallel assignment](https://i.imgur.com/ODy3ws4.png)

However, model placement is not the whole story. The training process encompasses not just the forward propagation as shown in the figure, but also loss calculation, backward propagation, and parameter update. A graphical representation of this whole process from [2] can be seen in the figure below.

![Synchronous Pipeline Parallelism in Deep Learning Training](https://i.imgur.com/IHbuIm0.png)

If we refer back to the previous section, we can see that pipeline parallelism **encompasses a large part of the training loop**. A key consideration for a programming interface that enables pipeline parallelism is: how does it interact with the code within the training loop (as opposed to the code in the model?)

## Motivation - Pipeline Parallel Training with as Few Edits as Possible

We would like to deliver a pipeline parallelism solution that is as unintrusive as possible to the developer. However, given that 1) PyTorch does not vend a standard training loop abstraction and 2) pipeline parallel training encompasses a large part of the training loop, we must design a novel solution to run model training in a pipelined way.

To set a goal, we would like to run the training loop from the beginning of the document in a pipeline parallel fashion with as few code changes as possible. In this document, we explore multiple approaches for implementing Pipeline Parallel training and examine how it affects the training loop authoring process.

## Desiderata

We would like to consider the following (lifted from [[RFC] Ceci n'est pas pipeline parallelism (Pipeline Parallelism 2021Q4/2022 Plan))](https://github.com/pytorch/rfcs/blob/master/RFC-0021-Distributed-Pipeline-Parallelism.md)) when comparing alternatives (D* is an identifier for later reference, P* is a priority based on the roadmap):


* **D0 (P-1)** Cross-host pipeline parallel support (table stakes)
* **D1 (P0)** Support for passing arbitrary data types between stages (table stakes)
* **D2 (P0)** Support for pipeline parallel schedules (e.g. GPipe fill-drain, 1F1B, or interleaved 1F1B)
    * **P1** Support for arbitrary programmable schedules
* **D3 (P0)** Composability with other parallelism schemes (Tensor Parallelism, Data Parallelism) in a 3D parallelism scheme
* **D4 (P1)** Composability with other parallelism schemes in an *arbitrary scheme*
* **D5 (P1)** Off-the-shelf support for pipelining without manual conversion of a model to `nn.Sequential`
* **D6 (P2)** Support for asynchronous pipeline parallelism
    * Continuous data loading
    * Weight stashing/Weight prediction
* **D7 (P2)** Research: Fits into a unified, highly configurable programming model encompassing all parallelism schemes
* **D8 (P1)** The user can use an idiomatic Python-based training loop with no or minimal modifications from their “normal” training loop

## Approach 1: SPMD with Predicated Training Loop and Message Passing

**NOTE**: This approach is only an abstract proposal and the ideas are still in development.

Suppose I have a model consisting of 5 layers and I have 5 processors I want to run that model on. The model pseudocode might look something like:

```
def model(x):
  x = layer1(x) # Assigned to processor 1
  x = layer2(x) # Assigned to processor 2
  x = layer3(x) # Assigned to processor 3
  x = layer4(x) # Assigned to processor 4
  x = layer5(x) # Assigned to processor 5
  return x
```

One way to implement this is to convert the code into a form that is programmatically manipulable (such as a `torch.fx` Graph), partition that IR such that the code for each stage resides in a separate program, and distribute that program to each of the processors. A runtime on each of those processors would then handle loading data (from a DataLoader or from a previous stage), running the partitioned code, calculating the loss (on the last stage), running backpropagation, and applying gradient updates. This is the [approach](https://github.com/microsoft/DeepSpeed/blob/af443f63f483f6ea6769b78b4b0f2407023e9aed/deepspeed/runtime/pipe/engine.py#L46) DeepSpeed takes in pipeline parallel execution (for example, [see](https://www.deepspeed.ai/tutorials/pipeline/) how you must pass your DataLoader and use an opaque `engine.train_batch` API in DeepSpeed, relinquishing control to the opaque runtime). However, this runtime essentially replaces the PyTorch training loop, and converting your training process to this scheme may be a burden for the end user (see Approach 4 for a comprehensive treatment of the design considerations for this approach).

@zdevito proposed transparently [splitting a model into stages](https://colab.research.google.com/drive/1lGg2NqlvDwVmvBqejzni2yTmYE9rxfdr?usp=sharing) by employing a type of predicated execution[3]  For each stage, the Python code of the whole model is run, however operations outside of the ones relevant to each specific pipeline stage are “no-op”ed.

![Predicated execution of a model in pipeline parallelism](https://i.imgur.com/opv5Wic.png)

One thing not represented in this diagram is a 3rd dimension of time, i.e. there will be a “fill” stage where `forward` micro-batches fill in from left-to-right, and a drain stage where `backward` micro-batches drain out of the pipeline from right-to-left. The diagram above represents the pipeline in a “steady-state” condition.


### Expanding Predication to Include the Training Loop

The next logical step is to expand this proposal to encompass the training loop, allowing the user to continue to write their arbitrary Python training loop, but overlaying pipeline parallelism semantics on this using predicated execution.

We can pull out the parts of the canonical training loop from the beginning and investigate how they should be executed under a predicated, pipeline parallel training loop:

**Data Loader**

The data loader should only load input data on rank 0. We can view this as predicating `true` on rank 0 and false on all other ranks. We can also commingle this with `recv` for stages != 0. i.e. under the hood the dataloader object will return an input micro-batch on rank 0, but will return the intermediate value received from `rank - 1` on all ranks != 0.

**Optimizer - Zero Grad**

In synchronous PP, the optimizer should zero out gradients at the beginning of the entire mini-batch (i.e. during micro-batch 0) and *not* zero the gradients for subsequent micro-batches. Gradients for each of the micro-batches should be accumulated, but not applied, preserving the mathematical integrity of SGD optimization. The GPipe diagram from above is reproduced to demonstrate this:

![Synchronous Pipeline Parallelism in Deep Learning Training](https://i.imgur.com/IHbuIm0.png)

This scheme of zeroing grads on the first micro-batch can be trivially implemented by predicating `zero_grads` as `True` for the first micro-batch on each pipeline stage, and predicating it as `False` for each subsequent micro-batch. Applying the update with accumulated gradients is covered later.


**Forward Propagation**

Predication of forward-propagation can be done as in Zach’s [proposal](https://colab.research.google.com/drive/1lGg2NqlvDwVmvBqejzni2yTmYE9rxfdr?usp=sharing).

Note that we may extend the predicated training loop scheme to include schedules such as 1F1B or interleaved 1F1B, discussed later. 

**Loss Calculation**

The loss calculation is only valid for the last pipeline stage, i.e. `rank == world_size - 1`. We can predicate loss calculation as `False` for all ranks except the last one. There may be some complication here, as (as in the transfer learning example), there may be additional computation between the model forward pass and the loss calculation, such as a `max()` operation for extracting a single top prediction. Potentially we could extend the predicated tensor data-type from Zach’s proposal and simply use that scheme to predicate out the loss and any interstitial computation.

**Backward Propagation**

I think back-prop could happen similarly to Zach’s proposal. Similar extensions for 1F1B schedules etc apply.

**Optimizer - Step**

As described in the Zero Grad section, the optimizer step should only be applied after the backwards pass for every micro-batch has gone through the pipeline. The optimizer step call should predicate `False` for every invocation except the very last one on each stage.

**LR Scheduler Step**

This occurs outside the micro-batch region, so I think this can execute as-is


**Graphical Representation**

![Predicated execution of the whole training loop](https://i.imgur.com/9RlorjQ.png)

Note that forward and backward stages do not necessarily always run in a given rank, for example in the “fill” or "drain" phases of pipelined execution or according to specific pipeline schedules. Time should be considered as an axis into and out of the page where different states of the pipeline (fill/drain) can be represented.

### Pros and Cons of the Approach

**Pro**

* Similarly to other SPMD schemes, there is no possibility for this approach to be “front-end bound”. There is no remote coordinating process and there is no network latency delaying commands to each of the workers. The program text resides locally on each processor.
* (**D8**) It is not necessary to use specialized DistributedLoss or DistributedOptimizer structures. There are no RPCs occurring during loss calculation or optimization.
    * Note: to implement this as a clean API, maybe need to create PredicatedLoss or PredicatedOptimizer, so this may be a wash, but at least we save the RPC cost.
* (**D2**) Pipeline parallel schedules can be programmed by controlling the number of iterations each stage executes and predicating `true` the parts that should execute on that iteration. Arbitrary schedules can be programmed by the user via Python code.
* (**D3, D4, D7**) Composes with SPMD execution model; can likely readily interoperate well with SPMD Tensor Parallelism. Is potentially the basis for converging parallelism on the SPMD model (other alternative is converging parallelism on the Actor model)(needs more research - likely can be the basis for (a) paper(s))
* (**D5**) Does not require the user to manually partition their model into an `nn.Sequential`
* (**D6**) There is no concept of a synchronous “call” or “dispatch” into the training loop; this scheme can likely readily support asynchronous pipeline parallelism with continuous data loading and training.

**Con**

* Value predication imposes certain restrictions on the classes of programs that can be used in this scheme
    * For a predicated tensor without metadata, restrictions would be essentially equivalent to those in FX tracing
    * For a predicated tensor with metadata (i.e. carrying forward a MetaTensor), restrictions would be equivalent to those for MetaTensor capability.
    * Potentially, we could short-circuit evaluate Module dispatches where all Tensor inputs are predicated `false`. This would require knowing the output type of the Module, however, to return a value with the correct structure.
* Program text predication (e.g. as in CUDA) would be ideal, but I’m not aware of any global “code identifier” structure we could use here. (**TODO**: investigate Python runtime structures that might be useful here?)
* (**D8**) The practice of having the training loop tiled over multiple machines may be confusing to the user. Outside of the components covered here (e.g. DataLoader, optimizer, etc), the user may need to reason about what is and isn’t valid in their training loop under an SPMD execution model.

## Approach 2 - RPC with RemoteModule and torchgpipe-style single coordinator (@pritamdamania87 RFC)

One proposal for an API for pipeline parallelism is the `pipeline_sync` API proposed in @pritamdamania87’s [RFC](https://github.com/pytorch/pytorch/issues/44827) (Certain lines are called out with end-of-line comments containing an alphabetical identifier): 

```
# Note: This API is very similar to torchgpipe and inspired from it.
# torchgpipe API for reference: https://torchgpipe.readthedocs.io/en/stable/api.html

torch.distributed.pipeline_sync(
    pipeline: nn.Sequential,
    checkpoint: CheckpointEnum = EXCEPT_LAST, # ALWAYS, EXCEPT_LAST, NEVER 
    chunks: int = 1) -> PipelineSyncModel

Arguments:

pipeline: nn.Sequential (https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) where each nn.Module in the list is placed on the 
          appropriate device(CPU or GPU)/machine by the user. Note that 
          nn.Sequential could also consist of RemoteModule (https://github.com/pytorch/pytorch/blob/master/torch/distributed/nn/api/remote_module.py#L147) for cross host 
          pipelining.
checkpoint: Enum that determines which checkpointing mode to use.
chunks: Number of micro-batches.

Returns:
    An instance of PipelineSyncModel

Forward Method

PipelineSyncModel.forward(self, *input, **kwargs) -> RRef

Returns:
    RRef to output corresponding to the result of the minibatch. 
    Since we plan to support cross host pipelining, the RRef could be on a 
    device on a different host.
    
Example:

# This is an example of a pipeline across two machines each using one GPU.
# On worker 0
layer1 = nn.Linear(10, 5).cuda(0)
# Need to enhance RemoteModule to include device for this purposes.
layer2 = RemoteModule("worker1", device="cuda:0", nn.Linear, 5, 1) 

pipeline = nn.Sequential(layer1, layer2)
model = torch.distributed.pipeline_sync(pipeline, chunks = 4) # A

rref_params = [RRef(param) for param in layer1.parameters()]
# Need to enhance RemoteModule for `get_rref_parameters`
rref_params.append(layer2.get_rref_parameters())

dist_optim = DistributedOptimizer(optim.SGD, rref_params, lr=0.05) # B

# Helper functions
def compute_loss(output_rref, target_rref):
    return F.cross_entropy*(*output_rref.local_value()*,* target_rref.local_value()*)*

def identity_fn(inp):
    return inp

for epoch in range(epochs):
    for minibatch, target in data:
        # Use dist autograd context for distributed autograd.
        with dist_autograd.context() as context_id:
    target_rref = rpc.remote("worker1", identity_fn, target) # C
    output_rref = model(minibatch) # D
            loss_rref = rpc.remote("worker1", compute_loss, output_rref, target_rref) # E
            # Can enhance RRef to ensure this calls "dist_autograd.backward" on the last 
            # node in the pipeline.
    loss_rref.backward(context_id) # F
        dist_optim****.step() # G
```

This proposal has the training loop running on a single machine and makes copious use of the `torch.distributed.rpc` APIs available in PyTorch. We can examine different parts of the loop, highlighted with alphabetical labels:

**A - Model Pipelining**

The `pipeline_sync` function wraps the given `Sequential` of layers in a runtime that will do micro-batch splitting and execution in a pipelined fashion. I believe by the comment, the implication is that this `pipeline_sync` API functions similarly to `torchgpipe`, where a single driver schedules commands for each pipeline stage onto some “stream” abstraction, issuing subsequent micro-batches on the same stream on the forward pass and scheduling “virtual” dependencies in the autograd graph to serialize execution of gradient computation on micro-batches on each stage (see section 3.2.2 of the torchgpipe [paper](https://arxiv.org/abs/2004.09910)).

`pipeline_sync` returns an `RRef` referring to the output of the pipeline *for the whole mini-batch*.

**B - Distributed Optimizer**

The training script instantiates a [DistributedOptimizer](https://pytorch.org/docs/master/distributed.optim.html) to wrap the vanilla SGD optimizer. This DistributedOptimizer takes RRefs to the parameters distributed among the pipeline stages. During the `step()` call in stage (G), the DistributedOptimizer will make async RPC calls to all of the remotes the run an optimizer step.

**D - Model Execution**

As mentioned in (A), this is likely going to follow the `torchgpipe` logic, which issues commands for each stage according to a schedule (e.g. GPipe fill/drain schedule, see section 3.2.1 of torchgpipe [paper](https://arxiv.org/abs/2004.09910)). In this case, `forward()` calls would be issued as calls to the RemoteModule instances in sequence. Calls to `forward()` would be recorded in the distributed autograd context for later backpropagation in stage (F).

**C/E - Loss execution**

The loss calculation is something that would usually happen directly in the training loop. However, in the case of Pipeline Parallel execution, the output of forward propagation resides on the last pipeline stage and backprop should begin from that stage, moving in reverse order through the pipeline. Thus, the loss computation is formulated as an RPC onto the last stage. The training loop calls `rpc.remote`, feeding the loss calculation as target and the returned minibatch output RRef as argument (along with target values moved to the remote). This then returns an RRef referring to the loss value calculated on the remote that is backpropagated through in stage (F).

**F - Backprop**

This proposal uses the distributed autograd engine to backpropagate through the forward passes computed in pipeline parallel execution. As mentioned earlier, the per-stage execution order is likely mediated by “fork” and “join” virtual dependencies in the autograd graph, due to section 3.2.2 in the [paper](https://arxiv.org/abs/2004.09910).

**NOTE**: I don’t believe that forward and backward jobs are serialized; they may run concurrently. Is this true?

**G - Optimizer**

As mentioned in stage (B), the DistributedOptimizer will make async RPC calls to all stages to apply the selected optimizer to the parameters contained within that stage.

### Pros and Cons of the Approach

**Pro**

* (**D8**) Training loop looks pretty close to the original PyTorch code. Training loop runs on a single machine, so user does not need to reason about correctness of their training loop under SPMD, as in Approach 1.

**Con**

* (**D5**) In its current conception, requires manual splitting of the model into an `nn.Sequential` instance
* High possibility of being “front-end bound”. Every forward computation, loss calculation, autograd, and optimizer are all mediated by RPCs from a central coordinator.
    * Speed of issuing these commands on the host may be an issue. torchgpipe addresses this via scheduling jobs in execution order (section 3.2.1 of the [paper](https://arxiv.org/abs/2004.09910.pdf)). However, `torchgpipe` still finds that this is not fast enough, so uses worker threads to actually run the forward computations (likely to parallelize CPU-bound tasks such as CUDA allocator invocation). Depending on the overhead of issuing these commands over RPC, the central coordinator overhead may become an issue
    * When expanding the torchgpipe single-coordinator scheme to cross-host execution, network latency, jitter, and instability may contribute to front-end-boundedness issues
* (**D2**) In its current conception, it’s not clear to me if schedules are representable in this scheme due to reliance on distributed autograd execution. GPipe’s fill-drain schedule is implemented via careful data dependency programming in the autograd graph. It’s not clear to me if things like 1F1B, interleaved 1F1B, Varuna scheduling, or other research schedules are (easily) implementable in this scheme.
* (**D6**) This approach has a strong concept of “synchronous dispatch” into the pipeline. The single coordinator calls into the pipeline with a mini-batch, the execution is scheduled internally to that, and the pipeline returns an RRef to the result value. It’s not clear how continuous, asynchronous training would fit into this without retrofitting an event-driven handler for the training loop to feed another mini-batch in.

## Approach 3 - RPC with RemoteModule and message passing (fairscale experimental)

This describes the approach used in the fairscale experimental [distributed pipeline](https://github.com/facebookresearch/fairscale/tree/main/fairscale/experimental/nn/distributed_pipeline). A sample training loop implementation can be seen starting [here](https://github.com/wayi1/pipeline_experiments/blob/7e0fe6f884edfab026379cce1b5ae03b5c2489cd/BERT/main.py#L200). The syntax in that example is not particularly clean, but it is similar to the loop in approach (2). We can distill its essence in the following:

```
layers = nn.Sequential(...)
graph = make_graph(layers)
model = DistributedPipline(graph,chunks=chunks) # A

optimizer = DistributedOptimizer(torch.optim.SGD, model.parameter_rrefs(), lr=lr) # B

criterion = nn.CrossEntropyLoss()
class Loss(nn.Module):
    def __init__(self, criterion, ntokens):
        super().__init__()
        self.ntokens = ntokens
        self.criterion = criterion
        #self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, target):
        return self.criterion(input.view(-1, self.ntokens), target.to(input.device))

for epoch in range(epochs):
    loss_module = DistributedLoss(Loss, criterion, ntokens)
    
    for minibatch, targets in dataloader:
        with dist_autograd.context() as context_id:
            minibatch = minibatch.transpose(0, 1)
 output = model(minibatch) # C
 loss = loss_module(output, rpc.RRef(targets)).to_here() # D
 dist_autograd.backward(context_id, [loss]) # E
 optimizer.step(context_id) # F
```

This proposal has the training loop running on a single machine and makes copious use of the `torch.distributed.rpc` APIs available in PyTorch. We can examine different parts of the loop, highlighted with alphabetical labels:

**A - Model Pipelining**

As opposed to the torchgpipe-based Approach 2, this approach instantiates actors (specifically [PartitionHandler](https://github.com/facebookresearch/fairscale/blob/6f3931a4d3464056231f1bbb92a67fbd14f30d96/fairscale/experimental/nn/distributed_pipeline/partition_handler.py#L140) instances) that execute the pipeline in an event-driven manner.  PartitionHandler instances own a [DistributedPipelineRecord](https://github.com/facebookresearch/fairscale/blob/6f3931a4d3464056231f1bbb92a67fbd14f30d96/fairscale/experimental/nn/distributed_pipeline/partition_handler.py#L27) instance, which has a “feed” method to be called via RPC to add a data item for processing. 

**B - Distributed Optimizer**

DistributedOptimizer is used in the same way as Approach 2. The training script instantiates a [DistributedOptimizer](https://pytorch.org/docs/master/distributed.optim.html) to wrap the vanilla SGD optimizer. This DistributedOptimizer takes RRefs to the parameters distributed among the pipeline stages. During the `step()` call in stage (G), the DistributedOptimizer will make async RPC calls to all of the remotes the run an optimizer step.

**C - Model Execution**

`PartitionHandler` has a worker thread that runs the pipeline stage given the input data in series. Then, it forwards the result to the successor by calling its `feed()` method via RPC.

**D - Loss calculation**

Loss calculation happens similarly to in Approach 2, the single driver calls into [DistributedLoss](https://github.com/facebookresearch/fairscale/blob/6f3931a4d3464056231f1bbb92a67fbd14f30d96/fairscale/experimental/nn/distributed_pipeline/loss.py#L16), which under the hood makes an async RPC to the last pipeline stage to execute the loss calculation.

**E - Backprop**

Backpropagation through the pipeline is similarly implemented via distributed autograd, as in Approach 2. Note that the same fork/join barrier approach is used to [serialize](https://github.com/facebookresearch/fairscale/blob/6f3931a4d3464056231f1bbb92a67fbd14f30d96/fairscale/experimental/nn/distributed_pipeline/partition_handler.py#L103) execution of micro-batches on the backward pass. 

**NOTE**: I don’t believe that forward and backward jobs are serialized; they may run concurrently. Is this true?

**F - Optimizer Step**

The optimizer step uses DistributedOptimizer in the same was as Approach 2. DistributedOptimizer will make async RPC calls to all stages to apply the selected optimizer to the parameters contained within that stage.

### Pros and Cons of the Approach

**Pro**


* (**D8**) Training loop looks pretty close to the original PyTorch code. Training loop runs on a single machine, so user does not need to reason about correctness of their training loop under SPMD, as in Approach 1.
    * OTOH, some of the set-up in the [example](https://github.com/wayi1/pipeline_experiments/blob/7e0fe6f884edfab026379cce1b5ae03b5c2489cd/BERT/main.py#L200) is pretty hairy and could probably be improved
* Compared to Approach 2, much less risk of being “front-end bound”. The burden of issuing commands is distributed throughout the ranks, i.e. a rank receives micro-batches and dispatches completed micro-batches to its successor.

**Con**


* (**D5**) In its current conception, requires manual splitting of the model into an `nn.Sequential` instance
* The system may still be “front-end bound” for loss calculation, distributed autograd, and DistributedOptimizer step.
* (**D2**) In its current conception, it’s not clear to me if schedules are representable in this scheme due to reliance on distributed autograd execution. GPipe’s fill-drain schedule is implemented via careful data dependency programming in the autograd graph. It’s not clear to me if things like 1F1B, interleaved 1F1B, Varuna scheduling, or other research schedules are (easily) implementable in this scheme.
* (**D6**) This approach has a strong concept of “synchronous dispatch” into the pipeline. The single coordinator calls into the pipeline with a mini-batch, the execution is scheduled internally to that, and the pipeline returns an RRef to the result value. It’s not clear how continuous, asynchronous training would fit into this without retrofitting an event-driven handler for the training loop to feed another mini-batch in.

## Approach 4 - MPMD with a custom interpreter/instruction format and message passing (DeepSpeed)

This is the approached used in [DeepSpeed pipeline parallelism](https://www.deepspeed.ai/tutorials/pipeline/). From that tutorial, we can see that the training loop ends up looking like this:

```
class AlexNetPipe(AlexNet):
    def to_layers(self):
        layers = [
            *self.features,
            self.avgpool,
            lambda x: torch.flatten(x, 1),
            *self.classifier
        ]
        return layers

from deepspeed.pipe import PipelineModule
net = AlexNetPipe()
net = PipelineModule(layers=net.to_layers(), num_stages=2)

engine, _, _, _ = deepspeed.initialize(
    args=args,
    model=net,
    model_parameters=[p for p in net.parameters() if p.requires_grad],
    training_data=cifar_trainset())

for step in range(args.steps):
    loss = engine.train_batch()
```

The DeepSpeed `engine` here encapsulates the runtime semantics of the training loop, including data loading, pipeline parallel execution, scheduling, and optimization. The implementation of [train_batch](https://github.com/microsoft/DeepSpeed/blob/488105ebd200bbd1f6d7cbe863412e41d9ab4221/deepspeed/runtime/pipe/engine.py#L278) shows that DeepSpeed uses a type of programmable interpreter to run the instructions constituting pipeline parallel execution. The method constructs a [TrainSchedule](https://github.com/microsoft/DeepSpeed/blob/488105ebd200bbd1f6d7cbe863412e41d9ab4221/deepspeed/runtime/pipe/schedule.py#L182), which yields [commands](https://github.com/microsoft/DeepSpeed/blob/488105ebd200bbd1f6d7cbe863412e41d9ab4221/deepspeed/runtime/pipe/schedule.py#L189) for the processor to run to implement the proper sequencing of events for pipeline parallel execution. The instructions available for this interpreter are the following (with self-explanatory names):


* OptimizerStep
* ReduceGrads
* [ReduceTiedGrads](https://www.deepspeed.ai/tutorials/pipeline/#tied-layers)
* LoadMicroBatch
* ForwardPass
* BackwardPass
* SendActivation
* RecvActivation
* SendGrad
* RecvGrad


The implementations for each of these instructions can be referenced from this [lookup table](https://github.com/microsoft/DeepSpeed/blob/488105ebd200bbd1f6d7cbe863412e41d9ab4221/deepspeed/runtime/pipe/engine.py#L1307).

### Pros and Cons of the Approach

**Pro**

* (**D2**) (hypothetically) supports arbitrary schedules through the [PipeSchedule](https://github.com/microsoft/DeepSpeed/blob/488105ebd200bbd1f6d7cbe863412e41d9ab4221/deepspeed/runtime/pipe/schedule.py#L6) abstraction. However, there don’t seem to be any schedules implemented beyond the default
* (**D3, D4?**) Usable in 3d parallelism, as detailed by the [blog post](https://www.deepspeed.ai/tutorials/pipeline/).
* (**D6**) Since data is pulled from the data loader rather than being pushed by a synchronous call in the training loop, this approach could *hypothetically* support async PP. 
* (**D7**) The approach seems to account for many different types of parallelism.

**Con**

* (**D1**) Does not support passing arbitrary data between stages, only supports Tensor and tuple of Tensor (because of `nn.Sequential` front-end)
* (**D5**) Only supports models fit into an `nn.Sequential`
* (**D8**) This approach takes control away from the user. The training loop is now implemented by the DeepSpeed engine abstraction, rather than being free-form Python code.

## Approach 5: RPC with remote modules and generalized Module-server architecture (SageMaker)

The [SageMaker model parallelism](https://arxiv.org/abs/2111.05972) design uses a single Python-native training loop with a “module-server” architecture. The system divides the model based on the Module hierarchy and assigns each module onto a specific pipeline parallel rank (PP_RANK). During execution, when there is a dispatch to a `Module` that resides on another PP_RANK, a remote request-response RPC is made to run the appropriate forward/backward pass for the Module on the remote PP_RANK.

[Image: req_resp.png]

PP_RANK 0 drives the process by scheduling instances of the training loop function (a UDF annotated by `@smp.step`): two for each micro-batch (one for forward, one for backward). PP_RANK 0 can implement different “schedules” by dispatching these `(micro-batch, phase)` tuples in a given order. The orders that they present are:


* Simple pipeline (aka GPipe fill-drain). This is implemented by having PP_RANK 0 dispatch the `phase=forward` tuples for each micro-batch in sequence. Then, dispatching the `phase=backward` tuples for each micro-batch in sequence.
* “interleaved” pipeline (**NB**: this is not the same as the *interleaved 1F1B* from Narayanan, 2021). PP_RANK 0 will schedule `phase=forward` jobs and opportunistically schedule `phase=backward` jobs *as soon as the forward pass for that micro-batch is done*.

![Module-server request-response execution in SageMaker pipeline parallelism](https://i.imgur.com/y9MZJ3b.png)

**NOTE**: The schedules here do not necessarily run stage in a given order on each stage. Network latency and other affects may change the order of when micro-batches are executed.

### Pros and Cons of the Approach

**Pro**

* (**D1**) I *believe* passing arbitrary types works, assuming their P2P communication backend supports it. There is no fundamental limitation precluding this from working with this design.
* (**D2**) (split pro/con) Support for *some* kind of schedules, but not strictly implemented as those described in the literature
* (**D3/D4**) Composes with other parallelism schemes
* (**D5**) Does not require model to be an `nn.Sequential`
* (**D8**) User’s original training loop is preserved with only slight modifications (`@smp.step` annotation and other things)

**Con**

* (**D2**) (split pro/con) Support for *some* kind of schedules, but not strictly implemented as those described in the literature
* (**D6**) (not sure about this one) Seems to still rely on synchronous training loop. May need modifications to support async (but not sure if there are any fundamental limitations?)

## Approach 6: SPMD with Program capture/JIT compilation and message passing (OneFlow)

[OneFlow](https://arxiv.org/abs/2110.15032) is a deep learning framework that purports to redesign the deep learning programming model to better support distributed computation. The framework uses a “consistent view” abstraction to represent distributed memory. The framework sports a compiler that can run global optimization of device placement. It also uses a unified actor model abstraction for its distributed runtime.

An example of using OneFlow for pipeline parallelism can be seen in this [tutorial](https://docs.oneflow.org/en/master/parallelism/06_pipeline.html). The example instantiates a sequential two-stage model architecture and uses the `to_consistent` API to move the submodules to the appropriate CUDA devices. It also sets `config.stage_id` on each of the submodules to give a monotonically increasing stage number. Finally, on the `nn.Graph` it uses `config.set_gradient_accumulation_steps` to delay optimization for 2 micro-batches, and calls `add_optimizer` to add the optimizer class.

`oneflow.distributed.launch` will launch the processes for each “actor”. Eager mode and graph mode reportedly go down the same (or similar) code path for distributed graph processing, with eager mode using a LazyTensor-like model to capture the program. The compiler knows which rank the compilation is for and will emit code specifically for that rank. Micro-batch splitting and `1f1b` seem to be hard-coded, if they are implemented.

### Pros and Cons of the Approach

**Pro**

* (split pro/con) (**D2**) Not clear if `1f1b` or other schedules are implemented
* (**D3/D4**) Composable with other parallelism schemes via their “Consistent View” definition
* (**D5**) `nn.Sequential` not needed, but potentially an `nn.Graph` instance may be needed in some cases
* (**D6**) Async is probably supportable but not clear. From their presentation, the actor/register model with backpressure can implement on-demand data loading, but I’m not 100% sure what that API looks like
* (**D7**) Unified programming model that already exists

**Con**

* (split pro/con) (**D2**) Not clear if `1f1b` or other schedules are implemented/implementable?
* (**D8**) Not clear what the training loop abstraction looks like. The optimizer is installed via an `nn.Graph` API. Loss calculation is created in the `nn.Graph.build()` method.

## Final Analysis

### General Design Axes

Looking through the various approaches above, we can pull out some general design axes we should consider:


* (**DA1**) Single-coordinator vs. distributed coordination
    * Sub-decisions: model execution, autograd, optimizers
* (**DA2**) Python training loop vs. custom encapsulated training loop (e.g. as in DeepSpeed)
* (**DA3**) `nn.Sequential` vs. more free-form partitioning
* (**DA4**) Local vs. distributed loss
* (**DA5**) Distributed autograd framework vs. manual autograd
    * As it relates to pipeline schedules
* (**DA6**) Local vs. distributed optimizer
* (**DA7**) Synchronous pipeline dispatch vs. asynchronous
* (**DA8**) Predication scheme (Approach 1 only)
* (**DA9**) Instruction format (Approach 4 only)


We can start analyzing the approaches by these design axes

* Approach 1: SPMD with Predicated Training Loop and message passing
* Approach 2: RPC with RemoteModule and torchgpipe-style single coordinator (@pritamdamania87 RFC)
    * Single-coordinator
        * CUDA command buffer analogy
    * Continuation passing
* Approach 3: MPMD with RemoteModule and message passing (fairscale experimental)
* Approach 4: MPMD with a custom interpreter/instruction format and message passing (DeepSpeed)
* Approach 5: RPC with remote modules and generalized Module-server architecture (SageMaker)
* Approach 6: SPMD with Program capture/JIT compilation and message passing (OneFlow)

|	|DA1	|DA2	|DA3	|DA4	|DA5	|DA6	|DA7	|DA8	|DA9	|Notes	|
|---	|---	|---	|---	|---	|---	|---	|---	|---	|---	|---	|
|Approach 1	|multi	|py	|FF	|local	|manual	|local	|async	|?	|X	|	|
|Approach 2	|single	|py	|seq	|dist	|dist	|dist	|sync	|X	|X	|	|
|Approach 3	|single	|py	|seq	|dist	|dist	|dist	|sync	|X	|X	|	|
|Approach 4	|multi	|interp	|seq	|local?	|manual	|local	|async*	|X	|?	|	|
|Approach 5	|single*	|py	|FF	|local?	|manual	|local?	|sync?	|X	|X	|Schedules?	|
|Approach 6	|multi	|interp	|FF	|local?	|manual? (graph?)	|local?	|async?	|X	|X	|	|

### Decision - Approach 3 with Modifications

After deliberation, we want to build the API with the least complexity, at least initially. We will modify/build the API in FairScale experimental with a few modifications:


* (**DA5**) Rather than using torchgpipe-style virtual dependencies in the distributed autograd graph, we want each stage to manually handle running `forward` and `backward` stages (the latter by explicitly calling `torch.autograd.backward()`). This will give easier and finer-grained control over the execution schedule of the pipeline
* (**DA9**) We want to abstract the runtime for each actor (similar to DeepSpeed) so that the actor can run forward/backward phases in a prescribed order. This will allow us to program schedules like `1f1b` or `interleaved 1f1b` . Further, we can define an instruction format similar to DeepSpeed to make these schedules arbitrarily programmable by the end user.
* (**DA4**) In the current implementation, the loss is implemented via DistributedLoss, which is issued in the training loop over the whole mini-batch. This will not be compatible with arbitrary pipeline schedules, which need to compute the loss and launch backward micro-batches asynchronously. So, the loss will need to be implemented as a callback that the pipeline can schedule on its own, rather than something called in the training loop


Approach 3 with modifications then looks like:


|	|DA1	|DA2	|DA3	|DA4	|DA5	|DA6	|DA7	|DA8	|DA9	|Notes	|
|---	|---	|---	|---	|---	|---	|---	|---	|---	|---	|---	|
|Approach 3 with Modifications	|single	|py	|seq	|local	|manual	|dist	|sync	|X	|?	|	|


**Future Extensibility**

This approach leaves some design improvements on the table. In particular, we will have to keep an eye on these extensibility points:


* (**DA3**) Expanding this API to work on programs that are not already `nn.Sequential`. We can explore using a predicated scheme such as in Zach’s [notebook](https://colab.research.google.com/drive/1lGg2NqlvDwVmvBqejzni2yTmYE9rxfdr?usp=sharing) or a `torch.fx`-based scheme as in Wanchao’s proposal. Further, we could create a "V2" API down the road using technology from Approaches 4-6 or if new research finds a better programming model.
* (**DA4**) In the current implementation, optimization is implemented via a `DistributedOptimizer` step that is called explicitly from the training loop. This is likely okay for synchronous PP. We should keep an eye on (a) the overhead from having a single coordinator issue RPCs to do the optimization step and (b) how this design might change with asynchronous pipeline parallelism, considering the pipeline itself will need to trigger all these events (async PP will likely be a different API, but it would be nice if it didn’t need to be)

## References


1. https://arxiv.org/abs/1806.03377
2. https://arxiv.org/abs/1811.06965
3. https://en.wikipedia.org/wiki/Predication_(computer_architecture)
