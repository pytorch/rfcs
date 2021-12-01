# [RFC] Ceci n'est pas pipeline parallelism (Pipeline Parallelism 2021Q4/2022 Plan)

This is an RFC for the strategic plan for further developing pipeline parallelism in PyTorch. **We invite our users and partners to comment on this plan and the corresponding technical plan** to help us develop the best APIs for PP in PyTorch.

Goal: *Provide a flexible, composable, and reconfigurable interface for pipeline parallelism in PyTorch that allows scaling a wide variety of PyTorch models on a wide variety of hardware configurations*.

## Motivation

* Pipeline parallelism (PP) is used as a lower-communication-volume technique for model parallelism. It is especially applicable when data must be transmitted across comparatively slower interconnects.
* Several research-oriented frameworks exist that implement PP (e.g. fairscale, megatron, deepspeed), but we would like to provide a production-quality implementation and support contract for PP.
* The existing PP implementation in PyTorch (`torch.distributed.pipeline.sync`) only supports intra-host pipeline parallelism across GPUs and does not support techniques like 1F1B scheduling. We can deliver inter-host pipelining and other features.
* `nn.Sequential` requirement creates a huge barrier to users who have models that don't lend easily to be converted to `nn.Sequential`. In particular with models that have dynamic control flow for some large segments (e.g. conditional encoder).
* Ultimately, we want to use this body of work as a driving force for research in delivering both performance AND usability of parallelism paradigms. We invite developers and researchers to participate in the design and development of this project.

## Stage 1: Requirements Gathering (2021Q4)

We have spent a good amount of time this calendar quarter researching the user requirements and systems research directions of pipeline parallelism and will continue to do so going forward. **We invite additional comments to fill in details we have not captured here**, if any.

### Prior Work

The research literature[1-12] has a rich body of work. This includes:

* Synchronous vs. Asynchronous pipeline parallelism.
    * Synchronous pipeline parallelism where a mini-batch is split into micro batches and the pipeline is filled and drained, blocking until the mini-batch is completed. This is the typical use case we are designing for
    * Asynchronous pipeline parallelism that keeps the pipeline continually occupied. Various clever techniques such as weight stashing and weight prediction have been proposed to address the consistency issues from the "locking" nature of SGD in these cases. These techniques may introduce additional design concerns in a pipeline parallelism API.
* Pipeline scheduling, where the execution order of `forward` or `backward` micro-batches follows a specified policy. The infrastructure for implementing these schedules can be an important consideration for the design on a PP API.
    * Fill-drain schedule, where all forward micro-batches are run to completion before all backward micro-batches are run and parameter updates are applied.
    * 1F1B schedule, where `backward` micro-batches are triggered by the last pipeline stage and stages ensure that they alternate between running `forward` and `backward` micro-batches at steady-state. This helps to reduce the amount of state stored on each pipeline stage.
    * More, including interleaved 1F1B and new research schedules.

### Key Stakeholders

This section is meant to capture key users/researchers who would benefit from such a pipeline parallelism API. **We invite additional comments to fill in users/researchers who would benefit from this API and would like to see their requirements satisfied**.

#### P0: HF Transformers

HF transformers [wants to](https://github.com/huggingface/transformers/issues/13690) incorporate 3d parallelism including Pipeline Parallelism, however the [current PyTorch implementation](https://github.com/pytorch/pytorch/blob/9f4e004abd8c5d11fc23f4ab705328cb9b4050bb/torch/distributed/pipeline/sync/pipe.py#L220) has limitations that we should address (Px is a priority, with lower x being higher priority. We assigned these priorities based on a) user need and b) implementation time/complexity, but we can adjust them based on user feedback):

* Frontend limitations:
    * **P0**: Cannot pass arbitrary data types between pipeline stages
    * **P0**: Unclear composability in 3d parallelism scheme (data, pipeline, model parallel)
    * **P1**: User needs to rewrite their model as an `nn.Sequential` instance
* Backend Limitations:
    * **P(-1)**: No cross-host support for PT pipeline parallelism API
    * **P0**: No support off-the-shelf schedules (1F1B or interleaving)
    * **P1**: No support arbitrary programmable schedules
* Non-requirements:
    * Composability with ZeRO-2/3 is not required. Theoretically possible, but reportedly will not give any perf gain.
* Success Criteria:
    * **to be determined**: Feedback on this would be appreciated

### Prior Implementations and Proposed Approach

An analysis of prior implementations and a proposed technical approach for pipeline parallelism can be seen in [[RFC] Distributed Pipeline Parallel Training Technical Approach](https://github.com/pytorch/rfcs/blob/master/RFC-0021-Distributed-Pipeline-Parallel-Technical.md). In this document, we further split execution into stages and correlate those to the PyTorch external release schedule.

## Stage 2: Ship prototype synchronous multi-node pipeline parallelism (torchgpipe-style) (1.11 Prototype Release)

### P(-1): Implement cross-host support for pipeline parallelism

Existing approaches that support this (in no particular order):

* Fairscale [experimental distributed pipeline parallelism](https://github.com/facebookresearch/fairscale/tree/main/fairscale/experimental/nn/distributed_pipeline)
* Sagemaker [model parallelism](https://arxiv.org/abs/2111.05972)
* [DeepSpeed pipeline parallelism](https://www.deepspeed.ai/tutorials/pipeline/)
* [OneFlow](https://github.com/Oneflow-Inc/oneflow)

Proposed approach short-list: (all approaches can be seen in [[RFC] Distributed Pipeline Parallel Training Technical Approach](https://github.com/pytorch/rfcs/blob/master/RFC-0021-Distributed-Pipeline-Parallel-Technical.md)

1. Selected approach: "Approach 3 with Modifications"
    * Inherit RemoteModule + torchpipe-based implementation from fairscale [experimental distributed pipeline parallelism](https://github.com/facebookresearch/fairscale/tree/main/fairscale/experimental/nn/distributed_pipeline).
    * Switch autograd off of distributed autograd and onto manual handling of autograd in the pipeline, to facilitate implementing schedules (e.g. 1F1B)
    * Abstract the runtime for each RemoteModule to allow for programming in execution schedules
    * Switch from using DistributedLoss to having a loss callback to facilitate the last pipeline stage calling the loss locally rather than relying on the training loop to calculate the loss via RPC and call distributed autograd. This will be necessary with arbitrary schedules.

### P0: Implement support for passing arbitrary data-types between pipeline stages

Existing approaches that support this (in no particular order):

* Some amount of [support](https://github.com/pytorch/pytorch/issues/53952) in existing PT implementation

Proposed approach short-list:

1. Hopefully should just work out of the box with the RPC API, but need to keep it in mind.

### P0: 1.11 Prototype Release and out-of-tree demo on HF Transformers

* Release API as prototype in the 1.11 release to facilitate gathering feedback
* Validation: Out-of-tree demo on HF transformers repo - hack it together to get it to work and pull out work items to improve the API to remove places where code edits are needed
* 1.11 Release Dates
    * Feature submission: 11/30 EOD
    * Branch cut 1/31/2022



### **P0**: Support off-the-shelf schedules (1F1B or interleaving)

Existing approaches that support this (in no particular order):

* Megatron hardcoded schedules: [1f1b](https://github.com/NVIDIA/Megatron-LM/blob/5ac5571ba0265af4c491ee0af1508ca7589450c6/megatron/schedules.py#L517), [interleaved](https://github.com/NVIDIA/Megatron-LM/blob/5ac5571ba0265af4c491ee0af1508ca7589450c6/megatron/schedules.py#L187)

Proposed approach short-list:

1. "Approach 3 with Modifications"
    * Once manual autograd is handled, and we abstract the workers, we can implement 1F1B or interleaved 1F1B using that infrastructure.

### P0: Composability of PP with TP and DP (3d Parallelism)

Existing approaches that support this (in no particular order):

* `torch.distributed` APIs via module wrapper composition
* [DeepSpeed](https://www.deepspeed.ai/tutorials/pipeline/)

Proposed approach short-list:

1. DistributedDataParallel wraps PipelineParallel API which operates on upcoming ShardedTensors
2. Unified programming model (Stage 5)

## Stage 3: Figure out how to reconcile local pipeline vs. distributed pipeline (2022H1)

The existing approaches live in different corners of a 2-dimensional space with axes on **single-driver vs. actors** and **local vs. distributed**.


|	|single-driver	|actors	|
|---	|---	|---	|
|local	|torchgpipe/fairscale Pipe/distributed.sync.Pipe	|	|
|distributed	|	|Fairscale distributed_pipeline, DeepSpeed, Megatron-LM	|

### Design Speculation

We can interpolate the missing spaces:

* **single-driver, distributed**: “macro SIMD” style distributed execution. I believe this is actually what was envisioned in @pritamdamania87’s [RFC](https://github.com/pytorch/pytorch/issues/44827) with the `torch.distributed.pipeline_sync` API. The current `distributed.sync.Pipe` API is a fork of the `torchgpipe` implementation (transitively forked in `fairscale`), which is hard-coded for single-node execution issuing commands via CUDA streams (or a fake CPU stream stand-in they implemented)
* **actors, local**: We can take the event-driven approach taken in fairscale’s `distributed_pipeline` and extend that to having worker processes/threads that both a) feed a corresponding CUDA device and b) feed data through to the successor in the pipeline. This is sort-of already done by the `torchgpipe` lineage of implementations which use [worker threads](https://github.com/pytorch/pytorch/blob/master/torch/distributed/pipeline/sync/worker.py) that run the actual forward computation but still have a central coordinating thread issuing each of those workers commands nonetheless. Potentially if done in a multi-process setting, this could lead to higher performance (need to measure).

I believe the way to go in the future may be to consolidate on actors for both local and distributed. This may represent lower complexity than the torchgpipe-style execution (at least when I think about it) and can avoid issues with a single driver process being a bottleneck (as evidenced by the fact that `torchgpipe` already uses threads for speed).


## Stage 4: Generalize pipeline parallelism interface to allow for more coverage of different techniques in the literature (e.g. async, scheduling, auto-partitioning, composition with tensor parallelism) (2022, OSS releases 1.11-1.15)

### P1: Pipeline parallelism without `nn.Sequential` rewrite

Existing approaches/proposals that support this (in no particular order):

* Sagemaker [model parallelism](https://drive.google.com/file/d/1N2eo5Yr_QOw0EtKv-MYBDWKvyRYxKv2o/view)
* @zdevito's [sequential-free splitting approach](https://colab.research.google.com/drive/1lGg2NqlvDwVmvBqejzni2yTmYE9rxfdr?usp=sharing)
* [OneFlow](https://github.com/Oneflow-Inc/oneflow)
* [[RFC] Model Partitioning in Pipeline Parallelism](https://github.com/pytorch/rfcs/blob/master/RFC-0022-Model-Partitioning-in-Pipeline-Parallelism.md)

Proposed approach short-list:

1. [[RFC] Model Partitioning in Pipeline Parallelism](https://github.com/pytorch/rfcs/blob/master/RFC-0022-Model-Partitioning-in-Pipeline-Parallelism.md)
2. @zdevito's [sequential-free splitting approach](https://colab.research.google.com/drive/1lGg2NqlvDwVmvBqejzni2yTmYE9rxfdr?usp=sharing)
3. Construct a pipeline parallelism API that uses a different approach, such as the one used in SageMaker model parallelism. This introduces trade-offs elsewhere, such as in support for schedules/the requirement for an optimization pass to be applied to implement "true" pipeline parallelism.

These approaches can be composed on top of an existing API that takes an `nn.Sequential`. We may consider in the future to develop a "v2" API that is centered more natively around non-`nn.Sequential` models using technologies from Sagemaker, OneFlow, or other research developments.

### P1: Support arbitrary programmable schedules (e.g. fill-drain, 1F1B, interleaved 1F1B) 

Existing approaches that support this (in no particular order):

* DeepSpeed [PipeSchedule](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/schedule.py) is an instruction format that allows customizing the order in which forward/backward jobs on different stages should be executed.

Proposed approach short-list:

1. Programmable instruction stream + interpreter à la [PipeSchedule](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/pipe/schedule.py). This should be enabled by the infrastructural work done in Stage 2.

### P2: Asynchronous Pipeline Parallelism - Mechanics of Asynchronous Training Loop

**Call for Stakeholders**: Do you have a project that would benefit from Asynchronous Pipeline Parallelism in PyTorch? Please comment on the RFC and we can incorporate your requirements.

* async training is like a self-perpetuating engine v.s. a synchronous procedure call as is typical in Python. How do we bridge these two? What would the Pythonic experience for async look like?

### P2: Asynchronous Pipeline Parallelism - Weight stashing

**Call for Stakeholders**: Do you have a project that would benefit from Asynchronous Pipeline Parallelism in PyTorch? Please comment on the RFC and we can incorporate your requirements.

* [Parametrization](https://pytorch.org/tutorials/intermediate/parametrizations.html) as an approach?

### P2: Asynchronous Pipeline Parallelism - Double-Buffered Weight Stashing

**Call for Stakeholders**: Do you have a project that would benefit from Asynchronous Pipeline Parallelism in PyTorch? Please comment on the RFC and we can incorporate your requirements.

* [Parametrization](https://pytorch.org/tutorials/intermediate/parametrizations.html) as an approach?

### P2: Asynchronous Pipeline Parallelism - Weight Prediction

**Call for Stakeholders**: Do you have a project that would benefit from Asynchronous Pipeline Parallelism in PyTorch? Please comment on the RFC and we can incorporate your requirements.

* [Parametrization](https://pytorch.org/tutorials/intermediate/parametrizations.html) as an approach?

## Stage 5: Integrate into Unified Programming Models Research (2022?)

Going into the future, we would like to develop theory and implementation for a unified distributed, parallel programming model that brings together all of data parallel, model parallel, pipeline parallel, expert parallel, and more. Various ideas are floating around, including building on top of the Actor model (as in Ray, OneFlow, etc) or extending the MPI-style SPMD model to support spatial parallelism like pipeline parallelism and predicated expert parallelism. Hopefully, this pipeline parallelism project will help to inform us on the correct model here and we can publish our findings in the future.


## References

1. Efficient and Robust Parallel DNN Training through Model Parallelism on Multi-GPU Platform https://arxiv.org/abs/1809.02839
2. ElasticPipe: An Efficient and Dynamic Model-Parallel Solution to DNN Training https://dl.acm.org/doi/10.1145/3322795.3331463
3. XPipe: Efficient Pipeline Model Parallelism for Multi-GPU DNN Training https://arxiv.org/abs/1911.04610
4. PipeDream: Fast and Efficient Pipeline Parallel DNN Training https://arxiv.org/abs/1806.03377
5. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism https://arxiv.org/abs/1811.06965
6. torchgpipe: On-the-fly Pipeline Parallelism for Training Giant Models https://arxiv.org/abs/2004.09910
7. Pipelined Backpropagation at Scale: Training Large Models without Batches https://arxiv.org/abs/2003.11666
8. Memory-Efficient Pipeline-Parallel DNN Training https://arxiv.org/abs/2006.09503
9. Efficient Large-Scale Language Model Training on GPU Clusters https://arxiv.org/abs/2104.04473
10. Performance analysis of a pipelined backpropagation parallel algorithm https://ieeexplore.ieee.org/document/286892
11. PipeMare: Asynchronous Pipeline Parallel DNN Training https://arxiv.org/abs/1910.05124
12. Scaling Language Model Training to a Trillion Parameters Using Megatron
 https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/