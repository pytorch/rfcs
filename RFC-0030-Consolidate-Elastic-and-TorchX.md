

<details>
<summary>Instructions - click to expand</summary>

- Fork the rfcs repo: https://github.com/pytorch/rfcs
- Copy `RFC-0000-template.md` to `RFC-00xx-my-feature.md`, or write your own open-ended proposal. Put care into the details.
- Submit a pull request titled `RFC-00xx-my-feature`. 
    - Assign the `draft` label while composing the RFC. You may find it easier to use a WYSIWYG editor (like Google Docs) when working with a few close collaborators; feel free to use whatever platform you like. Ideally this document is publicly visible and is linked to from the PR.
    - When opening the RFC for general discussion, copy your document into the `RFC-00xx-my-feature.md` file on the PR and assign the `commenting` label.
- Build consensus for your proposal, integrate feedback and revise it as needed, and summarize the outcome of the discussion via a [resolution template](https://github.com/pytorch/rfcs/blob/rfc-process/RFC-0000-template.md#resolution).
    - If the RFC is idle here (no activity for 2 weeks), assign the label `stalled` to the PR.
- Once the discussion has settled, assign a new label based on the level of support:
    - `accepted` if a decision has been made in the RFC
    - `draft` if the author needs to rework the RFC’s proposal
    - `shelved` if there are no plans to move ahead with the current RFC’s proposal. We want neither to think about evaluating the proposal
nor about implementing the described feature until some time in the future.
- A state of `accepted` means that the core team has agreed in principle to the proposal, and it is ready for implementation. 
- The author (or any interested developer) should next open a tracking issue on Github corresponding to the RFC.
    - This tracking issue should contain the implementation next steps. Link to this tracking issue on the RFC (in the Resolution > Next Steps section)
- Once all relevant PRs are merged, the RFC’s status label can be finally updated to `closed`.

</details>





# Consolidate TorchElastic and TorchX into `torch.x`

**Authors:**
* @kiukchung (Kiuk Chung)
* @d4l3k (Tristan Rice)


## **Summary**

Consolidate TorchElastic (`torch.distributed.elastic` + `torchrun`) and [TorchX](https://github.com/pytorch/torchx) 
into a single module called `torch.x` that launches PyTorch scripts (single-process & distributed)
both locally and as a job on remote schedulers (e.g. SLURM, Kubernetes, etc).

## **Background**

The sections below provide background/historical context on TorchElastic and TorchX. We do not go into
the details of how each library works but rather focus on the differences and similarities. Please refer 
to the corresponding documentation for further details

### TorchElastic

[torchelastic](https://pytorch.org/docs/stable/distributed.elastic.html), hereafter used interchangeably with
[`torchrun`](https://pytorch.org/docs/stable/elastic/run.html),
is the defacto local **process launcher** to kick-off PyTorch Distributed (PTD) scripts. 
Prior to `torch-1.9` TorchElastic resided in the PyTorch GitHub organization but under the 
[pytorch/elastic](https://github.com/pytorch/elastic) repository. 
In `torch-1.9` TorchElastic was upstreamed to torch under the `torch.distributed.elastic` submodule.

#### `torchrun` vs `torch.distributed.launch` vs `torch.distributed.run`
**TL;DR - All three tools use `torch.distributed.run` (torchelastic) under the hood**

The following table summarizes when each tool was introduced to PyTorch and how they relate to each other.

| **PyTorch Version**  | `torch.distributed.launch`                                   | `torch.distributed.run` | `torchrun`                                                                                                      |
|----------------------|--------------------------------------------------------------|-------------------------|-----------------------------------------------------------------------------------------------------------------|
| <1.9                 | Uses `torch.multiprocessing`                                 | --                      | --                                                                                                              |
| 1.9                  | Wraps `torch.distributed.run` with backwards compatible args | Uses `torchelastic`     | --                                                                                                              |
| 1.10                 | [same as above]                                              | [same as above]         | CLI [console_script](https://github.com/pytorch/pytorch/blob/master/setup.py#L980) to `torch.distributed.run` |


### TorchX

[TorchX](https://pytorch.org/torchx)
on the other hand is a **job launcher** that can submit PyTorch scripts as a job onto various
OSS schedulers such as SLURM, Kubernetes, and Ray as well as
cloud-provider-specific batch compute service offerings such as AWS Batch, GCP Batch, and Azure Batch.
TorchX covers all the most widely used AI schedulers today. By controlling the specifications and 
parameters of the job, *TorchX acts as a plugin-point
to the infrastructure*, much like how TorchDynamo/Inductor is a plugin-point for hardware accelerators.


When submitting distributed jobs, TorchX instructs the target scheduler to
run `torchrun` on each node (see diagram below).

![`torchx` vs `torchrun`](RFC-0030-assets/torchx_torchrun.png)

TorchX is currently in the PyTorch GitHub organization but under the 
[pytorch/torchx](https://github.com/pytorch/torchx) repository.


## **Motivation**

Uniform and improved user-experience (UX) in running PyTorch applications.

Enhancing `torchrun` with `torchx` capabilities enables users to quickly integrate 
PyTorch applications to their existing infrastructures. Consolidating to a single tool simplifies
the UX and setup. Through `torchx`'s infrastructure plugin points, PyTorch can leverage infra features
that reduce authoring complexity while improving reliability. One such example is in-place retries for 
distributed training on schedulers that support node-replacements. In-place retries have
a much lower restart overhead compared to job-level retries, but requires the user to carefully
coordinate the configuration of the infra (scheduler) as well as the application (`torchrun`).
Yet another example is running DDP training on GPUs where each rank occupies exactly one CUDA device. 
The user has to ensure that they request the job to be run on hosts with the GPU count
matching the `local_world_size`. TorchX makes this simple by taking a single argument: either the host
type or `nproc_per_node` and automatically configures the other with-respect-to the user specified one.

PyTorch applications are becoming more complex. Distributed training is a norm with LLMs.
Cloud providers are putting forth custom chips. Some, such as TPU (GCP) and Trainium (AWS), are AI-accelerators
with implications on compatibility with existing PyTorch models and ops. Others, such as IPUs in AWS' Nitro system
and EFA, do not directly run compute but need to be configured correctly to work with PyTorch.
Either way, there exists more runtime couplings between PyTorch and the infrastructure than one might think or like.
Unfortunately today, `torch` (out-of-the-box) does not offer any type of specifications, standards, or
plugin-points to the infrastructure. Users are on their own and unsurprisingly,
it is easier to author a DDP or FSDP trainer than it is to get it to run on a cluster.
This UX gap has created an opportunity for CSPs and 3rd party platforms to create
a myriad of "MLOps" offerings, enticing users with easy on-boarding and quick results.
Unfortunately, most of these tools are vertically integrated with the provider's ecosystem,
with a thick air-gap between native PyTorch and the user.

An interesting observation is that despite the amount of fragmentation in the tooling for PyTorch,
when it comes to process launchers, most of these tools either directly invoke or are compatible with
`torchrun`. This is the case with [Accelerate](https://huggingface.co/blog/pytorch-ddp-accelerate-transformers),
[DeepSpeed](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/launcher/launch.py#L29), 
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/main/examples/run_text_generation_server_345M.sh#L17),
and [SageMaker](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#launching-a-distributed-training-job).
So there is something to be said about the acceptance and adoption of standards when it belongs in 
the core pytorch library versus it being offered as a separate and optional install. 
Having TorchX specs/APIs in core encourages the community to adopt and extend PyTorch-defined infrastructure
standards. For the user, this means better less fragmentation, better portability and uniform UX. 
For PyTorch, this means being able to leverage and dictate its runtime requirements and environment.
 

## **Proposed Implementation**

**TL;DR - Upstream TorchX (specs and sensible default impls only) to PyTorch under a new sub-module called `torch.x` and pull TorchElastic out of
`torch.distributed` and merge it with `torch.x`**

The diagram below depicts the proposed module move + upstream (optional merge of the `multiprocessing` module not shown).

![`torch.x` Before and After](RFC-0030-assets/modules_before_after.png)

**The following changes are proposed (refer to diagram above):**

1. Move `torch.distribted.elastic.*` &rarr; `torch.x.elastic.*`
2. Upstream `torchx.*` &rarr; `torch.x.*`. Here we have three options:
     1. (Option 1) Upstream specs only:
          1. `torchx.specs` (the interfaces and APIs)
          2. Default implementations of said specs
          3. Other implementations would remain in TorchX repo
          4. PROS: Upstreams the minimal set of functionalities to have torch benefit from the topics mentioned in the [Motivation](#motivation) section 
          5. CONS: 
              1. Effectively splits the TorchX repo in two hence makes maintenance and CI/CD (testing) more complex
              2. No built-in support for launching remote jobs
     2. **[ PREFERRED ] (Option 2) Option 1 + Kubernetes and SLURM support**
          1. PROS: Keeps the minimalistic approach of Option 1 while providing out-of-the-box for CSP agnostic remote scheduler support
          2. CONS: TorchX still split into two repos
         
        > NOTE: The rest of the doc assumes Option 2 as this is the recommended option
     3. (Option 3) Upstream everything in TorchX:
          1. PROS: Makes maintenance and CI/CD simple since changes to the specs can be tested for downstream BC easily
          2. CONS: bloat-torch
3. Merge the functionalities of `torchx` CLI to
   `torchrun` (a python [`console_script`](https://python-packaging.readthedocs.io/en/latest/command-line-scripts.html)
   that points to `torch.distributed.run`) 
   allowing  `torchrun` to:
     1. Submit PyTorch scripts as a job to remote schedulers (9 x OSS schedulers + 2 x FB-internal schedulers)
     2. Run (simulate) multi-node distributed jobs locally on a single machine with multiple GPUs
4. **(Alternative Option)** keep `torchrun` for BC (eventually to be removed), and consolidate around `torchx` CLI.
4. **(Optional)** Merge `torch.distributed.elastic.multiprocessing` into `torch.multiprocessing`. Adds the following features to `torch.multiprocessing`:
     1. Launch binaries (e.g. `/home/kiuk/train.py`) in addition to functions (e.g. `pkg.my.train:main`)
     2. Write each sub-proc's stdout and stderr to log files (configurable)
     3. Prefix each line of stdout and stderr with local rank information
     4. Propagate exception traces from sub-proc back to the main proc for better error handling + propagation

### Backwards Compatibility (BC)

#### non-BC Breaking Changes

1. `torch/distributed/launch.py` and `torch/distributed/run.py` would have to remain for a few releases,
   but will both point to `torch/x/run.py`.
2. If we decide to consolidate the CLI into `torchrun`, the arguments to the CLI `torchrun` would have to remain to ensure BC.
   We propose an "opt-in" mechanism where if the user drops a marker file in (say) `~/.torch/.torchx`, then `torchrun` 
   operates like today's `torchx` CLI, otherwise, if the marker file is not found, then `torchrun` operates like the existing `torchrun`.
3. (Meta-internal only) `torchx` CLI is widely used internally at Meta. In order to minimize disruptions, `torchx` CLI
    can be made a symlink to `torchrun` and gradually have users shift over to `torchrun`. This can be measured by tracking
    direct calls to `torchx` in Scuba.


#### BC Breaking Changes

The use-cases below will be BC breaking under this proposal. For each item, we discuss ways to make them non-BC breaking.
However we propose that we clearly document the migration process and add references to these docs in the release notes
in favor of introducing code complexity and/or tech-debt to keep things BC.


1. Programmatic usages of torchelastic (e.g. `import torch.distributed.elastic`) are **NOT** BC and the user has to codemod
   references to `torch.distributed.elastic` to `torch.x.elastic`.
    1. **Impact**: Besides Meta-internal use-cases (which can be easily codemoded) the only external programmatic usage
        of torchelastic is in DeepSpeed (see [GitHub](https://github.com/microsoft/DeepSpeed/search?q=torch.distributed.elastic))
        for which we can work with the project owner to resolve.
    1. **To make BC**: We could re-import `torch.x.elastic.**` from `torch/elastic/**/__init__.py`. For example,  
        ```python
        # In torch/distributed/elastic/__init__.py
        import torch.x.elastic.* # noqa F401
        ```
2. (OSS-only) `torchx` CLI users would have to switch over to `torchrun`.
    1. **Impact**: Every OSS user that currently uses `torchx` CLI would have to one-time opt-in and switch over to torchrun
    2. **To make BC**: 
         1. **(Option 1)** Add console script `torchx` to PyTorch's `setup.py`, which would act as a symlink discussed in the non-BC section for Meta-internal use-case
         2. **(Option 2)** Release a final version of `torchx` wheel where the `torchx` CLI walks the user
            through the opt-in step and asks the user to switch over to `torchrun`.  
 

### Dependencies, and CI

#### Dependencies 

The additional install deps that upstreaming TorchX to PyTorch would bring are:

```
docstring-parser (==0.8.1)
urllib3 (<1.27,>=1.21.1)
tabulate # not strictly needed (can be removed, but requires code change in torchx CLI)
pyyaml # already defined in pytorch/requirements.txt but not a install-time dep of torch
fsspec # already defined in pytorch/requirements.txt but not a install-time dep of torch
```

TorchX defines [`extras-require`](https://github.com/pytorch/torchx/blob/main/setup.py#L83) based on 
the type of infrastructure the user wants to use TorchX on. For instance:

```
pip install torchx[kubernetes] # torchx + kubernetes>=11
pip install torchx[gcp_batch] # torchx + google-cloud-(batch + logging + runtimeconfig)
pip install torchx[ray] # torchx + ray>=1.12.1
```

For `extras-require` we can either:

1. **(Option 1)** Create `[extras]-requirements.txt` (e.g. `kubernetes-requirements.txt`) and
    have the user `pip install -r [extras]-requirements.txt`
2. **(Option 2)** At runtime display a warning message if the user hits a codepath and does not have the deps installed:
    ```shell-session
    # sample warning message
    $ pip install torch
    $ torchrun -s aws_batch dist.ddp -j 1x2 train.py
    =================================================
    ERROR: Unsatisfied dependencies [boto3, docker]
    To install run:

    $ pip install boto3==1.24.59, docker
    =================================================
    ```
3. **(Option 3 - not desirable, including for completeness)** Adding `extras-require` to torch installs (e.g. `pip install torch[gcp]`)

#### CI

1. **Linting**: static type checker needs to change from pyre (currently used by torchx) &rarr; mypy (used by torch).
   Once migrated over to mypy the regular PyTorch [Lint GitHub action](https://github.com/pytorch/pytorch/blob/main/.github/workflows/lint.yml) should suffice.
2. **Conditional Integration Tests**: TorchX runs two types of end-to-end integ tests to ensure that it can submit jobs correctly to the supported schedulers:
     1. [Components Integration Test](https://github.com/pytorch/torchx/blob/main/.github/workflows/components-integration-tests.yaml):
        Runs locally on the CI runner by using mock/local schedulers (e.g. minikube for kubernetes)  
     2. [SCHEDULER] Integration Tests (e.g. [aws-batch-integration-tests](https://github.com/pytorch/torchx/blob/main/.github/workflows/aws-batch-integration-tests.yaml)):
       Runs against scheduler instances running on an actual cloud (e.g. aws, gcp) account.
   
   While (i) is lighter weight compared to (ii) both are rather expensive to run at the pace of commits to PyTorch.
   Therefore, we propose that these integ tests are run:
     1. Only for commits that touch files in `torch/x/**`
     2. On nightly releases of torch 


### Documentation

The docs page https://pytorch.org/torchx/latest/ should move to https://pytorch.org/docs/stable/index.html
under the `torch.x` submodule. `torch.distributed.elastic` docs should get merged into `torch.x`.

## **Drawbacks**

See section on BC above.

## **Alternatives**

> NOTE: Where appropriate we discussed alternatives inlined with itemized **(Option #)**, **(Alternative Option)** or **(Optional)**
>  for more local context and clarity on the pros and cons.

### Upstreaming TorchX vs Pulling TorchElastic
We considered pulling `torch.distributed.elastic` (torchelastic) out to the pytorch/torchx repository and
consolidate TorchElastic and TorchX *outside* the pytorch/pytorch repo. However, due to the prevalent usage
of `torchrun`, pulling torchelastic out of PyTorch would mean:

1. Makes `torch-2.x` backwards incompatible since users would have to separately install `torchx` to get `torchrun`. 
    The installation instructions on https://pytorch.org could include `torchx` as part of the `pip` command 
    (similar to `torchvision`).
2. -- or -- (to make things BC) have PyTorch take a pip dependency on `torchx`.

Additionally, due to the reasons mentioned in the [Motivation](#motivation) section, there is value in upstreaming
TorchX's specs APIs to core.

### Level of Support
Pending...

Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.


#### Additional Context
N/A


### Next Steps

1. Pending RFC feedback
2. Meet-up between TorchX/Elastic team at Meta and external maintainers proposed (date and place pending)


#### Tracking issue
N/A

