

<details>
<summary>Instructions - click to expand</summary>

- Fork the rfcs repo: https://github.com/pytorch/rfcs
- Copy `RFC-0000-template.md` to `RFC-00xx-my-feature.md`, or write your own open-ended proposal. Put care into the details.
- Submit a pull request titled `RFC-00xx-my-feature`. 
    - Assign the `draft` label while composing the RFC. You may find it easier to use a WYSIWYG editor (like Google Docs) when working with a few close collaborators; feel free to use whatever platform you like. Ideally this document is publicly visible and is linked to from the PR.
    - When opening the RFC for general discussion, copy your document into the `RFC-00xx-my-feature.md` file on the PR and assign the `commenting` label.
- Build consensus for your proposal, integrate feedback and revise it as needed, and summarize the outcome of the discussion via a [resolution template](https://github.com/pytorch/rfcs/blob/master/RFC-0000-template.md#resolution).
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





# Redesign torch distributed interface in an object-oriented way

**Authors:**
* [@youkaichao](https://github.com/youkaichao)


## **Summary**

Remove global states in torch distributed, make it object-oriented.

## **Motivation**

I'm a developer of the [vLLM](https://github.com/vllm-project/vllm) project. vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs. One of its key features is distributed inference: we support tensor parallel and pipeline parallel, so that large models can be served efficiently.

When we develop vLLM, we find `torch.distributed` becomes a major pain point. It heavily relies on global states.

For example:

- If we have process A and process B called `torch.distributed.init_process_group` to form a group, there will be a global default process group object in process A and B. If we want to form another group with process A, B, C, D, we cannot call `torch.distributed.init_process_group` again, because the default group in process A and process B has been initialized. In vLLM, we have a [workaround](https://github.com/vllm-project/vllm/blob/ad39bd640cdaaf2963cd07a7cc912c1dde516ed0/vllm/distributed/utils.py#L94) to bypass this limitation, by calling many `torch.distributed` internal functions directly. (I got this idea when I discuss the issue with [@ezyang](https://github.com/ezyang) and [@wconstab](https://github.com/wconstab) during the PyTorch conference this year, and [@kwen2501](https://github.com/kwen2501) helped me settle down the final implementation.)

- Some distributed communication operations are naturally object-oriented, but they are forced to be linked to the global default group. For example, [`torch.distributed.send`](https://pytorch.org/docs/stable/distributed.html#torch.distributed.send)'s argument
`dst` means "Destination rank on global process group (regardless of group argument)", even if we provide a `group` argument, which is very unnatural. In vLLM, we create a [wrapper](https://github.com/vllm-project/vllm/blob/7371749d54db40999d896c4a7f8935bc6984c093/vllm/distributed/parallel_state.py#L557) around `ProcessGroup`, and make the `dst` argument to be the rank **inside** the group.

Before we go any further to call many PyTorch internal functions and build our own mini version of `torch.distributed`, I'd like to propose an RFC here first, and discuss with the PyTorch team to see if we have an aligned vision.

## **Proposed Implementation**

We should have a new subpackage `torch.distributed2`, with a new set of object oriented APIs. I don't think the engineering effort is too big, because we can reuse most of the existing code for the underlying communication primitives. We just need to refactor the APIs to be object-oriented.

- Creation of a new group should be an object-oriented operation. The usage should be like: `import torch.distributed2 as dist; group = dist.init_process_group(...)`. The arguments for `init_process_group` can be the same as the current `torch.distributed.init_process_group`, but the major difference is that it returns a `ProcessGroup` object, instead of placing the group as global state.
- When we have a `ProcessGroup` object, we can query the group size, rank, and other group properties, directly from the object. `group.rank` will naturally be the rank inside the group, not the global rank. In fact, there should not be a global rank anymore.
- Any processes can happily form multiple groups, without worrying about the global default group. E.g. process A and process B called `group1 = dist.init_process_group(...)`, and process A, B, C, D called `group2 = dist.init_process_group(...)`. They can happily communicate with each other using `group1` and `group2`. Process C and D do not need to know the existence of `group1`.
- All communication ops should be called from the `ProcessGroup` object. E.g. `group.send(tensor, dst=1)` will send `tensor` to rank 1 inside the group. The `dst` argument should be the rank inside the group, not the global rank. The same applies to `group.recv`, `group.broadcast`, `group.all_reduce`, etc.


## **Drawbacks**

The main drawback is that it is a breaking change. There are many legacy codebases that rely on the current global state. Therefore I'm proposing a new subpackage `torch.distributed2`, instead of modifying the existing `torch.distributed`.

Since the new design is object-oriented, it can naturally coexist with the old design. Users who want to use the new design can import `torch.distributed2` and use the new APIs. Users who want to use the old design can still use `torch.distributed`.

## **Unresolved questions**
This RFC mainly aims to tackle the usability and flexibility issue of `torch.distributed`. The underlying communication primitives are not changed. (I do think there's room for improvement in the underlying communication primitives, `torch.distributed` does too many additional things than purely communication, but that's another topic, maybe for another RFC.)
