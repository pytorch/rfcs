

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





# [viterbi-decoding]

**Authors:**
* @CameronChurchwell
* @maxmorrison


## **Summary**
Add Viterbi Algorithm Decoding operation to torch!


## **Motivation**
Viterbi decoding is a very common algorithm which achieves the maximum likelihood over a sequence of states at the cost of speed. In our research on speech editing, we have observed a massive improvement in pitch reconstruction accuracy when the pitch inputs are decoded using Viterbi as opposed to other methods, but commonly used implementations like that in Librosa are quite slow and are not feasible for use with larger datasets.


## **Proposed Implementation**
We have implemented the Viterbi Decoding Algorithm in five parts:
* A python wrapper module ([torbi](https://github.com/maxrmorrison/torbi))
    * A C++, Pybind11 style Torch extension ([viterbi.cpp](https://github.com/maxrmorrison/torbi/blob/main/torbi/viterbi.cpp))
        * A `viterbi_make_trellis_cpu` CPU function which uses OpenMP (with SIMD) to parallelize some loops. ([viterbi.cpp](https://github.com/maxrmorrison/torbi/blob/main/torbi/viterbi.cpp))
        * A `viterbi_make_trellis_kernel` CUDA kernel which parallelizes one sequence per thread block ([viterbi_kernel.cu](https://github.com/maxrmorrison/torbi/blob/main/torbi/viterbi_kernel.cu))
        * A `viterbi_backtrace_trellis_cpu` CPU function which does the final decoding ([viterbi.cpp](https://github.com/maxrmorrison/torbi/blob/main/torbi/viterbi.cpp))
        * A `viterbi_backtrace_tellis_kernel` CUDA kernel which does the final decoding on the GPU ([viterbi_kernel.cu](https://github.com/maxrmorrison/torbi/blob/main/torbi/viterbi_kernel.cu))

We have also implemented a series of tests and [benchmarks](https://github.com/maxrmorrison/torbi/blob/main/torbi/evaluate/core.py) to evaluate our method against the implementation in Librosa. See [metrics](#metrics)

### CUDA Algorithm

Our CUDA algorithm makes efficient use of warps to cache posterior probabilities in shared memory. The core design is nested loop, first over timesteps, and then over possible states. One warp is assigned to each state to compute posterior distributions and then perform a parallel argmax (with reduction) to find the best next state from the current state that the warp is assigned to.

The warps iterate over the input states for cases where there are more than 32 (#warps in a block) input states.

Instead of storing the entire posterior distribution as in the Librosa implementation, we only store the current and next timesteps, drastically reducing memory usage. To avoid expensive memory copies, we use pointers to switch which array stores current values and which stores next values. In addition, to support a variable number of input states, these two arrays are just pointers to the two halves of a shared memory array which is sized externally.

Because we use only a single block per input sequence, we can process a batch of input sequences very quickly in parallel, depending on the GPU in use. This also cuts down on the number of kernel-invocation-style syncs that must be performed.

## **Metrics**
All recorded with batch size 512 on VCTK

| Method  | Real Time Factor (higher is better) |
| ------------- | ------------- |
| Librosa (1x cpu)|  |
| Librosa (16x cpu)| |
| Torbi (1x cpu)|  |
| Torbi (16x cpu)| |
| Torbi (1x RTX 4090)| |

## **Drawbacks**
* The only real drawback is that it could be considered bloat by some.


## **Alternatives**
* Our design is currently open source so anyone wanting to make use of it need only install it. Unfortunately, due to the [well known difficulties](https://github.com/pytorch/builder/issues/468#issuecomment-661943587) with packaging torch extensions, it must be built from source which requires users to have installed the cuda toolkit and g++ which satisfy version constraints.
* We tested a variety of other implementations which ultimately were all slower:
    * Pure Python torch implementation
    * Cython numpy implementation
    * Cython implementaiton (without numpy operations)
    * C++ implementation without OpenMP
    * Librosa Numba implementation


## **Prior Art**
[Current librosa implementation](https://librosa.org/doc/main/generated/librosa.sequence.viterbi.html)


## **How we teach this**
* No reorganization of documentation would be necessary to the best of my knowledge.
* Ideally, this would take no more work to document than any other `torch.nn.functional` function.


## **Unresolved questions**
* Right now our implementation is written as a pytorch extension. How can it be converted to something like a `TORCH_MODULE_FRAGMENT`?
* How can our implementation be changed to support float16 and float64 types in addition?
* Currently our kernel only supports recent compute capabilities (7 and later?) and makes assumptions about that capability. Ideally this would be generalized to easily support new compute capabilities as they are announced. The assumptions made are the following:
    * The number of threads in a block
    * The number of threads in a warp
    * The number of warps in a block
* Does torch allow the use of OpenMP?


## Resolution
We decided to do it. X% of the engineering team actively approved of this change.

### Level of Support
Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.


#### Additional Context
Some people were in favor of it, but some people didn’t want it for project X.


### Next Steps
Will implement it. 


#### Tracking issue
<github issue URL>


#### Exceptions
Not implementing on project X now. Will revisit the decision in 1 year.
