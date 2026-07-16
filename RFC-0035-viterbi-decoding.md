

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
* @maxrmorrison


## **Summary**

We want to add Viterbi decoding to PyTorch. Viterbi decoding is a well-known algorithm that finds the path of maximum likelihood over a time-varying distribution. It is used in automatic speech recognition, bioinformatics, digital communications, and other tasks that produce models that infer or generate sequences of probability distributions. No implementation of Viterbi decoding exists in PyTorch, and no convenient alternative implementation exists for ML practitioners that is fast enough to scale to large datasets. We have created batched CPU and GPU implementations of Viterbi decoding significantly faster than available implementations. We have found our implementations useful for our own research tasks, and believe the community may find them useful as well.


## **Motivation**

Viterbi decoding is a generally useful algorithm that is missing from the PyTorch library, with applications in automatic speech recognition, bioinformatics, digital communications, and more. However, Viterbi decoding is O(C^2T) for C classes and T timesteps, making it challenging to scale to large datasets and real-time applications. A commonly-used implementation of Viterbi decoding exists in Librosa (`librosa.sequence.viterbi`). We use Librosa's implementation as a reference for correctness and as a baseline for benchmarking. Our benchmark uses `C = 1,440` states and approximately `T ~= 20 million` time steps across approximately 40k files. We compare our proposed implementation to the reference implementation in Librosa ([`librosa.sequence.viterbi`](https://librosa.org/doc/main/generated/librosa.sequence.viterbi.html)) that uses just-in-time compilation via numba.

| Method  | Timesteps decoded per second |
| ------------- | ------------- |
| Librosa (1x cpu)| 208 |
| Librosa (16x cpu)| 1,382* |
| Proposed (1x cpu)| 171 |
| Proposed (16x cpu)| **2,240** |
| Proposed (1x a40 gpu, batch size 1)| **3,944,452** |
| Proposed (1x a40 gpu, batch size 512)| **692,160,422** |

*By default, librosa.sequence.viterbi uses one CPU thread. We use a Multiprocessing pool to parallelize.

Our proposed implementation is fast enough that we are considering novel use cases of Viterbi decoding in future work, such as decoding optimal high-resolution sequences during the training of a neural network.


## **Proposed Implementation**

Viterbi decoding consists of two stages: (1) construction of a _trellis_ matrix containing path probabilities, and (2) backtracing along the maximal path. We have developed and open-sourced fast CPU and CUDA implementations of both stages. We think our implementations would be a viable starting point for adding Viterbi decoding to PyTorch.

Our current implementation is structured as follows.

* A python wrapper module ([torbi](https://github.com/maxrmorrison/torbi))
    * A C++, Pybind11 style Torch extension ([viterbi.cpp](https://github.com/maxrmorrison/torbi/blob/main/torbi/viterbi.cpp))
        * A `viterbi_make_trellis_cpu` CPU function which uses OpenMP (with SIMD) to parallelize some loops. ([viterbi.cpp](https://github.com/maxrmorrison/torbi/blob/main/torbi/viterbi.cpp))
        * A `viterbi_make_trellis_kernel` CUDA kernel which parallelizes one sequence per thread block ([viterbi_kernel.cu](https://github.com/maxrmorrison/torbi/blob/main/torbi/viterbi_kernel.cu))
        * A `viterbi_backtrace_trellis_cpu` CPU function which does the final decoding ([viterbi.cpp](https://github.com/maxrmorrison/torbi/blob/main/torbi/viterbi.cpp))
        * A `viterbi_backtrace_tellis_kernel` CUDA kernel which does the final decoding on the GPU ([viterbi_kernel.cu](https://github.com/maxrmorrison/torbi/blob/main/torbi/viterbi_kernel.cu))


We propose a Python API and underlying C++/CUDA extensions for Viterbi decoding in PyTorch. This proposal is a draft; we welcome input and opinions naming and implementation. Specifically, we propose adding a `torch.viterbi.decode` function (and corresponding stateful `torch.viterbi.Decoder`) that makes use of underlying functions `torch.viterbi.make_trellis` and `torch.viterbi.backtrace_trellis`.


#### `torch.viterbi.decode`

```
def decode(
    observation: torch.Tensor,
    batch_frames: torch.Tensor,
    transition: torch.Tensor,
    initial: torch.Tensor
):
    """Decode a time-varying categorical distribution

    Args:
        observation: :math:`(N, T, S)` or :math:`(T, S)`
            where `S = the number of states`,
            `T = the length of the sequence`,
            and `N = batch size`.
            Time-varying categorical distribution
        batch_frames :math:`(N)`
            Sequence length of each batch item
        transition :math:`(S, S)`
            Categorical transition matrix
        initial :math:`(S)`
            Categorical initial distribution

    Return:
        indices: :math:`(N, T)`
            The decoded bin indices

    Example::

            >>> observation = torch.tensor([[
            >>>     [0.25, 0.5, 0.25],
            >>>     [0.25, 0.25, 0.5],
            >>>     [0.33, 0.33, 0.33]
            >>> ]])
            >>> batch_frames = torch.tensor([3])
            >>> transition = torch.tensor([
            >>>     [0.5, 0.25, 0.25],
            >>>     [0.33, 0.34, 0.33],
            >>>     [0.25, 0.25, 0.5]
            >>> ])
            >>> initial = torch.tensor([0.4, 0.35, 0.25])
            >>> bins = torch.viterbi.decode(
            >>>     observation,
            >>>     batch_frames,
            >>>     transition,
            >>>     initial)
    """
```


#### `torch.viterbi.make_trellis`

```
def make_trellis(
    observation: torch.Tensor,
    batch_frames: torch.Tensor,
    transition: torch.Tensor,
    initial: torch.Tensor
) -> torch.Tensor:
    """Perform first step of Viterbi decoding to construct the path trellis

   Args:
        observation: :math:`(N, T, S)` or :math:`(T, S)`
            where `S = the number of states`,
            `T = the length of the sequence`,
            and `N = batch size`.
            Time-varying categorical distribution
        batch_frames :math:`(N)`
            Sequence length of each batch item
        transition :math:`(S, S)`
            Categorical transition matrix
        initial :math:`(S)`
            Categorical initial distribution

    Return:
        trellis: :math:`(N, T, S)`
            Matrix of minimum path indices for backtracing
    """
```


#### `torch.viterbi.backtrace_trellis`

```
def backtrace_trellis(
    trellis: torch.Tensor,
    batch_frames: torch.Tensor,
    transition: torch.Tensor,
    initial: torch.Tensor
) -> torch.Tensor:
    """Perform second step of Viterbi decoding to backtrace optimal path

    Args:
        trellis: :math:`(N, T, S)`
            Matrix of minimum path indices for backtracing
        batch_frames :math:`(N)`
            Sequence length of each batch item
        transition :math:`(S, S)`
            Categorical transition matrix
        initial :math:`(S)`
            Categorical initial distribution

    Return:
        indices: :math:`(N, T)`
            The decoded bin indices
    """
```


#### `torch.viterbi.Decoder`

```
class Decoder:
    """Stateful Viterbi decoder that stores transition and initial matrices"""

    def __init__(
        self,
        transition: torch.Tensor,
        initial: torch.Tensor
    ) -> None:
        """
        Args:
            transition :math:`(S, S)`
                Categorical transition matrix
            initial :math:`(S)`
                Categorical initial distribution
        """

    def decode(
        self,
        observation: torch.Tensor,
        batch_frames: torch.Tensor
    ) -> torch.Tensor:
        """Decode a time-varying categorical distribution

        Args:
            observation: :math:`(N, T, S)` or :math:`(T, S)`
                where `S = the number of states`,
                `T = the length of the sequence`,
                and `N = batch size`.
                Time-varying categorical distribution
            batch_frames :math:`(N)`
                Sequence length of each batch item

        Return:
            indices: :math:`(N, T)`
                The decoded bin indices
        """

    def make_trellis(
        self,
        observation: torch.Tensor,
        batch_frames: torch.Tensor
    ) -> torch.Tensor:
        """Perform first step of Viterbi decoding to construct the path trellis

        Args:
            observation: :math:`(N, T, S)` or :math:`(T, S)`
                where `S = the number of states`,
                `T = the length of the sequence`,
                and `N = batch size`.
                Time-varying categorical distribution
            batch_frames :math:`(N)`
                Sequence length of each batch item

        Return:
            trellis: :math:`(N, T, S)`
                Matrix of minimum path indices for backtracing
        """

    def backtrace_trellis(
        self,
        trellis: torch.Tensor,
        batch_frames: torch.Tensor
    ) -> torch.Tensor:
        """Perform second step of Viterbi decoding to backtrace optimal path

        Args:
            trellis: :math:`(N, T, S)`
                Matrix of minimum path indices for backtracing
            batch_frames :math:`(N)`
                Sequence length of each batch item

        Return:
            trellis: :math:`(N, T, S)`
                Matrix of minimum path indices for backtracing
        """
```

### CUDA Algorithm

Our CUDA algorithm makes efficient use of warps to cache posterior probabilities in shared memory. The core design is nested loop, first over timesteps, and then over possible states. One warp is assigned to each state to compute posterior distributions and then perform a parallel argmax (with reduction) to find the best next state from the current state that the warp is assigned to.

The warps iterate over the input states for cases where there are more than 32 input states (i.e., the number of warps in a block).

Instead of storing the entire posterior distribution (as in the Librosa implementation), we only store the current and next timesteps, reducing memory usage. To avoid expensive memory copies, we use pointers to switch which array stores current values and which stores next values. In addition, to support a variable number of input states, these two arrays are just pointers to the two halves of a shared memory array which is sized externally.

Because we use only a single block per input sequence, we can process a batch of input sequences very quickly in parallel, depending on the GPU in use. This also cuts down on the number of kernel-invocation-style syncs that must be performed.


## **Prior Art**
* We tested a variety of other implementations, which were all slower:
    * Pure Python torch implementation
    * Cython numpy implementation
    * Cython implementation (without numpy operations)
    * C++ implementation without OpenMP
    * [Librosa Numba implementation](https://librosa.org/doc/main/generated/librosa.sequence.viterbi.html)
* Our implementation is [currently open source](https://github.com/maxrmorrison/torbi/). However, due to the [complexities of packaging cross-platform torch extensions](https://github.com/pytorch/builder/issues/468#issuecomment-661943587), it currently must be built from source. Adding our implementation to `torch` allows us to use `torch`'s existing cross-platform build system instead of hand-rolling our own.


## **Discussion questions**

* Are there desired changes in the naming conventions?
* Right now our implementation is written as a PyTorch extension. How can it be converted to something like a `TORCH_MODULE_FRAGMENT`?
* Are there recommended methods for ensuring compliance over a set of allowed dtypes? Our implementation currently works for torch.float32, but is not guaranteed to work for all types.
* Currently our kernel only supports recent compute capabilities (7 and later?) and makes assumptions about that capability. Ideally this would be generalized to easily support new compute capabilities as they are announced. The assumptions made are the following:
    * The number of threads in a block
    * The number of threads in a warp
    * The number of warps in a block
* Does torch allow the use of OpenMP, as we use in our CPU implementation? If not, what is used instead?
