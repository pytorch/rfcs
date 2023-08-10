# [PT2.1 Feature Proposal] SDPA (Scaled-Dot Product Attention) CPU Optimization

This ticket is as part of PT 2.1 feature proposal process.

## **Motivation**
As LLM tends to accept a large batch size and a long context length, the requirement of large memory may lead to OOM issues or result in bad performance. To reduce memory usage and provide a substantial speedup for attention-related models, it is important to optimize SDPA. The fused SDPA, e.g. flash attention, is one type of the optimized SDPA algorithms designed for memory-bound problems, with better parallelism and memory access patterns. In PT 2.0, there exist both the basic unfused SDPA and the fused SDPA for CUDA, while only the unfused SDPA has CPU implementation. To fill the gap between CPU and CUDA, it is proposed to optimize SDPA by implementing the fused SDPA for CPU in PT 2.1.

## **Implementation**
We submitted PRs for CPU SDPA optimization and demonstrated up to 3x performance speedup on attention-related benchmarks.

Here are the detailed implementation items:

*   The flash attention CPU kernel is added, in which both forward and backward paths are implemented for data types float32 and bfloat16. Blocking is applied on dimensions of query length and kv length and the fusion of gemm + softmax update + gemm is done at once for each block. Specifically, FP32In-FP32Out and BF16In-FP32Out adopt the mkl gemm and BF16In-BF16Out adopts the OneDNN one. Parallelization is on the dimensions of batch size, head number and query length for forward path, and on the dimensions of batch size and head number for backward path. In addition, the causal attention mask is supported. As the attention is masked for the unseen tokens, early termination is applied and we only calculate the blocks in the lower triangular part.
*   Write an SDPA selecting function for CPU to automatically choose one SDPA implementation among several ones. There are two CPU implementations which could be chosen: the unfused SDPA and flash attention. In general, flash attention has a higher priority than the unfused SDPA. For cases where flash attention is not applicable, such as maually disabling flash attention or the inputs not 4 dimensional, the unfused SDPA is chosen.

The following will be nice to have for PT 2.1:

*   Support data type of float16.
*   Enable the SDPA graph rewriting for Inductor.
*   Further block-related tuning for the fused SDPA.
*   Support Dropout for the fused SDPA.


## **Performance**
All validations are run on SPR machine.

### NanoGPT's SDPA kernel
Using benchmark [repo](https://github.com/mingfeima/bench_sdpa/blob/main/README.md), with one socket.
Shape: Batch size 1, Sequence length 1024, Head number 25, Head size 64.

| Dtype    | Causal   | Mode      | SDPA            | Time (ms per iter) | Speedup |
| -------- | -------- | -------   | -------         | -------            | ------- |
| float32  | FALSE    | Inference | Unfused         | 3.081              |         |
|          |          |           | Flash attention | 1.665              | 1.85045 |
| float32  | TRUE     | Inference | Unfused         | 3.463              |         |
|          |          |           | Flash attention | 1.662              | 2.083634|
| bfloat16 | FALSE    | Inference | Unfused         | 1.203              |         |
|          |          |           | Flash attention | 1.154              | 1.042461|
| bfloat16 | TRUE     | Inference | Unfused         | 1.543              |         |
|          |          |           | Flash attention | 1.154              | 1.337088|
| float32  | FALSE    | Training  | Unfused         | 54.938             |         |
|          |          |           | Flash attention | 23.029             | 2.385601|
| float32  | TRUE     | Training  | Unfused         | 58.266             |         |
|          |          |           | Flash attention | 17.835             | 3.266947|
| bfloat16 | FALSE    | Training  | Unfused         | 18.924             |         |
|          |          |           | Flash attention | 18.886             | 1.002012|
| bfloat16 | TRUE     | Training  | Unfused         | 21.08              |         |
|          |          |           | Flash attention | 14.172             | 1.48744 |

### Stable Diffusion
Following model's [BKM](https://github.com/intel-innersource/frameworks.ai.models.intel-models/blob/develop/quickstart/diffusion/pytorch/stable_diffusion/inference/cpu/README.md).

| Dtype    | SDPA                    | Throughput (fps) | Speedup SDPA | Total Time (ms) | Speedup |
| -------- | --------                | -------          | -------      | -------         | ------- |
| float32  | Unfused                 | 1.63             |              | 1139            |         |
|          | Flash attention         | 1.983            | 1.216564     | 547.488         | 2.080411|
| bfloat16 | Flash attention in IPEX | 4.784            |              | 429.051         |         |
|          | Flash attention         | 4.857            | 1.015259     | 408.823         | 1.049479|

## **Related PRs**
Flash attention Implementation:
*   [#103826 Flash attention forward](https://github.com/pytorch/pytorch/pull/103826)
*   [#104693 Flash attention backward](https://github.com/pytorch/pytorch/pull/103826)
*   [#104863 enable bfloat16 on flash attention](https://github.com/pytorch/pytorch/pull/104863)

SDPA selecting function:
*   [#105131 add sdpa choice and UT](https://github.com/pytorch/pytorch/pull/105131)

Some additional works:
*   [#104583 enable mkl’s bfloat16 gemm on PT](https://github.com/pytorch/pytorch/pull/104583)
*   [#104584 expand functional utils on CPU vectorization path](https://github.com/pytorch/pytorch/pull/104584)

## **Discussion**

For the SDPA optimization, there are two things that needed to be discussed and I hope to have your precious opinions.

One is about the util functions for SDPA selection. The current util functions are under the CUDA folder, i.e. `transformers/cuda/sdp_utils`. For CPU, we have similar functions in `transformers/sdp_utils_cpp` (see #105131). It is good to know whether we need to make them a unified API.

The other one is about GQA (Grouped-Query Attention), used in llama2. It interpolates between multi-head and multi-query attention and should be presented as a new feature in SDPA. If this feature is regarded as necessary, we can do this later.
