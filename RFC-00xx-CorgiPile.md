# Add new shuffle mechanism for SGD/Adam --- CorgiPile without full data shuffle

**Authors:**

* Lijie Xu ( xulijie AT iscas DOT ac DOTcn )
* Zheng Qin ( qinzheng19 AT otcaix DOT iscas DOT ac DOT cn )
* Ran Yu ( yuran24 AT otcaix DOT iscas DOT ac DOT cn )


## **Summary**
**Stochastic Gradient Descent** (SGD) and **SGD-like** methods (e.g., Adam) are commonly used in **PyTorch** to train ML models. However, these methods rely on random data order to converge, which usually require a **full data shuffle**, leading to **low** I/O performance for disk-based storage. 

We proposed a simple but novel two-level data shuffling strategy named **CorgiPile** (https://link.springer.com/article/10.1007/s00778-024-00845-0), which can avoid a full data shuffle while maintaining **comparable convergence rate** as if a full shuffle were performed. CorgiPile first **samples** and **shuffles** data at the *block-level*, and then **shuffles** data at the *tuple-level* within the sampled data blocks, i.e., firstly shuffling data blocks, and then merging sampled blocks in a small buffer, and finally shuffling tuples in the buffer for SGD. We have implemented CorgiPile inside PyTorch (https://github.com/DS3Lab/CorgiPile-PyTorch), and extensive experimental results show that our CorgiPile  can achieve **comparable** convergence rate with the full-shuffle based SGD, and **faster** than PyTorch with full data shuffle.


## **Motivation** and Response from Community
In typical PyTorch usage scenarios, whether on a **single machine** or in **distributed settings**, training machine learning models with Stochastic Gradient Descent (SGD) or SGD-like optimizers (e.g., Adam) often involves handling **large** datasets that **exceed memory capacity**. This necessitates reading data from disk, **shuffling** it to ensure randomness for convergence, and then feeding it into the optimizer. However, the full data shuffle process introduces significant **performance bottlenecks**: disk I/O is inherently slow, especially for massive datasets, and generating a fully shuffled version of the data requires substantial additional storage space and computational overhead to create and manage new data files or structures.

Current approaches in PyTorch (and similarly in TensorFlow) mitigate this by using **sliding-window** techniques to load data into a buffer in chunks, followed by shuffling within that buffer. While this reduces the need for a complete upfront shuffle, it only achieves **partial randomization**, which can lead to **suboptimal convergence rates**. The limited shuffle scope within the buffer often results in correlated batches, biasing gradients and slowing down or destabilizing training, particularly in scenarios where data order matters for generalization.

Therefore, there is a clear need for PyTorch to incorporate more **advanced randomization mechanisms** in shuffling, with a primary focus on preserving convergence quality. This is especially critical for large-scale datasets common in modern ML tasks, such as **image datasets** like ImageNet, **video** processing pipelines, or vast **LLM** corpora, where inefficient shuffling can amplify I/O delays, increase training times, and degrade model performance without the benefits of true randomness.

To address these challenges, We have already implemented **CorgiPile** within PyTorch, and our extensive experiments demonstrate that it achieves **equivalent convergence** to traditional full-shuffle methods but runs **faster** than PyTorch's  with full data shuffle, making it ideal for disk-bound, large-scale training.

CorgiPile has garnered positive feedback and adoption in various communities and systems beyond our initial implementation. For instance:

-  **CorgiPile-PyTorch**:  Integrated into PyTorch for efficient data shuffling in database-driven ML pipelines, avoiding the full shuffle while maintaining comparable convergence rate by designing new parallel/distributed shuffle operators as well as a new DataSet API (https://github.com/DS3Lab/CorgiPile-PyTorch).
-  **CorgiPile-PostgreSQL**: Integrated into PostgreSQL for efficient data shuffling in database-driven ML pipelines, improving query and training performance on large stored datasets (https://github.com/DS3Lab/CorgiPile-PostgreSQL).
- **CorgiPile-openGauss (GaussML)**: Adopted in the openGauss , enhancing shuffled data access for distributed ML workloads with reduced I/O latency (https://ieeexplore.ieee.org/document/10597842).
- **Mobileye's Corg²**: An improved variant used by Mobileye, which applies CorgiPile twice—once offline for initial data preparation and once online during training—to further optimize for real-time autonomous driving data processing (https://arxiv.org/pdf/2309.01640).
- **LLM Training Systems**: Enhanced versions of CorgiPile have been employed in MLSys-inspired frameworks for LLM pretraining, where handling terabyte-scale corpora benefits from the method's efficiency, as evidenced by community discussions and adaptations in open-source LLM tools (https://openreview.net/forum?id=I2LF8QHaua).


## **Design**
The following figure illustrates the implementation of CorgiPile with new operators and double-buffering optimization, in PyTorch.

![](RFC-00xx-asset/SingleMachine.jpg)

The key idea of CorgiPile lies in the following two-level hierarchical shuffling mechanism:

We first **randomly select** a set of blocks (each block refers to a set of contiguous tuples) and put them into an in-memory buffer; we then randomly **shuffle** all tuples in the buffer and use them for the SGD computation.

At each epoch (say, the s-th epoch), CorgiPile runs the following steps:

1. **(Sample)** Randomly sample n blocks out of N data blocks without replacement and load the n blocks into the buffer. Note that we use sample without replacement to avoid visiting the same tuple multiple times for each epoch, which can converge faster and is a standard practice in most ML system
2. **(Shuffle)** Shuffle all tuples in the buffer. We use ψ<sub>s </sub> to denote an ordered set, whose elements are the indices of the shuffled tuples at the s-th epoch. The size of ψ<sub>s </sub> is bn, where b is the number of tuples per block. ψ<sub>s </sub>(k) is the k-th element in ψ<sub>s </sub> .
3. **(Update)** Perform gradient descent by scanning each tuple with the shuffle indices in ψ<sub>s </sub>, yielding the updating rule

$$
\mathbf{x}_k^s=\mathbf{x}_{k-1}^s-\eta_s \nabla f_{\boldsymbol{\psi}_s(k)}\left(\mathbf{x}_{k-1}^s\right)
$$

We have demonstrated that CorgiPile-SGD serves as an intermediate approach between full Gradient Descent (GD) and standard Stochastic Gradient Descent (SGD). Our analysis, detailed in Section 4.2 of the paper (https://link.springer.com/article/10.1007/s00778-024-00845-0), proves that CorgiPile achieves comparable convergence rates to standard SGD while requiring less randomization overhead. Additionally, we have shown that CorgiPile maintains similar convergence behavior in both single-machine and distributed settings, making it a robust solution for large-scale training scenarios.


## **Implementation **
The main challenge is how to extend our single-process CorgiPile to work in the parallel/distributed environment of PyTorch, which usually use multiple processes with multiple GPUs to train models.  

<img src="RFC-00xx-asset/Distributed.png" style="zoom: 50%;" />

CorgiPile can be naturally extended to work in a multi-process mode, by enhancing the tuple-level shuffle under the data-parallel computation paradigm. We can naturally implement block-level shuffle by randomly distributing data blocks to different processes. For tuple-level shuffle, we can use multi-buffer based shuffling instead of single-buffer based shuffling — in each process we allocate a local buffer to read blocks and shuffle their tuples. PyTorch can then read the shuffled tuples when running SGD to perform the forward/backward/update computation as well as gradient/parameter communication /synchronization among different processes.

We implement this enhanced multi-process CorgiPile as a new **CorgiPileDataset** API in PyTorch:

```python
train_dataset = CorgiPileDataset(dataset_path, block_index, other_args)
train_loader = torch.utils.data.DataLoader(train_dataset, other_args)
train(train_loader, model, other_args)
```

**Similar** to usage of the original Dataset API, users only need to **initialize** the CorgiPileDataset with necessary parameters, including block_index, etc., and then use it as usual in the DataLoader API offered by PyTorch. The train() method constantly extracts a batch of tuples from DataLoader and then performs mini-batch SGD. Multi-process CorgiPile can achieve random data order similar to that of the single-process CorgiPile.

## **Metrics **

**End-to-end exeuction time** to of training process with different data shuffle strategies

**Convergence rates** to indicate convergence behavior of all strategies

## **Drawbacks**

The design of CorgiPile does not introduce additional drawbacks on PyTorch's existing framework and methods; it simply provides an efficient data shuffling method as an enhancement. 

Users only need to initialize the CorgiPileDataset with necessary parameters and then use it as usual in the DataLoader API offered by PyTorch.



