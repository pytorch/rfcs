<!--watch-latex-md

This document is processed by watch_latex_md.py program, see

  https://github.com/Quansight/pearu-sandbox/latex_in_markdown/

You can edit this document as you wish. You can also edit the LaTeX
data in img elements, but only the content of `latex-data`:

  1. To automatically update the LaTeX rendering in img element, edit
     the file while watch_latex_md.py is running.

  2. Never change the beginning (`<img latex-data="...`) and the end
     (`...alt="latex">`) parts of the LaTeX img elements as these are
     used by the watch_latex_md.py script.

  3. Changes to other parts of the LaTeX img elements will be
     overwritten.

Enjoy LaTeXing!

watch-latex-md:no-force-rerender
-->

# Literature

Since sparse tensor usage in deep learning is a new topic, in this document
we intend to collect various publications related to sparse tensor/matrix
computational techniques and how they can possibly be used in PyTorch.

<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-generate-toc again -->
**Table of Contents**

- [Literature](#literature)
    - [Papers](#papers)
        - [Ginkgo: A Modern Linear Operator Algebra Framework for HighPerformance Computing](#ginkgo-a-modern-linear-operator-algebra-framework-for-highperformance-computing)
        - [Preparing Ginkgo for AMD GPUs –A Testimonial on Porting CUDA Code to HIP](#preparing-ginkgo-for-amd-gpus-a-testimonial-on-porting-cuda-code-to-hip)
        - [Adaptive SpMV/SpMSpV on GPUs forInput Vectors of Varied Sparsity](#adaptive-spmvspmspv-on-gpus-forinput-vectors-of-varied-sparsity)
        - [Sparse GPU Kernels for Deep Learning](#sparse-gpu-kernels-for-deep-learning)
        - [High-Performance Sparse Matrix-Matrix Products on Intel KNL and Multicore Architectures](#high-performance-sparse-matrix-matrix-products-on-intel-knl-and-multicore-architectures)
        - [Batched Sparse Matrix Multiplication for Accelerating Graph Convolutional Networks](#batched-sparse-matrix-multiplication-for-accelerating-graph-convolutional-networks)

<!-- markdown-toc end -->


## Papers

### Ginkgo: A Modern Linear Operator Algebra Framework for HighPerformance Computing

Link: https://arxiv.org/pdf/2006.16852.pdf

This paper describes the architecture of Ginkgo, a library implemented fast algorithms
for sparse linear algebra on GPUs. They show how C++ abstractions using runtime polymorphism
allow choosing between generic and hardware-specific algorithms that can be specially implemented
for various backends such as CUDA, HIP and openMP (see Fig. 1 in the paper for details).

Currently Ginkgo provides a list of Krylov solvers (BICG, BiCGSTAB, CG, CGS, FCG, GMRES) for
for iterative solutions of sparse matrices, along with various preconditioners.
The library has been designed from the ground up keeping in mind its usage within other frameworks
and libraries. Therefore it has a component based interface that allows combining various components
of say a solver and building one from the ground up (example in listing 3).

Ginkgo is licensed under the BSD-3 clause license.

Results reported between Gingko's SpMV and that from cuSPARSE and hipSPARSE show that Gingko
is faster in some cases. The tests are conducted on the matrices from SuiteSparse. It can be
seen that Gingko's algorithms cover most of the problems for a given slow down factor compared to
the fastest implementation for the given problem. The reporting of results does not convey the
actual speed of the kernels compared to HIP or CUSP but we do get an idea that Gingko is in 
general faster for most of the matrices in the problem set.

![Image of Gingko perf vs. CUSP and HIP.](images/gingko-perf-comp.png) 


### Preparing Ginkgo for AMD GPUs –A Testimonial on Porting CUDA Code to HIP

Link: https://arxiv.org/pdf/2006.14290.pdf

This paper is about how the linear algebra routines in Ginkgo were ported to use HIP by
leveraging the extensible software architecture of Ginkgo as reported earlier. The author
states that Ginkgo is now the only library that supports various matrix formats (), linear
solvers and preconditiners on both CUDA and HIP platforms for sparse computation.

Further, they show that CUDA spMV is faster for most cases over HIP, as can be seen in the
figure below.

![HIP vs CUDA perf comparison.](images/cuda-hip-comp-gingko.png) 

### Adaptive SpMV/SpMSpV on GPUs forInput Vectors of Varied Sparsity

Link: https://arxiv.org/pdf/2006.16767.pdf

This paper explores the SpMSpV algorithm, where the matrix and vector being worked on are both sparse.
They report speeding up the computation by use of the CSR format for storage of the sparse vector and 
using a machine-learning based approach for selecting appropriate comupute kernels depending on the
sparsity of the vector. Comparisons are done on K40, P100 and V100 GPUs.

![Image of spmspv product.](images/spvspm-product.png)

### Sparse GPU Kernels for Deep Learning

Link: https://arxiv.org/pdf/2006.10901.pdf

This paper is a survey about various applications of sparse matrices in deep learning appliciations.

### High-Performance Sparse Matrix-Matrix Products on Intel KNL and Multicore Architectures

Link: https://dl.acm.org/doi/abs/10.1145/3229710.3229720

### Batched Sparse Matrix Multiplication for Accelerating Graph Convolutional Networks

Link: https://arxiv.org/abs/1903.11409

TODO:

* https://ieeexplore.ieee.org/document/7092787
* https://ieeexplore.ieee.org/abstract/document/7965199?casa_token=jFN4uanr7p4AAAAA:jD4CKEczgvpg6bZ09NHFRSrQivmw8CzmP7ak2owWzr1nSqnGh1MIJYzl1OtMj-LOXVBljRDwIwzf
