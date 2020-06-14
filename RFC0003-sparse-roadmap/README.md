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

# Roadmap for PyTorch Sparse Tensors


|            |                 |
| ---------- | --------------- |
| Authors    | Pearu Peterson  |
| Status     | Draft           |
| Type       | Process         |
| Created    | 2020-04-23      |
| Resolution | TBD             |

## Abstract

The aim of this document is to define the future of PyTorch sparse
tensors. We propose that
> :large_blue_circle:<!--:proposal:--> a sparse tensor is a drop-in
> replacement of dense tensors for all tensor operations.

## Motivation and Scope

PyTorch implements the *Tensor* object as the basic structure of
storing data in a multi-dimensional array structure that can be
processed on various computational devices such as GPUs as well as
multicore CPUs efficiently by taking advantage of parallelism that
these devices support. PyTorch has advanged in developing
computational tools for various ML and AI algorithms that can be
applied to so-called dense or strided tensors. However, the
developement of tools applicable to sparse tensors has not passed the
experimental stage after two or more years of development, dispite the
fact that a sparse tensor would be a natural data storage format for
many ML&AI applications where the data characterizes some pairwise
relations between study subjects, or when analyzing only a small set
of features that dominate in a big system, for instance. Choosing a
sparse tensor format to store the data may lead to more performant
programs due to the efficient uses of memory as well as computational
resources.

The starting point of this proposal and the far-reaching goal is that
the PyTorch tensor represents a multi-dimensional array whereas the
used data storage format is an implementation detail that users can
select depending on their data to (indirectly) control the choise of
computational tools that would be most efficient for users'
applications. Switching a storage format of data to another data
format (dense to sparse, sparse to dense, sparse to sparse, etc)
should not require changes to the data processing algorithms written
by the end-users or other high-level software tools. This approach of
supporting data storage format agnostic algorithms allows faster
adoption of newly developed algorithms and mathematical models as
software programs.

PyTorch already implements a dispatcher system for executing programs
on a particular device depending on the data storage location. The
same dispatcher system can be used for selecting computational tools
depending on the data storage format. So, the implementation of this
proposal will be about
1. providing sparse tensor implementations and the corresponding
low-level computational tools
2. reusing PyTorch dispatcher system to make sparse tensors as drop-in
replacements of dense tensors when used as inputs to high-level
computational tools.

A prioritization of features sets is required in order to make the
stable sparse tensors support accessible for users' programs as soon
as possible.

### About the used language and notation

In PyTorch as well as in other software/hardware project
(e.g. TensorFlow, Tensor cores in NVIDIA GPU devices, etc), a
multi-dimensional array is called a tensor. However, in mathematics, a
*tensor* is a more general algebra concept than an array (see
https://en.wikipedia.org/wiki/Tensor): while a tensor can be
represented as a multi-dimensional array, not every multi-dimensional
array is a representation of a tensor.  In the following, we use the
term *tensor* only within the context of PyTorch project, and the term
*array* within mathematical context that is applicable in a more
general settings.

To represent various mathematical concepts and relations in a concise
way, we often use Python language syntax in a free way as a
replacement of mathematical typesetting. So, the reader of this
document should be familiar with the Python syntax and semantics.

We use emoji :large_blue_circle:<!--:proposal:--> to mark a proposal
statement and :large_blue_diamond:<!--:impl:--> to mark an implementation option.

## Mathematical background of arrays

One of the most common data representation in scientific and
engineering computations is an array of elements (or items or entries)
that share the same data type and size and that can be generalized to
any dimensions. The array elements are indexed using tuples of
integers that represent the coordinates in a multidimensional discrete
space of data values.  Mathematically, arrays can be interpreted as
vectors, matrices, or tensors that form algebras with various algebra
operations as well as functions on algebra elements.

When representing data as an array in computer memory, a array storage
format must be selected. The simplest and most general array storage
format is the strided array format where the data values are stored
continuously in memory so that the storage location any array element
can be easily (<img data-latex="$O(1)$" src=".images/ef0cdc8ea95d866c1fb30b9f9724a739.svg"  valign="-4.289px" width="37.737px" height="17.186px" style="display:inline;" alt="latex">) determined from its index tuple and therefore
efficient concurrent algorithms can be designed that process data in
the most efficient way.  However, when data poses certain
characteristics that would lead to arrays where majority of array
elements have the same value, choosing a strided array format may lead
to inefficient use of memory and processor resources. For such cases
various sparse array formats have been proposed that provide
considerable storage savings as well as more efficient processing
possibilities but a the cost of increased complexity in the sparse
storage format.

To be specific, in this document we consider three array storage
formats: strided dense, COO sparse, and CSR sparse array storage
formats. However, implementing other storage formats are encouraged if
these have potential to increase the performance of existing array
operations.

To relate the different array storage formats, we define for each
format a transformation to a canonical array storage format for which
we use a contiguous array storage format. We denote this
transformation as `contiguous(A)` where `A` denotes an array using any
storage format.

We can now formulate the aim of this proposal as follows: let `op` be
any array operation on arrays `A0`, `A1`, `...`, that returns
arrays `B0`, `B1`, `...` as results:
```python
B0, B1, ... = op(A0, A1, ...)
```

:large_blue_circle:<!--:proposal:--> We aim at ensuring that the following equation holds:
```python
map(contiguous, op(*map(contiguous, (A0, A1, ...)))) == map(contiguous, (B0, B1, ...))
```

Note 1: when operation `op` returns arrays with the same storage
format as the input arrays use, the equation simplifies to
```python
op(*map(contiguous, (A0, A1, ...))) == map(contiguous, (B0, B1, ...))
```
that is, the operations `op` and `contiguous` are commutative in this
particular case.

Note 2: PyTorch uses COO array storage format where the values of
elements can be (dense) arrays. As a result, its shape consists of
sparse dimensions followed by dense dimensions. If one interprets such
hybrid array as an array of scalar elements then operations that
permute array dimensions may need to return a fully dense array of
scalar elements.  As an example, transposing a 2-D array with one
sparse dimension and one dense dimension must result in a 2-D dense
array because of the (implementation specific) constraint that sparse
dimensions must preceed dense dimensions in the PyTorch COO sparse
storage format.  For arrays with a large number of elements such
densifying operations may lead to inefficient use of memory resources
that users might want to prevent, for instance, by reorganizing their
algorithms. Different strategies are possible to acknowledge users
about such events: :large_blue_diamond:<!--:impl:--> report these as
warnings or raise exceptions when a certain threshold of memory usage
has been exceeded or are about to exceed. Introducing such threshold
would be advantageous for smaller arrays when such densifying
operations would not lead to memory usage issues, and therefore, would
be allowed.

### Strided array storage format

Let `A` be a `N`-dimensional array with dimensions `A.shape = (d0, d1,
...)` and `A[I]` denotes the value of the array element with index
`I=(i0, i1, ...)`.  The strided array storage format introduces the
concept of array strides `A.strides=(s0, s1, ...)` that is a `N`-tuple
of integers. For a contiguous row-major storage of array elements the
strides can be computed as follows:
```python
A.strides[N - 1] = 1
for i in range(N - 1):
    A.strides[N - i - 2] = A.strides[N - i - 1] * A.shape[N - i - 1]
```

To constuct an array with strided array storage format, we use
the following notation:
```python
A = Strided(strides, values, shape)
```
where `values` represents a C-contiguous array of elements stored on some
memory storage device and encaptures in itself also the data type
information of array elements.

The memory location of the strided array element `A[I]` is given as:
```
p[I] = b + sum(A.strides[i] * I[i] for i in range(N))
```
where `b` is a reference location (e.g. the starting point of the
allocated memory).


#### Contiguous arrays

The following results about contiguous arrays are required for
generalizing the CRS/CCS sparse storage format to multidimensional
array cases.

Contiguous arrays are strided arrays that values are stored linearly
in memory with no caps, that is, the span of memory for storing all
values of a contiguous array has a lenght equal to the number of all
array elements (multiplied by the storage size of a single element).

Given the dimensionality <img data-latex="$N$" src=".images/eac9fa71715f08df45c75d17adab2305.svg"  width="19.594px" height="11.764px" style="display:inline;" alt="latex"> of a
multidimensional array, there exists <img data-latex="$N!$" src=".images/7e71066fce4e8af212a9daa976c3af47.svg"  width="23.892px" height="11.955px" style="display:inline;" alt="latex"> ways to store
the array contiguously. For instance, there are so-called C-contiguous
(row-major storage order) and F-contiguous (column-major storage
order) storage methods for two-dimensional arrays.

Each way of contiguous storage corresponds to a certain permutation of
array dimensions and is uniquely determined by the strides <img data-latex="$(s_0, \dots, s_{N-1})$" src=".images/2f051d6d7e70b6aae15a57f29d035f28.svg"  valign="-4.289px" width="102.752px" height="17.186px" style="display:inline;" alt="latex"> and shape of an array <img data-latex="$(d_0, \ldots, d_{N-1})$" src=".images/465cc0d9fb21403ec057aaaefceb38a9.svg"  valign="-4.289px" width="104.39px" height="17.186px" style="display:inline;" alt="latex">. We
use C-contiguous storage as a reference storage method that
corresponds to the following strides values:

<img data-latex="
\begin{equation}
\label{eq:strides}
\begin{aligned}
s_{N-1} &= 1\\
s_{N-2} &= d_{N-1}\\
&\vdots\\
s_{i} &= d_{i+1} s_{i+1}\\
&\vdots\\
s_{0} &= d_1\cdots d_{N-1}
\end{aligned}
\end{equation}
" src=".images/e150f494a46f0feebf94f3663e90c6f1.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:strides" alt="latex">

which have notable property of being sorted (non-strickly) decreasing
order.

Given a contiguous array <img data-latex="$A$" src=".images/bb2243c9fce59491f0436432e1193900.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">, there exists a
permutation <img data-latex="$\sigma$" src=".images/c6c9705c4e8b8046a4d324da5bd0132e.svg"  width="14.496px" height="7.412px" style="display:inline;" alt="latex"> and a
C-contiguous array <img data-latex="$A'$" src=".images/b7b4c45be253388f29fe03e77b15ffba.svg"  width="20.222px" height="12.889px" style="display:inline;" alt="latex"> such that

<img data-latex="
$$
A[i_0,\ldots,i_{N-1}] = A'[i_{\sigma(0)},\ldots,i_{\sigma(N-1)}]
$$
" src=".images/97727f6e78d6df54d5ea56976a1eb47e.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

or

<img data-latex="
$$
A'[j_0,\ldots,j_{N-1}] = A[j_{\sigma^{-1}(0)},\ldots,j_{\sigma^{-1}(N-1)}]
$$
" src=".images/0f99f36aeda3fc17f4e6a1de47490945.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

for all <img data-latex="$i_k \in \{0,\ldots,d_{k}-1\} $" src=".images/7c3a3e4ca7666d6eaff92c6b2fc6d513.svg"  valign="-4.304px" width="145.629px" height="17.215px" style="display:inline;" alt="latex">, <img data-latex="$k\in\{0,\ldots,N-1\}$" src=".images/33fb17302867288586ad47f6adb82526.svg"  valign="-4.304px" width="141.785px" height="17.215px" style="display:inline;" alt="latex">.  The shape of <img data-latex="$A'$" src=".images/b7b4c45be253388f29fe03e77b15ffba.svg"  width="20.222px" height="12.889px" style="display:inline;" alt="latex"> is <img data-latex="$(d_{\sigma(0)},\ldots,d_{\sigma(N-1)})$" src=".images/509bb43f5e7c50532b787b0e05370640.svg"  valign="-6.025px" width="136.764px" height="18.922px" style="display:inline;" alt="latex">. We can represent the permutation <img data-latex="$\sigma$" src=".images/c6c9705c4e8b8046a4d324da5bd0132e.svg"  width="14.496px" height="7.412px" style="display:inline;" alt="latex"> as a 1-D array
of integers from <img data-latex="$\{0,\ldots,N-1\}$" src=".images/d37868c847a06905ef8d199ab089c5ca.svg"  valign="-4.304px" width="111.399px" height="17.215px" style="display:inline;" alt="latex"> so that <img data-latex="$\sigma[i] \equiv \sigma(i)$" src=".images/d1576f454bc01aa516c875898a3a7343.svg"  valign="-4.289px" width="79.939px" height="17.186px" style="display:inline;" alt="latex"> and <img data-latex="$\sigma[\sigma^{-1}[k]] \equiv k$" src=".images/6e13f39e56c17dad86747f56eb5774e7.svg"  valign="-4.289px" width="99.179px" height="18.241px" style="display:inline;" alt="latex">.

Given a strided array <img data-latex="$A$" src=".images/bb2243c9fce59491f0436432e1193900.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">, a method for determining if `A` is contiguous is as follows:
1. Let <img data-latex="$n = \prod_{k=0}^{N-1} d_k$" src=".images/2db2b709ed8bea88efd6ab2781c6e883.svg"  valign="-4.902px" width="96.965px" height="21.367px" style="display:inline;" alt="latex"> be the total number of <img data-latex="$A$" src=".images/bb2243c9fce59491f0436432e1193900.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex"> elements.

2. Sort the pairs of <img data-latex="$A$" src=".images/bb2243c9fce59491f0436432e1193900.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex"> strides and
   dimension numbers in descending order as follows:

   <img data-latex="
$$
\mathrm{sort}( ((s_k, k): 0\leqslant k < N ) ) \rightarrow ((s_{\sigma(k)}, \sigma(k)): 0\leqslant k<N)
$$
" src=".images/c8b20be30a7e8b06ae46a1b00c37a2c5.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">


   If <img data-latex="$s_{\sigma(N-1)}\not=1$" src=".images/6f1021918b4e8ba2ddb45b55e6f85d73.svg"  valign="-6.025px" width="84.549px" height="17.981px" style="display:inline;" alt="latex">, we conclude that <img data-latex="$A$" src=".images/bb2243c9fce59491f0436432e1193900.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex"> is not a contiguous array.

   Otherwise, the strides of <img data-latex="$A'$" src=".images/b7b4c45be253388f29fe03e77b15ffba.svg"  width="20.222px" height="12.889px" style="display:inline;" alt="latex"> is <img data-latex="$(s_{\sigma(0)}, \ldots, s_{\sigma(N-1)})$" src=".images/ec80f06a1007615d120a7291222852ae.svg"  valign="-6.025px" width="135.127px" height="18.922px" style="display:inline;" alt="latex">.

3. Compute the shape of <img data-latex="$A'$" src=".images/b7b4c45be253388f29fe03e77b15ffba.svg"  width="20.222px" height="12.889px" style="display:inline;" alt="latex">:

   <img data-latex="
\begin{align*}
d_{\sigma(N-1)} &= s_{\sigma(N-2)}\\
&\vdots\\
d_{\sigma(k)} &= s_{\sigma(k-1)} / s_{\sigma(k)}\\
&\vdots\\
d_{\sigma(1)} &= s_{\sigma(0)} / s_{\sigma(1)}\\
d_{\sigma(0)} &= n / s_{\sigma(0)}
\end{align*}
" src=".images/5a052599f821cd1bc3cf9cd355d2bb1f.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

   The array <img data-latex="$A$" src=".images/bb2243c9fce59491f0436432e1193900.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex"> is a contiguous array iff <img data-latex="$\prod_{k=0}^{N-1} d_{\sigma(k)}$" src=".images/60fa0426b90fd24596bab80830a48004.svg"  valign="-6.025px" width="81.134px" height="22.491px" style="display:inline;" alt="latex"> is equal to the total number of all <img data-latex="$A$" src=".images/bb2243c9fce59491f0436432e1193900.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex"> elements `n`.

#### Dimensionality reduction

Storing a multidimensional array linearly in memory is a special case
of dimensionality reduction: the dimensionality <img data-latex="$N$" src=".images/eac9fa71715f08df45c75d17adab2305.svg"  width="19.594px" height="11.764px" style="display:inline;" alt="latex"> is reduced to
<img data-latex="$1$" src=".images/39a02353ea65bee09e817d2af7362f2b.svg"  width="12.193px" height="11.097px" style="display:inline;" alt="latex"> by relating the
indices of a <img data-latex="$N$" src=".images/eac9fa71715f08df45c75d17adab2305.svg"  width="19.594px" height="11.764px" style="display:inline;" alt="latex">-dimensional
array to a memory address of element storage location. Let use denote the linear array representing the memory by <img data-latex="$M$" src=".images/6e6f00098ba14a79c1d939d351e3410c.svg"  width="22.404px" height="11.764px" style="display:inline;" alt="latex">, then we have

<img data-latex="
$$
M[p] = A[i_0, \ldots, i_{N-1}]
$$
" src=".images/9f8fc245e624125d849b9c30785f60fd.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

for all <img data-latex="$0\leqslant i_k<d_k$" src=".images/916bf571f1bad0d3e9969946b44f5a68.svg"  valign="-2.582px" width="85.808px" height="14.537px" style="display:inline;" alt="latex">, <img data-latex="$0\leqslant k<N$" src=".images/2f10e35b9dbe88b831e2c8ead5e3641d.svg"  valign="-2.353px" width="82.463px" height="14.308px" style="display:inline;" alt="latex"> where

<img data-latex="
$$
p = \sum_{k=0}^{N-1} s_{k} i_{k} = s_0 i_0 + \cdots + s_{N-1} i_{N-1}
$$
" src=".images/52089c8b1dcf53c561a72da06cbbd4e4.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

If <img data-latex="$A$" src=".images/bb2243c9fce59491f0436432e1193900.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex"> is a C-contiguous array then

<img data-latex="
$$
p = \sum_{k=0}^{N-1} s_{k} i_{k} = ((\ldots((i_0 d_1 +i_1) d_{2} + i_2) \ldots + i_{N-3})d_{N-2} + i_{N-2} ) d_{N-1}  + i_{N-1}
$$
" src=".images/aea8f6a805b8c0d9fdf1a5a526185b56.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

and there exists inverse relation:

<img data-latex="
$$
A[i_0, \ldots, i_{N-1}] = M[p]
$$
" src=".images/ce07d814494502c8cda92370e275807a.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

for all <img data-latex="$0\leqslant p<m$" src=".images/e4eb71b7d13561f4e0bebba981437106.svg"  valign="-3.347px" width="81.026px" height="14.445px" style="display:inline;" alt="latex"> where <img data-latex="$m=\prod_{k=0}^{N-1}d_k$" src=".images/95d1291fb3f3bea03b00996543787c32.svg"  valign="-4.902px" width="101.648px" height="21.367px" style="display:inline;" alt="latex"> and

<img data-latex="
\begin{align*}
i_{N-1} &= p \pmod{d_{N-1}}\\
i_{N-2} &= (p - i_{N-1}) / d_{N-1}\pmod{d_{N-2}}\\
i_{N-3} &= (p - i_{N-1} - i_{N-2}d_{N-1}) / d_{N-2}\pmod{d_{N-3}}\\
&\vdots\\
i_{k} &= \left.\left(p - \sum_{j=k+1}^{N-1} s_j i_j\right) \right/ d_{k+1}\pmod{d_k}\\
&\vdots\\
i_0 &= \ldots
\end{align*}
" src=".images/147916817ac0016ee054e3a00e0020c1.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

The above results can be generalized for contiguous arrays by
determining the permutation function <img data-latex="$\sigma$" src=".images/c6c9705c4e8b8046a4d324da5bd0132e.svg"  width="14.496px" height="7.412px" style="display:inline;" alt="latex"> of <img data-latex="$A$" src=".images/bb2243c9fce59491f0436432e1193900.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex"> and relating the associated C-contiguous <img data-latex="$A'$" src=".images/b7b4c45be253388f29fe03e77b15ffba.svg"  width="20.222px" height="12.889px" style="display:inline;" alt="latex"> to <img data-latex="$M$" src=".images/6e6f00098ba14a79c1d939d351e3410c.svg"  width="22.404px" height="11.764px" style="display:inline;" alt="latex">.

The juice of the above becomes from the fact that the dimensionality
<img data-latex="$N$" src=".images/eac9fa71715f08df45c75d17adab2305.svg"  width="19.594px" height="11.764px" style="display:inline;" alt="latex"> reduction to 1
has a straightforward extension to a reduction to any target
dimensionality. In particular, we are interested in target
dimensionality 2 because this is used as a basis for generalization of
CRS/CSS format to multidimensional sparse arrays. For that, we need to first
decide the distribution of the dimensions numbers to two
non-intersecting ordered sets of sizes <img data-latex="$l$" src=".images/55ecd5c7d0ac6174d3b30b9c94c24cb5.svg"  width="9.697px" height="11.955px" style="display:inline;" alt="latex"> and <img data-latex="$N-l$" src=".images/7dda62af50616bb6c4cda32f54646d2e.svg"  valign="-1.435px" width="46.035px" height="13.39px" style="display:inline;" alt="latex">, <img data-latex="$1 < l < N-2$" src=".images/a36fbc5aa1afc741273c333dcc7717b7.svg"  valign="-1.435px" width="107.175px" height="13.39px" style="display:inline;" alt="latex">:

<img data-latex="
$$
(d_0, d_1, \ldots, d_{N-1}) \Rightarrow (d_{\pi(0)}, \ldots, d_{\pi(l-1)}, d_{\rho(l)}, \ldots, d_{\rho(N-1)}) 
$$
" src=".images/9e61797379b2f3f3e3584b428d07f6f1.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

and then define

<img data-latex="
$$
M[p_0, p_1] = A[i_0, \ldots, i_{N-1}]
$$
" src=".images/058f41cd2a153676fe23e4453fcc2083.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

for all <img data-latex="$0\leqslant i_k<d_k$" src=".images/916bf571f1bad0d3e9969946b44f5a68.svg"  valign="-2.582px" width="85.808px" height="14.537px" style="display:inline;" alt="latex"> and <img data-latex="$0\leqslant k<N$" src=".images/2f10e35b9dbe88b831e2c8ead5e3641d.svg"  valign="-2.353px" width="82.463px" height="14.308px" style="display:inline;" alt="latex"> where

<img data-latex="
\begin{align*}
p_0 &= \sum_{k=0}^{l-1} s'_{k} i_{\pi^{-1}(k)},  & s'_{k}&= s'_{k+1} d_{\pi^{-1}(k)}, & s'_{l-1}&=1\\
p_1 &= \sum_{k=l}^{N-1} s''_{k} i_{\rho^{-1}(k)},& s''_{k}&= s''_{k+1} d_{\rho^{-1}(k)}, & s''_{l-1}&=1
\end{align*}
" src=".images/3da5685dace883c635442c0b94396090.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

Clearly, there is also an inverse relation if <img data-latex="$A$" src=".images/bb2243c9fce59491f0436432e1193900.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex"> is a contiguous array:

<img data-latex="
$$
A[i_0, \ldots, i_{N-1}] = M[p_0, p_1]
$$
" src=".images/a03b7a2f413da5a9c94248cf8cf5f4f9.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

for all <img data-latex="$0\leqslant p_0<m_0$" src=".images/b46d280e9a531b8f52f6d3c803335836.svg"  valign="-3.347px" width="93.231px" height="14.445px" style="display:inline;" alt="latex"> and <img data-latex="$0\leqslant p_1<m_1$" src=".images/e46e22e98bcee3d1d6feaad6fe042461.svg"  valign="-3.347px" width="93.231px" height="14.445px" style="display:inline;" alt="latex"> where <img data-latex="$m_0=\prod_{k=0}^{l-1}d_{\pi^{-1}(k)}$" src=".images/b480383a8b10e59e032d441f2b13b902.svg"  valign="-6.933px" width="133.074px" height="23.531px" style="display:inline;" alt="latex"> and <img data-latex="$m_1=\prod_{k=l}^{N-1}d_{\rho^{-1}(k)}$" src=".images/3dc7fc92022e67b71d00ccda89a695d2.svg"  valign="-6.933px" width="136.369px" height="23.398px" style="display:inline;" alt="latex">.

<!--
In terms of Mathematics of Arrays (MoA,
https://www.researchgate.net/publication/308893116_A_Mathematics_of_Arrays),
the memory location of an array element can be expressed in terms of a
integer-valued *&#120574; function* such that
```python
A[I] == ravel(A)[gamma(I, A.shape, ...)]
```
where `A.shape, ...` denotes any array storage format specific parameters
that are required for determining the storage location of the array element
in memory. So, we write
```
p[I] = gamma(I, A.shape, A.strides)
```
for strided arrays.

Note that one could define `ravel(A)` for any array storage format as
`ravel(A) = ravel(contiguous(A))` but that would be impractical for
defining the `gamma` function for arrays using sparse storage
formats. However, `ravel(A)` can be left undefined for such cases and
define `gamma` according to the actual implementation of the
particular storage format: the value of array element `A[I]` has a
storage address equal to
```
p[I] = gamma(I, A)
```
.

Note: Creating a regular slice of an array is a matter of computing
new strides for the slice while using the same reference point `b` as
the base array. For creating non-regular slices a copy of base array
memory may be required.

-->

### COO sparse array storage format (PyTorch)

#### Construction

COO sparse array storage format can be described via two arrays:
`indices` and `values`. The `indices` array is a 2-D array of integers
with dimensions `(Ns, NNZ)`, and the `values` array is a 2-D array of
values with dimensions `(NNZ, Nd)` such that `Ns + Nd = N`. Here the
subscripts `s` and `d` denote the sparse and the dense parts of the
COO sparse array storage (usually `d == 1` but PyTorch generalizes COO
sparse arrays to a sparse-dense hybrid of a COO sparse arrays).

To constuct an array with COO sparse array storage format, we use
the following notation:
```python
A = COO(indices, values, shape)
```
where `indices` and `values` are strided arrays satisfying the
constraints defined above.

PyTorch supports uncoalesced sparse arrays that means there may exists
columns of `indices` that are equal, that is, the equation
```python
A.indices[:, n] == Is
```
may have multiple solutions for `n` that correspond to the same array
element `A[Is + Id]`. So, in general, the value of an array element is
defined as
```python
A[Is + Id] = sum(A.values[n] for n in range(NNZ) if A.indices[:, n] == Is)
```

On the other hand, when no `n` exists such that `A.indices[:, n] ==
Is`, then the corresponding array element is classified as unspecified
element. The unspecified elements may have many
interpretations. Often, the unspecified elements are interpreted as
elements with zero values, but other interpretations exists as
well. For instance, for sparse softmax operation the unspecified
elements are interpreted as negative infinities. In the cases of using
sparse arrays to represent graphs, the unspecified elements are
interpreted as undefined.

<!--
The memory location of a COO sparse array element `A[Is + Id]` is
determined as follows: find `n` such that
```python
A.indices[:, n] == Is
```
holds. Then
```
p[Is + Id] = gamma(Id, A.values[n])
```


The equation `A.indices[:, n] == Is` may have multiple solutions.
This is the case for uncoalesced COO sparse arrays in PyTorch and it
means that the value of the array element `A[Is + Id]` is
`sum(A.values[n] for all n such that A.indices[:, n] == Is
holds)`. Using uncoalesced COO sparse arrays is computationally
efficient for additive operations such as as adding arrays
element-wise because it has an efficient implementation that involves
concatenating the indices and values of the two sparse array. On the
other hand, non-additive operations such as element-wise
multiplication of two sparse tensor, and many other operations,
require coalescing the operands before preforming the operation. In
PyTorch, coalesce operation also sorts `indices` using lexicographic
ordering (this allows fast conversations to CSR sparse array format).

When the equation `A.indices[:, n] == Is` does not have a solution then
the corresponding array element is classified as unspecified
element. The unspecified elements can have many interpretations. In
most cases these are considered as elements with zero values but other
interpetations exists as well. For instance, for sparse softmax
operation the unspecified elements are interpreted as negative
infinities. In the cases of using sparse arrays to represent graphs,
the interpretation of unspecified elements is none.
-->

#### Unspecified elements - fill-value

Many mathematical operations map zero values to non-zero values, for
instance, <img data-latex="$\cos 0=1$" src=".images/4f6d621c30d592e4da59090e5310bbc7.svg"  width="65.992px" height="11.097px" style="display:inline;" alt="latex">
etc. With the aim of defining such
nonhomogeneous operations on sparse arrays in a memory efficient way,
we propose that > :large_blue_circle:<!--:proposal:--> Element-wise
operations on coalesced sparse arrays result > sparse arrays with the
same set of indices as the input.

For nonhomogeneous operations on sparse arrays, we'll need to
introduce fill-value property that represents the value of unspecified
elements. When the fill-value is defined, it has to have the same
properties that array elements have.

##### Example 1

PyTorch COO sparse storage format supports array elements
with dense array values. With this interpretation, the fill-value is
formally a dense array.

##### Example 2

Let us consider a 2-D PyTorch COO sparse storage format
with one sparse and one dense dimensions:

<img data-latex="
$$
A = \begin{bmatrix}
11 & ? & 13 & ? & 15\\
21 & ? & 23 & ? & 25
\end{bmatrix}
$$
" src=".images/ea550b2ce1400a64d1a3c677bc359a13.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">


where `?` represents unspecified entries, the sparse dimension is
horizontal and dense dimension is vertical.

Let's apply softmax operation
(https://en.wikipedia.org/wiki/Softmax_function) along `A` sparse
dimension while interpreting the unspecified elements as zero valued
elements. We denote the softmax normalization factors along sparse
dimension as follows:

<img data-latex="
\begin{align*}
d_1 &= \exp(11) + \exp(0) + \exp(13) + \exp(0) + \exp(15)\\
d_2 &= \exp(21) + \exp(0) + \exp(23) + \exp(0) + \exp(25)
\end{align*}
" src=".images/7a1bcad2ed640da5f4bbc3f00ac9cf6d.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

so that

<img data-latex="
$$
\mathbf{softmax}(A) = 
\begin{bmatrix}
  \exp(11)/d_1 & 1/d_1  & \exp(13)/d_1 & 1/d_1 & \exp(15)/d_1\\
  \exp(21)/d_2 & 1/d_2 &  \exp(23)/d_2 & 1/d_2 & \exp(25)/d_2
\end{bmatrix}
$$
" src=".images/046f962195414fdd52dc1a351eebdd3c.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">


To represent this result as a sparse matrix with the same sparsity as
the input, the fill-value must be a 1-D array:

<img data-latex="
$$
\begin{bmatrix}
1/d_1\\
1/d_2
\end{bmatrix}
$$
" src=".images/f367b4731af9e0c15e4859dc22d12367.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">



##### Example 3

In graph theory, sparse arrays can be used to represent adjacency
relations between nodes while the array elements are certain
transmission coefficents between the nodes. The unspecified elements
indicate the lack of connections between the corresponding nodes.
When applying element-wise operations on such sparse arrays, the nature
of an unspecified element of being a lack of connection should not
change. Hence, the fill-value of such sparse arrays must be undefined.

In conclusion,

> :large_blue_circle:<!--:proposal:--> The fill-value of a PyTorch COO
> sparse storage format array, when defined, is either a scalar or a
> dense array with the shape of dense dimensions.

This holds independent of all possible interpretations of COO sparse
array elements.

##### :large_blue_circle:<!--:proposal:-->Fill-value API

The construction method of PyTorch COO sparse tensor is extended with
an optional fill-value argument:
```python
A = sparse_coo_tensor(indices, values, size=None, fill_value=None, dtype=None, device=None, requires_grad=False)
```

where `fill_value` can be `None` (default fill-value), or a single
scalar value, or any array-like object that can be converted to a
`Tensor` object that satisfies the requirements of a fill-value (see below).

There exists three kinds of fill-values:
1. Scalar fill-value is a `Tensor` with `shape == ()`. For example, `fill_value = 0`.
2. Undefined fill-value is a `Tensor` with `shape == (0, )`. For example, `fill_value = []`
3. Array fill-value is a `Tensor` with `shape == values.shape[1:]`.

The fill-value can be acquired via `_fill_value()` function.

The default fill-value is undefined.

For convenience, the specified fill-value data type and storage
location can be used to determine the unspecified data type and
storage location of the sparse tensor only if `values` is an empty
array-like object except `Tensor`.

The data type and storage location of a defined array fill-value must
match with the data type and storage location of `values`.
If this condition is not satisfied, a `TypeError` exception is raised.

Note 1: The proposed API is advantageous for performing element-wise
operations on sparse arrays:
```python
op(COO(indices, values, fill_value=<fill-value>)) == COO(indices, op(values), fill_value=op(<fill-value>))
```

Note 2: Implementation-wise, the fill-value is stored as a separate
attribute in an array instance and exposed as a get-only property
named `_fill_value` of an array instance.

#### Fill-value alternative is additive-value

In parallel to `A = COO(indices, values, shape, fill_value)` representation,
there exists an alternative `A' = COO'(indices, values, shape,
additive_value)` where the additive-value is the same as fill-value
for unspecified elements but for specified elements the array element
is a sum of the corresponding value and the additive-value. Compare:
```
A[I] = values[indices.index(I)] if I in indices else fill_value
A'[I] = additive_value + (values[indices.index(I)] if I in indices else 0)
```
That is, `A' = additive_value + COO(indices, values, shape, fill_value=0)`.

While `A'` is equivalent to `A` for arithmetic operations, it is not
suitable representation for sparse arrays applications in graph theory
and we shall skip discussing its possible advantages.

#### PyTorch implementation of COO sparse format

##### COO layout and constructor

The sparse layout that manifests as the `torch.layout` object is implemented
as a C++ function in the Layout.h file in c10. This file defines the enum
`kSparse` that represents the COO sparse matrix within the C++ source code.
The `torch.sparse_coo_tensor` method is the main method for construction
of a COO sparse matrix. Various versions of this method can be found
in [native\_functions.yaml_](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml#L3355), and each is defined in [SparseTensor.cpp](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/sparse/SparseTensor.cpp#L144). 

Unlike other ATen native functions, the `sparse_coo_tensor` implementation cannot be
only done by specifying the functions in `native\_functions.yaml` and then implementing
it in the CPP file. Edward Yang in a [slack communication]() states you need to create
manual bindings for this function, and the same will need to be done for a CSR format
constructor too. However, making changes to [gen\_python\_functions.py](https://github.com/pytorch/pytorch/blob/master/tools/autograd/gen_python_functions.py#L56) can potentially
fix this problem and allow autogeneration of both COO and CSR python bindings. The manual
python bindings for COO can be found in [python\_torch\_functions.cpp](https://github.com/pytorch/pytorch/blob/master/tools/autograd/templates/python_torch_functions.cpp#L427).

The general flow of the COO constructor is:
1. Check dimensions and data of supplied arguments.
2. Compute sparse and dense dimensions.
3. Actually construct the tensor object.

##### Autograd on COO tensors 

A very important consideration here is the implementation of autograd on COO tensors.
The autograd support for COO constructor is implemented via a separate function
called `\_sparse\_coo\_tensor\_with\_dims\_and\_tensors`. This is done because
abstract methods (those with a type-specific dispatch) lose autograd tracking on
the actual method that they dispatch to (`\_sparse\_coo\_tensor\_with\_dims\_and\_tensors`
in this case). A note in [native\_functions.yaml](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml#L3238) describes this in detail.

Therefore, when implementing autograd compatible routines for CSR tensors, it important
to build the constructors in such a way that they call end up calling a single method
with multiple dispatch that in turn has derivatives defined on it.

##### Method calls on COO sparse tensors

The internal implementation of COO tensors is done within the `SparseTensorImpl` class defined
in [SparseTensorImpl.cpp](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/SparseTensorImpl.cpp).
This is a subclass of `TensorImpl` that specialize it for COO sparse
by including some extra members and functions found in [SparseTensorImpl.h](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/SparseTensorImpl.h).

##### Sparse method dispatch

The `dispatch` parameter that is specified for functions written in `native\_functions.yaml` can also accept
a dispatch parameter of type `SparseCPU` or `SparseCUDA` that allows listing functions that are specifically
targeted at sparse data layout. Notable examples of such functions can be found in `torch.add()` that defines
different dispatch calls for different device and layout types. The impact on CSR sparse can be that we can
make additions to code generators in
[preprocess\_declarations.py](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/preprocess_declarations.py)
and add another `SparseCSRCPU` dispatch
backend and write specialized function for any method just by adding a new entry into `native\_functions.yaml`.

### CRS/CCS sparse array storage format

The [CRS/CCS](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)) sparse array storage format has a long history (since
mid-1960s) for representing sparse matrices because this format is memory
efficient and convinient for multiplying sparse matrices.  However,
the CRS/CCS format was designed for storing sparse two-dimensional
arrays. A number of generalizations of CRS/CCS format exists that
makes the choice of generalization for PyTorch not as straightforward
as one could wish.

Here we propose to use our modification of [Generalized Compressed
Row/Column Storage format
(GCRS/GCCS)](https://ieeexplore.ieee.org/document/7237032). The basic
idea behind GCRS/GCCS is as follows:
- a multidimensional array is represented as two-dimensional array
  using [dimensionality reduction](#dimensionality-reduction),
- the associated two-dimensional array is stored using the standard
  Compressed Row/Column Storage format (CRS/CCS).

The original version of GCRS/GCCS format uses a specific choice of
dimensionality reduction where the dimensions are split into two parts
using the evenness of dimensions numbers. This choice does not seem to
have obvious advantages over other possible choises from the view
point of data locality and performance. So, our modification will be
parameterized verion of GCRS/GCCS where the choice of splitting the
set of dimensions to two sets can be user's choice. Also, this
approach allows one implementation for both CRS and CCS format cases.

A particularly important property of the GCRS/GCCS format is that in
the two-dimensional case it will coincide with the standard CRS/CCS
format so that existing high-performance libraries implementing the
CRS/CSS format support can be used within PyTorch at low cost.

#### Example

The GCRS paper explains an idea of representing a N-dimensional sparse
array as a two-dimensional array that is stored in the standard
CRS/CSS sparse array storage format. The mapping of N-dimensional
indexes to two-dimensional indices is choosen so that all the even/odd
dimensions of N-D indices (dimensions are numbered starting from 0)
contribute the row/column index of the 2-D indices. For example, a
five-dimensional index <img data-latex="$(i_0, i_1, i_2, i_3, i_4)$" src=".images/c99132adb8a9296e6dda9a9bc1012cef.svg"  valign="-4.289px" width="107.206px" height="17.186px" style="display:inline;" alt="latex"> of an array <img data-latex="$A$" src=".images/bb2243c9fce59491f0436432e1193900.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex"> with the shape <img data-latex="$(d_0, d_1, d_2, d_3, d_4)$" src=".images/e7f47b0dbd3844274016bfb2d8212945.svg"  valign="-4.289px" width="122.248px" height="17.186px" style="display:inline;" alt="latex"> is mapped to two-dimensional
index <img data-latex="$(a, b)$" src=".images/b1eae3c210005046a31e1777e9e3c513.svg"  valign="-4.289px" width="40.058px" height="17.186px" style="display:inline;" alt="latex"> of a sparse array <img data-latex="$A'$" src=".images/b7b4c45be253388f29fe03e77b15ffba.svg"  width="20.222px" height="12.889px" style="display:inline;" alt="latex"> as follows:

<img data-latex="
\begin{align*}
a_0 &= i_0 d_4 d_2 + i_2 d_4 + i_4\\
a_1 &= i_1 d_3 + i_3
\end{align*}
" src=".images/0bf4c59d8fec08c14625acd0c6be1710.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

and we have

<img data-latex="
$$
A[i_0, i_1, i_2, i_3, i_4] = A'[a_0, a_1]
$$
" src=".images/cc03110ccc3bff6e30efb8d391078c2a.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

where the two-dimensional array <img data-latex="$A'$" src=".images/b7b4c45be253388f29fe03e77b15ffba.svg"  width="20.222px" height="12.889px" style="display:inline;" alt="latex"> has a shape <img data-latex="$(\delta_0, \delta_1)$" src=".images/2ddba22fae20976bf7ea71b0273dff8e.svg"  valign="-4.289px" width="51.676px" height="17.186px" style="display:inline;" alt="latex"> where

<img data-latex="
\begin{align*}
\delta_0 &= d_0 d_2 d_4\\
\delta_1 &= d_1 d_3
\end{align*}
" src=".images/88c0a42e30c3bcfb3e6e9dc3a1a7116e.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

and the array <img data-latex="$A'$" src=".images/b7b4c45be253388f29fe03e77b15ffba.svg"  width="20.222px" height="12.889px" style="display:inline;" alt="latex"> will be stored using CRS format.
To reconstruct <img data-latex="$A$" src=".images/bb2243c9fce59491f0436432e1193900.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex"> from <img data-latex="$A'$" src=".images/b7b4c45be253388f29fe03e77b15ffba.svg"  width="20.222px" height="12.889px" style="display:inline;" alt="latex">, the following inverse mapping is used:

<img data-latex="
\begin{align*}
i_4 &= a_0 \% d_4\\
i_3 &= a_1 \% d_3\\
i_2 &= (a_0 - i_4) / d_4\\
i_1 &= (a_1 - i_3) / d_3\\
i_0 &= (a_0 - i_2d_4 - i_4)/(d_4d_2)
\end{align*}
" src=".images/12c7c4760c984fe4a9b75b8d0cf39f27.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

## Operations with PyTorch tensors

TBD: what is implemented and what is not? Comparison with SciPy,
PyData, etc sparse array projects.

### Construction

### Indexing operations

### Arithmetic operations

Let `A` and `B` be two sparse tensors and apply a element-wise
binary operation `op` (addition, multiplication, substraction, etc):
```
C = A op B
```
Clearly, `A.shape == B.shape` must hold.

### Tensor algebra operations

### Functions on tensors

#### Utility functions

* `torch.numel(input) → int`
  Returns the total number of elements in the input tensor.

  The `numel` function is often used as a predicate for non-empty
  tensor test.  The number of elements in the input tensor is
  `product(input.shape)` and its function is implementation detail
  agnostic.


## Implementation plan

TBD: prioritize

TBD: existing matmut proposal for CSR format

TBD: high-priority fill value/offset support

## Appendix: User stories

- A sparse matrix fits to GPU memory but dense matrix doesn't. https://github.com/cupy/cupy/issues/2360

- Algorithms on sparse tensors outperform the ones on dense tensors when the density of data is sufficiently low. https://github.com/pytorch/pytorch/pull/36305#issuecomment-627624391

- I'd love to be able to pass a sparse tensor in to torch.distributions... https://github.com/pytorch/rfcs/pull/4#issuecomment-619459572

- My hunch is that most users want sparse tensors to be a drop-in replacement as much as possible... https://github.com/pytorch/rfcs/pull/4#issuecomment-619459572

- Where do I need sparse tensor?... https://github.com/pytorch/pytorch/issues/10043

- The state of sparse tensors  https://github.com/pytorch/pytorch/issues/9674

<!--EOF-->
