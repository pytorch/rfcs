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

# Dimension reduction example

Consider a 3-dimensional array
<img data-latex="$A$" src=".images/68abca2748084d89c9d6d9bb958e7960.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
with shape
<img data-latex="$(d_0=3, d_1=2, d_2=4)$" src=".images/2fb23541e494b9ecae2d8330477a1f5e.svg"  valign="-4.289px" width="166.48px" height="17.186px" style="display:inline;" alt="latex">:

<img data-latex="
\begin{tikzpicture}
%\def\xs{5} %shift in x direction
%\def\ys{2} %shift in y direction
% shape
\def\di{3}
\def\dj{2}  % y, max is 4
\def\dk{4}  % x
% scales
\def\sj{1.5}
\def\sk{1.5}
\tikzmath{\xs = \dk*\sk;}
\tikzmath{\ys = (\dj-0.5)*\sj;}
%stride
\tikzmath{int \shape0; \shape0 = \di;}
\tikzmath{int \shape1; \shape1 = \dj;}
\tikzmath{int \shape2; \shape2 = \dk;}
\tikzmath{int \stride0; \stride0 = \shape1*\shape2;}
\tikzmath{int \stride1; \stride1 = \shape2;}
\tikzmath{int \stride2; \stride2 = 1;}
\foreach \ri in {1,2,...,\di}
{
  \tikzmath{int \i; \i = \di - \ri + 1;}
  \tikzmath{\x0 = (\i-1)*\xs;}
  \tikzmath{\y0 = -(\i-1)*\ys;}
  \tikzmath{\x1 = \x0 + \sk*\dk;}
  \tikzmath{\y1 = \y0 - \sj*\dj;}
  \fill[white] (\x0-0.2*\sk, \y0) rectangle (\x1+0.2*\sk, \y1);
  \draw[thick] (\x0-0.2*\sk, \y0) rectangle (\x1+0.2*\sk, \y1);
  \foreach \j in {1,...,\dj}
  {
    \foreach \k in {1,...,\dk}
    {
      \tikzmath{\x = (\k-1+0.5)*\sk+\x0;}
      \tikzmath{\y = -(\j-1+0.5)*\sj+\y0;}
      \tikzmath{int \i0; \i0 = \i - 1;}
      \tikzmath{int \i1; \i1 = \j - 1;}
      \tikzmath{int \i2; \i2 = \k - 1;}
      \tikzmath{int \p; \p = \stride0*\i0 + \stride1*\i1 + \stride2*\i2;}
      \draw (\x, \y) node {\inlinemath{A_{\i0, \i1, \i2}^{(\p)}}};
    }
  }
}
\draw [dashed,gray](\sk*\dk+0.2*\sk, 0) -- (\sk*\dk+0.2*\sk+\di*\xs-\xs, - \di*\ys + \ys);
\draw [dashed,gray](\sk*\dk+0.2*\sk, -\sj*\dj) -- (\sk*\dk+0.2*\sk+\di*\xs-\xs, - \di*\ys + \ys - \sj*\dj);
\draw [dashed,gray](-0.2*\sk, -\sj*\dj) -- (-0.2*\sk+\di*\xs-\xs, - \di*\ys + \ys - \sj*\dj);
\end{tikzpicture}
" src=".images/9179ea86615298c8b81196f6d29a8a10.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">


**A dimension reduction** is an indexing operation that produces a new array with reduced number of dimensions but with the same set of array elements. 

A dimension reduction is specified by the **dimension selector function**
<img data-latex="$\kappa$" src=".images/2a6a442d0aebe6707c9c7cb2729342fa.svg"  width="14.001px" height="7.412px" style="display:inline;" alt="latex">
and the **partition** of a dimensions range 
 <img data-latex="$[0, N)$" src=".images/9a7315803e42828f64a3a1d57520793f.svg"  valign="-4.289px" width="45.435px" height="17.186px" style="display:inline;" alt="latex">.

For example, here is a list of all possible dimension selector functions for 
 <img data-latex="$N=3$" src=".images/fcdd39dfa46d3107c8205b2b09edc240.svg"  width="49.446px" height="11.764px" style="display:inline;" alt="latex">:

<img data-latex="
$$
\begin{aligned}
\kappa &= \{(0, 0), (1, 1), (2, 2)\}&&\text{--- identity}\\
\kappa &= \{(0, 0), (1, 2), (2, 1)\}&&\text{--- swap last dimensions}\\
\kappa &= \{(0, 1), (1, 0), (2, 2)\}&&\text{--- swap first dimensions}\\
\kappa &= \{(0, 2), (1, 1), (2, 0)\}&&\text{--- swap first and last dimensions}\\
\kappa &= \{(0, 1), (1, 2), (2, 0)\}&&\text{--- roll dimensions right}\\
\kappa &= \{(0, 2), (1, 0), (2, 1)\}&&\text{--- roll dimensions left}
\end{aligned}
$$
" src=".images/a3acd52808e2d4e68c4a557ed044db0f.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

Dimension selector represents dimensions (axes) permutations.

For example, here is a list of all partitions of a dimension range for 
 <img data-latex="$N=3$" src=".images/fcdd39dfa46d3107c8205b2b09edc240.svg"  width="49.446px" height="11.764px" style="display:inline;" alt="latex">:

<img data-latex="
$$\begin{aligned}\\
[0, 3) &= \{0\} \cup \{1, 2\} &&\text{--- reduction from 3 to 2}\\
[0, 3) &= \{0, 1\} \cup  \{2\} &&\text{--- reduction from 3 to 2}\\
[0, 3) &= \{0, 1, 2\} &&\text{--- reduction from 3 to 1}\\
[0, 3) &= \{0\} \cup \{1\} \cup \{2\} &&\text{--- trivial reduction from 3 to 3}
\end{aligned}
$$
" src=".images/49d1310fabc1e55716a477080c7e9b37.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

The partitions represents which dimensions will be collapsed to a
given reduced dimension.


Let us apply the first dimension reduction from 3 to 2 with identity dimension selector to the 3-dimensional array:

1. Compute the strides matrix

   <img data-latex="
$$
S =
\begin{bmatrix}
1 & 0 & 0\\
0 & d_2 & 1
\end{bmatrix} =
\begin{bmatrix}
1 & 0 & 0\\
0 & 4 & 1
\end{bmatrix}
$$
" src=".images/993954b520e7196bf5bd8770061778f8.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

   so that 
   <img data-latex="$ (j_0, j_1) = S (i_0, i_1, i_2) = (i_0, 4 i_1 + i_2) $" src=".images/7e6f86382ad14da13f3283436ffb097e.svg"  valign="-4.289px" width="253.564px" height="17.186px" style="display:inline;" alt="latex">.
   
2. Compute the shape of reduced array:

    <img data-latex="
$$
(d'_0, d'_1) = (d_0, d_1 d_2) = (3, 8)
    $$
" src=".images/7bc9fba59a49b15b8f5f273699989e71.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

3. Construct the reduced array:

    <img data-latex="
$$
A'[j_0, j_1] =
\begin{bmatrix}
A'_{0, 0} & A'_{0, 1} & A'_{0, 2} & A'_{0, 3} & A'_{0, 4} & A'_{0, 5} & A'_{0, 6} & A'_{0, 7}\\
A'_{1, 0} & A'_{1, 1} & A'_{1, 2} & A'_{1, 3} & A'_{1, 4} & A'_{1, 5} & A'_{1, 6} & A'_{1, 7}\\
A'_{2, 0} & A'_{2, 1} & A'_{2, 2} & A'_{2, 3} & A'_{2, 4} & A'_{2, 5} & A'_{2, 6} & A'_{2, 7}
\end{bmatrix}
$$
" src=".images/86a4736f37572f2b7345c7a4b2401010.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

    such that 
<img data-latex="$A'[j_0, j_1] = A[i_0, i_1, i_2]$" src=".images/0f38e5a4ac5ac566ef00fcb054226bb6.svg"  valign="-4.289px" width="158.113px" height="17.186px" style="display:inline;" alt="latex">:

    <img data-latex="
$$
    A' = 
\begin{bmatrix}
A_{0,0,0}^{(0)} & A_{0,0,1}^{(1)} & A_{0,0,2}^{(2)} & A_{0,0,3}^{(3)} &
A_{0,1,0}^{(4)} & A_{0,1,1}^{(5)} & A_{0,1,2}^{(6)} & A_{0,1,3}^{(7)}\\
A_{1,0,0}^{(8)} & A_{1,0,1}^{(9)} & A_{1,0,2}^{(10)} & A_{1,0,3}^{(11)} &
A_{1,1,0}^{(12)} & A_{1,1,1}^{(13)} & A_{1,1,2}^{(14)} & A_{1,1,3}^{(15)}\\
A_{2,0,0}^{(16)} & A_{2,0,1}^{(17)} & A_{2,0,2}^{(18)} & A_{2,0,3}^{(19)} &
A_{2,1,0}^{(20)} & A_{2,1,1}^{(21)} & A_{2,1,2}^{(22)} & A_{2,1,3}^{(23)}
\end{bmatrix}
    $$
" src=".images/016dab35e59afe6260e49004b9b4e609.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">



To perceive the patterns of different reductions better, let's
consider the 3-dimensional array with concrete values:

<img data-latex="
\begin{tikzpicture}
%\def\xs{5} %shift in x direction
%\def\ys{2} %shift in y direction
% shape
\def\di{3}
\def\dj{2}  % y, max is 4
\def\dk{4}  % x
% scales
\def\sj{1.5}
\def\sk{1.5}
\tikzmath{\xs = \dk*\sk;}
\tikzmath{\ys = (\dj-0.5)*\sj;}
%stride
\tikzmath{int \shape0; \shape0 = \di;}
\tikzmath{int \shape1; \shape1 = \dj;}
\tikzmath{int \shape2; \shape2 = \dk;}
\tikzmath{int \stride0; \stride0 = \shape1*\shape2;}
\tikzmath{int \stride1; \stride1 = \shape2;}
\tikzmath{int \stride2; \stride2 = 1;}
\foreach \ri in {1,2,...,\di}
{
  \tikzmath{int \i; \i = \di - \ri + 1;}
  \tikzmath{\x0 = (\i-1)*\xs;}
  \tikzmath{\y0 = -(\i-1)*\ys;}
  \tikzmath{\x1 = \x0 + \sk*\dk;}
  \tikzmath{\y1 = \y0 - \sj*\dj;}
  \fill[white] (\x0-0.2*\sk, \y0) rectangle (\x1+0.2*\sk, \y1);
  \draw[thick] (\x0-0.2*\sk, \y0) rectangle (\x1+0.2*\sk, \y1);
  \foreach \j in {1,...,\dj}
  {
    \foreach \k in {1,...,\dk}
    {
      \tikzmath{\x = (\k-1+0.5)*\sk+\x0;}
      \tikzmath{\y = -(\j-1+0.5)*\sj+\y0;}
      \tikzmath{int \i0; \i0 = \i - 1;}
      \tikzmath{int \i1; \i1 = \j - 1;}
      \tikzmath{int \i2; \i2 = \k - 1;}
      \tikzmath{int \p; \p = 1 + \stride0*\i0 + \stride1*\i1 + \stride2*\i2;}
      \draw (\x, \y) node {\inlinemath{\p}};
    }
  }
}
\draw [dashed,gray](\sk*\dk+0.2*\sk, 0) -- (\sk*\dk+0.2*\sk+\di*\xs-\xs, - \di*\ys + \ys);
\draw [dashed,gray](\sk*\dk+0.2*\sk, -\sj*\dj) -- (\sk*\dk+0.2*\sk+\di*\xs-\xs, - \di*\ys + \ys - \sj*\dj);
\draw [dashed,gray](-0.2*\sk, -\sj*\dj) -- (-0.2*\sk+\di*\xs-\xs, - \di*\ys + \ys - \sj*\dj);
\end{tikzpicture}
" src=".images/a443473cc52d07583a1289f78b6a1b64.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

Then there exists 24 combinations of dimension reductions, out of 18
are non-trivial that are shown below:

<img data-latex="
\begin{equation}
\label{eq:case-0-1}
\left\{
\begin{aligned}
\kappa&=\{(0, 0), (1, 1), (2, 2)\}\\
[0, 3)&=\{0\} \cup \{1, 2\}\\
A'& = \begin{bmatrix}
1 & 2 & 3 & 4 & 5 & 6 & 7 & 8\\
9 & 10 & 11 & 12 & 13 & 14 & 15 & 16\\
17 & 18 & 19 & 20 & 21 & 22 & 23 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/e3c752aa38ed837a951544de3b6a0a82.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-0-1" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-0-2}
\left\{
\begin{aligned}
\kappa&=\{(0, 0), (1, 1), (2, 2)\}\\
[0, 3)&=\{0, 1\} \cup \{2\}\\
A'& = \begin{bmatrix}
1 & 2 & 3 & 4\\
5 & 6 & 7 & 8\\
9 & 10 & 11 & 12\\
13 & 14 & 15 & 16\\
17 & 18 & 19 & 20\\
21 & 22 & 23 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/dbcd43ccb2e8f74eb89ce3dec2d6ca81.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-0-2" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-0-3}
\left\{
\begin{aligned}
\kappa&=\{(0, 0), (1, 1), (2, 2)\}\\
[0, 3)&=\{0, 1, 2\}\\
A'& = \begin{bmatrix}
1 & 2 & 3 & 4 & 5 & 6 & 7 & \ldots & 23 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/673b33e651591c91debed1146b179a33.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-0-3" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-1-1}
\left\{
\begin{aligned}
\kappa&=\{(0, 0), (1, 2), (2, 1)\}\\
[0, 3)&=\{0\} \cup \{1, 2\}\\
A'& = \begin{bmatrix}
1 & 5 & 2 & 6 & 3 & 7 & 4 & 8\\
9 & 13 & 10 & 14 & 11 & 15 & 12 & 16\\
17 & 21 & 18 & 22 & 19 & 23 & 20 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/ef8ba9838358f7b977f815045eb9f116.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-1-1" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-1-2}
\left\{
\begin{aligned}
\kappa&=\{(0, 0), (1, 2), (2, 1)\}\\
[0, 3)&=\{0, 1\} \cup \{2\}\\
A'& = \begin{bmatrix}
1 & 5\\
2 & 6\\
3 & 7\\
4 & 8\\
9 & 13\\
10 & 14\\
11 & 15\\
12 & 16\\
17 & 21\\
18 & 22\\
19 & 23\\
20 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/c3d857e133819430c7db6a412f035d03.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-1-2" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-1-3}
\left\{
\begin{aligned}
\kappa&=\{(0, 0), (1, 2), (2, 1)\}\\
[0, 3)&=\{0, 1, 2\}\\
A'& = \begin{bmatrix}
1 & 5 & 2 & 6 & 3 & 7 & 4 & \ldots & 20 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/1839072f9c50743c8ca9c7160b4e3e54.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-1-3" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-2-1}
\left\{
\begin{aligned}
\kappa&=\{(0, 1), (1, 0), (2, 2)\}\\
[0, 3)&=\{0\} \cup \{1, 2\}\\
A'& = \begin{bmatrix}
1 & 2 & 3 & 4 & 9 & 10 & 11 & \ldots & 19 & 20\\
5 & 6 & 7 & 8 & 13 & 14 & 15 & \ldots & 23 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/2d09cf8ad90a9fd730efd5e07874d314.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-2-1" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-2-2}
\left\{
\begin{aligned}
\kappa&=\{(0, 1), (1, 0), (2, 2)\}\\
[0, 3)&=\{0, 1\} \cup \{2\}\\
A'& = \begin{bmatrix}
1 & 2 & 3 & 4\\
9 & 10 & 11 & 12\\
17 & 18 & 19 & 20\\
5 & 6 & 7 & 8\\
13 & 14 & 15 & 16\\
21 & 22 & 23 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/ed10dcf7176de691472a0d31463f9df4.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-2-2" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-2-3}
\left\{
\begin{aligned}
\kappa&=\{(0, 1), (1, 0), (2, 2)\}\\
[0, 3)&=\{0, 1, 2\}\\
A'& = \begin{bmatrix}
1 & 2 & 3 & 4 & 9 & 10 & 11 & \ldots & 23 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/92340bc52fa01a3774300c5b32bd6593.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-2-3" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-3-1}
\left\{
\begin{aligned}
\kappa&=\{(0, 1), (1, 2), (2, 0)\}\\
[0, 3)&=\{0\} \cup \{1, 2\}\\
A'& = \begin{bmatrix}
1 & 9 & 17 & 2 & 10 & 18 & 3 & \ldots & 12 & 20\\
5 & 13 & 21 & 6 & 14 & 22 & 7 & \ldots & 16 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/662ff3c2e275737d80e82651522d7356.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-3-1" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-3-2}
\left\{
\begin{aligned}
\kappa&=\{(0, 1), (1, 2), (2, 0)\}\\
[0, 3)&=\{0, 1\} \cup \{2\}\\
A'& = \begin{bmatrix}
1 & 9 & 17\\
2 & 10 & 18\\
3 & 11 & 19\\
4 & 12 & 20\\
5 & 13 & 21\\
6 & 14 & 22\\
7 & 15 & 23\\
8 & 16 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/794d25db494292960e6b991ab9dd3504.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-3-2" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-3-3}
\left\{
\begin{aligned}
\kappa&=\{(0, 1), (1, 2), (2, 0)\}\\
[0, 3)&=\{0, 1, 2\}\\
A'& = \begin{bmatrix}
1 & 9 & 17 & 2 & 10 & 18 & 3 & \ldots & 16 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/ca8b98177a14fb1d8ba5a0232c7219c8.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-3-3" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-4-1}
\left\{
\begin{aligned}
\kappa&=\{(0, 2), (1, 0), (2, 1)\}\\
[0, 3)&=\{0\} \cup \{1, 2\}\\
A'& = \begin{bmatrix}
1 & 5 & 9 & 13 & 17 & 21\\
2 & 6 & 10 & 14 & 18 & 22\\
3 & 7 & 11 & 15 & 19 & 23\\
4 & 8 & 12 & 16 & 20 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/2161e3f8dde9a7afbe3110c54cd89d55.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-4-1" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-4-2}
\left\{
\begin{aligned}
\kappa&=\{(0, 2), (1, 0), (2, 1)\}\\
[0, 3)&=\{0, 1\} \cup \{2\}\\
A'& = \begin{bmatrix}
1 & 5\\
9 & 13\\
17 & 21\\
2 & 6\\
10 & 14\\
18 & 22\\
3 & 7\\
11 & 15\\
19 & 23\\
4 & 8\\
12 & 16\\
20 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/2aac99e687c0dc455e6ee2058fb98c77.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-4-2" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-4-3}
\left\{
\begin{aligned}
\kappa&=\{(0, 2), (1, 0), (2, 1)\}\\
[0, 3)&=\{0, 1, 2\}\\
A'& = \begin{bmatrix}
1 & 5 & 9 & 13 & 17 & 21 & 2 & \ldots & 20 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/c458e2e9d22673c502451d851b7a04fb.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-4-3" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-5-1}
\left\{
\begin{aligned}
\kappa&=\{(0, 2), (1, 1), (2, 0)\}\\
[0, 3)&=\{0\} \cup \{1, 2\}\\
A'& = \begin{bmatrix}
1 & 9 & 17 & 5 & 13 & 21\\
2 & 10 & 18 & 6 & 14 & 22\\
3 & 11 & 19 & 7 & 15 & 23\\
4 & 12 & 20 & 8 & 16 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/311e4368a3f24c4e95abfebd2d31c60a.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-5-1" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-5-2}
\left\{
\begin{aligned}
\kappa&=\{(0, 2), (1, 1), (2, 0)\}\\
[0, 3)&=\{0, 1\} \cup \{2\}\\
A'& = \begin{bmatrix}
1 & 9 & 17\\
5 & 13 & 21\\
2 & 10 & 18\\
6 & 14 & 22\\
3 & 11 & 19\\
7 & 15 & 23\\
4 & 12 & 20\\
8 & 16 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/07eacdc140219305ffb934de2e8840a8.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-5-2" alt="latex">

<img data-latex="
\begin{equation}
\label{eq:case-5-3}
\left\{
\begin{aligned}
\kappa&=\{(0, 2), (1, 1), (2, 0)\}\\
[0, 3)&=\{0, 1, 2\}\\
A'& = \begin{bmatrix}
1 & 9 & 17 & 5 & 13 & 21 & 2 & \ldots & 16 & 24
\end{bmatrix}
\end{aligned}
\right.
\end{equation}
" src=".images/57d2b99b971010ddcc2349d83061bab9.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" id="eq:case-5-3" alt="latex">





<!--EOF-->
