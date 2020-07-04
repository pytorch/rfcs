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

|            |                 |
| ---------- | --------------- |
| Authors    | Pearu Peterson  |
| Created    | 2020-06-30      |

# Array Fundamentals

## Introduction

The aim of this document is to describe relations between different
array storage formats and develop a theory on how these relations
change on array indexing operations such as slicing, indexing, or
swapping axes.

## Definitions

An array is a structured collection of its elements. The essential
part of the array definition is that each element in this collection
is labeled with a fixed size tuple of integers, called indices, that
can be used to uniquely identify the array elements. Here we assume
that each integer value in the tuple of indices is bounded to a
specified range starting from 0 and having fixed length. The
collection of such ranges define the shape of an array.  The number of
indices is the dimensionality of an array.

We use the following notation: let 
<img data-latex="$A$" src=".images/2dc20c494a7c17abcb2fafd76a498fe7.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
be an 
<img data-latex="$N$" src=".images/4f96c072fefbe775ee976ac3d45be396.svg"  width="19.594px" height="11.764px" style="display:inline;" alt="latex">-dimensional
array then its elements are denoted by 
<img data-latex="$A_{{\boldsymbol i}}$" src=".images/949edb128df3a3244565ec0eb1741b14.svg"  valign="-2.582px" width="21.774px" height="14.346px" style="display:inline;" alt="latex">
where
<img data-latex="${\boldsymbol i}=(i_0,\ldots,i_{N-1})$" src=".images/26bd0c34ce9c468ecceb9baae6e01999.svg"  valign="-4.289px" width="127.298px" height="17.186px" style="display:inline;" alt="latex">
is the indices tuple such that
<img data-latex="$0\leqslant i_n<d_n$" src=".images/d17a22ade9851787aff63bae8f5ebbf8.svg"  valign="-2.582px" width="86.805px" height="14.537px" style="display:inline;" alt="latex">
and 
<img data-latex="${\boldsymbol d}=(d_0,\ldots,d_{N-1})$" src=".images/99d5b8906819fcc46e40511fda5eb6d8.svg"  valign="-4.289px" width="136.843px" height="17.186px" style="display:inline;" alt="latex">
is the shape of the array. We also write 
<img data-latex="$A[i_0,\dots,i_{N-1}]$" src=".images/fa4c37e4c076fc7dc4ad7b465fe77b27.svg"  valign="-4.289px" width="107.412px" height="17.186px" style="display:inline;" alt="latex">
for 
<img data-latex="$A_{{\boldsymbol i}}$" src=".images/949edb128df3a3244565ec0eb1741b14.svg"  valign="-2.582px" width="21.774px" height="14.346px" style="display:inline;" alt="latex">.


Indexing of an array is an indexing operation that selects one array
element with given index. It is often convinent to consider the
element of an array as a 0-dimensional array that shape is empty
tuple.

Slicing of an array is an indexing operation that selects certain
array elements to form a new array:

<img data-latex="
$$
A'[i'_0,\ldots,i'_{N'-1}] = A[\iota_0({\boldsymbol i}'), \ldots, \iota_{N-1}({\boldsymbol i}')]
$$
" src=".images/c6c2e88a9e2e78c0bf58c85a7a83d4d3.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

where 
<img data-latex="$\iota_n$" src=".images/2e769fb76c9d072a7dfe6de65d179b0f.svg"  valign="-2.582px" width="17.243px" height="9.995px" style="display:inline;" alt="latex">, 
<img data-latex="$0\leqslant n<N$" src=".images/a75ee54dc8ac56f630e1771bb11e69df.svg"  valign="-2.353px" width="83.18px" height="14.117px" style="display:inline;" alt="latex">,
are integer-valued indexing functions and the equality holds for all indices 
<img data-latex="${\boldsymbol i}'$" src=".images/9079272b4553fc7f9ec848de16a7e7d5.svg"  width="14.554px" height="13.96px" style="display:inline;" alt="latex">.

## Indexing operations

In the following we assume that the indexing functions are linear. Then

<img data-latex="
$$
{\boldsymbol i} = {\boldsymbol b} + D {\boldsymbol i}'
$$
" src=".images/3d7ed5ee3745b97d0cceeeafa9dbc658.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

where 
<img data-latex="$D$" src=".images/d32f3c1ba4986897eb32619849d3261d.svg"  width="18.762px" height="11.764px" style="display:inline;" alt="latex">
is 
<img data-latex="$N\times N'$" src=".images/904a8ffe9c1ee4570073efb3c3ba0a25.svg"  valign="-1.435px" width="59.22px" height="14.324px" style="display:inline;" alt="latex">
matrix, 
<img data-latex="${\boldsymbol b}$" src=".images/fa6e7e6b1b44935d92b745c79adafc5d.svg"  width="13.264px" height="11.955px" style="display:inline;" alt="latex">
is a column vector of offsets, and 
<img data-latex="$0\leqslant i'_k<d'_k$" src=".images/a9dbedd047b01f90614210a5a45e26bd.svg"  valign="-4.809px" width="85.808px" height="17.698px" style="display:inline;" alt="latex">, 
<img data-latex="$0\leqslant k < N'$" src=".images/1b48ec418010e78f9fa8b1a1f295511d.svg"  valign="-2.353px" width="85.751px" height="15.242px" style="display:inline;" alt="latex">.

Clearly, the same form applies for other indexing operation such as element selection or swapping axes.

Python programming language defines a slicing operation that is
parametrized by a triple of integers 
<img data-latex="$(b, e, \Delta)$" src=".images/e739a355bca33962bc3a16e466a277ab.svg"  valign="-4.289px" width="59.865px" height="17.186px" style="display:inline;" alt="latex">
where 
<img data-latex="$b$" src=".images/c2e19a6043f094a095aca911bf7da6f7.svg"  width="11.465px" height="11.955px" style="display:inline;" alt="latex">
is the starting index, 
<img data-latex="$e$" src=".images/a78cc572d52088fe4e4bac0935d925ed.svg"  width="12.11px" height="7.412px" style="display:inline;" alt="latex">
is stopping index (non-inclusive), and 
<img data-latex="$\Delta$" src=".images/66895ad4db5488ece2ac619fd8fe5506.svg"  width="17.589px" height="11.761px" style="display:inline;" alt="latex">
is the step of index increment. The Python slice operation along the
<img data-latex="$n$" src=".images/19ec9347027c17cab874ac9e2e406e12.svg"  width="14.36px" height="7.412px" style="display:inline;" alt="latex">-th
dimension is defined as follows:

<img data-latex="
$$
A'_{b_n:e_n:\Delta_n}[i'_0, \ldots, i'_n, \ldots, i'_{N-1}] = A[i'_0, \ldots, \tilde b_n+i'_n\Delta_n, \ldots, i'_{N-1}]
$$
" src=".images/4d7a56a9adb51f6146ed2bd4c257d0ba.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

where 
<img data-latex="$\tilde b_n=b_n \mod d_n$" src=".images/c71c2c93df589b7ebf23bcec7b60eecd.svg"  valign="-2.582px" width="118.953px" height="18.419px" style="display:inline;" alt="latex">, 
<img data-latex="$b_n$" src=".images/e13f9b45c5d3a6187c446c431387cd25.svg"  valign="-2.582px" width="18.452px" height="14.537px" style="display:inline;" alt="latex">
and 
<img data-latex="$e_n$" src=".images/e1dcc2f9829f0cd263d900007a885e3d.svg"  valign="-2.582px" width="19.098px" height="9.995px" style="display:inline;" alt="latex">
are reduced with respect to 
<img data-latex="$d_n$" src=".images/78dac938b4b6d6348f1aaf58e89aba59.svg"  valign="-2.582px" width="20.044px" height="14.537px" style="display:inline;" alt="latex">,
and the equality holds for all 
<img data-latex="$0\leqslant i'_n<d'_n$" src=".images/3c368612c8bc0f17c2268dd1bf723642.svg"  valign="-4.256px" width="86.805px" height="17.145px" style="display:inline;" alt="latex">,

<img data-latex="
$$
d'_n=\left\lfloor(e_n-b_n+\Delta_n - \mathrm{sign} \Delta_n)/\Delta_n\right\rfloor.
$$
" src=".images/18980cee209afb4f89bfadfa824e0d42.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

For a general Python slicing operation we have

<img data-latex="
$$
{\boldsymbol i} =
\begin{pmatrix}
b_0\\
b_1\\
\vdots\\
b_{N-1}
\end{pmatrix} +
\begin{bmatrix}
\Delta_0 & 0 & \ldots & 0\\
0 & \Delta_1 & \ldots & 0\\
\vdots&\vdots&\ddots&\vdots\\
0 & 0 &\ldots & \Delta_{N-1}
\end{bmatrix}_{N\times N}
{\boldsymbol i}'
$$
" src=".images/6ead325cb3ba04267321e057f0eab969.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

where
<img data-latex="$0\leqslant b_n < d_n$" src=".images/649a5fbd97f65c41902b200caa6d4544.svg"  valign="-2.582px" width="88.221px" height="14.537px" style="display:inline;" alt="latex">,
<img data-latex="$0\leqslant b_n+\Delta_n i'_n < d_n$" src=".images/094f7ce6d9762f5e2ba1bce2b5d19160.svg"  valign="-4.256px" width="142.278px" height="17.145px" style="display:inline;" alt="latex">,
<img data-latex="$0\leqslant i'_n < d'_n$" src=".images/0b8ad05a815466608644eefd1be096a4.svg"  valign="-4.256px" width="86.805px" height="17.145px" style="display:inline;" alt="latex">.

Selecting an array element with index 
<img data-latex="$(i_0,\ldots, i_{N-1})$" src=".images/a62f1022bed705f70e58f63ea03d5607.svg"  valign="-4.289px" width="98.373px" height="17.186px" style="display:inline;" alt="latex">
is a special case of Python slicing operation: 
<img data-latex="$b_n = i_n$" src=".images/098ddcb1301c64d54676b0795c9feb09.svg"  valign="-2.582px" width="53.645px" height="14.537px" style="display:inline;" alt="latex">, 
<img data-latex="$\Delta_n=0$" src=".images/d611150636e7717361cc276878972194.svg"  valign="-2.582px" width="54.927px" height="14.344px" style="display:inline;" alt="latex">,
and we define 
<img data-latex="$d'_n\equiv 0$" src=".images/ccfd1550dbc916afd76228f2685a2700.svg"  valign="-4.256px" width="51.392px" height="17.145px" style="display:inline;" alt="latex">, 
<img data-latex="$0\leqslant n<N$" src=".images/a75ee54dc8ac56f630e1771bb11e69df.svg"  valign="-2.353px" width="83.18px" height="14.117px" style="display:inline;" alt="latex">.

For selecting a subarray with fixed 
<img data-latex="$n$" src=".images/19ec9347027c17cab874ac9e2e406e12.svg"  width="14.36px" height="7.412px" style="display:inline;" alt="latex">-th
index we have

<img data-latex="
$$
{\boldsymbol i} =
\begin{pmatrix}
0\\
\vdots\\
0\\
i_n\\
0\\
\vdots\\
0
\end{pmatrix} +
\begin{bmatrix}
1 & \ldots & 0 & 0 & \ldots & 0\\
\vdots&&\vdots&\vdots&&\vdots\\
0 & \ldots & 1 & 0 & \ldots & 0\\
0 & \ldots & 0 & 0 & \ldots & 0\\
0 & \ldots & 0 & 1 & \ldots & 0\\
\vdots&&\vdots&\vdots&\ddots&\vdots\\
0 & \ldots & 0 & 0 & \ldots & 1\\
\end{bmatrix}_{N\times N-1}
{\boldsymbol i}'
$$
" src=".images/1d4757cde182396315c3e7f96731e2fe.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

and 
<img data-latex="$d'_k=d_k$" src=".images/25a2c836da3fb7eda66c222eff7a2cc1.svg"  valign="-4.809px" width="57.249px" height="17.698px" style="display:inline;" alt="latex">
if 
<img data-latex="$k<n$" src=".images/91957f12667a3a693d813ab7586b903c.svg"  valign="-0.459px" width="46.38px" height="12.414px" style="display:inline;" alt="latex">
and 
<img data-latex="$d'_k=d_{k+1}$" src=".images/ec77779dccf603028fb1f54ba30631ac.svg"  valign="-4.809px" width="72.207px" height="17.698px" style="display:inline;" alt="latex">
if 
<img data-latex="$n\leqslant k<N-1$" src=".images/f2a49a38ba9728de991c6324a86fa221.svg"  valign="-2.353px" width="113.566px" height="14.308px" style="display:inline;" alt="latex">.

For swapping the 
<img data-latex="$m$" src=".images/13dd3181e0c19f1ae2f261721137586a.svg"  width="19.042px" height="7.412px" style="display:inline;" alt="latex">-th
and 
<img data-latex="$n$" src=".images/19ec9347027c17cab874ac9e2e406e12.svg"  width="14.36px" height="7.412px" style="display:inline;" alt="latex">-th
dimension of an array we have

<img data-latex="
$$
{\boldsymbol i} =
\begin{pmatrix}
0\\
\vdots\\
0\\
\vdots\\
0\\
\vdots\\
0
\end{pmatrix} +
\begin{bmatrix}
1 & \ldots & 0 & \ldots & 0 & \ldots & 0\\
\vdots&\ddots&\vdots&&\vdots& &\vdots\\
0 & \ldots & 0 & \ldots &1 & \ldots & 0\\
\vdots& &\vdots&\ddots&\vdots& &\vdots\\
0 & \ldots & 1 & \ldots &0 & \ldots & 0\\
\vdots& &\vdots&&\vdots&\ddots&\vdots\\
0 & \ldots & 0 & \ldots &0 & \ldots & 1
\end{bmatrix}_{N\times N}
{\boldsymbol i}'
$$
" src=".images/2361c36ea4b26559be6f9a0897424805.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

and 
<img data-latex="$d'_k=d_k$" src=".images/25a2c836da3fb7eda66c222eff7a2cc1.svg"  valign="-4.809px" width="57.249px" height="17.698px" style="display:inline;" alt="latex">
if 
<img data-latex="$k\not\in\{m,n\}$" src=".images/08717a5f1383f414daabcbfe74a3556a.svg"  valign="-4.304px" width="84.257px" height="17.215px" style="display:inline;" alt="latex">, 
<img data-latex="$d'_m=d_n$" src=".images/5f35bc89ef5aada980e1dd14698ee3ea.svg"  valign="-4.256px" width="61.497px" height="17.145px" style="display:inline;" alt="latex">, 
<img data-latex="$d'_n=d_m$" src=".images/68b16e50818e1659c52b5207c0686399.svg"  valign="-4.256px" width="61.497px" height="17.145px" style="display:inline;" alt="latex">.

Indexing operations can be composed: if 
<img data-latex="$A'[{\boldsymbol i}'] = A[{\boldsymbol \iota}({\boldsymbol i}')]$" src=".images/a62fc7093ebdc537d4ac96dfca02a63a.svg"  valign="-4.289px" width="113.299px" height="18.25px" style="display:inline;" alt="latex">
and 
<img data-latex="$A''[{\boldsymbol i}''] = A'[{\boldsymbol \iota}'({\boldsymbol i}'')]$" src=".images/92c478d11b93e40ed941d0e2d9e382bc.svg"  valign="-4.289px" width="130.734px" height="18.25px" style="display:inline;" alt="latex">
then 
<img data-latex="$A''[{\boldsymbol i}''] = A[{\boldsymbol \iota}({\boldsymbol i}'')]$" src=".images/0bfe78296cf6c2ef01005f80cff2ef34.svg"  valign="-4.289px" width="123.162px" height="18.25px" style="display:inline;" alt="latex">
where

<img data-latex="
$$
{\boldsymbol i} = {\boldsymbol b} + D ({\boldsymbol b}' + D' {\boldsymbol i}'') = ({\boldsymbol b} + D {\boldsymbol b}') +  D' {\boldsymbol i}''
$$
" src=".images/4602e7300da7b4469befae179e447c8b.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

but one must be careful in computing the new dimensions values 
<img data-latex="$d''_k$" src=".images/8cd99c068ad13b0a550b308657e37fff.svg"  valign="-4.809px" width="19.632px" height="17.698px" style="display:inline;" alt="latex">, 
<img data-latex="$0\leqslant k<N''$" src=".images/5594bac7f425994f3999b3dfec89137b.svg"  valign="-2.353px" width="89.038px" height="15.242px" style="display:inline;" alt="latex">.

## Dimension reduction and promotion of arrays

Dimension reduction of an array is an indexing operation that produces
a new array with reduced number of dimensions but with the same set of
array elements. The inverse operation of dimension reduction is
dimension promotion.

A dimension reduction of an array is about coding subsets of its
indexes to a smaller set of indices. Let 
<img data-latex="$\kappa$" src=".images/cc2d81529ddc5c64b9b65ed97eebc0c4.svg"  width="14.001px" height="7.412px" style="display:inline;" alt="latex">
be a dimension selector: 
<img data-latex="$0\leqslant\kappa(j)<N$" src=".images/8f34ee7ce2fe7eba466aab2bc34d25cd.svg"  valign="-4.289px" width="102.957px" height="17.186px" style="display:inline;" alt="latex">, 
<img data-latex="$0\leqslant j<N$" src=".images/9a82ca08ed38543a40e1fb94665fb6cc.svg"  valign="-3.347px" width="81.061px" height="15.111px" style="display:inline;" alt="latex">,
and
<img data-latex="$\kappa(j)=\kappa(j') \Leftrightarrow j=j'$" src=".images/81e2e3c02deeeb218a7f333c5270a358.svg"  valign="-4.289px" width="157.626px" height="17.186px" style="display:inline;" alt="latex">.
Let us partition the range 
<img data-latex="$[0, N)$" src=".images/f933fa82b9744a24c75dacef7f881d21.svg"  valign="-4.289px" width="45.435px" height="17.186px" style="display:inline;" alt="latex">
into 
<img data-latex="$M$" src=".images/835fb1cc3fd10587ed6be53ee082396d.svg"  width="22.404px" height="11.764px" style="display:inline;" alt="latex">-subranges 
<img data-latex="$[N_0, N_1), \ldots, [N_{M-1}, N_{M})$" src=".images/54bf3140159fa4937cf607f107b3bea2.svg"  valign="-4.289px" width="185.938px" height="17.186px" style="display:inline;" alt="latex">
and define associated strides for each subrange: 
<img data-latex="$s_{j,N_{j}-1}=1, s_{j, k} = s_{j, k+1} d_{\kappa(k+1)}, N_{j}\leqslant k < N_{j+1}$" src=".images/7972b29c26ff6a8d6d7e29a1077acf18.svg"  valign="-6.972px" width="333.512px" height="18.927px" style="display:inline;" alt="latex">, 
<img data-latex="$s_{j,k'}=0$" src=".images/5548429bb46fe10edd85ba30949851a3.svg"  valign="-5.383px" width="60.432px" height="16.481px" style="display:inline;" alt="latex">
for 
<img data-latex="$k'<N_{j}$" src=".images/3897b338a8d20ff9345e9d33e486aa18.svg"  valign="-4.907px" width="59.117px" height="17.796px" style="display:inline;" alt="latex">
and 
<img data-latex="$k'\geqslant N_{j+1}$" src=".images/4d83eebe44efced2d8a7e60217be2a11.svg"  valign="-4.907px" width="74.354px" height="17.796px" style="display:inline;" alt="latex">, 
<img data-latex="$0\leqslant j<M$" src=".images/14b8390c4c079ccefb05ccbcfaeaabe7.svg"  valign="-3.347px" width="83.87px" height="15.111px" style="display:inline;" alt="latex">.

Then the indices of dimension reduction are defined as follows:

<img data-latex="
$$
i'_j = \sum_{k=0}^{N-1} s_{j, k} i_{\kappa(k)}
$$
" src=".images/1c24a9f345a36e64a51724e48e983a9a.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

such that 
<img data-latex="$0\leqslant i'_j<d'_j\equiv\prod_{k=N_j}^{N_{j+1}-1} d_{\kappa(k)}$" src=".images/1204fb339a2af4f162acbff1309a9c81.svg"  valign="-9.517px" width="200.457px" height="28.006px" style="display:inline;" alt="latex">
for all 
<img data-latex="$0\leqslant i_{\kappa(k)} < d_{\kappa(k)}$" src=".images/90ba2d7071635a085e83683970f06a75.svg"  valign="-6.025px" width="117.495px" height="17.981px" style="display:inline;" alt="latex">
and the dimension reduction of an array 
<img data-latex="$A$" src=".images/2dc20c494a7c17abcb2fafd76a498fe7.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
from 
<img data-latex="$N$" src=".images/4f96c072fefbe775ee976ac3d45be396.svg"  width="19.594px" height="11.764px" style="display:inline;" alt="latex">
to 
<img data-latex="$M$" src=".images/835fb1cc3fd10587ed6be53ee082396d.svg"  width="22.404px" height="11.764px" style="display:inline;" alt="latex">
with dimension selector 
<img data-latex="$\kappa$" src=".images/cc2d81529ddc5c64b9b65ed97eebc0c4.svg"  width="14.001px" height="7.412px" style="display:inline;" alt="latex">
reads

<img data-latex="
$$
A'[{\boldsymbol i}'] = A[{\boldsymbol i}],
$$
" src=".images/57fcf1d1e936852052a03e0b5f64f778.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

<img data-latex="${\boldsymbol i}'=S {\boldsymbol i}$" src=".images/78065b5c3d162e48ed8e17ea99c1dd2e.svg"  width="55.347px" height="13.96px" style="display:inline;" alt="latex">
where 
<img data-latex="$S$" src=".images/afb859031c2a86cafd54381d8a706e73.svg"  width="15.667px" height="11.764px" style="display:inline;" alt="latex">
is 
<img data-latex="$M\times N$" src=".images/ab55e4fa03bec5add5b9b33c7c10f18e.svg"  valign="-1.435px" width="58.741px" height="13.198px" style="display:inline;" alt="latex">
matrix of strides. 


There exists an inverse to this relation of indices that can be resolved using the following algorithm leading to the dimension promotion of an array:

<img data-latex="
$$
\begin{aligned}
i_{\kappa(N_{j+1}-1)} &= i'_j \mod d_{\kappa(N_{j+1}-1)}\\
&\vdots\\
i_{\kappa(k)} &= \left.\left(i'_j - \sum_{k'=k+1}^{N_{j+1}-1} s_{j, k'} i_{\kappa(k')} \right)\right/d_{\kappa(k+1)} \mod d_{\kappa(k)}\\
&\vdots\\
i_{\kappa(N_{j})} &= \ldots
\end{aligned}
$$
" src=".images/2dbbd0dd871e69f29915c6e038084675.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

Note that this algorithm assumes that the stride values are
ordered: 
<img data-latex="$s_{j,k} \leqslant s_{j,k+1}$" src=".images/e5db6b392ed36b59ab1f83c09fcfa577.svg"  valign="-4.907px" width="87.8px" height="15.867px" style="display:inline;" alt="latex">
for 
<img data-latex="$N_{j}\leqslant k < N_{j+1}$" src=".images/cdc44213a9af8d1bc55043de28545b99.svg"  valign="-4.907px" width="112.755px" height="16.862px" style="display:inline;" alt="latex">.
If these are not, the indices must be permuted to achieve the correct ordering before applying the algorithm.


### Example: 3 -> 1

Consider a 3-dimensional array with a shape 
<img data-latex="$(d_0, d_1, d_2)$" src=".images/591d22eb500d23adef29f03d970d91a8.svg"  valign="-4.289px" width="76.925px" height="17.186px" style="display:inline;" alt="latex">
and define a dimension selector 
<img data-latex="$\kappa(i)=i, 0\leqslant i<3$" src=".images/2e1e45d1b0ce91a3db99797bbf8c24fa.svg"  valign="-4.289px" width="134.373px" height="17.186px" style="display:inline;" alt="latex">.
For reducing the array dimensions to one, we have

<img data-latex="
$$
i' =
\begin{bmatrix}
d_2d_1 & d_1 & 1
\end{bmatrix}
\begin{pmatrix}
i_2\\
i_1\\
i_0
\end{pmatrix}
$$
" src=".images/5334471a5f37e2891e6ce7fdbbf8fce8.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

that is the classical storage model of a C-contiguous strided array.

### Example: 3 -> 2

Consider a 3-dimensional array with a shape 
<img data-latex="$(d_0, d_1, d_2)$" src=".images/591d22eb500d23adef29f03d970d91a8.svg"  valign="-4.289px" width="76.925px" height="17.186px" style="display:inline;" alt="latex">.
Define a dimension selector
<img data-latex="$\kappa=\{(0, 2), (1, 1), (2, 0)\}$" src=".images/8232916decccb0a646a72db3539ea365.svg"  valign="-4.304px" width="174.882px" height="17.215px" style="display:inline;" alt="latex">,
partition
<img data-latex="$[0, 3)$" src=".images/01ce47cbfac7b77f87cb8e318a826532.svg"  valign="-4.289px" width="38.034px" height="17.186px" style="display:inline;" alt="latex">
into two subranges
<img data-latex="$[0, 2)$" src=".images/f2303cd01be155d3c6a4a1b02ede8919.svg"  valign="-4.289px" width="38.034px" height="17.186px" style="display:inline;" alt="latex">
and
<img data-latex="$[2, 3)$" src=".images/94f35555515e2238fedec1a80e48310a.svg"  valign="-4.289px" width="38.034px" height="17.186px" style="display:inline;" alt="latex">,
hence
<img data-latex="$M=2$" src=".images/7d3f4fbbb1429d4051ed1c91f0c3cf4b.svg"  width="52.255px" height="11.764px" style="display:inline;" alt="latex">.
We have
<img data-latex="$s_{0,1}=1$" src=".images/aaf1c4c8551314c3fa4106e904a1fcdd.svg"  valign="-4.907px" width="57.545px" height="16.004px" style="display:inline;" alt="latex">, 
<img data-latex="$s_{0,0}=s_{0,1}d_{\kappa(1)}=d_{1}$" src=".images/0cc8f2dba974e5453e88e9fc825ef363.svg"  valign="-6.025px" width="140.568px" height="17.981px" style="display:inline;" alt="latex">, 
<img data-latex="$s_{0,2}=0$" src=".images/6e238e9a4307aca9813df4255b589eb1.svg"  valign="-4.907px" width="57.545px" height="16.004px" style="display:inline;" alt="latex">
and
<img data-latex="$s_{1,2}=1$" src=".images/4ad85d83d4b5ac2aba6745b51cd8b491.svg"  valign="-4.907px" width="57.545px" height="16.004px" style="display:inline;" alt="latex">, 
<img data-latex="$s_{1,0} = s_{1,1} = 0$" src=".images/24e84448126879f14a572b93085fab53.svg"  valign="-4.907px" width="102.898px" height="16.004px" style="display:inline;" alt="latex">
so that

<img data-latex="
$$
\begin{pmatrix}
i'_0\\
i'_1
\end{pmatrix} =
\begin{bmatrix}
d_1 & 1 & 0\\
0 & 0 & 1
\end{bmatrix}
\begin{pmatrix}
i_2\\
i_1\\
i_0
\end{pmatrix}
$$
" src=".images/e9bfc2d7167d576c996036043b5fac3a.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

<img data-latex="$0\leqslant i'_0 < d'_0=d_2 d_1$" src=".images/b0d80ae2f00f9de748fa9f8bfd7d3866.svg"  valign="-4.256px" width="136.712px" height="17.145px" style="display:inline;" alt="latex">, 
<img data-latex="$0\leqslant i'_0 < d'_0 = d_0$" src=".images/7c7250099cb3e4bc1e22392a4d972612.svg"  valign="-4.256px" width="121.602px" height="17.145px" style="display:inline;" alt="latex">.
Notice that the dimension reduction can be always reversed: 
<img data-latex="$i_0=i'_1$" src=".images/76e49dc96acf59caec418604f7255eff.svg"  valign="-4.256px" width="49.959px" height="17.145px" style="display:inline;" alt="latex">, 
<img data-latex="$i_1= i'_0 \mod d_1$" src=".images/33607284dd5e3f841c3ce635dff9c0d5.svg"  valign="-4.256px" width="112.716px" height="17.145px" style="display:inline;" alt="latex">, 
<img data-latex="$i_2=(i'_0 - i_1)/d_1 \mod d_2$" src=".images/a8ab5ed0cf49ef312e957d5a0ec06967.svg"  valign="-4.304px" width="181.59px" height="17.215px" style="display:inline;" alt="latex">.

## The problem of slicing array when given in dimension reduced form

Let array 
<img data-latex="$A$" src=".images/2dc20c494a7c17abcb2fafd76a498fe7.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
be given by its reduction 
<img data-latex="$A_r[S{\boldsymbol i}] = A[{\boldsymbol i}]$" src=".images/8854481e987ec699c1da5efea884095b.svg"  valign="-4.289px" width="100.123px" height="17.186px" style="display:inline;" alt="latex">, 
<img data-latex="$\forall {\boldsymbol i}$" src=".images/a9640d1a5b77e4f85b124a17dd13d410.svg"  width="20.831px" height="11.955px" style="display:inline;" alt="latex">.
Find the reduction of
<img data-latex="$A$" src=".images/2dc20c494a7c17abcb2fafd76a498fe7.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
slice 
<img data-latex="$A'[{\boldsymbol i}'] = A[{\boldsymbol b}+D{\boldsymbol i}']$" src=".images/38898c424957ea54d3f08f93334749a5.svg"  valign="-4.289px" width="137.487px" height="18.25px" style="display:inline;" alt="latex">
such that 
<img data-latex="$A'_r[S'{\boldsymbol i}'] = A'[{\boldsymbol i}']$" src=".images/48befc748b1f8bcd1c28ea0b2c2ba545.svg"  valign="-4.289px" width="115.266px" height="18.25px" style="display:inline;" alt="latex">, 
<img data-latex="$\forall {\boldsymbol i}'$" src=".images/37e39a63fdbc2ae4527adf55f7a7d8b2.svg"  width="24.118px" height="13.96px" style="display:inline;" alt="latex">.

Recall that 
<img data-latex="$A$" src=".images/2dc20c494a7c17abcb2fafd76a498fe7.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
dimension reduction is defined by dimension selector
<img data-latex="$\kappa$" src=".images/cc2d81529ddc5c64b9b65ed97eebc0c4.svg"  width="14.001px" height="7.412px" style="display:inline;" alt="latex">
and a partition of 
<img data-latex="$[0, N)$" src=".images/f933fa82b9744a24c75dacef7f881d21.svg"  valign="-4.289px" width="45.435px" height="17.186px" style="display:inline;" alt="latex">
that with the given shape
<img data-latex="${\boldsymbol d}$" src=".images/383f74a17eabe7c3ff99056e9b4cca69.svg"  width="14.794px" height="11.955px" style="display:inline;" alt="latex">
of 
<img data-latex="$A$" src=".images/2dc20c494a7c17abcb2fafd76a498fe7.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
leads to the strides matrix 
<img data-latex="$S$" src=".images/afb859031c2a86cafd54381d8a706e73.svg"  width="15.667px" height="11.764px" style="display:inline;" alt="latex">.
From
<img data-latex="${\boldsymbol b}$" src=".images/fa6e7e6b1b44935d92b745c79adafc5d.svg"  width="13.264px" height="11.955px" style="display:inline;" alt="latex">, 
<img data-latex="$D$" src=".images/d32f3c1ba4986897eb32619849d3261d.svg"  width="18.762px" height="11.764px" style="display:inline;" alt="latex">,
and 
<img data-latex="${\boldsymbol d}$" src=".images/383f74a17eabe7c3ff99056e9b4cca69.svg"  width="14.794px" height="11.955px" style="display:inline;" alt="latex">
we can compute the shape of 
<img data-latex="$A'$" src=".images/a3da0dd3f72451c051d0833272fade16.svg"  width="20.222px" height="12.889px" style="display:inline;" alt="latex">: 
<img data-latex="${\boldsymbol d}'$" src=".images/d6692407a67fc3227c8baf4bd574a1f3.svg"  width="18.082px" height="13.981px" style="display:inline;" alt="latex">.
We need to choose the dimension selector
<img data-latex="$\kappa'$" src=".images/d8b6e8e925e1413b6f8e489a4f8052b8.svg"  width="17.289px" height="12.889px" style="display:inline;" alt="latex">
and partition of 
<img data-latex="$[0, N')$" src=".images/7bd581345b65a3ef29dd45dc7128054f.svg"  valign="-4.289px" width="49.221px" height="17.186px" style="display:inline;" alt="latex">.
Then we can compute the strides matrix 
<img data-latex="$S'$" src=".images/861da079a6c24ee0c9b6444475614228.svg"  width="18.955px" height="12.889px" style="display:inline;" alt="latex">.
Now create a loop over all
<img data-latex="${\boldsymbol i}'$" src=".images/9079272b4553fc7f9ec848de16a7e7d5.svg"  width="14.554px" height="13.96px" style="display:inline;" alt="latex">
and initialize the dimension reduction of array slice as follows:

<img data-latex="
$$
A'_r[S'{\boldsymbol i}'] = A_r[S{\boldsymbol b} + SD{\boldsymbol i}'].
$$
" src=".images/2ed9027581285f45cdc4e8810885e0c6.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

For a sparse 
<img data-latex="$A$" src=".images/2dc20c494a7c17abcb2fafd76a498fe7.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">,
only a subset of 
<img data-latex="${\boldsymbol i}'$" src=".images/9079272b4553fc7f9ec848de16a7e7d5.svg"  width="14.554px" height="13.96px" style="display:inline;" alt="latex">
indices, denote it by 
<img data-latex="$I'$" src=".images/c7225c26fa915a518562c2605f792e48.svg"  width="16.373px" height="12.889px" style="display:inline;" alt="latex">,
needs to be iterated. To determine this subset, let us assume that the specified elements of sparse array 
<img data-latex="$A$" src=".images/2dc20c494a7c17abcb2fafd76a498fe7.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
are 
<img data-latex="$A[{\boldsymbol i}]$" src=".images/72510a91d725734b6262711aeedf53a4.svg"  valign="-4.289px" width="32.498px" height="17.186px" style="display:inline;" alt="latex">
where 
<img data-latex="${\boldsymbol i}\in I\subset \text{``all array indices''}$" src=".images/9646c430437e90db9d2e973146d010af.svg"  valign="-3.348px" width="188.967px" height="15.304px" style="display:inline;" alt="latex">.
So, the indices set 
<img data-latex="$I'$" src=".images/c7225c26fa915a518562c2605f792e48.svg"  width="16.373px" height="12.889px" style="display:inline;" alt="latex">
is defined by the following equation:

<img data-latex="
$$
I' = \left\{{\boldsymbol i}' \left| S{\boldsymbol b}+SD{\boldsymbol i}' = S{\boldsymbol i}, {\boldsymbol i}\in I\right.\right\}
$$
" src=".images/ccd4e7d999c2833db0359fc93b2687fb.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

or when the specified elements are defined by 
<img data-latex="$A$" src=".images/2dc20c494a7c17abcb2fafd76a498fe7.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
 and 
<img data-latex="$A'$" src=".images/a3da0dd3f72451c051d0833272fade16.svg"  width="20.222px" height="12.889px" style="display:inline;" alt="latex">
 reductions, that is, 
<img data-latex="${\boldsymbol j}\in J$" src=".images/0b1e5448fedda6fa7b6b99b69d842e87.svg"  valign="-3.347px" width="45.426px" height="15.282px" style="display:inline;" alt="latex">
 such that 
<img data-latex="${\boldsymbol j} =S{\boldsymbol i}$" src=".images/12b988ecf1bf3edc984c2c9eb9044146.svg"  valign="-3.347px" width="53.769px" height="15.282px" style="display:inline;" alt="latex">, 
<img data-latex="${\boldsymbol i}\in I$" src=".images/56ce0d5a5aa6071fc1c041fb5f2d2ebf.svg"  valign="-0.673px" width="41.096px" height="12.608px" style="display:inline;" alt="latex">:

<img data-latex="
$$
J' = \left\{{\boldsymbol j}' \left| 
{\boldsymbol j}'=S'{\boldsymbol i}',
S{\boldsymbol b}+SD{\boldsymbol i}' = {\boldsymbol j}, 
{\boldsymbol j}\in J\right.\right\}
$$
" src=".images/8967efadd163d7d743370f5a6a6b52ab.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

Solving the system 
<img data-latex="$S{\boldsymbol b}+SD{\boldsymbol i}' = {\boldsymbol j}$" src=".images/c3e59318ff2e1647a918ca58ee3f1c77.svg"  valign="-3.347px" width="112.397px" height="17.308px" style="display:inline;" alt="latex">
for 
<img data-latex="${\boldsymbol i}'$" src=".images/9079272b4553fc7f9ec848de16a7e7d5.svg"  width="14.554px" height="13.96px" style="display:inline;" alt="latex">
constitutes the main problem of slicing sparse array in dimension reduced form.

### Example: slicing of a sparse array

Let 
<img data-latex="$A$" src=".images/2dc20c494a7c17abcb2fafd76a498fe7.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
be a
<img data-latex="$N$" src=".images/4f96c072fefbe775ee976ac3d45be396.svg"  width="19.594px" height="11.764px" style="display:inline;" alt="latex">-dimensional
sparse array with an indices set
<img data-latex="$I$" src=".images/231893c1f3b46a6887755f86f9524376.svg"  width="13.086px" height="11.764px" style="display:inline;" alt="latex">
of specified array elements. Let's assume a trivial reduction of 
<img data-latex="$A$" src=".images/2dc20c494a7c17abcb2fafd76a498fe7.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
where 
<img data-latex="$M=N$" src=".images/1574a81254a7f79fd7a63b313e0f9546.svg"  width="59.657px" height="11.764px" style="display:inline;" alt="latex">
and 
<img data-latex="$\kappa(i)=i$" src=".images/88de2a1655ce0158b8a920c20dc85a8a.svg"  valign="-4.289px" width="59.652px" height="17.186px" style="display:inline;" alt="latex">,
that is, 
<img data-latex="$S$" src=".images/afb859031c2a86cafd54381d8a706e73.svg"  width="15.667px" height="11.764px" style="display:inline;" alt="latex">
is identity matrix and
<img data-latex="$J = I$" src=".images/e2b079a2017c9d4d2b040ce0e8738643.svg"  width="45.953px" height="11.764px" style="display:inline;" alt="latex">.
For the slice
<img data-latex="$({\boldsymbol b, D})$" src=".images/122b5a1bddbdc1ec67cbb8db71fbcddc.svg"  valign="-4.289px" width="47.473px" height="17.186px" style="display:inline;" alt="latex">
of the array, let 
<img data-latex="$\kappa'(i)=i$" src=".images/ce82d39246a64d7fff9179feb7683ab0.svg"  valign="-4.289px" width="63.438px" height="17.186px" style="display:inline;" alt="latex">
but choose 
<img data-latex="$M=1$" src=".images/d4117387c4be22b84b7c6857cfca48bf.svg"  width="52.255px" height="11.764px" style="display:inline;" alt="latex">,
that is, the sliced array is a 1-dimensional array with stride matrix

<img data-latex="
$$
S'=
\begin{bmatrix}
d'_0\cdots d'_{N'-2}, \ldots, d'_0, 1
\end{bmatrix}
$$
" src=".images/34d6c394f7434d6a2fa4ad3ac7f3334c.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

and the set of indices of the slice array in reduced form is

<img data-latex="
$$
J' = \left\{j \left|
j = \sum_{k=0}^{N'-1}s'_{0,k} i'_k,
{\boldsymbol b}+D{\boldsymbol i}' = {\boldsymbol i}, 
{\boldsymbol i}\in I\right.\right\}
$$
" src=".images/1277f823a1f1994b1a5e36723135c500.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

Notice that when for a given
<img data-latex="${\boldsymbol i}\in I$" src=".images/56ce0d5a5aa6071fc1c041fb5f2d2ebf.svg"  valign="-0.673px" width="41.096px" height="12.608px" style="display:inline;" alt="latex">
there exists no such 
<img data-latex="${\boldsymbol i}'$" src=".images/9079272b4553fc7f9ec848de16a7e7d5.svg"  width="14.554px" height="13.96px" style="display:inline;" alt="latex">
that 
<img data-latex="${\boldsymbol b}+D{\boldsymbol i}' = {\boldsymbol i}$" src=".images/aaa7462a464d734ee3f32a74dff35003.svg"  valign="-1.093px" width="87.451px" height="15.054px" style="display:inline;" alt="latex">,
and 
 <img data-latex="$0\leqslant i'_k<d'_k$" src=".images/a9dbedd047b01f90614210a5a45e26bd.svg"  valign="-4.809px" width="85.808px" height="17.698px" style="display:inline;" alt="latex">
hold, then the corresponding index 
 <img data-latex="${\boldsymbol i}\in I$" src=".images/56ce0d5a5aa6071fc1c041fb5f2d2ebf.svg"  valign="-0.673px" width="41.096px" height="12.608px" style="display:inline;" alt="latex">
will be skipped.

<!--EOF-->
