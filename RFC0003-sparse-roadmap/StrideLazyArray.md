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
| Created    | 2020-07-10      |

# Slicing sparse arrays as views

## Introduction

The implementation of a strided array is defined by

- a pointer to the storage of array values,

- a shape tuple that contains the array dimensions numbers,

- a strides tuple that is used for relating the array element index
  with the corresponding array element value in the storage of array
  values.


The implementation of a sparse array in COO format is defined by

- a pointer to the storage of array values,

- a shape tuple that contains the array dimensions numbers,

- a pointer to the storage of array indices. The storages of array
  values and array indices must have one-to-one relation.


## Note 1: slicing of arrays, view vs copy

Slicing of a strided array is performed by computing new dimension numbers,
new strides, and offsetting the pointer to the storage of array
values. The storage of array values is unchanged leading to the
concept of the array view. Changing the array element value of a
sliced array will change the corresponding array element value of the
original array, and vice-versa.

Slicing of a sparse array is performed by computing new dimension
numbers, new indicies, and selecting the corresponding values from the
storage of array values. The selection must include creating a copy of
the storage of array values because the storages of array values and
array indices must have one-to-one relation.

## Note 2: implicit or explicit storage of array indices

A notable difference of strided and sparse arrays is about implicit
and explicit storage of array indices. In the strided array case, the
array indices do not require storage because the relation between the
index of an array element and the storage location of the
corresponding array element value can be computed using strides.  In
the sparse array case, only the specified array elements require
storage which leads to the need to store the indices of specified
array elements.

## Note 3: refactoring strides

The strides of a strided array comprises the information of two
operations: (i) locating array values in the storage and (ii)
selecting the array values according to slicing operation. 

An idea: split the strides of a strided array into two strides sets,
one representing the location operation of array values, and another
representing the selection operation from slicing operation.

Such a split of the strides involves also splitting the shape of a
strided array into the shape of original array and the shape of
strided array. Lastly, also the value of the pointer to the storage of
array values needs to be decomposed to the pointer value of original
array and an offset value induced from the slicing operation.

## Note 4: lazy slicing

The above idea can be reformulated as a lazy slicing of an array where
subsequent slicing operations on the array are composed to a single
slice object while applying the slice to the storage of array values
is postponed. To be specific, consider an array 
<img data-latex="$A$" src=".images/17cb9db9f13d3e7780cbd9f5cf0ca178.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
with shape
<img data-latex="${\boldsymbol d}=(d_0,\ldots, d_{N-1})$" src=".images/301b95351ea0b223a9bd085958bd13c5.svg"  valign="-4.289px" width="136.843px" height="17.186px" style="display:inline;" alt="latex">
and a slicing operation represented by a pair 
<img data-latex="$({\boldsymbol b}, S)$" src=".images/f53320c149b2d23179460f56cd30f5b1.svg"  valign="-4.289px" width="44.378px" height="17.186px" style="display:inline;" alt="latex">
such that

<img data-latex="
$$
A'[{\boldsymbol i}'] = A[{\boldsymbol b} + D{\boldsymbol i}']\qquad\forall{\boldsymbol i}'
$$
" src=".images/24b113b38de0ba947376fad7731d048b.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

is the corresponding sliced array with shape
<img data-latex="${\boldsymbol d}'=(d'_0,\ldots, d'_{N'-1})$" src=".images/d9e52e0d0bd67577a3d2d639b68afacc.svg"  valign="-6.492px" width="143.867px" height="20.473px" style="display:inline;" alt="latex">. We denote 
 <img data-latex="$A'\equiv A\{{\boldsymbol b}, D\}$" src=".images/edf0453540832f88d4d50a45e016c921.svg"  valign="-4.304px" width="104.508px" height="17.215px" style="display:inline;" alt="latex">.

Define a slice-lazy array
<img data-latex="$\tilde A$" src=".images/fed77f4fb4e2cf6500b6ac87302b5f9f.svg"  width="16.934px" height="15.646px" style="display:inline;" alt="latex">
corresponding to array 
<img data-latex="$A$" src=".images/17cb9db9f13d3e7780cbd9f5cf0ca178.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
and slicing operation with parameters
<img data-latex="$({\boldsymbol b}, D)$" src=".images/82fc53988f766807a824fe50911fb2ad.svg"  valign="-4.289px" width="47.473px" height="17.186px" style="display:inline;" alt="latex">
as follows:

<img data-latex="
$$
\tilde A_{A, {\boldsymbol b}, D}[{\boldsymbol i}'] = A[{\boldsymbol b}+D{\boldsymbol i}'],\qquad\forall{\boldsymbol i}'
$$
" src=".images/734a679242c450fea74ace31f22da120.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

that is, the implementation of a slice-lazy array constitutes of the original array and the slicing parameters.

Slicing a slice-lazy array is a slice-lazy array with the same original array but with computed slicing parameters:

<img data-latex="
$$
\tilde A_{A, {\boldsymbol b}, D}\{{\boldsymbol b}', D'\} = \tilde A_{A, {\boldsymbol b} + D{\boldsymbol b}', D'}
$$
" src=".images/cece50dc5d42d8f12cc3253826ed3641.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

A slice-lazy array requires a materialization operation, denote it by 
 <img data-latex="$M$" src=".images/bfe2792bda47257647e9131b83ddf87f.svg"  width="22.404px" height="11.764px" style="display:inline;" alt="latex">, that constructs a new array with slicing operation applied: 
 <img data-latex="$M(\tilde A) = A'$" src=".images/5f708b8fe62fb6849bf05332403baa5d.svg"  valign="-4.289px" width="85.114px" height="19.935px" style="display:inline;" alt="latex">, or

<img data-latex="
$$
M(\tilde A_{A, {\boldsymbol b}, D})[{\boldsymbol i}'] = A[{\boldsymbol b}+D{\boldsymbol i}']=A'[{\boldsymbol i}'],\qquad\forall{\boldsymbol i}'.
$$
" src=".images/c86e1241da328b6545063a0291301515.svg"  style="display:block;margin-left:50px;margin-right:auto;padding:0px" alt="latex">

If the array
<img data-latex="$A$" src=".images/17cb9db9f13d3e7780cbd9f5cf0ca178.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
is a strided array then with the slice-lazy array concept the strides of 
<img data-latex="$A$" src=".images/17cb9db9f13d3e7780cbd9f5cf0ca178.svg"  width="16.934px" height="11.764px" style="display:inline;" alt="latex">
are used purely for locating the array values in the storage.

The materialization operation 
<img data-latex="$M$" src=".images/bfe2792bda47257647e9131b83ddf87f.svg"  width="22.404px" height="11.764px" style="display:inline;" alt="latex">
can be parameterized with parameters
of the target storage format. For instance, one can materialize a
stride-lazy array so that the result has C-contiguous data layout, or
Fortran-contiguous layout, or be a strided array with any specified
strides.

## Final note: slicing a sparse array as a view

The concept of stride-lazy array can be applied to sparse arrays: a
sparse array acts as the storage of array values and the
materialization operation produces a sliced sparse array as discussed
in the Note 1 above.

<!--EOF-->
