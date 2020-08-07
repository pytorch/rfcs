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


# Intermediate plan for the sparse tensor support

## What we have?

- pytorch supports two array data layouts: strided and sparse COO
  storage formats. Currently, about 12% of pytorch functions ([48 out of
  365](https://github.com/Quansight-Labs/rfcs/blob/pearu/rfc0005/RFC0003-sparse-roadmap/SparseSupportState.md))
  work with sparse tensor inputs on CPU.

- typical failures on sparse inputs:

  - unsupported backend - 130 cases
  - unsupported layout - 52 cases
  - sparse tensors do not have strides - 23 cases
  - sparse tensors do not have is_contiguous - 12 cases
  - not implemented - 9 cases
  - ASSERT FAILED - 8 cases
  - only supports strided layout - 7 cases
  - varia - ca 65 cases, some are failures for strided format

## What is needed?

- Expand the scope of pytorch sparse tensors

  - Increase the number of pytorch functions that support sparse inputs
  
  - Introduce (a procedure for) new sparse storage formats -> increased performance for domain-specific sparse applications

  - Estimate when using a sparse storage can advantageous performance-wise. The estimate can be function and data dependent.

## Low-hanging and impactful actions

- Introduce fill-value attribute to sparse tensors, enables supporting
  a large number of functions that result sparse tensors with non-zero
  fill value

- Introduce sparse tensors using CSR/CSC storage format to increase
  matrix multiplication performance

- Prioritize pytorch functions that require sparse tensor support ->
  increases the impact of sparse tensor support in short-term

  - implement sparse storage format support for functions with highest priority


<!--EOF-->
