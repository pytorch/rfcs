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


This file is auto-generated, do not edit!

# The state of PyTorch tensor layouts support

The following table summarizes the state of PyTorch tensor layouts for
different PyTorch functions from the following namespaces:
- torch
- torch.nn.functional
- torch.sparse

|                              Section                               |             strided@cpu             |                  sparse_coo@cpu                  |
| :----------------------------------------------------------------- | :---------------------------------- | :----------------------------------------------- |
| <a href="#tensor-constructors">Tensor constructors</a>             | PASSED: 19, SKIPPED: 3              | FAILED: 12, PASSED: 6, SKIPPED: 4                |
| <a href="#trigonometry-functions">Trigonometry functions</a>       | PASSED: 45                          | FAILED: 41, PASSED: 4                            |
| <a href="#arithmetics-functions">Arithmetics functions</a>         | PASSED: 22                          | FAILED: 17, PARTIAL: 2, PASSED: 3                |
| <a href="#linear-algebra-algorithms">Linear Algebra algorithms</a> | PASSED: 31                          | FAILED: 27, PASSED: 4                            |
| <a href="#convolution-operations">Convolution operations</a>       | PASSED: 9                           | FAILED: 9                                        |
| <a href="#matrix-functions">Matrix functions</a>                   | PASSED: 22                          | FAILED: 22                                       |
| <a href="#general-array-functions">General array functions</a>     | PASSED: 42                          | FAILED: 31, PASSED: 11                           |
| <a href="#array-reductions">Array reductions</a>                   | FAILED: 1, PASSED: 23               | FAILED: 21, PASSED: 3                            |
| <a href="#normalization-functions">Normalization functions</a>     | PASSED: 7                           | FAILED: 7                                        |
| <a href="#logical-functions">Logical functions</a>                 | PASSED: 10                          | FAILED: 9, PASSED: 1                             |
| <a href="#comparison-operations">Comparison operations</a>         | PASSED: 25                          | FAILED: 23, PASSED: 2                            |
| <a href="#random-sampling">Random sampling</a>                     | PASSED: 10, SKIPPED: 6              | FAILED: 10, SKIPPED: 6                           |
| <a href="#harmonic-anaysis">Harmonic anaysis</a>                   | PASSED: 10                          | FAILED: 10                                       |
| <a href="#nn">NN</a>                                               |                                     |                                                  |
| <a href="#nndistance-functions">NN/Distance functions</a>          | PASSED: 3                           | FAILED: 3                                        |
| <a href="#nnloss-operations">NN/Loss operations</a>                | PASSED: 18                          | FAILED: 18                                       |
| <a href="#nnvision-operations">NN/Vision operations</a>            | PASSED: 10                          | FAILED: 10                                       |
| <a href="#nnactivation-functions">NN/Activation functions</a>      | FAILED: 2, PASSED: 25               | FAILED: 25, PASSED: 2                            |
| <a href="#nnpool-functions">NN/Pool functions</a>                  | PASSED: 35                          | FAILED: 35                                       |
| <a href="#nndropout-functions">NN/Dropout functions</a>            | PASSED: 5                           | FAILED: 3, PASSED: 2                             |
| <a href="#nnsparse-operations">NN/Sparse operations</a>            | PASSED: 3, SKIPPED: 1               | FAILED: 3, SKIPPED: 1                            |
| <a href="#quantization-functions">Quantization functions</a>       | PASSED: 1, SKIPPED: 5               | SKIPPED: 6                                       |
| <a href="#utility-functions">Utility functions</a>                 | PASSED: 7, SKIPPED: 14              | FAILED: 2, PASSED: 5, SKIPPED: 14                |
| Total                                                              | FAILED: 3, PASSED: 382, SKIPPED: 29 | FAILED: 338, PARTIAL: 2, PASSED: 43, SKIPPED: 31 |


The functions and possible failure messages are listed in the tables
of subsequent sections.

## Ranking of failures

The following table lists the ranking of failure messages:

|                  Status detail                   | strided@cpu | sparse_coo@cpu |
| :----------------------------------------------- | :---------- | :------------- |
| PASSED                                           | 382         | 43             |
| RuntimeError: backend not supported              | 3           | 189            |
| RuntimeError: unsupported layout                 | 0           | 54             |
| RuntimeError: no strides                         | 0           | 40             |
| RuntimeError: no is_contiguous                   | 0           | 12             |
| RuntimeError: requires strided layout            | 0           | 9              |
| RuntimeError: unsupported memory format option   | 0           | 8              |
| RuntimeError: memory format option not supported | 0           | 8              |
| RuntimeError: unimplemented reshape              | 0           | 6              |
| RuntimeError: not implemented for layout         | 0           | 6              |
| RuntimeError: no storage                         | 0           | 2              |



# Tensor constructors

|                                                      Function                                                      | strided@cpu |              sparse_coo@cpu              |
| :----------------------------------------------------------------------------------------------------------------- | :---------- | :--------------------------------------- |
| <a href="https://pytorch.org/docs/master/generated/torch.arange.html">torch.arange</a>                             | PASSED      | RuntimeError: backend not supported      |
| <a href="https://pytorch.org/docs/master/generated/torch.as_strided.html">torch.as_strided</a>                     | PASSED      | RuntimeError: backend not supported      |
| <a href="https://pytorch.org/docs/master/generated/torch.as_tensor.html">torch.as_tensor</a>                       | PASSED      | PASSED                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.dequantize.html">torch.dequantize</a>                     | SKIPPED     | SKIPPED                                  |
| <a href="https://pytorch.org/docs/master/generated/torch.empty.html">torch.empty</a>                               | PASSED      | PASSED                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.empty_like.html">torch.empty_like</a>                     | PASSED      | PASSED                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.empty_strided.html">torch.empty_strided</a> D             | PASSED      | RuntimeError: backend not supported      |
| <a href="https://pytorch.org/docs/master/generated/torch.eye.html">torch.eye</a> !                                 | PASSED      | RuntimeError: backend not supported      |
| <a href="https://pytorch.org/docs/master/generated/torch.from_numpy.html">torch.from_numpy</a>                     | PASSED      | SKIPPED                                  |
| <a href="https://pytorch.org/docs/master/generated/torch.full.html">torch.full</a> F                               | PASSED      | RuntimeError: not implemented for layout |
| <a href="https://pytorch.org/docs/master/generated/torch.full_like.html">torch.full_like</a> F                     | PASSED      | RuntimeError: unsupported layout         |
| <a href="https://pytorch.org/docs/master/generated/torch.linspace.html">torch.linspace</a>                         | PASSED      | RuntimeError: backend not supported      |
| <a href="https://pytorch.org/docs/master/generated/torch.logspace.html">torch.logspace</a>                         | PASSED      | RuntimeError: backend not supported      |
| <a href="https://pytorch.org/docs/master/generated/torch.ones.html">torch.ones</a> F                               | PASSED      | RuntimeError: not implemented for layout |
| <a href="https://pytorch.org/docs/master/generated/torch.ones_like.html">torch.ones_like</a> F                     | PASSED      | RuntimeError: unsupported layout         |
| <a href="https://pytorch.org/docs/master/generated/torch.quantize_per_channel.html">torch.quantize_per_channel</a> | SKIPPED     | SKIPPED                                  |
| <a href="https://pytorch.org/docs/master/generated/torch.quantize_per_tensor.html">torch.quantize_per_tensor</a>   | SKIPPED     | SKIPPED                                  |
| <a href="https://pytorch.org/docs/master/generated/torch.range.html">torch.range</a>                               | PASSED      | RuntimeError: backend not supported      |
| <a href="https://pytorch.org/docs/master/generated/torch.sparse_coo_tensor.html">torch.sparse_coo_tensor</a>       | PASSED      | RuntimeError: unexpected layout          |
| <a href="https://pytorch.org/docs/master/generated/torch.tensor.html">torch.tensor</a>                             | PASSED      | PASSED                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.zeros.html">torch.zeros</a>                               | PASSED      | PASSED                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.zeros_like.html">torch.zeros_like</a>                     | PASSED      | PASSED                                   |



# Trigonometry functions

|                                                              Function                                                              | strided@cpu |           sparse_coo@cpu            |
| :--------------------------------------------------------------------------------------------------------------------------------- | :---------- | :---------------------------------- |
| <a href="https://pytorch.org/docs/master/generated/torch.abs.html">torch.abs</a> !                                                 | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.absolute.html">torch.absolute</a> !                                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.acos.html">torch.acos</a> F                                               | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.acosh.html">torch.acosh</a> F                                             | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.angle.html">torch.angle</a> F                                             | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.asin.html">torch.asin</a> !                                               | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.asinh.html">torch.asinh</a> !                                             | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.atan.html">torch.atan</a> !                                               | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.atan2.html">torch.atan2</a> !                                             | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.atanh.html">torch.atanh</a> !                                             | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.conj.html">torch.conj</a> !                                               | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.cos.html">torch.cos</a> F                                                 | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.cosh.html">torch.cosh</a> F                                               | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.deg2rad.html">torch.deg2rad</a> !                                         | PASSED      | PASSED                              |
| <a href="https://pytorch.org/docs/master/generated/torch.digamma.html">torch.digamma</a> F                                         | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.erf.html">torch.erf</a> !                                                 | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.erfc.html">torch.erfc</a> F                                               | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.erfinv.html">torch.erfinv</a> !                                           | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.exp.html">torch.exp</a> F                                                 | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.expm1.html">torch.expm1</a> F                                             | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.hardsigmoid">torch.nn.functional.hardsigmoid</a> F | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.imag.html">torch.imag</a> !                                               | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/master/generated/torch.lgamma.html">torch.lgamma</a> F                                           | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.log.html">torch.log</a> F                                                 | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.log10.html">torch.log10</a> F                                             | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.log1p.html">torch.log1p</a> !                                             | PASSED      | PASSED                              |
| <a href="https://pytorch.org/docs/master/generated/torch.log2.html">torch.log2</a> F                                               | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.logaddexp.html">torch.logaddexp</a>                                       | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.logaddexp2.html">torch.logaddexp2</a>                                     | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.logit.html">torch.logit</a> F                                             | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.mvlgamma.html">torch.mvlgamma</a> F                                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.polygamma.html">torch.polygamma</a> F                                     | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.rad2deg.html">torch.rad2deg</a> !                                         | PASSED      | PASSED                              |
| <a href="https://pytorch.org/docs/master/generated/torch.real.html">torch.real</a> !                                               | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/master/generated/torch.rot90.html">torch.rot90</a>                                               | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.rsqrt.html">torch.rsqrt</a> !                                             | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.sigmoid">torch.nn.functional.sigmoid</a> F         | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.sigmoid.html">torch.sigmoid</a> F                                         | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.sin.html">torch.sin</a> !                                                 | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.sinh.html">torch.sinh</a> !                                               | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.sqrt.html">torch.sqrt</a> !                                               | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.square.html">torch.square</a> F                                           | PASSED      | PASSED                              |
| <a href="https://pytorch.org/docs/master/generated/torch.tan.html">torch.tan</a> !                                                 | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.tanh">torch.nn.functional.tanh</a> !               | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.tanh.html">torch.tanh</a> !                                               | PASSED      | RuntimeError: backend not supported |



# Arithmetics functions

|                                              Function                                              | strided@cpu |                                                    sparse_coo@cpu                                                     |
| :------------------------------------------------------------------------------------------------- | :---------- | :-------------------------------------------------------------------------------------------------------------------- |
| <a href="https://pytorch.org/docs/master/generated/torch.add.html">torch.add</a>                   | PASSED      | PASSED                                                                                                                |
| <a href="https://pytorch.org/docs/master/generated/torch.addcdiv.html">torch.addcdiv</a>           | PASSED      | RuntimeError: unsupported layout                                                                                      |
| <a href="https://pytorch.org/docs/master/generated/torch.addcmul.html">torch.addcmul</a>           | PASSED      | RuntimeError: unsupported layout                                                                                      |
| <a href="https://pytorch.org/docs/master/generated/torch.ceil.html">torch.ceil</a>                 | PASSED      | RuntimeError: backend not supported                                                                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.clamp.html">torch.clamp</a>               | PASSED      | RuntimeError: backend not supported                                                                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.div.html">torch.div</a>                   | PASSED      | RuntimeError: Sparse division requires a scalar or zero-dim dense tensor divisor (got shape [2, 2] for divisor)       |
| <a href="https://pytorch.org/docs/master/generated/torch.floor.html">torch.floor</a>               | PASSED      | RuntimeError: backend not supported                                                                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.floor_divide.html">torch.floor_divide</a> | PASSED      | RuntimeError: Sparse floor division requires a scalar or zero-dim dense tensor divisor (got shape [2, 2] for divisor) |
| <a href="https://pytorch.org/docs/master/generated/torch.fmod.html">torch.fmod</a>                 | PASSED      | RuntimeError: backend not supported                                                                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.frac.html">torch.frac</a>                 | PASSED      | RuntimeError: unsupported layout                                                                                      |
| <a href="https://pytorch.org/docs/master/generated/torch.gcd.html">torch.gcd</a>                   | PASSED      | RuntimeError: unsupported layout                                                                                      |
| <a href="https://pytorch.org/docs/master/generated/torch.lcm.html">torch.lcm</a>                   | PASSED      | RuntimeError: unsupported layout                                                                                      |
| <a href="https://pytorch.org/docs/master/generated/torch.mul.html">torch.mul</a>                   | PASSED      | PASSED                                                                                                                |
| <a href="https://pytorch.org/docs/master/generated/torch.neg.html">torch.neg</a>                   | PASSED      | RuntimeError: backend not supported                                                                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.pow.html">torch.pow</a>                   | PASSED      | RuntimeError: backend not supported                                                                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.reciprocal.html">torch.reciprocal</a>     | PASSED      | RuntimeError: unsupported layout                                                                                      |
| <a href="https://pytorch.org/docs/master/generated/torch.remainder.html">torch.remainder</a>       | PASSED      | RuntimeError: backend not supported                                                                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.round.html">torch.round</a>               | PASSED      | RuntimeError: backend not supported                                                                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.sign.html">torch.sign</a>                 | PASSED      | RuntimeError: backend not supported                                                                                   |
| <a href="https://pytorch.org/docs/master/generated/torch.sub.html">torch.sub</a>                   | PASSED      | PASSED                                                                                                                |
| <a href="https://pytorch.org/docs/master/generated/torch.true_divide.html">torch.true_divide</a>   | PASSED      | RuntimeError: Sparse true division requires a scalar or zero-dim dense tensor divisor (got shape [2, 2] for divisor)  |
| <a href="https://pytorch.org/docs/master/generated/torch.trunc.html">torch.trunc</a>               | PASSED      | RuntimeError: backend not supported                                                                                   |



# Linear Algebra algorithms

|                                                          Function                                                          | strided@cpu |           sparse_coo@cpu            |
| :------------------------------------------------------------------------------------------------------------------------- | :---------- | :---------------------------------- |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.bilinear">torch.nn.functional.bilinear</a> | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.cholesky.html">torch.cholesky</a>                                 | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.cholesky_inverse.html">torch.cholesky_inverse</a>                 | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.cholesky_solve.html">torch.cholesky_solve</a>                     | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/master/generated/torch.det.html">torch.det</a>                                           | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.eig.html">torch.eig</a>                                           | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.geqrf.html">torch.geqrf</a>                                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.ger.html">torch.ger</a>                                           | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.inverse.html">torch.inverse</a>                                   | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.lerp.html">torch.lerp</a>                                         | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.linear">torch.nn.functional.linear</a>     | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/master/generated/torch._lobpcg.lobpcg.html">torch._lobpcg.lobpcg</a>                     | PASSED      | PASSED                              |
| <a href="https://pytorch.org/docs/master/generated/torch.logdet.html">torch.logdet</a>                                     | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.lstsq.html">torch.lstsq</a>                                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.lu.html">torch.functional.lu</a>                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.lu_solve.html">torch.lu_solve</a>                                 | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.lu_unpack.html">torch.functional.lu_unpack</a>         | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.orgqr.html">torch.orgqr</a>                                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.ormqr.html">torch.ormqr</a>                                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch._lowrank.pca_lowrank.html">torch._lowrank.pca_lowrank</a>         | PASSED      | PASSED                              |
| <a href="https://pytorch.org/docs/master/generated/torch.pinverse.html">torch.pinverse</a>                                 | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.qr.html">torch.qr</a>                                             | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.slogdet.html">torch.slogdet</a>                                   | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.solve.html">torch.solve</a>                                       | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/master/generated/torch.svd.html">torch.svd</a>                                           | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch._lowrank.svd_lowrank.html">torch._lowrank.svd_lowrank</a>         | PASSED      | PASSED                              |
| <a href="https://pytorch.org/docs/master/generated/torch.symeig.html">torch.symeig</a>                                     | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.t.html">torch.t</a>                                               | PASSED      | PASSED                              |
| <a href="https://pytorch.org/docs/master/generated/torch.trace.html">torch.trace</a>                                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.trapz.html">torch.trapz</a>                                       | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/master/generated/torch.triangular_solve.html">torch.triangular_solve</a>                 | PASSED      | RuntimeError: no strides            |



# Convolution operations

|                                                           Function                                                           | strided@cpu |           sparse_coo@cpu            |
| :--------------------------------------------------------------------------------------------------------------------------- | :---------- | :---------------------------------- |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.conv1d">torch.conv1d</a>                     | PASSED      | RuntimeError: no is_contiguous      |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.conv2d">torch.conv2d</a>                     | PASSED      | RuntimeError: no is_contiguous      |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.conv3d">torch.conv3d</a>                     | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.conv_tbc.html">torch.conv_tbc</a>                                   | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.conv_transpose1d">torch.conv_transpose1d</a> | PASSED      | RuntimeError: no is_contiguous      |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.conv_transpose2d">torch.conv_transpose2d</a> | PASSED      | RuntimeError: no is_contiguous      |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.conv_transpose3d">torch.conv_transpose3d</a> | PASSED      | RuntimeError: no is_contiguous      |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.fold">torch.nn.functional.fold</a>           | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.unfold">torch.nn.functional.unfold</a>       | PASSED      | RuntimeError: backend not supported |



# Matrix functions

|                                                         Function                                                         | strided@cpu |                  sparse_coo@cpu                  |
| :----------------------------------------------------------------------------------------------------------------------- | :---------- | :----------------------------------------------- |
| <a href="https://pytorch.org/docs/master/generated/torch.addbmm.html">torch.addbmm</a>                                   | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.addmm.html">torch.addmm</a>                                     | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/sparse.html#torch.sparse.addmm">torch.sparse.addmm</a>                          | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.addmv.html">torch.addmv</a>                                     | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.addr.html">torch.addr</a>                                       | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.baddbmm.html">torch.baddbmm</a>                                 | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.bmm.html">torch.bmm</a>                                         | PASSED      | RuntimeError: operand must be dense              |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.chain_matmul.html">torch.functional.chain_matmul</a> | PASSED      | RuntimeError: no strides                         |
| <a href="https://pytorch.org/docs/master/generated/torch.cross.html">torch.cross</a>                                     | PASSED      | RuntimeError: memory format option not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.dot.html">torch.dot</a>                                         | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.matmul.html">torch.matmul</a>                                   | PASSED      | RuntimeError: no strides                         |
| <a href="https://pytorch.org/docs/master/generated/torch.matrix_power.html">torch.matrix_power</a>                       | PASSED      | RuntimeError: no strides                         |
| <a href="https://pytorch.org/docs/master/generated/torch.matrix_rank.html">torch.matrix_rank</a>                         | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.mm.html">torch.mm</a>                                           | PASSED      | RuntimeError: no strides                         |
| <a href="https://pytorch.org/docs/stable/sparse.html#torch.sparse.mm">torch.sparse.mm</a>                                | PASSED      | RuntimeError: no strides                         |
| <a href="https://pytorch.org/docs/master/generated/torch.mv.html">torch.mv</a>                                           | PASSED      | RuntimeError: no strides                         |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.tensordot.html">torch.functional.tensordot</a>       | PASSED      | RuntimeError: no strides                         |
| <a href="https://pytorch.org/docs/master/generated/torch.tril.html">torch.tril</a>                                       | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.tril_indices.html">torch.tril_indices</a>                       | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.triu.html">torch.triu</a>                                       | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.triu_indices.html">torch.triu_indices</a>                       | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.vander.html">torch.vander</a>                                   | PASSED      | RuntimeError: unsupported layout                 |



# General array functions

|                                                              Function                                                              | strided@cpu |                           sparse_coo@cpu                           |
| :--------------------------------------------------------------------------------------------------------------------------------- | :---------- | :----------------------------------------------------------------- |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.block_diag.html">torch.functional.block_diag</a>               | PASSED      | RuntimeError: no strides                                           |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.broadcast_tensors.html">torch.functional.broadcast_tensors</a> | PASSED      | PASSED                                                             |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.cartesian_prod.html">torch.functional.cartesian_prod</a>       | PASSED      | RuntimeError: backend not supported                                |
| <a href="https://pytorch.org/docs/master/generated/torch.cat.html">torch.cat</a>                                                   | PASSED      | PASSED                                                             |
| <a href="https://pytorch.org/docs/master/generated/torch.chunk.html">torch.chunk</a>                                               | PASSED      | RuntimeError: no strides                                           |
| <a href="https://pytorch.org/docs/master/generated/torch.clone.html">torch.clone</a>                                               | PASSED      | PASSED                                                             |
| <a href="https://pytorch.org/docs/master/generated/torch.combinations.html">torch.combinations</a>                                 | PASSED      | RuntimeError: backend not supported                                |
| <a href="https://pytorch.org/docs/master/generated/torch.detach.html">torch.detach</a>                                             | PASSED      | PASSED                                                             |
| <a href="https://pytorch.org/docs/master/generated/torch.diag.html">torch.diag</a>                                                 | PASSED      | RuntimeError: backend not supported                                |
| <a href="https://pytorch.org/docs/master/generated/torch.diag_embed.html">torch.diag_embed</a>                                     | PASSED      | RuntimeError: no storage                                           |
| <a href="https://pytorch.org/docs/master/generated/torch.diagflat.html">torch.diagflat</a>                                         | PASSED      | RuntimeError: no is_contiguous                                     |
| <a href="https://pytorch.org/docs/master/generated/torch.diagonal.html">torch.diagonal</a>                                         | PASSED      | RuntimeError: no storage                                           |
| <a href="https://pytorch.org/docs/master/generated/torch.flatten.html">torch.flatten</a>                                           | PASSED      | RuntimeError: unimplemented reshape                                |
| <a href="https://pytorch.org/docs/master/generated/torch.flip.html">torch.flip</a>                                                 | PASSED      | RuntimeError: backend not supported                                |
| <a href="https://pytorch.org/docs/master/generated/torch.fliplr.html">torch.fliplr</a>                                             | PASSED      | RuntimeError: backend not supported                                |
| <a href="https://pytorch.org/docs/master/generated/torch.flipud.html">torch.flipud</a>                                             | PASSED      | RuntimeError: backend not supported                                |
| <a href="https://pytorch.org/docs/master/generated/torch.gather.html">torch.gather</a>                                             | PASSED      | RuntimeError: backend not supported                                |
| <a href="https://pytorch.org/docs/master/generated/torch.index_add.html">torch.index_add</a>                                       | PASSED      | RuntimeError: unsupported memory format option                     |
| <a href="https://pytorch.org/docs/master/generated/torch.index_copy.html">torch.index_copy</a>                                     | PASSED      | RuntimeError: unsupported memory format option                     |
| <a href="https://pytorch.org/docs/master/generated/torch.index_fill.html">torch.index_fill</a>                                     | PASSED      | RuntimeError: unsupported memory format option                     |
| <a href="https://pytorch.org/docs/master/generated/torch.index_put.html">torch.index_put</a>                                       | PASSED      | RuntimeError: unsupported memory format option                     |
| <a href="https://pytorch.org/docs/master/generated/torch.index_select.html">torch.index_select</a>                                 | PASSED      | PASSED                                                             |
| <a href="https://pytorch.org/docs/master/generated/torch.masked_fill.html">torch.masked_fill</a>                                   | PASSED      | RuntimeError: unsupported memory format option                     |
| <a href="https://pytorch.org/docs/master/generated/torch.masked_scatter.html">torch.masked_scatter</a>                             | PASSED      | RuntimeError: unsupported memory format option                     |
| <a href="https://pytorch.org/docs/master/generated/torch.masked_select.html">torch.masked_select</a>                               | PASSED      | RuntimeError: backend not supported                                |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.meshgrid.html">torch.functional.meshgrid</a>                   | PASSED      | RuntimeError: backend not supported                                |
| <a href="https://pytorch.org/docs/master/generated/torch.narrow.html">torch.narrow</a>                                             | PASSED      | RuntimeError: no strides                                           |
| <a href="https://pytorch.org/docs/master/generated/torch.numel.html">torch.numel</a>                                               | PASSED      | PASSED                                                             |
| <a href="https://pytorch.org/docs/master/generated/torch.repeat_interleave.html">torch.repeat_interleave</a>                       | PASSED      | RuntimeError: unimplemented reshape;RuntimeError: no is_contiguous |
| <a href="https://pytorch.org/docs/master/generated/torch.reshape.html">torch.reshape</a>                                           | PASSED      | RuntimeError: unimplemented reshape                                |
| <a href="https://pytorch.org/docs/master/generated/torch.roll.html">torch.roll</a>                                                 | PASSED      | RuntimeError: backend not supported                                |
| <a href="https://pytorch.org/docs/master/generated/torch.scatter.html">torch.scatter</a>                                           | PASSED      | RuntimeError: unsupported memory format option                     |
| <a href="https://pytorch.org/docs/master/generated/torch.scatter_add.html">torch.scatter_add</a>                                   | PASSED      | RuntimeError: unsupported memory format option                     |
| <a href="https://pytorch.org/docs/master/generated/torch.select.html">torch.select</a>                                             | PASSED      | PASSED                                                             |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.split.html">torch.functional.split</a>                         | PASSED      | RuntimeError: no strides                                           |
| <a href="https://pytorch.org/docs/master/generated/torch.squeeze.html">torch.squeeze</a>                                           | PASSED      | RuntimeError: no strides                                           |
| <a href="https://pytorch.org/docs/master/generated/torch.stack.html">torch.stack</a>                                               | PASSED      | PASSED                                                             |
| <a href="https://pytorch.org/docs/master/generated/torch.take.html">torch.take</a>                                                 | PASSED      | RuntimeError: backend not supported                                |
| <a href="https://pytorch.org/docs/master/generated/torch.transpose.html">torch.transpose</a>                                       | PASSED      | PASSED                                                             |
| <a href="https://pytorch.org/docs/master/generated/torch.unbind.html">torch.unbind</a>                                             | PASSED      | PASSED                                                             |
| <a href="https://pytorch.org/docs/master/generated/torch.unsqueeze.html">torch.unsqueeze</a>                                       | PASSED      | PASSED                                                             |
| <a href="https://pytorch.org/docs/master/generated/torch.where.html">torch.where</a>                                               | PASSED      | RuntimeError: unsupported layout                                   |



# Array reductions

|                                                               Function                                                               |             strided@cpu             |            sparse_coo@cpu             |
| :----------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------- | :------------------------------------ |
| <a href="https://pytorch.org/docs/master/generated/torch.argmax.html">torch.argmax</a>                                               | PASSED                              | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.argmin.html">torch.argmin</a>                                               | PASSED                              | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.cdist.html">torch.functional.cdist</a>                           | PASSED                              | RuntimeError: no strides              |
| <a href="https://pytorch.org/docs/master/generated/torch.cummax.html">torch.cummax</a>                                               | PASSED                              | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.cummin.html">torch.cummin</a>                                               | PASSED                              | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.cumprod.html">torch.cumprod</a>                                             | PASSED                              | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.cumsum.html">torch.cumsum</a>                                               | PASSED                              | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.dist.html">torch.dist</a>                                                   | PASSED                              | PASSED                                |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.einsum.html">torch.functional.einsum</a>                         | PASSED                              | RuntimeError: no strides              |
| <a href="https://pytorch.org/docs/master/generated/torch.logcumsumexp.html">torch.logcumsumexp</a>                                   | PASSED                              | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.logsumexp.html">torch.logsumexp</a>                                         | PASSED                              | RuntimeError: requires strided layout |
| <a href="https://pytorch.org/docs/master/generated/torch.mean.html">torch.mean</a>                                                   | PASSED                              | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.median.html">torch.median</a>                                               | PASSED                              | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.mode.html">torch.mode</a>                                                   | PASSED                              | RuntimeError: requires strided layout |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.norm.html">torch.functional.norm</a>                             | PASSED                              | PASSED                                |
| <a href="https://pytorch.org/docs/master/generated/torch.prod.html">torch.prod</a>                                                   | PASSED                              | RuntimeError: no strides              |
| <a href="https://pytorch.org/docs/master/generated/torch.std.html">torch.std</a>                                                     | PASSED                              | RuntimeError: requires strided layout |
| <a href="https://pytorch.org/docs/master/generated/torch.std_mean.html">torch.std_mean</a>                                           | PASSED                              | RuntimeError: requires strided layout |
| <a href="https://pytorch.org/docs/stable/sparse.html#torch.sparse.sum">torch.sparse.sum</a>                                          | RuntimeError: backend not supported | PASSED                                |
| <a href="https://pytorch.org/docs/master/generated/torch.sum.html">torch.sum</a>                                                     | PASSED                              | RuntimeError: no strides              |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.unique.html">torch.functional.unique</a>                         | PASSED                              | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.unique_consecutive.html">torch.functional.unique_consecutive</a> | PASSED                              | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.var.html">torch.var</a>                                                     | PASSED                              | RuntimeError: requires strided layout |
| <a href="https://pytorch.org/docs/master/generated/torch.var_mean.html">torch.var_mean</a>                                           | PASSED                              | RuntimeError: requires strided layout |



# Normalization functions

|                                                                     Function                                                                     | strided@cpu |            sparse_coo@cpu             |
| :----------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :------------------------------------ |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.batch_norm">torch.nn.functional.batch_norm</a>                   | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.group_norm">torch.nn.functional.group_norm</a>                   | PASSED      | RuntimeError: no is_contiguous        |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.instance_norm">torch.nn.functional.instance_norm</a>             | PASSED      | RuntimeError: no strides              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.layer_norm">torch.nn.functional.layer_norm</a>                   | PASSED      | RuntimeError: no is_contiguous        |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.local_response_norm">torch.nn.functional.local_response_norm</a> | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.normalize">torch.nn.functional.normalize</a>                     | PASSED      | RuntimeError: requires strided layout |
| <a href="https://pytorch.org/docs/master/generated/torch.renorm.html">torch.renorm</a>                                                           | PASSED      | RuntimeError: backend not supported   |



# Logical functions

|                                             Function                                             | strided@cpu |            sparse_coo@cpu             |
| :----------------------------------------------------------------------------------------------- | :---------- | :------------------------------------ |
| <a href="https://pytorch.org/docs/master/generated/torch.all.html">torch.all</a>                 | PASSED      | RuntimeError: requires strided layout |
| <a href="https://pytorch.org/docs/master/generated/torch.any.html">torch.any</a>                 | PASSED      | PASSED                                |
| <a href="https://pytorch.org/docs/master/generated/torch.bitwise_and.html">torch.bitwise_and</a> | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.bitwise_not.html">torch.bitwise_not</a> | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.bitwise_or.html">torch.bitwise_or</a>   | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.bitwise_xor.html">torch.bitwise_xor</a> | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.logical_and.html">torch.logical_and</a> | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.logical_not.html">torch.logical_not</a> | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.logical_or.html">torch.logical_or</a>   | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.logical_xor.html">torch.logical_xor</a> | PASSED      | RuntimeError: backend not supported   |



# Comparison operations

|                                               Function                                               | strided@cpu |           sparse_coo@cpu            |
| :--------------------------------------------------------------------------------------------------- | :---------- | :---------------------------------- |
| <a href="https://pytorch.org/docs/master/generated/torch.allclose.html">torch.allclose</a>           | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.argsort.html">torch.argsort</a>             | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.bincount.html">torch.bincount</a>           | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.bucketize.html">torch.bucketize</a>         | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.count_nonzero.html">torch.count_nonzero</a> | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.eq.html">torch.eq</a>                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.equal.html">torch.equal</a>                 | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.ge.html">torch.ge</a>                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.gt.html">torch.gt</a>                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.histc.html">torch.histc</a>                 | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.is_nonzero.html">torch.is_nonzero</a>       | PASSED      | PASSED                              |
| <a href="https://pytorch.org/docs/master/generated/torch.isclose.html">torch.isclose</a>             | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.isfinite.html">torch.isfinite</a>           | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.isinf.html">torch.isinf</a>                 | PASSED      | RuntimeError: unsupported layout    |
| <a href="https://pytorch.org/docs/master/generated/torch.isnan.html">torch.isnan</a>                 | PASSED      | PASSED                              |
| <a href="https://pytorch.org/docs/master/generated/torch.kthvalue.html">torch.kthvalue</a>           | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.le.html">torch.le</a>                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.lt.html">torch.lt</a>                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.max.html">torch.max</a>                     | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.min.html">torch.min</a>                     | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.ne.html">torch.ne</a>                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.nonzero.html">torch.nonzero</a>             | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.searchsorted.html">torch.searchsorted</a>   | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.sort.html">torch.sort</a>                   | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.topk.html">torch.topk</a>                   | PASSED      | RuntimeError: backend not supported |



# Random sampling

|                                                      Function                                                      | strided@cpu |                  sparse_coo@cpu                  |
| :----------------------------------------------------------------------------------------------------------------- | :---------- | :----------------------------------------------- |
| <a href="https://pytorch.org/docs/master/generated/torch.bernoulli.html">torch.bernoulli</a>                       | PASSED      | RuntimeError: memory format option not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.random.get_rng_state.html">torch.random.get_rng_state</a> | SKIPPED     | SKIPPED                                          |
| <a href="https://pytorch.org/docs/master/generated/torch.random.initial_seed.html">torch.random.initial_seed</a>   | SKIPPED     | SKIPPED                                          |
| <a href="https://pytorch.org/docs/master/generated/torch.random.manual_seed.html">torch.random.manual_seed</a>     | SKIPPED     | SKIPPED                                          |
| <a href="https://pytorch.org/docs/master/generated/torch.multinomial.html">torch.multinomial</a>                   | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.normal.html">torch.normal</a>                             | SKIPPED     | SKIPPED                                          |
| <a href="https://pytorch.org/docs/master/generated/torch.poisson.html">torch.poisson</a>                           | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.rand.html">torch.rand</a>                                 | PASSED      | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/master/generated/torch.rand_like.html">torch.rand_like</a>                       | PASSED      | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/master/generated/torch.randint.html">torch.randint</a>                           | PASSED      | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/master/generated/torch.randint_like.html">torch.randint_like</a>                 | PASSED      | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/master/generated/torch.randn.html">torch.randn</a>                               | PASSED      | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/master/generated/torch.randn_like.html">torch.randn_like</a>                     | PASSED      | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/master/generated/torch.randperm.html">torch.randperm</a>                         | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch.random.seed.html">torch.random.seed</a>                   | SKIPPED     | SKIPPED                                          |
| <a href="https://pytorch.org/docs/master/generated/torch.random.set_rng_state.html">torch.random.set_rng_state</a> | SKIPPED     | SKIPPED                                          |



# Harmonic anaysis

|                                                  Function                                                  | strided@cpu |              sparse_coo@cpu              |
| :--------------------------------------------------------------------------------------------------------- | :---------- | :--------------------------------------- |
| <a href="https://pytorch.org/docs/master/generated/torch.bartlett_window.html">torch.bartlett_window</a>   | PASSED      | RuntimeError: not implemented for layout |
| <a href="https://pytorch.org/docs/master/generated/torch.blackman_window.html">torch.blackman_window</a>   | PASSED      | RuntimeError: not implemented for layout |
| <a href="https://pytorch.org/docs/master/generated/torch.fft.html">torch.fft</a>                           | PASSED      | RuntimeError: backend not supported      |
| <a href="https://pytorch.org/docs/master/generated/torch.hamming_window.html">torch.hamming_window</a>     | PASSED      | RuntimeError: not implemented for layout |
| <a href="https://pytorch.org/docs/master/generated/torch.hann_window.html">torch.hann_window</a>           | PASSED      | RuntimeError: not implemented for layout |
| <a href="https://pytorch.org/docs/master/generated/torch.ifft.html">torch.ifft</a>                         | PASSED      | RuntimeError: backend not supported      |
| <a href="https://pytorch.org/docs/master/generated/torch.irfft.html">torch.irfft</a>                       | PASSED      | RuntimeError: backend not supported      |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.istft.html">torch.functional.istft</a> | PASSED      | RuntimeError: unimplemented reshape      |
| <a href="https://pytorch.org/docs/master/generated/torch.rfft.html">torch.rfft</a>                         | PASSED      | RuntimeError: backend not supported      |
| <a href="https://pytorch.org/docs/master/generated/torch.functional.stft.html">torch.functional.stft</a>   | PASSED      | RuntimeError: backend not supported      |



# NN



## NN/Distance functions

|                                                                   Function                                                                   | strided@cpu |          sparse_coo@cpu          |
| :------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :------------------------------- |
| <a href="https://pytorch.org/docs/master/generated/torch.cosine_similarity.html">torch.cosine_similarity</a>                                 | PASSED      | RuntimeError: no strides         |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pairwise_distance">torch.nn.functional.pairwise_distance</a> | PASSED      | RuntimeError: unsupported layout |
| <a href="https://pytorch.org/docs/master/generated/torch.pdist.html">torch.pdist</a>                                                         | PASSED      | RuntimeError: no is_contiguous   |



## NN/Loss operations

|                                                                                  Function                                                                                  | strided@cpu |                  sparse_coo@cpu                  |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :----------------------------------------------- |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.binary_cross_entropy">torch.nn.functional.binary_cross_entropy</a>                         | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.binary_cross_entropy_with_logits">torch.nn.functional.binary_cross_entropy_with_logits</a> | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.cosine_embedding_loss">torch.nn.functional.cosine_embedding_loss</a>                       | PASSED      | RuntimeError: no strides                         |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.cross_entropy">torch.nn.functional.cross_entropy</a>                                       | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.ctc_loss">torch.nn.functional.ctc_loss</a>                                                 | PASSED      | RuntimeError: no is_contiguous                   |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.hinge_embedding_loss">torch.nn.functional.hinge_embedding_loss</a>                         | PASSED      | RuntimeError: memory format option not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.kl_div">torch.nn.functional.kl_div</a>                                                     | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.l1_loss">torch.nn.functional.l1_loss</a>                                                   | PASSED      | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.margin_ranking_loss">torch.nn.functional.margin_ranking_loss</a>                           | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.mse_loss">torch.nn.functional.mse_loss</a>                                                 | PASSED      | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.multi_margin_loss">torch.nn.functional.multi_margin_loss</a>                               | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.multilabel_margin_loss">torch.nn.functional.multilabel_margin_loss</a>                     | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.multilabel_soft_margin_loss">torch.nn.functional.multilabel_soft_margin_loss</a>           | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.nll_loss">torch.nn.functional.nll_loss</a>                                                 | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.poisson_nll_loss">torch.nn.functional.poisson_nll_loss</a>                                 | PASSED      | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.smooth_l1_loss">torch.nn.functional.smooth_l1_loss</a>                                     | PASSED      | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.soft_margin_loss">torch.nn.functional.soft_margin_loss</a>                                 | PASSED      | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.triplet_margin_loss">torch.nn.functional.triplet_margin_loss</a>                           | PASSED      | RuntimeError: unsupported layout                 |



## NN/Vision operations

|                                                                   Function                                                                   | strided@cpu |            sparse_coo@cpu             |
| :------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :------------------------------------ |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional._pad">torch.nn.functional._pad</a>                           | PASSED      | RuntimeError: unsupported layout      |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional._pad_circular">torch.nn.functional._pad_circular</a>         | PASSED      | RuntimeError: no strides              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.affine_grid">torch.nn.functional.affine_grid</a>             | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.channel_shuffle.html">torch.channel_shuffle</a>                                     | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample">torch.nn.functional.grid_sample</a>             | PASSED      | RuntimeError: requires strided layout |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate">torch.nn.functional.interpolate</a>             | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/master/generated/torch.pixel_shuffle.html">torch.pixel_shuffle</a>                                         | PASSED      | RuntimeError: unimplemented reshape   |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.upsample">torch.nn.functional.upsample</a>                   | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.upsample_bilinear">torch.nn.functional.upsample_bilinear</a> | PASSED      | RuntimeError: backend not supported   |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.upsample_nearest">torch.nn.functional.upsample_nearest</a>   | PASSED      | RuntimeError: backend not supported   |



## NN/Activation functions

|                                                                Function                                                                |             strided@cpu             |                  sparse_coo@cpu                  |
| :------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------- | :----------------------------------------------- |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.celu">torch.nn.functional.celu</a>                     | PASSED                              | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.elu">torch.nn.functional.elu</a>                       | PASSED                              | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.gelu">torch.nn.functional.gelu</a>                     | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.glu">torch.nn.functional.glu</a>                       | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.gumbel_softmax">torch.nn.functional.gumbel_softmax</a> | PASSED                              | RuntimeError: memory format option not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.hardshrink">torch.nn.functional.hardshrink</a>         | PASSED                              | RuntimeError: memory format option not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.hardshrink.html">torch.hardshrink</a>                                         | PASSED                              | RuntimeError: memory format option not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.hardswish">torch.nn.functional.hardswish</a>           | PASSED                              | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.hardtanh">torch.nn.functional.hardtanh</a>             | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.leaky_relu">torch.nn.functional.leaky_relu</a>         | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch._C._nn.log_sigmoid.html">torch._C._nn.log_sigmoid</a>                         | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.log_softmax">torch.nn.functional.log_softmax</a>       | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/sparse.html#torch.sparse.log_softmax">torch.sparse.log_softmax</a>                            | RuntimeError: backend not supported | PASSED                                           |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.prelu">torch.nn.functional.prelu</a>                   | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.relu">torch.nn.functional.relu</a>                     | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.relu6">torch.nn.functional.relu6</a>                   | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.rrelu">torch.nn.functional.rrelu</a>                   | PASSED                              | RuntimeError: memory format option not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.selu">torch.nn.functional.selu</a>                     | PASSED                              | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.silu">torch.nn.functional.silu</a>                     | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.softmax">torch.nn.functional.softmax</a>               | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/sparse.html#torch.sparse.softmax">torch.sparse.softmax</a>                                    | RuntimeError: backend not supported | PASSED                                           |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.softmin">torch.nn.functional.softmin</a>               | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/master/generated/torch._C._nn.softplus.html">torch._C._nn.softplus</a>                               | PASSED                              | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/master/generated/torch._C._nn.softshrink.html">torch._C._nn.softshrink</a>                           | PASSED                              | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.softsign">torch.nn.functional.softsign</a>             | PASSED                              | RuntimeError: unsupported layout                 |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.tanhshrink">torch.nn.functional.tanhshrink</a>         | PASSED                              | RuntimeError: backend not supported              |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.threshold">torch.nn.functional.threshold</a>           | PASSED                              | RuntimeError: backend not supported              |



## NN/Pool functions

|                                                                                    Function                                                                                    | strided@cpu |           sparse_coo@cpu            |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :---------------------------------- |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional._adaptive_max_pool1d">torch.nn.functional._adaptive_max_pool1d</a>                             | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional._adaptive_max_pool2d">torch.nn.functional._adaptive_max_pool2d</a>                             | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional._adaptive_max_pool3d">torch.nn.functional._adaptive_max_pool3d</a>                             | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional._fractional_max_pool2d">torch.nn.functional._fractional_max_pool2d</a>                         | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional._fractional_max_pool3d">torch.nn.functional._fractional_max_pool3d</a>                         | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional._max_pool1d">torch.nn.functional._max_pool1d</a>                                               | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional._max_pool2d">torch.nn.functional._max_pool2d</a>                                               | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional._max_pool3d">torch.nn.functional._max_pool3d</a>                                               | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.adaptive_avg_pool1d.html">torch.adaptive_avg_pool1d</a>                                                               | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.adaptive_avg_pool2d">torch.nn.functional.adaptive_avg_pool2d</a>                               | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.adaptive_avg_pool3d">torch.nn.functional.adaptive_avg_pool3d</a>                               | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.adaptive_max_pool1d">torch.nn.functional.adaptive_max_pool1d</a>                               | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.adaptive_max_pool1d_with_indices">torch.nn.functional.adaptive_max_pool1d_with_indices</a>     | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.adaptive_max_pool2d">torch.nn.functional.adaptive_max_pool2d</a>                               | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.adaptive_max_pool2d_with_indices">torch.nn.functional.adaptive_max_pool2d_with_indices</a>     | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.adaptive_max_pool3d">torch.nn.functional.adaptive_max_pool3d</a>                               | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.adaptive_max_pool3d_with_indices">torch.nn.functional.adaptive_max_pool3d_with_indices</a>     | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch.avg_pool1d.html">torch.avg_pool1d</a>                                                                                 | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/master/generated/torch._C._nn.avg_pool2d.html">torch._C._nn.avg_pool2d</a>                                                                   | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/master/generated/torch._C._nn.avg_pool3d.html">torch._C._nn.avg_pool3d</a>                                                                   | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.fractional_max_pool2d">torch.nn.functional.fractional_max_pool2d</a>                           | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.fractional_max_pool2d_with_indices">torch.nn.functional.fractional_max_pool2d_with_indices</a> | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.fractional_max_pool3d">torch.nn.functional.fractional_max_pool3d</a>                           | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.fractional_max_pool3d_with_indices">torch.nn.functional.fractional_max_pool3d_with_indices</a> | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.lp_pool1d">torch.nn.functional.lp_pool1d</a>                                                   | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.lp_pool2d">torch.nn.functional.lp_pool2d</a>                                                   | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.max_pool1d">torch.nn.functional.max_pool1d</a>                                                 | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.max_pool1d_with_indices">torch.nn.functional.max_pool1d_with_indices</a>                       | PASSED      | RuntimeError: no strides            |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.max_pool2d">torch.nn.functional.max_pool2d</a>                                                 | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.max_pool2d_with_indices">torch.nn.functional.max_pool2d_with_indices</a>                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.max_pool3d">torch.nn.functional.max_pool3d</a>                                                 | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.max_pool3d_with_indices">torch.nn.functional.max_pool3d_with_indices</a>                       | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.max_unpool1d">torch.nn.functional.max_unpool1d</a>                                             | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.max_unpool2d">torch.nn.functional.max_unpool2d</a>                                             | PASSED      | RuntimeError: backend not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.max_unpool3d">torch.nn.functional.max_unpool3d</a>                                             | PASSED      | RuntimeError: backend not supported |



## NN/Dropout functions

|                                                                       Function                                                                       | strided@cpu |                  sparse_coo@cpu                  |
| :--------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :----------------------------------------------- |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.alpha_dropout">torch.nn.functional.alpha_dropout</a>                 | PASSED      | PASSED                                           |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.dropout">torch.nn.functional.dropout</a>                             | PASSED      | RuntimeError: memory format option not supported |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.dropout2d">torch.nn.functional.dropout2d</a>                         | PASSED      | RuntimeError: no is_contiguous                   |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.dropout3d">torch.nn.functional.dropout3d</a>                         | PASSED      | RuntimeError: no is_contiguous                   |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.feature_alpha_dropout">torch.nn.functional.feature_alpha_dropout</a> | PASSED      | PASSED                                           |



## NN/Sparse operations

|                                                                              Function                                                                              | strided@cpu |           sparse_coo@cpu            |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- | :---------------------------------- |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.embedding">torch.nn.functional.embedding</a>                                       | PASSED      | RuntimeError: unimplemented reshape |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.embedding_bag">torch.nn.functional.embedding_bag</a>                               | PASSED      | RuntimeError: unimplemented reshape |
| <a href="https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.multi_head_attention_forward">torch.nn.functional.multi_head_attention_forward</a> | SKIPPED     | SKIPPED                             |
| <a href="https://pytorch.org/docs/master/generated/torch._C._nn.one_hot.html">torch._C._nn.one_hot</a>                                                             | PASSED      | RuntimeError: backend not supported |



# Quantization functions

|                                                           Function                                                           | strided@cpu | sparse_coo@cpu |
| :--------------------------------------------------------------------------------------------------------------------------- | :---------- | :------------- |
| <a href="https://pytorch.org/docs/master/generated/torch.int_repr.html">torch.int_repr</a>                                   | PASSED      | SKIPPED        |
| <a href="https://pytorch.org/docs/master/generated/torch.q_per_channel_axis.html">torch.q_per_channel_axis</a>               | SKIPPED     | SKIPPED        |
| <a href="https://pytorch.org/docs/master/generated/torch.q_per_channel_scales.html">torch.q_per_channel_scales</a>           | SKIPPED     | SKIPPED        |
| <a href="https://pytorch.org/docs/master/generated/torch.q_per_channel_zero_points.html">torch.q_per_channel_zero_points</a> | SKIPPED     | SKIPPED        |
| <a href="https://pytorch.org/docs/master/generated/torch.q_scale.html">torch.q_scale</a>                                     | SKIPPED     | SKIPPED        |
| <a href="https://pytorch.org/docs/master/generated/torch.q_zero_point.html">torch.q_zero_point</a>                           | SKIPPED     | SKIPPED        |



# Utility functions

|                                                              Function                                                              | strided@cpu |      sparse_coo@cpu      |
| :--------------------------------------------------------------------------------------------------------------------------------- | :---------- | :----------------------- |
| <a href="https://pytorch.org/docs/master/generated/torch.can_cast.html">torch.can_cast</a>                                         | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch.compiled_with_cxx11_abi.html">torch.compiled_with_cxx11_abi</a>           | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch._C.get_default_dtype.html">torch._C.get_default_dtype</a>                 | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch.get_device.html">torch.get_device</a>                                     | PASSED      | PASSED                   |
| <a href="https://pytorch.org/docs/master/generated/torch._C.get_num_interop_threads.html">torch._C.get_num_interop_threads</a>     | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch._C.get_num_threads.html">torch._C.get_num_threads</a>                     | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch.is_complex.html">torch.is_complex</a>                                     | PASSED      | PASSED                   |
| <a href="https://pytorch.org/docs/master/generated/torch.is_floating_point.html">torch.is_floating_point</a>                       | PASSED      | PASSED                   |
| <a href="https://pytorch.org/docs/master/generated/torch.is_signed.html">torch.is_signed</a>                                       | PASSED      | PASSED                   |
| <a href="https://pytorch.org/docs/master/generated/torch.serialization.load.html">torch.serialization.load</a>                     | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch.promote_types.html">torch.promote_types</a>                               | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch.result_type.html">torch.result_type</a>                                   | PASSED      | PASSED                   |
| <a href="https://pytorch.org/docs/master/generated/torch.serialization.save.html">torch.serialization.save</a>                     | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch.set_default_dtype.html">torch.set_default_dtype</a>                       | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch.set_default_tensor_type.html">torch.set_default_tensor_type</a>           | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch._C.set_flush_denormal.html">torch._C.set_flush_denormal</a>               | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch._C.set_num_interop_threads.html">torch._C.set_num_interop_threads</a>     | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch._C.set_num_threads.html">torch._C.set_num_threads</a>                     | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch._tensor_str.set_printoptions.html">torch._tensor_str.set_printoptions</a> | SKIPPED     | SKIPPED                  |
| <a href="https://pytorch.org/docs/master/generated/torch.view_as_complex.html">torch.view_as_complex</a>                           | PASSED      | RuntimeError: no strides |
| <a href="https://pytorch.org/docs/master/generated/torch.view_as_real.html">torch.view_as_real</a>                                 | PASSED      | RuntimeError: no strides |

<!--EOF-->