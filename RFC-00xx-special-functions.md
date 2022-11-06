# Special Functions

_Author’s note—This RFC is a work-in-progress._

## Authors

* Allen Goodman (@0x00b1)

## Summary

This proposal concerns adding new operators to PyTorch's special functions module (i.e., `torch.special`). The proposed operators have a wide range of use in scientific computing and numerical methods.

This RFC proposes:

* a coherent philosophy for PyTorch’s special functions module ([torch.special](https://pytorch.org/docs/stable/special.html)) that clearly distinguishes PyTorch’s elementary from special functions; and
* a set of new [torch](https://pytorch.org/docs/stable/torch.html) and [torch.special](https://pytorch.org/docs/stable/special.html) operators that provide a robust numerical foundation for PyTorch and adhere to the aforementioned philosophy.

## Motivation

PyTorch users:

* solves the variety of common scientific and engineering problems that special functions address.

PyTorch maintainers:

* provides much needed standardization to committing future operators to PyTorch.
* provides an extremely useful set of operators that can and should be used for tricky numerical problems (e.g., implementing challenging distribution functions and gradients) and useful decomposition targets.
 
## Special Functions

There’s no formal definition of a *special function*. Colloquially, and for the purpose of this RFC, a special function is a non-elementary function that has an established name and notation due to its importance and ubiquity.

### Elementary Functions

Unlike special functions,  *elementary functions* have a rigorous definition but, for simplicity, this RFC uses a simplified definition, categorizing the following functions as elementary:

* Power Functions
  * torch.pow
  * torch.sqrt
* Exponential Functions
  * torch.exp
  * torch.exp2
  * torch.expm1
* Logarithmic Functions
  * torch.log
  * torch.log10
  * torch.log1p
  * torch.log2
  * torch.xlog1py
  * torch.xlogy
* Trigonometric Functions
  * torch.sin
  * torch.cos
  * torch.tan
  * torch.cot
  * torch.csc
  * torch.sec
* Inverse Trigonometric Functions 
  * torch.asin
  * torch.acos
  * torch.atan
  * torch.atan
  * torch.acot
  * torch.acsc
  * torch.asec
* Hyperbolic Functions
  * torch.sinh
  * torch.cosh
  * torch.tanh
  * torch.coth
  * torch.csch
  * torch.sech
* Inverse Hyperbolic Functions
  * torch.asinh
  * torch.acosh
  * torch.atanh
  * torch.atanh
  * torch.acoth
  * torch.acsch
  * torch.asech
* Product Logarithms
* Roots
  * `torch.root` *See the “Polynomial Root” section for the proposed operator*
* Maximum and Minimum
  * torch.max
  * torch.min
* Cardinal Functions
  * torch.sinc
* Greatest Common Divisor (GCD) and Least Common Multiple (LCM)
  * torch.gcd
  * torch.lcm
* Rounding and Congruence Functions
  * torch.ceil
  * torch.floor
  * torch.floor_divide
  * torch.remainder
  * torch.round
* Discontinuous Functions
* Functions with Singular Support

### Special Functions

PyTorch’s mathematical operators are categorized as either “elementary” or “special.” 

An elementary function is a mathematical function whose corresponding operator is available from the `torch` module. A special function is a mathematical function whose corresponding operator is available from the `torch.special` module. Regardless of whether an operator implements an elementary or special function, users should expect that each operator share the following properties:

* A name that adheres to the naming policy.
* CPU and CUDA implementations
* C++ and Python function documentation that state a function’s:
  * common mathematical definition;
  * real and complex domains;
  * real and complex graphs; and, if relevant,
  * mathematical and scientific applications.
* If differentiable, derivtatives for each variable.

#### Naming Policy

## Implementation

### c10

#### Constants

To ease implementation, c10 will provide the following lookup tables (LUTs):

* factorial, $n!$, where $n = 1, 2, \cdots, 170;$
* natural logarithm of factorial, $\ln{\left(n!\right)}$, where $n = 1, 2, \cdots, 300;$
* double factorial, $n!!$, where $n = -1,000, -999, \cdots, 300;$
* natural logarithm of double factorial, $\ln{\left(n!!\right)}$, where $n = -1,000, -999, \cdots, 300;$ and
* prime number, $p\left(n\right)$, where $n = 1, 2, \cdots, 10,000.$

#### Functions

To ease implementation, c10 will provide the following numerical approximations:

* Lanczos approximation;
* Spouge’s approximation; and
* Stirling’s approximation.

##### Lanczos Approximation

Numerical method for computing the gamma function. It exhibits faster convergence than Spouge’s approximation.

##### Spouge’s approximation

Numerical method for computing the gamma function. Its coefficients are faster to compute than Lanczos approximation coefficients.

##### Stirling’s Approximation

Numerical method for computing factorials.

### ATen

Each special function has a corresponding ATen operator.

#### CPU

#### CUDA

### torch

## C++ API

## Python API

<details>
<summary>Factorials</summary>

### Factorials

#### Factorial

```Python
factorial(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

$n^{\text{th}}$ factorial, if $n \in \mathbb{N}$:

$$n! = \prod_{k = 1}^{n}k.$$

Otherwise:

$$n! = \Gamma(n + 1),$$

where $\Gamma$ is the gamma function.

$n!$ is defined for $\left\\{n \in \mathbb{R} \mid n \geq 0 \vee n \notin \mathbb{Z} \right\\}$ and $\left\\{n \in \mathbb{C} \mid \operatorname{Re}(n) \geq 0 \vee n \notin \mathbb{Z} \right\\}$.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Natural Logarithm of Factorial

```Python
ln_factorial(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Natural logarithm of $n^{\text{th}}$ factorial, $\ln{(n!)}$.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Double Factorial

```Python
double_factorial(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

$n^{\text{th}}$ double factorial, if $n \in \mathbb{N}$:

$$n!!=\prod_{k = 0}^{\left\lceil\tfrac{n}{2}\right\rceil-1}(n-2k).$$

Otherwise:

$$n!!=\left(\frac{2}{\pi}\right)^{\frac{1}{4}(1-\cos(\pi n))}2^{\tfrac{2}{n}}\Gamma\left(\frac{n}{2}+1\right),$$

where $\Gamma$ is the gamma function.

$n!!$ is defined for $\left\\{n \in \mathbb{R} \mid n \geq 0 \vee \tfrac{n}{2} \notin \mathbb{Z} \right\\}$ and $\left\\{n \in \mathbb{C} \mid n \geq 0 \vee \tfrac{n}{2} \notin \mathbb{Z} \right\\}$.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Natural Logarithm of Double Factorial

```Python
ln_double_factorial(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Natural logarithm of $n^{\text{th}}$ double factorial, $\ln{(n!!)}$.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Rising Factorial

```Python
rising_factorial(
    z: Tensor, 
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Rising factorial:

$$z^{n}=\frac{\Gamma(z + n)}{\Gamma(z)},$$

where $\Gamma$ is the gamma function.

$z^{n}$ is defined for all real and complex $z$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $n$ is a number, $z$ must be a tensor.

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – exponent. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Natural Logarithm of Rising Factorial

```Python
ln_rising_factorial(
    z: Tensor, 
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Natural logarithm of rising factorial, $\operatorname{ln}{(z^{n})}$.

$\operatorname{ln}{(z^{n})}$ is defined for all real and complex $z$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $n$ is a number, $z$ must be a tensor.

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – exponent. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Falling Factorial

```Python
falling_factorial(
    z: Tensor, 
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Falling factorial:

$$z_{n}=\frac{\Gamma(z + 1)}{\Gamma(z - n + 1)},$$

where $\Gamma$ is the gamma function.

$z_{n}$ is defined for all real and complex $z$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $n$ is a number, $z$ must be a tensor.

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – exponent. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Natural Logarithm of Falling Factorial

```Python
ln_falling_factorial(
    z: Tensor, 
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Natural logarithm of falling factorial, $\operatorname{ln}{(z_{n})}$.

$\operatorname{ln}{(z_{n})}$ is defined for all real and complex $z$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $n$ is a number, $z$ must be a tensor.

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – exponent. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Combinatorial Numbers and Functions</summary>

### Combinatorial Numbers and Functions

#### Binomial Coefficient

```Python
binomial_coefficient(
    n: Tensor,
    k: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

Binomial coefficient:

$${\binom{n}{k}} = {\frac{n!}{k!(n - k)!}}.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $k$ is a number, $n$ must be a tensor.

**k** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – exponent. If $n$ is a number, $k$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Natural Logarithm of Binomial Coefficient

```Python
ln_binomial_coefficient(
    n: Tensor,
    k: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

Natural logarithm of binomial coefficient, $\operatorname{ln}{{\binom{n}{k}}}$.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $k$ is a number, $n$ must be a tensor.

**k** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – exponent. If $n$ is a number, $k$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Catalan Number

```Python
catalan_number_c(
    n: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

$n^{\text{th}}$ Catalan number:

$$C_{z} = \frac{2^{2z}\Gamma(z + \tfrac{1}{2})}{\sqrt{\pi} \Gamma(z + 2)}.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Stirling Number of the First Kind

```Python
stirling_number_s_1(
    n: Tensor,
    m: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

Stirling number of the first kind:

$$s(n, k) = $$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Stirling Number of the Second Kind

```Python
stirling_number_s_2(
    n: Tensor,
    m: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

Stirling number of the second kind:

$$S(n, k) = {\frac{1}{k!}}\sum_{i = 0}^{k}(-1)^{i}{\binom{k}{i}}(k - i)^{n}.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Bell Number

```Python
bell_number_b(
    n: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

$n^{\text{th}}$ Bell number:

$$B_{n}=\sum_{k = 0}^{n}S(n, k).$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Delannoy Number

```Python
delannoy_number_d(
    m: Tensor,
    n: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

Delannoy number:

$$D(m, n) = \sum_{k = 0}^{\min(m, n)}\binom{m + n - k}{m}\binom{m}{k}.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Motzkin Number

```Python
motzkin_number_m(
    n: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

$n^{\text{th}}$ Motzkin number:

$$M_{n} = \sum_{k = 0}^{\lfloor \frac{n}{2} \rfloor}{\binom{n}{2k}}C_{k}.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Narayana Number

```Python
narayana_number_n(
    n: Tensor,
    k: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

Narayana number:

$$N(n, k) = {\frac{1}{n}}\binom{n}{k}\binom{n}{k - 1}.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Schröder Number

```Python
schroder_number_r(
    n: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

$n^{\text{th}}$ Schröder number:

$$r_{n} = D(n, n) - D(n + 1, n - 1).$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Gamma and Related Functions</summary>

### Gamma and Related Functions

#### Gamma Function

```Python
gamma(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Gamma function:

$$\Gamma(z)=\int_{0}^{\infty}t^{z-1}e^{-t}dt.$$

$\Gamma(z)$ is defined for $\left\\{n \in \mathbb{R} \mid n > 0 \vee n \notin \mathbb{Z} \right\\}$ and $\left\\{n \in \mathbb{C} \mid \operatorname{Re}(n) > 0 \vee n \notin \mathbb{Z} \right\\}$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Reciprocal Gamma Function

```Python
reciprocal_gamma(
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Polygamma Function

```Python
polygamma(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Polygamma function, if, and only if, $n \in \mathbb{Z} \land n > 0$:

$$\psi^{n}(z)=(-1)^{n+1} n! \sum _{k=0}^{\infty }\frac{1}{(k+z)^{n+1}}.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – derivative.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Digamma Function

```Python
digamma(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Digamma function:

$$\psi(z)=\sum _{k=1}^{\infty } \left(\frac{1}{k}-\frac{1}{k+z-1}\right)-\gamma.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Trigamma Function

```Python
trigamma(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Logarithm of the Gamma Function

```Python
log_gamma(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Logarithm of the gamma function:

$$\log{\Gamma}(z)=\sum _{k=1}^{\infty } \left(\frac{z}{k}-\log \left(1+\frac{z}{k}\right)\right)-\gamma  z-\log (z).$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Sign of the Gamma Function

```Python
sign_gamma(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Beta Function

```Python
beta(
    a: Tensor, 
    b: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Beta function:

$$\operatorname{B}(a, b) = \frac{\Gamma(a) \Gamma(b)}{\Gamma(a + b)}$$

where $\Gamma$ is the gamma function.

##### Parameters

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**b** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Logarithm of the Beta Function

```Python
log_beta(
    a: Tensor, 
    b: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**b** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Exponential and Logarithmic Integrals</summary>

### Exponential and Logarithmic Integrals

#### Exponential Integral, $\operatorname{Ein}$

```Python
exponential_integral_ein(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Exponential integral:

$$\operatorname{Ein}(z)=\int_{0}^{z}(1-e^{-t}){\frac{dt}{t}}.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Exponential Integral, $\operatorname{Ei}$

```Python
exponential_integral_ei(
    x: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Exponential integral:

$$\operatorname{Ei}(z) = \sum_{k = 1}^{\infty} \frac{z^{k}}{k k!} + \gamma + \frac{1}{2}(\ln{(z)} - \ln{(\tfrac{1}{z})}).$$

##### Parameters

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Exponential Integral, $E_{1}$

```Python
exponential_integral_e_1(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Exponential integral:

$$E_{1}(z)=\int _{z}^{\infty}{\frac{e^{-t}}{t}}.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Exponential Integral, $E_{n}$

```Python
exponential_integral_e(
    n: Tensor, 
    x: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Exponential integral:

$$E_{n}(x)=\int_{1}^{\infty}{\frac{e^{-xt}}{t^{n}}}dt.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Logarithmic Integral

```Python
logarithmic_integral_li(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Logarithmic integral:

$$\operatorname{li}(z)=\int_{0}^{z}{\frac{1}{\ln{(t)}}}dt.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Error and Related Functions</summary>

### Error and Related Functions

#### Error Function

```Python
error_erf(z: Tensor, *, out: Optional[Tensor] = None) -> Tensor
```

Error function: 

$$\operatorname{erf}(z) = \frac{2}{\sqrt{\pi}} \sum_{k = 0}^{\infty } \frac{-1^{k} z^{2k + 1}}{k!(2k + 1)}.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Complementary Error Function

```Python
error_erfc(z: Tensor, *, out: Optional[Tensor] = None) -> Tensor
```

Complementary error function:

$$\operatorname{erfc}(z) = 1 - \frac{2}{\sqrt{\pi}} \sum_{k = 0}^{\infty} \frac{-1^{k} z^{2k + 1}}{k!(2k + 1)}.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Imaginary Error Function

```Python
error_erfi(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Imaginary error function:

$$\operatorname{erfc}(z) = \frac{2}{\sqrt{\pi}} \sum_{k = 0}^{\infty} \frac{z^{2k + 1}}{k!(2k + 1)}.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Inverse Error Function

```Python
error_inverse_erf(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Inverse error function:

$$\operatorname{erf}^{-1}(z)=\sum_{k=0}^{\infty}{\frac{c_{k}}{2k+1}}({\frac{\sqrt{\pi}}{2}}z)^{2k+1}$$

where $c_{0}=1$ and:

$$c_{k}=\sum_{m=0}^{k-1}{\frac{c_{m}c_{k-1-m}}{(m+1)(2m+1)}}.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Inverse Complementary Error Function

```Python
error_inverse_erfc(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Dawson and Fresnel Integrals</summary>

### Dawson and Fresnel Integrals

#### Dawson’s Integral

```Python
dawson_integral_f(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Dawson’s integral:

$$\operatorname{F}(z)=e^{-z^{2}}\int_{0}^{z}e^{t^{2}}dt.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Sine Fresnel Integral

```Python
fresnel_integral_s(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Fresnel integral:

$$\operatorname{S}(z)=\int_{0}^{x}\sin{t^{2}}dt.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Cosine Fresnel Integral

```Python
fresnel_integral_c(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Fresnel integral:

$$\operatorname{C}(z)=\int_{0}^{x}\cos{t^{2}}dt.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Trigonometric and Hyperbolic Integrals</summary>

### Trigonometric and Hyperbolic Integrals

#### Sine Integral ($operatorname{Sin}$)

```Python
sine_integral_sin(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Sine integral:

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Sine Integral ($\operatorname{Si}$)

$$\operatorname{Sin}(z)=\int_{0}^{z}{\frac{\sin{t}}{t}}dt.$$

```Python
sine_integral_si(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Sine integral:

$$\operatorname{Si}(z)=-\int_{z}^{\infty }{\frac{\sin{t}}{t}}dt.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Cosine Integral ($\operatorname{Cin}$)

```Python
cosine_integral_cin(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Cosine integral:

$$\operatorname{Cin}(z)=\int_{0}^{z}{\frac{1-\cos{t}}{t}}dt.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Cosine Integral ($\operatorname{Ci}$)

```Python
cosine_integral_ci(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Cosine integral:

$$\operatorname{Ci}(z)=\gamma+\ln{z}-\int_{0}^{z}{\frac{1-\cos{t}}{t}}$$

where $\gamma$ is the Euler–Mascheroni constant.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Hyperbolic Sine Integral

```Python
hyperbolic_sine_integral_shi(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Hyperbolic sine integral:

$$\operatorname{Shi}(z)=\int_{0}^{z}{\frac{\sinh{t}}{t}}dt.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Hyperbolic Cosine Integral

```Python
hyperbolic_cosine_integral_chi(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Hyperbolic cosine integral:

$$\operatorname{Chi}(z)=\gamma+\ln{z}+\int_{0}^{z}{\frac{\cosh{t-1}}{t}}dt.$$

where $\gamma$ is the Euler–Mascheroni constant.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Incomplete Gamma and Related Functions</summary>

### Incomplete Gamma and Related Functions

#### Incomplete Gamma Function ($\gamma$)

```Python
lower_incomplete_gamma(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Lower incomplete gamma function:

$$\Gamma(s,z)=\int_{z}^{\infty}t^{s-1}e^{-t}dt.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Incomplete Gamma Function ($\Gamma$)

```Python
upper_incomplete_gamma(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Upper incomplete gamma function:

$$\gamma(s,z)=\int_{0}^{z}t^{s-1}e^{-t}dt.$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Incomplete Beta Function

```Python
incomplete_beta(
    z: Tensor, 
    a: Tensor, 
    b: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**b** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Airy Functions</summary>

### Airy Functions

Functions defined as the two, linearly independent solutions to:

$$y'' - yz = 0.$$

#### Airy Function of the First Kind

```Python
airy_ai(
    z: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

Airy function of the first kind:

$$\operatorname{Ai}(z)={\frac{1}{3^{\tfrac{2}{3}}\Gamma(\tfrac{2}{3})}} {_0F_1(; \tfrac{2}{3}; \tfrac{1}{9}; z^{3})} - \frac{z}{3^{\tfrac{1}{3}}\Gamma(\tfrac{1}{3})} {_0F_1(; \tfrac{4}{3}; \tfrac{1}{9}; z^{3})}$$

where $\Gamma$ is the gamma function and $_0F_1(; a; z)$ is the confluent hypergeometric limit function.

$\operatorname{Ai}(z)$ is defined for all real and complex values.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Airy Function of the Second Kind

```Python
airy_bi(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Airy function of the second kind:

$$\operatorname{Bi}(z)=\frac{_0F_1\left(;\frac{2}{3};\frac{z^3}{9}\right)}{\sqrt[6]{3} \Gamma \left(\frac{2}{3}\right)}+\frac{\sqrt[6]{3} z _0F_1\left(;\frac{4}{3};\frac{z^3}{9}\right)}{\Gamma \left(\frac{1}{3}\right)}$$

where $\Gamma$ is the gamma function and $_0F_1(; a; z)$ is the confluent hypergeometric limit function.

$\operatorname{Bi}(z)$ is defined for all real and complex values.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Derivative of the Airy Function of the First Kind

```Python
airy_ai_derivative(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Derivative of the Airy function of the first kind:

$$\operatorname{Ai}'(z)=\frac{z^2 \\,_0F_1\left(;\frac{5}{3};\frac{z^3}{9}\right)}{2\ 3^{\tfrac{2}{3}} \Gamma \left(\frac{2}{3}\right)}-\frac{\\, _0F_1\left(;\frac{1}{3};\frac{z^3}{9}\right)}{\sqrt[3]{3} \Gamma \left(\frac{1}{3}\right)},$$

where $\Gamma$ is the gamma function and $_0F_1(; a; z)$ is the confluent hypergeometric limit function.

$\operatorname{Ai}'(z)$ is defined for all real and complex values.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Derivative of the Airy Function of the Second Kind

```Python
airy_bi_derivative(
    z: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

Derivative of the Airy function of the second kind:

$$\operatorname{Bi}'(z)=\frac{z^2 \\, _0F_1\left(;\frac{5}{3};\frac{z^3}{9}\right)}{2 \sqrt[6]{3} \\, \Gamma \left(\frac{2}{3}\right)}+\frac{\sqrt[6]{3} \\, _0F_1\left(;\frac{1}{3};\frac{z^3}{9}\right)}{\Gamma \left(\frac{1}{3}\right)},$$

where $\Gamma$ is the gamma function and $_0F_1(; a; z)$ is the confluent hypergeometric limit function.

$\operatorname{Bi}'(z)$ is defined for all real and complex values.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Exponentially Scaled Airy Function of the First Kind

```Python
exp_airy_ai(
    z: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

Exponentially scaled Airy function of the first kind, $\operatorname{exp}(\operatorname{Ai}(z))$.

$\exp{(\operatorname{Ai}(z))}$ is defined for all real and complex values.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Exponentially Scaled Airy Function of the Second Kind

```Python
exp_airy_bi(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Exponentially scaled Airy function of the second kind, $\operatorname{exp}(\operatorname{Bi}(z))$.

$\exp{(\operatorname{Bi}(z))}$ is defined for all real and complex values.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Exponentially Scaled Derivative of the Airy Function of the First Kind

```Python
exp_airy_ai_derivative(
    z: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

Exponentially scaled Derivative of the Airy function of the first kind, $\operatorname{exp}(\operatorname{Ai'}(z))$.

$\exp{(\operatorname{Ai}'(z))}$ is defined for all real and complex values.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Exponentially Scaled Derivative of the Airy Function of the Second Kind

```Python
exp_airy_bi_derivative(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Exponentially scaled Derivative of the Airy function of the second kind, $\operatorname{exp}(\operatorname{Bi'}(z))$.

$\exp{(\operatorname{Bi}'(z))}$ is defined for all real and complex values.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Bessel Functions</summary>

### Bessel Functions

#### Bessel Function of the First Kind

```Python
bessel_j(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Bessel function of the first kind:

$$J_{n}(z)=\sum _{k=0}^{\infty } \frac{(-1)^k \left(\frac{z}{2}\right)^{2k+n}}{\Gamma (k+\nu +1) k!},$$

where $\Gamma$ is the gamma function.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $n$ is a number, $z$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Bessel Function of the First Kind of Order 0

```Python
bessel_j_0(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Bessel function of the first kind of order $0$, $J_{0}(z)$

$J_{0}(z)$ is defined for all real and complex $z$.

#### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Bessel Function of the First Kind of Order 1

```Python
bessel_j_1(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Bessel function of the first kind of order $1$, $J_{1}(z)$

$J_{1}(z)$ is defined for all real and complex $z$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Bessel Function of the Second Kind

```Python
bessel_y(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Bessel function of the second kind, if, and only if $\nu \notin \mathbb{Z}$:

$$Y_{\nu }(z)=\csc (\pi  \nu ) (\cos (\nu  \pi ) J_{\nu }(z)-J_{-\nu }(z)),$$

where $J_{n}(z)$ is the Bessel function of the first kind.

If $z \in \mathbb{R}$, $Y_{n}(z)$ is defined for $z > 0$. 

If $z \in \mathbb{C}$, $Y_{n}(z)$ is defined for $z \neq 0$. 

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $n$ is a number, $z$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Bessel Function of the Second Kind of Order 0

```Python
bessel_y_0(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Bessel function of the second kind of order $0$, $Y_{0}(z)$.

$Y_{0}(z)$ is defined for $\\{z \in \mathbb{R}\\}$ and $\\{z \in \mathbb{C} \mid z \neq 0\\}$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Bessel Function of the Second Kind of Order 1

```Python
bessel_y_1(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Bessel function of the second kind of order $1$, $Y_{1}(z)$.

$Y_{1}(z)$ is defined for $\\{z \in \mathbb{R} \mid z > 0\\}$ and $\\{z \in \mathbb{C} \mid z \neq 0\\}$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Hankel Functions</summary>

### Hankel Functions

#### Hankel Function of the First Kind

```Python
hankel_h_1(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Hankel function of the first kind:

$$H_{n}^{1}(z) = J_{n}(z) + i Y_{n}(z),$$

where $J_{n}(z)$ is the Bessel function of the first kind, $i$ is the imaginary unit, and $Y_{n}(z)$ is the Bessel function of the second kind.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Hankel Function of the Second Kind

```Python
hankel_h_2(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Hankel function of the second kind:

$$H_{n}^{2}(z) = J_{n}(z) - i Y_{n}(z),$$

where $J_{n}(z)$ is the Bessel function of the first kind, $i$ is the imaginary unit, and $Y_{n}(z)$ is the Bessel function of the second kind.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Modified Bessel Functions</summary>

### Modified Bessel Functions

#### Modified Bessel Function of the First Kind

```Python
modified_bessel_i(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Modified Bessel function of the first kind:

$$I_{\nu }(z)=\sum _{k=0}^{\infty } \frac{\left(\frac{z}{2}\right)^{2 k+\nu }}{\Gamma (k+\nu +1) k!}.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Modified Bessel Function of the First Kind of Order 0

```Python
modified_bessel_i_0(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Modified Bessel function of the first kind of order $0$, $I_{0}(z)$.

$I_{0}(z)$ is defined for all real and complex $z$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Modified Bessel Function of the First Kind of Order 1

```Python
modified_bessel_i_1(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Modified Bessel function of the first kind of order $1$, $I_{1}(z)$.

$I_{1}(z)$ is defined for all real and complex $z$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Modified Bessel Function of the Second Kind

```Python
modified_bessel_k(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Modified Bessel function of the second kind:

$$K_{n}(z) = \frac{1}{2} \pi i^{n + 1} H_n^{1}(i z),$$

where $i$ is the imaginary unit and $H_{n}^{1}(z)$ is the Hankel function of the first kind.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Modified Bessel Function of the Second Kind of Order 0

```Python
modified_bessel_k_0(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Modified Bessel function of the first kind of order $0$, $K_{0}(z)$.

$K_{0}(z)$ is defined for $\\{z \in \mathbb{R} \mid z > 0\\}$ and $\\{z \in \mathbb{C} \mid z \neq 0\\}$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Modified Bessel Function of the Second Kind of Order 1

```Python
modified_bessel_k_1(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Modified Bessel function of the first kind of order $1$, $K_{1}(z)$.

$K_{1}(z)$ is defined for $\\{z \in \mathbb{R} \mid z > 0\\}$ and $\\{z \in \mathbb{C} \mid z \neq 0\\}$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Spherical Bessel Functions</summary>

### Spherical Bessel Functions

#### Spherical Bessel Function of the First Kind

```Python
spherical_bessel_j(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Spherical Bessel function of the first kind:

$$j_{n}(x)={\sqrt{\frac{\pi}{2x}}}J_{n + {\frac{1}{2}}}(x).$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Spherical Bessel Function of the First Kind of Order 0

```Python
spherical_bessel_j_0(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Spherical Bessel function of the first kind:

$$j_{n}(x)={\sqrt{\frac{\pi}{2x}}}J_{n + {\frac{1}{2}}}(x)$$

where $n = 0$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Spherical Bessel Function of the First Kind of Order 1

```Python
spherical_bessel_j_1(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Spherical Bessel function of the first kind:

$$j_{n}(x)={\sqrt{\frac{\pi}{2x}}}J_{n + {\frac{1}{2}}}(x)$$

where $n = 1$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Spherical Bessel Function of the Second Kind

```Python
spherical_bessel_y(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Spherical Bessel function of the second kind:

$$y_{n}(x)={\sqrt{\frac{\pi}{2x}}}Y_{n+{\frac{1}{2}}}(x).$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Spherical Bessel Function of the Second Kind of Order 0

```Python
spherical_bessel_y_0(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Spherical Bessel function of the second kind:

$$y_{n}(x)={\sqrt{\frac{\pi}{2x}}}Y_{n+{\frac{1}{2}}}(x)$$

where $n = 0$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Spherical Bessel Function of the Second Kind of Order 1

```Python
spherical_bessel_y_1(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Spherical Bessel function of the second kind:

$$y_{n}(x)={\sqrt{\frac{\pi}{2x}}}Y_{n+{\frac{1}{2}}}(x)$$

where $n = 1$.

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Spherical Hankel Functions</summary>

### Spherical Hankel Functions

#### Spherical Hankel Function of the First Kind

```Python
spherical_hankel_h_1(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Spherical Hankel function of the first kind:

$$h_{n}^{1}(z)=\frac{\sqrt{\frac{\pi}{2}}H_{n +\frac{1}{2}}^{1}(z)}{\sqrt{z}},$$

where $H_{n}^{1}$ is the Hankel function of the first kind.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Spherical Hankel Function of the Second Kind

```Python
spherical_hankel_h_2(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Spherical Hankel function of the second kind:

$$h_{n}^{2}(z)=\frac{\sqrt{\frac{\pi}{2}}H_{n+\frac{1}{2}}^{2}(z)}{\sqrt{z}},$$

where $H_{n}^{2}$ is the Hankel function of the second kind.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Modified Spherical Bessel Functions</summary>

### Modified Spherical Bessel Functions

#### Modified Spherical Bessel Function of the First Kind

```Python
modified_spherical_bessel_i(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Modified spherical Bessel function of the first kind:

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Modified Spherical Bessel Function of the First Kind of Order 0

```Python
modified_spherical_bessel_i_0(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Modified Spherical Bessel Function of the First Kind of Order 1

```Python
modified_spherical_bessel_i_1(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Modified Spherical Bessel Function of the Second Kind

```Python
modified_spherical_bessel_k(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Modified Spherical Bessel Function of the Second Kind of Order 0

```Python
modified_spherical_bessel_k_0(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Modified Spherical Bessel Function of the Second Kind of Order 1

```Python
modified_spherical_bessel_k_1(
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Kelvin Functions</summary>

### Kelvin Functions

#### Kelvin Function of the First Kind ($\operatorname{ber}$)

```Python
kelvin_ber(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

$$\mathrm {ber} _{n}(x)=({\frac {x}{2}})^{n}\sum _{k\geq 0}{\frac {\cos [({\frac {3n}{4}}+{\frac {k}{2}})\pi ]}{k!\Gamma (n+k+1)}}({\frac {x^{2}}{4}})^{k}.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Kelvin Function of the First Kind ($\operatorname{bei}$)

```Python
kelvin_bei(
    n: Tensor,
    z: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Kelvin Function of the Second Kind ($\operatorname{kei}$)

```Python
kelvin_kei(
    n: Tensor,
    z: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Kelvin Function of the Second Kind ($\operatorname{ker}$)

```Python
kelvin_ker(
    n: Tensor,
    z: Tensor,
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Struve and Modified Struve Functions</summary>

### Struve and Modified Struve Functions

#### Struve Function

```Python
struve_h(
    n: Tensor, 
    z: Tensor, 
    *,
    out: Optional[Tensor] = None,
) -> Tensor
```

Struve function:

$$\mathbf{H}_{n}(z)=\left(\frac{z}{2}\right)^{n+1} \sum _{k=0}^{\infty } \frac{(-1)^k \left(\frac{z}{2}\right)^{2 k}}{\Gamma \left(k+\frac{3}{2}\right) \Gamma \left(k+n+\frac{3}{2}\right)},$$

where $\Gamma(z)$ is the gamma function.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Modified Struve Function

```Python
modified_struve_l(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Modified Struve function:

$$\mathbf{L}_{n}(z)=\left(\frac{z}{2}\right)^{n+1} \sum _{k=0}^{\infty } \frac{\left(\frac{z}{2}\right)^{2 k}}{\Gamma \left(k+\frac{3}{2}\right) \Gamma \left(k+n+\frac{3}{2}\right)},$$

where $\Gamma(z)$ is the gamma function.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Lommel Functions</summary>

### Lommel Functions

Functions defined as the solutions to the Lommel differential equation:

$$z^{2}{\frac{d^{2}y}{dz^{2}}}+z{\frac{dy}{dz}}+(z^{2}-n^{2})y=z^{m+1}.$$

#### Lommel Function of the First Kind

#### Lommel Function of the Second Kind
</details>

<details>
<summary>Anger and Weber Functions</summary>

### Anger and Weber Functions

#### Anger Function

```Python
anger_j(
    n: Tensor,
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Anger function:

$$\mathbf{J}_{n}(z)={\frac{1}{\pi}}\int_{0}^{\pi}\cos(n\theta-z\sin\theta)d\theta.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Weber Function

```Python
weber_e(
    n: Tensor,
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Weber function:

$$\mathbf{E}_{n}(z)={\frac{1}{\pi}}\int_{0}^{\pi}\sin(n\theta-z\sin \theta )d\theta.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Parabolic Cylinder Functions</summary>

### Parabolic Cylinder Functions

#### Parabolic Cylinder Function

```Python
parabolic_cylinder_d(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None
) -> Tensor
```

Parabolic cylinder function:

$$D_{n}(z) = \sqrt{\pi} 2^{\tfrac{n}{2}} e^{-\frac{z^2}{4}} \left(\frac{ _1F_1(-\frac{n }{2}; \frac{1}{2}; \frac{z^2}{2})}{\Gamma (\frac{1-n }{2})}-\frac{\sqrt{2} z  _1F_1(\frac{1-n }{2};\frac{3}{2};\frac{z^2}{2})}{\Gamma (-\frac{n }{2})} \right),$$

where $_1F_1$ is the confluent hypergeometric function of the first kind and $\Gamma$ is the gamma function.
</details>

<details>
<summary>Confluent Hypergeometric Functions</summary>

### Confluent Hypergeometric Functions

#### Confluent Hypergeometric Function of the First Kind

```Python
confluent_hypergeometric_1_f_1(
    a: Tensor,
    b: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Confluent hypergeometric function of the first kind:

$$_{1}{F}_{1}(a; b; z) = \sum_{k = 0}^{\infty} \frac{a_{k}}{b_{k}} \frac{z^{k}}{k!},$$

where $a_{k}$ and $b_{k}$ are rising factorials.

#### Confluent Hypergeometric Function of the Second Kind
</details>

<details>
<summary>Whittaker Functions</summary>

### Whittaker Functions

Whittaker functions are defined as a special solution of Whittaker’s equation:

$${\frac  {d^{2}w}{dz^{2}}}+(-{\frac  {1}{4}}+{\frac  {\kappa }{z}}+{\frac  {1/4-\mu ^{2}}{z^{2}}})w=0.$$

#### Whittaker Function ($M_{\kappa, \mu}$)

```Python
whittaker_m(
    k: Tensor,
    m: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Whittaker function:

$${\displaystyle M_{\kappa ,\mu }(z)=\exp (-z/2)z^{\mu +{\tfrac {1}{2}}}M(\mu -\kappa +{\tfrac {1}{2}},1+2\mu ,z)}.$$

##### Parameters

**k** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Whittaker Function ($W_{\kappa, \mu}$)

```Python
whittaker_w(
    k: Tensor,
    m: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Whittaker function:

$${\displaystyle W_{\kappa ,\mu }(z)=\exp (-z/2)z^{\mu +{\tfrac {1}{2}}}U(\mu -\kappa +{\tfrac {1}{2}},1+2\mu ,z).}$$

##### Parameters

**k** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Legendre Functions</summary>

### Legendre Functions

#### Legendre Function of the First Kind

```Python
legendre_p(
    n: Tensor,
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Legendre function of the first kind:

$$P_{n}(z)=\operatorname{_2F_2}\left(-n,n+1;1;\frac{1-z}{2}\right),$$

where $\operatorname{_2F_2}$ is the generalized hypergeometric function.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Legendre Function of the Second Kind

```Python
legendre_q(
    n: Tensor,
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Legendre function of the second kind:

$$Q_{n}(z)=Q_{n}^0(z),$$

where $Q_{n}^m$ is the associated Legendre function of the second kind.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – order. If $z$ is a number, $n$ must be a tensor.

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – input. If $z$ is a number, $n$ must be a tensor.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Associated Legendre Functions</summary>

### Associated Legendre Functions

#### Associated Legendre Function of the First Kind

```Python
associated_legendre_p(
    n: Tensor,
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Associated Legendre function of the first kind:

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – 

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Associated Legendre Function of the Second Kind

```Python
associated_legendre_q(
    n: Tensor,
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Associated Legendre function of the second kind:

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) – 

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Ferrers Functions</summary>

### Ferrers Functions

#### Ferrers Function of the First Kind

#### Ferrers Function of the Second Kind
</details>

<details>
<summary>Appell Functions</summary>

### Appell Functions

#### Appell Function $\left(F_{1}\right)$

```Python
appell_f_1(
    a: Tensor, 
    b: Tensor,
    c: Tensor,
    d: Tensor,
    x: Tensor,
    y: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Appell function:

$$F_{1}(a; b, c; d; x, y) = \sum_{m, n=0}^{\infty}{\frac{a_{m + n}b_{m}c_{n}}{d_{m + n}m!n!}}x^{m}y^{n}.$$

##### Parameters

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**b** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**c** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**y** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Appell Function $\left(F_{2}\right)$

```Python
appell_f_2(
    a: Tensor, 
    b: Tensor,
    c: Tensor,
    d: Tensor,
    e: Tensor,
    x: Tensor,
    y: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**b** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**c** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**y** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Appell Function $\left(F_{3}\right)$

```Python
appell_f_3(
    a: Tensor, 
    b: Tensor,
    c: Tensor,
    d: Tensor,
    e: Tensor,
    x: Tensor,
    y: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**b** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**c** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**y** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Appell Function $\left(F_{4}\right)$

```Python
appell_f_4(
    a: Tensor, 
    b: Tensor,
    c: Tensor,
    d: Tensor,
    x: Tensor,
    y: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**b** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**c** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**y** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>$q$-Hypergeometric and Related Functions</summary>

### $q$-Hypergeometric and Related Functions

#### $q$-Factorial

#### $q$-Binomial Coefficient

#### $q$-Gamma Function

#### $q$-Digamma Function

#### $q$-Polygamma Function
</details>

<details>
<summary>Orthogonal Polynomials</summary>

### Orthogonal Polynomials

#### Legendre Polynomial

#### Laguerre Polynomial
</details>

<details>
<summary>Chebyshev Polynomials</summary>

### Chebyshev Polynomials

#### Chebyshev Polynomial of the First Kind

```Python
chebyshev_polynomial_t(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Chebyshev polynomial of the first kind, if, and only if, $n \in \mathbb{N} \wedge n \geq 0$:

$$T_n(z)=\frac{\delta _{n,0}}{2}+\frac{1}{2} n \sum _{k=1}^{\left\lfloor\frac{n}{2}\right\rfloor } \frac{(-1)^k (n-k-1)! (2 z)^{n-2 k}}{k! (n-2 k)!}+2^{n-1}z^n,$$

where $\delta$ is the Kronecker delta function.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Chebyshev Polynomial of the Second Kind

```Python
chebyshev_polynomial_u(
    n: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Chebyshev polynomial of the second kind, if, and only if, $n \in \mathbb{N} \wedge n \geq 0$:

$$U_n(z)=\sum _{k=0}^{\left\lfloor \frac{n}{2}\right\rfloor } \frac{(-1)^k (n-k)! (2 z)^{n-2 k}}{k! (n-2 k)!}.$$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Chebyshev Polynomial of the Third Kind

```Python
chebyshev_polynomial_v(
    n: Tensor, 
    x: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Chebyshev polynomial of the third kind, $V_{n}(x).$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Chebyshev Polynomial of the Fourth Kind

```Python
chebyshev_polynomial_w(
    n: Tensor, 
    x: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Chebyshev polynomial of the fourth kind, $W_{n}(x).$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Shifted Chebyshev Polynomials</summary>

### Shifted Chebyshev Polynomials

#### Shifted Chebyshev Polynomial of the First Kind

```Python
shifted_chebyshev_polynomial_t(
    n: Tensor, 
    x: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Shifted Chebyshev polynomial of the first kind, $T_{n}^{\ast}(x).$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Shifted Chebyshev Polynomial of the Second Kind

```Python
shifted_chebyshev_polynomial_u(
    n: Tensor, 
    x: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Shifted Chebyshev polynomial of the second kind, $U_{n}^{\ast}(x).$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Shifted Chebyshev Polynomial of the Third Kind

```Python
shifted_chebyshev_polynomial_v(
    n: Tensor, 
    x: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Shifted Chebyshev polynomial of the third kind, $V_{n}^{\ast}(x).$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Shifted Chebyshev Polynomial of the Fourth Kind

```Python
shifted_chebyshev_polynomial_w(
    n: Tensor, 
    x: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Shifted Chebyshev polynomial of the fourth kind, $W_{n}^{\ast}(x).$

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Hermite Polynomials</summary>

### Hermite Polynomials

#### Probabilist’s Hermite Polynomial

```Python
hermite_polynomial_he(
    n: Tensor, 
    x: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Probabilist’s Hermite polynomial:

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Physicist’s Hermite Polynomial

```Python
hermite_polynomial_h(
    n: Tensor, 
    x: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Physicist’s Hermite polynomial:

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Legendre Forms of Elliptic Integrals</summary>

### Legendre Forms of Elliptic Integrals

#### Elliptic Integral of the First Kind

```Python
legendre_elliptic_integral_f(
    m: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Legendre form of the incomplete elliptic integral of the first kind:

The incomplete elliptic integral is defined in terms of the parameter $m$ instead of the elliptic modulus $k$. $m$ is defined as $m = k^{2}$.

##### Parameters

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Elliptic Integral of the Second Kind

```Python
legendre_elliptic_integral_e(
    m: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Legendre form of the incomplete elliptic integral of the second kind:

The incomplete elliptic integral is defined in terms of the parameter $m$ instead of the elliptic modulus $k$. $m$ is defined as $m = k^{2}$.

##### Parameters

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Elliptic Integral of the Third Kind

```Python
legendre_elliptic_integral_pi(
    n: Tensor, 
    m: Tensor, 
    z: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Legendre form of the incomplete elliptic integral of the third kind:

The incomplete elliptic integral is defined in terms of the parameter $m$ instead of the elliptic modulus $k$. $m$ is defined as $m = k^{2}$.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Legendre Forms of Complete Elliptic Integrals</summary>

### Legendre Forms of Complete Elliptic Integrals

#### Complete Elliptic Integral of the First Kind

```Python
complete_legendre_elliptic_integral_k(
    m: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Legendre form of the complete elliptic integral of the first kind:

$$K(m) = F({\tfrac{\pi}{2}}, m).$$

The complete elliptic integral is defined in terms of the parameter $m$ instead of the elliptic modulus $k$. $m$ is defined as $m = k^{2}$.

##### Parameters

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Complete Elliptic Integral of the Second Kind

```Python
complete_legendre_elliptic_integral_e(
    m: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Legendre form of the complete elliptic integral of the second kind:

$$E(m) = \int_{0}^{\tfrac{\pi}{2}}{\sqrt{1 - m \sin^{2} \theta}}d\theta.$$

The complete elliptic integral is defined in terms of the parameter $m$ instead of the elliptic modulus $k$. $m$ is defined as $m = k^{2}$.

##### Parameters

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Complete Elliptic Integral of the Third Kind

```Python
complete_legendre_elliptic_integral_pi(
    n: Tensor, 
    m: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Legendre form of the complete elliptic integral of the third kind:

$$\Pi(n, m) = \int_{0}^{\frac{\pi}{2}}{\frac{d\theta}{(1 - n\sin^{2}\theta){\sqrt{1 - m\sin ^{2}\theta }}}}.$$

The complete elliptic integral is defined in terms of the parameter $m$ instead of the elliptic modulus $k$. $m$ is defined as $m = k^{2}$.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Carlson Symmetric Forms of Elliptic Integrals</summary>

### Carlson Symmetric Forms of Elliptic Integrals

#### Carlson Elliptic Integral $\left(R_{C}\right)$

```Python
carlson_elliptic_integral_r_c(
    x: Tensor, 
    y: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**y** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Carlson Elliptic Integral $\left(R_{D}\right)$

```Python
carlson_elliptic_integral_r_d(
    x: Tensor, 
    y: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**y** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Carlson Elliptic Integral $\left(R_{E}\right)$

```Python
carlson_elliptic_integral_r_e(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Carlson Elliptic Integral $\left(R_{F}\right)$

```Python
carlson_elliptic_integral_r_f(
    x: Tensor, 
    y: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**y** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Carlson Elliptic Integral $\left(R_{G}\right)$

```Python
carlson_elliptic_integral_r_g(
    x: Tensor, 
    y: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**y** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Carlson Elliptic Integral $\left(R_{J}\right)$

```Python
carlson_elliptic_integral_r_j(
    x: Tensor, 
    y: Tensor,
    z: Tensor,
    p: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**y** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Carlson Elliptic Integral $\left(R_{K}\right)$

```Python
carlson_elliptic_integral_r_k(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Carlson Elliptic Integral $\left(R_{M}\right)$

```Python
carlson_elliptic_integral_r_m(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```
</details>

<details>
<summary>Theta Functions</summary>

### Theta Functions

#### Theta Function $\left(\vartheta_1\right)$

```Python
theta_1(
    z: Tensor,
    q: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Theta function:

$$\vartheta _1(z,q)=2 \sqrt[4]{q} \sum _{k=0}^{\infty } (-1)^k q^{k (k+1)} \sin ((2 k+1) z).$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Derivative of Theta Function $\left(\vartheta_1'\right)$

```Python
theta_1_prime(
    z: Tensor,
    q: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Derivative of theta function:

$$\vartheta_1^{\prime}(z,q)=2 \sqrt[4]{q} \sum _{k=0}^{\infty } (-1)^k q^{k (k+1)} (2 k+1) \cos ((2 k+1) z).$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Theta Function $\left(\vartheta_2\right)$

```Python
theta_2(
    z: Tensor,
    q: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Theta function:

$$\vartheta _2(z,q)=2 \sqrt[4]{q} \sum _{k=0}^{\infty } q^{k (k+1)} \cos ((2 k+1) z).$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Derivative of Theta Function $\left(\vartheta_2'\right)$

```Python
theta_2_prime(
    z: Tensor,
    q: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Derivative of theta function:

$$\vartheta _2^{\prime }(z,q)=-2 \sqrt[4]{q} \sum _{k=0}^{\infty } q^{k (k+1)} (2 k+1) \sin ((2 k+1) z).$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Theta Function $\left(\vartheta_3\right)$

```Python
theta_3(
    z: Tensor,
    q: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Theta function:

$$\vartheta _3(z,q)=1+2 \sum _{k=1}^{\infty } q^{k^2} \cos (2 k z).$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Derivative of Theta Function $\left(\vartheta_3'\right)$

```Python
theta_3_prime(
    z: Tensor,
    q: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Derivative of theta function:

$$\vartheta _3^{\prime }(z,q)=-4 \sum _{k=1}^{\infty } q^{k^2} k \sin (2 k z).$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Theta Function $\left(\vartheta_4\right)$

```Python
theta_4(
    z: Tensor,
    q: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Theta function:

$$\vartheta _4(z,q)=1+2 \sum _{k=1}^{\infty } (-1)^k q^{k^2} \cos (2 k z).$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Jacobi Elliptic and Related Functions</summary>

#### Derivative of Theta Function $\left(\vartheta_4'\right)$

```Python
theta_4_prime(
    z: Tensor,
    q: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Derivative of theta function:

$$\vartheta_1^{\prime}(z,q)=2 \sqrt[4]{q} \sum _{k=0}^{\infty } (-1)^k q^{k (k+1)} (2 k+1) \cos ((2 k+1) z).$$

##### Parameters

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

### Jacobi Elliptic and Related Functions

#### Jacobi Amplitude Function

```Python
jacobi_amplitude_am(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Jacobi Elliptic Function $\left(\operatorname{sn}\right)$

```Python
jacobi_elliptic_sn(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

$$\operatorname{sn}(z \mid m) = \sin(\operatorname{am}(z \mid m))$$

#### Jacobi Elliptic Function $\left(\operatorname{cn}\right)$

```Python
jacobi_elliptic_cn(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

$$\operatorname{cn}(z \mid m) = \cos(\operatorname{am}(z \mid m))$$

#### Jacobi Elliptic Function $\left(\operatorname{dn}\right)$

```Python
jacobi_elliptic_dn(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Jacobi Elliptic Function $\left(\operatorname{sd}\right)$

```Python
jacobi_elliptic_sd(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Jacobi Elliptic Function $\left(\operatorname{cd}\right)$

```Python
jacobi_elliptic_cd(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

$$\operatorname{cd}(z \mid m) = \frac{\operatorname{cn}(z \mid m)}{\operatorname{dn}(z \mid m)}$$

#### Jacobi Elliptic Function $\left(\operatorname{sc}\right)$

```Python
jacobi_elliptic_sc(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```
</details>

<details>
<summary>Inverse Jacobi Elliptic Functions</summary>

### Inverse Jacobi Elliptic Functions

#### Inverse Jacobi Elliptic Function $\left(\operatorname{sn}\right)$

```Python
inverse_jacobi_elliptic_sn(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Inverse Jacobi Elliptic Function $\left(\operatorname{cn}\right)$

```Python
inverse_jacobi_elliptic_cn(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Inverse Jacobi Elliptic Function $\left(\operatorname{dn}\right)$

```Python
inverse_jacobi_elliptic_dn(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Inverse Jacobi Elliptic Function $\left(\operatorname{sd}\right)$

```Python
inverse_jacobi_elliptic_sd(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Inverse Jacobi Elliptic Function $\left(\operatorname{cd}\right)$

```Python
inverse_jacobi_elliptic_cd(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Inverse Jacobi Elliptic Function $\left(\operatorname{sc}\right)$

```Python
inverse_jacobi_elliptic_sc(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```
</details>

<details>
<summary>Weierstrass Elliptic Functions</summary>

### Weierstrass Elliptic Functions

#### Weierstrass Elliptic Function (p)

#### Weierstrass Elliptic Function (\zeta)

#### Weierstrass Elliptic Function (\sigma)
</details>

<details>
<summary>Inverse Weierstrass Elliptic Functions</summary>

### Inverse Weierstrass Elliptic Functions

#### Inverse Weierstrass Elliptic Function (p)

#### Inverse Weierstrass Elliptic Function (q)
</details>

<details>
<summary>Modular Functions</summary>

### Modular Functions

#### Elliptic Function (\lambda)

#### Klein’s Complete Invariant Function
</details>

<details>
<summary>Bernoulli Number and Polynomial</summary>

### Bernoulli Number and Polynomial

#### Bernoulli Number

```Python
bernoulli_number_b(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

$n^{\text{th}}$ Bernoulli number, $B_{n}$.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Bernoulli Polynomial

```Python
bernoulli_polynomial_b(
    n: Tensor, 
    x: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Bernoulli polynomial, $B_{n}(x)$.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**x** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Euler Number and Polynomial</summary>

### Euler Number and Polynomial

#### Euler Number

```Python
euler_number_e(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

$n^{\text{th}}$ Euler number, $E_{n}$.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Euler Polynomial

```Python
euler_polynomial_e(
    n: Tensor, 
    x: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Euler polynomial, $E_{n}(x)$.

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Zeta and Related Functions</summary>

### Zeta and Related Functions

#### Riemann Zeta Function

```Python
riemann_zeta(
    s: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Riemann zeta function:

$$\zeta(s)=\sum _{n=1}^{\infty}{\frac{1}{n^{s}}}.$$

##### Parameters

**s** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – input.

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Hurwitz Zeta Function

```Python
hurwitz_zeta(
    s: Tensor,
    a: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Hurwitz zeta function:

$$\zeta(s, a) = \sum _{n = 0}^{\infty}{\frac{1}{(n + a)^{s}}}.$$

##### Parameters

**s** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Polylogarithm

```Python
polylogarithm_li(
    s: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Polylogarithm:

$$\operatorname{Li}_{s + 1}(z ) = \int_{0}^{z}{\frac{\operatorname{Li}_{s}(t)}{t}}dt.$$

##### Parameters

**s** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Lerch Zeta Function

```Python
lerch_zeta_l(
    l: Tensor,
    z: Tensor,
    a: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**l** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Lerch Transcendent

#### Dirichlet L-Function
</details>

<details>
<summary>Multiplicative Number Theoretic Functions</summary>

### Multiplicative Number Theoretic Functions

#### Prime Number

```Python
prime_number_p(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

$n^{\text{th}}$ prime number, $p(n)$.

#### Euler’s Totient Function

```Python
euler_totient_phi(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Divisor Function

#### Jordan’s Totient Function

#### Möbius Function

```Python
mobius_mu(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Liouville Function

```Python
liouville_lambda(
    n: Tensor, 
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```
</details>

<details>
<summary>Matthieu Characteristic Values</summary>

### Mathieu Characteristic Values

#### Matthieu Characteristic Value $\left(a\right)$

```Python
matthieu_characteristic_a(
    r: Tensor,
    q: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

#### Matthieu Characteristic Value $\left(b\right)$

```Python
matthieu_characteristic_b(
    r: Tensor,
    q: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```
</details>

<details>
<summary>Angular Matthieu Functions</summary>

### Angular Mathieu Functions

#### Angular Mathieu Function ($\operatorname{ce}$)

Even Mathieu function:

#### Angular Mathieu Function ($\operatorname{se}$)
</details>

<details>
<summary>Radial Mathieu Functions</summary>

### Radial Mathieu Functions

#### Radial Matthieu Function ($\operatorname{M}c$)

#### Radial Matthieu Function ($\operatorname{M}s$)
</details>

<details>
<summary>Lamé Functions</summary>

### Lamé Functions

#### Lamé Function $\left(Ec_{n}^{j}\right)$

```Python
lame_ec(
    n: Tensor,
    j: Tensor,
    z: Tensor,
    m: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**j** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Derivative of the Lamé Function $\left(Ec_{n}^{j}\right)$

```Python
lame_ec_prime(
    n: Tensor,
    j: Tensor,
    z: Tensor,
    m: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**j** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Lamé Function $\left(Es_{n}^{j}\right)$

```Python
lame_es(
    n: Tensor,
    j: Tensor,
    z: Tensor,
    m: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**j** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Derivative of the Lamé Function $\left(Es_{n}^{j}\right)$

```Python
lame_es_prime(
    n: Tensor,
    j: Tensor,
    z: Tensor,
    m: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**n** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**j** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

**m** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Heun Functions</summary>

### Heun Functions

#### Heun Function

```Python
heun(
    p: Tensor,
    q: Tensor,
    a: Tensor,
    b: Tensor,
    g: Tensor,
    d: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Heun function:

##### Parameters

**p** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**b** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**g** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Derivative of the Heun Function

```Python
heun_prime(
    p: Tensor,
    q: Tensor,
    a: Tensor,
    b: Tensor,
    g: Tensor,
    d: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Derivative of the Heun function:

##### Parameters

**p** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**b** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**g** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Confluent Heun Function

```Python
confluent_heun(
    q: Tensor,
    a: Tensor,
    g: Tensor,
    d: Tensor,
    e: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Confluent Heun function:

##### Parameters

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**g** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Derivative of the Confluent Heun Function

```Python
confluent_heun_prime(
    q: Tensor,
    a: Tensor,
    g: Tensor,
    d: Tensor,
    e: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Derivative of the confluent Heun function:

##### Parameters

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**g** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Doubly-Confluent Heun Function

```Python
confluent_heun(
    q: Tensor,
    a: Tensor,
    g: Tensor,
    d: Tensor,
    e: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Doubly-confluent Heun function:

##### Parameters

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**g** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Derivative of the Doubly-Confluent Heun Function

```Python
confluent_heun_prime(
    q: Tensor,
    a: Tensor,
    g: Tensor,
    d: Tensor,
    e: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Derivative of the doubly-confluent Heun function:

##### Parameters

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**g** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Bi-Confluent Heun Function

```Python
biconfluent_heun(
    q: Tensor,
    a: Tensor,
    g: Tensor,
    d: Tensor,
    e: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Bi-confluent Heun function:

##### Parameters

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**g** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Derivative of the Bi-Confluent Heun Function

```Python
biconfluent_heun_prime(
    q: Tensor,
    a: Tensor,
    g: Tensor,
    d: Tensor,
    e: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Derivative of the bi-confluent Heun function:

##### Parameters

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**g** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Tri-Confluent Heun Function

```Python
triconfluent_heun(
    q: Tensor,
    a: Tensor,
    g: Tensor,
    d: Tensor,
    e: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Tri-confluent Heun function:

##### Parameters

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**g** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Derivative of the Tri-Confluent Heun Function

```Python
triconfluent_heun_prime(
    q: Tensor,
    a: Tensor,
    g: Tensor,
    d: Tensor,
    e: Tensor,
    z: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

Derivative of the tri-confluent Heun function:

##### Parameters

**q** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**a** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**g** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**d** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**z** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

<details>
<summary>Coulomb Wave Functions</summary>

#### Coulomb Wave Function

```Python
coulomb_f(
    l: Tensor,
    e: Tensor,
    r: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**l** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**r** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Irregular Coulomb Wave Function

```Python
coulomb_g(
    l: Tensor,
    e: Tensor,
    r: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**l** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**r** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Outgoing Irregular Coulomb Wave Function

```Python
coulomb_h_positive(
    l: Tensor,
    e: Tensor,
    r: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**l** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**r** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.

#### Incoming Irregular Coulomb Wave Function

```Python
coulomb_h_negative(
    l: Tensor,
    e: Tensor,
    r: Tensor,
    *, 
    out: Optional[Tensor] = None,
) -> Tensor
```

##### Parameters

**l** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**e** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

**r** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor) *or Number*) –

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
</details>

## Prior Art

### Cephes Mathematical Library

Suite of elementary and special functions written by Stephen L. Moshier. It was initially created as a supplement to his numerical analysis textbook, “Methods and Programs for Mathematical Functions.” Unsure of the original publication date, but I estimate somewhere between 1984 and 1986. It was most recently updated, by Moshier, in 2018. Its use is ubiquitous (it’s even presently used in PyTorch), and DeepMind recently maintained a library wrapper for PyTorch.

### specfun

Suite of elementary and special functions authored by W. J. Cody, a numerical analysis pioneer, written in Fortran. It’s abandonware.

### Wolfram Language

General-purpose, commercial, multi-paradigm programming language written and maintained by Wolfram Research. The Wolfram Language touts one of the best suites of elementary and special functions. Most of these functions support either symbolic, or relevant to PyTorch, numerical evaluation and differentiation.

### MATLAB
### International Mathematics and Statistics Library (IMSL)
### NAG Numerical Library
### GNU Octave
### GNU Scientific Library (GSL)

### SciPy

SciPy is a free and open-source Python package for scientific computing. It contains modules for linear algebra, integration, optimization, etc. SciPy also contains a robust suite of special functions. Most of the special function implementations rely on third-party packages, many featured elsewhere in this subsection (e.g., Cephes and specfun).
