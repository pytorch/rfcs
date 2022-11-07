<details>
<summary>Instructions - click to expand</summary>

- Fork the rfcs repo: https://github.com/pytorch/rfcs
- Copy `RFC-0000-template.md` to `RFC-00xx-my-feature.md`, or write your own open-ended proposal. Put care into the details.
- Submit a pull request titled `RFC-00xx-my-feature`. 
    - Assign the `draft` label while composing the RFC. You may find it easier to use a WYSIWYG editor (like Google Docs) when working with a few close collaborators; feel free to use whatever platform you like. Ideally this document is publicly visible and is linked to from the PR.
    - When opening the RFC for general discussion, copy your document into the `RFC-00xx-my-feature.md` file on the PR and assign the `commenting` label.
- Build consensus for your proposal, integrate feedback and revise it as needed, and summarize the outcome of the discussion via a [resolution template](https://github.com/pytorch/rfcs/blob/rfc-process/RFC-0000-template.md#resolution).
    - If the RFC is idle here (no activity for 2 weeks), assign the label `stalled` to the PR.
- Once the discussion has settled, assign a new label based on the level of support:
    - `accepted` if a decision has been made in the RFC
    - `draft` if the author needs to rework the RFC’s proposal
    - `shelved` if there are no plans to move ahead with the current RFC’s proposal. We want neither to think about evaluating the proposal
nor about implementing the described feature until some time in the future.
- A state of `accepted` means that the core team has agreed in principle to the proposal, and it is ready for implementation. 
- The author (or any interested developer) should next open a tracking issue on Github corresponding to the RFC.
    - This tracking issue should contain the implementation next steps. Link to this tracking issue on the RFC (in the Resolution > Next Steps section)
- Once all relevant PRs are merged, the RFC’s status label can be finally updated to `closed`.
</details>

# Automatic Implicit Differentiation

_Author’s note—This RFC is a work-in-progress._

## Authors

* Allen Goodman ([@0x00b1](https://github.com/0x00b1))

## Summary

Combine the 
[implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem) 
and automatic differentation to unify and expand PyTorch optimization module 
([torch.optim](https://pytorch.org/docs/stable/optim.html)).

### Example

Users define an objective function to capture the optimality conditions of their 
problem:

```Python
import functorch
from torch import Tensor


def f(x: Tensor, y: Tensor) -> Tensor:
    …

g = functorch.grad(objective, argnums=0)
```

*`torch.optim` would provide a robust set of operations to easily express `g`.*

`g` is passed to the `@torch.optim.root` decorator and the decorator is prepended to a differentiable solver:

```Python
@torch.optim.root(function)
def h(x: Tensor, y: Tensor) -> Tensor:
    …
```

PyTorch will combine the implicit function theorem and automatic differentation of `g` to automatically differentiate the solution:

```Python
functorch.jacrev(h, argnums=0)(y, 10.0)
```

*`torch.optim` would provide a robust set of solvers*

## Motivation

Automatic differentiation encourages PyTorch users to express complex computations by creatively composing elementary computations, removing the tediousness of manually computing derivatives. It’s *the* fundamental feature of PyTorch. Meanwhile, the differentiation of optimization solutions has become fundamental to machine learning practitioners. A prominent example is bi-level optimization, computing the derivatives of an inner optimization problem to solve an outer one. Applications in machine learning include hyper-parameter optimization, neural networks, and meta learning. Unfortunately, optimization solutions usually do not enjoy an explicit formula for their inputs, so these functions cannot use automatic differentiation. 

Two strategies have been commonly used in recent years to circumvent this problem. 

1. The first strategy, unrolling the iterations of an optimization method using the final iteration as a proxy for the solution, enables the explicit construction of an automatically differentiable graph. But, unrolling requires reimplementing the optimization method using automatically differentiable operators, and many algorithms are unfriendly to automatic differentiation. Furthermore, forward-mode automatic differentiation has a time complexity that scales linearly with the number of variables, and reverse-mode automatic differentiation has a memory complexity that scales linearly with the number of iterations. 

2. The second strategy, implicitly relating an optimization solution to its inputs using optimality conditions, is comparatively advantageous since reimplementation is unnecessary. In machine learning, implicit differentiation is successfully used in stationarity conditions, Karush–Kuhn–Tucker (KKT) conditions, and the proximal gradient methods. Yet, implicit differentiation has, so far, remained difficult to use for practitioners, as it requires case-by-case, tedious mathematical derivation and implementation.

Here, I'll propose the adoption of a third strategy, automatic implicit differentiation, an approach that adds implicit differentiation to existing optimization methods. First, the practitioner defines a mapping, $F$, that captures the optimality conditions of the problem solved by the algorithm. Next, the automatic differentiation of $F$ is combined with the implicit function theorem to differentiate the optimization solution automatically. This strategy is generic yet exploits the efficiency of state-of-the-art optimization methods. Moreover, it combines the benefits of both implicit differentiation and automatic differentiation. It could achieve for optimization solutions what automatic differentiation did for computational graphs. 

This strategy was first summarized in the following paper:

```bibtex
@article{jaxopt_implicit_diff,
  title={Efficient and Modular Implicit Differentiation},
  author={Blondel, Mathieu and Berthet, Quentin and Cuturi, Marco and Frostig, Roy
   and Hoyer, Stephan and Llinares-L{\'o}pez, Felipe and Pedregosa, Fabian
   and Vert, Jean-Philippe},
  journal={arXiv preprint arXiv:2105.15183},
  year={2021}
}
```

It is _the_ primary reference for this proposal.

## Notation

Hessian of $f$: $\mathbb{R}^{d}\rightarrow\mathbb{R}$ evaluated at $x\in\mathbb{R}^{d}$ by $\nabla f\left(x\right)\in\mathbb{R}^{d}$ and $\nabla^{2}f\left(x\right)\in\mathbb{R}^{d\times d}$.

Jacobian of $F$: $\mathbb{R}^{d} \rightarrow \mathbb{R}^{p}$ evaluated at $x\in\mathbb{R}^{d}$ by $\partial{F\left(x\right)}\in\mathbb{R}^{p\times d}$.

If $f$ or $F$ have several arguments, the gradient is denoted, Hessian and Jacobian, in the $i^{\text{th}}$ argument by $\nabla_{i}$, $\nabla_{i}^{2}$, and $\partial_{i}$.

The standard simplex is denoted:

$$\Delta^{d} := \left\\{ x \in \mathbb{R}^{d} : ||x||_{1} = 1, x \geq 0 \right\\}.$$

For any set $C \subset \mathbb{R}^{d}$, the indicator function, $I_{C}$, is denoted:

$$\mathbb{R}^{d} \rightarrow \mathbb{R} \cup \left\\{ +\infty \right\\}$$

where $I_{C}\left(x \right) = 0$ if $x \in C$, $I_{C}\left(x \right) = +\infty$ otherwise.

For a vector or matrix, $A$, we note $||A||$ the Frobenius (or Euclidean) norm, and $||A||_{\text{operator}}$ the operator norm.

## Overview

### Zero of a Function

$F$: $\mathbb{R}^{d} \times \mathbb{R}^{n} \rightarrow \mathbb{R}^{d}$ is a map, capturing the optimality conditions.

An optimal solution, $x^{\ast}\left(\theta\right)$, should be a [zero](https://en.wikipedia.org/wiki/Zero_of_a_function) of $F$:

$$F\left ( x^{\ast}\left ( \theta \right ), \theta \right ) = 0.$$

$x^{\ast}\left(\theta\right)$ is an implicitly defined function of $\theta \in \mathbb{R}^{n}$ (i.e. $x^{\ast}: \mathbb{R}^{n} \rightarrow \mathbb{R}^{d}$).

For $(x_{0}, \theta_{0})$, satisfying $F(x_{0}, \theta_{0}) = 0$ with a continuously differentiable $F$, if the Jacobian $\partial_{1}F$, evaluated at $(x_{0}, \theta_{0})$, is a square invertible matrix, then, by the implicit function theorem,  there exists a function $x^{\ast}\left(\cdot \right)$ defined on a neighborhood of $\theta_{0}$ such that $x^{\ast}\left ( \theta_{0} \right ) = x_{0}$.

For all $\theta$ in this neighborhood, $F(x^{\ast}(\theta), \theta) = 0$ and $\partial x^{\ast}(\theta)$ exists. 

Using the chain rule, the Jacobian $\partial x^{\ast}(\theta)$ satisfies:

$$\partial_{1}F(x^{\ast}(\theta), \theta) \partial x^{\ast}(\theta)+\partial_{2}F(x^{\ast}(\theta), \theta) \partial x^{\ast}(\theta)=0.$$

Comparing $\partial x^{\ast}(\theta)$ is the resolution of the linear system of equations:

$$-\partial_{1}F(x^{\ast}(\theta), \theta) \partial x^{\ast}(\theta) = \partial_{2}F(x^{\ast}(\theta), \theta).$$

When $F(x^{\ast}(\theta), \theta) = 0$ is a one-dimensional root-finding problem, $d = 2$, the linear system of equations:

$$-\partial_{1}F(x^{\ast}(\theta), \theta) \partial x^{\ast}(\theta) = \partial_{2}F(x^{\ast}(\theta), \theta).$$

is straightforward because we have $\nabla x^{\ast}(\theta) = \frac{B^{\top}}{A}$, where $A$ is a scalar.

Many existing and new implicit differentiation methods reduce to this principle. This strategy is efficient as it can be added to any solver and modular as the optimality condition is decoupled from the implicit differentiation method.

### Fixed Point

In many applications, $x^{\ast}\left(\theta\right)$ is instead implicitly defined through a fixed point:

$$x^{\ast}\left(\theta\right)=T\left(x^{\ast}(\theta),\theta\right),$$

where $T$: $\mathbb{R}^{d} \times \mathbb{R}^{n} \rightarrow \mathbb{R}^{d}$. This can be seen as a particular case of (1) by defining the residual:

$$F(x,\theta)=T(x,\theta)-x.$$

In this case, when $T$ is continuously differentiable, using the chain rule:

$$A=-\partial_{1}F(x^{\ast}(\theta),\theta)=I-\partial_{1}T(x^{\ast}(\theta),\theta)$$

and

$$B=-\partial_{2}F(x^{\ast}(\theta),\theta)=I-\partial_{2}T(x^{\ast}(\theta),\theta).$$

### JVP and VJP

In most practical scenarios, it is not necessary to explicitly form the Jacobian matrix, and instead it is sufficient to left-multiply or right-multiply by $\partial_{1}F$ and $\partial_{2}F$. These are called vector-Jacobian product (VJP) and Jacobian-vector product (JVP), and are useful for integrating $x^{\ast}(\theta)$ with reverse-mode and forward-mode autodiff, respectively. Oftentimes, $F$ will be explicitly defined. In this case, computing the VJP or JVP can be done by automatic differentiation. In some cases, $F$ may itself be implicitly defined, for instance when $F$ involves the solution of a variational problem. In this case, computing the VJP or JVP will itself involve implicit differentiation.

The right-multiplication (JVP) between $J = \partial x^{\ast}(\theta)$ and a vector $v$, $J_{v}$, can be computed efficiently by solving $A(J_{v})=B_{v}$. The Left-multiplication (VJP) of $v^{\top}$ with $J$, $v^{\top}J$, can be computed by first solving $A^{\top}u=v$. Then, we can obtain $v^{\top}J$ by $v^{\top}J$ by $v^{\top}J=u^{\top}AJ=u{\top}B$. When $B$ changes but $A$ and $v$ remain the same, you do not need to solve $A^{\top}u=v$ once again. This allows to compute VJP in regard to different variables while solving only one linear system.

To solve these linear systems, we can use the conjugate gradient method when $A$ is symmetric positive semi-definite and GMRES or BiCGSTAB otherwise. These algorithms are all matrix-free: they only require matrix-vector products. Thus, all we need from $F$ is its JVPs or VJPs. An alternative to GMRES and BiCGSTAB is to solve the normal equation $AA^{\top}u=Av$ using conjugate gradient. This can be implemented using the transpose operator. In case of non-invertibility, a common heuristic is to solve a least squares $\operatorname{min}_{J}||AJ-B||^ {2}$ instead.

### Pre-Processing and Post-Processing Mappings

Oftentimes, the goal of a practitioner is not to differentiate $\theta$ per se, but the parameters of a function producing $\theta$. 

One example of such pre-processing is to convert the parameters to be differentiated from one form to another canonical form, such as quadratic or conic programs. Another example is when $x^{\ast}\left(\theta\right)$ is used as the output of a neural network layer, in which case $\theta$ is produced by the previous layer. Likewise, $x^{\ast}\left(\theta\right)$ will often not be the final output we want to differentiate. One example of such post-processing is when $x^{\ast}\left(\theta\right)$ is the solution of a dual program and we apply the dual-primal mapping to recover the solution of the primal program. Another example is the application of a loss function, in order to reduce $x^{\ast}\left(\theta\right)$ to a scalar value. 

PyTorch should leave the differentiation of such pre-and-post-processing mappings to the automatic differentiation system, allowing to compose functions in complex ways.

### Examples

#### Gradients

##### Mirror Descent

#### Newton’s Method

Let $x$ be a root of $G(\cdot, \theta)$, i.e., $(Gx, \theta) = 0$. The fixed point iteration of Newton’s method for root-finding is:

$$T(x,\theta)=x-\eta[\partial_{1}G(x,\theta)]^{-1}G(x,\theta).$$

Applying the chain and product rules:

$$\partial_{1}T(x, \theta)= I - \eta(\cdots)G(x, \theta) - \eta[\partial_{1}G(x, \theta)]^{-1} \partial_{1} G (x, \theta) = (1 - \eta)^{I}.$$

So $A = -\partial_{1} F (x, \theta) = \eta I$. Similarly:

$$B=\partial_{2}T(x,\theta)=\partial_{2}F(x,\theta)=-\eta[\partial_{1}G(x,\theta)]^{-1}\partial_{2}G(x, \theta).$$

Newton’s method is obtained by $G(x, \theta) = \delta_{1}f(x, \theta)$, yielding:

$$T(x,\theta)=x-\eta[\nabla^{2}_{1}f(x,\theta)]^{-1}\nabla_{1}f(x,\theta).$$

The LU decomposition of $\partial_{1}G(x,\theta)$, or if, and only if, $\partial_{1}G(x,\theta)$ is positive and semi-definite, the Cholesky decomposition, is pre-computable.

#### Convex Optimization

##### Frank–Wolfe

### Python API

#### Decorators

##### `@torch.optim.root`

```Python
from typing import Callable

def root(f: Callable, g: Callable):
    """
    Add implicit differentiation to a root method.
  
    Args:
      f: equation, ``f(parameters, *args)``. 
        
         Invariant is ``f(solution, *args) == 0`` at ``solution``.
      g: linear solver of the form ``g(a, b)``.
  
    Returns:
      A solver function decorator, i.e., ``root(f)(g)``.
    """
    pass
```

When a solver function is decorated with `@torch.optim.root`, PyTorch adds custom JVP and VJP methods to the [Function](https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function) instance, overriding PyTorch’s default behavior. Linear system solvers based on matrix-vector products are used and only need access to $F$ through the JVP or VJP with $\partial_{1}F$ and $\partial_{2}F$. 

$F$ including a gradient map $\nabla_{1}f(x, \theta)$ is not an issue since PyTorch transparently supports higher-order derivatives. 

##### `@torch.optim.fixed_point`

```Python
from typing import Callable

def fixed_point(f: Callable, g: Callable):
    """
    Add implicit differentiation to a fixed point method.
  
    Args:
      f: equation, ``f(parameters, *args)``. 
        
         Invariant is ``f(solution, *args) == 0`` at ``solution``.
      g: linear solver of the form ``g(a, b)``.

    Returns:
      A solver function decorator, i.e., ``fixed_point(f)(g)``.
    """
    pass
```

#### Functions

##### `torch.optim.root_jvp`

```Python
from typing import Any, Callable, Tuple


def root_jvp(
        f: Callable,
        g: Callable,
        x: Tuple[Any],
        y: Tuple[Any],
        z: Any,
):
    """
    Jacobian-vector product of a root.
  
    Args:
      f: 
      g: linear solver of the form ``g(a, b)``.
      x: arguments.
      y: tangents.
      z: solution.
    """
    pass
```

##### `torch.optim.root_vjp`

```Python
from typing import Any, Callable, Tuple


def root_vjp(
        f: Callable,
        g: Callable,
        x: Tuple[Any],
        y: Tuple[Any],
        z: Any,
):
    """
    Vector-Jacobian product of a root.
  
    Args:
      f: 
      g: linear solver of the form ``g(a, b)``.
      x: arguments.
      y: cotangents.
      z: solution.
    """
    pass
```

#### Bracketing Optimizers

PyTorch provides a variety of bracketing methods for univariate functions, or functions involving a single variable.

Bracketing is the process of identifying an interval in which a local minimum lies and then successively shrinking the interval. For many functions, derivative information can be helpful in directing the search for an optimum, but, for some functions, this information may not be available or might not exist. 

##### Bisection Method

```Python
from typing import Callable, Optional, Tuple

from torch import Tensor
from torch.optim import Optimizer


class Bisection(Optimizer):
    def __init__(
            self,
            f: Callable[[Tensor], Tensor],
            bracket: Tuple[float, float],
            tolerance: Optional[float] = None,
            maximum_iterations: Optional[int] = None,
    ):
        raise NotImplementedError
```

###### Example

```Python
import functorch
from torch import Tensor
from torch.optim import Bisection
import torch


def f(x: Tensor) -> Tensor:
  return x ** 3 - x - 2

assert Bisection(f, (1, 2)) == torch.tensor([1.521])

def g(x: Tensor, k: Tensor) -> Tensor:
    return k * x ** 3 - x - 2
    
def root(k: Tensor) -> Tensor:  
    return Bisection(g, (1, 2))(k).params

# Derivative of the root of `f` with respect to `k` where $k = 2.0$.
functorch.grad(root)(2.0)
```

##### Brent’s Method

Brent’s method is an extension of the bisection method. It is a root-finding algorithm that combines elements of the secant method and inverse quadratic interpolation. It has reliable and fast convergence properties, and it is the univariate optimization algorithm of choice in many popular numerical optimization packages.

```Python
from typing import Callable, Optional, Tuple, Union

from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class Brent(Optimizer):
    def __init__(
            self,
            f: Callable[[Tensor], Tensor],
            bracket: Union[Tuple[float, float], Tuple[float, float, float]],
            tolerance: float,
            maximum_iterations: Optional[int] = None,
    ):
        raise NotImplementedError
```

#### Local Descent Optimizers

Local models incrementally improve a design point until some convergence criterion is met.

A common approach to optimization is to incrementally improve a design point $x$ by taking a step that minimizes the objective value based on a local model. The local model may be obtained, for example, from a first-or-second-order Taylor approximation. Optimization algorithms that follow this general approach are referred to as descent direction methods. They start with a design point $x^{1}$ and then generate a sequence of points, iterates, to converge to a local minimum.

The iterative descent direction procedure involves the following steps:

1. Check whether $x^{k}$ satisfies the termination conditions. If it does, terminate; otherwise proceed to the next step.
2. Determine the descent direction $d^{k}$ using local information such as the gradient or Hessian. Some algorithms assume $||d^{k}|| = 1$, but others do not.
3. Determine the step size or learning rate $\alpha^{k}$. Some algorithms attempt to optimize the step size so that the step maximally decreases $f$.
4. Compute the next design point according to:

$$x^{k + 1} \leftarrow x^{k} + \alpha^{k}d^{k}$$

There are many different optimization methods, each with their own ways of determining $\alpha$ and $d$.

##### Line Search

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class LineSearch(Optimizer):
    def __init__():
        raise NotImplementedError
```

##### Trust Region

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class TrustRegion(Optimizer):
    def __init__():
        raise NotImplementedError
```

##### Backtracking Line Search

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class BacktrackingLineSearch(Optimizer):
    def __init__():
        raise NotImplementedError
```

#### Non-Linear Least Squares Optimizers

Problems of the form:

$$\min_{x} \frac{1}{2} ||\textbf{r}(x, \theta)||^2,$$

where $\textbf{r}$ is a residual function, $x$ are the parameters with respect to which the function is minimized, and $\theta$ are optional additional arguments.

##### Gauss–Newton

The update equation is solved for every iteration to find the update to the parameters:

$$\mathbf{J} \mathbf{J^T} h_{gn} = - \mathbf{J^T} \mathbf{r}$$

where $\mathbf{J}$ is the Jacobian of the residual function w.r.t. parameters.

```Python
from typing import Callable, Optional

from torch import Optimizer

class GaussNewton(Optimizer):
    def __init__(
            self,
            f: Callable,
            g: Optional[Callable],
            maximum_iterations: int,
            tolerance: float,
    ):
        pass
```

##### Levenberg–Marquardt

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class LevenbergMarquardt(Optimizer):
    def __init__():
        raise NotImplementedError
```

#### First-Order Method Optimizers

First-order methods rely on gradient information to help direct the search for a minimum, which can be obtained using derivatives and gradients.

##### Gradient Descent

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class GradientDescent(Optimizer):
    def __init__():
        raise NotImplementedError
```

##### Conjugate Gradient Method

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class ConjugateGradientDescent(Optimizer):
    def __init__():
        raise NotImplementedError
```

##### Adagrad

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class Adagrad(Optimizer):
    def __init__():
        raise NotImplementedError
```

##### RMSProp

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class RMSProp(Optimizer):
    def __init__():
        raise NotImplementedError
```

##### Adadelta

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class Adadelta(Optimizer):
    def __init__():
        raise NotImplementedError
```

##### Adam

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class Adam(Optimizer):
    def __init__():
        raise NotImplementedError
```

#### Second-Order Method Optimizers

Second-order methods leverage second-order approximations that use the second derivative in univariate optimization or the Hessian in multivariate optimization to direct the search. This additional information can help improve the local model used for informing the selection of directions and step lengths in descent algorithms.

##### Newton’s Method

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class Newton(Optimizer):
    def __init__():
        raise NotImplementedError
```

##### Secant Method

Newton’s method for univariate function minimization requires the first and second derivatives $f'$ and $f''$. In many cases, $f'$ is known but the second derivative is not. The secant method applies Newton’s method using estimates of the second derivative and thus only requires $f′$. This property makes the secant method more convenient to use in practice.

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class Secant(Optimizer):
    def __init__():
        raise NotImplementedError
```

##### Limited-Memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) Method

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class LBFGS(Optimizer):
    def __init__():
        raise NotImplementedError
```

#### Derivative-Free Optimizers

Derivative-free methods rely only on the objective function, $f$. Derivative-free methods do not rely on derivative information to guide them toward a local minimum or identify when they have reached a local minimum. They use other criteria to choose the next search direction and to judge when they have converged.

##### Coordinate Descent

##### Powell’s Method

Powell’s method can search in directions that are not orthogonal to each other. The method can automatically adjust for long, narrow valleys that might otherwise require a large number of iterations for cyclic coordinate descent or other methods that search in axis-aligned directions.

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class Powell(Optimizer):
    def __init__():
        raise NotImplementedError
```

##### Nelder-Mead Method

The Nelder-Mead method uses a simplex to traverse the space in search of a minimum. A simplex is a generalization of a tetrahedron to $n$-dimensional space. A simplex in one dimension is a line, and in two dimensions it is a triangle. The simplex derives its name from the fact that it is the simplest possible polytope in any given space.

```Python
from torch import Tensor

from torch import Tensor
from torch.optim import Optimizer

class NelderMead(Optimizer):
    def __init__():
        raise NotImplementedError
```

#### Linear Constrained Optimizers

## Drawbacks

* The automatic implicit differentiation strategy _theoretically_ only applies to situations where the implicit function theorem is valid, namely, where optimality conditions satisfy the differentiability and invertibility conditions. While this covers a wide range of situations, even for non-smooth optimization problems (e.g., under mild assumptions the solution of a Lasso regression can be differentiated a.e. with respect to the regularization parameter), this strategy could be extended to handle cases where the differentiability and invertibility conditions are not satisfied (e.g., using non-smooth implicit function theorems).

## Prior Art

### CasADi

> CasADi is an open-source tool for nonlinear optimization and algorithmic differentiation. It facilitates rapid — yet efficient — implementation of different methods for numerical optimal control, both in an offline context and for nonlinear model predictive control (NMPC).

### [JAX](https://jax.readthedocs.io)

#### [JAXopt](https://jaxopt.github.io)

#### [Optax](https://optax.readthedocs.io)
