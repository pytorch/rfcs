# Interpolation

## Authors

* Allen Goodman (@0x00b1)

## Summary

Interpolation is a technique for adding new data points in a range of a set of known data points. You can use interpolation to fill-in missing data, smooth existing data, make predictions, and more.

Interpolation operators operate on:

* data on a regular grid (i.e., predetermined, not necessarily, uniform, spacing); or 
* scattered data on an irregular grid.

### `torch.interpolation.interpolate`

```Python
from typing import Callable, Optional, Tuple

from torch import Tensor

def interpolate(
        x: Tuple[Tensor],
        v: Tensor,
        q: Tuple[Tensor],
        f: Callable[[Tuple[Tensor]], Tuple[Tensor]],
        *,
        out: Optional[Tensor] = None
):
    raise NotImplementedError
```

Interpolate $n$-dimensional data on a regular grid (i.e., predetermined, not necessarily, uniform, spacing).

### `torch.interpolation.unstructured_interpolate`

```Python
from typing import Callable, Optional, Tuple

from torch import Tensor

def unstructured_interpolate(
        input: Tensor,
        points: Tuple[Tensor],
        x_i: Tuple[Tensor],
        interpolant: Callable[[Tensor], Tensor],
        *,
        out: Optional[Tensor] = None
):
    raise NotImplementedError
```

Interpolate scattered data on an irregular grid.

**Note**—Using this operation in dimensions greater than six is impractical because the memory required by the underlying Delaunay triangulation grows exponentially with its rank.

**Note**—Because this operator uses a Delaunay triangulation, it can be sensitive to scaling issues in `input`. When this occurs, you should standardize `input` to improve the results.

##### Parameters

**input** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

**points** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

**x_i** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

**interpolant** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)) – 

##### Keyword Arguments

**out** ([Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor), *optional*) – output.
