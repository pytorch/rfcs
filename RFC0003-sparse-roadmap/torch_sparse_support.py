
import os
import sys
import inspect
import torch
from torch_signatures import scan_module
import text_utils
from collections import defaultdict


def get_layouts():
    """Return a list of public layouts.
    """
    return [a for n, a in torch.__dict__.items() if isinstance(a, torch.layout) and not n.startswith('_')]


def random_coo(shape, dtype, sparsity=0.75, coalesce=True, rdist='random'):
    total = 1
    for dim in shape:
        total *= dim
    nnz = int(total * (1 - sparsity))
    nnz = max(0, min(nnz, total))
    i, j, d = [], [], set()
    indices = [[] for dim in shape]
    for n in range(nnz):
        while 1:
            _index = tuple(torch.randint(0, dim-1, ()) for dim in shape)
            if _index in d:
                continue
            d.add(_index)
            break
    for _index in (sorted(d) if coalesce else d):
        for _i in range(len(shape)):
            indices[_i].append(_index[_i])
    values = make_tensor((nnz,), torch.strided, dtype=dtype, rdist=rdist)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=dtype)


def make_tensor(shape, layout, dtype=float, rdist='randn'):
    if layout == torch.strided:
        if dtype in [bool]:
            return torch.randint(0, 1, shape, dtype=dtype)
        if dtype in [int]:
            return torch.randint(0, 10, shape, dtype=dtype)
        if rdist == 'uniform':
            t = torch.empty(shape, dtype=dtype)
            t.uniform_()
        elif rdist == 'randn':
            if dtype in [torch.qint32]:
                t = torch.randint(-10, 10, shape, dtype=torch.float32)
                t = torch.quantize_per_tensor(t, 1.0, 0, dtype=dtype)
            else:
                t = torch.randn(shape, dtype=dtype)
        elif rdist == 'posdefined':
            t = torch.randn(shape, dtype=dtype)
            for i in range(len(shape)):
                t[(i,) * len(shape)] += 5
        else:
            raise NotImplementedError(rdist)
        return t
    if layout == torch.sparse_coo:
        return random_coo(shape, dtype, rdist=rdist)
    raise NotImplementedError(layout)


def get_test_args(fname, sig, layout):
    rdist = 'randn'
    dtype = float
    size = (2, 2)
    if fname == 'arange': return (10,), dict(layout=layout)
    if fname == 'range': return (0, 10), dict(layout=layout)
    if fname == 'bartlett_window': return (3,), dict(layout=layout)
    if fname == 'blackman_window': return (3,), dict(layout=layout)
    if fname == 'hamming_window': return (3,), dict(layout=layout)
    if fname == 'hann_window': return (3,), dict(layout=layout)
    if fname == 'empty': return (2, 2), dict(layout=layout)
    if fname == 'empty_like': return (make_tensor((2, 2), layout, rdist=rdist, dtype=dtype),), dict(layout=layout)
    if fname == 'empty_strided': return ((2, 2), (2, 1)), dict(layout=layout)
    if fname == 'ones': return (2, 2), dict(layout=layout)
    if fname == 'ones_like': return (make_tensor((2, 2), layout, rdist=rdist, dtype=dtype),), dict(layout=layout)
    if fname == 'zeros': return (2, 2), dict(layout=layout)
    if fname == 'zeros_like': return (make_tensor((2, 2), layout, rdist=rdist, dtype=dtype),), dict(layout=layout)
    if fname == 'eye': return (2,), dict(layout=layout)
    if fname == 'full': return ((2, 2), 0), dict(layout=layout, dtype=float)
    if fname == 'full_like': return (make_tensor((2, 2), layout, rdist=rdist, dtype=dtype), 0), dict(layout=layout)
    if fname == 'linspace': return (0, 10), dict(layout=layout)
    if fname == 'logspace': return (0, 10), dict(layout=layout)
    if fname == 'rand': return (2, 2), dict(layout=layout)
    if fname == 'randn': return (2, 2), dict(layout=layout)
    if fname == 'rand_like': return (make_tensor((2, 2), layout, rdist=rdist, dtype=dtype),), dict(layout=layout)
    if fname == 'randn_like': return (make_tensor((2, 2), layout, rdist=rdist, dtype=dtype),), dict(layout=layout)
    if fname == 'randint': return (0, 10, (2, 2)), dict(layout=layout)
    if fname == 'randint_like': return (make_tensor((2, 2), layout, rdist=rdist, dtype=dtype), 0, 10), dict(layout=layout)
    if fname == 'randperm': return (2,), dict(layout=layout)
    if fname == 'tril_indices': return (2, 2), dict(layout=layout)
    if fname == 'triu_indices': return (2, 2), dict(layout=layout)
    if fname == 'addbmm': return (make_tensor((2, 2), layout, rdist=rdist, dtype=dtype), make_tensor((3, 2, 2), layout, rdist=rdist, dtype=dtype), make_tensor((3, 2, 2), layout, rdist=rdist, dtype=dtype)), {}
    if fname == 'addmv': return (make_tensor((2,), layout, rdist=rdist, dtype=dtype), make_tensor((2, 2), layout, rdist=rdist, dtype=dtype), make_tensor((2,), layout, rdist=rdist, dtype=dtype)), {}
    if fname == 'addr': return (make_tensor((2, 2), layout, rdist=rdist, dtype=dtype), make_tensor((2,), layout, rdist=rdist, dtype=dtype), make_tensor((2,), layout, rdist=rdist, dtype=dtype)), {}
    if fname in ['all', 'any']: return (make_tensor((2, 2), layout, dtype=bool, rdist=rdist),), {}
    if fname == 'baddbmm': return (make_tensor((3, 2, 2), layout, rdist=rdist, dtype=dtype), make_tensor((3, 2, 2), layout, rdist=rdist, dtype=dtype), make_tensor((3, 2, 2), layout, rdist=rdist, dtype=dtype)), {}
    if fname in ['lu_unpack', 'lu_unpacktensor']:
        t = make_tensor((3, 2, 2), layout, rdist=rdist, dtype=dtype)
        A_LU, pivots = t.lu()
        return (A_LU, pivots), {}
    if fname == 'mv': return (make_tensor((2, 2), layout, rdist=rdist, dtype=dtype), make_tensor((2,), layout, rdist=rdist, dtype=dtype)), {}
    if fname == 'batch_norm':
        return (make_tensor((2, 2), layout, rdist=rdist, dtype=dtype), make_tensor((2, ), layout, rdist=rdist, dtype=dtype), make_tensor((2, ), layout, rdist=rdist, dtype=dtype)), {}
    
    if fname in ['bernoulli', 'cholesky', 'poisson']:
        rdist = 'uniform'
    if fname in ['cholesky']:
        rdist = 'posdefined'
    if fname in ['bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor',
                 'int_repr']:
        dtype = int
    if fname in ['bincount', 'combinations', 'dot', 'ger', 'vander']:
        size = (2,)
    if fname in ['bmm', 'conv1d', 'conv_transpose1d']:
        size = (2, 2, 2)
    if fname in ['conv2d', 'conv_transpose2d']:
        size = (2, 2, 2, 2)
    if fname in ['conv3d', 'conv_transpose3d']:
        size = (2, 2, 2, 2, 2)
    if fname in ['cross']:
        size = (3, 3)
    if fname in ['is_nonzero']:
        size = (1, )
    if fname in ['lobpcg']:
        size = (9, 9)
    if fname in ['lu_unpack']:
        size = (3, 2, 2)
    if fname in ['imag', 'real', 'view_as_real']:
        dtype = torch.cfloat
    if fname in ['dequantize', 'int_repr', 'q_per_channel_axis', 'q_per_channel_scales', 'q_per_channel_zero_points',
                 'q_scale', 'q_zero_point']:
        dtype = torch.qint32
    args = []
    for argname, params in sig.parameters.items():
        if params.default is not inspect.Parameter.empty:
            break
        annot = params.annotation
        if annot is torch.Tensor:
            args.append(make_tensor(size, layout, dtype=dtype, rdist=rdist))
        else:
            raise NotImplementedError((fname, sig, layout))
    return tuple(args), {}


def get_doclink(func):
    if func.__module__ == 'torch.sparse':
        return f'https://pytorch.org/docs/stable/sparse.html#torch.sparse.{func.__name__}'
    if func.__module__ == 'nn.functional.html':
        return f'https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.{func.__name__}'
    if func.__name__ in ['conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']:
        return f'https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.{func.__name__}'
    return f'https://pytorch.org/docs/master/generated/{func.__module__}.{func.__name__}.html'


def get_docname(func, sig):
    #return f'{func.__module__}.{func.__name__}'
    return f'<a href="{get_doclink(func)}">{func.__module__}.{func.__name__}</a>'
    return f'<a href="{get_doclink(func)}" title="{func.__name__}{sig}">{func.__module__}.{func.__name__}</a>'


def all_functions(filter=lambda func, sig: True, _cache=[]):
    if not _cache:
        _cache.extend(scan_module(module=[torch, torch.nn.functional, torch.sparse]))
    for func, sig in _cache:
        if filter(func, sig):
            yield func, sig


def has_layout(func, sig):
    return 'layout' in sig.parameters


def has_all_tensor_inputs(func, sig):
    if func.__name__.endswith('_'):
        return False
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            break
        if not (param.annotation == torch.Tensor):
            return False
    return True


def try_layout(func, sig, layout, must_pass=False):
    allow_fail = [
        'q_per_channel_axis',
        'q_per_channel_scales',
        'q_per_channel_zero_points',
        'q_scale',
        'q_zero_point',
    ]
    try:
        test_args, test_kwargs = get_test_args(func.__name__, sig, layout)
        ok = True
    except Exception as msg:
        fail = str(msg).strip().splitlines()[0]
        status = fail
        ok = False
    if ok:
        try:
            test_result = func(*test_args, **test_kwargs)
            status = 'OK'
        except Exception as msg:
            fail = str(msg).strip().splitlines()[0]
            if 'is not implemented for' in fail:
                status = f'{type(msg).__name__}: not implemented'
            elif fail.startswith('unsupported tensor layout'):
                status = f'{type(msg).__name__}: unsupported layout'
            elif fail.startswith('Could not run') and 'with arguments from the' in fail:
                status = f'{type(msg).__name__}: unsupported backend'
            else:
                status = fail
            if must_pass and func.__name__ not in allow_fail:
                print(func.__doc__)
                print(f'{func.__name__}{test_args}')
                #raise
    if len(status) > 80:
        status = status[:78] + '...'
    return status


def main(working_dir=None):
    layouts = get_layouts()
    if working_dir is None:
        working_dir = os.path.dirname(__file__)

    f = open(os.path.join(working_dir, 'SparseSupportState.md'), 'w')

    f.write('## Functions with layout argument\n\n')

    headers = ['Function'] + list(map(str, layouts))
    
    lst = []
    failures = defaultdict(list)
    for func, sig in all_functions(has_layout):
        row = [get_docname(func, sig)]
        for layout in layouts:
            row.append(try_layout(func, sig, layout))
        lst.append(row)
    text_utils.table(f, lst, list(range(len(headers))), headers)

    f.write('\n\n')

    f.write('## Functions with tensor inputs\n\n')

    lst = []
    for func, sig in all_functions(has_all_tensor_inputs):
        print(func.__module__, func.__name__, sig)
        row = [get_docname(func, sig)]
        for layout in layouts:
            row.append(try_layout(func, sig, layout, must_pass=layout==torch.strided))
        lst.append(row)

    text_utils.table(f, lst, list(range(len(headers))), headers)

    f.close()


if __name__ == '__main__':
    main()
