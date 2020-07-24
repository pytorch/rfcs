
import io
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
            return torch.randint(0, 5, shape, dtype=dtype)
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
    extra_kwargs = {}
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
    if fname == 'cross_entropy':
        return (make_tensor((5, 5), layout, rdist=rdist, dtype=dtype), make_tensor((5,), layout, rdist=rdist, dtype=int),), {}
    if fname == 'ctc_loss':
        return (make_tensor((5, 5, 5), layout, rdist=rdist, dtype=dtype),
                make_tensor((5, 5), layout, rdist=rdist, dtype=int),
                make_tensor((5,), layout, rdist=rdist, dtype=int),
                make_tensor((5,), layout, rdist=rdist, dtype=int),
        ), {}
    if fname == 'multilabel_margin_loss':
        return (make_tensor((5, 5), layout, rdist=rdist, dtype=dtype),
                make_tensor((5, 5), layout, rdist=rdist, dtype=int),
        ), {}
    if fname == 'nll_loss':
        return (make_tensor((5, 5), layout, rdist=rdist, dtype=dtype),
                make_tensor((5, ), layout, rdist=rdist, dtype=int),
        ), {}
    if fname == 'prelu':
        return (make_tensor((5, 5), layout, rdist=rdist, dtype=dtype),
                make_tensor((5, ), layout, rdist=rdist, dtype=dtype),
        ), {}
    if fname == 'addmm':
        return (make_tensor((5, 5), torch.strided, rdist=rdist, dtype=dtype),
                make_tensor((5, 5), layout, rdist=rdist, dtype=dtype),
                make_tensor((5, 5), torch.strided, rdist=rdist, dtype=dtype),
        ), {}
    
    if fname in ['bernoulli', 'cholesky', 'poisson', 'binary_cross_entropy']:
        rdist = 'uniform'
    if fname in ['cholesky']:
        rdist = 'posdefined'
    if fname in ['bincount', 'bitwise_and', 'bitwise_not', 'bitwise_or', 'bitwise_xor',
                 'int_repr']:
        dtype = int
    if fname in ['bincount', 'combinations', 'dot', 'ger', 'vander']:
        size = (2,)
    if fname in ['bmm', 'conv1d', 'conv_transpose1d', 'bilinear']:
        size = (2, 2, 2)
    if fname in ['conv2d', 'conv_transpose2d', 'grid_sample']:
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
    if fname == 'interpolate':
        size = (2, 2, 2)
        extra_kwargs.update(size=1)
    if fname in ['upsample']:
        size = (2, 2, 2)
        extra_kwargs.update(size=1)
    if fname in ['upsample_bilinear', 'upsample_nearest']:
        size = (2, 2, 2, 2)
        extra_kwargs.update(size=1)
    args = []
    for argname, params in sig.parameters.items():
        if params.default is not inspect.Parameter.empty:
            break
        annot = params.annotation
        if annot is torch.Tensor:
            args.append(make_tensor(size, layout, dtype=dtype, rdist=rdist))
        else:
            raise NotImplementedError(f'{fname}')
            raise NotImplementedError(f'{fname}{sig}')
    return tuple(args), extra_kwargs


def get_doclink(func):
    if func.__module__ == 'torch.sparse':
        return f'https://pytorch.org/docs/stable/sparse.html#torch.sparse.{func.__name__}'
    if func.__module__ == 'torch.nn.functional':
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

def l_not(predicate):
    def op(*args):
        return not predicate(*args)
    return op


def l_and(*predicates):
    def op(*args):
        for predicate in predicates:
            if not predicate(*args):
                return False
        return True
    return op

def l_or(*predicates):
    def op(*args):
        for predicate in predicates:
            if predicate(*args):
                return True
        return False
    return op


def has_layout(func, sig):
    if func.__name__.endswith('_'):
        return False
    return 'layout' in sig.parameters


def returns_tensor(func, sig):
    if func.__name__.endswith('_'):
        return False
    return sig.return_annotation == torch.Tensor


def has_all_tensor_inputs(func, sig):
    if func.__name__.endswith('_'):
        return False
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            break
        if not (param.annotation == torch.Tensor):
            return False
    return True


def has_tensor_input(func, sig):
    if func.__name__.endswith('_'):
        return False
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            break
        if param.annotation == torch.Tensor:
            return True
    return False


def try_layout(func, sig, layout, must_pass=False):
    try:
        test_args, test_kwargs = get_test_args(func.__name__, sig, layout)
        ok = True
    except Exception as msg:
        fail = str(msg).strip().splitlines()[0]
        if len(fail) > 40:
            fail = fail[:38] + '...'
        status = f'{type(msg).__name__}: {fail}'
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
                if len(fail) > 40:
                    fail = fail[:38] + '...'
                status = f'{type(msg).__name__}: {fail}'
            if must_pass:
                print(func.__doc__)
                print(f'{func.__name__}{test_args}')
                raise
    return status


def main(working_dir=None):
    layouts = get_layouts()
    if working_dir is None:
        working_dir = os.path.dirname(__file__)

    predicates = []

    f = io.StringIO('')
    headers = ['Function'] + list(map(str, layouts))

    f.write('# Tensor constructors\n\n')
    
    f.write('## Functions with layout argument\n\n')

    lst = []
    failures = defaultdict(list)
    predicates.append(has_layout)
    for func, sig in all_functions(predicates[-1]):
        row = [get_docname(func, sig)]
        for layout in layouts:
            row.append(try_layout(func, sig, layout))
        lst.append(row)
    text_utils.table(f, lst, list(range(len(headers))), headers)

    f.write('\n\n')

    f.write('## Functions with tensor inputs\n\n')

    lst = []
    predicates.append(has_all_tensor_inputs)
    for func, sig in all_functions(predicates[-1]):
        print(func.__module__, func.__name__, sig)

        allow_fail = func.__name__ in [
            'q_per_channel_axis',
            'q_per_channel_scales',
            'q_per_channel_zero_points',
            'q_scale',
            'q_zero_point',
        ]
        if func.__module__ == 'torch.sparse':
            native_layout = torch.sparse_coo
        else:
            native_layout = torch.strided
        row = [get_docname(func, sig)]
        for layout in layouts:
            row.append(try_layout(func, sig, layout, must_pass=(layout==native_layout and not allow_fail)))
        lst.append(row)

    text_utils.table(f, lst, list(range(len(headers))), headers)

    f.write('# Functions not covered above\n\n')

    lst = []
    failures = defaultdict(list)
    for func, sig in all_functions(l_not(l_or(*predicates))):
        row = [get_docname(func, sig)]
        lst.append(row)
    text_utils.table(f, lst, list(range(1)), headers[:1])

    f.write('\n\n')
    s = f.getvalue()
    f = open(os.path.join(working_dir, 'SparseSupportState.md'), 'w')
    f.write(s)
    f.close()

if __name__ == '__main__':
    main()
