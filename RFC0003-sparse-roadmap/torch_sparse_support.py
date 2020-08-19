
import io
import os
import torch
import numpy
import re
import configparser
import types
import typing
import text_utils
from collections import defaultdict

modules = [torch, torch.nn.functional, torch.sparse]


def all_functions(module=modules, recursive=False, _cache=set()):
    """Iterator of functions from a module or modules.
    """
    if isinstance(module, (list, tuple)):
        for m in module:
            for r in all_functions(m, recursive=recursive):
                yield r
        return

    if module.__name__ in _cache:
        return
    _cache.add(module.__name__)
    for name, member in sorted(module.__dict__.items()):
        if isinstance(member, (bool, str, type, dict, list, tuple, int,
                               float, complex)):
            continue
        if isinstance(member, typing._Final):
            continue
        if not callable(member):
            if name == 'classes':
                continue
            if isinstance(member, types.ModuleType) and recursive:
                if not member.__name__.startswith('torch.'):
                    continue
                for r in all_functions(module=member, recursive=recursive):
                    yield r
            continue
        if not getattr(member, '__doc__', None):
            continue
        if ((member.__name__.startswith('_')
             and not member.__name__.startswith('__'))):
            continue
        if member.__module__ is None:
            member.__module__ = module.__name__
        yield member


def random_coo(shape, dtype, sparsity=0.75, coalesce=True, rdist='random',
               requires_grad=False,
               require_positive=False, device=None):
    total = 1
    for dim in shape:
        total *= dim
    nnz = int(total * (1 - sparsity))
    nnz = max(0, min(nnz, total))
    d = set()
    indices = [[] for dim in shape]
    for n in range(nnz):
        while 1:
            _index = tuple(
                torch.randint(0, max(1, dim-1), ()) for dim in shape)
            if _index in d:
                continue
            d.add(_index)
            break
    for _index in (sorted(d) if coalesce else d):
        for _i in range(len(shape)):
            indices[_i].append(_index[_i])
    values = make_tensor(
        (nnz,), dtype=dtype, rdist=rdist,
        device=device, require_positive=require_positive)
    if dtype in [complex, torch.complex128]:
        values += 1j * make_tensor(
            (nnz,), dtype=dtype, rdist=rdist,
            device=device, require_positive=require_positive)
        dtype = torch.complex128
    return torch.sparse_coo_tensor(
        indices, values, shape, dtype=dtype, device=device,
        requires_grad=requires_grad)


def make_tensor(shape_or_data, dtype=float, rdist='randn',
                requires_grad=False,
                require_positive=False, **params):
    if isinstance(shape_or_data, tuple):
        data, shape = None, shape_or_data
    else:
        data, shape = shape_or_data, None
    layout = params.get('layout', torch.strided)
    device = params.get('device', None)
    if layout == torch.strided:
        if data is not None:
            return torch.tensor(data, dtype=dtype, device=device,
                                requires_grad=requires_grad)
        if dtype in [bool]:
            return torch.randint(0, 1, shape, dtype=dtype, device=device,
                                 requires_grad=requires_grad)
        if dtype in [int]:
            return torch.randint(0, 5, shape, dtype=dtype, device=device,
                                 requires_grad=requires_grad)
        if rdist == 'uniform':
            t = torch.empty(shape, dtype=dtype, device=device,
                            requires_grad=requires_grad)
            t.uniform_()
        elif rdist == 'randn':
            if dtype in [torch.qint32]:
                t = torch.randint(-10, 10, shape,
                                  dtype=torch.float32, device=device)
                t = torch.quantize_per_tensor(t, 1.0, 0, dtype=dtype)
            elif dtype in [complex]:
                t = torch.randn(shape, dtype=float, device=device,
                                requires_grad=requires_grad)
                t = t + 1j * torch.randn(shape, dtype=float, device=device,
                                         requires_grad=requires_grad)
            else:
                t = torch.randn(shape, dtype=dtype, device=device,
                                requires_grad=requires_grad)
                if require_positive:
                    t = t * t
        elif rdist == 'posdefined':
            t = torch.randn(shape, dtype=dtype, device=device,
                            requires_grad=requires_grad)
            for i in range(len(shape)):
                t[(i,) * len(shape)] += 5
        else:
            raise NotImplementedError(rdist)
        return t
    if layout == torch.sparse_coo:
        if data is not None:
            t = torch.tensor(
                data, dtype=dtype, device=device).to_sparse().coalesce()
            return torch.sparse_coo_tensor(
                t.indices(), t.values(), t.size(),
                dtype=t.dtype, device=device, requires_grad=requires_grad)
        return random_coo(shape, dtype=dtype, rdist=rdist,
                          requires_grad=requires_grad,
                          require_positive=require_positive, device=device)
    raise NotImplementedError(layout)


def get_seclink(section_title):
    lst = section_title.lower().replace('/', '').split()
    return '#' + '-'.join(lst)


def get_secname(section_title):
    return f'<a href="{get_seclink(section_title)}">{section_title}</a>'


def get_doclink(func):
    if func.__module__ == 'torch.sparse':
        return (f'https://pytorch.org/docs/stable/sparse.html'
                f'#torch.sparse.{func.__name__}')
    if func.__module__ == 'torch.nn.functional':
        return (f'https://pytorch.org/docs/stable/nn.functional.html'
                f'#torch.nn.functional.{func.__name__}')
    if func.__name__ in [
            'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
            'conv_transpose2d', 'conv_transpose3d']:
        return (f'https://pytorch.org/docs/stable/nn.functional.html'
                f'#torch.nn.functional.{func.__name__}')
    return (f'https://pytorch.org/docs/master/generated/'
            f'{func.__module__}.{func.__name__}.html')


def get_docname(func, sig):
    return (f'<a href="{get_doclink(func)}">'
            f'{func.__module__}.{func.__name__}</a>')


class Classifier:

    @classmethod
    def fromfile(cls, filename, working_dir=None):
        if working_dir is None:
            working_dir = os.path.dirname(__file__)
        ini_file = os.path.join(working_dir, filename)
        config = configparser.RawConfigParser()
        config.read([ini_file])
        return cls(config)

    def __init__(self, config, working_dir=None, verbose=True):
        self.verbose = verbose
        if working_dir is None:
            working_dir = os.path.dirname(__file__)
        # replace names string content with a list
        self.attrs = {}
        for section in config.sections():
            if section == 'ATTRIBUTES':
                continue
            if config.has_option(section, 'names'):
                names = []
                for name in config.get(section, 'names').split():
                    name = name.strip()
                    if not name:
                        continue
                    if name.endswith('!') or name.endswith('?'):
                        i = name.index(name[-1])
                        name, attr = name[:i], name[i:]
                        self.attrs[name] = attr
                    elif '|' in name:
                        name, attr = name.split('|', 1)
                        self.attrs[name] = attr
                    names.append(name)
                config.set(section, 'names', names)
        #
        self.working_dir = working_dir
        self.config = config

    def iter_sections(self):
        for section in self.config.sections():
            if section == 'ATTRIBUTES':
                continue
            yield section

    def attach(self, func, sig):
        """Attach function and its signature to classifier. Return section
        when defined.
        """
        for section in self.iter_sections():
            if not self.config.has_option(section, 'names'):
                continue
            names = self.config.get(section, 'names')
            if func.__name__ in names:
                if not self.config.has_option(section, '_funcs'):
                    self.config.set(section, '_funcs', [])
                funcs = self.config.get(section, '_funcs')
                funcs.append((func, sig))
                return section

    def __str__(self):
        lines = []
        for section in self.config.sections():
            lines.append(f'[{section}]')
            for option in self.config.options(section):
                value = self.config.get(section, option)
                if isinstance(value, list):
                    value = ' '.join(map(str, value))
                lines.append(f'{option}: {value}')
        return '\n'.join(lines)

    def get_layouts(self):
        return [a for n, a in torch.__dict__.items()
                if isinstance(a, torch.layout) and not n.startswith('_')]

    def get_devices(self):
        devices = ['cpu']
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                devices.append(f'cuda:{count-1}')
        return devices

    def iter_tensor_parameters(self):
        """A iterator of parameters that determine the tensor kind, such
        as tensors with various layouts (strided, sparse_coo, etc) and
        storage locations (cpu, cuda, etc)
        """
        layouts = self.get_layouts()
        devices = self.get_devices()

        for layout in layouts:
            for device in devices:
                title = f'{str(layout).split(".")[-1]}@{device}'
                yield dict(layout=layout, device=device), title

    def _skip(self, func, sig, tensor_parameters):
        attr = self.attrs.get(func.__name__)
        if attr == 'I':
            return True
        if tensor_parameters.get('layout') == torch.strided and attr == 'S':
            return True
        if tensor_parameters.get('layout') != torch.strided and attr == 'D':
            return True

    def iter_func_args(self, func, sig, tensor_parameters):
        layout_device_kwargs = dict(
            layout=tensor_parameters.get('layout'),
            device=tensor_parameters.get('device'),
            requires_grad=tensor_parameters.get('requires_grad', False))
        device_kwargs = dict(device=tensor_parameters.get('device'))
        if self._skip(func, sig, tensor_parameters):
            pass
        elif func.__name__ in ['arange', 'range']:
            if func.__name__ == 'arange':
                yield (5, ), layout_device_kwargs
            yield (1, 5), layout_device_kwargs
            yield (1, 5, 0.5), layout_device_kwargs
        elif func.__name__ in ['randint']:
            yield (1, 5, (2, 2)), layout_device_kwargs
        elif func.__name__ in [
                'randperm', 'bartlett_window', 'blackman_window',
                'hamming_window', 'hann_window']:
            yield (5, ), layout_device_kwargs
        elif func.__name__ in ['randint_like']:
            yield ((make_tensor((2, 2), **tensor_parameters), 1, 5),
                   layout_device_kwargs)
        elif func.__name__ == 'as_strided':
            yield (make_tensor((3, 3), **tensor_parameters),
                   (2, 2), (1, 2)), {}
            yield (make_tensor((3, 3), **tensor_parameters),
                   (2, 2), (1, 2), 1), {}
        elif func.__name__ == 'as_tensor':
            yield ([[1, 2], [3, 4]],), device_kwargs
            yield (make_tensor((3, 3), **tensor_parameters), ), {}
            if tensor_parameters.get('device', 'cpu') != 'cpu':
                yield ((make_tensor((3, 3), **tensor_parameters), ),
                       dict(device='cpu'))
        elif func.__name__ in [
                'dequantize', 'quantize_per_channel', 'quantize_per_tensor',
                'get_rng_state', 'initial_seed', 'manual_seed', 'normal',
                'seed', 'set_rng_state', 'multi_head_attention_forward',
                'can_cast', 'compiled_with_cxx11_abi', 'get_default_dtype',
                'get_num_interop_threads', 'get_num_threads', 'load', 'seek',
                'promote_types', 'save', 'set_default_dtype',
                'set_default_tensor_type', 'set_flush_denormal',
                'set_num_interop_threads', 'set_num_threads',
                'set_printoptions']:
            pass
        elif func.__name__ in [
                'empty', 'ones', 'zeros', 'tril_indices', 'triu_indices',
                'rand', 'randn']:
            yield (2, 2), layout_device_kwargs
        elif func.__name__ in [
                'empty_like', 'ones_like', 'zeros_like', 'rand_like',
                'randn_like']:
            yield ((make_tensor((2, 2), **tensor_parameters),),
                   layout_device_kwargs)
        elif func.__name__ == 'empty_strided':
            yield ((3, 3), (2, 2)), layout_device_kwargs
        elif func.__name__ == 'eye':
            yield (2,), layout_device_kwargs
            yield (2, 3), layout_device_kwargs
        elif func.__name__ == 'from_numpy':
            if (tensor_parameters.get('device', 'cpu') == 'cpu'
                and tensor_parameters.get(
                    'layout', torch.strided) == torch.strided):
                yield (numpy.array([1, 2]),), {}
        elif func.__name__ == 'full':
            yield ((2, 2), 1.5), layout_device_kwargs
            yield ((2, 2), 0), dict(dtype=int, **layout_device_kwargs)
        elif func.__name__ == 'full_like':
            yield ((make_tensor((2, 2), **tensor_parameters), 1.5),
                   layout_device_kwargs)
            yield ((make_tensor((2, 2), **tensor_parameters), 0),
                   layout_device_kwargs)
        elif func.__name__ in ['linspace', 'logspace']:
            yield (1, 5), layout_device_kwargs
            yield (1, 5), dict(steps=10, **layout_device_kwargs)
        elif func.__name__ == 'sparse_coo_tensor':
            indices = make_tensor([[0, 1, 1], [2, 0, 2]],
                                  **layout_device_kwargs)
            values = make_tensor([0.1, 0.2, 0.3], **layout_device_kwargs)
            yield (indices, values, (2, 4)), device_kwargs
        elif func.__name__ == 'tensor':
            yield ([[1, 2], [3, 4]],), device_kwargs
            for tp, _ in self.iter_tensor_parameters():
                data = make_tensor((2, 2), **tp)
                yield (data,), device_kwargs
        elif func.__name__ in [
                'abs', 'absolute', 'acos', 'acosh',
                'angle', 'asin', 'asinh', 'atan', 'atanh', 'cos',
                'cosh', 'deg2rad', 'digamma', 'erf', 'erfc', 'erfinv',
                'exp', 'expm1', 'hardsigmoid', 'lgamma', 'log',
                'log10', 'log1p', 'log2', 'logit', 'rad2deg', 'rsqrt',
                'sigmoid', 'sin', 'sinh', 'sqrt', 'square', 'tan',
                'tanh', 'ceil', 'floor', 'frac', 'neg', 'reciprocal',
                'round', 'sign', 'trunc', 'det', 'geqrf', 'inverse',
                'logdet', 'lu', 'pinverse', 'qr', 'slogdet', 't',
                'trace', 'matrix_rank', 'tril', 'triu', 'clone',
                'detach', 'flatten', 'fliplr', 'flipud', 'numel',
                'unbind', 'argmax', 'argmin', 'mean', 'median',
                'mode', 'norm', 'std', 'std_mean', 'sum', 'unique',
                'unique_consecutive', 'var', 'var_mean', 'normalize',
                'count_nonzero', 'histc', 'isfinite', 'isinf',
                'isnan', 'max', 'min', 'nonzero', 'sort', 'pdist',
                'celu', 'elu', 'gelu', 'glu', 'gumbel_softmax',
                'hardshrink', 'hardswish', 'hardtanh', 'leaky_relu',
                'log_sigmoid', 'relu', 'relu6', 'rrelu', 'selu',
                'silu', 'softplus', 'softshrink', 'softsign',
                'tanhshrink', 'alpha_dropout', 'dropout', 'dropout2d',
                'feature_alpha_dropout', 'get_device', 'is_complex',
                'is_floating_point', 'is_signed', 'view_as_complex',
                'is_tensor',
                'selu_', 'rrelu_', 'relu_', 'leaky_relu_', 'hardtanh_',
                'elu_', 'celu_', 'isposinf', 'isneginf', 'nansum',
                'matrix_exp', 'atleast_1d', 'atleast_2d', 'atleast_3d',
                'signbit', 'arccosh']:
            # unary operations
            yield (make_tensor((2, 2), **tensor_parameters),), {}
        elif func.__name__ in [
                'imag', 'conj', 'real', 'view_as_real', 'isreal']:
            yield (make_tensor((2, 2), dtype=complex,
                               **tensor_parameters),), {}
        elif func.__name__ in [
                'atan2', 'logaddexp', 'logaddexp2',
                'mul', 'pow', 'cdist', 'dist', 'allclose', 'eq',
                'equal', 'ge', 'gt', 'le', 'lt', 'isclose', 'ne',
                'cosine_similarity', 'pairwise_distance',
                'binary_cross_entropy_with_logits', 'result_type',
                'polar', 'hypot', 'complex', 'nextafter']:
            # binary operations
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['gcd', 'lcm', 'true_divide']:
            yield (make_tensor((2, 2), dtype=int, **tensor_parameters),
                   make_tensor((2, 2), dtype=int, **tensor_parameters)), {}
        elif func.__name__ in ['mvlgamma']:
            yield ((make_tensor((2, 2), require_positive=True,
                                **tensor_parameters), 1), {})
        elif func.__name__ in ['polygamma']:
            yield (1, make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['rot90']:
            yield (make_tensor((2, 2), **tensor_parameters), 1, [0, 1]), {}
        elif func.__name__ in ['add', 'sub']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), {}
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), dict(alpha=1.5)
        elif func.__name__ in ['addcdiv', 'addcmul']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters),), {}
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), dict(value=1.5)
        elif func.__name__ in ['clamp', 'clip']:
            yield ((make_tensor((2, 2), **tensor_parameters),),
                   dict(min=-1, max=1))
        elif func.__name__ in ['quantile']:
            yield (make_tensor((2, 2), **tensor_parameters), 0.75), {}
        elif func.__name__ in ['div', 'floor_divide', 'fmod', 'remainder']:
            yield (make_tensor((2, 2), **tensor_parameters), 1.5), {}
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['bilinear']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2, 2), require_positive=True,
                               **tensor_parameters)), {}
        elif func.__name__ in ['linear']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), require_positive=True,
                               **tensor_parameters)), {}
        elif func.__name__ in ['cholesky', 'cholesky_inverse']:
            yield (make_tensor((2, 2), rdist='posdefined',
                               **tensor_parameters),), {}
        elif func.__name__ in ['cholesky_solve', 'solve']:
            yield (make_tensor((2, 2), rdist='posdefined',
                               **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['triangular_solve']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), rdist='posdefined',
                               **tensor_parameters)), {}
        elif func.__name__ in ['eig', 'symeig']:
            yield (make_tensor((2, 2), **tensor_parameters),), {}
            yield ((make_tensor((2, 2), **tensor_parameters),),
                   dict(eigenvectors=True))
        elif func.__name__ in ['lobpcg', 'pca_lowrank', 'svd',
                               'svd_lowrank']:
            yield (make_tensor((6, 6), **tensor_parameters),), {}
        elif func.__name__ in ['ger', 'outer']:
            yield (make_tensor((2,), **tensor_parameters),
                   make_tensor((2,), **tensor_parameters)), {}
        elif func.__name__ in ['lerp']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), require_positive=True,
                               **tensor_parameters)), {}
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters),
                   0.75), dict()
        elif func.__name__ in ['lstsq']:
            yield (make_tensor((5, 2), **tensor_parameters),
                   make_tensor((5, 2), **tensor_parameters)), {}
        elif func.__name__ in ['lu_solve']:
            A = make_tensor((2, 3, 3))
            b = make_tensor((2, 3, 1), **tensor_parameters)
            A_LU = torch.lu(A)
            A_LU = tuple(make_tensor(t, dtype=t.dtype, **tensor_parameters)
                         for t in A_LU)
            yield (b, ) + A_LU, {}
        elif func.__name__ in ['lu_unpack']:
            A = make_tensor((2, 3, 3))
            b = make_tensor((2, 3, 1), **tensor_parameters)
            A_LU = torch.lu(A)
            A_LU = tuple(make_tensor(t, dtype=t.dtype, **tensor_parameters)
                         for t in A_LU)
            yield A_LU, {}
        elif func.__name__ in ['orgqr']:
            output = torch.geqrf(make_tensor((2, 2)))
            output = tuple(make_tensor(t, dtype=t.dtype, **tensor_parameters)
                           for t in output)
            yield output, {}
        elif func.__name__ in ['ormqr']:
            output = torch.geqrf(make_tensor((2, 2)))
            output = tuple(make_tensor(t, dtype=t.dtype, **tensor_parameters)
                           for t in output)
            yield output + (make_tensor((2, 2), **tensor_parameters),), {}
        elif func.__name__ in ['trapz']:
            y = make_tensor((2, 3), **tensor_parameters)
            x = make_tensor([[1, 3, 4], [1, 2, 3]], **tensor_parameters)
            yield (y, x), {}
        elif func.__name__ in ['conv1d', 'conv_transpose1d']:
            yield (make_tensor((2, 2, 2), **tensor_parameters),
                   make_tensor((2, 2, 2), require_positive=True,
                               **tensor_parameters)), {}
        elif func.__name__ in ['conv2d', 'conv_transpose2d']:
            yield (make_tensor((2, 2, 2, 2), **tensor_parameters),
                   make_tensor((2, 2, 2, 2), require_positive=True,
                               **tensor_parameters)), {}
        elif func.__name__ in ['conv3d', 'conv_transpose3d']:
            yield (make_tensor((2, 2, 2, 2, 2), **tensor_parameters),
                   make_tensor((2, 2, 2, 2, 2), require_positive=True,
                               **tensor_parameters)), {}
        elif func.__name__ in ['conv_tbc']:
            yield (make_tensor((2, 2, 2), **tensor_parameters),
                   make_tensor((2, 2, 2), require_positive=True,
                               **tensor_parameters),
                   make_tensor((2,), **tensor_parameters)), {}
        elif func.__name__ in ['fold']:
            yield ((make_tensor((2, 2, 1), **tensor_parameters),),
                   dict(output_size=1, kernel_size=1))
        elif func.__name__ in ['unfold']:
            yield ((make_tensor((2, 2, 2, 2), **tensor_parameters),),
                   dict(kernel_size=1))
        elif func.__name__ in ['addbmm', 'baddbmm']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2, 2), **tensor_parameters),
                   make_tensor((2, 2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['addmm', 'chain_matmul']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['addmv']:
            yield (make_tensor((2,), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2,), **tensor_parameters)), {}
        elif func.__name__ in ['addr']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2,), **tensor_parameters),
                   make_tensor((2,), **tensor_parameters)), {}
        elif func.__name__ in ['bmm']:
            yield (make_tensor((2, 2, 2), **tensor_parameters),
                   make_tensor((2, 2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['cross']:
            yield (make_tensor((2, 3), **tensor_parameters),
                   make_tensor((2, 3), **tensor_parameters)), {}
        elif func.__name__ in ['dot', 'cartesian_prod']:
            yield (make_tensor((2,), **tensor_parameters),
                   make_tensor((2,), **tensor_parameters)), {}
        elif func.__name__ in ['matmul', 'mm', 'tensordot', 'block_diag',
                               'broadcast_tensors']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['matmul', 'mv']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2,), **tensor_parameters)), {}
        elif func.__name__ in ['matrix_power']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   2), {}
        elif func.__name__ in ['vander', 'squeeze']:
            yield (make_tensor((2, ), **tensor_parameters),), {}
        elif func.__name__ in ['cat', 'meshgrid']:
            yield ((make_tensor((2,), **tensor_parameters),
                    make_tensor((2,), **tensor_parameters)),), {}
        elif func.__name__ in ['chunk', 'unsafe_chunk']:
            yield (make_tensor((2, 2), **tensor_parameters), 2), {}
        elif func.__name__ in ['movedim']:
            yield (make_tensor((2, 2), **tensor_parameters), 0, 1), {}
        elif func.__name__ in ['combinations', 'bincount']:
            yield (make_tensor((2, ), dtype=int, **tensor_parameters),), {}
        elif func.__name__ in ['diag', 'diag_embed', 'diagflat', 'diagonal']:
            yield (make_tensor((2, 2), **tensor_parameters),), {}
            yield (make_tensor((2, 2), **tensor_parameters), 1), {}
        elif func.__name__ in ['flip']:
            yield (make_tensor((2, 2), **tensor_parameters), [0, 1]), {}
        elif func.__name__ in ['gather']:
            yield (make_tensor((2, 2), **tensor_parameters), 1,
                   make_tensor([[0, 0], [1, 0]], dtype=int,
                               **tensor_parameters)), {}
        elif func.__name__ in ['index_add', 'index_copy']:
            yield (make_tensor((2, 2), **tensor_parameters), 0,
                   make_tensor([0, 0], dtype=int, **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['index_fill']:
            yield (make_tensor((2, 2), **tensor_parameters), 0,
                   make_tensor([0, 0], dtype=int,
                               **tensor_parameters), 1.5), {}
        elif func.__name__ in ['index_put']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   (make_tensor([0, 0], dtype=int, **tensor_parameters), ),
                   make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['index_select']:
            yield (make_tensor((2, 2), **tensor_parameters), 0,
                   make_tensor([0, 0], dtype=int, **tensor_parameters)), {}
        elif func.__name__ in ['masked_fill']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), dtype=bool,
                               **tensor_parameters), 1.5), {}
        elif func.__name__ in ['masked_scatter']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), dtype=bool, **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['masked_select']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), dtype=bool, **tensor_parameters)), {}
        elif func.__name__ in ['narrow']:
            yield (make_tensor((2, 2), **tensor_parameters), 0, 0, 2), {}
        elif func.__name__ in ['repeat_interleave']:
            yield (make_tensor((2, 2), **tensor_parameters), 2), {}
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor([2], dtype=int, **tensor_parameters)), {}
        elif func.__name__ in ['reshape']:
            yield (make_tensor((2, 2), **tensor_parameters), (2, 2)), {}
            yield (make_tensor((2, 2), **tensor_parameters), (4, )), {}
        elif func.__name__ in ['roll', 'select']:
            yield (make_tensor((2, 2), **tensor_parameters), 0, 1), {}
        elif func.__name__ in ['scatter', 'scatter_add']:
            yield (make_tensor((2, 2), **tensor_parameters), 0,
                   make_tensor([[0], [0]], dtype=int, **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['split', 'unsqueeze', 'cummax', 'cummin',
                               'cumprod', 'cumsum', 'logcumsumexp',
                               'logsumexp', 'prod', 'argsort',
                               'kthvalue', 'topk',
                               'softmax', 'softmin', 'unsafe_split']:
            yield (make_tensor((2, 2), **tensor_parameters), 1), {}
        elif func.__name__ in ['stack', 'vstack', 'hstack', 'dstack']:
            yield ([make_tensor((2, 2), **tensor_parameters),
                    make_tensor((2, 2), **tensor_parameters)], ), {}
        elif func.__name__ in ['take']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor([0, 2], dtype=int, **tensor_parameters), ), {}
        elif func.__name__ in ['transpose']:
            yield (make_tensor((2, 2), **tensor_parameters), 0, 1), {}
        elif func.__name__ in ['where']:
            x = make_tensor((2, 2), **tensor_parameters)
            y = make_tensor((2, 2), **tensor_parameters)
            yield (make_tensor([[True, False], [False, True]], dtype=bool,
                               **tensor_parameters), x, y), {}
        elif func.__name__ in ['einsum']:
            yield ('i,j->ji', make_tensor((2,), **tensor_parameters),
                   make_tensor((2,), **tensor_parameters)), {}
        elif func.__name__ in ['batch_norm', 'instance_norm']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2,), **tensor_parameters),
                   make_tensor((2,), **tensor_parameters)), {}
        elif func.__name__ in ['group_norm']:
            yield (make_tensor((2, 2), **tensor_parameters), 1), {}
        elif func.__name__ in ['layer_norm']:
            yield (make_tensor((2, 2), **tensor_parameters), (2,)), {}
        elif func.__name__ in ['local_response_norm']:
            yield (make_tensor((2, 2, 2), **tensor_parameters), 1), {}
        elif func.__name__ in ['renorm']:
            yield (make_tensor((2, 2), **tensor_parameters), 1, 0, 5), {}
        elif func.__name__ in ['all', 'any', 'bitwise_not', 'logical_not']:
            yield (make_tensor((2, 2), dtype=bool, **tensor_parameters), ), {}
        elif func.__name__ in ['bitwise_and', 'bitwise_or', 'bitwise_xor',
                               'logical_and', 'logical_or', 'logical_xor']:
            yield (make_tensor((2, 2), dtype=bool, **tensor_parameters),
                   make_tensor((2, 2), dtype=bool, **tensor_parameters)), {}
        elif func.__name__ in ['bucketize']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2,), **tensor_parameters)), {}
        elif func.__name__ in ['is_nonzero']:
            yield (make_tensor((1, ), **tensor_parameters)), {}
        elif func.__name__ in ['searchsorted']:
            sorted_sequence = make_tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]],
                                          **tensor_parameters)
            values = make_tensor([[3, 6, 9], [3, 6, 9]], **tensor_parameters)
            yield (sorted_sequence, values,), {}
        elif func.__name__ in ['bernoulli']:
            yield (make_tensor((2, 2), rdist='uniform',
                               **tensor_parameters),), {}
        elif func.__name__ in ['multinomial']:
            yield (make_tensor((2, 2), rdist='uniform',
                               **tensor_parameters), 1), {}
        elif func.__name__ in ['poisson']:
            yield (make_tensor((2, 2), require_positive=True,
                               **tensor_parameters),), {}
        elif func.__name__ in ['fft', 'ifft', 'irfft', 'rfft', 'log_softmax']:
            yield (make_tensor((2, 2), **tensor_parameters), 1), {}
        elif func.__name__ in ['istft']:
            yield (make_tensor((2, 2, 2), **tensor_parameters), 2, 1), {}
        elif func.__name__ in ['stft']:
            yield (make_tensor((2, ), **tensor_parameters), 2, 1), {}
        elif func.__name__ in ['binary_cross_entropy']:
            yield (make_tensor((2, 2), rdist='uniform', **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters),), {}
        elif func.__name__ in ['cross_entropy']:
            yield (make_tensor((2, 2), rdist='uniform', **tensor_parameters),
                   make_tensor([1, 1], dtype=int, **tensor_parameters),), {}
        elif func.__name__ in ['cosine_embedding_loss', 'margin_ranking_loss',
                               'triplet_margin_loss']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['ctc_loss']:
            yield (make_tensor((2, 2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters),
                   make_tensor([2, 2], dtype=int, **tensor_parameters),
                   make_tensor([2, 2], dtype=int, **tensor_parameters)), {}
        elif func.__name__ in ['hinge_embedding_loss', 'kl_div', 'l1_loss',
                               'mse_loss', 'multilabel_soft_margin_loss',
                               'poisson_nll_loss', 'smooth_l1_loss',
                               'soft_margin_loss']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['prelu']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor((2,), **tensor_parameters)), {}
        elif func.__name__ in ['multi_margin_loss', 'nll_loss']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor([1, 1], dtype=int, **tensor_parameters)), {}
        elif func.__name__ in ['multilabel_margin_loss']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   make_tensor([[0, 0], [0, 0]],
                               dtype=int, **tensor_parameters)), {}
        elif func.__name__ in ['_pad', '_pad_circular']:
            yield (make_tensor((2, 2, 2), **tensor_parameters),
                   (2, 2)), {}
        elif func.__name__ in ['affine_grid']:
            yield (make_tensor((2, 2, 3), **tensor_parameters),
                   (2, 2, 2, 2)), {}
        elif func.__name__ in ['channel_shuffle']:
            yield (make_tensor((2, 2, 2), **tensor_parameters), 1), {}
        elif func.__name__ in ['grid_sample']:
            yield (make_tensor((2, 2, 2, 2), **tensor_parameters),
                   make_tensor((2, 2, 2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['interpolate', 'pixel_shuffle', 'upsample',
                               'upsample_bilinear', 'upsample_nearest']:
            yield (make_tensor((2, 2, 2, 2), **tensor_parameters), 1), {}
        elif func.__name__ in ['threshold', 'threshold_']:
            yield (make_tensor((2, 2), **tensor_parameters),
                   0.75, 0.5), {}
        elif func.__name__ in ['_adaptive_max_pool1d', 'adaptive_avg_pool1d',
                               'adaptive_max_pool1d',
                               'adaptive_max_pool1d_with_indices']:
            yield ((make_tensor((2, 2, 2), **tensor_parameters),),
                   dict(output_size=(2,)))
        elif func.__name__ in ['_adaptive_max_pool2d',
                               'adaptive_avg_pool2d',
                               'adaptive_max_pool2d',
                               'adaptive_max_pool2d_with_indices']:
            yield ((make_tensor((2, 2, 2, 2), **tensor_parameters),),
                   dict(output_size=(2, 2)))
        elif func.__name__ in ['_adaptive_max_pool3d', 'adaptive_avg_pool3d',
                               'adaptive_max_pool3d',
                               'adaptive_max_pool3d_with_indices']:
            yield ((make_tensor((2, 2, 2, 2), **tensor_parameters),),
                   dict(output_size=(2, 2, 2)))
        elif func.__name__ in ['_fractional_max_pool2d',
                               'fractional_max_pool2d',
                               'fractional_max_pool2d_with_indices']:
            yield ((make_tensor((2, 2, 2, 2), **tensor_parameters),),
                   dict(output_size=(2, 2), kernel_size=(1, 1)))
        elif func.__name__ in ['_fractional_max_pool3d',
                               'fractional_max_pool3d',
                               'fractional_max_pool3d_with_indices']:
            yield ((make_tensor((3, 3, 3, 3, 3), **tensor_parameters),),
                   dict(output_size=(2, 2, 2), kernel_size=(1, 1, 1)))
        elif func.__name__ in ['_max_pool1d', 'max_pool1d',
                               'max_pool1d_with_indices', 'avg_pool1d']:
            yield ((make_tensor((2, 2, 2), **tensor_parameters),),
                   dict(kernel_size=(1,)))
        elif func.__name__ in ['_max_pool2d', 'max_pool2d',
                               'max_pool2d_with_indices', 'avg_pool2d']:
            yield ((make_tensor((2, 2, 2, 2), **tensor_parameters),),
                   dict(kernel_size=(1, 1)))
        elif func.__name__ in ['_max_pool3d', 'max_pool3d',
                               'max_pool3d_with_indices', 'avg_pool3d']:
            yield ((make_tensor((2, 2, 2, 2, 2), **tensor_parameters),),
                   dict(kernel_size=(1, 1, 1)))
        elif func.__name__ in ['lp_pool1d']:
            yield ((make_tensor((2, 2, 2), **tensor_parameters), 0.75),
                   dict(kernel_size=1))
        elif func.__name__ in ['lp_pool2d']:
            yield ((make_tensor((2, 2, 2, 2), **tensor_parameters), 0.75),
                   dict(kernel_size=(1, 1)))
        elif func.__name__ in ['lp_pool3d']:
            yield ((make_tensor((2, 2, 2, 2, 2), **tensor_parameters), 0.75),
                   dict(kernel_size=(1, 1, 1)))
        elif func.__name__ in ['max_unpool1d']:
            output = torch.max_pool1d_with_indices(
                make_tensor((2, 2, 2)), kernel_size=(1,))
            output = tuple(make_tensor(t, dtype=t.dtype, **tensor_parameters)
                           for t in output)
            yield output, dict(kernel_size=(1,))
        elif func.__name__ in ['max_unpool2d']:
            output = torch.nn.functional.max_pool2d_with_indices(
                make_tensor((2, 2, 2, 2)), kernel_size=(1, 1))
            output = tuple(make_tensor(t, dtype=t.dtype, **tensor_parameters)
                           for t in output)
            yield output, dict(kernel_size=(1, 1))
        elif func.__name__ in ['max_unpool3d']:
            output = torch.nn.functional.max_pool3d_with_indices(
                make_tensor((2, 2, 2, 2, 2)), kernel_size=(1, 1, 1))
            output = tuple(make_tensor(t, dtype=t.dtype,
                                       **tensor_parameters) for t in output)
            yield output, dict(kernel_size=(1, 1, 1))
        elif func.__name__ in ['dropout3d']:
            yield (make_tensor((2, 2, 2), **tensor_parameters),), {}
        elif func.__name__ in ['embedding', 'embedding_bag']:
            yield (make_tensor([[0, 0], [0, 0]], dtype=int,
                               **tensor_parameters),
                   make_tensor((2, 2), **tensor_parameters)), {}
        elif func.__name__ in ['one_hot']:
            yield (make_tensor((2, 2), dtype=int, **tensor_parameters),), {}
        elif func.__name__ in ['int_repr']:
            # cannot create quantized sparse tensor
            if tensor_parameters.get('layout', torch.strided) == torch.strided:
                yield (make_tensor((2, 2), dtype=torch.qint32,
                                   **tensor_parameters),), {}
        elif func.__name__ in ['q_per_channel_axis',
                               'q_per_channel_scales',
                               'q_per_channel_zero_points', 'q_scale',
                               'q_zero_point',
                               'quantized_lstm', 'quantized_gru']:
            pass
        else:
            print('-'*80)
            print(func.__name__)
            print(func.__doc__)
            yield NotImplemented, None

    def get_status(self, func, sig, tensor_parameters):
        """Return the support (status, state, fail_details) tuple of a
        function for given tensor parameters.
        """
        success_count = 0
        fail_count = 0
        fail_details = []
        notests = True
        for args, kwargs in self.iter_func_args(func, sig, tensor_parameters):
            notests = False
            if args is NotImplemented:
                return 'NOTIMPL', 'not impl', []
                break
            try:
                if tensor_parameters.get('requires_grad'):
                    func(*args, **kwargs)
                else:
                    func(*args, **kwargs)
                success_count += 1
            except RuntimeError as msg:
                fail_count += 1
                detail = str(msg).splitlines()[0]
                if re.match('.+? is only available for these backends:',
                            detail):
                    detail = 'RuntimeError: backend not supported'
                elif re.match('.+? is not implemented for .+? layout',
                              detail):
                    detail = 'RuntimeError: not implemented for layout'
                elif re.match('unsupported tensor layout: .+', detail):
                    detail = 'RuntimeError: unsupported layout'
                elif re.match(r'expected \w+ to be a .+?, but got \w+ of'
                              r' layout \w+', detail):
                    detail = 'RuntimeError: unexpected layout'
                elif re.match(r'\w+ tensors do not have strides', detail):
                    detail = 'RuntimeError: no strides'
                elif re.match('sparse tensors do not have is_contiguous',
                              detail):
                    detail = 'RuntimeError: no is_contiguous'
                elif re.match('.+? must be dense', detail):
                    detail = 'RuntimeError: operand must be dense'
                elif func.__name__ in ['div', 'floor_divide', 'true_divide']:
                    detail = f'RuntimeError: {detail}'
                elif re.match('memory format option is only supported'
                              ' by strided tensors', detail):
                    detail = 'RuntimeError: memory format option not supported'
                elif re.match("Could not run.*?memory_format['] is only"
                              " available for these backends.*", detail):
                    detail = 'RuntimeError: memory format unavailable'
                elif re.match(r'\w+ tensors do not have storage', detail):
                    detail = 'RuntimeError: no storage'
                elif re.match(r'reshape is not implemented for \w+ tensors',
                              detail):
                    detail = 'RuntimeError: unimplemented reshape'
                elif re.match(r'unsupported memory format option \w+', detail):
                    detail = 'RuntimeError: unsupported memory format option'
                elif (re.match(r'.+? only supports strided layout.*', detail)
                      or re.match(
                          r'.*?expected .+? to have torch[.]strided layout.*',
                          detail)):
                    detail = 'RuntimeError: requires strided layout'

                fail_details.append(detail)
            except Exception as msg:
                fail_count += 1
                detail = str(msg).splitlines()[0]
                fail_details.append(f'{type(msg).__name__}: {detail}')

        if notests:
            return 'SKIPPED', 'skipped', []
        count = fail_count + success_count
        status, state = '?', ''
        if count == 0:
            fail_details.append('untested')
            status, state = 'UNTESTED', 'untested'
        elif fail_count == 0:
            status, state = 'PASSED', 'passed'
        elif success_count == 0:
            status, state = 'FAILED', 'failed'
        else:
            status, state = 'PARTIAL', f'{success_count}/{count} passed'
        if fail_details:
            nl = '\n  '
            print(f'{func.__module__}.{func.__name__}:'
                  f'{nl}{nl.join(fail_details)}')
        return status, state, fail_details

    def make_sparse_support_state(self):
        tensor_parameters_list, titles = zip(*self.iter_tensor_parameters())
        lines = []
        section_title_status = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int)))
        title_status = defaultdict(lambda: defaultdict(int))
        detail_title = defaultdict(lambda: defaultdict(int))
        detail_total = defaultdict(int)
        for section_id, section in enumerate(self.iter_sections()):
            funcs = []
            if self.config.has_option(section, '_funcs'):
                funcs = list(self.config.get(section, '_funcs'))
            funcs1 = sorted((func.__name__, str(sig), i)
                            for i, (func, sig) in enumerate(funcs))
            funcs = list(funcs[i] for _, _, i in funcs1)

            if self.verbose:
                print(f'section: {section}')
            section_title = self.config.get(section, 'title')

            f = io.StringIO('')
            headers = ['Function'] + list(titles)
            lst = []
            for func, sig in funcs:
                dname = get_docname(func, sig)
                attr = self.attrs.get(func.__name__)
                if attr is not None:
                    dname += ' ' + attr
                row = [dname]
                for tensor_parameters, title in zip(
                        tensor_parameters_list, titles):
                    status, state, fail_details \
                        = self.get_status(func, sig, tensor_parameters)
                    fail_details = list(set(fail_details))
                    detail = ';'.join(fail_details) or status
                    row.append(detail)
                    section_title_status[section][title][status] += 1
                    detail_title[detail][title] += 1
                    detail_total[detail] += 1
                lst.append(row)

            level = section.count('/') + 1
            lines.append(f'\n\n{level * "#"} {section_title}\n')

            if lst:
                text_utils.table(f, lst, list(range(len(headers))), headers)
                lines.append(f.getvalue())

        f = io.StringIO('')
        headers = ['Section'] + list(titles)
        lst = []
        for section in self.iter_sections():
            section_title = self.config.get(section, 'title')
            row = []
            row.append(get_secname(section_title))
            for title in titles:
                l1 = []
                for status, count in (section_title_status[section][title]
                                      .items()):
                    if count:
                        l1.append(f'{status}: {count}')
                        title_status[title][status] += count
                row.append(', '.join(sorted(l1)))
            lst.append(row)
        row = ['Total']
        for title in titles:
            l1 = []
            for status, count in title_status[title].items():
                if count:
                    l1.append(f'{status}: {count}')
            row.append(', '.join(sorted(l1)))
        lst.append(row)
        text_utils.table(f, lst, list(range(len(headers))), headers)
        namespaces = '\n'.join([f'- {m.__name__}' for m in modules])

        lines_header = [
            f'''
This file is auto-generated, do not edit!

# The state of PyTorch tensor layouts support

The following table summarizes the state of PyTorch tensor layouts for
different PyTorch functions from the following namespaces:
{namespaces}\n''',
            f.getvalue(),
            '''
The functions and possible failure messages are listed in the tables
of subsequent sections.

## Ranking of failures

The following table lists the ranking of failure messages:\n''',
        ]

        f = io.StringIO('')
        headers = ['Status detail'] + list(titles)
        lst = []

        for total, detail in reversed(
                sorted((v, k) for k, v in detail_total.items())):
            if detail == 'SKIPPED':
                continue
            if total <= 1:
                # don't show single failures
                break
            row = [detail]
            for title in titles:
                row.append(str(detail_title[detail][title]))
            lst.append(row)
        text_utils.table(f, lst, list(range(len(headers))), headers)
        lines_header.append(f.getvalue())

        #
        lines = lines_header + lines

        fn = os.path.join(self.working_dir, 'SparseSupportState.md')
        print(f'Creating {fn}')
        f = open(fn, 'w')
        f.write('\n'.join(lines))
        f.close()


def main(working_dir=None):
    if working_dir is None:
        working_dir = os.path.dirname(__file__)
    classifier = Classifier.fromfile('pytorch_functions.ini',
                                     working_dir=working_dir)
    functions = list(set(all_functions()))
    for _, i in sorted([(func.__name__, i)
                        for i, func in enumerate(functions)]):
        func = functions[i]
        section = classifier.attach(func, None)
        if section is None:
            print(f'TODO: add {func.__name__} from {func.__module__} to'
                  f' {working_dir or "."}/pytorch_functions.ini')
    classifier.make_sparse_support_state()


if __name__ == '__main__':
    main()
