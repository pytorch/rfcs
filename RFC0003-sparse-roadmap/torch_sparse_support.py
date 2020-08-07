
import io
import os
import sys
import inspect
import itertools
import torch
import numpy
import typing
import configparser
import torch_signatures
from torch_signatures import scan_module
import text_utils
from collections import defaultdict, abc

modules = [torch, torch.nn.functional, torch.sparse]

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
    default_int = 1
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
    if fname in ['channel_shuffle']:
        size = (2, 2, 2)
        default_int = 2
    if fname == 'conv_tbc':
        size = (2, 2, 2)
    args = []
    for argname, params in sig.parameters.items():
        if params.default is not inspect.Parameter.empty:
            break
        annot = params.annotation
        if annot is torch.Tensor:
            args.append(make_tensor(size, layout, dtype=dtype, rdist=rdist))
        elif annot is int:
            args.append(default_int)
        else:
            raise NotImplementedError(f'{annot}')
    return tuple(args), extra_kwargs


def get_seclink(section_title):
    l = section_title.lower().replace('/', '').split()
    return '#' + '-'.join(l)

def get_secname(section_title):
    return f'<a href="{get_seclink(section_title)}">{section_title}</a>'


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
        _cache.extend(scan_module(module=modules))
    for func, sig in set(_cache):
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
            if isinstance(predicate, (list, tuple)):
                for p in predicate:
                    if p(*args):
                        return True
            else:
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
    count = 0
    for name, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            break
        if not (param.annotation == torch.Tensor):
            return False
        count += 1
    return count > 0


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


def get_required_atypes(sig):
    lst = []
    for aname, param in sig.parameters.items():
        if param.default is not inspect.Parameter.empty:
            break
        lst.append(param)
    return tuple(lst)
        
def variants(typ):
    if typ in [torch.CompilationUnit, torch.ExtraFilesMap,
                 torch.dtype, numpy.ndarray]:
        yield typ
        return
    elif isinstance(typ, tuple):
        for v in itertools.product(*(tuple(variants(a)) for a in typ)):
            yield v
        return
    elif isinstance(typ, inspect.Parameter):
        if typ.annotation is not typ.empty:
            for v in variants(typ.annotation):
                yield v
        else:
            yield typ
        return
    elif isinstance(typ, typing._Final):
        if isinstance(typ, typing.TypeVar):
            yield  typ
        elif typ.__origin__ == typing.Union:
            for t in typ.__args__:
                for v in variants(t):
                    yield v
        elif typ.__origin__ == tuple:
            if typ.__args__ is None:
                yield typ
            else:
                for v in itertools.product(*(tuple(variants(a)) for a in typ.__args__)):
                    yield typing.Tuple[v]
        elif typ.__origin__ == list:
            if typ.__args__ is None:
                yield typ
            else:
                for v in itertools.product(*(tuple(variants(a)) for a in typ.__args__)):
                    yield typing.List[v]
        elif typ.__origin__ == abc.Sequence:
            if typ.__args__ is None:
                yield typ
            else:
                for v in itertools.product(*(tuple(variants(a)) for a in typ.__args__)):
                    yield typing.Sequence[v]
        elif typ.__origin__ == abc.Callable:
            assert typ.__args__ == ()  # otherwise not implemented
            yield typ
        elif typ.__origin__ == type:
            yield typ
        else:
            raise NotImplementedError((typ, type(typ), type(typ).__bases__))
    elif typ is Ellipsis:
        yield typ
    elif isinstance(typ, type):
        yield typ
    elif isinstance(typ, torch_signatures.Name):
        yield typ
    else:
        raise NotImplementedError((typ, type(typ), type(typ).__bases__))

def has_tensor(typ):
    if typ == torch.Tensor:
        return True
    elif isinstance(typ, typing._Final) and typ.__origin__ == typing.Union:
        for t in typ.__args__:
            if has_tensor(t):
                return True
    return False

def has_int(typ):
    if typ == int:
        return True
    elif isinstance(typ, typing._Final) and typ.__origin__ == typing.Union:
        for t in typ.__args__:
            if has_int(t):
                return True
    return False


def make_test_value(typ):
    if typ == torch.Tensor:
        return f'make_{typ.__name__}(shape=(default_int, default_int), dtype=default_dtype, layout=default_layout, device=default_device)'
    if isinstance(typ, type):
        return f'default_{typ.__name__}'
    if isinstance(typ, typing._Final):
        if typ.__origin__ == tuple:
            if typ.__args__ is None:
                return 'default_tuple'
            lst = []
            for atyp in typ.__args__:
                if atyp == Ellipsis:
                    lst = lst + lst
                    continue
                lst.append(make_test_value(atyp))
            return '('+', '.join(lst)+',)'
        elif typ.__origin__ == list:
            if typ.__args__ is None:
                return 'default_list'
            return '['+', '.join(map(make_test_value, typ.__args__))+',]'
    if str(typ) == '*tensors':
        return make_test_value(typing.Tuple[torch.Tensor, torch.Tensor])[1:-1]
    if str(typ) == '*matrices':
        return make_test_value(typing.Tuple[torch.Tensor, torch.Tensor])[1:-1]
    if str(typ) == '*operands':
        return make_test_value(typing.Tuple[torch.Tensor, torch.Tensor])[1:-1]
    if str(typ) == '*size':
        return make_test_value(typing.Tuple[int, int])[1:-1]
    if str(typ) == '*args':
        return '*tuple()'
    if str(typ) == '**kwargs':
        return '**dict()'
    if typ == 'torch.dtype':
        return 'default_dtype'
    if typ == typing.Type[torch.Tensor]:
        return 'default_Tensor_type'
    if typ == typing.Callable:
        return 'default_callable'
    if isinstance(typ, inspect.Parameter):
        return f'{typ}'
    raise NotImplementedError((typ, type(typ), str(typ)))

def ops2set(operations):
    return set(w for w in operations.strip().split() if w)



def make_classification_file(working_dir=None):   # OBSOLETE
    """
    Classifiers:
    notensor - a function with no Tensor input nor output
    constructor - a function that constructs new Tensor instances
    inplace - a function that changes tensor inplace
    unary - a function that represents an unary operation
    binary - a function that represents a binary operation
    reduction - a function that represents a reduction
    elementwise - a function that is applied to tensor elementwise
    array - a function that represents an array operation
    """
    
    if working_dir is None:
        working_dir = os.path.dirname(__file__)
    classifier = Classifier.fromfile('pytorch_functions.ini', working_dir=working_dir)

    count_classified = 0
    count_unclassified = 0
    lst = []
    for func, sig in all_functions():
        fullname = func.__module__ + '.' + func.__name__
        sig_str = str(sig)
        classifiers = classifier.get_classifiers(func, sig)
        continue
        section = classifier.get_section(func.__name__)
        if section is not None:
            classifiers.append(section)
        
        if 'Tensor' not in sig_str:
            classifiers.append('notensor')
        if 0:
            if 'torch.layout' in sig_str:
                classifiers.append('haslayout')
            if 'torch.device' in sig_str:
                classifiers.append('hasdevice')

        if ('torch.layout' in sig_str or 'requires_grad' in sig_str or 'torch.device' in sig_str):
            if sig.return_annotation == torch.Tensor:
                classifiers.append('constructor')

        if func.__name__.endswith('_'):
            classifiers.append('inplace')

        req_atypes = get_required_atypes(sig)

        if (0 and len(req_atypes) == 1
            and req_atypes[0].name == 'input'
            and has_tensor(req_atypes[0].annotation)
            and sig.return_annotation == torch.Tensor
            and [param for name, param in sig.parameters.items() if name in ['dim', 'dims'] and has_int(param.annotation)]
            and 'array' not in classifiers
        ):
            classifiers.append('reduction')
        
        if (0 and len(req_atypes) == 1
            and req_atypes[0].name == 'input'
            and has_tensor(req_atypes[0].annotation)
            and sig.return_annotation == torch.Tensor
            and 'reduction' not in classifiers
            and 'array' not in classifiers
        ):
            classifiers.append('unary')
        if (0 and len(req_atypes) == 2
            and req_atypes[0].name == 'input'
            and req_atypes[1].name == 'other'
            and has_tensor(req_atypes[0].annotation)
            and has_tensor(req_atypes[1].annotation)
            and sig.return_annotation == torch.Tensor
            and 'array' not in classifiers
        ):
            classifiers.append('binary')

        if classifiers:
            print(fullname, sig)
            print('  classifiers:', ', '.join(classifiers))    
            count_classified += 1
            continue
        count_unclassified += 1
        print(fullname, sig)
        #print('  classifiers:', ', '.join(classifiers))    
        #for atypes in variants(req_atypes):
        #    print('  sample:', ', '.join(map(make_test_value, atypes)))

    print(f'unclassified/classified counts={count_unclassified}/{count_classified}')


def main(working_dir=None):  # OBSOLETE
    layouts = get_layouts()
    if working_dir is None:
        working_dir = os.path.dirname(__file__)

    func_cache = set()
        
    predicates = []

    f = io.StringIO('')
    headers = ['Function'] + list(map(str, layouts)) + ['Signature']

    f.write('# Tensor constructors\n\n')
    
    f.write('## Functions with layout argument\n\n')

    lst = []
    failures = defaultdict(list)
    predicates.append(has_layout)
    for func, sig in all_functions(predicates[-1]):
        if func in func_cache: continue
        func_cache.add(func)
        row = [get_docname(func, sig)]
        for layout in layouts:
            row.append(try_layout(func, sig, layout))
        row.append(str(sig))
        lst.append(row)
    text_utils.table(f, lst, list(range(len(headers))), headers)

    f.write('\n\n')

    f.write('## Functions with tensor inputs\n\n')

    lst = []
    predicates.append(has_all_tensor_inputs)
    for func, sig in all_functions(predicates[-1]):
        if func in func_cache: continue
        func_cache.add(func)
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
        row.append(str(sig))
        lst.append(row)

    text_utils.table(f, lst, list(range(len(headers))), headers)

    f.write('## Functions with tensor input\n\n')

    lst = []
    predicates.append(has_tensor_input)
    for func, sig in all_functions(predicates[-1]):
        if func in func_cache: continue
        func_cache.add(func)
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
        row.append(str(sig))
        lst.append(row)

    text_utils.table(f, lst, list(range(len(headers))), headers)
    
    f.write('# Functions not covered above\n\n')

    lst = []
    failures = defaultdict(list)
    for func, sig in all_functions():
        if func in func_cache: continue
        func_cache.add(func)
        row = [get_docname(func, sig)]
        row.append(str(sig))
        lst.append(row)

    text_utils.table(f, lst, list(range(2)), headers[:1] + headers[-1:])

    f.write('\n\n')
    s = f.getvalue()
    f = open(os.path.join(working_dir, 'SparseSupportState.md'), 'w')
    f.write(s)
    f.close()


class Classifier:

    @classmethod
    def fromfile(cls, filename, working_dir=None):
        if working_dir is None:
            working_dir = os.path.dirname(__file__)
        ini_file = os.path.join(working_dir, filename)
        config = configparser.RawConfigParser()
        config.read([ini_file])
        return cls(config)

    def __init__(self, config, working_dir=None):
        if working_dir is None:
            working_dir = os.path.dirname(__file__)
        # replace names string content with a list
        for section in config.sections():
            if config.has_option(section, 'names'):
                names = config.get(section, 'names')
                names = list(w for w in names.strip().split() if w)
                config.set(section, 'names', names)
        #
        self.working_dir = working_dir
        self.config = config

    def get_section(self, func, sig):
        for section in self.config.sections():
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

    def get_classifiers(self, func, sig):
        sig_str = str(sig)
        
        classifiers = []
        section = self.get_section(func, sig)
        if section is not None:
            classifiers.append(section)

        if 'Tensor' not in sig_str:
            classifiers.append('notensor')

        if func.__name__.endswith('_'):
            classifiers.append('inplace')

        if not classifiers:
            print(f'no classifiers for {func.__name__}')

        return classifiers

    def make_sparse_support_state(self):
        layouts = get_layouts()
        lines = []
        section_layout_ok_lst = defaultdict(lambda : defaultdict(int))
        section_layout_skip_lst = defaultdict(lambda : defaultdict(int))
        section_layout_fail_lst = defaultdict(lambda : defaultdict(int))
        total_layout_ok_lst = defaultdict(int)
        total_layout_skip_lst = defaultdict(int)
        total_layout_fail_lst = defaultdict(int)
        for section in self.config.sections():
            funcs = []
            if self.config.has_option(section, '_funcs'):
                funcs = self.config.get(section, '_funcs')

            print(f'section: {section}')
            section_title = self.config.get(section, 'title')

            skips = []
            if self.config.has_option(section, 'skip_try_layout'):
                skips = self.config.get(section, 'skip_try_layout')
                skips = list(w for w in skips.strip().split() if w)

            f = io.StringIO('')
            headers = ['Function'] + list(map(str, layouts)) + ['Signature']
            lst = []
            for func, sig in funcs:
                row = [get_docname(func, sig)]
                for layout in layouts:
                    #print(f'  try {layout}: {func.__name__}')
                    if func.__name__ in skips:
                        status = 'SKIP'
                    else:
                        status = try_layout(func, sig, layout)
                    row.append(status)
                    if status == 'OK':
                        section_layout_ok_lst[section][layout] += 1
                    elif status == 'SKIP':
                        section_layout_skip_lst[section][layout] += 1
                    else:
                        section_layout_fail_lst[section][layout] += 1
                row.append(str(sig))
                lst.append(row)                


            level = section.count('/') + 1
            lines.append(f'\n\n{level * "#"} {section_title}\n')

            if lst:
                text_utils.table(f, lst, list(range(len(headers))), headers)
                lines.append(f.getvalue())

        f = io.StringIO('')
        headers = ['Section'] + list(map(str, layouts))
        lst = []
        for section in self.config.sections():
            section_title = self.config.get(section, 'title')
            row = []
            row.append(get_secname(section_title))
            for layout in layouts:
                l = []
                v = section_layout_ok_lst[section][layout]
                if v:
                    total_layout_ok_lst[layout] += v
                    l.append(f'{v} passed')
                v = section_layout_fail_lst[section][layout]
                if v:
                    total_layout_fail_lst[layout] += v
                    l.append(f'{v} failed')
                v = section_layout_skip_lst[section][layout]
                if v:
                    total_layout_skip_lst[layout] += v
                    l.append(f'{v} skipped')
                row.append(', '.join(l))
            lst.append(row)
        row = ['Total']
        for layout in layouts:
            l = []
            v = total_layout_ok_lst[layout]
            if v:
                l.append(f'{v} passed')
            v = total_layout_fail_lst[layout]
            if v:
                l.append(f'{v} failed')
            v = total_layout_skip_lst[layout]
            if v:
                l.append(f'{v} skipped')
            row.append(', '.join(l))
        lst.append(row)

        text_utils.table(f, lst, list(range(len(headers))), headers)

        namespaces = '\n'.join([f'- {m.__name__}' for m in modules])

        lines_header = [
            f'''
This file is auto-generated, do not edit!

# The state of PyTorch tensor layouts support

The following table summarizes the state of PyTorch tensor layouts for
different PyTorch functions from the following namespaces:
{namespaces}

The functions and possible failure messages are listed in the tables
of subsequent sections.

            ''',
            f.getvalue(),
            '''

Notes:
* TODO: all tests for strided layout should pass
* TODO: enable tests for different devices: CPU, CUDA
            '''
        ]
        lines = lines_header + lines

        fn = os.path.join(self.working_dir, 'SparseSupportState.md')
        print(f'Creating {fn}')
        f = open(fn, 'w')
        f.write('\n'.join(lines))
        f.close()


def main2(working_dir=None):
    if working_dir is None:
        working_dir = os.path.dirname(__file__)
    classifier = Classifier.fromfile('pytorch_functions.ini', working_dir=working_dir)
    for func, sig in all_functions():
        classifiers = classifier.get_classifiers(func, sig)

    classifier.make_sparse_support_state()

if __name__ == '__main__':
    main2()
    #make_classification_file()
