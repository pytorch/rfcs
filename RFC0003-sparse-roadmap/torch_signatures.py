"""
get the signatures of torch functions using their user-facing doc strings
"""
# Author: Pearu Peterson
# Created: July 2020

import re
import inspect
import types
import typing
import torch

Value = typing.Union[int, float, complex, bool]

def resolve_annotation(annot, orig_line, name):
    if 'optional' in annot:
        annot = annot.replace(', optional', '')
        annot = annot.replace(',optional', '')
    if annot.startswith(':class:`'):
        annot = annot[8:-1].strip()
    if annot == 'None':
        return None
    if annot == 'object':
        return object
    if annot in ['``int``', 'int']:
        return int
    if annot in ['boolean', 'bool', '(bool)']:
        return bool
    if annot == 'tuple':
        return tuple
    if annot in ['string', 'str']:
        return str
    if annot == 'float':
        return float
    if annot == 'int...':
        return typing.Sequence[int]
    if annot == 'seq':
        return typing.Sequence
    if annot == 'array_like':
        return typing.Sequence
    if annot in ['tuple of ints', 'Tuple[int]']:
        return typing.Tuple[int]
    if annot == 'int or tuple of ints':
        return typing.Union[int, typing.Tuple[int]]
    if annot == 'tuple or ints':  # should be 'tuple of ints'
        return typing.Tuple[int]
    if annot == 'a list or tuple':
        return typing.Union[typing.List, typing.Tuple]
    if annot == 'List of Tensors':
        return typing.List[torch.Tensor]
    if annot == 'sequence of Tensors':
        return typing.Sequence[torch.Tensor]
    if annot == 'LongTensor or tuple of LongTensors':
        return typing.Union[torch.LongTensor, typing.Tuple[torch.LongTensor]]
    if annot == 'tuple of LongTensor':
        return typing.Tuple[torch.LongTensor]
    if annot == 'Tensor':
        return torch.Tensor
    if annot == 'LongTensor':
        return torch.LongTensor
    if annot == 'IntTensor':
        return torch.IntTensor
    if annot == 'BoolTensor':
        return torch.BoolTensor
    if annot == 'Generator':
        return torch.Generator
    if annot == 'dtype':
        return torch.dtype
    if annot in ('Tensor or float', 'float or tensor'):
        return typing.Union[torch.Tensor, float]
    if annot == 'Number':
        return typing.Union[int, float, complex]
    if annot == 'Value':
        return Value
    if annot == 'Tensor or Number':
        return typing.Union[torch.Tensor, int, float, complex]
    if annot == 'Tensor or int':
        return typing.Union[torch.Tensor, int]
    if annot == 'Tensor or Scalar':
        return typing.Union[torch.Tensor, int, float, complex, bool]
    if annot == '(Tensor, LongTensor)':
        return typing.Tuple[torch.Tensor, torch.LongTensor]
    if annot == '(Tensor, Tensor)':
        return typing.Tuple[torch.Tensor, torch.Tensor]
    if annot == '(Tensor, Tensor, Tensor)':
        return typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    if annot == 'list, tuple, or :class:`torch.Size`':
        return typing.Union[list, tuple, torch.Size]
    if annot == 'list or :class:`torch.Size`':
        return typing.Union[list, torch.Size]
    if annot == 'c10::FunctionSchema':
        return torch.FunctionSchema
    if annot == 'torch::jit::Graph':
        return torch.Graph
    if annot == 'List[List[torch.autograd.ProfilerEvent]]':
        return typing.List[typing.List[torch.autograd.ProfilerEvent]]
    if annot == 'List[at::Tensor]':
        return typing.List[torch.Tensor]
    if annot == 'List[int]':
        return typing.List[int]
    if annot == 'List[List[int]]':
        return typing.List[typing.List[int]]
    if annot == 'List[bool]':
        return typing.List[bool]
    if annot == 'List[torch.distributed.ProcessGroup]':
        return typing.List[torch.distributed.ProcessGroup]
    if annot.startswith('torch.'):
        return eval(annot, dict(torch=torch), {})
    if annot.startswith('Device ordinal (Integer)'):
        return int
    raise NotImplementedError((annot, orig_line, name))

def get_signature_from_doc(name, member, doc, first_arg=''):
    doc = doc.lstrip()
    if name == 'pad':
        if '->' not in doc:
            doc = "pad(input, pad:Tuple[int], mode:str='constant', value:Value=0) -> Tensor\n" + doc
    if name == 'device':
        if '->' not in doc:
            doc = "device(device) -> None\n" + doc
    if name == 'stream':
        if '->' not in doc:
            doc = "stream(stream:torch.cuda.Stream) -> None\n" + doc
    if name == 'channel_shuffle':
        if '->' not in doc:
            doc = 'channel_shuffle(input:Tensor, groups:int) -> Tensor\n' + doc
    if name == 'pixel_shuffle':
        if '->' not in doc:
            doc = 'pixel_shuffle(input:Tensor, upscale_factor:int) -> Tensor\n' + doc
    if name in ('isinf', 'isfinite', 'isnan'):
        if '->' not in doc:
            doc = name + '(input:Tensor) -> Tensor\n' + doc
    if name == 'conv_tbc':
        if '->' not in doc:
            doc = name + '(input:Tensor, weight:Tensor, bias:Tensor, pad:int) -> Tensor\n' + doc
    if name in ('detach', 'detach_'):
        if '->' not in doc:
            doc = 'detach() -> Tensor'
    _sig_match = re.compile(r'(torch[.]|[.][.]\sfunction[:][:]\s*|)' + name.rstrip('_') + r'_?\s*[(](?P<args>[^)]*)[)]').match
    return_annotation = inspect.Signature.empty
    parameters = None
    parameters_without_annotations = {}
    for line in doc.splitlines():
        line = line.lstrip()
        if not line:
            continue
        if parameters is None:
            m = _sig_match(line)
            if m is not None:
                parameters = []
                kw_only = False
                # fix documentation typos, TODO: create pytorch issue
                args = m.group('args')
                if name == 'poisson':
                    args = args.replace('input *', 'input, *')
                if name in ['empty', 'ones', 'rand', 'randn', 'zeros']:
                    args = args.replace('*size, out=', '*size, *, out=')
                args_lst = args.split(',')
                if first_arg:
                    args_lst.insert(0, first_arg)
                for a in args_lst:
                    a = a.strip()
                    if not a:
                        continue
                    if a in ['*', r'\*']:
                        kw_only = True
                        continue
                    if a == '...':
                        a = '*tripledot'
                    if a.startswith('**'):
                        aname = a[2:]
                        kind = inspect.Parameter.VAR_KEYWORD
                        default = inspect.Parameter.empty
                    elif a.startswith('*'):
                        aname = a[1:]
                        kind = inspect.Parameter.VAR_POSITIONAL
                        default = inspect.Parameter.empty
                    elif '=' in a:
                        aname, default = a.split('=', 1)
                        default = eval(default, dict(torch=torch), {})
                        if kw_only:
                            kind = inspect.Parameter.KEYWORD_ONLY
                        else:
                            kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
                    else:
                        aname, default = a, inspect.Parameter.empty
                        if kw_only:
                            kind = inspect.Parameter.KEYWORD_ONLY
                        else:
                            kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
                    if ':' in aname:
                        aname, annotation = aname.split(':', 1)
                        aname = aname.strip()
                        annotation = resolve_annotation(annotation.strip(), line, name)
                    else:
                        annotation = inspect.Parameter.empty
                        if kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                            parameters_without_annotations[aname] = len(parameters)
                    if aname.endswith('*'):
                        print('todo:', line)
                        aname = aname[:-1].rstrip()

                    parameters.append(inspect.Parameter(aname, kind, default=default, annotation=annotation))
                rest = line[m.end():].lstrip()
                if rest.startswith('->'):
                    return_annotation = resolve_annotation(rest[2:].strip(), line, name)
                continue

        for aname, index in tuple(parameters_without_annotations.items()):
            if name == 'solve' and line.startswith('out ((Tensor, Tensor), optional):'):
                annot = typing.Tuple[torch.Tensor, torch.Tensor]
                parameters[index] = parameters[index].replace(annotation=annot)
                del parameters_without_annotations[aname]
                break
            _annot_match = re.compile(r'\A' + aname + r'\s*[(](?P<annot>[^)]*)[)]').match
            m = _annot_match(line)
            if m is not None and ':' in line:
                annot = m.group('annot')
                annot = resolve_annotation(annot, line, name)
                parameters[index] = parameters[index].replace(annotation=annot)
                del parameters_without_annotations[aname]
                break

    for aname, index in tuple(parameters_without_annotations.items()):
        default = parameters[index].default
        if default is not inspect.Parameter.empty:
            if isinstance(default, bool):
                parameters[index] = parameters[index].replace(annotation=bool)
                del parameters_without_annotations[aname]                
                continue

        if aname in ['input', 'input1', 'input2']:
            if name in ['trace', 'threshold_', 'selu_', 'rrelu_', 'relu_', 'pdist', 'lu_solve', 'frac',
                        'equal', 'dot', 'conv_transpose3d', 'conv_transpose2d', 'conv_transpose1d',
                        'conv3d', 'conv2d', 'conv1d', 'celu_', 'bitwise_xor', 'bitwise_or', 'bitwise_and',
                        'avg_pool1d', 'adaptive_avg_pool1d', 'absolute',
                        'tanhshrink', 'tanh', 'softsign', 'softshrink', 'softplus', 'soft_margin_loss',
                        'sigmoid', 'selu', 'rrelu', 'relu6', 'relu', 'prelu', 'mse_loss', 'logsigmoid',
                        'leaky_relu_', 'leaky_relu', 'l1_loss', 'hinge_embedding_loss', 'hardtanh_',
                        'hardtanh', 'hardsigmoid', 'hardshrink', 'gelu', 'elu_', 'celu', 'avg_pool3d',
                        'avg_pool2d', 'multilabel_soft_margin_loss', 'multilabel_margin_loss', 'leaky_relu_',
                        'margin_ranking_loss', 'cosine_embedding_loss'
            ]:
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
        elif aname == 'out':
            if name in ['full_like', 'frac', 'digamma', 'absolute']:
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
        elif aname == 'target':
            if name in ['soft_margin_loss', 'multilabel_soft_margin_loss', 'multilabel_margin_loss',
                        'mse_loss', 'margin_ranking_loss', 'l1_loss', 'hinge_embedding_loss', 'cosine_embedding_loss']:
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
        elif aname == 'other':
            if name in ['equal', 'bitwise_xor', 'bitwise_or', 'bitwise_and', 'sub']:
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
        elif aname in ['tensor', 'tensor1']:
            if name in ['dot', 'resize_as_', 'index_put']:
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
        elif name == 'threshold_':
            if 0 and aname == 'inplace':
                parameters[index] = parameters[index].replace(annotation=bool)
                del parameters_without_annotations[aname]
            if aname in ('value', 'threshold'):
                parameters[index] = parameters[index].replace(annotation=Value)
                del parameters_without_annotations[aname]
        elif name == 'take' and aname == 'index':  # pytorch doc bug
            parameters[index] = parameters[index].replace(annotation=torch.LongTensor)
            del parameters_without_annotations[aname]
        elif name in ('set_num_interop_threads', 'set_num_threads') and aname == 'int':
            parameters[index] = parameters[index].replace(annotation=int)
            del parameters_without_annotations[aname]
        elif name in ['rrelu_', 'rrelu']:
            if aname in ('lower', 'upper'):
                parameters[index] = parameters[index].replace(annotation=typing.Union[int, float])
                del parameters_without_annotations[aname]
        elif name == 'pdist' and aname == 'p':
            parameters[index] = parameters[index].replace(annotation=typing.Union[int, float])
            del parameters_without_annotations[aname]
        elif name in ('full_like', 'full') and aname=='fill_value':
            parameters[index] = parameters[index].replace(annotation=Value)
            del parameters_without_annotations[aname]
        elif aname == 'ndarray':
            import numpy
            parameters[index] = parameters[index].replace(annotation=numpy.ndarray)
            del parameters_without_annotations[aname]
        elif name in ('conv_transpose3d', 'conv_transpose2d', 'conv_transpose1d',
                      'conv3d', 'conv2d', 'conv1d'):
            if aname in ('weight', 'bias'):
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
            elif aname in ('stride', 'padding', 'output_padding', 'dilation'):
                parameters[index] = parameters[index].replace(annotation=typing.Union[int, typing.Tuple[int]])
                del parameters_without_annotations[aname]
            elif aname == 'groups':
                parameters[index] = parameters[index].replace(annotation=int)
                del parameters_without_annotations[aname]
        elif name in ('avg_pool1d', 'avg_pool2d', 'avg_pool3d'):
            if aname in ('stride', 'padding', 'kernel_size'):
                parameters[index] = parameters[index].replace(annotation=typing.Union[int, typing.Tuple[int]])
                del parameters_without_annotations[aname]
            if aname == 'divisor_override':
                parameters[index] = parameters[index].replace(annotation=int)
                del parameters_without_annotations[aname]
        elif name in ('celu_', 'elu_', 'celu') and aname == 'alpha':
            parameters[index] = parameters[index].replace(annotation=float)
            del parameters_without_annotations[aname]
        elif name == 'adaptive_avg_pool1d' and aname == 'output_size':
            parameters[index] = parameters[index].replace(annotation=int)
            del parameters_without_annotations[aname]
        elif name in ['softshrink', 'hardshrink'] and aname == 'lambd':
            parameters[index] = parameters[index].replace(annotation=float)
            del parameters_without_annotations[aname]
        elif name == 'softplus' and aname in ('beta', 'threshold'):
            parameters[index] = parameters[index].replace(annotation=float)
            del parameters_without_annotations[aname]
        elif name in ['soft_margin_loss', 'multilabel_soft_margin_loss', 'multilabel_margin_loss',
                      'mse_loss', 'margin_ranking_loss', 'l1_loss', 'hinge_embedding_loss', 'cosine_embedding_loss']:
            if aname in ['size_average', 'reduce']:
                parameters[index] = parameters[index].replace(annotation=bool)
                del parameters_without_annotations[aname]
            elif aname == 'reduction':
                parameters[index] = parameters[index].replace(annotation=str)
                del parameters_without_annotations[aname]
            elif aname == 'weight':
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
            elif aname == 'margin':
                parameters[index] = parameters[index].replace(annotation=float)
                del parameters_without_annotations[aname]
        elif name == 'prelu' and aname == 'weight':
            parameters[index] = parameters[index].replace(annotation=torch.Tensor)
            del parameters_without_annotations[aname]
        elif name in ['leaky_relu_', 'leaky_relu'] and aname == 'negative_slope':
            parameters[index] = parameters[index].replace(annotation=float)
            del parameters_without_annotations[aname]
        elif name in ['hardtanh_', 'hardtanh']:
            if aname in ('min_val', 'max_val'):
                parameters[index] = parameters[index].replace(annotation=float)
                del parameters_without_annotations[aname]
        elif name == 'sub' and aname == 'alpha':
            parameters[index] = parameters[index].replace(annotation=typing.Union[int, float])
            del parameters_without_annotations[aname]
        elif name == 'fill_' and aname == 'value':
            parameters[index] = parameters[index].replace(annotation=Value)
            del parameters_without_annotations[aname]
    if parameters_without_annotations:
        print(doc)
        print(name, parameters_without_annotations)

    if name in ['arange', 'randint', 'range']:
        print(f'{name}: remove default from {parameters[0]}')
        parameters[0] = parameters[0].replace(default=inspect.Parameter.empty)
    elif name in ['randint_like']:
        print(f'{name}: remove default from {parameters[1]}')
        parameters[1] = parameters[1].replace(default=inspect.Parameter.empty)

    if parameters is None:
        return

    return inspect.Signature(parameters=parameters, return_annotation=return_annotation)

def scan_module(module=torch):
    """
    yield (<torch function>, <torch function signature>)
    """
    for name in sorted(dir(module)):
        try:
            member = getattr(module, name)
        except Exception as msg:
            print(f'accessing {module.__name__}.{name} failed: {msg}')
            continue
        if isinstance(member, (bool, str, type, dict, list, tuple, int, float, complex)):
            continue
        if not callable(member):
            if name == 'classes':
                continue
            if isinstance(member, types.ModuleType):
                if not member.__name__.startswith('torch.'):
                    continue
                # TODO: scan torch modules recursively
            continue
        doc = getattr(member, '__doc__')
        first_arg = ''

        if not doc:
            if name in ['scatter_add', 'index_put']:
                doc = getattr(getattr(torch.Tensor, name + '_'), '__doc__')
                first_arg = 'input:Tensor'
            if name in ['clamp_', 'addmv_', 'atanh_']:
                doc = getattr(getattr(torch, name[:-1]), '__doc__')

        if name.endswith('_') and doc and 'In-place version of' in doc:
            mth = getattr(module, name[:-1])
            doc = getattr(mth, '__doc__')

        if not name.endswith('_') and doc and 'Out-of-place version of' in doc:
            mth = getattr(module, name + '_')
            doc = getattr(mth, '__doc__')

        if not doc:
            if isinstance(member, types.BuiltinMethodType):
                mth = getattr(torch.Tensor, name, None)
                if mth is not None:
                    doc = getattr(mth, '__doc__')
                    first_arg = 'input:Tensor'
                if doc and 'Out-of-place version of' in doc:
                    mth_ = getattr(torch.Tensor, name + '_', None)
                    if mth_ is not None:
                        doc = getattr(mth_, '__doc__')
                        first_arg = 'input:Tensor'
            if not doc:
                #print(f'{name}: empty or no __doc__')
                continue

        sig = get_signature_from_doc(name, member, doc, first_arg=first_arg)

        if sig is None:
            try:
                sig = inspect.signature(member)
            except Exception as msg:
                print(f'{name}: {msg}')
                print('-' * 40)                
                continue
        if sig is None:
            print(f'{name}: failed to construct signature')

        yield member, sig

if __name__ == '__main__':
    for func, sig in scan_module():
        print(func.__name__, sig)

