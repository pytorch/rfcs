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
    if annot in ['object', 'Object']:
        return object
    if annot in ['``int``', 'int', 'Optional[int]', 'integer']:
        return int
    if annot in ['boolean', 'bool', '(bool)']:
        return bool
    if annot == 'tuple':
        return tuple
    if annot == 'list':
        return list
    if annot == 'dict':
        return dict
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
    if annot in ['Tensor', 'Optional[torch.Tensor]', 'tensor', 'at::Tensor']:
        return torch.Tensor
    if annot == 'LongTensor':
        return torch.LongTensor
    if annot == 'IntTensor':
        return torch.IntTensor
    if annot == 'ByteTensor':
        return torch.ByteTensor
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
    if annot == 'callable':
        return typing.Callable
    if annot == 'type or string':
        return typing.Union[type, str]
    if annot == 'int or tuple of two lists of integers':
        return typing.Union[int, typing.Tuple[typing.List[int]]]
    if annot == '(Tensor, IntTensor, Optional[IntTensor])':
        return typing.Tuple[torch.Tensor, torch.IntTensor, typing.Optional[torch.IntTensor]]
    if annot == 'Sequence[Tensor]':
        return typing.Sequence[torch.Tensor]
    if annot == 'Tuple[Tensor]':
        return typing.Tuple[torch.Tensor]
    if annot == 'Module':
        return types.ModuleType
    if annot == 'function':
        return types.FunctionType
    if annot in ('IValue', 'handle', 'cpp_function'):
        print(f'{name}: returning annotation {annot!r} as it is')
        return annot
    if annot == 'torch::jit::Module':
        return torch.jit.Module
    if annot == 'c10::Type':
        return torch.Type
    if annot == 'torch::jit::Gradient':
        return torch.Gradient
    if annot == 'Tuple[Callable[[torch._C.ScriptModule], None], Callable[[torch._C.ScriptFunction], None]]':
        return typing.Tuple[typing.Callable[[torch.ScriptModule], None], typing.Callable[[torch.ScriptModule], None]]
    if annot == 'torch::jit::Block':
        return torch.Block
    if annot == 'int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]':
        return typing.Union[int, typing.Tuple[int]]
    if annot == 'float or Tuple[float]':
        return typing.Union[float, typing.Tuple[float]]
    if annot in ['int or Tuple[int, int]', 'int or Tuple[int, int] or Tuple[int, int, int]']:
        return typing.Union[int, typing.Tuple[int]]
    if annot == 'int or tuple':
        return typing.Union[int, typing.Tuple]
    if annot == 'SparseTensor':
        return torch.Tensor
    if '[' in annot and annot.endswith(']'):
        i = annot.index('[')
        t = annot[:i]
        a = annot[i+1:-1].strip()
        if hasattr(typing, t):
            args = []
            for a_ in a.split(','):
                a_ = a_.strip()
                if not a_:
                    continue
                args.append(resolve_annotation(a_, orig_line, name))
            return getattr(typing, t)[tuple(args)]
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
    if name == 'elu_':
        if '->' not in doc:
            doc = 'elu_(input:Tensor, alpha:float=1.) -> Tensor'
    if name == 'threshold_':
        if '->' not in doc:
            doc = 'threshold_(input:Tensor, threshold:Value, value:Value, inplace:bool=False) -> Tensor'
    if name == 'multi_margin_loss':
        doc = doc.replace('\n', '', 1)
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
                if args == 'arg0: Dict[str, IValue]':
                    args_lst = [args]
                elif args == 'input:torch.Tensor, dim:int, dtype:Union[int, NoneType]=None':
                    args_lst = ['input:torch.Tensor', 'dim:int', 'dtype:Union[int, None]=None']
                else:
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
                        if name in ['load', 'save'] and aname == 'pickle_module' and default.startswith("<module 'pickle'"):
                            import pickle
                            default = pickle
                            if ':' not in aname:
                                aname += ':Module'
                        else:
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
            if name == 'norm' and line.startswith('p ('):
                annot = typing.Union[int, float, str]
                parameters[index] = parameters[index].replace(annotation=annot)
                del parameters_without_annotations[aname]
                break
            if name in ('nonzero', 'norm') and line.startswith('dim ('):
                annot = typing.Union[int, typing.Tuple[int]]
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
                        'margin_ranking_loss', 'cosine_embedding_loss', 'unfold', 'threshold', 'smooth_l1_loss',
                        'poisson_nll_loss', 'normalize', 'nll_loss',
                        'max_unpool1d', 'max_unpool2d', 'max_unpool3d',
                        '_adaptive_max_pool1d', '_adaptive_max_pool2d', '_adaptive_max_pool3d',
                        'adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_max_pool3d',
                        '_adaptive_avg_pool1d', '_adaptive_avg_pool2d', '_adaptive_avg_pool3d',
                        'adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d',
                        '_fractional_max_pool1d', '_fractional_max_pool2d', '_fractional_max_pool3d',
                        'fractional_max_pool1d', 'fractional_max_pool2d', 'fractional_max_pool3d',
                        'fractional_max_pool1d_with_indices', 'fractional_max_pool2d_with_indices', 'fractional_max_pool3d_with_indices',
                        '_max_pool1d', '_max_pool2d', '_max_pool3d',
                        'max_pool1d', 'max_pool2d', 'max_pool3d',
                        'max_pool1d_with_indices', 'max_pool2d_with_indices', 'max_pool3d_with_indices',
                        '_pad', '_pad_circular', 'alpha_dropout',
                        'binary_cross_entropy', 'binary_cross_entropy_with_logits',
                        'conv1d', 'conv2d', 'conv3d',
                        'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d',
                        'dropout', 'dropout2d', 'dropout3d', 'local_response_norm', 'linear',
                        'layer_norm', 'kl_div', 'hardswish', 'group_norm',
                        'fold', 'feature_alpha_dropout', 'elu',
                        'adaptive_max_pool1d_with_indices', 'adaptive_max_pool2d_with_indices', 'adaptive_max_pool3d_with_indices',
                        'bilinear', 'batch_norm', 'instance_norm',
                        'lp_pool1d', 'lp_pool2d', 'lp_pool3d',
            ]:
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
        elif aname == 'out':
            if name in ['full_like', 'frac', 'digamma', 'absolute']:
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
        elif aname == 'target':
            if name in ['soft_margin_loss', 'multilabel_soft_margin_loss', 'multilabel_margin_loss',
                        'mse_loss', 'margin_ranking_loss', 'l1_loss', 'hinge_embedding_loss', 'cosine_embedding_loss',
                        'smooth_l1_loss', 'poisson_nll_loss', 'nll_loss', 'kl_div',
                        'binary_cross_entropy', 'binary_cross_entropy_with_logits']:
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
                      'conv3d', 'conv2d', 'conv1d',
                      'avg_pool1d', 'avg_pool2d', 'avg_pool3d',
                      'unfold', 'fold',
                      'max_pool1d_with_indices', 'max_pool2d_with_indices', 'max_pool3d_with_indices',
                      'max_unpool1d', 'max_unpool2d', 'max_unpool3d',
                      '_max_pool1d', '_max_pool2d', '_max_pool3d',
                      'max_pool1d', 'max_pool2d', 'max_pool3d',
                      'lp_pool1d', 'lp_pool2d', 'lp_pool3d',
                      'fractional_max_pool1d_with_indices', 'fractional_max_pool2d_with_indices', 'fractional_max_pool3d_with_indices',
                      '_fractional_max_pool1d', '_fractional_max_pool2d', '_fractional_max_pool3d',
                      'fractional_max_pool1d', 'fractional_max_pool2d', 'fractional_max_pool3d',
                      '_adaptive_max_pool1d', '_adaptive_max_pool2d', '_adaptive_max_pool3d',
                      'adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_max_pool3d',
                      'adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d',
                      'adaptive_max_pool1d_with_indices', 'adaptive_max_pool2d_with_indices', 'adaptive_max_pool3d_with_indices',
        ):
            if aname in ('weight', 'bias', 'indices'):
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
            elif aname in ('stride', 'padding', 'output_padding', 'dilation', 'kernel_size', 'output_size'):
                parameters[index] = parameters[index].replace(annotation=typing.Union[int, typing.Tuple[int]])
                del parameters_without_annotations[aname]
            elif aname in ('groups', 'divisor_override'):
                parameters[index] = parameters[index].replace(annotation=int)
                del parameters_without_annotations[aname]
            elif aname in ('output_ratio', 'norm_type'):
                parameters[index] = parameters[index].replace(annotation=float)
                del parameters_without_annotations[aname]
            elif aname in ('_random_samples',):
                parameters[index] = parameters[index].replace(annotation=None)
                del parameters_without_annotations[aname]
        elif name in ('_pad_circular',):
            if aname in ('padding',):
                parameters[index] = parameters[index].replace(annotation=typing.Tuple[int])
                del parameters_without_annotations[aname]
        elif name in ('celu_', 'elu_', 'celu', 'elu') and aname == 'alpha':
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
        elif name == 'set_printoptions':
            if aname in ['precision', 'threshold', 'edgeitems', 'linewidth']:
                parameters[index] = parameters[index].replace(annotation=int)
                del parameters_without_annotations[aname]
            elif aname == 'profile':
                parameters[index] = parameters[index].replace(annotation=str)
                del parameters_without_annotations[aname]
            elif aname == 'sci_mode':
                parameters[index] = parameters[index].replace(annotation=bool)
                del parameters_without_annotations[aname]
        elif name in ('save', 'load'):
            if aname == 'obj':
                parameters[index] = parameters[index].replace(annotation=object)
                del parameters_without_annotations[aname]
            elif aname == 'f':
                parameters[index] = parameters[index].replace(annotation=typing.Union[str, object])
                del parameters_without_annotations[aname]
            elif aname == 'pickle_protocol':
                parameters[index] = parameters[index].replace(annotation=int)
                del parameters_without_annotations[aname]
            elif aname == 'map_location':
                parameters[index] = parameters[index].replace(annotation=typing.Union[types.FunctionType, str, dict, torch.device])
                del parameters_without_annotations[aname]
        elif name == 'lobpcg':
            if aname in ('ortho_iparams', 'ortho_fparams', 'ortho_bparams'):
                parameters[index] = parameters[index].replace(annotation=dict)
                del parameters_without_annotations[aname]
        elif name == 'cdist':
            if aname == 'p':
                parameters[index] = parameters[index].replace(annotation=float)
                del parameters_without_annotations[aname]
            if aname == 'compute_mode':
                parameters[index] = parameters[index].replace(annotation=str)
                del parameters_without_annotations[aname]
        elif name in ('triplet_margin_loss', 'smooth_l1_loss'):
            if aname in ('margin', 'eps'):
                parameters[index] = parameters[index].replace(annotation=float)
                del parameters_without_annotations[aname]
            elif aname in ('size_average', 'reduce', 'swap'):
                parameters[index] = parameters[index].replace(annotation=bool)
                del parameters_without_annotations[aname]
            elif aname in ['anchor', 'positive', 'negative']:
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
            elif aname == 'reduction':
                parameters[index] = parameters[index].replace(annotation=str)
                del parameters_without_annotations[aname]
            elif aname == 'p':
                parameters[index] = parameters[index].replace(annotation=int)
                del parameters_without_annotations[aname]
        elif name == 'threshold':
            if aname in ['threshold', 'value']:
                parameters[index] = parameters[index].replace(annotation=float)
                del parameters_without_annotations[aname]
        elif name in ('softmin', 'softmax', 'log_softmax', 'log_softmin'):
            if aname == '_stacklevel':
                parameters[index] = parameters[index].replace(annotation=int)
                del parameters_without_annotations[aname]
        elif name == 'pairwise_distance':
            if aname in ('x1', 'x2'):
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
            elif aname == 'p':
                parameters[index] = parameters[index].replace(annotation=int)
                del parameters_without_annotations[aname]
            elif aname == 'eps':
                parameters[index] = parameters[index].replace(annotation=float)
                del parameters_without_annotations[aname]
        elif name in ('pad', '_pad'):
            if aname == 'mode':
                parameters[index] = parameters[index].replace(annotation=str)
                del parameters_without_annotations[aname]
            elif aname == 'value':
                parameters[index] = parameters[index].replace(annotation=float)
                del parameters_without_annotations[aname]
        elif name in ('dropout', 'dropout2d', 'dropout3d', 'alpha_dropout', 'feature_alpha_dropout'):
            if aname == 'p':
                parameters[index] = parameters[index].replace(annotation=float)
                del parameters_without_annotations[aname]
        elif name in ('linear', 'bilinear', 'batch_norm', 'group_norm', 'local_response_norm', 'instance_norm', 'layer_norm'):
            if aname in ('weight', 'bias', 'running_mean', 'running_var'):
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
            elif aname in ['eps', 'momentum', 'alpha', 'beta', 'k']:
                parameters[index] = parameters[index].replace(annotation=float)
                del parameters_without_annotations[aname]
            elif aname in ['num_groups', 'size']:
                parameters[index] = parameters[index].replace(annotation=int) # guess
                del parameters_without_annotations[aname]
            elif aname in ['normalized_shape']:
                parameters[index] = parameters[index].replace(annotation=typing.Union[int, typing.List[int], torch.Size])
                del parameters_without_annotations[aname]
        elif name in ['has_torch_function', 'handle_torch_function']:
            if aname == 'relevant_args':
                parameters[index] = parameters[index].replace(annotation=typing.Sequence)
                del parameters_without_annotations[aname]
            elif aname == 'public_api':
                parameters[index] = parameters[index].replace(annotation=typing.Callable)
                del parameters_without_annotations[aname]
        elif name == 'gumbel_softmax':
            if aname in ('tau', 'eps'):
                parameters[index] = parameters[index].replace(annotation=float)
                del parameters_without_annotations[aname]
            elif aname == 'logits':
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
        elif name == 'ctc_loss':
            if aname in ('tau', 'eps'):
                parameters[index] = parameters[index].replace(annotation=float)
                del parameters_without_annotations[aname]
            elif aname in ('log_probs', 'targets', 'input_lengths', 'target_lengths'):
                parameters[index] = parameters[index].replace(annotation=torch.Tensor)
                del parameters_without_annotations[aname]
    if parameters_without_annotations and 0:
        print(doc)
        print(name, parameters_without_annotations)

    if name in ['arange', 'randint', 'range']:
        print(f'{name}: remove default from {parameters[0]}')
        parameters[0] = parameters[0].replace(default=inspect.Parameter.empty)
    elif name in ['randint_like']:
        print(f'{name}: remove default from {parameters[1]}')
        parameters[1] = parameters[1].replace(default=inspect.Parameter.empty)

    if parameters is None:
        print(name, doc)
        return

    return inspect.Signature(parameters=parameters, return_annotation=return_annotation)

def scan_module(module=torch, recursive=False, _cache=set()):
    """
    yield (<torch function>, <torch function signature>)
    """
    if isinstance(module, (list, tuple)):
        for m in module:
            for r in scan_module(m, recursive=recursive):
                yield r
        return

    if module.__name__ in _cache:
        return
    _cache.add(module.__name__)
    for name, member in sorted(module.__dict__.items()):
        if isinstance(member, (bool, str, type, dict, list, tuple, int, float, complex, typing._FinalTypingBase)):
            continue
        if not callable(member):
            if name == 'classes':
                continue
            if isinstance(member, types.ModuleType) and recursive:
                if not member.__name__.startswith('torch.'):
                    continue
                for r in scan_module(module=member, recursive=recursive):
                    yield r
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

        if isinstance(member, types.FunctionType) and '->' not in doc:
            try:
                s = name + str(inspect.signature(member))
            except Exception as msg:
                print(f'{name}: failed to get signature: {msg}')
                continue
            if '->' not in s:
                if name in ['block_diag', 'cartesian_prod', 'cdist', 'chain_matmul', 'istft', 'load', 'norm',
                            'stft', 'tensordot', 'upsample_nearest', 'upsample_bilinear', 'upsample',
                            'unfold', '_pad', '_pad_circular', 'affine_grid', 'alpha_dropout',
                            'batch_norm', 'bilinear', 'binary_cross_entropy', 'binary_cross_entropy_with_logits',
                            'cross_entropy', 'ctc_loss', 'dropout', 'dropout2d', 'dropout3d',
                            'elu', 'embedding', 'embedding_bag', 'feature_alpha_dropout', 'fold',
                            'grid_sample', 'group_norm', 'gumbel_softmax', 'hardswish', 'interpolate', 'kl_div',
                            'layer_norm', 'linear', 'local_response_norm', 'log_softmax', 'lp_pool1d', 'lp_pool2d',
                            'max_pool1d_with_indices', 'max_pool2d', 'max_pool2d_with_indices', 'max_pool3d',
                            'max_pool3d_with_indices', 'max_unpool1d', 'max_unpool2d', 'max_unpool3d', 'nll_loss',
                            'normalize', 'pad', 'pairwise_distance', 'poisson_nll_loss', 'smooth_l1_loss', 'softmax',
                            'softmin', 'threshold', 'triplet_margin_loss']:
                    s += '-> Tensor'
                elif name in ['compiled_with_cxx11_abi', 'is_deterministic', 'is_storage', 'is_tensor', 'has_torch_function']:
                    s += '-> bool'
                elif name == 'get_rng_state':
                    s += '-> ByteTensor'
                elif name in ('initial_seed', 'save', 'set_default_dtype', 'set_default_tensor_type',
                              'set_deterministic', 'set_printoptions', 'set_rng_state'):
                    s += '-> None'
                elif name == 'lobpcg':
                    s += '-> (Tensor, Tensor)'
                elif name == 'lu':
                    s += '-> (Tensor, IntTensor, Optional[IntTensor])'
                elif name in ['lu_unpack', 'pca_lowrank', 'svd_lowrank']:
                    s += '-> (Tensor, Tensor, Tensor)'
                elif name == 'manual_seed':
                    s += '-> torch.Generator'
                elif name == 'meshgrid':
                    s += '-> Sequence[Tensor]'
                elif name == 'split':
                    s += '-> Tuple[Tensor]'
                elif name == 'seed':
                    s += '-> int'
                elif name in ['unique', 'unique_consecutive']:
                    s += '-> Tuple[Tensor]'
                elif name in ['_adaptive_max_pool1d', '_adaptive_max_pool2d', '_adaptive_max_pool3d',
                              'adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_max_pool3d',
                              'adaptive_avg_pool1d', 'adaptive_avg_pool2d', 'adaptive_avg_pool3d',
                              'adaptive_max_pool1d_with_indices', 'adaptive_max_pool2d_with_indices', 'adaptive_max_pool3d_with_indices',
                              '_fractional_max_pool2d', '_fractional_max_pool3d', '_max_pool1d',
                              'fractional_max_pool2d', 'fractional_max_pool3d', 'max_pool1d',
                              '_max_pool2d', '_max_pool3d',
                              'fractional_max_pool1d_with_indices', 'fractional_max_pool2d_with_indices', 'fractional_max_pool3d_with_indices',
                              'boolean_dispatch', 'instance_norm'
                ]:
                    s += '-> None'
                elif name == 'handle_torch_function':
                    s += '-> object'
                elif name == 'multi_head_attention_forward':
                    s += '-> (Tensor, Tensor)'
                elif name in ['sum', 'addmm', 'mm']:
                    s += '-> Tensor'
                else:
                    #print(doc)
                    print(f'{name}: no return value')
                    #raise
            doc = s + '\n' + doc

        try:
            sig = get_signature_from_doc(name, member, doc, first_arg=first_arg)
        except Exception as msg:
            print(name)
            print(doc)
            print(f'{name}: failed to construct signature: {msg}')
            raise
            continue

        if sig is None:
            print(f'{name}: failed to construct signature')

        if member.__module__ is None:
            member.__module__ = module.__name__
        yield member, sig

if __name__ == '__main__':
    for func, sig in scan_module(module=[torch, torch.nn.functional]):
        print(func.__name__, sig)
        pass
