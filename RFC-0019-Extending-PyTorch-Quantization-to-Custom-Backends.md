# Summary
Today PyTorch Quantization has well documented support for two backends: fbgemm (x86) or qnnpack (arm or x86), with some customizations exposed but no easy way for customers to quantize models for other backends.  This document proposes how to extend PyTorch Quantization to properly support custom backends, such as Intel NNPI (A*), NVIDIA V-100/A-100 and others. We hope that this design will:
* Allow for pytorch users to perform Quantization Aware Training or Post Training quantization for backends beyond server and mobile CPUs
* Provide a simple and clear API for custom backend developers to integrate a custom backend with PyTorch Quantization
* Provide a simple and clear API for model developers to quantize models targeting custom backends

The workflow for custom backend developers who want to extend PyTorch Quantization to work on their backend looks like following:
* Define the configurations for quantized operators (including fused and quantized operators) with backend_config_dict api
* Define a lowering pass that transforms a model with reference quantized functions to a model that can be understood by the custom backend

The workflow for model developers who want to quantize for a particular backend need to do the following:
* Get the backend configuration for the custom backend
* Specifying how the model should be quantized by defining the qconfig_dict, the qconfig_dict should be valid given the backend configuration
Quantize the model
* Lower the model to custom backend by calling the lowering function for the backend

Note: This design is based on [FX Graph Mode Quantization](https://pytorch.org/docs/stable/quantization.html#quantization-api-summary).

# Reference Quantized Model
We introduce the concept of a reference pattern which serves as a standard format for quantized operators in all backends, a reference quantized model is a quantized model with these reference pattern. Reference patterns provide a close approximation to backends using fp32 ops and type conversion ops. If a more accurate match is desired, we need emulation operators that accurately model numerics for a backend. A reference quantized model serves two purposes:
1. Standard format for lowering quantized models
2. Emulate model numerics with approximate reference operators on a devserver for debugging.

The property of a quantized operator can be decomposed into two dimensions:
  1. Signature
  2. Numerics

Currently PyTorch quantization supports two backends:  fbgemm (server x86 CPU)  and qnnpack (ARM CPU and x86 on mobile) and they (almost) match in both dimensions, however, there might be other backends that differ in signature or numerics than what is provided by PyTorch Quantization right now.

In general, when we have a new backend, there is no guarantee that the quantized operators supported by the new backend would match in either of the dimensions, so we propose to add an extra layer of indirection between the model produced by the quantization flow and the model that’s actually used for execution in the backends.

Therefore, quantization flow can produce a model with reference patterns (dequant - float_op - quant). And there will be an extra step to lower this to a model that is executable by various backends, including the current native backends in PyTorch (fbgemm/qnnpack). One thing to call out here is that we did not address some behavior of the custom backend, for example fp16/int16 accumulations since that would require changes to the implementation of reference pattern.

Here are examples of reference pattern for single operator and fused operators:
```python
# single operator with torch op/functional
def forward(x):
    ...
    x = x.dequantize()
    x = torch.sigmoid(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, dtype)
    ...

# single operator with module
def forward(x):
    ...
    x = x.dequantize()
    x = self.sigmoid(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, dtype)
    ...

# fused operators
def forward(x):
    ...
    x = x.dequantize()
    x = self.conv2d(x)
    x = torch.nn.functional.relu(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, dtype)
    ...
```



# Quantization Workflow
![Quantization Workflow](https://docs.google.com/drawings/d/e/2PACX-1vQ6tgl6MOkSPLcVV4ZRWwCBLebj-ugMIpBJgw8OL2FxqYg2u5rpp8UKSVQUg_Ie1HsHyVJf3A5dPIb_/pub?w=950&h=700)

As we can see from the above diagram, we’ll separate the generation of a quantized model in PyTorch and the actual runnable quantized model on a specific backend. PyTorch will produce a reference quantized model which contains reference patterns that can be fused into quantized functions, this model will act as a unified representation of a quantized model and we do not give guarantees on either numerics or performance.

Accuracy of a model on a specific backend can be emulated using reference ops as long as the numerics of the backend are well approximated by reference patterns (i.e by a sequence of dequant-fp32-quant ops). If this is not the case, then reference patterns need to be lowered to numerically accurate emulation functions for purposes of emulating accuracy.

To get a model runnable on a specific backends, we will need to have an extra lowering step that transforms this reference model to a backend specific model (a model that only runs on that backend, for example: fbgemm/qnnpack). We may also transform the reference model to a backend with fake ops that simulates the numerics of the ops that run on backends.
Backend Configurations
A backend is a hardware or kernel library (NNPI, FBGEMM, QNNPACK etc.), each hardware/kernel library has a set of settings that can differ from the default setting, We can define them by following:
1. **Quantization Scheme** (symmetric vs asymmetric, per-channel vs per-tensor)
2. **Data Type** (float32, float16, int8, int8, bfloat16, etc)
3. **Quantized (and Fused) Operators and Mapping** The quantized operators supported by the backend. For example: quantized conv2d, quantized linear etc.
Some quantized operators may have different numerics compared to a naive (dequant - float_op - quant) implementation
For weighted operators (conv and linear) we need to define a reference module and a mapping
4. **QAT Module Mapping** For modules with weights, e.g. Conv2d and Linear, we need to swap them with qat (quantization aware training) module that adds fake quantization to the weights

Note that this is general to all backends, not just custom backends. Current default backends in PyTorch (fbgemm/qnnpack) can be defined in the above terms as well.


| | Fbgemm/qnnpack (supported by default)|
| ----| ----|
|Quantization Scheme | activation: per tensor, weight: per tensor or per channel |
| Data Type | activation: quint8 (reduce_range for fbgemm), weight: qint8 |
| Quantized (and Fused) Operators and Mapping | For example: nn.Conv2d → torch.ao.nn.quantized.reference.Conv2d |
| QAT Module Mapping | Conv, Conv - Bn, Linear etc.|

# Proposed APIs for Custom Backend Developers (Demo Purpose Only)
```python
from enum import Enum
# Demonstration purpose only, the api is subject to change
class QuantizedOperatorType(Enum):
    NEED_OBSERVER_FOR_BOTH_INPUTS_AND_OUTPUTS = 0
    OUTPUT_IS_SHARING_OBSERVER_WITH_INPUT = 1

conv_module_config = {
    “type”: QuantizedOperatorType.NEED_OBSERVER_FOR_BOTH_INPUTS_AND_OUTPUTS,
    “float_to_quantized_operator_mapping”: {
        # contains mapping for float op and pattern to reference function or reference module
        “static”: {
            torch.nn.Conv2d: torch.ao.nn.quantized.reference.Conv2d,
            (torch.nn.ReLU, torch.nn.Conv2d): torch.ao.nn.quantized.reference.ConvReLU2d,
            (torch.nn.ReLU, torch.nn.qat.Conv2d): torch.ao.nn.quantized.reference.ConvReLU2d,
            torch.nn.intrinsic.qat.ConvBn2d: torch.ao.nn.quantized.reference.Conv2d
        },
    “qat_mapping”: {
        “static”: {
            torch.nn.Conv2d: torch.nn.qat.Conv2d
            (torch.nn.BatchNorm2d, torch.nn.Conv2d): torch.nn.intrinsic.qat.ConvBn2d
        }
    }
}

conv_functional_config = {
“type”: QuantizedOperatorType.NEED_OBSERVER_FOR_BOTH_INPUTS_AND_OUTPUTS,
}

bmm_config = {
    “type”: QuantizedOperatorType.NEED_OBSERVER_FOR_BOTH_INPUTS_AND_OUTPUTS,
“fusions”: [(torch.nn.Softmax, torch.bmm)] # quantized_bmm_softmax
}

custom_backend_config_dict = {
    # optional
    "name": "custom_backend",
    # quantized operator config is a map from
    # module/functional/torch ops to their configurations
    “operator”: {
        torch.nn.Conv2d: conv_module_config,
        torch.nn.functional.conv2d: conv_functional_config,
        torch.bmm: bmm_config
    }
}

# define a function to return the backend config dict
def get_custom_backend_config_dict():
    return custom_backend_config_dict

# We'll also provide utility functions to get the backend configurations for a given backend:
my_backend_config_dict = get_my_backend_config_dict()
```

Note the apis here are demo purpose only, for the most up to date apis, please refer to the code: https://github.com/pytorch/pytorch/tree/master/torch/ao/quantization/fx/backend_config_dict, we'll also have tutorials later when this is fully implemented.

# Proposed APIs for Model Developers
```python
from torch.quantization.quantize_fx import prepare_fx, convert_to_reference_fx
from custom_backend_library import get_my_backend_config_dict, lower_to_custom_backend

backend_config_dict = get_my_backend_config_dict()
model = prepare_fx(model, qconfig_dict, prepare_custom_config_dict=..., backend_config_dict = backend_config_dict)
# calibration
...
model = convert_to_reference_fx(model, convert_custom_config_dict=..., backend_config_dict=backend_config_dict)

# get the lower_to_custom_backend function defined by custom backend developers and call the function to lower a Reference Quantized Model to a model that runs on a custom backend
model = lower_to_custom_backend(model)
```

# Use Cases
## Use Case 1: Quantizing a Model for Inference on Server/Mobile
```python
from torch.quantization.quantize_fx import prepare_fx
model = model.eval()
qconfig_mapping = QConfigMapping().set_global(torch.quantization.default_qconfig)
model = prepare_fx(model, qconfig_mapping)
calibration(model, ...)
model = convert_to_reference_fx(model)
```

The model produced here is a reference model that contains reference patterns, it is runnable since it is using quantize_per_tensor/dequantize/floating point operators to simulate quantized operators. For numerics we’ll provide an approximation to backend numerics even though it may not have the exact same numerics as any backends, or same speed up as the backends.

### Backend Lowering (fbgemm/qnnpack)
```python
from torch.quantization.quantize_fx import prepare_fx
model = model.eval()
qconfig_mapping = QConfigMapping().set_global(torch.quantization.default_qconfig)
model = prepare_fx(model, qconfig_mapping)
calibration(model, ...)
model = convert_to_reference_fx(model)

# This step will transform a model with reference patterns to a model with
# fbgemm/qnnpack ops, e.g. torch.ops.quantized.conv2d
model = lower_to_fbgemm_fx(model)  # lower_to_qnnpack_fx(model)
```


### Example Implementation lower_to_fbgemm_fx

```python
from torch.ao.nn.quantized.functional.reference import quantized_sigmoid
import torch.fx
from torch.fx import subgraph_rewriter
from torch.quantization.fx.graph_module import QuantizedGraphModule
def relu_pattern(x, scale, zero_point):
    x = x.dequantize()
    x = torch.nn.functional.relu(x)
    x = torch.quantize_per_tensor(x, scale, zero_point, torch.quint8)
    return x

def relu_replacement(x, scale, zero_point):
    x = torch.nn.functional.relu(x)
    return x


def _get_all_patterns_and_replacements():
    return [
        (relu_pattern, relu_replacement)
    ]


def get_fbgemm_patterns_and_replacements():
    return _get_all_patterns_and_replacements()


def _lower_to_native_backend(model: QuantizedGraphModule) -> torch.nn.Module:
    """ Lower a quantized reference model (with reference quantized operator patterns)
    to the native backend in PyTorch (fbgemm/qnnpack), both backends shares the same
    operator signature so they can be lowered with the same function
    """
    module_dict = dict(model.named_modules())
    for pattern, replacement in get_fbgemm_patterns_and_replacements():
        subgraph_rewriter.replace_pattern(model, pattern, replacement)
    model.graph.lint()
    return model

def lower_to_fbgemm_fx(model: QuantizedGraphModule) -> torch.nn.Module:
    return _lower_to_native_backend(model)

def lower_to_qnnpack_fx(model: QuantizedGraphModule) -> torch.nn.Module:
    return _lower_to_native_backend(model)
```

As is shown from the code, there are two requirements that must be met when we add a new lowering pass for a specific backed:
We need to register the backend quantized operator in `torch.ops` namespace, this can be achieved with PyTorch custom operator registration: https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/quantized/cpu/qconv.cpp#L881.

## Use Case 2: Quantizing the Same Model for Inference on Custom Backend
If a backend does not need to modify the quantization flow, that is, it does not have extra quantized operators, fused quantized operators or customizations of quantization flow, we only need to write a lowering pass to transform a reference quantized model to a model runnable on a custom backend.

```python
from torch.quantization.quantize_fx import prepare_fx
model = model.eval()
qconfig_mapping = QConfigMapping().set_global(torch.quantization.default_qconfig)
model = prepare_fx(model, qconfig_mapping)
calibration(model, ...)
model = convert_to_reference_fx(model)

# This optional step will transform a model with reference
# functions to a model with fakeNNPI ops, e.g. torch.ops.fakeNNPI.sigmoidFP16
# This is useful for bit exact model emulation on a server.

fake_nnpi_model = lower_to_fakennpi_fx(model)

# This function will transform the model with reference patterns to a model with nnpi ops (need to register NNPI ops in torch.ops namespace)
nnpi_model = lower_to_nnpi_fx(model)
```


There are multiple lowering options:
* **Lowering with `torch.fx`** We can lower directly with `torch.fx` transformations. Please take a look at Extending PyTorch Quantization to Custom Backends for an example implementation of lowering in fx, for this we need to make sure all backend operators are exposed in `torch` namespace, for example: `torch.ops.custom_backend.quantized_conv2d`
* **Lowering with TorchScript** We can also first script/trace the model and then lower with TorchScript, we need to make sure TorchScript pass can recognize the reference quantized functions, for example, this can be done by looking at the name of a CallFunction Node
* **Lowering with a combination of 1 and 2** People can also do some transformations in `torch.fx` and then script/trace the model and do some other transformations in TorchScript, for example, in `torch.fx` we can transform reference quantized functions to actual (dequant - float_op - quant) pattern and in TorchScript we can fuse the patterns
* **Custom Lowering** Users can also provide their own lowering functions that does not use `torch.fx` or TorchScript that can transform a Reference Quantized Model to a model that’s runnable on the target backend. Basically Reference Quantized Model is the standard format that is expected by backend developers.

## Use Case 3: Extending Quantization to Support Custom Quantized Operators and Fusions for Inference in Custom Backend

If a backend supports more quantized operators, more fusions, different ways of quantization, we will need to customize the flow. We only provide limited support for quantized (and fused) operator configurations at this point and we’ll work on improving the support in the future to allow arbitrary custom behaviors.

Here we’ll give an example of supporting custom quantized operators and fusions.

### Custom Operator support
If you need to support an operator that does not have a reference operator implementation, you will need to implement a custom function/module to specify the quantized operator implementation. In addition, you will need to write a handler class that specifies how to convert a observed node to a quantized node.

#### Example: Quantized BMM
To add support for new quantized ops, we need to do the following:
* Get fbgemm config dictionary and add the new entry for the operator to it, and expose the backend config with a function
* Define/modify the lowering function of the custom backend that can fuse the pattern

```python
from torch.quantization.quantize_fx import prepare_fx
from torch.quantization.quantize_fx import get_backend_config_dict, QuantizedOperatorType, update_operator_config
import torch.fx


# Let's say we want to modify the backend_config of fbgemm and add new quantized operator
custom_backend_config_dict = get_backend_config_dict("fbgemm")
bmm_config = {
    "type": QuantizedOperatorType.NEED_OBSERVER_FOR_BOTH_INPUTS_AND_OUTPUTS,
}
update_operator_config(custom_backend_config_dict, torch.bmm, bmm_config)
custom_backend_config_dict["name"] = "custom_backend"

# define/modify lowering function to support lowering quantized_bmm to the custom backend, we now expect quantized models to have `dequant - torch.bmm - quant` patterns in the model if the input model contains a `torch.bmm` operator
```



### Custom Fusion Support
For custom fusions, we need to do the following:
* Get fbgemm config dictionary and add the new entry for the fused op to it, and expose the backend config with a function
* Define/modify the lowering function of the custom backend that can fuse the pattern

#### Example: Quantized Fused BMM and Sigmoid

```python
from torch.quantization.quantize_fx import prepare_fx
from torch.quantization.quantize_fx import get_backend_config_dict, QuantizedOperatorType, update_operator_config
import torch.fx

# Let's say we want to modify the backend_config of fbgemm and add new quantized operators
custom_backend_config_dict = get_backend_config("fbgemm").copy()
# Notice that the order of the pattern is reversed, this is
# because we want to support a graph
bmm_config = {
    "type": QuantizedOperatorType.NEED_OBSERVER_FOR_BOTH_INPUTS_AND_OUTPUTS,
}
update_operator_config(custom_backend_config_dict, torch.bmm, bmm_config)
update_operator_config(custom_backend_config_dict, (torch.softmax, torch.bmm), bmm_config)
custom_backend_config_dict["name"] = "custom_backend"

# define/modify lowering function to support lowering pattern for quantized_bmm and quantized_bmm_softmax to the custom backend
# we now expect quantized models to have `dequant - torch.bmm - quant`
# and `dequant - torch.bmm - torch.softmax - quant` patterns
```

# QConfig, QConfigMapping and BackendConfig
## [QConfig](https://pytorch.org/docs/master/generated/torch.quantization.qconfig.QConfig.html#torch.quantization.qconfig.QConfig)
QConfig is what we use to specify how to observe an operator (e.g. conv) or operator pattern (e.g. conv - relu) in the model, for example:
```
qconfig = QConfig(activation=HistogramObserver(dtype=torch.quint8, quant_min=0, quant_max=255), weight=PerChannelMinMaxObserver(dtype=torch.qint8, quant_min=-128, quant_max=127)
```
means we want to insert `HistogramObserver` for the input and output of the operator/operator pattern (in FX Graph Mode Quantization), and insert `PerChannelMinMaxObserver` to the weight of the operator/operator pattern. Note that this is relatively restrictive today since we can only specify the same observer for both input and output and only for weight, we plan to make this more general in the future to support specifying different observers for different inputs and outputs. Example:
```
QConfig(input_args=(input_act_obs,), input_kwargs={}, attribute = {“weight”: weight_obs, “bias”: bias_obs}, output=output_act_obs)
```

## [QConfigMapping](https://pytorch.org/docs/master/generated/torch.ao.quantization.qconfig_mapping.QConfigMapping.html#torch.ao.quantization.qconfig_mapping.QConfigMapping)
`QConfigMapping` is used to configure how to quantize a model, it is a set of rules that applies to the model graph to to decide what should be the `QConfig` for each operator or operator pattern, right now we are mostly supporting operators, but in the future we'll support operator pattern as well.
Example:
```
qconfig_mapping = QConfigMapping()
    .set_global(global_qconfig)
    .set_object_type(torch.nn.Linear, qconfig1)
    .set_object_type(torch.nn.ReLU, qconfig1)
    .set_module_name_regex("foo.*bar.*conv[0-9]+", qconfig1)
    .set_module_name_regex("foo.*", qconfig2)
    .set_module_name("module1", qconfig1)
    .set_module_name("module2", qconfig2)
    .set_module_name_object_type_order("foo.bar", torch.nn.functional.linear, 0, qconfig3)
```
Setting `qconfig1` for `torch.nn.Linear` means, whenever we see a `call_module` to an module instance of `torch.nn.Linear`, we'll apply `qconfig1`, we'll insert input and output observer for this `call_module` node based on the observer constructors specified in `qconfig1`.

Note that we only apply QConfig if the configuration is supported by `BackendConfig`, for more details please see https://github.com/pytorch/pytorch/blob/master/torch/ao/quantization/fx/README.md#13-quantdequantstub-and-observerfakequantize-insertion

## [BackendConfig](https://pytorch.org/docs/master/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig)
`BackendConfig` is used to configure what are the different ways of quantization are supported for an operator or operator pattern, e.g. a backend may support both static and dynamic quantized linear, they can express this information with `BackendConfig`, and if a modeling user requests a type of quantization that is not supported by the backend. In general, there are three possible cases:
(1). The operator or operator pattern is recognized (the entry exists in BackendConfig) and the requested type of quantization is supported (e.g. static int8 quantization), we'll insert observers based on the `QConfig`
(2). The operator or operator pattern is recognized, but the requested type of quantization (e.g. requested fp16, but only static int8 is supported)  is not supported in BackendConfig: we'll print a warning
(3). The operator or operator pattern is not recognized, we'll ignore the request

# Appendix
* More explanations on patterns: https://docs.google.com/document/d/1kSM0n05vI5Y939n2U3YPD5MWOQNkgXMTpt10ASU-Kns/edit
