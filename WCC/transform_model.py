from torch import nn
from .WCC import WCC
from .QuantConv2d import QuantConv2d


def quantize_deeplabmobilev2(model, bit_w, bit_a):
    first_conv = model.backbone.low_level_features[0][0]
    last_layer = model.classifier.classifier[3]
    quantize_module(model, bit_w, bit_a)
    model.backbone.low_level_features[0][0] = first_conv
    model.classifier.classifier[3] = last_layer


def wavelet_deeplabmobilev2(model, levels, compress_rate, bit_w, bit_a):
    first_conv = model.backbone.low_level_features[0][0]
    last_layer = model.classifier.classifier[3]
    wavelet_module(model, levels, compress_rate, bit_w, bit_a)
    model.backbone.low_level_features[0][0] = first_conv
    model.classifier.classifier[3] = last_layer


def quantize_module(module, bit_w, bit_a):
    new_module = module
    if isinstance(module, nn.Conv2d):
        new_module = QuantConv2d(module.in_channels,
                                 module.out_channels,
                                 module.kernel_size,
                                 module.stride,
                                 module.padding,
                                 module.dilation,
                                 module.groups,
                                 module.bias is not None,
                                 bit_w,
                                 bit_a)
        new_module.weight = module.weight
        new_module.bias = module.bias
    for name, child in module.named_children():
        new_module.add_module(name, quantize_module(child, bit_w, bit_a))
    return new_module


def change_module_bits(module, bit_w, bit_a):
    if isinstance(module, QuantConv2d) or isinstance(module, WCC):
        module.change_bit(bit_w, bit_a)
    else:
        for name, child in module.named_children():
            change_module_bits(child, bit_w, bit_a)


def wavelet_module(module, levels, compress_rate, bit_w, bit_a):
    new_module = module
    if isinstance(module, nn.Conv2d):
        if module.kernel_size[0] > 1:
            new_module = QuantConv2d(module.in_channels,
                                     module.out_channels,
                                     module.kernel_size,
                                     module.stride,
                                     module.padding,
                                     module.dilation,
                                     module.groups,
                                     module.bias is not None,
                                     bit_w,
                                     bit_a)
            new_module.weight = module.weight
            new_module.bias = module.bias
        else:
            new_module = WCC(module.in_channels,
                             module.out_channels,
                             module.stride[0],
                             module.padding[0],
                             module.dilation[0],
                             module.groups,
                             module.bias is not None,
                             levels,
                             compress_rate,
                             bit_w,
                             bit_a)
            new_module.weight = nn.Parameter(module.weight.squeeze(-1))
            new_module.bias = module.bias
        if isinstance(module, QuantConv2d):
            new_module.act_quant.a_alpha = module.act_quant.a_alpha
            new_module.weight_quant.w_alpha = module.weight_quant.w_alpha
    else:
        for name, child in module.named_children():
            new_module.add_module(name, wavelet_module(child, levels, compress_rate, bit_w, bit_a))
    return new_module


def change_module_wt_params(module, compress_rate, levels):
    if isinstance(module, WCC):
        module.change_wt_params(compress_rate, levels)
    else:
        for name, child in module.named_children():
            change_module_wt_params(child, compress_rate, levels)
