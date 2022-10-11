from typing import Tuple, Union

import torch.nn as nn
import torch.nn.functional as F

from .util.quantization import weight_quantize_fn, act_quantize_fn


class QuantConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size: Union[int, Tuple], stride: Union[int, Tuple] = 1,
                 padding: Union[int, Tuple] = 0, dilation: Union[int, Tuple] = 1, groups: int = 1, bias=False, bit_w=8,
                 bit_a=8):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias)
        self.layer_type = 'QuantConv2d'
        self.bit_w = bit_w
        self.bit_a = bit_a
        self.weight_quant = weight_quantize_fn(self.bit_w)
        self.act_quant = act_quantize_fn(self.bit_a)

    def forward(self, x):
        weight_q = self.weight_quant(self.weight)
        x = self.act_quant(x)
        return F.conv2d(x, weight_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def change_bit(self, bit_w, bit_a):
        self.bit_w = bit_w
        self.bit_a = bit_a
        self.weight_quant.change_bit(bit_w)
        self.act_quant.change_bit(bit_a)
