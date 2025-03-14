# Author: yida
# Time is: 2023/9/3 1:22
# this Code: 变形场预测模块

import sys
import math
import numpy as np
import einops
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.utils.checkpoint as checkpoint
from torch.distributions.normal import Normal
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from torchsummary import summary

import numbers
from timm.models import register_model

from typing import Optional

import os
import kornia.utils as KU

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MNV4ConvSmall_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [32, 32, 3, 2],
            [32, 32, 1, 1]
        ]
    },
    "layer2": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [32, 96, 3, 2],
            [96, 64, 1, 1]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            [64, 96, 5, 5, True, 2, 3],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 0, 3, True, 1, 2],
            [96, 96, 3, 0, True, 1, 4],
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            [96, 128, 3, 3, True, 2, 6],
            [128, 128, 5, 5, True, 1, 4],
            [128, 128, 0, 5, True, 1, 4],
            [128, 128, 0, 5, True, 1, 3],
            [128, 128, 0, 3, True, 1, 4],
            [128, 128, 0, 3, True, 1, 4],
        ]
    },
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [128, 960, 1, 1],
            [960, 1280, 1, 1]
        ]
    }
}

MNV4ConvMedium_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [32, 48, 2, 4.0, True]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 80, 3, 5, True, 2, 4],
            [80, 80, 3, 3, True, 1, 2]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 8,
        "block_specs": [
            [80, 160, 3, 5, True, 2, 6],
            [160, 160, 3, 3, True, 1, 4],
            [160, 160, 3, 3, True, 1, 4],
            [160, 160, 3, 5, True, 1, 4],
            [160, 160, 3, 3, True, 1, 4],
            [160, 160, 3, 0, True, 1, 4],
            [160, 160, 0, 0, True, 1, 2],
            [160, 160, 3, 0, True, 1, 4]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            [160, 256, 5, 5, True, 2, 6],
            [256, 256, 5, 5, True, 1, 4],
            [256, 256, 3, 5, True, 1, 4],
            [256, 256, 3, 5, True, 1, 4],
            [256, 256, 0, 0, True, 1, 4],
            [256, 256, 3, 0, True, 1, 4],
            [256, 256, 3, 5, True, 1, 2],
            [256, 256, 5, 5, True, 1, 4],
            [256, 256, 0, 0, True, 1, 4],
            [256, 256, 0, 0, True, 1, 4],
            [256, 256, 5, 0, True, 1, 2]
        ]
    },
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [256, 960, 1, 1],
            [960, 1280, 1, 1]
        ]
    }
}

MNV4ConvLarge_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 24, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [24, 48, 2, 4.0, True]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 96, 3, 5, True, 2, 4],
            [96, 96, 3, 3, True, 1, 4]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            [96, 192, 3, 5, True, 2, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 5, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 3, 0, True, 1, 4]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 13,
        "block_specs": [
            [192, 512, 5, 5, True, 2, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 3, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 3, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4]
        ]
    },
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [512, 960, 1, 1],
            [960, 1280, 1, 1]
        ]
    }
}


def mhsa(num_heads, key_dim, value_dim, px):
    if px == 24:
        kv_strides = 2
    elif px == 12:
        kv_strides = 1
    query_h_strides = 1
    query_w_strides = 1
    use_layer_scale = True
    use_multi_query = True
    use_residual = True
    return [
        num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides,
        use_layer_scale, use_multi_query, use_residual
    ]


MNV4HybridConvMedium_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [32, 48, 2, 4.0, True]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 80, 3, 5, True, 2, 4],
            [80, 80, 3, 3, True, 1, 2]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 8,
        "block_specs": [
            [80, 160, 3, 5, True, 2, 6],
            [160, 160, 0, 0, True, 1, 2],
            [160, 160, 3, 3, True, 1, 4],
            [160, 160, 3, 5, True, 1, 4, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 3, True, 1, 4, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 0, True, 1, 4, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 3, True, 1, 4, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 0, True, 1, 4]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 12,
        "block_specs": [
            [160, 256, 5, 5, True, 2, 6],
            [256, 256, 5, 5, True, 1, 4],
            [256, 256, 3, 5, True, 1, 4],
            [256, 256, 3, 5, True, 1, 4],
            [256, 256, 0, 0, True, 1, 2],
            [256, 256, 3, 5, True, 1, 2],
            [256, 256, 0, 0, True, 1, 2],
            [256, 256, 0, 0, True, 1, 4, mhsa(4, 64, 64, 12)],
            [256, 256, 3, 0, True, 1, 4, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 5, True, 1, 4, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 0, True, 1, 4, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 0, True, 1, 4]
        ]
    },
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [256, 960, 1, 1],
            [960, 1280, 1, 1]
        ]
    }
}

MNV4HybridConvLarge_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [3, 24, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [24, 48, 2, 4.0, True]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 96, 3, 5, True, 2, 4],
            [96, 96, 3, 3, True, 1, 4]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            [96, 192, 3, 5, True, 2, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 3, True, 1, 4],
            [192, 192, 3, 5, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4],
            [192, 192, 5, 3, True, 1, 4, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True, 1, 4, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True, 1, 4, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True, 1, 4, mhsa(8, 48, 48, 24)],
            [192, 192, 3, 0, True, 1, 4]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 14,
        "block_specs": [
            [192, 512, 5, 5, True, 2, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 3, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 0, True, 1, 4],
            [512, 512, 5, 3, True, 1, 4],
            [512, 512, 5, 5, True, 1, 4, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4]
        ]
    },
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [512, 960, 1, 1],
            [960, 1280, 1, 1]
        ]
    }
}

MODEL_SPECS = {
    "MobileNetV4ConvSmall": MNV4ConvSmall_BLOCK_SPECS,
    "MobileNetV4ConvMedium": MNV4ConvMedium_BLOCK_SPECS,
    "MobileNetV4ConvLarge": MNV4ConvLarge_BLOCK_SPECS,
    "MobileNetV4HybridMedium": MNV4HybridConvMedium_BLOCK_SPECS,
    "MobileNetV4HybridLarge": MNV4HybridConvLarge_BLOCK_SPECS
}


# --------------------------模型-------------------------------#

def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
) -> int:
    """
    This function is copied from here
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"

    This is to ensure that all layers have channels that are divisible by 8.

    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.

    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.InstanceNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU6())
    return conv


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, act=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=1, stride=1))
        self.block.add_module('conv_3x3',
                              conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
                 inp,
                 oup,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 middle_dw_downsample,
                 stride,
                 expand_ratio
                 ):
        super().__init__()
        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_,
                                      groups=expand_filters)
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)

        # Ending depthwise conv.
        # this not used
        # _end_dw_kernel_size = 0
        # self._end_dw = conv_2d(oup, oup, kernel_size=_end_dw_kernel_size, stride=stride, groups=inp, act=False)

    def forward(self, x):
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        return x


def build_blocks(layer_spec):
    if not layer_spec.get('block_name'):
        return nn.Sequential()
    block_names = layer_spec['block_name']
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ['inp', 'oup', 'kernel_size', 'stride']
        args = {}
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"convbn_{i}", conv_2d(**args))
    elif block_names == "uib":
        schema_ = ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride',
                   'expand_ratio']
        args = {}
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
    elif block_names == "fused_ib":
        schema_ = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
        args = {}
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers


# Restormer
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AttentionBase(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False, ):
        super(AttentionBase, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv1 = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.qkv2 = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding='same', bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        b, c, h, w = x.shape
        qkv = self.qkv2(self.qkv1(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.proj(out)
        return out


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding='same', groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding='same', groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding='same', groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))

        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)

        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(x)
        x = x + self.ffn(self.norm2(x))

        return x


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

class CALayer(nn.Module):  # input shape: n, h, w, c
    """Squeeze-and-excitation block for channel attention."""

    def __init__(self, num_channels, reduction=4, use_bias=True):
        super().__init__()

        self.Conv_0 = nn.Conv2d(num_channels, num_channels // reduction, kernel_size=1, stride=1, bias=use_bias)
        self.relu = nn.ReLU()
        self.Conv_1 = nn.Conv2d(num_channels // reduction, num_channels, kernel_size=1, stride=1, bias=use_bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in):
        x = torch.mean(x_in, dim=(2, 3), keepdim=True)
        x = self.Conv_0(x)
        x = self.relu(x)
        x = self.Conv_1(x)
        w = self.sigmoid(x)

        x_out = x_in * w
        return x_out


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
class Correlation(nn.Module):
    def __init__(self, max_disp=1, kernel_size=1, stride=1, use_checkpoint=False):
        assert kernel_size == 1, "kernel_size other than 1 is not implemented"
        assert stride == 1, "stride other than 1 is not implemented"
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.max_disp = max_disp
        self.padlayer = nn.ConstantPad2d(max_disp, 0)

    def forward_run(self, x_1, x_2):
        x_2 = self.padlayer(x_2)
        offsetx, offsety = torch.meshgrid([torch.arange(0, 2 * self.max_disp + 1),
                                           torch.arange(0, 2 * self.max_disp + 1)], indexing='ij')

        h, w = x_1.shape[2], x_1.shape[3]  # 获取输入张量的高度和宽度

        # 确保切片操作不会超出边界
        # x_out = torch.cat([torch.mean(
        #     x_1 * x_2[:, :, dy:dy + h, dx:dx + w], 1, keepdim=True)
        #     for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))
        #     if dy + h <= x_2.shape[2] and dx + w <= x_2.shape[3]
        # ], 1)

        # 确保切片操作不会超出边界
        x_out = torch.cat([
            torch.mean(x_1 * x_2[:, :, dy:dy + h, dx:dx + w], 1, keepdim=True)
            for dx, dy in zip(offsetx.reshape(-1), offsety.reshape(-1))
            if (dy >= 0 and dy + h <= x_2.shape[2]) and (dx >= 0 and dx + w <= x_2.shape[3])
        ], 1)

        return x_out

    def forward(self, x_1, x_2):
        # if self.use_checkpoint and x_1.requires_grad and x_2.requires_grad:
        #     x = checkpoint(self.forward_run, x_1, x_2)
        # else:
        x = self.forward_run(x_1, x_2)
        return x


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

class Restormer(nn.Module):
    def __init__(self,
                 dim=64,
                 num_blocks=4,
                 heads=8,
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Restormer, self).__init__()

        self.Corr = Correlation(max_disp=1)

        self.Conv = nn.Conv2d(dim * 2 + 9, dim, kernel_size=3, stride=1, padding='same')
        self.restormer = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                                          bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])

    def forward(self, x, y):
        x_corr = self.Corr(x, y)
        x = torch.cat([x, x_corr, y], dim=1)
        x = self.Conv(x)
        x = self.restormer(x) + x
        return x


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #


class FlowUpsampler(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=2, mode='bilinear'):
        super(FlowUpsampler, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        # 定义转置卷积层，上采样倍数由scale_factor决定
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_c,  # 光流的通道数为2（水平和垂直位移）
            out_channels=out_c,
            kernel_size=4,
            stride=scale_factor,
            padding=1,
            output_padding=scale_factor - 2,
            bias=False
        )

    def forward(self, flow, target_size):
        """
        :param flow: 输入的光流，形状为 (B, 2, H, W)
        :param target_size: 目标特征图的尺寸，形状为 (H_target, W_target)
        :return: 上采样并调整尺寸后的光流，形状为 (B, 2, H_target, W_target)
        """
        # B, C, H, W = flow.shape
        # H_target, W_target = target_size

        # 使用转置卷积上采样光流
        upsampled_flow = self.deconv(flow)

        # 如果上采样后的尺寸与目标尺寸不一致，使用插值调整
        if (upsampled_flow.shape[2], upsampled_flow.shape[3]) != target_size:
            upsampled_flow = F.interpolate(
                upsampled_flow,
                size=target_size,
                mode=self.mode,
                align_corners=True
            )

        # # 光流值需要根据尺寸变化进行缩放
        # scale_y = H_target / H
        # scale_x = W_target / W
        # upsampled_flow[:, 0, :, :] *= scale_x  # 水平位移
        # upsampled_flow[:, 1, :, :] *= scale_y  # 垂直位移

        return upsampled_flow


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #


class SpatialTransformer(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]  # 交换坐标轴 不调整会让图像是倒的

        return nnf.grid_sample(src, torch.clamp(new_locs, min=-1, max=1), align_corners=True, mode=self.mode, padding_mode='reflection')


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

class Mobilenetv4_block(nn.Module):
    def __init__(self,
                 in_channels=16, num_blocks=1):
        super(Mobilenetv4_block, self).__init__()

        model_sep = {
            "block_name": "uib",
            "num_blocks": num_blocks,
            "block_specs": [
                [in_channels, in_channels, 3, 3, True, 1, 4],
                [in_channels, in_channels, 5, 5, True, 1, 4],
                [in_channels, in_channels, 5, 5, True, 1, 4],
                [in_channels, in_channels, 7, 7, True, 1, 4],
                [in_channels, in_channels, 7, 7, True, 1, 4],
                [in_channels, in_channels, 4, 3, True, 1, 4],
            ]
        }

        self.LayerNorm = nn.LayerNorm(in_channels)

        self.mobilenetv4_block = build_blocks(model_sep)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, bias=True, padding='same'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, bias=True, padding='same'),
        )
        self.ca_layer = CALayer(in_channels)

    def forward(self, x):
        shortcut = x
        x = x.permute(0, 2, 3, 1)
        x = self.LayerNorm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.mobilenetv4_block(x)

        x = self.ca_layer(x)
        x = x + shortcut
        return x


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

class Reg_flow(nn.Module):
    def __init__(self, in_channels: int = 16):
        super().__init__()

        self.reg_head = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding='same')
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))

    def forward(self, x):
        flow = self.reg_head(x)
        return flow


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

class Reg_decoder(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()

        # 第四个尺度
        self.mobilenetv4_block4 = Mobilenetv4_block(in_channels=in_channels * 8, num_blocks=1)
        self.restormer4 = Restormer(dim=in_channels * 8, num_blocks=1, heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')

        self.reg_flow4_a = Reg_flow(in_channels=in_channels * 8)
        self.reg_flow4_e = Reg_flow(in_channels=in_channels * 8)
        self.reg_flow4_f = Reg_flow(in_channels=4)

        self.flow4_a_up = FlowUpsampler(in_c=2, out_c=2, scale_factor=2)
        self.flow4_e_up = FlowUpsampler(in_c=2, out_c=2, scale_factor=2)
        self.flow4_f_up = FlowUpsampler(in_c=2, out_c=2, scale_factor=2)
        self.feature4_up = FlowUpsampler(in_c=in_channels * 8, out_c=in_channels * 4, scale_factor=2)

        # 第三个尺度
        self.mobilenetv4_block3 = Mobilenetv4_block(in_channels=in_channels * 4, num_blocks=1)
        self.restormer3 = Restormer(dim=in_channels * 4, num_blocks=1, heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.restormer3_fus = Restormer(dim=in_channels * 4, num_blocks=1, heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')

        self.reg_flow3_a = Reg_flow(in_channels=in_channels * 4)
        self.reg_flow3_e = Reg_flow(in_channels=in_channels * 4)
        self.reg_flow3_f = Reg_flow(in_channels=4)

        self.flow3_a_up = FlowUpsampler(in_c=2, out_c=2, scale_factor=2)
        self.flow3_e_up = FlowUpsampler(in_c=2, out_c=2, scale_factor=2)
        self.flow3_f_up = FlowUpsampler(in_c=2, out_c=2, scale_factor=2)
        self.feature3_up = FlowUpsampler(in_c=in_channels * 4, out_c=in_channels * 2, scale_factor=2)

        # 第二个尺度
        self.mobilenetv4_block2 = Mobilenetv4_block(in_channels=in_channels * 2, num_blocks=1)
        self.restormer2 = Restormer(dim=in_channels * 2, num_blocks=1, heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.restormer2_fus = Restormer(dim=in_channels * 2, num_blocks=1, heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')

        self.reg_flow2_a = Reg_flow(in_channels=in_channels * 2)
        self.reg_flow2_e = Reg_flow(in_channels=in_channels * 2)
        self.reg_flow2_f = Reg_flow(in_channels=4)

        self.flow2_a_up = FlowUpsampler(in_c=2, out_c=2, scale_factor=2)
        self.flow2_e_up = FlowUpsampler(in_c=2, out_c=2, scale_factor=2)
        self.flow2_f_up = FlowUpsampler(in_c=2, out_c=2, scale_factor=2)
        self.feature2_up = FlowUpsampler(in_c=in_channels * 2, out_c=in_channels * 1, scale_factor=2)

        # 第一个尺度
        self.mobilenetv4_block1 = Mobilenetv4_block(in_channels=in_channels, num_blocks=1)
        self.restormer1 = Restormer(dim=in_channels, num_blocks=1, heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')
        self.restormer1_fus = Restormer(dim=in_channels, num_blocks=1, heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias')

        self.reg_flow1_a = Reg_flow(in_channels=in_channels)
        self.reg_flow1_e = Reg_flow(in_channels=in_channels)
        self.reg_flow1_f = Reg_flow(in_channels=4)

        # self.flow1_a_up = FlowUpsampler(scale_factor=2)
        # self.flow1_e_up = FlowUpsampler(scale_factor=2)
        # self.flow1_f_up = FlowUpsampler(scale_factor=2)
        # self.feature1_up = FlowUpsampler(scale_factor=2)

        # 空间变换函数
        self.SpatialTransformer = SpatialTransformer()

    def forward(self, x_fix, x_mov):
        x_fix_1, x_fix_2, x_fix_3, x_fix_4 = x_fix
        x_mov_1, x_mov_2, x_mov_3, x_mov_4 = x_mov

        # 1、第四个尺度
        x_4 = self.restormer4(x_fix_4, x_mov_4)
        flow4_a = self.reg_flow4_a(x_4)
        x_4 = self.mobilenetv4_block4(x_4)
        flow4_e = self.reg_flow4_e(x_4)
        flow4_f = self.reg_flow4_f(torch.cat([flow4_a, flow4_e], dim=1))

        # 与第三个尺度进行对齐
        flow4_a = self.flow4_a_up(flow4_a, x_mov_3.shape[2:4])
        flow4_e = self.flow4_e_up(flow4_e, x_mov_3.shape[2:4])
        flow4_f = self.flow4_f_up(flow4_f, x_mov_3.shape[2:4])
        temp = self.feature4_up(x_4, x_mov_3.shape[2:4])

        # 第三个尺度特征的空间变换
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow4_a)
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow4_e)
        x_mov_3 = self.SpatialTransformer(x_mov_3, flow4_f)

        x_3 = self.restormer3(x_fix_3, x_mov_3)
        x_3 = self.restormer3_fus(x_3, temp)
        flow3_a = self.reg_flow3_a(x_3) + flow4_a
        x_3 = self.mobilenetv4_block3(x_3)
        flow3_e = self.reg_flow3_e(x_3) + flow4_e
        flow3_f = self.reg_flow3_f(torch.cat([flow3_a, flow3_e], dim=1)) + flow4_f

        # 2、第三个尺度
        # 与第二个尺度进行对齐
        flow3_a = self.flow3_a_up(flow3_a, x_mov_2.shape[2:4])
        flow3_e = self.flow3_e_up(flow3_e, x_mov_2.shape[2:4])
        flow3_f = self.flow3_f_up(flow3_f, x_mov_2.shape[2:4])
        temp = self.feature3_up(x_3, x_mov_2.shape[2:4])

        # 第二个尺度特征的空间变换
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow3_a)
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow3_e)
        x_mov_2 = self.SpatialTransformer(x_mov_2, flow3_f)

        x_2 = self.restormer2(x_fix_2, x_mov_2)
        x_2 = self.restormer2_fus(x_2, temp)
        flow2_a = self.reg_flow2_a(x_2) + flow3_a
        x_2 = self.mobilenetv4_block2(x_2)
        flow2_e = self.reg_flow2_e(x_2) + flow3_e

        # 3、第二个尺度
        flow2_f = self.reg_flow2_f(torch.cat([flow2_a, flow2_e], dim=1)) + flow3_f

        # 与第一个尺度进行对齐
        flow2_a = self.flow2_a_up(flow2_a, x_mov_1.shape[2:4])
        flow2_e = self.flow2_e_up(flow2_e, x_mov_1.shape[2:4])
        flow2_f = self.flow2_f_up(flow2_f, x_mov_1.shape[2:4])
        temp = self.feature2_up(x_2, x_mov_1.shape[2:4])

        # 第一个尺度特征的空间变换
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow2_a)
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow2_e)
        x_mov_1 = self.SpatialTransformer(x_mov_1, flow2_f)

        x_1 = self.restormer1(x_fix_1, x_mov_1)
        x_1 = self.restormer1_fus(x_1, temp)
        flow1_a = self.reg_flow1_a(x_1) + flow2_a
        x_1 = self.mobilenetv4_block1(x_1)
        flow1_e = self.reg_flow1_e(x_1) + flow2_e

        # 4、第一个尺度
        flow1_f = self.reg_flow1_f(torch.cat([flow1_a, flow1_e], dim=1)) + flow2_f

        return flow1_a, flow1_e, flow1_f


class Multiscale_feature(nn.Module):

    def __init__(self,
                 in_channels: int):
        super().__init__()

        self.down_sample = nn.AvgPool2d(2, stride=2)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm2d(2 * in_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(2 * in_channels, 2 * in_channels, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm2d(2 * in_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(2 * in_channels, 4 * in_channels, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm2d(4 * in_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(4 * in_channels, 4 * in_channels, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm2d(4 * in_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(4 * in_channels, 8 * in_channels, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm2d(8 * in_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(8 * in_channels, 8 * in_channels, kernel_size=3, stride=1, padding='same'),
            nn.InstanceNorm2d(8 * in_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        # in_channels //1
        x_1 = self.layer1(x)

        # 2 * in_channels //2
        x_2 = self.down_sample(x_1)
        x_2 = self.layer2(x_2)

        # 4 * in_channels //4
        x_3 = self.down_sample(x_2)
        x_3 = self.layer3(x_3)

        # 8 * in_channels //8
        x_4 = self.down_sample(x_3)
        x_4 = self.layer4(x_4)
        return [x_1, x_2, x_3, x_4]


class RegModel(nn.Module):

    def __init__(self,
                 in_channels: int = 32,
                 use_bn: bool = False):
        super().__init__()

        # 使用encoder
        self.Multiscale_feature = Multiscale_feature(in_channels=in_channels)

        self.Decoder = Reg_decoder(in_channels=in_channels)

        self.SpatialTransformer = SpatialTransformer(mode='bilinear')

    def forward(self, fixed, moving):
        # 多尺度特征
        x_fix = self.Multiscale_feature(fixed)
        x_mov = self.Multiscale_feature(moving)

        # decoder
        flow_a, flow_e, flow = self.Decoder(x_fix, x_mov)

        # 施加到moving上
        warped = self.SpatialTransformer(moving, flow_a)
        warped = self.SpatialTransformer(warped, flow_e)
        warped = self.SpatialTransformer(warped, flow)

        return warped, flow, x_fix, x_mov


# 共享部分
class DeformableFieldPredictor(nn.Module):
    def __init__(self, in_channels, use_bn=False):
        super(DeformableFieldPredictor, self).__init__()
        self.deformation = RegModel(in_channels=in_channels, use_bn=use_bn)

    def forward(self, ir_concat_features, vis_concat_features):
        """
        :param img_fusion_features: 融合特征用于预测变形场
        :return: 变换参数，分别作用到不同特征
        """
        warped, flow, x_fix, x_mov = self.deformation(fixed=vis_concat_features, moving=ir_concat_features)

        return warped, flow, x_fix, x_mov


if __name__ == '__main__':
    # 初始化模型和输入
    channels = 32  # 假设输入特征的通道数为16
    model = DeformableFieldPredictor(channels, use_bn=True).to(device)
    print(model)

    input1 = torch.rand((1, channels, 224, 224)).to(device)  # 假设每个输入特征都有形状 (1, 64, 224, 224)

    # 使用变形场预测模块 不使用蒸馏
    outputs, deformation, _, _ = model(input1, input1)
    print("Output feature shape1:", deformation.shape)  # 应该与输入特征有相同的形状

    # 使用变形场预测模块 使用蒸馏
    outputs, deformation, _, _ = model(input1, input1)
    print("Output feature shape1:", deformation.shape)  # 应该与输入特征有相同的形状
    import thop

    flops, params = thop.profile(model, inputs=(input1, input1,))
    # 格式化输出
    print("%-10s | %-12s | %-12s" % ("Model", "Params(M)", "FLOPs(G)"))
    print("-----------|--------------|--------------")
    print("%-10s | %-12.3f | %-12.3f" % ("MyModel", params / 1e6, flops / 1e9))

