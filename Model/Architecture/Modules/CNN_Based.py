# Author: yida
# Time is: 2023/9/2 23:36
# this Code: 基于CNN提取红外与可见光的细节特征

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
from timm.models import register_model

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# from models.model_config import MODEL_SPECS


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
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
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


class CNN_Based_Model(nn.Module):
    def __init__(self, in_channel, use_bn=False):
        super(CNN_Based_Model, self).__init__()

        model_sep = {
            "block_name": "uib",
            "num_blocks": 6,
            "block_specs": [
                [in_channel, in_channel, 3, 3, True, 1, 4],
                [in_channel, in_channel, 5, 5, True, 1, 4],
                [in_channel, in_channel, 5, 5, True, 1, 4],
                [in_channel, in_channel, 7, 7, True, 1, 4],
                [in_channel, in_channel, 7, 7, True, 1, 4],
                [in_channel, in_channel, 3, 3, True, 1, 4],
            ]
        }
        self.mobilenetv4_block = build_blocks(model_sep)

    def forward(self, x):
        x = self.mobilenetv4_block(x) + x
        return x


if __name__ == '__main__':
    inputs = torch.rand(1, 16, 256, 256).to(device)
    model = CNN_Based_Model(in_channel=16, use_bn=False).to(device)
    print(model)
    outputs = model(inputs)
    print(outputs.shape)
    summary(model, (16, 256, 256))
