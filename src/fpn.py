"""Queryinst feature pyramid network."""

import mindspore.nn as nn
import numpy as np
import mindspore.ops as ops
from mindspore import Tensor, context, dtype as mstype
import mindspore.common.initializer as init


def bias_init_zeros(shape):
    """Bias init method."""
    return Tensor(np.array(np.zeros(shape).astype(np.float32)), dtype=mstype.float32)


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad'):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    weights = init.initializer("XavierUniform", shape=shape, dtype=mstype.float32)
    shape_bias = (out_channels,)
    biass = bias_init_zeros(shape_bias)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=True, bias_init=biass)


class FPN(nn.Cell):
    """
    Feature pyramid network cell, usually uses as network neck.

    Applies the convolution on multiple, input feature maps
    and output feature map with same channel size. if required num of
    output larger than num of inputs, add extra maxpooling for further
    downsampling;

    Args:
        in_channels (tuple) - Channel size of input feature maps.
        out_channels (int) - Channel size output.
        num_outs (int) - Num of output features.

    Returns:
        Tuple, with tensors of same channel size.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs):
        super(FPN, self).__init__()

        if context.get_context("device_target") == "Ascend":
            self.cast_type = mstype.float16
        else:
            self.cast_type = mstype.float32

        self.num_outs = num_outs
        self.in_channels = in_channels
        self.fpn_layer = len(self.in_channels)

        assert not self.num_outs < len(in_channels)

        self.lateral_convs_list_ = []
        self.fpn_convs_ = []

        for _, channel in enumerate(in_channels):
            l_conv = _conv(channel, out_channels, kernel_size=1, stride=1,
                           padding=0, pad_mode='valid').to_float(self.cast_type)
            fpn_conv = _conv(out_channels, out_channels, kernel_size=3, stride=1,
                             padding=0, pad_mode='same').to_float(self.cast_type)
            self.lateral_convs_list_.append(l_conv)
            self.fpn_convs_.append(fpn_conv)
        self.lateral_convs_list = nn.layer.CellList(self.lateral_convs_list_)
        self.fpn_convs_list = nn.layer.CellList(self.fpn_convs_)
        self.interpolate1 = ops.ResizeBilinear((48, 84))
        self.interpolate2 = ops.ResizeBilinear((96, 168))
        self.interpolate3 = ops.ResizeBilinear((192, 336))
        self.maxpool = ops.MaxPool(kernel_size=1, strides=2, pad_mode="same")

    def construct(self, inputs):
        x = ()
        for i in range(self.fpn_layer):
            x += (self.lateral_convs_list[i](inputs[i]),)

        y = (x[3],)
        y = y + (x[2] + ops.cast(self.interpolate1(y[self.fpn_layer - 4]), self.cast_type),)
        y = y + (x[1] + ops.cast(self.interpolate2(y[self.fpn_layer - 3]), self.cast_type),)
        y = y + (x[0] + ops.cast(self.interpolate3(y[self.fpn_layer - 2]), self.cast_type),)

        z = ()
        for i in range(self.fpn_layer - 1, -1, -1):
            z = z + (y[i],)

        outs = ()
        for i in range(self.fpn_layer):
            outs = outs + (self.fpn_convs_list[i](z[i]),)

        for i in range(self.num_outs - self.fpn_layer):
            outs = outs + (self.maxpool(outs[3]),)
        return outs


if __name__ == "__main__":
    from src.resnet import ResNet, ResidualBlock
    net = ResNet(ResidualBlock, [3, 4, 6, 3], [64, 256, 512, 1024], [256, 512, 1024, 2048], False)
    inputs = Tensor(np.ones([1, 3, 768, 1344]), mstype.float32)
    bb_output = net(inputs)
    neck = FPN([256, 512, 1024, 2048], 256, 4)
    n_output = neck(bb_output)
    print([x.shape for x in n_output])
    # Outputs:
    # [(1, 256, 192, 336), (1, 256, 96, 168), (1, 256, 48, 84), (1, 256, 24, 42)]
