from typing import List, Tuple, Union, Optional

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.common.initializer as init
from mindcv.models.layers import Conv2dNormActivation


class FPN(nn.Cell):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.
        upsample_cfg (:obj:`ConfigDict` or dict, optional): Config dict
            for interpolate layer. Defaults to dict(mode='nearest').
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        relu_before_extra_convs: bool = False,
        add_extra_convs: str = 'on_input',
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
        upsample_cfg: dict = dict(mode='nearest'),
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.add_extra_convs = add_extra_convs

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = nn.SequentialCell()
        self.fpn_convs = nn.SequentialCell()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = Conv2dNormActivation(
                in_channels[i],
                out_channels,
                1,
                norm=norm_cfg,
                activation=act_cfg)
            fpn_conv = Conv2dNormActivation(
                out_channels,
                out_channels,
                3,
                pad_mode='pad',
                padding=1,
                norm=norm_cfg,
                activation=act_cfg)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = Conv2dNormActivation(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    pad_mode='pad',
                    padding=1,
                    norm=norm_cfg,
                    activation=act_cfg)
                self.fpn_convs.append(extra_fpn_conv)
        self.init_weights()

    def construct(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + ops.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + ops.interpolate(
                    laterals[i], size=prev_shape, mode="nearest")

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(ops.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                extra_source = inputs[self.backbone_end_level - 1]
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](ops.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

    def init_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(init.XavierUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None and cell.bias.dim() > 1:
                    cell.bias.set_data(init.initializer(init.XavierUniform(), cell.bias.shape, cell.bias.dtype))


if __name__ == "__main__":
    import numpy as np
    from src.resnet import ResNet, ResidualBlock
    from mindspore import context

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

    net = ResNet(ResidualBlock, [3, 4, 6, 3], [64, 256, 512, 1024], [256, 512, 1024, 2048], False)
    inputs = Tensor(np.load("../data/inputs.npy")).unsqueeze(0)
    bb_output = net(inputs)
    neck = FPN([256, 512, 1024, 2048], 256, num_outs=4, add_extra_convs='on_input', start_level=0)
    n_output = neck(bb_output)
    print([x.shape for x in n_output])
    # Outputs:
    # [(1, 256, 200, 336), (1, 256, 100, 168), (1, 256, 50, 84), (1, 256, 25, 42)]
    for i, x in enumerate(n_output):
        np.save("../data/features{}.npy".format(i), x.asnumpy())
