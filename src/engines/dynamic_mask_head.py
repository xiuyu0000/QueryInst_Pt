from typing import Optional

import mindspore.common.initializer as init
import mindspore.nn as nn
from mindspore import Tensor
from mindcv.models.layers import Conv2dNormActivation

from .dii_head import DynamicConv


class DynamicMaskHead(nn.Cell):
    r"""Dynamic Mask Head for
    `Instances as Queries <http://arxiv.org/abs/2105.01928>`_

    Args:
        num_convs (int): Number of convolution layer.
            Defaults to 4.
        roi_feat_size (int): The output size of RoI extractor,
            Defaults to 14.
        in_channels (int): Input feature channels.
            Defaults to 256.
        conv_kernel_size (int): Kernel size of convolution layers.
            Defaults to 3.
        conv_out_channels (int): Output channels of convolution layers.
            Defaults to 256.
        num_classes (int): Number of classes.
            Defaults to 80
        class_agnostic (int): Whether generate class agnostic prediction.
            Defaults to False.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        upsample_cfg (:obj:`ConfigDict` or dict): The config for
            upsample layer.
        conv_cfg (:obj:`ConfigDict` or dict, optional): The convolution
            layer config.
        norm_cfg (:obj:`ConfigDict` or dict, optional): The norm layer config.
        dynamic_conv_cfg (:obj:`ConfigDict` or dict): The dynamic convolution
            layer config.
        loss_mask (:obj:`ConfigDict` or dict): The config for mask loss.
    """

    def __init__(self,
                 num_convs: int = 4,
                 in_channels: int = 256,
                 inner_channels: int = 64,
                 out_channels: Optional[int] = None,
                 input_feat_shape: int = 14,
                 conv_kernel_size: int = 3,
                 conv_out_channels: int = 256,
                 num_classes: int = 80,
                 class_agnostic: bool = False,
                 scale_factor: int = 2,
                 with_proj: bool = False,
                 norm_cfg: nn.Cell = nn.BatchNorm2d,
                 act: nn.Cell = nn.ReLU,
                 norm: nn.Cell = nn.LayerNorm,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_convs = num_convs
        self.fp16_enabled = False
        self.class_agnostic = class_agnostic
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.in_channels = in_channels
        self.out_channel = out_channels if out_channels else in_channels
        self.num_classes = num_classes
        self.scale_factor = scale_factor

        self.instance_interactive_conv = DynamicConv(in_channels,
                                                     inner_channels,
                                                     out_channels=self.out_channel,
                                                     input_feat_shape=input_feat_shape,
                                                     with_proj=with_proj,
                                                     act=act,
                                                     norm=norm)

        convs = []
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            convs.append(
                Conv2dNormActivation(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    norm=norm_cfg))
        self.convs = nn.SequentialCell(convs)

        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        self.upsample = nn.Conv2dTranspose(
            in_channels=upsample_in_channels,
            out_channels=self.conv_out_channels,
            kernel_size=self.scale_factor,
            stride=self.scale_factor,
            pad_mode='pad',
            has_bias=True)

        self.relu = nn.ReLU()

        predictor_cfg = dict(type='Conv'),
        self.predictor_cfg = predictor_cfg
        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = self.conv_out_channels
        self.conv_logits = nn.Conv2d(logits_in_channel, out_channels, 1)
        self.init_weights()

    def construct(self, roi_feat: Tensor, proposal_feat: Tensor) -> Tensor:
        """Forward function of DynamicMaskHead.

        Args:
            roi_feat (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            proposal_feat (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size*num_proposals, feature_dimensions)

          Returns:
            mask_preds (Tensor): Predicted foreground masks with shape
            (batch_size*num_proposals, num_classes, pooling_h*2, pooling_w*2).
        """
        proposal_feat = proposal_feat.reshape(-1, self.in_channels)
        proposal_feat_iic = self.instance_interactive_conv(roi_feat, proposal_feat)
        x = proposal_feat_iic.permute((0, 2, 1)).reshape(roi_feat.shape)

        for conv in self.convs:
            x = conv(x)
        if self.upsample is not None:
            x = self.upsample(x)
            x = self.relu(x)
        mask_preds = self.conv_logits(x)
        return mask_preds

    def init_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                if cell.weight.dim() > 1:
                    cell.weight.set_data(init.initializer(init.XavierUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None and cell.bias.dim() > 1:
                    cell.bias.set_data(init.initializer(init.XavierUniform(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm) or isinstance(cell, nn.BatchNorm2d):
                if cell.gamma.dim() > 1:
                    cell.gamma.set_data(init.initializer(init.XavierUniform(), cell.gamma.shape, cell.gamma.dtype))
                if cell.beta.dim() > 1:
                    cell.beta.set_data(init.initializer(init.XavierUniform(), cell.beta.shape, cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d) or isinstance(cell, nn.Conv2dTranspose):
                if cell.weight.dim() > 1:
                    cell.weight.set_data(init.initializer(init.XavierUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None and cell.bias.dim() > 1:
                    cell.bias.set_data(init.initializer(init.XavierUniform(), cell.bias.shape, cell.bias.dtype))
