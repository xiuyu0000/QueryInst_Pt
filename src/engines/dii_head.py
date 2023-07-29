from typing import Optional, Sequence, List

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, dtype as mstype
import mindspore.common.initializer as init


class DynamicConv(nn.Cell):
    """Implements Dynamic Convolution.

    This module generate parameters for each sample and
    use bmm to implement 1*1 convolution. Code is modified
    from the `official github repo <https://github.com/PeizeSun/
    SparseR-CNN/blob/main/projects/SparseRCNN/sparsercnn/head.py#L258>`_ .

    Args:
        in_channels (int): The input feature channel.
            Defaults to 256.
        feat_channels (int): The inner feature channel.
            Defaults to 64.
        out_channels (int, optional): The output feature channel.
            When not specified, it will be set to `in_channels`
            by default
        input_feat_shape (int): The shape of input feature.
            Defaults to 7.
        with_proj (bool): Project two-dimentional feature to
            one-dimentional feature. Default to True.
        act (nn.Cell): The activation of DynamicConv. Default ReLU.
        norm (nn.Cell): The normalization layer. Default
            layer normalization.
    """

    def __init__(self,
                 in_channels: int = 256,
                 feat_channels: int = 64,
                 out_channels: Optional[int] = None,
                 input_feat_shape: int = 7,
                 with_proj: bool = True,
                 act: nn.Cell = nn.ReLU,
                 norm: nn.Cell = nn.LayerNorm) -> None:
        super(DynamicConv, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.input_feat_shape = input_feat_shape
        self.with_proj = with_proj
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.in_channels * self.feat_channels
        self.num_params_out = self.out_channels * self.feat_channels
        self.dynamic_layer = nn.Dense(
            self.in_channels, self.num_params_in + self.num_params_out)

        self.norm_in = norm(self.feat_channels)
        self.norm_out = norm(self.out_channels)

        self.activation = act()

        num_output = self.out_channels * input_feat_shape ** 2
        if self.with_proj:
            self.fc_layer = nn.Dense(num_output, self.out_channels)
            self.fc_norm = norm(self.out_channels)

    def construct(self, param_feature: Tensor, input_feature: Tensor) -> Tensor:
        """Forward function for `DynamicConv`.

        Args:
            param_feature (Tensor): The feature can be used
                to generate the parameter, has shape
                (num_all_proposals, in_channels).
            input_feature (Tensor): Feature that
                interact with parameters, has shape
                (num_all_proposals, in_channels, H, W).

        Returns:
            Tensor: The output feature has shape
            (num_all_proposals, out_channels).
        """
        input_feature = input_feature.flatten(start_dim=2).permute(2, 0, 1, 3)

        input_feature = input_feature.permute(1, 0, 2, 3)
        parameters = self.dynamic_layer(param_feature)

        param_in = parameters[:, :self.num_params_in].view((
            -1, self.in_channels, self.feat_channels))
        param_out = parameters[:, -self.num_params_out:].view((
            -1, self.feat_channels, self.out_channels))

        # input_feature has shape (num_all_proposals, H*W, in_channels)
        # param_in has shape (num_all_proposals, in_channels, feat_channels)
        # feature has shape (num_all_proposals, H*W, feat_channels)
        features = ops.bmm(input_feature, param_in)
        features = self.norm_in(features)
        features = self.activation(features)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = ops.bmm(features, param_out)
        features = self.norm_out(features)
        features = self.activation(features)

        if self.with_proj:
            features = features.flatten(start_dim=1)
            features = self.fc_layer(features)
            features = self.fc_norm(features)
            features = self.activation(features)

        return features


class DeltaXYWHBBoxCoder:
    """Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
    """

    def __init__(self,
                 target_means: Sequence[float] = (0., 0., 0., 0.),
                 target_stds: Sequence[float] = (0.5, 0.5, 1., 1.)
                 ) -> None:
        super(DeltaXYWHBBoxCoder, self).__init__()
        self.target_means = Tensor(target_means)
        self.target_stds = Tensor(target_stds)

    def encoder(self, anchors: Tensor, gt_boxes: Tensor):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            anchors (Tensor): Source boxes,
                e.g., object proposals.
            gt_boxes (Tensor): Target of the
                transformation, e.g., ground-truth boxes.

        Returns:
            Tensor: Box transformation deltas
        """
        anchors_wh = anchors[..., [2, 3]] - anchors[..., [0, 1]]
        anchors_xy = anchors[..., [0, 1]] + 0.5 * anchors_wh
        gt_wh = (gt_boxes[..., [2, 3]] - gt_boxes[..., [0, 1]]).clamp(min=1.0)
        gt_xy = gt_boxes[..., [0, 1]] + 0.5 * gt_wh
        delta_xy = (gt_xy - anchors_xy) / anchors_wh
        delta_wh = (gt_wh / anchors_wh).log()

        delta_targets = ops.cat([delta_xy, delta_wh], axis=-1).sub(
            self.target_means).div(self.target_stds)

        return delta_targets

    def decoder(self, predicts, anchors):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            anchors (Tensor): Basic boxes. Shape
                (B, N, 4) or (N, 4)
            predicts (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.

        Returns:
            Tensor: Decoded boxes.
        """
        anchors_wh = anchors[:, [2, 3]] - anchors[:, [0, 1]]
        anchors_xy = anchors[:, [0, 1]] + 0.5 * anchors_wh
        scale_reg = predicts * self.target_stds
        scale_wh = scale_reg[..., 2:].exp() * anchors_wh
        scale_x1y1 = (anchors_xy + scale_reg[..., :2] * anchors_wh) - 0.5 * scale_wh
        scale_x2y2 = scale_x1y1 + scale_wh
        return ops.cat([scale_x1y1, scale_x2y2], axis=-1)


class DIIHead(nn.Cell):
    r"""Dynamic Instance Interactive Head for `Sparse R-CNN: End-to-End Object
    Detection with Learnable Proposals <https://arxiv.org/abs/2011.12450>`_

    Args:
        in_channel (int): The input feature channel.
        inner_channel (int): The inner feature channel.
        out_channels (int, optional): The output feature channel.
            When not specified, it will be set to `in_channels`
            by default
        num_cls (int): Number of class in dataset.
            Defaults to 80.
        dim_feedforward (int): The hidden dimension
            of FFNs. Defaults to 2048
        nhead (int): The hidden dimension of FFNs.
            Defaults to 8.
        dropout (float): Probability of drop the channel.
            Defaults to 0.0
        pooling_resolution (int): The shape of input feature.
            Defaults to 7.
        activation (Cell): The activation function.
            Defaults to nn.ReLU.
        cls_tower_num (int): The number of fully-connected
            layers in classification subnet. Defaults to 1.
        reg_tower_num (int): The number of fully-connected
            layers in regression subnet. Defaults to 3.
    """

    def __init__(self,
                 in_channel,
                 inner_channel,
                 out_channel: Optional[int] = None,
                 num_cls: int = 80,
                 dim_feedforward: int = 2048,
                 nhead: int = 8,
                 dropout: float = 0.0,
                 pooling_resolution: int = 7,
                 activation: nn.Cell = nn.ReLU,
                 cls_tower_num: int = 1,
                 reg_tower_num: int = 3,
                 **kwargs):
        super(DIIHead, self).__init__()
        self.self_attn = nn.MultiheadAttention(in_channel, nhead, dropout=dropout)
        self.out_channel = out_channel if out_channel else in_channel

        self.inst_interact = DynamicConv(in_channel,
                                         inner_channel,
                                         out_channels=self.out_channel,
                                         input_feat_shape=pooling_resolution,
                                         act=activation,
                                         **kwargs)

        self.feed_forward = nn.SequentialCell([
            nn.Dense(in_channel, dim_feedforward),
            activation(),
            nn.Dropout(dropout),
            nn.Dense(dim_feedforward, in_channel)
        ])

        self.norm1 = nn.LayerNorm(in_channel)
        self.norm2 = nn.LayerNorm(in_channel)
        self.norm3 = nn.LayerNorm(in_channel)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.cls_tower = list()
        for _ in range(cls_tower_num):
            self.cls_tower.append(nn.Dense(in_channel, in_channel, has_bias=False))
            self.cls_tower.append(nn.LayerNorm(in_channel))
            self.cls_tower.append(activation())
        self.cls_tower = nn.SequentialCell(self.cls_tower)

        self.reg_tower = list()
        for _ in range(reg_tower_num):
            self.reg_tower.append(nn.Dense(in_channel, in_channel, has_bias=False))
            self.reg_tower.append(nn.LayerNorm(in_channel))
            self.reg_tower.append(activation())
        self.reg_tower = nn.SequentialCell(self.reg_tower)

        self.class_logits = nn.Dense(in_channel, num_cls)
        self.bboxes_delta = nn.Dense(in_channel, 4)
        self.box_coder = DeltaXYWHBBoxCoder()
        self.init_weights()

    def construct(self, x: Tensor, params_x: Tensor):
        """Forward function of Dynamic Instance Interactive Head.

        Args:
            x (Tensor): Roi-pooling features with shape
                (batch_size*num_proposals, feature_dimensions,
                pooling_h , pooling_w).
            params_x (Tensor): Intermediate feature get from
                diihead in last stage, has shape
                (batch_size, num_proposals, feature_dimensions)

        Returns:
            tuple[Tensor]: Usually a tuple of classification scores
            and bbox prediction and an intermediate feature.

            - cls_out (Tensor): Classification scores for
              all proposals, has shape
              (batch_size, num_proposals, num_classes).
            - pred_bboxes (Tensor): Box energies / deltas for
              all proposals, has shape
              (batch_size, num_proposals, 4).
            - out (Tensor): Object feature before classification
              and regression subnet, has shape
              (batch_size, num_proposal, feature_dimensions).
            - params_attn (Tensor): Intermediate feature.
        """
        nxp, c, _, _ = x.shape
        n, p, _ = params_x.shape
        # [res**2,N * nr_boxes,in_channel]
        x = x.view(nxp, c, -1).permute((2, 0, 1))
        # [nr_boxes, N, in_channel]
        params_x = params_x.permute((1, 0, 2))
        params_attn = self.self_attn(params_x, params_x, value=params_x)[0]
        params_attn = self.norm1(params_x + self.dropout1(params_attn))

        params_x = params_attn.permute((1, 0, 2)).view((-1, params_x.size(2)))
        # [N*nr_boxes,in_channel]
        param_intersect = self.inst_interact(x, params_x)
        params_x = self.norm2(params_x + self.dropout2(param_intersect))

        param_feedforward = self.feed_forward(params_x)
        # [N*nr_boxes,in_channel]
        out = self.norm3(params_x + self.dropout3(param_feedforward))
        cls_tower = self.cls_tower(out)
        reg_tower = self.reg_tower(out)
        cls_out = self.class_logits(cls_tower)
        reg_delta = self.bboxes_delta(reg_tower)
        return cls_out.view(n, p, -1), reg_delta.view(
            n, p, -1), out.view(n, p, -1), params_attn

    def init_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                if cell.weight.dim() > 1:
                    cell.weight.set_data(init.initializer(init.XavierUniform(), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None and cell.bias.dim() > 1:
                    cell.bias.set_data(init.initializer(init.XavierUniform(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                if cell.gamma.dim() > 1:
                    cell.gamma.set_data(init.initializer(init.XavierUniform(), cell.gamma.shape, cell.gamma.dtype))
                if cell.beta.dim() > 1:
                    cell.beta.set_data(init.initializer(init.XavierUniform(), cell.beta.shape, cell.beta.dtype))

    def refine_bboxes(self,
                      sampling_results: List[dict],
                      bbox_results: dict,
                      batch_img_metas: List[dict]) -> List:
        """Refine bboxes during training.

        Args:
            sampling_results (List[dict]): Sampling results.
                :obj:`dict` is the real sampling results
                calculate from bbox_head or fake sampling results,
                e.g., in Sparse R-CNN or QueryInst, etc.
            bbox_results (dict): Usually is a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `rois` (Tensor): RoIs with the shape (n, 5) where the first
                  column indicates batch id of each RoI.
                - `bbox_targets` (tuple):  Ground truth for proposals in a
                  single image. Containing the following list of Tensors:
                  (labels, label_weights, bbox_targets, bbox_weights)
            batch_img_metas (List[dict]): List of image information.

        Returns:
            list[dict]: Refined bboxes of each image.
        """
        pos_is_gts = [res.pos_is_gt for res in sampling_results]
        # bbox_targets is a tuple
        labels = bbox_results['bbox_targets'][0]
        cls_scores = bbox_results['cls_score']
        rois = bbox_results['rois']
        bbox_preds = bbox_results['bbox_pred']
        if cls_scores.numel() == 0:
            return None
        if cls_scores.shape[-1] == self.num_classes + 1:
            # remove background class
            cls_scores = cls_scores[:, :-1]
        elif cls_scores.shape[-1] != self.num_classes:
            raise ValueError('The last dim of `cls_scores` should equal to '
                             '`num_classes` or `num_classes + 1`,'
                             f'but got {cls_scores.shape[-1]}.')
        labels_condition = labels == self.num_classes
        labels = ops.where(Tensor(labels_condition), cls_scores.argmax(1),
                           labels)

        img_ids, _ = ops.unique(rois[:, 0]).long()
        assert img_ids.numel() <= len(batch_img_metas)

        results_list = []
        for i in range(len(batch_img_metas)):
            inds = ops.nonzero(rois[:, 0] == i).squeeze(axis=1)
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = batch_img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep
            results = dict(bboxes=bboxes[keep_inds.astype(mstype.bool_)])
            results_list.append(results)

        return results_list

    def regress_by_class(self, priors: Tensor, label: Tensor,
                         bbox_pred: Tensor, img_meta: dict) -> Tensor:
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            priors (Tensor): Priors from `rpn_head` or last stage
                `bbox_head`, has shape (num_proposals, 4).
            label (Tensor): Only used when `self.reg_class_agnostic`
                is False, has shape (num_proposals, ).
            bbox_pred (Tensor): Regression prediction of
                current stage `bbox_head`. When `self.reg_class_agnostic`
                is False, it has shape (n, num_classes * 4), otherwise
                it has shape (n, 4).
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        reg_dim = self.bbox_coder.encode_size
        if not self.reg_class_agnostic:
            label = label * reg_dim
            inds = ops.stack([label + i for i in range(reg_dim)], 1)
            bbox_pred = ops.gather_elements(bbox_pred, 1, inds)
        assert bbox_pred.shape[1] == reg_dim

        max_shape = img_meta['img_shape']
        regressed_bboxes = self.bbox_coder.decode(
            priors, bbox_pred, max_shape=max_shape)
        return regressed_bboxes
