# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Optional, Union

import mindspore as ms
from mindspore import Tensor
import mindspore.ops as ops


def fp16_clamp(x, min=None, max=None):
    if x.dtype == ms.float16:
        return x.float().clamp(min, max).half()
    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.shape[-1] == 4 or bboxes1.shape[0] == 0)
    assert (bboxes2.shape[-1] == 4 or bboxes2.shape[0] == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    print(f"bboxes1: {bboxes1.shape}, bboxes2: {bboxes2.shape}")
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]

    if rows * cols == 0:
        return bboxes1.new_zeros(batch_shape + (rows,))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
            bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
            bboxes2[..., 3] - bboxes2[..., 1])

    lt = ops.maximum(bboxes1[..., :, None, :2],
                 bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
    rb = ops.minimum(bboxes1[..., :, None, 2:],
                 bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

    wh = fp16_clamp(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]

    if mode in ['iou', 'giou']:
        union = area1[..., None] + area2[..., None, :] - overlap
    else:
        union = area1[..., None]
    if mode == 'giou':
        enclosed_lt = ops.minimum(bboxes1[..., :, None, :2],
                              bboxes2[..., None, :, :2])
        enclosed_rb = ops.maximum(bboxes1[..., :, None, 2:],
                              bboxes2[..., None, :, 2:])

    eps = Tensor([eps], dtype=union.dtype)
    union = ops.maximum(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = ops.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


def bbox_xyxy_to_cxcywh(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), axis=-1)
    bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    return ops.cat(bbox_new, axis=-1)


class BaseMatchCost:
    """Base match cost class.

    Args:
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, weight: Union[float, int] = 1.) -> None:
        self.weight = weight

    @abstractmethod
    def __call__(self,
                 pred_instances: dict,
                 gt_instances: dict,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (dict): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (dict): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            img_meta (dict, optional): Image information.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pass


class BBoxL1Cost(BaseMatchCost):
    """BBoxL1Cost.

    Note: ``bboxes`` in ``InstanceData`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.
            Defaults to 'xyxy'.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 box_format: str = 'xyxy',
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self,
                 pred_instances: dict,
                 gt_instances: dict,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (:obj:`InstanceData`): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances["bboxes"]
        gt_bboxes = gt_instances["bboxes"]

        # convert box format
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
            pred_bboxes = bbox_xyxy_to_cxcywh(pred_bboxes)

        # normalized
        img_h, img_w = img_meta['img_shape']
        factor = Tensor([img_w, img_h, img_w, img_h], dtype=gt_bboxes.dtype).unsqueeze(0)
        gt_bboxes = gt_bboxes / factor
        pred_bboxes = pred_bboxes / factor

        bbox_cost = ops.cdist(pred_bboxes, gt_bboxes, p=1.)
        return bbox_cost * self.weight


class IoUCost(BaseMatchCost):
    """IoUCost.

    Note: ``bboxes`` in ``dict`` passed in is of format 'xyxy'
    and its coordinates are unnormalized.

    Args:
        iou_mode (str): iou mode such as 'iou', 'giou'. Defaults to 'giou'.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self, iou_mode: str = 'giou', weight: Union[float, int] = 1.):
        super().__init__(weight=weight)
        self.iou_mode = iou_mode

    def __call__(self,
                 pred_instances: dict,
                 gt_instances: dict,
                 img_meta: Optional[dict] = None,
                 **kwargs):
        """Compute match cost.

        Args:
            pred_instances (dict): ``bboxes`` inside is
                predicted boxes with unnormalized coordinate
                (x, y, x, y).
            gt_instances (dict): ``bboxes`` inside is gt
                bboxes with unnormalized coordinate (x, y, x, y).
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_bboxes = pred_instances["bboxes"]
        gt_bboxes = gt_instances["bboxes"]

        overlaps = bbox_overlaps(
            pred_bboxes, gt_bboxes, mode=self.iou_mode)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * self.weight


class FocalLossCost(BaseMatchCost):
    """FocalLossCost.

    Args:
        alpha (Union[float, int]): focal_loss alpha. Defaults to 0.25.
        gamma (Union[float, int]): focal_loss gamma. Defaults to 2.
        eps (float): Defaults to 1e-12.
        binary_input (bool): Whether the input is binary. Currently,
            binary_input = True is for masks input, binary_input = False
            is for label input. Defaults to False.
        weight (Union[float, int]): Cost weight. Defaults to 1.
    """

    def __init__(self,
                 alpha: Union[float, int] = 0.25,
                 gamma: Union[float, int] = 2,
                 eps: float = 1e-12,
                 weight: Union[float, int] = 1.) -> None:
        super().__init__(weight=weight)
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def _focal_loss_cost(self, cls_pred: Tensor, gt_labels: Tensor) -> Tensor:
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_queries, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            Tensor: cls_cost value with weight
        """
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + self.eps).log() * (
                1 - self.alpha) * cls_pred.pow(self.gamma)
        pos_cost = -(cls_pred + self.eps).log() * self.alpha * (
                1 - cls_pred).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

    def __call__(self,
                 pred_instances: dict,
                 gt_instances: dict,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (dict): Predicted instances which
                must contain ``scores`` or ``masks``.
            gt_instances (dict): Ground truth which must contain
                ``labels`` or ``mask``.
            img_meta (Optional[dict]): Image information. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        pred_scores = pred_instances["scores"]
        gt_labels = gt_instances["labels"]
        return self._focal_loss_cost(pred_scores, gt_labels)


MATCH_COSTS = dict(
    FocalLossCost=FocalLossCost,
    IoUCost=IoUCost,
    BBoxL1Cost=BBoxL1Cost,
)


def create_match_cost(match_cost_config: dict) -> BaseMatchCost:
    """Create match cost.

    Args:
        match_cost_config (dict): Config for creating match cost.

    Returns:
        BaseMatchCost: Created match cost.
    """
    if match_cost_config is None:
        return None
    print(f'Creating match cost {match_cost_config}')
    match_cost_type = match_cost_config.pop('type')
    match_cost = MATCH_COSTS[match_cost_type](**match_cost_config)
    assert isinstance(match_cost, BaseMatchCost)
    return match_cost
