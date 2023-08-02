from typing import List, Tuple, Optional

import mindspore.nn as nn
from mindspore import Tensor

from src.rpn_head import EmbeddingRPNHead
from src.roi_head import SparseRoIHead
from src.resnet import ResNet
from src.fpn_pt import FPN


class QueryInst(nn.Cell):
    """Base class for QueryInst detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 rpn_head: Optional[dict] = None,
                 roi_head: Optional[dict] = None, ) -> None:
        super().__init__()
        self.backbone = ResNet(**backbone)

        if neck is not None:
            self.neck = FPN(**neck)

        if rpn_head is not None:
            self.rpn_head = EmbeddingRPNHead(**rpn_head)

        if roi_head is not None:
            self.roi_head = SparseRoIHead(**roi_head)

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def construct(self, batch_inputs: Tensor,
                  batch_data_samples: List[dict]) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[dict]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        results = ()
        x = self.extract_feat(batch_inputs)

        rpn_results_list = self.rpn_head(
            x, batch_data_samples)

        roi_outs = self.roi_head(x, rpn_results_list,
                                 batch_data_samples)
        results = results + (roi_outs,)
        return results


if __name__ == '__main__':
    import numpy as np
    from src.resnet import ResidualBlock

    match_costs_config = [
        dict(type='FocalLossCost', weight=2.0),
        dict(type='BBoxL1Cost', weight=5.0),
        dict(type='IoUCost', iou_mode='giou')
    ]
    bbox_roi_extractor_config = dict(roi_layer=dict(out_size=7, sample_num=2),
                                     out_channels=256,
                                     featmap_strides=[4, 8, 16, 32])
    mask_roi_extractor_config = dict(roi_layer=dict(out_size=14, sample_num=2),
                                     out_channels=256,
                                     featmap_strides=[4, 8, 16, 32])
    bbox_head_config = dict(in_channel=256, inner_channel=64, out_channel=256)
    mask_head_config = dict(num_convs=4)

    net = QueryInst(backbone=dict(block=ResidualBlock,
                                  layer_nums=[3, 4, 6, 3],
                                  in_channels=[64, 256, 512, 1024],
                                  out_channels=[256, 512, 1024, 2048],
                                  weights_update=False),
                    neck=dict(in_channels=[256, 512, 1024, 2048],
                              out_channels=256,
                              num_outs=4,
                              add_extra_convs='on_input',
                              start_level=0),
                    rpn_head=dict(num_proposals=100, proposal_feature_channel=256),
                    roi_head=dict(match_costs_config=match_costs_config,
                                  bbox_roi_extractor=bbox_roi_extractor_config,
                                  mask_roi_extractor=mask_roi_extractor_config,
                                  bbox_head=bbox_head_config,
                                  mask_head=mask_head_config)
                    )
    inputs = Tensor(np.load("./data/inputs.npy")).unsqueeze(0)
    batch_gt_instances = {"bboxes": Tensor(np.load("./data/bboxes.npy")),
                          "labels": Tensor([1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 26, 0, 0])}
    batch_img_metas = {"img_shape": [768, 1344]}
    batch_data_samples = [{"gt_instances": batch_gt_instances, "metainfo": batch_img_metas}]
    res = net(inputs, batch_data_samples)
    print(res[0][-1][1]['mask_preds'].shape)
