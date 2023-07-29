from typing import List, Tuple, Optional

import mindspore.nn as nn
from mindspore import Tensor

from src.rpn_head import EmbeddingRPNHead
from src.roi_head import SparseRoIHead
from src.resnet import ResNet
from src.fpn import FPN


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
