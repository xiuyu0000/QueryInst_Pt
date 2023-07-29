from typing import List, Tuple, Optional
from collections import defaultdict

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, dtype as mstype

from src.engines.dii_head import DIIHead
from src.engines.dynamic_mask_head import DynamicMaskHead
from src.engines.single_roi_extractor import SingleRoIExtractor
from src.engines.hungarian_assigner import hungarian_assigner, pseudo_sampler


def unpack_gt_instances(batch_data_samples: List[dict]) -> tuple:
    """Unpack ``gt_instances``, ``gt_instances_ignore`` and ``img_metas`` based
    on ``batch_data_samples``

    Args:
        batch_data_samples (List[dict]): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

    Returns:
        tuple:

            - batch_gt_instances (list[dict]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            - batch_gt_instances_ignore (list[dict]):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            - batch_img_metas (list[dict]): Meta information of each image,
                e.g., image size, scaling factor, etc.
    """
    batch_gt_instances = []
    batch_gt_instances_ignore = []
    batch_img_metas = []
    for data_sample in batch_data_samples:
        batch_img_metas.append(data_sample["metainfo"])
        batch_gt_instances.append(data_sample["gt_instances"])
        if 'ignored_instances' in data_sample:
            batch_gt_instances_ignore.append(data_sample["ignored_instances"])
        else:
            batch_gt_instances_ignore.append(None)

    return batch_gt_instances, batch_gt_instances_ignore, batch_img_metas


def bbox2roi(bbox_list: List[Tensor]) -> Tensor:
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (List[Union[Tensor, :obj:`BaseBoxes`]): a list of bboxes
            corresponding to a batch of images.

    Returns:
        Tensor: shape (n, box_dim + 1), where ``box_dim`` depends on the
        different box types. For example, If the box type in ``bbox_list``
        is HorizontalBoxes, the output shape is (n, 5). Each row of data
        indicates [batch_ind, x1, y1, x2, y2].
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        img_inds = ops.fill(bboxes.dtype, (bboxes.shape[0], 1), img_id)
        rois = ops.cat([img_inds, bboxes], axis=-1)
        rois_list.append(rois)
    rois = ops.cat(rois_list, 0)
    return rois


class SparseRoIHead(nn.Cell):
    r"""The RoIHead for `Sparse R-CNN: End-to-End Object Detection with
    Learnable Proposals <https://arxiv.org/abs/2011.12450>`_
    and `Instances as Queries <http://arxiv.org/abs/2105.01928>`_

    Args:
        num_stages (int): Number of stage whole iterative process.
            Defaults to 6.
        proposal_feature_channel (int): Channel number of
            init_proposal_feature. Defaults to 256.
        bbox_roi_extractor (dict): Config of box
            roi extractor.
        mask_roi_extractor (dict): Config of mask
            roi extractor.
        bbox_head (dict): Config of box head.
        mask_head (dict): Config of mask head.
    """

    def __init__(self,
                 match_costs_config: List[dict],
                 num_stages: int = 6,
                 proposal_feature_channel: int = 256,
                 bbox_roi_extractor: Optional[dict] = None,
                 mask_roi_extractor: Optional[dict] = None,
                 bbox_head: Optional[dict] = None,
                 mask_head: Optional[dict] = None,
                 ) -> None:
        super().__init__()
        assert bbox_roi_extractor is not None
        assert bbox_head is not None

        self.num_stages = num_stages
        self.proposal_feature_channel = proposal_feature_channel
        self.share_roi_extractor = False
        self.bbox_roi_extractor = nn.SequentialCell()
        self.mask_roi_extractor = nn.SequentialCell()
        self.bbox_head = nn.SequentialCell()
        self.mask_head = nn.SequentialCell()
        self.match_costs_config = match_costs_config

        if bbox_head is not None:
            self.init_bbox_head(bbox_roi_extractor, bbox_head)

        if mask_head is not None:
            self.init_mask_head(mask_roi_extractor, mask_head)

    @property
    def with_bbox(self) -> bool:
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self) -> bool:
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    def init_bbox_head(self, bbox_roi_extractor: dict,
                       bbox_head: dict) -> None:
        """Initialize box head and box roi extractor.

        Args:
            bbox_roi_extractor (dict): Config of box roi extractor.
            bbox_head (dict): Config of box in box head.
        """
        if not isinstance(bbox_roi_extractor, list):
            bbox_roi_extractor = [
                bbox_roi_extractor for _ in range(self.num_stages)
            ]
        if not isinstance(bbox_head, list):
            bbox_head = [bbox_head for _ in range(self.num_stages)]
        assert len(bbox_roi_extractor) == len(bbox_head) == self.num_stages
        for roi_extractor, head in zip(bbox_roi_extractor, bbox_head):
            self.bbox_roi_extractor.append(SingleRoIExtractor(**roi_extractor))
            self.bbox_head.append(DIIHead(**head))

    def init_mask_head(self, mask_roi_extractor: dict,
                       mask_head: dict) -> None:
        """Initialize mask head and mask roi extractor.

        Args:
            mask_head (dict): Config of mask in mask head, out_size and sample_num.
            mask_roi_extractor (:obj:`ConfigDict`, dict or list):
                Config of mask roi extractor.
        """
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(DynamicMaskHead(**head))
        if mask_roi_extractor is not None:
            if not isinstance(mask_roi_extractor, list):
                mask_roi_extractor = [
                    mask_roi_extractor for _ in range(self.num_stages)
                ]
            assert len(mask_roi_extractor) == self.num_stages
            for roi_extractor in mask_roi_extractor:
                self.mask_roi_extractor.append(SingleRoIExtractor(**roi_extractor))
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor

    def _bbox_forward(self, stage: int, x: Tuple[Tensor], rois: Tensor,
                      object_feats: Tensor, batch_img_metas: List[dict]) -> dict:
        """Box head forward function used in both training and testing. Returns
        all regression, classification results and a intermediate feature.

        Args:
            stage (int): The current stage in iterative process.
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
                Each dimension means (img_index, x1, y1, x2, y2).
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            batch_img_metas (list[dict]): Meta information of each image.

        Returns:
            dict[str, Tensor]: a dictionary of bbox head outputs,
            Containing the following results:

            - cls_score (Tensor): The score of each class, has
              shape (batch_size, num_proposals, num_classes)
              when use focal loss or
              (batch_size, num_proposals, num_classes+1)
              otherwise.
            - decoded_bboxes (Tensor): The regression results
              with shape (batch_size, num_proposal, 4).
              The last dimension 4 represents
              [tl_x, tl_y, br_x, br_y].
            - object_feats (Tensor): The object feature extracted
              from current stage
            - detached_cls_scores (list[Tensor]): The detached
              classification results, length is batch_size, and
              each tensor has shape (num_proposal, num_classes).
            - detached_proposals (list[tensor]): The detached
              regression results, length is batch_size, and each
              tensor has shape (num_proposal, 4). The last
              dimension 4 represents [tl_x, tl_y, br_x, br_y].
        """
        num_imgs = len(batch_img_metas)
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs],
                                        rois)
        cls_score, bbox_pred, object_feats, attn_feats = bbox_head(
            bbox_feats, object_feats)

        fake_bbox_results = dict(
            rois=rois,
            bbox_targets=(rois.new_zeros(len(rois), dtype=mstype.int64)),
            bbox_pred=bbox_pred.view((-1, bbox_pred.shape[-1])),
            cls_score=cls_score.view((-1, cls_score.shape[-1])))
        fake_sampling_results = [
            dict(pos_is_gt=rois.new_zeros(object_feats.shape[1]))
            for _ in range(len(batch_img_metas))
        ]

        results_list = bbox_head.refine_bboxes(
            sampling_results=fake_sampling_results,
            bbox_results=fake_bbox_results,
            batch_img_metas=batch_img_metas)
        proposal_list = [res.bboxes for res in results_list]
        bbox_results = dict(
            cls_score=cls_score,
            decoded_bboxes=ops.cat(proposal_list),
            object_feats=object_feats,
            attn_feats=attn_feats,
            # detach then use it in label assign
            detached_cls_scores=[
                cls_score[i] for i in range(num_imgs)
            ],
            detached_proposals=[item for item in proposal_list])

        return bbox_results

    def bbox_forward(self,
                     stage: int,
                     x: Tuple[Tensor],
                     results_list: List[dict],
                     object_feats: Tensor,
                     batch_img_metas: List[dict],
                     batch_gt_instances: List[dict]) -> dict:
        """Perform forward propagation and loss calculation of the bbox head on
        the features of the upstream network.

        Args:
            stage (int): The current stage in iterative process.
            x (tuple[Tensor]): List of multi-level img features.
            results_list (List[dict]) : List of region
                proposals.
            object_feats (Tensor): The object feature extracted from
                the previous stage.
            batch_img_metas (list[dict]): Meta information of each image.
            batch_gt_instances (list[dict]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.

        Returns:
            dict[str, Tensor]
        """
        proposal_list = [res.bboxes for res in results_list]
        rois = bbox2roi(proposal_list)
        bbox_results = self._bbox_forward(stage, x, rois, object_feats,
                                          batch_img_metas)
        cls_pred_list = bbox_results['detached_cls_scores']
        proposal_list = bbox_results['detached_proposals']

        sampling_results = []
        for i in range(len(batch_img_metas)):
            pred_instances = defaultdict()
            # TODO: Enhance the logic
            pred_instances["bboxes"] = proposal_list[i]  # for assinger
            pred_instances["scores"] = cls_pred_list[i]
            pred_instances["priors"] = proposal_list[i]  # for sampler

            assign_result = hungarian_assigner(
                match_costs_config=self.match_costs_config,
                pred_instances=pred_instances,
                gt_instances=batch_gt_instances[i],
                img_meta=batch_img_metas[i])

            sampling_result = pseudo_sampler(
                assign_result, pred_instances, batch_gt_instances[i])
            sampling_results.append(sampling_result)

        bbox_results.update(sampling_results=sampling_results)

        # propose for the new proposal_list
        proposal_list = []
        for idx in range(len(batch_img_metas)):
            results = defaultdict()
            results["imgs_whwh"] = results_list[idx]["imgs_whwh"]
            results["bboxes"] = bbox_results['detached_proposals'][idx]
            proposal_list.append(results)
        bbox_results.update(results_list=proposal_list)
        return bbox_results

    def _mask_forward(self, stage: int, x: Tuple[Tensor], rois: Tensor,
                      attn_feats) -> dict:
        """Mask head forward function used in both training and testing.

        Args:
            stage (int): The current stage in Cascade RoI Head.
            x (tuple[Tensor]): Tuple of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.
            attn_feats (Tensot): Intermediate feature get from the last
                diihead, has shape
                (batch_size*num_proposals, feature_dimensions)

        Returns:
            dict: Usually returns a dictionary with keys:

            - `mask_preds` (Tensor): Mask prediction.
        """
        mask_roi_extractor = self.mask_roi_extractor[stage]
        mask_head = self.mask_head[stage]
        mask_feats = mask_roi_extractor(x[:mask_roi_extractor.num_inputs],
                                        rois)
        # do not support caffe_c4 model anymore
        mask_preds = mask_head(mask_feats, attn_feats)

        mask_results = dict(mask_preds=mask_preds)
        return mask_results

    def construct(self, x: Tuple[Tensor], rpn_results_list: List[Tensor],
                  batch_data_samples: List[dict]) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            x (List[Tensor]): Multi-level features that may have different
                resolutions.
            rpn_results_list (List[Tensor]): List of region
                proposals.
            batch_data_samples (list[dict]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns
            tuple: A tuple of features from ``bbox_head`` and ``mask_head``
            forward.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        all_stage_bbox_results = []
        object_feats = ops.cat(
            [res.pop('features')[None, ...] for res in rpn_results_list])
        results_list = rpn_results_list
        if self.with_bbox:
            for stage in range(self.num_stages):
                bbox_results = self.bbox_forward(
                    stage=stage,
                    x=x,
                    results_list=results_list,
                    object_feats=object_feats,
                    batch_img_metas=batch_img_metas,
                    batch_gt_instances=batch_gt_instances)

                bbox_results.pop('results_list')
                bbox_res = bbox_results.copy()
                bbox_res.pop('sampling_results')
                all_stage_bbox_results.append((bbox_res,))

                if self.with_mask:
                    attn_feats = bbox_results['attn_feats']
                    sampling_results = bbox_results['sampling_results']

                    pos_rois = bbox2roi(
                        [res["pos_priors"] for res in sampling_results])

                    attn_feats = ops.cat([
                        feats[res["pos_inds"]]
                        for (feats, res) in zip(attn_feats, sampling_results)
                    ])
                    mask_results = self._mask_forward(stage, x, pos_rois,
                                                      attn_feats)
                    all_stage_bbox_results[-1] += (mask_results,)
        return tuple(all_stage_bbox_results)
