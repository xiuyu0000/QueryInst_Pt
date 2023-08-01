from typing import List, Tuple
from collections import defaultdict

import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import Tensor


def bbox_cxcywh_to_xyxy(bbox: Tensor) -> Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.split((1, 1, 1, 1), axis=-1)
    bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return ops.cat(bbox_new, axis=-1)


class EmbeddingRPNHead(nn.Cell):
    """RPNHead in the `Sparse R-CNN <https://arxiv.org/abs/2011.12450>`_ .

    Unlike traditional RPNHead, this module does not need FPN input, but just
    decode `init_proposal_bboxes` and expand the first dimension of
    `init_proposal_bboxes` and `init_proposal_features` to the batch_size.

    Args:
        num_proposals (int): Number of init_proposals. Defaults to 100.
        proposal_feature_channel (int): Channel number of
            init_proposal_feature. Defaults to 256.

    """

    def __init__(self,
                 num_proposals: int = 100,
                 proposal_feature_channel: int = 256,
                 ) -> None:
        super().__init__()
        self.num_proposals = num_proposals
        self.proposal_feature_channel = proposal_feature_channel
        self._init_layers()
        self.init_weights()

    def _init_layers(self) -> None:
        """Initialize a sparse set of proposal boxes and proposal features."""
        self.init_proposal_bboxes = nn.Embedding(self.num_proposals, 4)
        self.init_proposal_features = nn.Embedding(
            self.num_proposals, self.proposal_feature_channel, embedding_table='normal')

    def init_weights(self) -> None:
        """Initialize the init_proposal_bboxes as normalized.

        [c_x, c_y, w, h], and we initialize it to the size of the entire
        image.
        """
        self.init_proposal_bboxes.embedding_table[:, :] = Tensor([[0.5, 0.5, 1., 1.]] * self.num_proposals,
                                                                 dtype=self.init_proposal_features.dtype)

    def _decode_init_proposals(self, x: Tuple[Tensor],
                               batch_data_samples: List[dict]) -> List:
        """Decode init_proposal_bboxes according to the size of images and
        expand dimension of init_proposal_features to batch_size.

        Args:
            x (Tuple[Tensor]): List of FPN features.
            batch_data_samples (List[dict]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            List[:obj:`InstanceData`:] Detection results of each image.
            Each item usually contains following keys.

            - proposals: Decoded proposal bboxes,
              has shape (num_proposals, 4).
            - features: init_proposal_features, expanded proposal
              features, has shape
              (num_proposals, proposal_feature_channel).
            - imgs_whwh: Tensor with shape
              (num_proposals, 4), the dimension means
              [img_width, img_height, img_width, img_height].
        """
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample["metainfo"])

        proposals = self.init_proposal_bboxes.embedding_table.clone()
        proposals = bbox_cxcywh_to_xyxy(proposals)
        imgs_whwh = []
        for meta in batch_img_metas:
            h, w = meta['img_shape'][:2]
            imgs_whwh.append(Tensor([[w, h, w, h]], dtype=x[0].dtype))
        imgs_whwh = ops.cat(imgs_whwh, axis=0)
        imgs_whwh = imgs_whwh[:, None, :]
        proposals = proposals * imgs_whwh

        rpn_results_list = []
        for idx in range(len(batch_img_metas)):
            rpn_results = defaultdict()
            rpn_results["bboxes"] = proposals[idx]
            rpn_results["imgs_whwh"] = imgs_whwh[idx].repeat(
                self.num_proposals, 1)
            rpn_results["features"] = self.init_proposal_features.embedding_table.clone()
            rpn_results_list.append(rpn_results)
        return rpn_results_list

    def construct(self, x: Tuple[Tensor], batch_data_samples: List[dict],
                  ) -> List:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network."""
        # `**kwargs` is necessary to avoid some potential error.
        return self._decode_init_proposals(
            x=x, batch_data_samples=batch_data_samples)


if __name__ == "__main__":
    import numpy as np
    # from mindspore import dtype as mstype
    # from src.resnet import ResNet, ResidualBlock
    # from src.fpn import FPN
    from mindspore import context
    #
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    # inputs = Tensor(np.ones([1, 3, 768, 1344]), mstype.float32)
    data_samples = [{"metainfo": {"img_shape": [768, 1344, 3]}}]
    #
    # net = ResNet(ResidualBlock, [3, 4, 6, 3], [64, 256, 512, 1024], [256, 512, 1024, 2048], False)
    # neck = FPN([256, 512, 1024, 2048], 256, num_outs=4)
    rpn_head = EmbeddingRPNHead()
    #
    # bb_output = net(inputs)
    # n_output = neck(bb_output)
    n_output = []
    for i in range(4):
        f = np.load("../data/features{}.npy".format(i))
        n_output.append(Tensor(f))
    n_output = tuple(n_output)
    rpn_results_list = rpn_head(n_output, data_samples)
    print([rpn_results_list[0][e].shape for e in rpn_results_list[0]])
    print([e for e in rpn_results_list[0]])
    # Output:
    # [(100, 4), (1, 400), (100, 256)]
    # ['bboxes', 'imgs_whwh', 'features']
    np.save("../data/rpn_results_list.npy", rpn_results_list, allow_pickle=True)
