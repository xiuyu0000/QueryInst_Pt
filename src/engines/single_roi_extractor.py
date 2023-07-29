import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, dtype as mstype


class SingleRoIExtractor(nn.Cell):
    """
    Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (list): Strides of input feature maps.
        batch_size (int)： Batchsize.
        finest_scale (int): Scale threshold of mapping to level 0.
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 batch_size=1,
                 finest_scale=56):
        super(SingleRoIExtractor, self).__init__()
        self.batch_size = batch_size
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.out_size = roi_layer['out_size']
        self.sample_num = roi_layer['sample_num']
        self.roi_layers = self.build_roi_layers(self.featmap_strides)
        self.roi_layers = nn.CellList(self.roi_layers)

        self.finest_scale_ = finest_scale
        self.clamp = ops.clip_by_value

        self.equal = ops.Equal()
        self.select = ops.Select()

        _mode_16 = False
        self.dtype = np.float16 if _mode_16 else np.float32
        self.ms_dtype = mstype.float16 if _mode_16 else mstype.float32
        self.set_train_local(training=True)

    def set_train_local(self, training=True):
        """Set training flag."""
        self.training_local = training

        # Init tensor
        self.ones = Tensor(np.array(np.ones((self.batch_size, 1)), dtype=self.dtype))
        finest_scale = np.array(np.ones((self.batch_size, 1)), dtype=self.dtype) * self.finest_scale_
        self.finest_scale = Tensor(finest_scale)
        self.epslion = Tensor(np.array(np.ones((self.batch_size, 1)), dtype=self.dtype) * self.dtype(1e-6))
        self.zeros = Tensor(np.array(np.zeros((self.batch_size, 1)), dtype=np.int32))
        self.max_levels = Tensor(np.array(np.ones((self.batch_size, 1)), dtype=np.int32) * (self.num_levels - 1))
        self.res_ = Tensor(np.array(np.zeros((self.batch_size, self.out_channels,
                                              self.out_size, self.out_size)), dtype=self.dtype))

    @property
    def num_inputs(self):
        return len(self.featmap_strides)

    def build_roi_layers(self, featmap_strides):
        roi_layers = []
        for s in featmap_strides:
            layer_cls = ops.ROIAlign(self.out_size, self.out_size,
                                     spatial_scale=1 / s,
                                     sample_num=self.sample_num)
            roi_layers.append(layer_cls)
        return roi_layers

    def _c_map_roi_levels(self, rois):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = ops.sqrt(rois[::, 3:4:1] - rois[::, 1:2:1] + self.ones) * ops.sqrt(
            rois[::, 4:5:1] - rois[::, 2:3:1] + self.ones)

        target_lvls = ops.log2(scale / self.finest_scale + self.epslion)
        target_lvls = ops.floor(target_lvls)
        target_lvls = ops.cast(target_lvls, mstype.int32)
        target_lvls = self.clamp(target_lvls, self.zeros, self.max_levels)

        return target_lvls

    def construct(self, features, rois):
        """SingleRoIExtractor"""

        res = self.res_
        target_lvls = self._c_map_roi_levels(rois)
        for i in range(self.num_levels):
            mask = self.equal(target_lvls, ops.ScalarToArray()(i))
            mask = ops.Reshape()(mask, (-1, 1, 1, 1))
            roi_feats_t = self.roi_layers[i](features[i], rois)
            mask = self.cast(ops.Tile()(self.cast(mask, mstype.int32),
                                        (1, 256, self.out_size, self.out_size)), mstype.bool_)
            res = self.select(mask, roi_feats_t, res)

        return res
