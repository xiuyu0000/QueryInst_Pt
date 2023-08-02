
from mmdet.datasets.coco import CocoDatasetQuery

train_dataset_configs = dict(
        ann_file="/data1/coco2017/annotations/instances_train2017.json",
        data_root="/data1/coco2017/train2017/",
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='mmdet.LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='mmdet.Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='mmdet.Pad', size_divisor=32),
            dict(type='mmdet.PackDetInputs')
        ])

train_datasets = CocoDatasetQuery(**train_dataset_configs)
