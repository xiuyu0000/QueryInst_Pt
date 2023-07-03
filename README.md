# QueryInst

> [Instances as Queries](https://openaccess.thecvf.com/content/ICCV2021/html/Fang_Instances_As_Queries_ICCV_2021_paper.html)

<!-- [ALGORITHM] -->

## Abstract

We present QueryInst, a new perspective for instance segmentation. QueryInst is a multi-stage end-to-end system that treats instances of interest as learnable queries, enabling query based object detectors, e.g., Sparse R-CNN, to have strong instance segmentation performance. The attributes of instances such as categories, bounding boxes, instance masks, and instance association embeddings are represented by queries in a unified manner. In QueryInst, a query is shared by both detection and segmentation via dynamic convolutions and driven by parallelly-supervised multi-stage learning. We conduct extensive experiments on three challenging benchmarks, i.e., COCO, CityScapes, and YouTube-VIS to evaluate the effectiveness of QueryInst in object detection, instance segmentation, and video instance segmentation tasks. For the first time, we demonstrate that a simple end-to-end query based framework can achieve the state-of-the-art performance in various instance-level recognition tasks.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/143971527-c1b7ff78-e95f-4edb-9d5e-3d6d7d902999.png"/>
</div>

## Results and Models

|   Model   | Backbone  |  Style  | Lr schd | Number of Proposals | Multi-Scale | RandomCrop | box AP | mask AP |                                 Config                                  |                                                                                                                                                                                                                       Download                                                                                                                                                                                                                       |
| :-------: | :-------: | :-----: | :-----: | :-----------------: | :---------: | :--------: | :----: | :-----: | :---------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| QueryInst | R-50-FPN  | pytorch |   1x    |         100         |    False    |   False    |  42.0  |  37.5   |                [config](./queryinst_r50_fpn_1x_coco.py)                 |                                                                         [model](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916-5a8f1998.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/queryinst/queryinst_r50_fpn_1x_coco/queryinst_r50_fpn_1x_coco_20210907_084916.log.json) |

## Citation

```latex
@InProceedings{Fang_2021_ICCV,
    author    = {Fang, Yuxin and Yang, Shusheng and Wang, Xinggang and Li, Yu and Fang, Chen and Shan, Ying and Feng, Bin and Liu, Wenyu},
    title     = {Instances As Queries},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {6910-6919}
}
```

