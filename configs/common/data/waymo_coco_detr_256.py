from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
    DatasetCatalog,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances

from detrex.data import DetrDatasetMapper

img_size = 256
if "waymococo_train2020" not in DatasetCatalog.list():
    root = "/private/home/francoisporcher/data/waymococo_f0"
    register_coco_instances(
        "waymococo_train2020",
        {},
        f"{root}/annotations/instances_train2020.json",
        f"{root}/train2020",
    )
    register_coco_instances(
        "waymococo_val2020",
        {},
        f"{root}/annotations/instances_val2020.json",
        f"{root}/val2020",
    )

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="waymococo_train2020"),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.Resize)(shape=(img_size, img_size)),
        ],
        augmentation_with_crop=None,
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=2,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="waymococo_val2020", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.Resize)(shape=(img_size, img_size)),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=8,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)