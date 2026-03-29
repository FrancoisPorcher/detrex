from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling import DINOv3Backbone, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .dino_r50 import model

# ViT Base Hyper-params
embed_dim = 384
model.model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
model.use_dinov3_backbone = True
model.num_classes = 3
model.img_size = 512

# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(DINOv3Backbone)(  # Single-scale ViT backbone
        patch_size=16,
        embed_dim=embed_dim,
        out_feature="last_feat",
        model_name=model.model_name,
        return_all_layers=True,
        use_last_layer_ln=True,
        add_noise_latents_gaussian=True,
        add_noise_latents_laplacian=False,
        noise_latents_std=2.0,
        dinov3_dropout_rate=0.0,
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(2.0, 1.0, 0.5),  # (4.0, 2.0, 1.0, 0.5) in ViTDet
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=512,
)

# modify neck config
model.neck.input_shapes = {
    "p3": ShapeSpec(channels=256),
    "p4": ShapeSpec(channels=256),
    "p5": ShapeSpec(channels=256),
    "p6": ShapeSpec(channels=256),
}
model.neck.in_features = ["p3", "p4", "p5", "p6"]
model.neck.num_outs = 4
model.transformer.num_feature_levels = 4
