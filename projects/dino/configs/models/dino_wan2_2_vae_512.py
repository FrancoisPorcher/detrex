from functools import partial
import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling import WanVAE, SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .dino_r50 import model

# WAN 2.1 VAE Hyper-params
embed_dim = 16
vae_pth = "/private/home/francoisporcher/Wan2.2/Wan2.2-I2V-A14B/Wan2.1_VAE.pth"
model.model_name = "wan2.1-vae"
model.use_dinov3_backbone = False
model.num_classes = 3
model.img_size = 512
model.pixel_mean = [127.5, 127.5, 127.5]
model.pixel_std = [127.5, 127.5, 127.5]

# Creates Simple Feature Pyramid from ViT backbone
model.backbone = L(SimpleFeaturePyramid)(
    net=L(WanVAE)(  # Single-scale VAE backbone
        patch_size=8,
        z_dim=embed_dim,
        out_feature="last_feat",
        vae_pth=vae_pth,
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(1.0, 0.5, 0.25),  # (4.0, 2.0, 1.0, 0.5) in ViTDet
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
