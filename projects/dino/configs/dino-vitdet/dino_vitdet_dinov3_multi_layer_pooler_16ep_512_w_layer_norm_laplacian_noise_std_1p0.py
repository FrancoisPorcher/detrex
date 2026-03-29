from detrex.config import get_config
from ..models.dino_vitdet_dinov3_multi_layer_pooler_512_w_layer_norm_laplacian_noise_std_1p0 import model

# get default config
dataloader = get_config("common/data/waymo_coco_detr_512.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_16ep
train = get_config("common/train.py").train


# modify training config
train.init_checkpoint = ""
train.output_dir = "./experiments/train_dinov3_detector/dino_vitdet_dinov3_multi_layer_pooler_16ep_512_w_layer_norm_laplacian_noise_std_1p0"
train.wandb.params.name = "dino_vitdet_dinov3_multi_layer_pooler_16ep_512_w_layer_norm_laplacian_noise_std_1p0"
model.img_size = 512
model.vis_period = 500

# max training iterations
train.max_iter = 120000

# run evaluation every 5000 iters
train.eval_period = 5000

# log training information every 10 iters
train.log_period = 10

# save checkpoint every 5000 iters
train.checkpointer.period = 5000

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 16

# please notice that this is total batch size.
# adjust your per-GPU batch size accordingly based on the number of devices.
dataloader.train.total_batch_size = 16

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
