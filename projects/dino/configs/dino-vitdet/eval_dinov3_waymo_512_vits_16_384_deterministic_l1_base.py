from detrex.config import get_config
from ..models.dino_vitdet_dinov3_multi_layer_pooler_512_wo_layer_norm import model

# get default config
dataloader = get_config("common/data/waymo_coco_detr_512.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_config("common/train.py").train


# modify training config
train.init_checkpoint = ""
train.output_dir = "./experiments/eval_dinov3_detector/dino_vitdet_dinov3_multi_layer_pooler_12ep_512"
train.wandb.params.name = "dino_vitdet_dinov3_multi_layer_pooler_12ep_512"

model.img_size = 512
model.vis_period = 500

# max training iterations
train.max_iter = 90000

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
dataloader.train.total_batch_size = 1

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

cfg_evaluation = dict(
    experiment_dir = "/private/home/francoisporcher/FutureLatents2/experiments/dinov3/deterministic/compile/384/dinov3_waymo_512_vits_16_384_deterministic_l1_base_compile_enabled",
    waymo_root_dir = "/private/home/francoisporcher/data/waymococo_f0/",
)
