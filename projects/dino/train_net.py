#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
from datetime import timedelta
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.events import (
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.utils.file_io import PathManager

from detrex.utils import WandbWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger("detrex")


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
            self.grad_scaler = grad_scaler

        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        with autocast(enabled=self.amp):
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )


def do_test(cfg, 
            model, 
            df_evaluation_metadata=None, 
            last_epoch_dir=None,
            cfg_evaluation=None,
            inference_from_latents=False):

    if inference_from_latents:
        import os
        import torch
        import torch.nn.functional as F
        import torchvision.transforms.functional as tvF
        from einops import rearrange
        from torch.utils.data import Dataset
        import pandas as pd

        num_samples_for_evaluation = 1000
        class WaymoVideoFrameDataset(Dataset):
            def __init__(self, df_metadata: pd.DataFrame, cfg, last_epoch_dir: str, num_samples_for_evaluation = None):
                self.df_metadata = df_metadata.reset_index(drop=True)
                self.cfg = cfg
                self.experiment_dir = cfg.experiment_dir
                self.waymo_root_dir = cfg.waymo_root_dir
                self.last_epoch_dir = last_epoch_dir
                self.num_samples_for_evaluation = num_samples_for_evaluation

                # Build index: each entry is (video_row_idx, frame_idx)
                # Waymo is a video dataset. The simplest way to make it compatible with COCO is to flatten the double indexing (video idx, frame idx) into a single index
                self.frame_index = []
                for row_idx, row in self.df_metadata.iterrows():
                    # Parse the image_ids list (stored as string)
                    image_ids = list(map(int, row["image_ids"].strip("[]").split(",")))
                    num_frames = len(image_ids)
                    for t in range(num_frames):
                        self.frame_index.append((row_idx, t))
                        
                # For quick evaluation, we can limit the number of samples
                if num_samples_for_evaluation is not None:
                    self.frame_index = self.frame_index[:num_samples_for_evaluation]

                self.image_ids_subset = []
                # Build image_ids_subset from the selected frames
                for (video_row_idx, frame_idx) in self.frame_index:
                    row = self.df_metadata.iloc[video_row_idx]
                    image_ids = list(map(int, row["image_ids"].strip("[]").split(",")))
                    self.image_ids_subset.append(int(image_ids[frame_idx]))

            def __len__(self):
                return len(self.frame_index)

            def __getitem__(self, idx):
                video_row_idx, frame_idx = self.frame_index[idx]
                row = self.df_metadata.iloc[video_row_idx]
                video_file_name = row["video_filename"]

                pth_video_tensor_pt = os.path.join(self.last_epoch_dir, video_file_name)
                video_tensor = torch.load(pth_video_tensor_pt)  
                frame = video_tensor[:,frame_idx,:,:]        # (D, H, W)

                # Parse image_ids again (could be cached, but fine for now)
                image_ids = list(map(int, row["image_ids"].strip("[]").split(",")))
                image_id = image_ids[frame_idx]
                
                original_height = list(map(int, row["original_height"].strip("[]").split(",")))            
                height = original_height[frame_idx]
                original_width = list(map(int, row["original_width"].strip("[]").split(",")))            
                width = original_width[frame_idx]
                # Optional: store some extra info
                metadata = row.to_dict()
                

                sample = {
                    "file_name": video_file_name,  # not strictly used if "image" is present
                    "image_id": int(image_id),
                    "image": frame,               # <- what the detector expects
                    "video_row_idx": int(video_row_idx),
                    "frame_idx": int(frame_idx),
                    "metadata": metadata,
                    "height": height,
                    "width": width,
                }
                return sample

        # import the waymo_dataset evaluator
        dataset_waymo_video = WaymoVideoFrameDataset(df_metadata = df_evaluation_metadata, 
                                                     cfg=cfg_evaluation,
                                                     last_epoch_dir=last_epoch_dir,
                                                     num_samples_for_evaluation=num_samples_for_evaluation)

        
        from detrex.data import DetrDatasetMapper

        import torch.utils.data as torchdata
        from detectron2.data import MapDataset
        from detectron2.data.samplers import InferenceSampler

        def trivial_batch_collator(batch):
            # Detectron2's version is just this: return the list as-is
            return batch

        def build_waymo_video_test_loader(
            dataset,
            mapper=None,
            num_workers: int = 0,
            batch_size: int = 1,
        ):
            """
            Minimal Detectron2-compatible test DataLoader for WaymoVideoFrameDataset.
            Returns batches as List[Dict], like the standard build_detection_test_loader.
            """
            if mapper is not None:
                dataset = MapDataset(dataset, mapper)

            sampler = InferenceSampler(len(dataset))

            data_loader = torchdata.DataLoader(
                dataset,
                batch_size=batch_size,          # 1 per GPU for eval
                sampler=sampler,
                drop_last=False,
                num_workers=num_workers,
                collate_fn=trivial_batch_collator,
            )
            return data_loader


        dataloader_waymo_videoframe = build_waymo_video_test_loader(
            dataset=dataset_waymo_video,
            mapper=None,
            num_workers=1,
            batch_size=1
            )

    if inference_from_latents:
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model = model,
                data_loader = dataloader_waymo_videoframe,
                evaluator = instantiate(cfg.dataloader.evaluator), 
                inference_from_backbone=True)
            print_csv_format(ret)
            return ret

    else:
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, 
                data_loader = instantiate(cfg.dataloader.test), 
                evaluator = instantiate(cfg.dataloader.evaluator), 
                inference_from_backbone=False,
                img_ids = None
            )
            print_csv_format(ret)
            return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    # this is an hack of train_net
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not match_name_keywords(n, ["backbone"])
                and not match_name_keywords(n, ["reference_points", "sampling_offsets"])
                and p.requires_grad
            ],
            "lr": 2e-4,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, ["backbone"]) and p.requires_grad
            ],
            "lr": 2e-5,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if match_name_keywords(n, ["reference_points", "sampling_offsets"])
                and p.requires_grad
            ],
            "lr": 2e-5,
        },
    ]
    optim = torch.optim.AdamW(param_dicts, 2e-4, weight_decay=1e-4)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
    )

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    if comm.is_main_process():
        output_dir = cfg.train.output_dir
        PathManager.mkdirs(output_dir)
        writers = [
            CommonMetricPrinter(cfg.train.max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]
        wandb_cfg = getattr(cfg.train, "wandb", None)
        # print the wandb name of the experiment
        if wandb_cfg is not None and wandb_cfg.enabled:
            print("Wandb experiment name:", wandb_cfg.params.name) 
        if wandb_cfg and wandb_cfg.enabled:
            PathManager.mkdirs(wandb_cfg.params.dir)
            writers.append(WandbWriter(cfg))

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                writers,
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )
    

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.debug:
        cfg.train.max_iter = 10
        cfg.train.eval_period = 10
        cfg.dataloader.train.total_batch_size = 2
        cfg.dataloader.test.batch_size = 2

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        

        # Evaluation part added
        cfg_evaluation = cfg.cfg_evaluation
        experiment_dir = cfg_evaluation['experiment_dir']
        waymo_root_dir = cfg_evaluation['waymo_root_dir']
        
        
        # Get video tensor dataframe
        dir_df_metadata_validation = os.path.join(waymo_root_dir, "validation_video_tensors", "df_metadata.csv")
        import pandas as pd
        df_metadata_validation = pd.read_csv(dir_df_metadata_validation)

        # get the predictions
        evaluation_dir = os.path.join(experiment_dir, "evaluation")
        last_epoch_folder_name = sorted(os.listdir(evaluation_dir))[-1]
        last_epoch_dir = os.path.join(evaluation_dir, last_epoch_folder_name)

        print(do_test(cfg, 
                      model, 
                      df_evaluation_metadata = df_metadata_validation,
                      last_epoch_dir = last_epoch_dir,
                      cfg_evaluation = cfg_evaluation,
                      inference_from_latents = args.inference_from_latents
                      ))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    # print which python executable we are using
    import sys
    print("Python executable:", sys.executable)
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="If set, use debug mode with fewer iterations.")
    args = parser.parse_args()

    # print args in a formatted way with for loop
    for k, v in sorted(vars(args).items()):
        print(f"{k}: {v}")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        timeout=timedelta(hours=2),
        args=(args,),
    )
