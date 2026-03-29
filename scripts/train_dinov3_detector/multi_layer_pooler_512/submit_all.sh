#!/usr/bin/env bash

set -euo pipefail

sbatch "/private/home/francoisporcher/detrex/scripts/train_dinov3_detector/multi_layer_pooler_512/dino_vitdet_dinov3_multi_layer_pooler_16ep_512_w_layer_norm.sh"
sbatch "/private/home/francoisporcher/detrex/scripts/train_dinov3_detector/multi_layer_pooler_512/dino_vitdet_dinov3_multi_layer_pooler_16ep_512_wo_layer_norm.sh"
