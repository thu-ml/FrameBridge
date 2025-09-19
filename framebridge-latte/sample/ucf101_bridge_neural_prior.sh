#!/bin/bash
torchrun --nnodes=1 --nproc_per_node=1 --master_port=29520 sample/sample.py \
--config ./configs/ucf101/ucf101_sample_bridge_nwarp.yaml \
--ckpt /path/to/FrameBridge/with/neural/prior/checkpoint \
--save_video_path ./sample_videos