#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=8 --master_port=29509 train.py --config ./configs/ucf101/ucf101_train_nwarp.yaml