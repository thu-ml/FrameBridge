#!/bin/bash

export NAME="bridge-i2v-2b"
export MODEL_PATH="/path/to/downloaded/CogVideoX-2B-modified"
export CACHE_PATH="~/.cache"
export DATASET_PATH="/path/to/WebVid-2M/metadata/file/results_2M_train.csv"
export OUTPUT_PATH="experiments"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate_args="--num_processes 4"

accelerate launch --config_file accelerate_config_machine_single.yaml $accelerate_args --multi_gpu\
  train_cogvideox_image_to_video.py \
  --num_validation_videos 1 \
  --pretrained_model_name_or_path $MODEL_PATH \
  --cache_dir $CACHE_PATH \
  --enable_tiling \
  --enable_slicing \
  --instance_data_root $DATASET_PATH \
  --video_folder /path/to/WebVid-2M/video/folder/2M_train \
  --validation_epochs 100 \
  --seed 42 \
  --mixed_precision no \
  --output_dir $OUTPUT_PATH \
  --height 256 \
  --width 256 \
  --max_num_frames 17 \
  --train_batch_size 8 \
  --num_train_epochs 30 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --report_to tensorboard \
  --logging_dir logs/$NAME \
  --on_fly_loading \
  --noised_image_dropout 0.0 \
  --text_dropout 0.1 \
  --validation_steps 10 \
  --checkpointing_steps 1000 \
  --sft_training \
  --noisy_cond \
  --noisy_prior \
  --non_zero_channel_padding \
  --use_bridge \
  --concat_type "channel" \
  --use_tf32 \
  --resume_from_checkpoint "latest" \