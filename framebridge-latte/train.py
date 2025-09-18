# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Latte using PyTorch DDP.
"""


import torch
import gc
# Maybe use fp16 percision training need to set to False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import io
import os
import math
import argparse

import torch.distributed as dist
from glob import glob
from time import time
from copy import deepcopy
from einops import rearrange
from models import get_models
from datasets import get_dataset
from models.clip import TextEmbedder
from diffusion import create_diffusion
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import (clip_grad_norm_, create_logger, update_ema, 
                   requires_grad, cleanup, create_tensorboard, 
                   write_tensorboard, setup_distributed,
                   get_experiment_dir, text_preprocessing)
import numpy as np
from transformers import T5EncoderModel, T5Tokenizer
import cv2
import imageio
from sample.sample import build_nwarp_model, nwarp_sampling
from video_utils import save_images, save_videos, dct_preprocess, dct_postprocess

#################################################################################
#                                  Training Loop                                #
#################################################################################

from download import find_model

def main(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    setup_distributed()
    # dist.init_process_group("nccl")
    # assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    # rank = dist.get_rank()
    # device = rank % torch.cuda.device_count()
    # local_rank = rank
    
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    seed = args.global_seed + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, local rank={local_rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., Latte-XL/2 --> Latte-XL-2 (for naming folders)
        num_frame_string = 'F' + str(args.num_frames) + 'S' + str(args.frame_interval)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}-{num_frame_string}-{args.dataset}"  # Create an experiment folder
        experiment_dir = get_experiment_dir(experiment_dir, args)
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        tb_writer = create_tensorboard(experiment_dir)
        OmegaConf.save(args, os.path.join(experiment_dir, 'config.yaml'))
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        tb_writer = None

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    sample_size = args.image_size // 8
    args.latent_size = sample_size
        
    model = get_models(args)
    # Note that parameter initialization is done within the Latte constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
        
    diffusion_kwargs = dict(timestep_respacing="",
                            learn_sigma=args.learn_sigma,
                            model_stage=args.model_stage,
                            model_predict_type=args.model_predict_type,
                            prior_encoder_type=args.prior_encoder_type,
                            condition_type=args.condition_type,
                            nwarp_supervision_type=args.nwarp_supervision_type,
                            nwarp_combine_lambda=args.nwarp_combine_lambda,
                            nwarp_condition_type=args.nwarp_condition_type,
                            dct_domain=args.dct_domain,
                            dct_coef_num=args.dct_coef_num,
                            snr_alignment=args.finetune_snr_alignment)
    diffusion = create_diffusion(**diffusion_kwargs)  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)

    # load possible neural warping network
    if args.nwarp_config is not None:
        nwarp_config = OmegaConf.load(args.nwarp_config)
        nwarp_config.ckpt = args.nwarp_ckpt
        nwarp_model = build_nwarp_model(nwarp_config)
    
    # # use pretrained model?
    if args.pretrained:
        checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            logger.info('Using ema ckpt!')
            checkpoint = checkpoint["ema"]

        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {}
        for k, v in checkpoint.items():
            if k in model_dict:
                if "final" in k and v.shape[0] == 2 * model.patch_size * model.patch_size * model.out_channels:
                    index = []
                    for i in range(model.patch_size * model.patch_size):
                        for j in range(model.out_channels):
                            index.append(i * model.out_channels * 2 + j)
                    v = v[index, ...]
                if k == "temp_embed" and v.shape[1] != model_dict[k].shape[1]:
                    v = torch.concat((v[:,0:1,:].repeat(1, model_dict[k].shape[1] - v.shape[1], 1), v), dim=1)
                pretrained_dict[k] = v
            else:
                logger.info('Ignoring: {}'.format(k))
        logger.info('Successfully Load {}% original pretrained model weights '.format(len(pretrained_dict) / len(checkpoint.items()) * 100))
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info('Successfully load model at {}!'.format(args.pretrained))

    if args.use_compile:
        model = torch.compile(model)

    if args.enable_xformers_memory_efficient_attention:
        logger.info("Using Xformers!")
        model.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        logger.info("Using gradient checkpointing!")
        model.enable_gradient_checkpointing()

    # set distributed training
    model = DDP(model.to(device), device_ids=[local_rank])

    cnt_params = 0
    for p in model.parameters():
        if p.requires_grad:
            cnt_params += p.numel()
    logger.info(f"Model Parameters: {cnt_params:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0)

    # Freeze vae and text_encoder
    vae.requires_grad_(False)

    # Setup data:
    dataset = get_dataset(args)

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.local_batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} videos ({args.data_path})")

    # Scheduler
    lr_scheduler = get_scheduler(
        name="constant",
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    running_loss_regression = 0
    running_loss_denoising = 0
    first_epoch = 0
    start_time = time()
    loss_accumulation = 0.

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(loader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        # TODO, need to checkout
        # Get the most recent checkpoint
        '''
        dirs = os.listdir(os.path.join(experiment_dir, 'checkpoints'))
        dirs = [d for d in dirs if d.endswith("pt")]
        dirs = sorted(dirs, key=lambda x: int(x.split(".")[0]))
        path = dirs[-1]
        '''
        path = args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint {path}")
        model.load_state(path)

        '''
        first_epoch = train_steps // num_update_steps_per_epoch
        resume_step = train_steps % num_update_steps_per_epoch
        '''

    '''
    if args.pretrained:
        train_steps = int(args.pretrained.split("/")[-1].split('.')[0])
    '''
    
    # Directly generate DCT coefficients
    if args.dct_domain and args.use_quantile:
        dct_quantile = torch.load(args.dct_quantile).to(device)[:, :args.dct_coef_num]
        B, H, W = args.local_batch_size, sample_size, sample_size
        data_quantile = dct_quantile.reshape(1, 1, 1, dct_quantile.shape[0], dct_quantile.shape[1])
        data_quantile = rearrange(data_quantile.repeat(B, H, W, 1, 1), 'b h w c f -> b f c h w').contiguous()
    else:
        data_quantile = None

    current_steps = 0
    for epoch in range(first_epoch, num_train_epochs):
        sampler.set_epoch(epoch)
        for step, video_data in enumerate(loader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                continue

            x = video_data['video'].to(device, non_blocking=True)
            
            condition_frame = None
            if args.prior_encoder_type is not None and "pixel" in args.prior_encoder_type:
                if args.prior_encoder_type == "mean_pixel":
                    x_mean = torch.mean(x, dim=1)
                    x_mean = vae.encode(x_mean).latent_dist.sample().mul_(0.18215)
                    condition_frame = x_mean.unsqueeze(1)
                else:
                    assert args.prior_encoder_type == "first_pixel"
                    condition_frame = x[:,0,:,:,:].unsqueeze(1)
                    condition_frame = rearrange(condition_frame, 'b f c h w -> (b f) c h w').contiguous()
                    condition_frame = vae.encode(condition_frame).latent_dist.sample().mul_(0.18215)
                    condition_frame = condition_frame.unsqueeze(1)
                
            video_name = video_data['video_name']
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                b, _, _, _, _ = x.shape
                x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()
            
            if args.dct_domain:
                x = dct_preprocess(x, condition_frame, data_quantile, args.dct_coef_num, use_quantile=args.use_quantile)

            # Training model to generate condition of bridge (i.e. coarse video)
            if args.model_stage == "nwarp" and args.dct_domain and args.dct_coef_num <= 2:
                x_nwarp = condition_frame.squeeze(1)
            elif args.model_stage == "nwarp":
                x_nwarp = condition_frame.repeat(1, x.shape[1], 1, 1, 1) # input for nwarp model
            else:
                x_nwarp = None
            
            # generate results of neural warping stage
            with torch.no_grad():
                if args.nwarp_config is not None:
                    nwarp_input = x[:, 0, :, :, :].unsqueeze(1).repeat(1, x.shape[1], 1, 1, 1)
                    coarse_video = nwarp_sampling(nwarp_input, video_name, nwarp_config, nwarp_model, data_quantile=data_quantile).to(device).float()
                    if args.dct_domain:
                        coarse_video = dct_preprocess(coarse_video, condition_frame, data_quantile, args.dct_coef_num, use_quantile=args.use_quantile)
                else:
                    coarse_video = None

            if args.extras == 78: # text-to-video
                raise 'T2V training are Not supported at this moment!'
            elif args.extras == 2:
                model_kwargs = dict(y=video_name)
            else:
                model_kwargs = dict(y=None)
            
            if args.condition_type in ["cross_attention", "adaln"]:
                model_kwargs["condition_frame"] = condition_frame[:,0,:,:]
            
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, condition_frame, coarse_video, x_nwarp, args.finetune_snr_alignment, model_kwargs)
            loss = loss_dict["loss"].mean() / args.gradient_accumulation_steps
            loss.backward()
            loss_accumulation += loss

            current_steps += 1
            if current_steps % args.gradient_accumulation_steps == 0:
                if train_steps < args.start_clip_iter: # if train_steps >= start_clip_iter, will clip gradient
                    gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=False)
                else:
                    gradient_norm = clip_grad_norm_(model.module.parameters(), args.clip_max_norm, clip_grad=True)

                opt.step()
                lr_scheduler.step()
                opt.zero_grad()
                update_ema(ema, model.module)

                log_steps += 1
                train_steps += 1

                # Log loss values:
                running_loss += loss_accumulation.detach().item()
                loss_accumulation = 0.
                if args.model_stage == "nwarp" and args.nwarp_supervision_type == "combination":
                    running_loss_regression += loss_dict["loss_regression"].mean().item()
                    running_loss_denoising += loss_dict["loss_denoising"].mean().item()
            
                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    if args.model_stage == "nwarp" and args.nwarp_supervision_type == "combination":
                        avg_loss_regression = torch.tensor(running_loss_regression / log_steps, device=device)
                        avg_loss_denoising = torch.tensor(running_loss_denoising / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    # logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    write_tensorboard(tb_writer, 'Train Loss', avg_loss, train_steps)
                    write_tensorboard(tb_writer, 'Gradient Norm', gradient_norm, train_steps)
                    # Reset monitoring variables:
                    running_loss = 0
                    if args.model_stage == "nwarp" and args.nwarp_supervision_type == "combination":
                        write_tensorboard(tb_writer, 'Regression Loss', avg_loss_regression, train_steps)
                        write_tensorboard(tb_writer, 'Denoising Loss', avg_loss_denoising, train_steps)
                        running_loss_regression = 0
                        running_loss_denoising = 0
                        logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Regression Loss: {avg_loss_regression:.4f}, Denoising Loss: {avg_loss_denoising:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    else:
                        logger.info(f"(step={train_steps:07d}/epoch={epoch:04d}) Train Loss: {avg_loss:.4f}, Gradient Norm: {gradient_norm:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")                  
                    log_steps = 0
                    start_time = time()

                # Save Latte checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            # "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            # "opt": opt.state_dict(),
                            # "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train Latte with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train.yaml")
    args = parser.parse_args()
    main(OmegaConf.load(args.config))
