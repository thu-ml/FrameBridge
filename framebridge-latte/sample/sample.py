# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained Latte.
"""
import os
import sys
import random
try:
    import utils

    from diffusion import create_diffusion
    from download import find_model
except:
    sys.path.append(os.path.split(sys.path[0])[0])

    import utils

    from diffusion import create_diffusion
    from download import find_model

import torch
import argparse
import torchvision

from einops import rearrange
from models import get_models
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from models.clip import TextEmbedder
import imageio
from omegaconf import OmegaConf
from datasets import get_dataset
from torch.utils.data import DataLoader
import cv2
import torch
from video_utils import save_videos, dct_2d_compress, dct_1d_compress, dct_postprocess, dct_preprocess

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def build_nwarp_model(args):
    # Load model:
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    model = get_models(args).to(device)

    if args.use_compile:
        model = torch.compile(model)
    if args.use_fp16:
        model.to(dtype=torch.float16)

    # a pre-trained model or load a custom Latte checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=False) # only preserve the nwarp model
    model.eval()
    return model

def nwarp_sampling(z, label, args, model, data_quantile=None):
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    z = z.to(device)
    if args.use_fp16:
        z = z.half()
    coarse_video = model.nwarp_model(z, None, torch.tensor(label, device=device), None, args.use_fp16)
    if args.dct_domain:
        coarse_video = dct_postprocess(coarse_video, z, data_quantile, args.num_frames, use_quantile=args.use_quantile).float()
    return coarse_video

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    print("configs: ", args)
    # device = "cpu"

    if args.ckpt is None:
        assert args.model == "Latte-XL/2", "Only Latte-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    using_cfg = (args.cfg_scale >= 1.0) and (args.model_stage != "nwarp")

    # Load model:
    latent_size = args.image_size // 8
    args.latent_size = latent_size
    model = get_models(args).to(device)

    if args.use_compile:
        model = torch.compile(model)

    # a pre-trained model or load a custom Latte checkpoint from train.py:
    ckpt_path = args.ckpt
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict, strict=True)

    model.eval()  # important!
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
                            snr_alignment=args.finetune_snr_alignment,
                            prepare_bridge_ode_sampler=False)
    original_diffusion = create_diffusion(**diffusion_kwargs)  # default: 1000 steps, linear noise schedule

    diffusion_kwargs = dict(timestep_respacing=str(args.num_sampling_steps),
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
                            snr_alignment=args.finetune_snr_alignment,
                            prepare_bridge_ode_sampler=(args.sample_method == "bridge_ode"),
                            original_aligned_t=original_diffusion.bridge_aligned_t if args.finetune_snr_alignment else None)
    diffusion = create_diffusion(**diffusion_kwargs)  # default: 1000 steps, linear noise schedule
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae").to(device)
    # text_encoder = TextEmbedder().to(device)

    if args.use_fp16:
        print('WARNING: using half percision for inferencing!')
        vae.to(dtype=torch.float16)
        model.to(dtype=torch.float16)
        # text_encoder.to(dtype=torch.float16)
        
    if args.nwarp_config is not None:
        nwarp_config = OmegaConf.load(args.nwarp_config)
        nwarp_config.ckpt = args.nwarp_ckpt
        nwarp_model = build_nwarp_model(nwarp_config)

    if args.dct_domain:
        if args.dct_quantile is None:
            data_quantile = None
        else:
            dct_quantile = torch.load(args.dct_quantile).to(device)[:, :args.dct_coef_num]
            B, H, W = 1, latent_size, latent_size
            data_quantile = dct_quantile.reshape(1, 1, 1, dct_quantile.shape[0], dct_quantile.shape[1])
            data_quantile = rearrange(data_quantile.repeat(B, H, W, 1, 1), 'b h w c f -> b f c h w').contiguous()

    # Labels to condition the model with (feel free to change):

    # Create sampling noise:
    label = None
    if ("prior_encoder_type" in args) and (args.prior_encoder_type is not None):
        if args.sample_prior == "ground_truth":
            dataset = get_dataset(args)
            loader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                pin_memory=True,
                drop_last=True
            )
            for _, sample in enumerate(loader):
                video = sample["video"]
                save_videos("sample_videos/ground_truth", video)
                
                label = sample["video_name"]
                print(torch.tensor(label), dataset.classes[label])
                x = video.to(device)
                if args.use_fp16:
                    x = x.half()
                b = x.shape[0]
                
                # image condition
                if args.prior_encoder_type == "first_pixel":
                    x = x[:,0,:,:,:].unsqueeze(1)
                    x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                elif args.prior_encoder_type == "mean_latent":
                    x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()
                    x = torch.mean(x, dim=1)
                elif args.prior_encoder_type == "mean_pixel":
                    x = torch.mean(x, dim=1).unsqueeze(1).to(device)
                    x = rearrange(x, 'b f c h w -> (b f) c h w').contiguous()
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                else:
                    raise NotImplementedError(args.prior_encoder_type)
                
                prior = vae.decode(x / 0.18215).sample
                prior_save_path = os.path.join(args.save_video_path, 'prior' + '.png')
                img = prior[0] * 0.5 + 0.5
                # img = ((img * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu()
                save_image(img, prior_save_path)
                
                x = rearrange(x, '(b f) c h w -> b f c h w', b=b).contiguous()
                frame_cond = x.repeat(1, args.num_frames, 1, 1, 1)
                
                # neural warp condition
                if args.use_pretrained_nwarp:
                    nwarp_input = frame_cond
                    coarse_video = nwarp_sampling(nwarp_input, label, nwarp_config, nwarp_model).to(device)
                    save_videos("sample_videos/coarse_video", vae.decode(coarse_video[0] / 0.18215).sample.unsqueeze(0))
                    if args.dct_domain:
                        coarse_video = dct_preprocess(coarse_video, frame_cond, data_quantile, args.dct_coef_num, use_quantile=args.use_quantile)
                else:
                    coarse_video = None
                break
        else:
            NotImplementedError(args.sample_prior)
    else:
        frame_cond = None
        coarse_video = None
    
    prior_num_frames = args.dct_coef_num if args.dct_domain else args.num_frames
    if args.use_fp16:
        z = torch.randn(1, prior_num_frames, 4, latent_size, latent_size, dtype=torch.float16, device=device) # b c f h w
    else:
        z = torch.randn(1, prior_num_frames, 4, latent_size, latent_size, device=device)
    
    if args.model_stage == "bridge":
        if args.nwarp_condition_type == "concat_only" or args.use_pretrained_nwarp == False:
            z = frame_cond
        else:
            assert args.nwarp_condition_type == "prior"
            z = coarse_video

    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    if using_cfg:
        z = torch.cat([z, z], 0)
        y = torch.randint(0, args.num_classes, (1,), device=device)
        if label is not None:
            y = torch.tensor(label, device=device)
        y_null = torch.tensor([101] * 1, device=device)
        y = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale, use_fp16=args.use_fp16)
        sample_fn = model.forward_with_cfg
        
        if frame_cond is not None:
            frame_cond = torch.cat([frame_cond, frame_cond], 0)
        if coarse_video is not None:
            coarse_video = torch.cat([coarse_video, coarse_video], 0)
    else:
        sample_fn = model.forward
        model_kwargs = dict(y=None, use_fp16=args.use_fp16)
    if frame_cond is not None:
        model_kwargs["condition_frame"] = frame_cond[:, 0, :, :, :]

    # Sample images:
    if args.model_stage == "nwarp":
        # input: first frame x L
        if args.model_stage == "nwarp" and args.dct_domain and args.dct_coef_num <= 2:
            x_nwarp = frame_cond[:,0,:,:,:]
        elif args.model_stage == "nwarp":
            x_nwarp = frame_cond[:,:args.dct_coef_num,:,:,:]# input for nwarp model
        else:
            x_nwarp = frame_cond
        coarse_video = model.nwarp_model(x_nwarp, None, torch.tensor(label, device=device), None, args.use_fp16)
        samples = coarse_video
    else:
        if args.sample_method == 'ddim':
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
        elif args.sample_method == 'ddpm':
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, frame_cond=frame_cond, coarse_video=coarse_video
            )
        elif args.sample_method == "bridge_ode":
            samples = diffusion.bridge_ode_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, frame_cond=frame_cond, coarse_video=coarse_video, snr_alignment=args.finetune_snr_alignment, order=args.bridge_ode_order
            )
        elif args.sample_method == "bridge_sde":
            samples = diffusion.bridge_sde_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, frame_cond=frame_cond, coarse_video=coarse_video, snr_alignment=args.finetune_snr_alignment
            )

    if args.dct_domain:
        if args.dct_coef_num <= 2:
            samples = torch.stack(torch.chunk(samples, args.dct_coef_num, dim=1), dim=1)
        samples = dct_postprocess(samples, frame_cond, data_quantile, args.num_frames, use_quantile=args.use_quantile).float()
        
    if args.use_fp16:
        samples = samples.to(dtype=torch.float16)
    b, f, c, h, w = samples.shape
    samples = rearrange(samples, 'b f c h w -> (b f) c h w')
    samples = vae.decode(samples / 0.18215).sample
    samples = rearrange(samples, '(b f) c h w -> b f c h w', b=b)
    # Save and display images:

    if not os.path.exists(args.save_video_path):
        os.makedirs(args.save_video_path)

    video_ = ((samples[0] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
    video_save_path = os.path.join(args.save_video_path, 'sample' + '.mp4')
    print(video_save_path)
    imageio.mimwrite(video_save_path, video_, fps=8, quality=10)
    original_tensor = video_
    vframes, aframes, info = torchvision.io.read_video(filename=video_save_path, pts_unit='sec', output_format='TCHW')
    print('save path {}'.format(args.save_video_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/ucf101/ucf101_sample.yaml")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--save_video_path", type=str, default="./sample_videos/")
    args = parser.parse_args()
    omega_conf = OmegaConf.load(args.config)
    omega_conf.ckpt = args.ckpt
    omega_conf.save_video_path = args.save_video_path
    main(omega_conf)
