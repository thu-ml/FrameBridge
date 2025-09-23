
version=256
seed=0
H=256
FS=3
name=sample

ckpt=/path/to/checkpoint
config=configs/inference_256_bridge.yaml

prompt_dir=/path/to/prompts
res_dir="results/"

python3 -m torch.distributed.launch \
scripts/evaluation/ddp_wrapper.py \
--module 'inference' \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name \
--n_samples 1 \
--bs 1 --height ${H} --width 256 \
--unconditional_guidance_scale 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 16 \
--frame_stride ${FS} \
--timestep_spacing 'uniform_trailing' \
--bridge_sampler 'bridge_sde' \