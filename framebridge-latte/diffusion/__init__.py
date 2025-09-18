# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear", 
    use_kl=False,
    sigma_small=False,
    model_predict_type="eps",
    learn_sigma=True,
    # learn_sigma=False,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    model_stage="diffusion",
    condition_type=None,
    prior_encoder_type=None,
    nwarp_supervision_type=None,
    nwarp_combine_lambda=None,
    nwarp_condition_type=None,
    dct_domain=False,
    dct_coef_num=16,
    snr_alignment=False,
    prepare_bridge_ode_sampler=False,
    original_aligned_t=None
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    
    if model_stage == "bridge":
        if model_predict_type == "eps_psi":
            model_mean_type = gd.ModelMeanType.BRIDGE_EPSILON_PSI
        else:
            raise NotImplementedError(model_mean_type)
    else:
        if model_predict_type == "eps":
            model_mean_type = gd.ModelMeanType.EPSILON
        else:
            model_mean_type = gd.ModelMeanType.START_X
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        model_stage=model_stage,
        condition_type=condition_type,
        prior_encoder_type=prior_encoder_type,
        nwarp_supervision_type=nwarp_supervision_type,
        nwarp_combine_lambda=nwarp_combine_lambda,
        nwarp_condition_type=nwarp_condition_type,
        dct_domain=dct_domain,
        dct_coef_num=dct_coef_num,
        snr_alignment=snr_alignment,
        prepare_bridge_ode_sampler=prepare_bridge_ode_sampler,
        original_aligned_t=original_aligned_t
        # rescale_timesteps=rescale_timesteps,
    )
