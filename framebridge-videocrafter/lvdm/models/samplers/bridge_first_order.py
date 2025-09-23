import numpy as np
from tqdm import tqdm
import torch
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps, rescale_noise_cfg
from lvdm.common import noise_like
from lvdm.common import extract_into_tensor
import copy


class BridgeFirstOrderSampler(object):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.model_num_timesteps = model.num_timesteps

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.sample_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.model_num_timesteps,verbose=verbose)
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        if self.model.use_dynamic_rescale:
            self.sample_scale_arr = self.model.scale_arr[self.sample_timesteps]
            self.sample_scale_arr_prev = torch.cat([self.sample_scale_arr[0:1], self.sample_scale_arr[:-1]])

        self.register_buffer('sampler_bridge_coef_x0', to_torch(self.model.bridge_coef_x0[self.sample_timesteps]))
        self.register_buffer('sampler_bridge_coef_x1', to_torch(self.model.bridge_coef_x1[self.sample_timesteps]))
        self.register_buffer('sampler_bridge_sqrt_var', to_torch(self.model.bridge_sqrt_var[self.sample_timesteps]))
        self.register_buffer('sampler_bridge_alphas', to_torch(self.model.bridge_alphas[self.sample_timesteps]))
        self.register_buffer('sampler_bridge_sigmas', to_torch(self.model.bridge_sigmas[self.sample_timesteps]))

        self.register_buffer('sampler_bridge_aligned_t', to_torch(self.model.bridge_aligned_t[self.sample_timesteps]))
        if self.model.use_dynamic_rescale:
            self.register_buffer('sampler_original_diffusion_rescale', to_torch(self.model.original_diffusion_rescale[self.sample_timesteps]))

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               verbose=True,
               schedule_verbose=False,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               precision=None,
               fs=None,
               timestep_spacing='uniform', #uniform_trailing for starting from last timestep,
               intermediate_results=False,
               **kwargs
               ):
        
        # check condition bs
        if conditioning is not None:
            if isinstance(conditioning, dict):
                try:
                    cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                except:
                    cbs = conditioning[list(conditioning.keys())[0]][0].shape[0]

                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_discretize=timestep_spacing, verbose=schedule_verbose)
        
        # make shape
        if len(shape) == 3:
            C, H, W = shape
            size = (batch_size, C, H, W)
        elif len(shape) == 4:
            C, T, H, W = shape
            size = (batch_size, C, T, H, W)

        samples, intermediates = self.bridge_first_order_sampling(conditioning, size,
                                                                x_T=x_T,
                                                                log_every_t=log_every_t,
                                                                unconditional_guidance_scale=unconditional_guidance_scale,
                                                                unconditional_conditioning=unconditional_conditioning,
                                                                verbose=verbose,
                                                                precision=precision,
                                                                fs=fs,
                                                                **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def bridge_first_order_sampling(self, cond, shape,
                                x_T=None,
                                log_every_t=100,
                                unconditional_guidance_scale=1.,
                                unconditional_conditioning=None,
                                verbose=True,
                                precision=None,
                                fs=None,
                                **kwargs):
        device = self.model.betas.device        
        b = shape[0]
        
        if x_T is None:
            img = cond["c_concat"][0]
        else:
            img = x_T
        bridge_prior = img
        
        if precision is not None:
            if precision == 16:
                img = img.to(dtype=torch.float16)
            
        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        
        timesteps = self.sample_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        
        if verbose:
            iterator = tqdm(time_range, desc='Bridge 1st Order Sampler', total=total_steps)
        else:
            iterator = time_range

        # cond_copy, unconditional_conditioning_copy = copy.deepcopy(cond), copy.deepcopy(unconditional_conditioning)
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            outs = self.p_sample_bridge_first_order(img, cond, ts, index=index,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      bridge_prior=bridge_prior,
                                      fs=fs, **kwargs)
            img, pred_x0 = outs

            if index % log_every_t == 0:
                # print("save intermedaite steps: ", index)
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_bridge_first_order(self, x, c, t, index, repeat_noise=False,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, bridge_prior=None, **kwargs):
        b, *_, device = *x.shape, x.device
        
        if x.dim() == 5:
            is_video = True
        else:
            is_video = False

        if is_video:
            size = (b, 1, 1, 1, 1)
        else:
            size = (b, 1, 1, 1)
            
        kwargs["bridge_prior"] = bridge_prior

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c, **kwargs) # unet denoiser
        else:
            ### do_classifier_free_guidance
            if isinstance(c, torch.Tensor) or isinstance(c, dict):
                e_t_cond = self.model.apply_model(x, t, c, **kwargs)
                e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning, **kwargs)
            else:
                raise NotImplementedError

            model_output = e_t_uncond + unconditional_guidance_scale * (e_t_cond - e_t_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        alpha_s, sigma_s = (torch.full(size, self.sampler_bridge_alphas[index], device=device),
                            torch.full(size, self.sampler_bridge_sigmas[index], device=device))
        
        bridge_a_t, bridge_b_t, bridge_c_t =  (torch.full(size, self.sampler_bridge_coef_x0[index], device=device), 
                                                torch.full(size, self.sampler_bridge_coef_x1[index], device=device),
                                                torch.full(size, self.sampler_bridge_sqrt_var[index], device=device))
        '''
        predict_x_0 = (x - bridge_b_t * bridge_prior - bridge_c_t * e_t) / bridge_a_t
        '''
        if self.model.parameterization == "eps_psi":
            predict_x_0 = (x - alpha_s * sigma_s * e_t) / alpha_s
        elif self.model.parameterization == "eps_psi_new":
            predict_x_0 = (x - bridge_b_t * bridge_prior - bridge_c_t * e_t) / bridge_a_t
        else:
            assert self.model.parameterization == "v_psi"
            coef_1 = bridge_a_t / ((bridge_a_t ** 2. + bridge_c_t ** 2.) ** 0.5)
            coef_2 = bridge_c_t / ((bridge_a_t ** 2. + bridge_c_t ** 2.) ** 0.5)
            predict_x_0 = coef_1 * x - coef_2 * model_output

        if self.model.use_dynamic_rescale:
            original_scale_t = torch.full(size, self.sampler_original_diffusion_rescale[index], device=device)
            predict_x_0 = predict_x_0 / original_scale_t

        if index == 0:
            return predict_x_0, predict_x_0

        alpha_t, sigma_t = (torch.full(size, self.sampler_bridge_alphas[index - 1], device=device), 
                            torch.full(size, self.sampler_bridge_sigmas[index - 1], device=device))
        
        coef_x_s = (alpha_t * (sigma_t ** 2.)) / (alpha_s * (sigma_s ** 2.))
        coef_x_0 = alpha_t * (1. - (sigma_t ** 2.) / (sigma_s ** 2.))
        coef_noise = alpha_t * sigma_t * torch.sqrt(1. - (sigma_t ** 2.) / (sigma_s ** 2.))

        noise = torch.randn_like(x)
        x_prev = coef_x_s * x + coef_x_0 * predict_x_0 + coef_noise * noise

        return x_prev, predict_x_0
