import torch
import numpy as np
from functools import partial

def snr_align(snr, sl, sr, tl, tr):
    ratio = (snr - sl) / (sr - sl)
    return tl + ratio * (tr - tl)

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class BridgeScheduler():
    def __init__(self, diffusion_alphas, diffusion_sigmas, bridge_beta_min=0.01, bridge_beta_max=50.0, snr_aligned=True, timesteps=1000):
        self.bridge_beta_min = bridge_beta_min
        self.bridge_beta_max = bridge_beta_max
        self.timesteps = timesteps

        bridge_alphas = np.ones(timesteps)
        bridge_alphas_bar = np.ones(timesteps)
        bridge_t = (np.arange(0, timesteps) + 1.) / (timesteps + 1)
        sigma_1 = ((self.bridge_beta_min + self.bridge_beta_max) / 2.) ** 0.5
        # Note that here "sigma" represents the notation of Bridge-TTS, and is different from the "sigmas" in common diffusion schedules (up to a scale of alpha_t)
        bridge_sigmas = (((0.5 * (self.bridge_beta_max - self.bridge_beta_min) * bridge_t * bridge_t + self.bridge_beta_min * bridge_t)) ** 0.5)
        bridge_sigmas_bar = (sigma_1 ** 2. - bridge_sigmas ** 2.) ** 0.5
        bridge_coef_x0 = bridge_alphas * (bridge_sigmas_bar ** 2.) / (sigma_1 ** 2.)
        bridge_coef_x1 = bridge_alphas_bar * (bridge_sigmas ** 2.) / (sigma_1 ** 2.)
        bridge_sqrt_var = bridge_alphas * bridge_sigmas_bar * bridge_sigmas / sigma_1
        
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.bridge_coef_x0 = to_torch(bridge_coef_x0)
        self.bridge_coef_x1 = to_torch(bridge_coef_x1)
        self.bridge_sqrt_var = to_torch(bridge_sqrt_var)
        self.bridge_alphas = to_torch(bridge_alphas)
        self.bridge_alphas_bar = to_torch(bridge_alphas_bar)
        self.bridge_sigmas = to_torch(bridge_sigmas)
        self.bridge_sigmas_bar = to_torch(bridge_sigmas_bar)
        
        self.diffusion_alphas = diffusion_alphas
        self.diffusion_sigmas = diffusion_sigmas

        # SNR Alignment
        if snr_aligned:
            bridge_snr = self.bridge_coef_x0 / self.bridge_sqrt_var
            diffusion_snr = diffusion_alphas / diffusion_sigmas
            
            bridge_aligned_t = np.ones(bridge_snr.shape)
            cur = 0
            for i in range(len(bridge_snr)):
                if bridge_snr[i] > diffusion_snr[1]:
                    bridge_aligned_t[i] = snr_align(bridge_snr[i], diffusion_snr[0] + bridge_snr[0], diffusion_snr[1], 0, 1)
                else:
                    while cur < len(diffusion_snr) - 1 and bridge_snr[i] < diffusion_snr[cur + 1]:
                        cur += 1
                    if bridge_snr[i] < diffusion_snr[-1]:
                        bridge_aligned_t[i] = snr_align(bridge_snr[i], diffusion_snr[-1], 0., len(diffusion_snr) - 1, len(diffusion_snr))
                    else:
                        bridge_aligned_t[i] = snr_align(bridge_snr[i], diffusion_snr[cur], diffusion_snr[cur + 1], cur, cur + 1)
            self.bridge_aligned_t = to_torch(bridge_aligned_t)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
        x_prior: torch.Tensor
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.bridge_coef_x0 = self.bridge_coef_x0.to(device=original_samples.device)
        self.bridge_coef_x1 = self.bridge_coef_x1.to(device=original_samples.device)
        self.bridge_sqrt_var = self.bridge_sqrt_var.to(device=original_samples.device)
        timesteps = timesteps.to(device=original_samples.device)
        return (extract_into_tensor(self.bridge_coef_x0, timesteps, original_samples.shape) * original_samples
                + extract_into_tensor(self.bridge_coef_x1, timesteps, original_samples.shape) * x_prior
                + extract_into_tensor(self.bridge_sqrt_var, timesteps, original_samples.shape) * noise)

    def get_v_target(self, X0: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor, Xt):
        self.bridge_coef_x0 = self.bridge_coef_x0.to(device=noise.device)
        self.bridge_coef_x1 = self.bridge_coef_x1.to(device=noise.device)
        self.bridge_sqrt_var = self.bridge_sqrt_var.to(device=noise.device)
        timesteps = timesteps.to(device=noise.device)
        bridge_a_t, bridge_b_t, bridge_c_t =  (extract_into_tensor(self.bridge_coef_x0, timesteps, noise.shape), 
                                                extract_into_tensor(self.bridge_coef_x1, timesteps, noise.shape),
                                                extract_into_tensor(self.bridge_sqrt_var, timesteps, noise.shape))
        return bridge_a_t / ((bridge_a_t ** 2. + bridge_c_t ** 2.) ** 0.5) * noise - bridge_c_t / ((bridge_a_t ** 2. + bridge_c_t ** 2.) ** 0.5) * X0
        
    def from_v_to_x0(self, model_output: torch.Tensor, x_t: torch.Tensor, timesteps: torch.IntTensor, bridge_prior: torch.Tensor):
        self.bridge_coef_x0 = self.bridge_coef_x0.to(device=x_t.device)
        self.bridge_coef_x1 = self.bridge_coef_x1.to(device=x_t.device)
        self.bridge_sqrt_var = self.bridge_sqrt_var.to(device=x_t.device)
        self.bridge_alphas = self.bridge_alphas.to(device=x_t.device)
        self.bridge_sigmas = self.bridge_sigmas.to(device=x_t.device)
        timesteps = timesteps.to(device=x_t.device)
        bridge_a_t, bridge_b_t, bridge_c_t =  (extract_into_tensor(self.bridge_coef_x0, timesteps, x_t.shape), 
                                                extract_into_tensor(self.bridge_coef_x1, timesteps, x_t.shape),
                                                extract_into_tensor(self.bridge_sqrt_var, timesteps, x_t.shape))
        sigma_t = extract_into_tensor(self.bridge_sigmas, timesteps, x_t.shape)
        alpha_t = extract_into_tensor(self.bridge_alphas, timesteps, x_t.shape)
        sqrt_snr = bridge_a_t / bridge_c_t
        norm_coef = (bridge_a_t ** 2. + bridge_c_t ** 2.) ** 0.5
        coef_z_T = sqrt_snr * bridge_b_t / norm_coef
        coef_z_t = sqrt_snr / norm_coef
        coef_z_0 = norm_coef / bridge_c_t
        return (coef_z_t * x_t - coef_z_T * bridge_prior - model_output) / coef_z_0
    
    def get_training_weights(self, timesteps, device, shape):
        timesteps = timesteps.to(device=device)
        self.bridge_alphas = self.bridge_alphas.to(device=device)
        self.bridge_sigmas = self.bridge_sigmas.to(device=device)
        sigma_t = extract_into_tensor(self.bridge_sigmas, timesteps, shape)
        alpha_t = extract_into_tensor(self.bridge_alphas, timesteps, shape)
        bridge_a_t, bridge_b_t, bridge_c_t =  (extract_into_tensor(self.bridge_coef_x0, timesteps, shape), 
                                                extract_into_tensor(self.bridge_coef_x1, timesteps, shape),
                                                extract_into_tensor(self.bridge_sqrt_var, timesteps, shape))
        return (alpha_t ** 2. * bridge_c_t ** 2.) / (sigma_t ** 2. * (bridge_a_t ** 2. + bridge_c_t ** 2.))

    def align_marginal_and_timesteps(self, x_noisy, timesteps, bridge_prior, X0=None, noise=None):
        self.bridge_coef_x0 = self.bridge_coef_x0.to(device=x_noisy.device)
        self.bridge_coef_x1 = self.bridge_coef_x1.to(device=x_noisy.device)
        self.bridge_sqrt_var = self.bridge_sqrt_var.to(device=x_noisy.device)
        timesteps = timesteps.to(device=x_noisy.device)
        self.bridge_aligned_t = self.bridge_aligned_t.to(device=x_noisy.device)
        bridge_a_t, bridge_b_t, bridge_c_t =  (extract_into_tensor(self.bridge_coef_x0, timesteps, x_noisy.shape), 
                                                extract_into_tensor(self.bridge_coef_x1, timesteps, x_noisy.shape),
                                                extract_into_tensor(self.bridge_sqrt_var, timesteps, x_noisy.shape))
        aligned_x_noisy = (x_noisy - bridge_b_t * bridge_prior) / (bridge_a_t ** 2. + bridge_c_t ** 2.) ** 0.5
        aligned_t = extract_into_tensor(self.bridge_aligned_t, timesteps, timesteps.shape)
        if X0 is not None:
            coef_X0 = bridge_a_t / (bridge_a_t ** 2. + bridge_c_t ** 2.) ** 0.5
            coef_noise = bridge_c_t / (bridge_a_t ** 2. + bridge_c_t ** 2.) ** 0.5
            aligned_x_noisy_from_ori = coef_X0 * X0 + coef_noise * noise
            return aligned_x_noisy, aligned_t, aligned_x_noisy_from_ori
        else:
            return aligned_x_noisy, aligned_t
    
    def step(self, model_output, current_t, next_t, x_t, bridge_prior):
        pred_x0 = self.from_v_to_x0(model_output, x_t, (current_t * torch.ones((x_t.shape[0],), device=x_t.device)).to(torch.int64), bridge_prior)
        if next_t is None:
            return pred_x0

        size = (x_t.shape[0], 1, 1, 1, 1)
        alpha_s, sigma_s = (torch.full(size, self.bridge_alphas[current_t], device=x_t.device),
                            torch.full(size, self.bridge_sigmas[current_t], device=x_t.device))

        alpha_t, sigma_t = (torch.full(size, self.bridge_alphas[next_t], device=x_t.device), 
                            torch.full(size, self.bridge_sigmas[next_t], device=x_t.device))
        
        coef_x_s = (alpha_t * (sigma_t ** 2.)) / (alpha_s * (sigma_s ** 2.))
        coef_x_0 = alpha_t * (1. - (sigma_t ** 2.) / (sigma_s ** 2.))
        coef_noise = alpha_t * sigma_t * torch.sqrt(1. - (sigma_t ** 2.) / (sigma_s ** 2.))
        noise = torch.randn_like(x_t)
        x_prev = coef_x_s * x_t + coef_x_0 * pred_x0 + coef_noise * noise
        return x_prev