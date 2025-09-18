from einops import rearrange
import cv2
import imageio
import torch
from scipy.fftpack import dct, idct
import copy
import numpy as np

def save_images(file_name, x):
    for i in range(x.shape[0]):
        image_save_path = file_name + f"{i}" + '.jpg'
        img = rearrange(x[i], 'c h w -> h w c')
        img = ((img * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().numpy()
        cv2.imwrite(image_save_path, img)

def save_videos(file_name, x):
    for i in range(x.shape[0]):
        video_ = ((x[i] * 0.5 + 0.5) * 255).add_(0.5).clamp_(0, 255).to(dtype=torch.uint8).cpu().permute(0, 2, 3, 1).contiguous()
        video_save_path = file_name + f"{i}" + '.mp4'
        imageio.mimwrite(video_save_path, video_, fps=8, quality=10)

def inverse_dct_1d(data, N):
    padding_shape = list(data.shape)
    padding_shape[-1] = N - data.shape[-1]
    padding = np.zeros(padding_shape)
    coef = idct(np.concatenate((data, padding), axis=-1), norm='ortho')
    return coef

def inverse_dct_2d(data, W, H):
    tmp = np.zeros((W, H))
    tmp[:data.shape[-2], :data.shape[-1]] = data
    coef = idct(idct(tmp.T, norm='ortho').T, norm='ortho')
    return coef

def forward_dct_1d(data, n):
    coef = dct(data, norm='ortho')
    return coef[..., :n]

def forward_dct_2d(data, w, h):
    coef = dct(dct(data.T, norm='ortho').T, norm='ortho')
    return coef[..., :w, :h]
    
def dct_2d_compress(data, w, h):
    tmp = data.reshape(-1, data.shape[-2], data.shape[-1])
    result = torch.zeros(tmp.shape, device=data.device)
    for i in range(tmp.shape[0]):
        feature = tmp[i].cpu().numpy()
        result[i] = torch.from_numpy(inverse_dct_2d(forward_dct_2d(feature, w, h), feature.shape[0], feature.shape[1]))
    result = result.reshape(data.shape)
    print(torch.max(torch.abs(data - result)))
    return result

def dct_1d_compress(data, n):
    tmp = data.reshape(-1, data.shape[-1])
    result = torch.zeros(tmp.shape, device=data.device)
    
    for i in range(tmp.shape[0]):
        feature = tmp[i].cpu().numpy()
        result[i] = torch.from_numpy(inverse_dct_1d(forward_dct_1d(feature, n), feature.shape[0]))
    result = result.reshape(data.shape)
    
    return result

def dct_postprocess(x, condition_frame, data_quantile, n, use_quantile=True):
    if use_quantile:
        dct_coef = torch.sign(x) * x ** 2. * data_quantile
    else:
        dct_coef = x
    data = torch.from_numpy(inverse_dct_1d(dct_coef.permute(0, 2, 3, 4, 1).contiguous().cpu().numpy(), n)).to(x.device).permute(0, 4, 1, 2, 3).contiguous()
    return data + condition_frame
    
def dct_preprocess(x, condition_frame, data_quantile, n, use_quantile=True):
    data = forward_dct_1d((x - condition_frame).permute(0, 2, 3, 4, 1).contiguous().cpu().numpy(), n)
    dct_coef = torch.from_numpy(data).to(x.device).permute(0, 4, 1, 2, 3).contiguous()
    if use_quantile:
        dct_coef = torch.sign(dct_coef) * torch.sqrt(torch.abs(dct_coef) / data_quantile)
    return dct_coef