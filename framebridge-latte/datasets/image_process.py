import os
import re
import json
import torch
import torchvision
import numpy as np
import argparse
import video_transforms
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import shutil
from tqdm import tqdm

def get_filelist(file_path):
    Filelist = []
    for home, dirs, files in os.walk(file_path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
            # Filelist.append( filename)
    return Filelist

def generate_image_dataset(config, video_transform):
    video_lists = get_filelist(config.video_path)
    for filename in tqdm(video_lists):
        print(filename)
        vframes, aframes, info = torchvision.io.read_video(filename=filename, pts_unit='sec', output_format='TCHW')
        video = video_transform(vframes)
        video_name = os.path.basename(filename)[:-4]
        for i in range(video.shape[0] // config.frame_interval):
            image = video[i * config.frame_interval].permute(1, 2, 0) * 255.0
            image = Image.fromarray(np.uint8(image))
            image.save(os.path.join(config.save_path, video_name + f'_{i}.jpg'))

def generate_image_txt(config):
    with open(config.txt_path, 'w') as f:
        for name in os.listdir(config.save_path):
            if "jpg" in name:
                f.write(name + '\n')

def categorize_image_dataset(config):
    image_list = os.listdir(config.image_path)
    for filename in tqdm(image_list):
        class_name = filename.split('_')[1]
        os.makedirs(os.path.join(config.save_path, class_name), exist_ok=True)
        shutil.copy(os.path.join(config.image_path, filename), os.path.join(config.save_path, class_name))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=3)
    parser.add_argument("--video-path", type=str, default="data/UCF-101")
    parser.add_argument("--image-path", type=str, default="data/UCF-101-image")
    parser.add_argument("--save-path", type=str, default="data/UCF-101-image")
    parser.add_argument("--txt-path", type=str, default="data/UCF-101-frame-data.txt")
    parser.add_argument("--opt", type=str, choices=["generate", "categorize"], default="generate")
    
    transform_ucf101 = transforms.Compose([
            video_transforms.ToTensorVideo(), # TCHW
            video_transforms.RandomHorizontalFlipVideo(),
            video_transforms.UCFCenterCropVideo(256),
        ])
    
    config = parser.parse_args()
    
    if config.opt == "generate":
        generate_image_dataset(config, transform_ucf101)
        generate_image_txt(config)
    else:
        categorize_image_dataset(config)