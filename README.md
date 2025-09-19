<h2 align="center">
  FrameBridge: Improving Image-to-Video Generation with Bridge Models
</h1>


## 1. Fine-tuning from CogVideoX

### Environment Setup

```bash
cd framebridge-cogvideox

conda create -n framebridge-cogvideox python=3.10
conda activate framebridge-cogvideox
pip install -r requirements.txt
```

### Fine-tuning from CogVideoX-2B

#### (1) Prepare the Initialization Model
First download the [CogVideoX-2B model](https://huggingface.co/zai-org/CogVideoX-2b). For our I2V fine-tuning, we slightly modify the transformer of CogVideoX (doubling the number of input channels for the input layer  to receive image input). You can directly download our modified version [CogVideoX-2B-modified](https://drive.google.com/drive/folders/1rm2JW0_qM3bmI1jcu5rZ0riZJg8Uvf7W?usp=drive_link).

#### (2) Prepare WebVid-2M Dataset
Download the metadata file `results_2M_train.csv` of WebVid-2M and put all videos in the folder `2M_train` with the following structure:

```
2M_train/
├── 00001.mp4
├── 00002.mp4
|   ......
└── *****.mp4
```


#### (3) Run the Fine-tuning Script
Modify the following path arguments in `finetune_single_rank_i2v_bridge_2b.sh`:

- `MODEL_PATH`:  path to CogVideoX-2B-modified
- `DATASET_PATH`:  path to `results_2M_train.csv`  
- `--video_folder`: path to `2M_train`  

Run the script:

```bash
bash finetune_single_rank_i2v_bridge_2b.sh
```

### Inference with Fine-tuned Model

The fine-tuned FrameBridge-CogVideoX model can be downloaded from the [Google Drive link](https://drive.google.com/drive/folders/194iwv7oJIK9Ob93mkjITuX5Ri_XeiCNd?usp=drive_link). (Unfortunately, due to limited computational resources and dataset quality, the performance of FrameBridge model is not as satisfactory as the official I2V version of CogVideoX.)

Before running inference, update the following arguments in `sample.sh`:

- `--model_path`: path to the original CogVideoX-2B model or CogVideoX-2B-modified (either option is ok as the transformer will be reloaded with the fine-tuned bridge model)
- `--image_or_video_path`: path to the image prompt  
- `--transformer_path`: path to the FrameBridge-CogVideoX model (or the transformer subfolder from fine-tuned models)  

Run the script:

```bash
bash sample.sh
```

## 2. Training from Scratch on UCF-101

### Environment Setup

```bash
cd framebridge-latte

conda create -n framebridge-latte python=3.9
cconda activate framebridge-latte
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Training

#### (1) Prepare UCF-101 Dataset and VAE Model
Download [UCF-101 dataset](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar). 

Download VAE model [stabilityai/sd-vae-ft-ema](https://huggingface.co/stabilityai/sd-vae-ft-ema) and put the files into the folder `checkpoints/vae`.

#### (2) Set Path Arguments in Config Files

We have three different config files for training different models:

- `configs/ucf101/ucf101_train_bridge.yaml`: training vanilla FrameBridge  
- `configs/ucf101/ucf101_train_nwarp.yaml`: training the neural prior model
- `configs/ucf101/ucf101_train_bridge_nwarp.yaml`: training FrameBridge with neural prior

Choose the corresponding config file based on the model you want to train, and set the path arguments inside:

- `data_path`: path to the extracted UCF101 folder  
- `pretrained_model_path`: path to the `checkpoints` folder which include the downloaded VAE.

If you want to train FrameBridge with neural prior, you also need to set:

- `nwarp_config`: path to `configs/ucf101/ucf101_sample_nwarp.yaml`
- `nwarp_ckpt`: path to checkpoint of trained neural prior model

#### (3) Run the Script

```bash
# Train vanilla FrameBridge
bash train_scripts/ucf101_bridge_train.sh

# Train neural prior model
bash train_scripts/ucf101_nwarp_train.sh

# Train FrameBridge with neural prior
bash train_scripts/ucf101_bridge_nwarp_train.sh
```

### Inference

#### (1) Set Path Arguments

You can download FrameBridge checkpoints from the Google Drive link:

- [FrameBridge trained without neural prior](https://drive.google.com/file/d/15_AfLQl6cnXmoBnxRibnHQzwwqxexp_P/view?usp=drive_link)
- [Neural prior model](https://drive.google.com/file/d/1fLRDaGctvedjYsMC_KG5FqBuD1HNyt2a/view?usp=drive_link)
- [FrameBridge trained with neural prior](https://drive.google.com/file/d/1O51G3ZPGkqItdlkde-jXOc765kbVBGbc/view?usp=drive_link)

There are also different config files for sampling with different models:

- `configs/ucf101/ucf101_sample_bridge.yaml`: sampling with vanilla FrameBridge  
- `configs/ucf101/ucf101_sample_bridge_nwarp.yaml`: sampling with FrameBridge with neural prior

Similarly, 
- set `pretrained_model_path` to the path of the `checkpoints` folder containing the VAE
- set `data_path` to the UCF-101 folder (the UCF-101 dataset is needed to obtain image prompts during sampling).

To use FrameBridge with neural prior, you also need to set:

- `nwarp_config`: path to `configs/ucf101/ucf101_sample_nwarp.yaml`
- `nwarp_ckpt`: path to checkpoint of trained neural prior model

in the config file `configs/ucf101/ucf101_sample_bridge_nwarp.yaml`.

#### (2) Run the Script

Set `--ckpt` arguments in the corresponding script with downloaded or trained checkpoint, and run the script:

```bash
# vanilla FrameBridge
bash sample/ucf101_bridge.sh

# FrameBridge with neural prior
bash sample/ucf101_bridge_neural_prior.sh
```

## Acknowledgements

This repository is built upon the excellent work of [CogVideoX](https://github.com/zai-org/CogVideo), [cogvideox-factory](https://github.com/huggingface/finetrainers), [Latte](https://github.com/Vchitect/Latte), and [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter). We sincerely thank the authors and contributors of these projects.
