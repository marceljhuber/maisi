from scripts.diff_model_setting import setup_logging
import copy
import os
import json
import numpy as np
import torch
import random
from PIL import Image
import glob
import subprocess

logger = setup_logging("notebook")

###################################################################################################
# RANDOM SEEDS
###################################################################################################
seed = 42 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # CUDA
torch.cuda.manual_seed_all(seed)  # multiple GPUs
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###################################################################################################
# UTILS
###################################################################################################
def list_image_files(directory_path):
    # Define common image file extensions
    image_extensions = ('.jpg', '.jpeg', '.png')

    # Use glob to find all files in directory and subdirectories
    files = glob.glob(os.path.join(directory_path, '**', '*.*'), recursive=True)

    # Filter files for image extensions
    image_files = [file for file in files if file.lower().endswith(image_extensions)]

    return image_files

def split_train_val_by_patient(image_names, train_ratio=0.9):
    # Extract unique patient IDs
    patient_ids = set(name.split('-')[1] for name in image_names)

    # Random split of patient IDs
    num_train = int(len(patient_ids) * train_ratio)
    train_patients = set(random.sample(list(patient_ids), num_train))

    # Split images based on patient IDs
    train_images = [img for img in image_names if img.split('-')[1] in train_patients]
    val_images = [img for img in image_names if img.split('-')[1] not in train_patients]

    return train_images, val_images

###################################################################################################
# PREPARE DATASET
###################################################################################################
# Get list of image files
directory_path = "/optima/exchange/mhuber/KermanyV3_resized/train"
image_files = list_image_files(directory_path)
train_imgs, val_imgs = split_train_val_by_patient(image_files)

# Create training dataset structure
work_dir = os.path.abspath("./temp_work_dir")
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)

dataroot_dir = os.path.join(work_dir, "processed_data")
if not os.path.isdir(dataroot_dir):
    os.makedirs(dataroot_dir)

# Process and save images in standardized format
def process_image(image_path, output_dir, prefix):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((256, 256))  # Resize to 256x256
    
    # Create standardized filename
    filename = os.path.basename(image_path)
    new_filename = f"{prefix}_{filename}"
    save_path = os.path.join(output_dir, new_filename)
    
    # Save processed image
    img.save(save_path)
    return save_path

# Process training and validation images
processed_train_imgs = []
processed_val_imgs = []

for img_path in train_imgs:
    processed_path = process_image(img_path, dataroot_dir, "train")
    processed_train_imgs.append({"image": processed_path})

for img_path in val_imgs:
    processed_path = process_image(img_path, dataroot_dir, "val")
    processed_val_imgs.append({"image": processed_path})

# Create datalist
datalist = {
    "training": processed_train_imgs,
    "validation": processed_val_imgs
}

datalist_file = os.path.join(work_dir, "datalist.json")
with open(datalist_file, "w") as f:
    json.dump(datalist, f)

###################################################################################################
# SETUP
###################################################################################################
# Create configuration for 2D image training
model_config = {
    "save_every": 1,
    "diffusion_unet_train": {
        "batch_size": 16,
        "cache_rate": 0,
        "lr": 0.0001,
        "n_epochs": 1000,
    },
    "diffusion_unet_inference": {
        "dim": [
            256,
            256,
            128
        ],
        "spacing": [
            1.0,
            1.0,
            0.75
        ],
        "top_region_index": [
            0,
            1,
            0,
            0
        ],
        "bottom_region_index": [
            0,
            0,
            1,
            0
        ],
        "random_seed": 42,
        "num_inference_steps": 10
    }
}

env_config = {
    "data_base_dir": "./data",
    "embedding_base_dir": "./embeddings",
    "json_data_list": "./temp_work_dir/datalist.json",
    "log_dir": "./logs",
    "model_dir": "./models",
    "model_filename": "diff_unet_ckpt.pt",
    "output_dir": "./predictions",
    "output_prefix": "unet_2d",
    "trained_autoencoder_path": "./models/autoencoder_epoch17.pt",
    "existing_ckpt_filepath": None
}

model_def = {
    "spatial_dims": 2,  # Changed from 3 to 2
    "image_channels": 1,
    "latent_channels": 1,  # Changed from 4 to 1 #
    "mask_generation_latent_shape": [
        1,  # Changed from 4 to 1 #
        256,  # Changed from 64 to 256 #
        256   # Changed from 64 to 256 #
    ],
    "autoencoder_def": {
        "_target_": "monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi",
        "spatial_dims": "@spatial_dims",
        "in_channels": "@image_channels",
        "out_channels": "@image_channels",
        "latent_channels": "@latent_channels",
        "num_channels": [
            64,
            128,
            256
        ],
        "num_res_blocks": [2,2,2],
        "norm_num_groups": 32,
        "norm_eps": 1e-06,
        "attention_levels": [
            False,
            False,
            False
        ],
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
        "use_checkpointing": False,
        "use_convtranspose": False,
        "norm_float16": True,
        "num_splits": 8,
        "dim_split": 1
    },
    "diffusion_unet_def": {
        "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
        "spatial_dims": "@spatial_dims",
        "in_channels": "@latent_channels",
        "out_channels": "@latent_channels",
        "num_channels": [
            64,
            128,
            256,
            512
        ],
        "attention_levels": [
            False,
            False,
            True,
            True
        ],
        "num_head_channels": [
            0,
            0,
            32,
            32
        ],
        "num_res_blocks": 2,
        "use_flash_attention": True,
        "include_top_region_index_input": False,  # Changed to false
        "include_bottom_region_index_input": False,  # Changed to false
        "include_spacing_input": False  # Changed to false
    },
    "controlnet_def": {
        "_target_": "monai.apps.generation.maisi.networks.controlnet_maisi.ControlNetMaisi",
        "spatial_dims": "@spatial_dims",
        "in_channels": "@latent_channels",
        "num_channels": [
            64,
            128,
            256,
            512
        ],
        "attention_levels": [
            False,
            False,
            True,
            True
        ],
        "num_head_channels": [
            0,
            0,
            32,
            32
        ],
        "num_res_blocks": 2,
        "use_flash_attention": True,
        "conditioning_embedding_in_channels": 8,
        "conditioning_embedding_num_channels": [8, 32, 64]
    },
    "mask_generation_autoencoder_def": {
        "_target_": "monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi",
        "spatial_dims": "@spatial_dims",
        "in_channels": 8,
        "out_channels": 125,
        "latent_channels": "@latent_channels",
        "num_channels": [
            32,
            64,
            128
        ],
        "num_res_blocks": [1, 2, 2],
        "norm_num_groups": 32,
        "norm_eps": 1e-06,
        "attention_levels": [
            False,
            False,
            False
        ],
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
        "use_flash_attention": False,
        "use_checkpointing": True,
        "use_convtranspose": True,
        "norm_float16": True,
        "num_splits": 8,
        "dim_split": 1
    },
    "mask_generation_diffusion_def": {
        "_target_": "monai.networks.nets.diffusion_model_unet.DiffusionModelUNet",
        "spatial_dims": "@spatial_dims",
        "in_channels": "@latent_channels",
        "out_channels": "@latent_channels",
        "channels":[64, 128, 256, 512],
        "attention_levels":[False, False, True, True],
        "num_head_channels":[0, 0, 32, 32],
        "num_res_blocks": 2,
        "use_flash_attention": True,
        "with_conditioning": True,
        "upcast_attention": True,
        "cross_attention_dim": 10
    },
    "mask_generation_scale_factor": 1.0055984258651733,
    "noise_scheduler": {
        "_target_": "monai.networks.schedulers.ddpm.DDPMScheduler",
        "num_train_timesteps": 1000,
        "beta_start": 0.0015,
        "beta_end": 0.0195,
        "schedule": "scaled_linear_beta",
        "clip_sample": False
    },
    "mask_generation_noise_scheduler": {
        "_target_": "monai.networks.schedulers.ddpm.DDPMScheduler",
        "num_train_timesteps": 1000,
        "beta_start": 0.0015,
        "beta_end": 0.0195,
        "schedule": "scaled_linear_beta",
        "clip_sample": False
    }
}

# Save configurations
os.makedirs(env_config["model_dir"], exist_ok=True)
os.makedirs(env_config["output_dir"], exist_ok=True)
os.makedirs(env_config["embedding_base_dir"], exist_ok=True)
os.makedirs(env_config["log_dir"], exist_ok=True)

env_config_filepath = os.path.join(work_dir, "environment.json")
model_config_filepath = os.path.join(work_dir, "model_config.json")
model_def_filepath = os.path.join(work_dir, "model_def.json")

with open(env_config_filepath, "w") as f:
    json.dump(env_config, f, sort_keys=True, indent=4)
with open(model_config_filepath, "w") as f:
    json.dump(model_config, f, sort_keys=True, indent=4)
with open(model_def_filepath, "w") as f:
    json.dump(model_def, f, sort_keys=True, indent=4)

print(f"Dumped all config settings to JSON-files.")

###################################################################################################
# TRAIN THE MODEL
###################################################################################################
logger.info("Training the model...")

from scripts.diff_model_train import diff_model_train

diff_model_train(
    env_config_filepath,
    model_config_filepath, 
    model_def_filepath,
    num_gpus=1,
    amp=True
)

