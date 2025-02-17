###################################################################################################
# IMPORTS
###################################################################################################
import argparse
import json
import os
import random
import tempfile

import numpy as np
import torch
from monai.config import print_config
from monai.utils import set_determinism

from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.sample import LDMSampler
from scripts.utils import define_instance

print_config()
###################################################################################################
# RANDOM SEEDS
###################################################################################################
seed = 42
set_determinism(seed=seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # CUDA
torch.cuda.manual_seed_all(seed)  # multiple GPUs
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###################################################################################################
# PATHS
###################################################################################################
directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory

# autoencoder_path = "./models/autoencoder_epoch17.pt"
# diffusion_path = "./models/checkpoint_epoch_5.pt"
###################################################################################################


###################################################################################################
# CONFIG
###################################################################################################
def load_configs(config_path="./configs/config_INFERENCE_v2.json"):
    """
    Load configurations from a single JSON file into an argparse.Namespace object
    """
    args = argparse.Namespace()

    # Load the consolidated config file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Process paths section
    for k, v in config["paths"].items():
        setattr(args, k, v)
        if "datasets/" in str(v):
            v = os.path.join(root_dir, v)
        print(f"{k}: {v}")
    print("Global config variables have been loaded.")

    # print(f"args.latent_channels:", args.latent_channels)
    print(f"config.latent_channels:", config["model_config"]["latent_channels"])
    # Process model configuration
    for k, v in config["model_config"].items():
        setattr(args, k, v)

    # Process inference configuration
    for k, v in config["inference"]["diffusion_unet_inference"].items():
        setattr(args, k, v)
        print(f"{k}: {v}")

    # Process training configuration
    for k, v in config["training"]["diffusion_unet_train"].items():
        setattr(args, k, v)

    # Process scheduler configurations
    for scheduler_type, scheduler_config in config["schedulers"].items():
        setattr(args, scheduler_type, scheduler_config)

    # Process mask generation configuration
    for k, v in config["mask_generation"].items():
        setattr(args, k, v)

    # Set model definitions
    for model_key in [
        "autoencoder_def",
        "controlnet_def",
        "diffusion_unet_def",
        "mask_generation_autoencoder_def",
        "mask_generation_diffusion_def",
    ]:
        if model_key in config["model_config"]:
            # Resolve @ references
            model_config = config["model_config"][model_key].copy()
            for k, v in model_config.items():
                if isinstance(v, str) and v.startswith("@"):
                    ref_key = v[1:]
                    model_config[k] = config["model_config"][ref_key]
            setattr(args, model_key, model_config)

    # Calculate and set latent shape
    print(f"args.latent_channels:", args.latent_channels)
    args.latent_shape = [
        args.latent_channels,
        args.dim[0] // 4,  # Using dim from inference config
        args.dim[1] // 4,
    ]
    print(f"latent_shape: {args.latent_shape}")

    print("Network definition and inference inputs have been loaded.")
    return args


# Load all configs
args = load_configs()
###################################################################################################


def load_autoencoder(trained_autoencoder_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = {
        "autoencoder": {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 4,
            "num_channels": [64, 128, 256],
            "num_res_blocks": [2, 2, 2],
            "norm_num_groups": 32,
            "norm_eps": 1e-6,
            "attention_levels": [False, False, False],
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "use_checkpointing": False,
            "use_convtranspose": False,
            "norm_float16": True,
            "num_splits": 1,
            "dim_split": 1,
        }
    }
    autoencoder = AutoencoderKlMaisi(**model["autoencoder"]).to(device)
    checkpoint = torch.load(trained_autoencoder_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    return autoencoder


###################################################################################################
# INITIALIZATION
noise_scheduler = define_instance(args, "noise_scheduler")

device = torch.device("cuda")

# Load model
autoencoder = load_autoencoder(args.trained_autoencoder_path)
autoencoder.eval()

diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
checkpoint_diffusion_unet = torch.load(args.trained_diffusion_path, weights_only=False)
diffusion_unet.load_state_dict(
    checkpoint_diffusion_unet["unet_state_dict"], strict=True
)
scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)

latent_shape = [4, 64, 64]
args.output_size = [256, 256]

print("All the trained model weights have been loaded.")
###################################################################################################


###################################################################################################
# LDM SAMPLER
###################################################################################################
print("image_output_ext:", args.image_output_ext)
ldm_sampler = LDMSampler(
    body_region=None,
    anatomy_list=None,
    all_mask_files_json=None,
    all_anatomy_size_condtions_json=None,
    all_mask_files_base_dir=None,
    label_dict_json=None,
    label_dict_remap_json=None,
    autoencoder=autoencoder,
    diffusion_unet=diffusion_unet,
    controlnet=None,
    noise_scheduler=noise_scheduler,
    scale_factor=scale_factor,
    mask_generation_autoencoder=None,
    mask_generation_diffusion_unet=None,
    mask_generation_scale_factor=None,
    mask_generation_noise_scheduler=None,
    device=device,
    latent_shape=latent_shape,
    mask_generation_latent_shape=None,
    output_size=None,
    output_dir=args.output_dir,
    controllable_anatomy_size=None,
    image_output_ext=args.image_output_ext,
    label_output_ext=None,
    real_img_median_statistics=args.real_img_median_statistics,
    spacing=None,
    num_inference_steps=args.num_inference_steps,
    mask_generation_num_inference_steps=None,
    random_seed=None,
    autoencoder_sliding_window_infer_size=None,
    autoencoder_sliding_window_infer_overlap=None,
)
###################################################################################################


###################################################################################################
# INFERENCE
###################################################################################################
print(f"The generated image/mask pairs will be saved in {args.output_dir}.")
output_filenames = ldm_sampler.sample_multiple_images(args.num_output_samples)
print("MAISI image/mask generation finished")
###################################################################################################
