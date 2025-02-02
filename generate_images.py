import argparse
import torch
import random
import numpy as np
import os
from monai.utils import set_determinism
from scripts.sample import LDMSampler
from scripts.utils import define_instance
import json

def set_seed(seed):
    set_determinism(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_configs(config_path):
    args = argparse.Namespace()
    with open(config_path, "r") as f:
        config = json.load(f)

    for section in ["paths", "model_config", "inference", "training", "schedulers", "mask_generation"]:
        if section in config:
            for k, v in config[section].items():
                setattr(args, k, v)

    return args

def load_models(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise_scheduler = define_instance(args, "noise_scheduler")
    mask_generation_noise_scheduler = define_instance(args, "mask_generation_noise_scheduler")

    # Load autoencoder with correct state dict structure
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    checkpoint = torch.load(args.trained_autoencoder_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])

    # Rest of model loading remains the same
    diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint_diffusion = torch.load(args.trained_diffusion_path, map_location=device)
    diffusion_unet.load_state_dict(checkpoint_diffusion["unet_state_dict"])
    scale_factor = checkpoint_diffusion["scale_factor"].to(device)

    controlnet = define_instance(args, "controlnet_def").to(device)
    checkpoint_controlnet = torch.load(args.trained_controlnet_path, map_location=device)
    controlnet.load_state_dict(checkpoint_controlnet["controlnet_state_dict"])

    mask_gen_autoencoder = define_instance(args, "mask_generation_autoencoder_def").to(device)
    checkpoint_mask_gen = torch.load(args.trained_mask_generation_autoencoder_path, map_location=device)
    mask_gen_autoencoder.load_state_dict(checkpoint_mask_gen)

    mask_gen_diffusion = define_instance(args, "mask_generation_diffusion_def").to(device)
    checkpoint_mask_gen_diff = torch.load(args.trained_mask_generation_diffusion_path, map_location=device)
    mask_gen_diffusion.load_state_dict(checkpoint_mask_gen_diff["unet_state_dict"])
    mask_gen_scale_factor = checkpoint_mask_gen_diff["scale_factor"]

    return (autoencoder, diffusion_unet, controlnet, noise_scheduler, scale_factor,
            mask_gen_autoencoder, mask_gen_diffusion, mask_gen_scale_factor,
            mask_generation_noise_scheduler)

def main():
    parser = argparse.ArgumentParser(description='Generate medical images')
    parser.add_argument('--num_images', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--config_path', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    set_seed(args.seed)

    print("Loading config from:", args.config_path)
    config = load_configs(args.config_path)

    print("Loading models...")
    models = load_models(config)
    (autoencoder, diffusion_unet, controlnet, noise_scheduler, scale_factor,
     mask_gen_autoencoder, mask_gen_diffusion, mask_gen_scale_factor,
     mask_generation_noise_scheduler) = models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    sampler = LDMSampler(
        body_region=["abdomen"],
        anatomy_list=["liver", "pancreas"],
        all_mask_files_json=config.all_mask_files_json,
        all_anatomy_size_condtions_json=config.all_anatomy_size_conditions_json,
        all_mask_files_base_dir=config.all_mask_files_base_dir,
        label_dict_json=config.label_dict_json,
        label_dict_remap_json=config.label_dict_remap_json,
        autoencoder=autoencoder,
        diffusion_unet=diffusion_unet,
        controlnet=controlnet,
        noise_scheduler=noise_scheduler,
        scale_factor=scale_factor,
        mask_generation_autoencoder=mask_gen_autoencoder,
        mask_generation_diffusion_unet=mask_gen_diffusion,
        mask_generation_scale_factor=mask_gen_scale_factor,
        mask_generation_noise_scheduler=mask_generation_noise_scheduler,
        device=device,
        latent_shape=config.latent_shape,
        mask_generation_latent_shape=config.mask_generation_latent_shape,
        output_size=config.output_size,
        output_dir=args.output_dir,
        controllable_anatomy_size=[],
        random_seed=args.seed
    )

    print(f"Generating {args.num_images} images...")
    output_filenames = sampler.sample_multiple_images(args.num_images)
    print(f"\nGenerated {len(output_filenames)} image-mask pairs:")
    for i, (img_path, mask_path) in enumerate(output_filenames):
        print(f"Pair {i+1}:")
        print(f"  Image: {img_path}")
        print(f"  Mask:  {mask_path}")

if __name__ == '__main__':
    main()