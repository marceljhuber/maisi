import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from monai.utils import set_determinism
from torch.cuda.amp import autocast

from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.sample import *


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
    with open(config_path, "r") as f:
        config = json.load(f)

    args = argparse.Namespace()

    # Set model config
    for k, v in config["model_config"].items():
        setattr(args, k, v)

    # Set paths
    for k, v in config["paths"].items():
        setattr(args, k, v)

    # Get dimensions from inference config
    args.dim = config["inference"]["diffusion_unet_inference"]["dim"]

    # Calculate latent dimensions
    args.latent_shape = [args.latent_channels, args.dim[0] // 4, args.dim[1] // 4]

    return args


def load_autoencoder(args):
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

    model_config = model["autoencoder"]
    autoencoder = AutoencoderKlMaisi(**model_config).to(device)

    print(f"Loading Autoencoder from: ", args.trained_autoencoder_path)
    checkpoint = torch.load(args.trained_autoencoder_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    return autoencoder


def print_model_structure(autoencoder):
    """Print detailed information about the autoencoder's structure"""
    print("\nAutoencoder Structure:")
    print("=====================")

    print("\nDecoder Blocks:")
    for i, block in enumerate(autoencoder.decoder.blocks):
        print(f"\nBlock {i}:")
        print(f"Type: {type(block).__name__}")
        print("Attributes:", [attr for attr in dir(block) if not attr.startswith("_")])

        if hasattr(block, "in_channels"):
            print(f"in_channels: {block.in_channels}")
        if hasattr(block, "out_channels"):
            print(f"out_channels: {block.out_channels}")

    if hasattr(autoencoder.decoder, "final_conv"):
        print("\nFinal Conv Layer:")
        final_conv = autoencoder.decoder.final_conv
        print(f"Type: {type(final_conv).__name__}")
        if hasattr(final_conv, "in_channels"):
            print(f"in_channels: {final_conv.in_channels}")
        if hasattr(final_conv, "out_channels"):
            print(f"out_channels: {final_conv.out_channels}")


def generate_vae_samples(autoencoder, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # z = torch.randn(batch_size, 4, 64, 64).to(device)
    # print(f"z.shape:", z.shape)

    z = initialize_noise_latents((batch_size, 4, 64, 64), device).squeeze(0)
    # print(z.mean(), z2.mean())
    # print(f"z2.shape:", z2.shape)

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=1).to(device)
    autoencoder.eval()

    with torch.no_grad():
        with autocast():
            # samples = autoencoder.decode(z)
            samples = recon_model(z)
            print(f"samples.size:", samples.size())
    return samples.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--config_path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args.seed)
    config = load_configs(args.config_path)

    autoencoder = load_autoencoder(config)

    def convert_model_to_float32(model):
        for param in model.parameters():
            param.data = param.data.float()
        return model

    # In your main function, before generating:
    autoencoder = convert_model_to_float32(autoencoder)
    autoencoder.eval()

    print(f"Generating {args.num_images} images...")
    images = generate_vae_samples(autoencoder)

    for i, img in enumerate(images):
        save_path = f"{args.output_dir}/vae_sample_{i}.png"
        # Convert tensor to numpy and normalize to 0-255 range
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        # If image has extra dimensions, squeeze them
        img_np = np.squeeze(img_np)
        # Save using PIL
        Image.fromarray(img_np).save(save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
