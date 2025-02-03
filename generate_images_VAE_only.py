import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from monai.utils import set_determinism
from torch.cuda.amp import autocast
from torchvision import transforms

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
    msg = autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    print(msg)
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


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2100], std=[0.0300]),
        ]
    )
    img = Image.open(image_path).convert("L")
    return transform(img).unsqueeze(0)


def generate_vae_samples(autoencoder, batch_size=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # z = initialize_noise_latents((batch_size, 4, 64, 64), device).squeeze(0)
    z = torch.randn(batch_size, 4, 64, 64).to(device)
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=1).to(device)
    # autoencoder.eval()

    std = 0.0300
    mean = 0.2100

    with torch.no_grad():
        with torch.amp.autocast("cuda"):  # Updated autocast import
            samples = autoencoder.decode(z)
            # samples = recon_model(z)
            # samples = samples.float()  # Convert to float32
            # samples = samples * std + mean
            print(f"samples.size:", samples.size())
    return samples.cpu()


def reconstruct(autoencoder, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.eval()

    x = preprocess_image(image_path).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            z = autoencoder.encode(x)[0]  # Get mean from encoder output
            x_hat = autoencoder.decode(z)
            print(f"z:", z)
            print(f"z.size:", z.size())
            print(f"x_hat.size:", x_hat.size())
    return z, x_hat.cpu()


def scale_latent(z, scale_factors=[0.8, 0.9, 1.0, 1.1, 1.2]):
    device = z.device
    scales = torch.tensor(scale_factors, device=device).view(-1, 1, 1, 1)
    noise = torch.rand(1, 4, 64, 64, device=device)
    print(f"z.mean:", torch.mean(z + noise * scales[0]))
    return z + noise * scales


def save_images(images, output_dir, prefix="vae_sample"):
    for i, img in enumerate(images):
        save_path = f"{output_dir}/{prefix}_{i}.png"
        img = (img - img.min()) / (img.max() - img.min())
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        img_np = np.squeeze(img_np)
        Image.fromarray(img_np).save(save_path)
        print(f"Saved: {save_path}")


def add_noise_to_latent(z, noise_level=0.1):
    """Add small Gaussian noise to latent vector"""
    noise = torch.randn_like(z) * noise_level
    return z + noise


def interpolate_latents(z1, z2, steps=5):
    """Create interpolations between two latent vectors"""
    alphas = torch.linspace(0, 1, steps).to(z1.device)
    return [(1 - a) * z1 + a * z2 for a in alphas]


def perturb_dimensions(z, dim_indices, amounts):
    """Modify specific dimensions of the latent space"""
    z_new = z.clone()
    for dim, amount in zip(dim_indices, amounts):
        z_new[:, dim] += amount
    return z_new


def flip_specific_channels(z, channel_pairs):
    """Flip specific channel pairs in the latent vector"""
    # Example pairs: [(0,3), (1,2)] to swap channels 0&3 and 1&2
    z_flipped = z.clone()
    for ch1, ch2 in channel_pairs:
        z_flipped[:, [ch1, ch2]] = z[:, [ch2, ch1]]
    return z_flipped


def generate_custom_random(
    shape=(1, 4, 64, 64),
    target_mean=0.0165,
    target_std=0.7485,
    target_min=-9.2422,
    target_max=8.2422,
    seed=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    if seed is not None:
        torch.manual_seed(seed)

    # Generate initial random values from normal distribution
    data = torch.randn(shape, device=device)

    # Scale and shift to match target std and mean
    data = (data - data.mean()) / data.std() * target_std + target_mean

    # Clip to enforce min/max bounds
    data = torch.clamp(data, target_min, target_max)

    # Readjust mean and std after clipping
    data = (data - data.mean()) / data.std() * target_std + target_mean

    return data


def sample_vectors_new(device="cuda" if torch.cuda.is_available() else "cpu"):
    vectors = torch.from_numpy(np.load("./encodings/encoded_vectors.npy")).to(device)
    indices = torch.randint(0, len(vectors), (2,), device=device)
    z1, z2 = vectors[indices[0]], vectors[indices[1]]
    return z1, z2


def sample_and_interpolate(
    n_steps, device="cuda" if torch.cuda.is_available() else "cpu"
):
    vectors = torch.from_numpy(np.load("./encodings/encoded_vectors.npy")).to(device)
    indices = torch.randint(0, len(vectors), (2,), device=device)
    z1, z2 = vectors[indices[0]], vectors[indices[1]]

    weights = torch.linspace(0, 1, n_steps, device=device)
    interpolated = torch.stack([(1 - w) * z1 + w * z2 for w in weights])

    return interpolated.squeeze(1)


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
    image_path = "/home/mhuber/Thesis/data/KermanyV3_resized/test/0/CNV-1569-1.jpeg"
    z, x_hat = reconstruct(autoencoder, image_path)
    print(f"z.mean:", z.mean())
    print(f"Encoded Image: ", z.shape)
    print(f"Decoded Image: ", x_hat.shape)
    images = generate_vae_samples(autoencoder)

    with torch.no_grad():
        with torch.amp.autocast("cuda"):  # Updated autocast import
            reconstructions = autoencoder.decode(
                # add_noise_to_latent(z, noise_level=0.05)
                # perturb_dimensions(z, dim_indices=[0, 1], amounts=[0.1, -0.1])
                # torch.tanh(z)
                # torch.rand(1, 4, 64, 64, device=z.device)
                # torch.randn(1, 4, 64, 64).to(z.device)
                # flip_specific_channels(z, channel_pairs=[(2, 3)])
                # generate_custom_random()
                sample_and_interpolate(n_steps=5)
            )

    save_images([x_hat], args.output_dir, prefix="reconstruction")
    save_images(images, args.output_dir, prefix="vae_sample")
    save_images(reconstructions, args.output_dir, prefix="variation")


if __name__ == "__main__":
    main()
