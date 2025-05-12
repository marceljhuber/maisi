#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
import json
import logging
from argparse import Namespace
from torch.amp import autocast

# Import needed modules from your codebase
from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.sample import ReconModel, initialize_noise_latents
from scripts.utils_data import split_grayscale_to_channels
from scripts.utils import define_instance


def setup_logger():
    """Set up a logger for the inference script"""
    logger = logging.getLogger("maisi.controlnet.inference")
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler("controlnet_inference.log")

    # Create formatters
    formatter = logging.Formatter(
        "[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add formatters to handlers
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def load_masks(mask_names, mask_path):
    """
    Load specified masks from the mask directory

    Args:
        mask_names: List of mask filenames to load
        mask_path: Path to the directory containing masks

    Returns:
        list: List of full paths to mask files
    """
    if not os.path.isdir(mask_path):
        raise ValueError(f"Mask path must be a directory: {mask_path}")

    mask_files = []
    for mask_name in mask_names:
        full_path = os.path.join(mask_path, mask_name)
        if not os.path.exists(full_path):
            print(f"Warning: Mask file not found: {full_path}")
            continue
        if not full_path.endswith(".png"):
            print(f"Warning: Mask file not a PNG: {full_path}")
            continue
        mask_files.append(full_path)

    if not mask_files:
        raise ValueError(f"No valid mask files found in {mask_path} with specified names")

    print(f"Found {len(mask_files)} valid mask(s)")
    return mask_files


def generate_noise_vectors(num_samples, latent_shape, device, output_dir):
    """
    Generate random noise vectors for diffusion sampling

    Args:
        num_samples: Number of noise vectors to generate
        latent_shape: Shape of the latent noise vector
        device: PyTorch device
        output_dir: Directory to save noise vectors

    Returns:
        list: List of noise latent tensors
    """
    noise_dir = os.path.join(output_dir, "noise_vectors")
    os.makedirs(noise_dir, exist_ok=True)

    # Check if noise vectors already exist
    existing_vectors = [f for f in sorted(os.listdir(noise_dir))
                        if f.startswith("noise_") and f.endswith(".pt")]

    if len(existing_vectors) >= num_samples:
        print(f"Using {num_samples} existing noise vectors from {noise_dir}")
        noise_vectors = []
        for i in range(num_samples):
            vector_path = os.path.join(noise_dir, existing_vectors[i])
            noise = torch.load(vector_path, map_location=device)
            noise_vectors.append(noise)
    else:
        print(f"Generating {num_samples} new noise vectors")
        noise_vectors = []
        for i in range(num_samples):
            noise = torch.randn(latent_shape, device=device, dtype=torch.float32)
            torch.save(noise, os.path.join(noise_dir, f"noise_{i:03d}.pt"))
            noise_vectors.append(noise)

    return noise_vectors


def denoise_sample_with_controlnet(unet, controlnet, noise_scheduler, condition, initial_latent, device, verbose=False):
    """
    Denoise a sample using ControlNet and UNet with autocast to handle mixed precision

    Args:
        unet: UNet model
        controlnet: ControlNet model
        noise_scheduler: Noise scheduler
        condition: Condition/mask tensor
        initial_latent: Initial noise latent
        device: PyTorch device
        verbose: Whether to show progress bar

    Returns:
        torch.Tensor: Denoised latent
    """
    # Create a copy of the input latent to avoid modifying the original
    latents = initial_latent.clone().to(dtype=torch.float32)

    # Ensure condition is float32
    condition = condition.to(dtype=torch.float32)

    # Get all timesteps from the noise scheduler
    timesteps = noise_scheduler.timesteps

    # Setup progress tracking
    progress_iter = tqdm(enumerate(timesteps), total=len(timesteps)) if verbose else enumerate(timesteps)

    # Denoise step by step
    for i, t in progress_iter:
        # Get current timestep as tensor
        current_timestep = torch.tensor([t], device=device)

        # Process through ControlNet and UNet with autocast
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
            down_block_res_samples, mid_block_res_sample = controlnet(
                x=latents,
                timesteps=current_timestep,
                controlnet_cond=condition,
            )

            # Ensure all tensors are in float32 for UNet
            noise_pred = unet(
                x=latents,
                timesteps=current_timestep,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

            # Update latent
            latents, _ = noise_scheduler.step(noise_pred, t, latents)

        # Clear intermediate tensors
        del noise_pred, down_block_res_samples, mid_block_res_sample

        # Periodically clear cache to avoid OOM issues
        if i % 50 == 0:
            torch.cuda.empty_cache()

    return latents


def denoise_sample_without_controlnet(unet, noise_scheduler, initial_latent, device, verbose=False):
    """
    Denoise a sample using only UNet (no ControlNet) with autocast

    Args:
        unet: UNet model
        noise_scheduler: Noise scheduler
        initial_latent: Initial noise latent
        device: PyTorch device
        verbose: Whether to show progress bar

    Returns:
        torch.Tensor: Denoised latent
    """
    # Create a copy of the input latent to avoid modifying the original
    latents = initial_latent.clone().to(dtype=torch.float32)

    # Get all timesteps from the noise scheduler
    timesteps = noise_scheduler.timesteps

    # Setup progress tracking
    progress_iter = tqdm(enumerate(timesteps), total=len(timesteps)) if verbose else enumerate(timesteps)

    # Denoise step by step
    for i, t in progress_iter:
        # Get current timestep as tensor
        current_timestep = torch.tensor([t], device=device)

        # Process through UNet with autocast (no ControlNet)
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
            # Get noise prediction directly from UNet
            noise_pred = unet(
                x=latents,
                timesteps=current_timestep,
            )

            # Update latent
            latents, _ = noise_scheduler.step(noise_pred, t, latents)

        # Clear intermediate tensors
        del noise_pred

        # Periodically clear cache to avoid OOM issues
        if i % 50 == 0:
            torch.cuda.empty_cache()

    return latents


def save_generated_image(generated_image, prefix, sample_idx, output_dir):
    """
    Save the generated image

    Args:
        generated_image: Generated image tensor
        prefix: Prefix for the filename
        sample_idx: Sample index
        output_dir: Output directory

    Returns:
        str: Path to the saved image
    """
    # Convert tensor to numpy for matplotlib
    gen_img_np = generated_image.squeeze(0).permute(1, 2, 0).numpy()

    # Normalize to 0-1 range if needed
    if gen_img_np.min() < 0 or gen_img_np.max() > 1:
        gen_img_np = (gen_img_np - gen_img_np.min()) / (gen_img_np.max() - gen_img_np.min())

    # Save as PNG
    output_path = os.path.join(output_dir, f"{prefix}_{sample_idx:03d}.png")

    # Create figure
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(gen_img_np, cmap="gray")
    plt.axis("off")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return output_path


def process_with_and_without_controlnet(mask_path, num_samples, noise_vectors, models, device, output_dir, logger):
    """
    Process noise vectors both with and without ControlNet

    Args:
        mask_path: Path to the mask file
        num_samples: Number of samples to generate per mask
        noise_vectors: List of noise vectors
        models: Tuple of (autoencoder, unet, controlnet, noise_scheduler, scale_factor)
        device: PyTorch device
        output_dir: Output directory
        logger: Logger instance

    Returns:
        tuple: Lists of paths to generated images (with_controlnet, without_controlnet)
    """
    autoencoder, unet, controlnet, noise_scheduler, scale_factor = models

    # Create reconstruction model
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(device)

    # Load and prepare mask
    mask = Image.open(mask_path).convert("L")
    mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

    # Split grayscale mask to channels if needed
    mask_channels = split_grayscale_to_channels(mask_tensor)

    # Get mask prefix for naming output files
    mask_prefix = os.path.splitext(os.path.basename(mask_path))[0]

    # Generate images
    with_controlnet_images = []
    without_controlnet_images = []

    for i, noise in enumerate(noise_vectors[:num_samples]):
        logger.info(f"Processing noise vector {i+1}/{num_samples}")

        try:
            # Ensure noise vector is float32
            noise = noise.to(dtype=torch.float32)

            # First, generate image without ControlNet
            logger.info(f"Generating original (no ControlNet) for vector {i+1}")
            original_latent = denoise_sample_without_controlnet(
                unet=unet,
                noise_scheduler=noise_scheduler,
                initial_latent=noise,
                device=device,
                verbose=True
            )

            # Decode the latent with autocast
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
                original_image = recon_model(original_latent)
                original_image = original_image.to(dtype=torch.float32)

            # Normalize to -1 to 1 range
            b_min, b_max = -1.0, 1.0
            original_image = torch.clip(original_image, b_min, b_max).cpu()
            original_image = (original_image - b_min) / (b_max - b_min)

            # Save the original image
            original_path = save_generated_image(
                generated_image=original_image,
                prefix="original",
                sample_idx=i,
                output_dir=output_dir
            )

            without_controlnet_images.append(original_path)
            logger.info(f"Saved original image to {original_path}")

            # Clear memory
            del original_latent, original_image
            torch.cuda.empty_cache()

            # Now generate with ControlNet
            logger.info(f"Generating ControlNet version for vector {i+1} with mask {mask_prefix}")
            controlled_latent = denoise_sample_with_controlnet(
                unet=unet,
                controlnet=controlnet,
                noise_scheduler=noise_scheduler,
                condition=mask_channels,
                initial_latent=noise,
                device=device,
                verbose=True
            )

            # Decode the latent with autocast
            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
                controlled_image = recon_model(controlled_latent)
                controlled_image = controlled_image.to(dtype=torch.float32)

            # Normalize to -1 to 1 range
            controlled_image = torch.clip(controlled_image, b_min, b_max).cpu()
            controlled_image = (controlled_image - b_min) / (b_max - b_min)

            # Save the controlled image
            controlled_path = save_generated_image(
                generated_image=controlled_image,
                prefix=mask_prefix,
                sample_idx=i,
                output_dir=output_dir
            )

            with_controlnet_images.append(controlled_path)
            logger.info(f"Saved ControlNet image to {controlled_path}")

            # Clear memory
            del controlled_latent, controlled_image
            torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error processing vector {i+1}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    return with_controlnet_images, without_controlnet_images


def convert_models_to_float32(models):
    """
    Converts model parameters to float32 to avoid precision issues

    Args:
        models: A tuple of (autoencoder, unet, controlnet, noise_scheduler, scale_factor)

    Returns:
        Tuple of models with parameters converted to float32
    """
    autoencoder, unet, controlnet, noise_scheduler, scale_factor = models

    # Convert autoencoder parameters
    for param in autoencoder.parameters():
        param.data = param.data.to(torch.float32)

    # Convert UNet parameters
    for param in unet.parameters():
        param.data = param.data.to(torch.float32)

    # Convert ControlNet parameters
    for param in controlnet.parameters():
        param.data = param.data.to(torch.float32)

    return autoencoder, unet, controlnet, noise_scheduler, scale_factor


def create_comparison_grid(masks, controlnet_images, original_images, output_dir, logger):
    """
    Create a grid of images comparing masks, ControlNet outputs, and original images

    Args:
        masks: List of mask file paths
        controlnet_images: Dictionary of mask_prefix -> list of image paths
        original_images: List of original image paths
        output_dir: Output directory
        logger: Logger instance
    """
    logger.info("Creating comparison grid...")

    # For each mask, create a grid
    for mask_path in masks:
        mask_prefix = os.path.splitext(os.path.basename(mask_path))[0]

        if mask_prefix not in controlnet_images:
            logger.warning(f"No images found for mask {mask_prefix}, skipping grid")
            continue

        # Load mask
        mask_img = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_img) / 255.0

        # Create figure
        num_samples = len(controlnet_images[mask_prefix])
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

        # If only one sample, make axes indexable
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        # Add a title to the figure
        fig.suptitle(f"Comparison for mask: {mask_prefix}", fontsize=16)

        # Column titles
        col_titles = ["Mask", "ControlNet Output", "Original (No ControlNet)"]
        for i, title in enumerate(col_titles):
            fig.text(0.15 + 0.35 * i, 0.98, title, ha='center', fontsize=14)

        # Fill the grid
        for i in range(num_samples):
            # First column: mask
            axes[i, 0].imshow(mask_np, cmap='gray')
            axes[i, 0].set_title(f"Mask")
            axes[i, 0].axis('off')

            # Second column: ControlNet output
            ctrl_image = Image.open(controlnet_images[mask_prefix][i])
            ctrl_np = np.array(ctrl_image)
            axes[i, 1].imshow(ctrl_np)
            axes[i, 1].set_title(f"Sample {i}")
            axes[i, 1].axis('off')

            # Third column: Original image
            orig_image = Image.open(original_images[i])
            orig_np = np.array(orig_image)
            axes[i, 2].imshow(orig_np)
            axes[i, 2].set_title(f"Sample {i}")
            axes[i, 2].axis('off')

        # Save the grid
        grid_path = os.path.join(output_dir, f"grid_{mask_prefix}.png")
        plt.tight_layout()
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved comparison grid to {grid_path}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="ControlNet Inference")
    parser.add_argument("--config_path", type=str, help="Path to config JSON file", default="./configs/config_CONTROLNET_germany.json")
    parser.add_argument("--mask_path", type=str, help="Path to directory containing mask files", default="/home/user/Thesis/data/retouch_masks")
    parser.add_argument("--masks", nargs="+", default=["0.png", "10.png", "11.png", "12.png", "13.png"], help="List of mask filenames to use (e.g., 0.png 1.png)")
    parser.add_argument("--num_samples", type=int, help="Number of samples to generate per mask", default=5)
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for generated images")
    cli_args = parser.parse_args()

    # Set up logger
    logger = setup_logger()

    # Set device (using the first available CUDA device)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    if cli_args.output_dir is None:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        cli_args.output_dir = os.path.join("./outputs/retouch", timestamp)

    os.makedirs(cli_args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {cli_args.output_dir}")

    # Load configuration
    with open(cli_args.config_path, "r") as f:
        config = json.load(f)

    # Create args object with attribute access for define_instance
    args = Namespace()

    # Merge config into args
    for section in ["environment", "model_def", "training", "model"]:
        if section in config:
            for k, v in config[section].items():
                setattr(args, k, v)

    # Override with command line arguments
    args.trained_autoencoder_path = config["environment"]["trained_autoencoder_path"]
    args.trained_diffusion_path = config["environment"]["trained_diffusion_path"]
    args.trained_controlnet_path = config["environment"]["trained_controlnet_path"]

    # Load models - this follows the pattern in your existing code
    logger.info("Loading models...")

    # Load autoencoder
    if not os.path.exists(args.trained_autoencoder_path):
        raise ValueError(f"Autoencoder checkpoint not found: {args.trained_autoencoder_path}")

    model_config = config["model"]["autoencoder"]
    autoencoder = AutoencoderKlMaisi(**model_config).to(device)

    checkpoint = torch.load(
        args.trained_autoencoder_path,
        map_location=device,
        weights_only=True,
    )
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    autoencoder.eval()
    logger.info(f"Loaded autoencoder from {args.trained_autoencoder_path}")

    # Convert autoencoder to float32
    for param in autoencoder.parameters():
        param.data = param.data.to(torch.float32)

    # Load UNet
    unet = define_instance(args, "diffusion_unet_def").to(device)

    # Load diffusion model checkpoint
    if not os.path.exists(args.trained_diffusion_path):
        raise ValueError(f"Diffusion model checkpoint not found: {args.trained_diffusion_path}")

    diffusion_model_ckpt = torch.load(
        args.trained_diffusion_path,
        map_location=device,
        weights_only=False,
    )
    unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"])
    scale_factor = diffusion_model_ckpt.get("scale_factor", 1.0)
    logger.info(f"Loaded diffusion model from {args.trained_diffusion_path}")
    logger.info(f"Using scale_factor: {scale_factor}")

    # Load ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)

    if not os.path.exists(args.trained_controlnet_path):
        raise ValueError(f"ControlNet checkpoint not found: {args.trained_controlnet_path}")

    controlnet_ckpt = torch.load(args.trained_controlnet_path, map_location=device, weights_only=True)
    controlnet.load_state_dict(controlnet_ckpt["controlnet_state_dict"])
    logger.info(f"Loaded ControlNet from {args.trained_controlnet_path}")

    # Initialize noise scheduler
    noise_scheduler = define_instance(args, "noise_scheduler")

    # Set all models to evaluation mode
    autoencoder.eval()
    unet.eval()
    controlnet.eval()

    # Package models
    models = (autoencoder, unet, controlnet, noise_scheduler, scale_factor)

    # Ensure all model parameters are in float32
    models = convert_models_to_float32(models)

    # Enable performance optimizations
    torch.backends.cudnn.benchmark = True

    # Load specified masks
    logger.info(f"Looking for masks: {cli_args.masks} in directory: {cli_args.mask_path}")
    mask_files = load_masks(cli_args.masks, cli_args.mask_path)

    # Generate or load noise vectors (latent shape is typically [1, 4, 64, 64])
    noise_vectors = generate_noise_vectors(
        num_samples=cli_args.num_samples,
        latent_shape=(1, 4, 64, 64),  # Default latent shape
        device=device,
        output_dir=cli_args.output_dir
    )

    # Process each mask
    all_controlnet_images = []
    all_original_images = []
    controlnet_images_by_mask = {}
    mask_file_paths = []
    start_time = time.time()

    for mask_file in mask_files:
        mask_prefix = os.path.splitext(os.path.basename(mask_file))[0]
        controlnet_images_by_mask[mask_prefix] = []
        mask_file_paths.append(mask_file)

        logger.info(f"\nProcessing mask: {mask_file}")
        controlnet_images, original_images = process_with_and_without_controlnet(
            mask_path=mask_file,
            num_samples=cli_args.num_samples,
            noise_vectors=noise_vectors,
            models=models,
            device=device,
            output_dir=cli_args.output_dir,
            logger=logger
        )
        controlnet_images_by_mask[mask_prefix] = controlnet_images
        all_controlnet_images.extend(controlnet_images)
        all_original_images.extend(original_images)

    total_time = time.time() - start_time
    logger.info(f"\nGenerated {len(all_controlnet_images)} ControlNet images and {len(all_original_images)} original images in {total_time:.2f} seconds")
    logger.info(f"Output directory: {cli_args.output_dir}")

    # Generate comparison grid
    create_comparison_grid(
        masks=mask_file_paths,
        controlnet_images=controlnet_images_by_mask,
        original_images=all_original_images,
        output_dir=cli_args.output_dir,
        logger=logger
    )

    # Print summary
    for mask_name in cli_args.masks:
        mask_prefix = os.path.splitext(mask_name)[0]
        logger.info(f"  • {mask_prefix}: {cli_args.num_samples} samples")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)