import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Force non-interactive backend
import numpy as np
import torch
from PIL import Image
import torch.distributed as dist
import torch.nn.functional as F
from monai.networks.utils import copy_model_state
from monai.utils import RankFilter

# from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from datetime import datetime
import wandb
from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.utils_data import (
    create_latent_dataloaders,
    create_cluster_dataloaders,
    create_oct_dataloaders,
    split_grayscale_to_channels,
)
from .utils import (
    define_instance,
    setup_ddp,
)

from tqdm import tqdm

from scripts.sample import ReconModel, initialize_noise_latents


def validate_and_visualize(
    autoencoder,
    unet,
    controlnet,
    noise_scheduler,
    val_loader,
    device,
    epoch,
    save_dir,
    scale_factor=1.0,
    num_samples=20,
    weighted_loss=1.0,
    weighted_loss_label=None,
    rank=0,
    generate_visuals=True,
):
    """
    Validate the model on the validation set, compute loss metrics,
    generate visualizations, and log everything to wandb.

    Args:
        autoencoder: The trained autoencoder model
        unet: The trained diffusion UNet model
        controlnet: The ControlNet model being trained
        noise_scheduler: Noise scheduler for the diffusion process
        val_loader: Validation data loader
        device: The device to run inference on
        epoch: Current epoch number
        save_dir: Directory to save the validation visualizations
        scale_factor: Scale factor for the latent space
        num_samples: Number of validation samples to visualize
        weighted_loss: Weight factor for loss computation on specific regions
        weighted_loss_label: Labels for regions with weighted loss
        rank: Process rank for distributed training
        generate_visuals: Whether to generate visualizations
    """

    # Set constant ranges for normalization
    IMG_RANGE = {"min": 0, "max": 255}  # PNG image intensity range
    LATENT_RANGE = {"min": -1.0, "max": 1.0}  # autoencoder output range

    # Set models to evaluation mode
    autoencoder.eval()
    controlnet.eval()
    unet.eval()

    # Create reconstruction model
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(
        device
    )

    # Create directory for validation visualizations if needed
    val_vis_dir = None

    if generate_visuals:
        val_vis_dir = os.path.join(save_dir, f"epoch_{epoch + 1}_validation")
        os.makedirs(val_vis_dir, exist_ok=True)
        print(f"Saving visualizations for validation set at {val_vis_dir}")

    # Set up inference timesteps
    # noise_scheduler.set_timesteps(1000, device=device)

    # Collect validation metrics
    metrics = _compute_validation_metrics(
        autoencoder=autoencoder,
        unet=unet,
        controlnet=controlnet,
        noise_scheduler=noise_scheduler,
        val_loader=val_loader,
        device=device,
        scale_factor=scale_factor,
        weighted_loss=weighted_loss,
        weighted_loss_label=weighted_loss_label,
    )

    # Generate visualizations if requested
    if generate_visuals:
        _generate_validation_visualizations(
            autoencoder=autoencoder,
            unet=unet,
            controlnet=controlnet,
            noise_scheduler=noise_scheduler,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            val_vis_dir=val_vis_dir,
            recon_model=recon_model,
            scale_factor=scale_factor,
            num_samples=num_samples,
            rank=rank,
        )

    # Log metrics to wandb
    if rank == 0 and wandb.run is not None:
        _log_validation_to_wandb(metrics, val_vis_dir, epoch, num_samples)

    # Return models to training mode
    controlnet.train()

    return metrics["val_loss"]


def _compute_validation_metrics(
    autoencoder,
    unet,
    controlnet,
    noise_scheduler,
    val_loader,
    device,
    scale_factor,
    weighted_loss,
    weighted_loss_label,
):
    """Memory-efficient validation metrics computation with mixed precision"""
    total_loss = 0.0
    total_samples = 0
    max_val_batches = min(len(val_loader), 5)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_val_batches:
                break

            # Get full batch data
            inputs = batch[0].squeeze(1).to(device) * scale_factor
            labels = batch[1].to(device)

            ############################################################################################################
            # First, convert labels to 5 channels
            labels_5ch = split_grayscale_to_channels(labels)  # [64, 5, 256, 256]

            # Scale up inputs from 64x64 to 256x256
            inputs_upscaled = F.interpolate(
                inputs, size=(256, 256), mode="bilinear", align_corners=False
            )
            # inputs_upscaled: [64, 4, 256, 256]

            # Concatenate along channel dimension (dim=1)
            combined_labels = torch.cat([labels_5ch, inputs_upscaled], dim=1)
            # Result: [64, 9, 256, 256] (5 + 4 = 9 channels)
            ############################################################################################################

            sub_batch_size = 1  # Process 1 sample at a time for max memory efficiency

            # Process in sub-batches
            for start_idx in range(0, inputs.shape[0], sub_batch_size):
                end_idx = min(start_idx + sub_batch_size, inputs.shape[0])

                sub_inputs = inputs[start_idx:end_idx].to(torch.float32)
                sub_combined_labels = combined_labels[
                    start_idx:end_idx
                ]  # Already 9 channels

                # Random timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (sub_inputs.shape[0],),
                    device=device,
                ).long()

                # Add noise
                noise = torch.randn_like(sub_inputs, dtype=torch.float32)
                noisy_latent = noise_scheduler.add_noise(sub_inputs, noise, timesteps)

                # Clear cache
                torch.cuda.empty_cache()

                # Use autocast for forward passes
                with autocast("cuda", enabled=True):
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        x=noisy_latent,
                        timesteps=timesteps,
                        controlnet_cond=sub_combined_labels.float(),  # 9-channel combined labels
                    )

                    noise_pred = unet(
                        x=noisy_latent,
                        timesteps=timesteps,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    )

                # Compute loss with explicit casting to float
                batch_loss = F.l1_loss(noise_pred.float(), noise.float())

                total_loss += batch_loss.item() * sub_inputs.shape[0]
                total_samples += sub_inputs.shape[0]

                # Free memory
                del (
                    noise_pred,
                    noisy_latent,
                    down_block_res_samples,
                    mid_block_res_sample,
                )
                torch.cuda.empty_cache()

    return {"val_loss": total_loss / max(total_samples, 1)}


def _create_weighted_loss_mask(inputs, labels, weighted_loss, weighted_loss_label):
    """Create a weighted loss mask for region-specific loss weighting"""
    weights = torch.ones_like(inputs)
    roi_mask = torch.zeros_like(inputs[:, :1], dtype=torch.float32)

    # Interpolate labels to match latent dimensions
    interpolate_label = F.interpolate(labels, size=inputs.shape[2:], mode="nearest")

    # For each target label, add to the mask
    for label in weighted_loss_label:
        for channel in range(interpolate_label.shape[1]):
            # Create mask for this channel/label combination
            channel_mask = (
                interpolate_label[:, channel : channel + 1] == label
            ).float()
            roi_mask = roi_mask + channel_mask

    # Convert to binary mask and apply to weights
    roi_mask = (roi_mask > 0).float()
    weights = weights.masked_fill(
        roi_mask.repeat(1, inputs.shape[1], 1, 1) > 0, weighted_loss
    )

    return weights


def _generate_validation_visualizations(
    autoencoder,
    unet,
    controlnet,
    noise_scheduler,
    val_loader,
    device,
    epoch,
    val_vis_dir,
    recon_model,
    scale_factor,
    num_samples,
    rank,
):
    """Generate validation visualizations for a subset of samples"""
    # Collect samples for visualization
    sample_batches = []
    sample_count = 0

    for batch in val_loader:
        sample_batches.append(batch)
        sample_count += batch[0].shape[0]
        if sample_count >= num_samples:
            break

    print(f"Generating visualizations for {min(sample_count, num_samples)} samples")

    # Define latent range for normalization
    b_min, b_max = -1.0, 1.0

    # Process each batch of samples
    for batch_idx, batch in tqdm(enumerate(sample_batches)):
        inputs = (
            batch[0].squeeze(1).to(device) * scale_factor
        )  # Latent [batch, 4, 64, 64]
        labels = batch[1].to(device)  # Condition/mask [batch, 1, 256, 256]

        ############################################################################################################
        # Apply the same preprocessing as in training/validation
        labels_5ch = split_grayscale_to_channels(labels)  # [batch, 5, 256, 256]

        # Scale up inputs from 64x64 to 256x256
        inputs_upscaled = F.interpolate(
            inputs, size=(256, 256), mode="bilinear", align_corners=False
        )
        # inputs_upscaled: [batch, 4, 256, 256]

        # Concatenate along channel dimension (dim=1)
        combined_labels = torch.cat([labels_5ch, inputs_upscaled], dim=1)
        # Result: [batch, 9, 256, 256] (5 + 4 = 9 channels)
        ############################################################################################################

        batch_size = inputs.shape[0]

        # Process samples in the batch
        with torch.no_grad(), torch.amp.autocast("cuda"):
            # For each sample in the batch
            for sample_idx in range(batch_size):
                if batch_idx * batch_size + sample_idx >= num_samples:
                    break

                sample_num = batch_idx * batch_size + sample_idx

                # Extract individual sample - now with 9 channels
                sample_condition = combined_labels[
                    sample_idx : sample_idx + 1
                ].float()  # [1, 9, 256, 256]

                try:
                    # Generate denoised image
                    generated_image = _denoise_sample(
                        unet=unet,
                        controlnet=controlnet,
                        noise_scheduler=noise_scheduler,
                        condition=sample_condition,  # Now 9 channels
                        recon_model=recon_model,
                        device=device,
                    )

                    # Normalize generated image
                    generated_image = torch.clip(generated_image, b_min, b_max).cpu()
                    generated_image = (generated_image - b_min) / (b_max - b_min)

                    # Get original mask (still use original labels for visualization)
                    original_mask = labels[sample_idx : sample_idx + 1].cpu()

                    # Get source paths
                    current_latent_path = val_loader.dataset.oct_paths[sample_num]
                    original_oct_path = current_latent_path.replace(
                        "_oct_latent.pt", "_oct.png"
                    )

                    # Create visualization
                    _create_and_save_visualization(
                        generated_image=generated_image,
                        original_mask=original_mask,
                        original_oct_path=original_oct_path,
                        output_path=f"{val_vis_dir}/sample_{sample_num}.png",
                        sample_num=sample_num,
                        epoch=epoch,
                        rank=rank,
                    )

                except Exception as e:
                    print(
                        f"Error generating visualization for sample {sample_num}: {e}"
                    )
                    continue


def _denoise_sample(unet, controlnet, noise_scheduler, condition, recon_model, device):
    """Denoise a single sample in a single pass"""
    # Initialize with random noise
    latents = initialize_noise_latents((4, 64, 64), device)

    # Get all timesteps from the noise scheduler
    timesteps = noise_scheduler.timesteps

    # Denoise all steps at once
    for i, t in tqdm(enumerate(timesteps)):
        # Get current timestep as tensor
        current_timestep = torch.tensor([t], device=device)

        # Process through ControlNet and UNet
        down_block_res_samples, mid_block_res_sample = controlnet(
            x=latents,
            timesteps=current_timestep,
            controlnet_cond=condition,
        )

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
            # print(f"Completed denoising step {i+1}/{len(timesteps)}")

    # Decode latents to images
    with torch.no_grad():
        generated_image = recon_model(latents)

    # Free latent memory
    del latents
    torch.cuda.empty_cache()

    return generated_image


def _create_and_save_visualization(
    generated_image,
    original_mask,
    original_oct_path,
    output_path,
    sample_num,
    epoch,
    rank,
):
    """Create and save a visualization comparing original OCT, generated image, and mask"""
    # Convert tensors to numpy for matplotlib
    gen_img_np = generated_image.squeeze(0).permute(1, 2, 0).numpy()
    mask_np = original_mask.squeeze(0).squeeze(0).numpy()

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Try to load and plot the original OCT image
    try:
        original_oct_img = Image.open(original_oct_path).convert("L")
        original_oct_np = np.array(original_oct_img)
        axes[0].imshow(original_oct_np, cmap="gray")
        axes[0].set_title("Original OCT Image")
    except Exception as e:
        print(f"Could not load original OCT image from {original_oct_path}: {e}")
        axes[0].text(
            0.5,
            0.5,
            "Original OCT\nnot available",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_title("Original OCT Image")

    axes[0].axis("off")

    # Plot generated image
    axes[1].imshow(gen_img_np, cmap="gray")
    axes[1].set_title("Generated Image")
    axes[1].axis("off")

    # Plot mask
    im = axes[2].imshow(mask_np, cmap="viridis")
    axes[2].set_title("Mask/Condition")
    axes[2].axis("off")

    # Add a colorbar for the mask
    plt.colorbar(im, ax=axes[2], shrink=0.6)

    # Save figure and clean up
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Log to wandb
    if rank == 0 and wandb.run is not None and sample_num < 5:
        wandb.log(
            {
                f"validation/sample_{sample_num}": wandb.Image(
                    output_path, caption=f"Epoch {epoch+1} - Sample {sample_num}"
                )
            }
        )


def _log_validation_to_wandb(metrics, val_vis_dir, epoch, num_samples):
    """Log validation metrics and visualizations to wandb"""
    # Log metrics
    log_dict = {
        "validation/loss": metrics["val_loss"],
        "validation/epoch": epoch + 1,
    }

    if "val_weighted_loss" in metrics and metrics["val_weighted_loss"] is not None:
        log_dict["validation/weighted_loss"] = metrics["val_weighted_loss"]

    wandb.log(log_dict)

    # Log validation sample grid
    if val_vis_dir and os.path.exists(val_vis_dir) and len(os.listdir(val_vis_dir)) > 0:
        sample_images = [
            f"{val_vis_dir}/sample_{i}.png" for i in range(min(5, num_samples))
        ]
        sample_images = [img for img in sample_images if os.path.exists(img)]

        if sample_images:
            wandb.log(
                {"validation/sample_grid": [wandb.Image(img) for img in sample_images]}
            )
