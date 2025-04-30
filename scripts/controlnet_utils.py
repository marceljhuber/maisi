import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    logger,
    scale_factor=1.0,
    num_samples=20,
    weighted_loss=1.0,
    weighted_loss_label=None,
    rank=0,
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
        logger: Logger instance
        scale_factor: Scale factor for the latent space
        num_samples: Number of validation samples to visualize
        weighted_loss: Weight factor for loss computation on specific regions
        weighted_loss_label: Labels for regions with weighted loss
    """
    # PNG image intensity range
    a_min = 0
    a_max = 255
    # autoencoder output intensity range
    b_min = -1.0
    b_max = 1.0

    autoencoder.eval()
    controlnet.eval()
    unet.eval()

    # Create directory for validation visualizations
    val_vis_dir = os.path.join(save_dir, f"epoch_{epoch + 1}_validation")
    os.makedirs(val_vis_dir, exist_ok=True)

    # Set up inference timesteps
    noise_scheduler.set_timesteps(1000, device=device)

    # Initialize loss tracking variables
    total_loss = 0.0
    total_samples = 0
    all_metrics = {
        "val_loss": 0.0,
        "val_weighted_loss": 0.0 if weighted_loss > 1.0 else None,
        "val_mse": 0.0,
        "val_psnr": 0.0,
    }

    # Get the first num_samples batches for visualization
    sample_batches = []
    sample_count = 0
    for batch_idx, batch in enumerate(val_loader):
        sample_batches.append(batch)
        sample_count += batch[0].shape[0]
        if sample_count >= num_samples:
            break

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(
        device
    )

    logger.info(
        f"Validating on {len(val_loader)} batches and visualizing {len(sample_batches)} batches"
    )

    with torch.no_grad(), torch.amp.autocast("cuda"):
        # First, compute validation loss on the entire validation set
        for batch_idx, batch in enumerate(val_loader):
            inputs = batch[0].squeeze(1).to(device) * scale_factor  # Latent
            labels = batch[1].to(device)  # Condition/mask
            labels = split_grayscale_to_channels(labels)

            # Random timesteps for diffusion process
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (inputs.shape[0],),
                device=device,
            ).long()

            # Create noise
            noise = torch.randn_like(inputs)

            # Create noisy latent
            noisy_latent = noise_scheduler.add_noise(
                original_samples=inputs, noise=noise, timesteps=timesteps
            )

            # Get controlnet output
            down_block_res_samples, mid_block_res_sample = controlnet(
                x=noisy_latent, timesteps=timesteps, controlnet_cond=labels.float()
            )

            # Get noise prediction from diffusion unet
            noise_pred = unet(
                x=noisy_latent,
                timesteps=timesteps,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

            # Compute L1 loss between predicted and actual noise
            batch_loss = F.l1_loss(noise_pred.float(), noise.float(), reduction="none")

            # Apply weighted loss if specified
            if weighted_loss > 1.0 and weighted_loss_label:
                weights = torch.ones_like(inputs).to(inputs.device)
                roi_mask = torch.zeros_like(inputs[:, :1], dtype=torch.float32)

                interpolate_label = F.interpolate(
                    labels, size=inputs.shape[2:], mode="nearest"
                )

                # For each target label, add to the mask
                for label in weighted_loss_label:
                    for channel in range(interpolate_label.shape[1]):
                        # Create mask for this channel/label combination
                        channel_mask = (
                            interpolate_label[:, channel : channel + 1] == label
                        ).float()
                        # Add masks
                        roi_mask = roi_mask + channel_mask

                # Convert to binary mask
                roi_mask = (roi_mask > 0).float()

                # Apply the mask to weights
                weights = weights.masked_fill(
                    roi_mask.repeat(1, inputs.shape[1], 1, 1) > 0, weighted_loss
                )

                weighted_batch_loss = (batch_loss * weights).mean()
                all_metrics["val_weighted_loss"] += (
                    weighted_batch_loss.item() * inputs.shape[0]
                )

            # Calculate final loss
            batch_loss = batch_loss.mean()
            total_loss += batch_loss.item() * inputs.shape[0]
            total_samples += inputs.shape[0]

            # Free up memory
            del noise_pred, noisy_latent, batch_loss
            torch.cuda.empty_cache()

        # Calculate average loss
        all_metrics["val_loss"] = total_loss / total_samples
        if weighted_loss > 1.0 and weighted_loss_label:
            all_metrics["val_weighted_loss"] = (
                all_metrics["val_weighted_loss"] / total_samples
            )

        logger.info(f"Validation Loss: {all_metrics['val_loss']:.6f}")

        # Now, generate visualizations for the sample batches
        for batch_idx, batch in enumerate(sample_batches):
            inputs = batch[0].squeeze(1).to(device) * scale_factor  # Latent
            labels = batch[1].to(device)  # Condition/mask
            labels_channels = split_grayscale_to_channels(labels)

            batch_size = inputs.shape[0]

            # For each sample in the batch
            for sample_idx in range(batch_size):
                if batch_idx * batch_size + sample_idx >= num_samples:
                    break

                # Get single sample
                sample_input = inputs[sample_idx : sample_idx + 1]
                sample_label = labels_channels[sample_idx : sample_idx + 1].float()

                # Generate denoised image
                # First, initialize with random noise
                latents = initialize_noise_latents((4, 64, 64), device)

                # Denoise step by step
                for i, t in enumerate(noise_scheduler.timesteps):
                    # Get timestep
                    timesteps = torch.tensor([t], device=device)

                    # Get controlnet output
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        x=latents.to(torch.float32),
                        timesteps=timesteps,
                        controlnet_cond=sample_label.to(torch.float32),
                    )

                    # Get noise prediction from diffusion unet
                    noise_pred = unet(
                        x=latents.to(torch.float32),
                        timesteps=timesteps,
                        down_block_additional_residuals=[
                            res.to(torch.float32) for res in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(
                            torch.float32
                        ),
                    )

                    # Update latent
                    latents, _ = noise_scheduler.step(noise_pred, t, latents)

                    # Clear CUDA cache every few steps
                    if i % 100 == 0:
                        torch.cuda.empty_cache()

                # Decode latents to images
                generated_image = recon_model(latents)

                # Ensure generated_image is float32 for proper manipulation
                generated_image = generated_image.to(torch.float32)

                # Clip values to valid range and move to CPU
                generated_image = torch.clip(generated_image, b_min, b_max).cpu()

                # Normalize to [0, 1] range for visualization
                generated_image = (generated_image - b_min) / (b_max - b_min)

                # Convert original mask to visualization format
                original_mask = labels[sample_idx : sample_idx + 1].cpu()

                # Create side-by-side visualization
                # Convert tensors to numpy for matplotlib
                gen_img_np = generated_image.squeeze(0).permute(1, 2, 0).numpy()
                mask_np = original_mask.squeeze(0).squeeze(0).numpy()

                # Create figure with two subplots side by side
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                # Plot generated image
                axes[0].imshow(gen_img_np)
                axes[0].set_title("Generated Image")
                axes[0].axis("off")

                # Plot mask
                axes[1].imshow(mask_np, cmap="viridis")
                axes[1].set_title("Mask/Condition")
                axes[1].axis("off")

                plt.tight_layout()

                # Save the visualization
                sample_num = batch_idx * batch_size + sample_idx
                plt.savefig(
                    f"{val_vis_dir}/sample_{sample_num}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close()

                # Log to wandb
                if (
                    rank == 0 and wandb.run is not None and sample_num < 5
                ):  # Only log first 5 to avoid cluttering wandb
                    wandb.log(
                        {
                            f"validation/sample_{sample_num}": wandb.Image(
                                f"{val_vis_dir}/sample_{sample_num}.png",
                                caption=f"Epoch {epoch+1} - Sample {sample_num}",
                            )
                        }
                    )

                # Free up memory
                del generated_image, latents, noise_pred
                torch.cuda.empty_cache()

    # Log metrics to wandb
    if rank == 0 and wandb.run is not None:
        log_dict = {
            "validation/loss": all_metrics["val_loss"],
            "validation/epoch": epoch + 1,
        }

        if all_metrics["val_weighted_loss"] is not None:
            log_dict["validation/weighted_loss"] = all_metrics["val_weighted_loss"]

        wandb.log(log_dict)

        # Also log a grid of the first few validation samples
        if os.path.exists(val_vis_dir) and len(os.listdir(val_vis_dir)) > 0:
            sample_images = [
                f"{val_vis_dir}/sample_{i}.png" for i in range(min(5, num_samples))
            ]
            sample_images = [img for img in sample_images if os.path.exists(img)]

            if sample_images:
                wandb.log(
                    {
                        "validation/sample_grid": [
                            wandb.Image(img) for img in sample_images
                        ]
                    }
                )

    # Return model to training mode
    controlnet.train()

    return all_metrics["val_loss"]
