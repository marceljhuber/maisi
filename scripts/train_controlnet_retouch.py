# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from scripts.controlnet_utils import validate_and_visualize
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



import argparse
import json
import logging
import os
import random
import sys
import time
import gc
from datetime import timedelta, datetime
from typing import Dict, List, Tuple, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from monai.networks.utils import copy_model_state
from monai.utils import RankFilter

import wandb
from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.controlnet_utils import validate_and_visualize
from scripts.utils_data import (
    create_latent_dataloaders,
    create_cluster_dataloaders,
    create_oct_dataloaders,
    split_grayscale_to_channels,
)
from .utils import define_instance, setup_ddp
from scripts.sample import ReconModel, initialize_noise_latents


class MemoryTracker:
    """Utility class to track memory usage"""

    @staticmethod
    def log_memory_usage(logger, prefix=""):
        """Log current GPU memory usage"""
        if torch.cuda.is_available():
            used_gb = torch.cuda.memory_allocated() / (1024**3)
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"{prefix} GPU Memory: {used_gb:.2f}GB / {total_gb:.2f}GB ({used_gb/total_gb*100:.1f}%)")

    @staticmethod
    def force_memory_cleanup():
        """Aggressively clean memory"""
        gc.collect()
        torch.cuda.empty_cache()

        # Extra aggressive approach on CUDA
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def train_controlnet(
        autoencoder: torch.nn.Module,
        unet: torch.nn.Module,
        controlnet: torch.nn.Module,
        noise_scheduler,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler,
        device: torch.device,
        config: Dict,
        logger: logging.Logger,
        scale_factor: float,
        weighted_loss: float = 1.0,
        weighted_loss_label: Optional[List[int]] = None,
        rank: int = 0,
        world_size: int = 1,
        use_ddp: bool = False,
) -> None:
    """
    Main training loop for ControlNet

    Args:
        autoencoder: AutoencoderKL model
        unet: UNet model
        controlnet: ControlNet model being trained
        noise_scheduler: Noise scheduler
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        device: Device to train on
        config: Configuration dictionary
        logger: Logger instance
        scale_factor: Scale factor for latent space
        weighted_loss: Weight factor for loss computation
        weighted_loss_label: Labels for regions with weighted loss
        rank: Process rank
        world_size: Total number of processes
        use_ddp: Whether using distributed training
    """
    args = config['training']
    n_epochs = args['controlnet_train']['n_epochs']
    scaler = GradScaler("cuda")
    total_step = 0
    best_train_loss = float('inf')
    best_val_loss = float('inf')

    # Create directories for model checkpoints and visualizations
    model_dir = config['environment']['model_dir']
    exp_name = config['environment']['exp_name']
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    vis_dir = os.path.join(model_dir, "visualizations")
    val_vis_dir = os.path.join(model_dir, "validation_visualizations")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(val_vis_dir, exist_ok=True)

    if weighted_loss > 1.0:
        logger.info(f"Applying weighted loss = {weighted_loss} on labels: {weighted_loss_label}")

    # Set models to correct states
    controlnet.train()
    unet.eval()
    autoencoder.eval()

    # Training loop
    logger.info(f"Starting training for {n_epochs} epochs")
    prev_time = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        batch_times = []
        losses = []

        # Log memory state at start of epoch
        MemoryTracker.log_memory_usage(logger, f"Epoch {epoch+1} start:")

        try:
            # Process each batch
            for step, batch in enumerate(train_loader):
                batch_start = time.time()

                # Inside batch loop:
                if step % 5 == 0:
                    grad_norm = 0.0
                    for p in controlnet.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5
                    logger.info(f"Gradient norm: {grad_norm:.4f}")

                try:
                    # Process batch
                    inputs = batch[0].squeeze(1).to(device) * scale_factor
                    labels = batch[1].to(device)
                    labels = split_grayscale_to_channels(labels)

                    # Clear gradients
                    optimizer.zero_grad(set_to_none=True)

                    # Forward pass with mixed precision
                    with autocast("cuda", enabled=True):
                        # Generate random noise
                        noise = torch.randn_like(inputs).to(device)

                        # Get random timesteps
                        timesteps = torch.randint(
                            0, noise_scheduler.num_train_timesteps,
                            (inputs.shape[0],), device=device
                        ).long()

                        # Add noise to inputs
                        noisy_latent = noise_scheduler.add_noise(
                            original_samples=inputs, noise=noise, timesteps=timesteps
                        )

                        # ControlNet forward pass
                        down_block_res_samples, mid_block_res_sample = controlnet(
                            x=noisy_latent, timesteps=timesteps, controlnet_cond=labels.float()
                        )

                        # UNet forward pass
                        noise_pred = unet(
                            x=noisy_latent,
                            timesteps=timesteps,
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        )

                        # Compute loss (with weighted regions if specified)
                        if weighted_loss > 1.0:
                            weights = create_weighted_loss_mask(
                                inputs, labels, weighted_loss, weighted_loss_label
                            )
                            loss = (
                                    F.l1_loss(noise_pred.float(), noise.float(), reduction="none")
                                    * weights
                            ).mean()
                        else:
                            loss = F.l1_loss(noise_pred.float(), noise.float())

                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    torch.cuda.synchronize()
                    scaler.update()

                    # Update learning rate
                    lr_scheduler.step()

                    # Track metrics
                    total_step += 1
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    losses.append(loss.item())
                    epoch_loss += loss.detach().item()

                    # Log progress on rank 0
                    if rank == 0:
                        # Log to wandb periodically
                        if step % 50 == 0 and wandb.run is not None:
                            wandb.log({
                                "train/loss": loss.item(),
                                "train/learning_rate": lr_scheduler.get_last_lr()[0],
                                "train/batch_time": batch_time,
                                "train/step": total_step,
                            })

                        # Print progress
                        if step % 2 == 0:  # Print more frequently to see progress
                            batches_done = step + 1
                            batches_left = len(train_loader) - batches_done
                            time_left = timedelta(seconds=batches_left * batch_time)

                            logger.info(
                                f"[Epoch {epoch+1}/{n_epochs}] [Batch {batches_done}/{len(train_loader)}] "
                                f"[LR: {lr_scheduler.get_last_lr()[0]:.8f}] [Loss: {loss.item():.4f}] "
                                f"ETA: {time_left}"
                            )

                    # Clean up memory
                    del noise_pred, noisy_latent, down_block_res_samples, mid_block_res_sample
                    if step % 5 == 0:  # Clean more frequently
                        MemoryTracker.force_memory_cleanup()

                except Exception as e:
                    logger.error(f"Error processing batch {step} in epoch {epoch+1}: {e}")
                    MemoryTracker.force_memory_cleanup()
                    continue

            # Compute epoch metrics
            avg_batch_time = sum(batch_times) / max(len(batch_times), 1)
            avg_loss = sum(losses) / max(len(losses), 1)
            epoch_loss = epoch_loss / max(len(train_loader), 1)

            # Sync metrics across processes if using DDP
            if use_ddp:
                dist.barrier()
                dist.all_reduce(torch.tensor(epoch_loss, device=device), op=torch.distributed.ReduceOp.AVG)

            # Validation and checkpointing on rank 0
            if rank == 0:
                logger.info(f"Completed epoch {epoch+1} training, starting validation")
                MemoryTracker.log_memory_usage(logger, "Before validation:")

                try:
                    # Determine whether to generate visuals
                    generate_visuals = (epoch % args['generate_every'] == 0)

                    # Run validation
                    logger.info(f"Running validation for epoch {epoch+1}...")
                    val_start_time = time.time()

                    # During epoch loop
                    try:
                        val_loss = validate_and_visualize(
                            autoencoder=autoencoder,
                            unet=unet,
                            controlnet=controlnet.module if world_size > 1 else controlnet,
                            noise_scheduler=noise_scheduler,
                            val_loader=val_loader,
                            device=device,
                            epoch=epoch,
                            save_dir=val_vis_dir,
                            logger=logger,
                            scale_factor=scale_factor,
                            num_samples=5,  # Reduced from 20
                            weighted_loss=weighted_loss,
                            weighted_loss_label=weighted_loss_label,
                            rank=rank,
                            generate_visuals=generate_visuals,
                        )
                        previous_val_loss = val_loss
                    except Exception as e:
                        logger.error(f"Validation error: {e}, continuing to next epoch")
                        val_loss = previous_val_loss

                    val_time = time.time() - val_start_time
                    logger.info(f"Validation completed in {val_time:.2f}s, loss: {val_loss:.6f}")

                except Exception as e:
                    logger.error(f"Error during validation for epoch {epoch+1}: {e}")
                    val_loss = float('inf')

                MemoryTracker.log_memory_usage(logger, "After validation:")

                # Log epoch metrics to wandb
                if wandb.run is not None:
                    wandb.log({
                        "epoch": epoch + 1,
                        "epoch/avg_train_loss": avg_loss,
                        "epoch/avg_val_loss": val_loss,
                        "epoch/avg_batch_time": avg_batch_time,
                        "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch/best_train_loss": min(best_train_loss, avg_loss),
                        "epoch/best_val_loss": min(best_val_loss, val_loss),
                    })

                # Save checkpoint with careful error handling
                try:
                    logger.info(f"Saving checkpoint for epoch {epoch+1}")
                    checkpoint_start = time.time()

                    # Get state dict
                    controlnet_state_dict = (
                        controlnet.module.state_dict() if world_size > 1 else controlnet.state_dict()
                    )

                    # Save regular checkpoint
                    checkpoint_path = os.path.join(checkpoints_dir, f"{exp_name}_{epoch}.pt")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "train_loss": epoch_loss,
                            "val_loss": val_loss,
                            "controlnet_state_dict": controlnet_state_dict,
                        },
                        checkpoint_path,
                    )

                    # Save best validation model if needed
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        logger.info(f"New best validation loss -> {best_val_loss:.6f}")
                        best_val_path = os.path.join(model_dir, f"{exp_name}_best_val.pt")
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "train_loss": epoch_loss,
                                "val_loss": best_val_loss,
                                "controlnet_state_dict": controlnet_state_dict,
                            },
                            best_val_path,
                        )

                    # Save best training model if needed
                    if epoch_loss < best_train_loss:
                        best_train_loss = epoch_loss
                        logger.info(f"New best training loss -> {best_train_loss:.6f}")
                        best_train_path = os.path.join(model_dir, f"{exp_name}_best.pt")
                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "train_loss": best_train_loss,
                                "val_loss": val_loss,
                                "controlnet_state_dict": controlnet_state_dict,
                            },
                            best_train_path,
                        )

                    checkpoint_time = time.time() - checkpoint_start
                    logger.info(f"Checkpoint saving completed in {checkpoint_time:.2f}s")

                except Exception as e:
                    logger.error(f"Error saving checkpoint for epoch {epoch+1}: {e}")

            # Sync processes if using DDP
            if use_ddp:
                torch.cuda.synchronize()  # Make sure CUDA operations are done
                dist.barrier()

            # Final cleanup after epoch
            MemoryTracker.force_memory_cleanup()
            logger.info(f"Completed epoch {epoch+1}/{n_epochs}")

        except Exception as e:
            logger.critical(f"Critical error in epoch {epoch+1}: {e}")
            MemoryTracker.force_memory_cleanup()

            # Try to save emergency checkpoint on rank 0
            if rank == 0:
                try:
                    emergency_path = os.path.join(checkpoints_dir, f"emergency_{exp_name}_{epoch}.pt")
                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "controlnet_state_dict": controlnet.module.state_dict() if world_size > 1 else controlnet.state_dict(),
                        },
                        emergency_path,
                    )
                    logger.info(f"Saved emergency checkpoint to {emergency_path}")
                except Exception as e:
                    logger.error(f"Failed to save emergency checkpoint: {e}")

    # Finish training
    logger.info("Training completed")

    # Clean up wandb if it was initialized on rank 0
    if rank == 0 and wandb.run is not None:
        wandb.finish()


def create_weighted_loss_mask(inputs, labels, weighted_loss, weighted_loss_label):
    """Create a weighted loss mask for specific regions"""
    weights = torch.ones_like(inputs)
    roi_mask = torch.zeros_like(inputs[:, :1], dtype=torch.float32)

    # Interpolate labels to match latent dimensions
    interpolate_label = F.interpolate(labels, size=inputs.shape[2:], mode="nearest")

    # For each target label, add to the mask
    for label in weighted_loss_label:
        for channel in range(interpolate_label.shape[1]):
            channel_mask = (interpolate_label[:, channel:channel+1] == label).float()
            roi_mask = roi_mask + channel_mask

    # Convert to binary mask and apply weights
    roi_mask = (roi_mask > 0).float()
    weights = weights.masked_fill(roi_mask.repeat(1, inputs.shape[1], 1, 1) > 0, weighted_loss)

    return weights


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="maisi.controlnet.training")
    parser.add_argument(
        "--config_path",
        default="./configs/config_CONTROLNET_germany.json",
        help="config json file that stores controlnet settings",
    )
    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of gpus per node"
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Setup logging
    logger = logging.getLogger("maisi.controlnet.training")

    # Setup DDP if using multiple GPUs
    use_ddp = args.gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"World size: {world_size}")

    # Initialize wandb on rank 0
    if rank == 0 and wandb.run is None:
        wandb_config = {
            "environment": config["environment"],
            "model_def": config["model_def"],
            "training": config["training"],
        }
        wandb.init(
            project="controlnet-training",
            name=f"{config['main']['jobname']}_{datetime.now().strftime('%Y_%m%d_%H%M')}",
            config=wandb_config,
            resume="allow",
        )
        logger.info("Initialized wandb for tracking")

    # Extract configuration values
    env_dict = config["environment"]
    model_def_dict = config["model_def"]
    training_dict = config["training"]

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in model_def_dict.items():
        setattr(args, k, v)
    for k, v in training_dict.items():
        setattr(args, k, v)

    # Create data loaders
    logger.info("Creating data loaders")
    train_loader, val_loader = create_oct_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, train_ratio=0.9
    )

    # Setup model directories
    args.model_dir = os.path.join(args.model_dir, "CONTROLNET", args.exp_name)
    os.makedirs(args.model_dir, exist_ok=True)

    # Load autoencoder
    logger.info("Loading autoencoder model")
    if args.trained_autoencoder_path is not None:
        if not os.path.exists(args.trained_autoencoder_path):
            raise ValueError("Autoencoder checkpoint not found.")

        model_config = config["model"]["autoencoder"]
        autoencoder = AutoencoderKlMaisi(**model_config).to(device)

        checkpoint = torch.load(
            args.trained_autoencoder_path,
            map_location=device,
            weights_only=True,
        )
        autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
        autoencoder.eval()
    else:
        logger.warning("No trained autoencoder provided.")
        autoencoder = None

    # Convert autoencoder to float32
    if autoencoder is not None:
        for param in autoencoder.parameters():
            param.data = param.data.to(torch.float32)

    # Load UNet
    logger.info("Loading UNet model")
    unet = define_instance(args, "diffusion_unet_def").to(device)

    # Load pretrained diffusion model
    scale_factor = 1.0
    if args.trained_diffusion_path is not None:
        if not os.path.exists(args.trained_diffusion_path):
            raise ValueError("Diffusion model checkpoint not found.")

        diffusion_model_ckpt = torch.load(
            args.trained_diffusion_path,
            map_location=device,
            weights_only=False,
        )

        unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"])
        scale_factor = diffusion_model_ckpt["scale_factor"]
        logger.info(f"Loaded diffusion model from {args.trained_diffusion_path}")
        logger.info(f"Loaded scale_factor from diffusion model: {scale_factor}")
    else:
        logger.warning("No trained diffusion model provided.")

    # Initialize ControlNet
    logger.info("Initializing ControlNet model")
    controlnet = define_instance(args, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())

    # Load pretrained ControlNet if available
    if args.trained_controlnet_path is not None:
        if not os.path.exists(args.trained_controlnet_path):
            raise ValueError("ControlNet checkpoint not found.")

        controlnet.load_state_dict(
            torch.load(args.trained_controlnet_path, map_location=device)["controlnet_state_dict"]
        )
        logger.info(f"Loaded ControlNet from {args.trained_controlnet_path}")
    else:
        logger.info("Training ControlNet from scratch")

    # Freeze UNet parameters
    for p in unet.parameters():
        p.requires_grad = False

    # Initialize noise scheduler
    noise_scheduler = define_instance(args, "noise_scheduler")

    # Wrap model in DDP if using multiple GPUs
    if use_ddp:
        controlnet = DDP(
            controlnet,
            device_ids=[device],
            output_device=rank,
            find_unused_parameters=True,
        )

    # Setup training configuration
    weighted_loss = args.controlnet_train["weighted_loss"]
    weighted_loss_label = args.controlnet_train["weighted_loss_label"]

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        params=controlnet.parameters(),
        lr=args.controlnet_train["lr"],
        weight_decay=0.01,  # Added weight decay for better regularization
    )

    # Calculate total training steps
    total_steps = (
                          args.controlnet_train["n_epochs"] * len(train_loader.dataset)
                  ) / args.controlnet_train["batch_size"]
    logger.info(f"Total number of training steps: {total_steps}")

    # Initialize learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=total_steps, power=2.0
    )

    # Log memory usage before training
    MemoryTracker.log_memory_usage(logger, "Initial:")

    # Run training loop
    try:
        train_controlnet(
            autoencoder=autoencoder,
            unet=unet,
            controlnet=controlnet,
            noise_scheduler=noise_scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            config=config,
            logger=logger,
            scale_factor=scale_factor,
            weighted_loss=weighted_loss,
            weighted_loss_label=weighted_loss_label,
            rank=rank,
            world_size=world_size,
            use_ddp=use_ddp,
        )
    except Exception as e:
        logger.critical(f"Critical training error: {e}", exc_info=True)

    # Clean up DDP if used
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add file handler for persistent logs
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler(
        f"logs/controlnet_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)

    # Start training
    main()