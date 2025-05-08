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


def generate_image_grid(
    autoencoder,
    unet,
    controlnet,
    noise_scheduler,
    device,
    epoch,
    save_dir,
    logger,
    scale_factor=1.0,
    num_seeds=10,
    num_classes=5,
):
    """
    Generate a grid of images for visualization after each epoch.

    Args:
        autoencoder: The trained autoencoder model
        unet: The trained diffusion UNet model
        controlnet: The ControlNet model being trained
        noise_scheduler: Noise scheduler for the diffusion process
        device: The device to run inference on
        epoch: Current epoch number
        save_dir: Directory to save the grid image
        scale_factor: Scale factor for the latent space
        num_seeds: Number of different seeds to use (columns)
        num_classes: Number of different classes to visualize (rows)
    """
    # PNG image intensity range
    a_min = 0
    a_max = 255
    # autoencoder output intensity range
    b_min = -1.0
    b_max = 1.0

    noise_factor = 1.0

    autoencoder.eval()
    controlnet.eval()
    unet.eval()

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(
        device
    )

    # Create a directory for visualizations if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Set up inference timesteps
    noise_scheduler.set_timesteps(1000, device=device)

    # Set fixed seeds for reproducibility across epochs
    fixed_seeds = [42, 1337, 7, 13, 999, 123, 456, 789, 101, 202]
    assert len(fixed_seeds) >= num_seeds, "Not enough fixed seeds provided"

    # Initialize grid to store generated images
    all_images = []

    # Create conditions for each class (one-hot encoded)
    # with torch.no_grad(), torch.cuda.amp.autocast():
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for class_idx in range(num_classes):
            logger.info(f"Generating images of class {class_idx}.")
            class_images = []

            # Create condition for this class
            condition = torch.zeros(
                (1, num_classes, 256, 256), dtype=torch.float32, device=device
            )
            condition[0, class_idx] = 1.0  # Set the corresponding class channel to 1

            for seed_idx in range(num_seeds):
                # Set seed for reproducibility
                torch.manual_seed(fixed_seeds[seed_idx])
                random.seed(fixed_seeds[seed_idx])
                np.random.seed(fixed_seeds[seed_idx])

                # Generate random noise
                latents = initialize_noise_latents((4, 64, 64), device) * noise_factor

                # Explicitly convert to float32 before any processing
                latents = latents.float()

                # Denoise step by step
                for i, t in enumerate(noise_scheduler.timesteps):
                    # Get timestep
                    timesteps = torch.tensor([t], device=device)

                    # Get controlnet output
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        x=latents.to(torch.float32),
                        timesteps=timesteps,
                        controlnet_cond=condition.to(torch.float32),
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

                    # Clear CUDA cache every few steps to prevent memory buildup
                    if i % 10 == 0:
                        torch.cuda.empty_cache()

                del noise_pred
                torch.cuda.empty_cache()

                # Decode latents to images
                synthetic_images = recon_model(latents)

                # Ensure synthetic_images is float32 for proper clipping and normalization
                synthetic_images = synthetic_images.to(torch.float32)

                # Clip values to valid range
                synthetic_images = torch.clip(synthetic_images, b_min, b_max).cpu()

                # Normalize images to [0, 1] range for visualization
                synthetic_images = (synthetic_images - b_min) / (b_max - b_min)

                class_images.append(synthetic_images)
                torch.cuda.empty_cache()

            # Collect all images for this class
            all_images.extend(class_images)

    # Create a grid from all generated images
    # Ensure all tensors have the same dtype before creating the grid
    all_images_tensor = torch.cat(all_images, dim=0).float()
    grid = make_grid(all_images_tensor, nrow=num_seeds, normalize=False)

    del all_images, all_images_tensor
    torch.cuda.empty_cache()

    # Convert to numpy for matplotlib
    grid_np = grid.permute(1, 2, 0).numpy()

    # Save the grid
    plt.figure(figsize=(20, 10))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.title(f"Epoch {epoch + 1} - Image Grid (4 classes × 10 seeds)")

    # Add class labels on the left
    for i in range(num_classes):
        plt.text(
            -10,
            i * (grid_np.shape[0] / num_classes) + (grid_np.shape[0] / num_classes) / 2,
            f"Class {i}",
            va="center",
            ha="right",
        )

    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch + 1}_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    controlnet.train()


def main():
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

    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Step 0: configuration
    logger = logging.getLogger("maisi.controlnet.training")
    # whether to use distributed data parallel
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
    logger.info(f"World_size: {world_size}")

    # Initialize wandb only for rank 0 process when using DDP
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

    env_dict = config["environment"]
    model_def_dict = config["model_def"]
    training_dict = config["training"]

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in model_def_dict.items():
        setattr(args, k, v)
    for k, v in training_dict.items():
        setattr(args, k, v)

    # Step 1: Define Data Loader
    train_loader, val_loader = create_oct_dataloaders(
        data_dir=args.data_dir, batch_size=40, num_workers=8, train_ratio=0.9
    )

    args.model_dir = os.path.join(
        args.model_dir,
        "CONTROLNET",
        args.exp_name,
    )
    vis_dir = os.path.join(args.model_dir, f"visualizations")
    val_vis_dir = os.path.join(args.model_dir, f"validation_visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(val_vis_dir, exist_ok=True)
    print(f"Created directories: {vis_dir}, {val_vis_dir}")

    # Step 2: Define Autoencoder, Unet and ControlNet
    ####################################################################################################################
    # Autoencoder
    ####################################################################################################################
    if args.trained_autoencoder_path is not None:
        if not os.path.exists(args.trained_autoencoder_path):
            raise ValueError("Please download the autoencoder checkpoint.")

        model_config = config["model"]["autoencoder"]

        # Load model
        autoencoder = AutoencoderKlMaisi(**model_config).to(device)
        checkpoint = torch.load(
            config["environment"]["trained_autoencoder_path"],
            map_location=device,
            weights_only=True,
        )
        autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
        autoencoder.eval()
    else:
        logger.info("trained autoencoder model is not loaded.")

    # Convert autoencoder parameters to float32
    for param in autoencoder.parameters():
        param.data = param.data.to(torch.float32)
    ####################################################################################################################
    # Unet
    ####################################################################################################################
    unet = define_instance(args, "diffusion_unet_def").to(device)

    if args.trained_diffusion_path is not None:
        if not os.path.exists(args.trained_diffusion_path):
            raise ValueError("Please download the trained diffusion unet checkpoint.")

        diffusion_model_ckpt = torch.load(
            args.trained_diffusion_path,
            map_location=device,
            weights_only=False,
        )

        unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"])
        # load scale factor from diffusion model checkpoint
        scale_factor = diffusion_model_ckpt["scale_factor"]
        logger.info(f"Load trained diffusion model from {args.trained_diffusion_path}.")
        logger.info(f"loaded scale_factor from diffusion model ckpt -> {scale_factor}.")
    else:
        logger.info("trained diffusion model is not loaded.")
        scale_factor = 1.0
        logger.info(f"set scale_factor -> {scale_factor}.")
    ####################################################################################################################
    # ControlNet
    ####################################################################################################################
    controlnet = define_instance(args, "controlnet_def").to(device)
    copy_model_state(controlnet, unet.state_dict())

    if args.trained_controlnet_path is not None:
        if not os.path.exists(args.trained_controlnet_path):
            raise ValueError("Please download the trained ControlNet checkpoint.")
        controlnet.load_state_dict(
            torch.load(args.trained_controlnet_path, map_location=device)[
                "controlnet_state_dict"
            ]
        )
        logger.info(
            f"load trained controlnet model from {args.trained_controlnet_path}"
        )
    else:
        logger.info("train controlnet model from scratch.")

    # we freeze the parameters of the diffusion model.
    for p in unet.parameters():
        p.requires_grad = False
    ####################################################################################################################

    noise_scheduler = define_instance(args, "noise_scheduler")

    if use_ddp:
        controlnet = DDP(
            controlnet,
            device_ids=[device],
            output_device=rank,
            find_unused_parameters=True,
        )

    # Step 3: Training config
    weighted_loss = args.controlnet_train["weighted_loss"]
    weighted_loss_label = args.controlnet_train["weighted_loss_label"]
    optimizer = torch.optim.AdamW(
        params=controlnet.parameters(), lr=args.controlnet_train["lr"]
    )
    total_steps = (
        args.controlnet_train["n_epochs"] * len(train_loader.dataset)
    ) / args.controlnet_train["batch_size"]
    logger.info(f"total number of training steps: {total_steps}.")

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=total_steps, power=2.0
    )

    # Step 4: Training
    n_epochs = args.controlnet_train["n_epochs"]
    # scaler = GradScaler()
    scaler = GradScaler("cuda")
    total_step = 0
    best_train_loss = 1e4
    best_val_loss = 1e4

    if weighted_loss > 1.0:
        logger.info(
            f"apply weighted loss = {weighted_loss} on labels: {weighted_loss_label}"
        )

    controlnet.train()
    unet.eval()
    prev_time = time.time()
    for epoch in range(n_epochs):
        epoch_loss_ = 0
        batch_times = []
        losses = []
        for step, batch in enumerate(train_loader):
            batch_start = time.time()

            inputs = batch[0].squeeze(1).to(device) * scale_factor  # Latent
            labels = batch[1].to(device)  # Latent

            labels = split_grayscale_to_channels(labels)

            # print(f"inputs.shape:", inputs.shape)
            # print(f"labels.shape:", labels.shape)

            optimizer.zero_grad(set_to_none=True)

            # with autocast(enabled=True):
            with autocast("cuda", enabled=True):
                noise_shape = list(inputs.shape)
                noise = torch.randn(noise_shape, dtype=inputs.dtype).to(device)

                controlnet_cond = labels.float()

                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (inputs.shape[0],),
                    device=device,
                ).long()

                # create noisy latent
                noisy_latent = noise_scheduler.add_noise(
                    original_samples=inputs, noise=noise, timesteps=timesteps
                )

                # get controlnet output
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=noisy_latent, timesteps=timesteps, controlnet_cond=controlnet_cond
                )
                # get noise prediction from diffusion unet
                noise_pred = unet(
                    x=noisy_latent,
                    timesteps=timesteps,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

            ##########################################################################
            if weighted_loss > 1.0:
                weights = torch.ones_like(inputs).to(inputs.device)

                # Create a mask in float32 format
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
                        # Add masks instead of OR operation
                        roi_mask = roi_mask + channel_mask

                # Convert to binary mask (any positive value becomes 1)
                roi_mask = (roi_mask > 0).float()

                # Apply the mask to weights
                weights = weights.masked_fill(
                    roi_mask.repeat(1, inputs.shape[1], 1, 1) > 0, weighted_loss
                )

                loss = (
                    F.l1_loss(noise_pred.float(), noise.float(), reduction="none")
                    * weights
                ).mean()
            else:
                loss = F.l1_loss(noise_pred.float(), noise.float())
            ##########################################################################

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_step += 1

            # Calculate batch processing time
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            losses.append(loss.item())

            if rank == 0:
                # Log metrics to wandb every 50 steps
                if step % 50 == 0 and wandb.run is not None:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/learning_rate": lr_scheduler.get_last_lr()[0],
                            "train/batch_time": batch_time,
                            "train/step": total_step,
                        }
                    )

                batches_done = step + 1
                batches_left = len(train_loader) - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                if (step - 1) % 500 == 0:
                    logger.info(
                        "\r[Epoch %d/%d] [Batch %d/%d] [LR: %.8f] [loss: %.4f] ETA: %s "
                        % (
                            epoch + 1,
                            n_epochs,
                            step + 1,
                            len(train_loader),
                            lr_scheduler.get_last_lr()[0],
                            loss.detach().cpu().item(),
                            time_left,
                        )
                    )
            epoch_loss_ += loss.detach()

        epoch_loss = epoch_loss_ / (step + 1)
        avg_batch_time = sum(batch_times) / len(batch_times)
        avg_loss = sum(losses) / len(losses)

        if use_ddp:
            dist.barrier()
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.AVG)

        # Run validation after each epoch
        if rank == 0:
            logger.info(f"Running validation for epoch {epoch + 1}...")

            if epoch % args.generate_every == 0:
                generate_visuals = True
            else:
                generate_visuals = False

            # Use our new validation function
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
                num_samples=5,  # Visualize 20 validation samples
                weighted_loss=weighted_loss,
                weighted_loss_label=weighted_loss_label,
                rank=rank,
                generate_visuals=generate_visuals,
            )

            logger.info(f"Validation loss for epoch {epoch + 1}: {val_loss:.6f}")

            # Log to wandb
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "epoch/avg_train_loss": avg_loss,
                        "epoch/avg_val_loss": val_loss,
                        "epoch/avg_batch_time": avg_batch_time,
                        "epoch/learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch/best_train_loss": min(best_train_loss, avg_loss),
                        "epoch/best_val_loss": min(best_val_loss, val_loss),
                    }
                )

            # save controlnet only on master GPU (rank 0)
            controlnet_state_dict = (
                controlnet.module.state_dict()
                if world_size > 1
                else controlnet.state_dict()
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "train_loss": epoch_loss,
                    "val_loss": val_loss,
                    "controlnet_state_dict": controlnet_state_dict,
                },
                f"{args.model_dir}/{args.exp_name}_{epoch}.pt",
            )

            # Log model checkpoint to wandb
            if wandb.run is not None:
                wandb.save(f"{args.model_dir}/{args.exp_name}_{epoch}.pt")

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"New best validation loss -> {best_val_loss}.")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "train_loss": epoch_loss,
                        "val_loss": best_val_loss,
                        "controlnet_state_dict": controlnet_state_dict,
                    },
                    f"{args.model_dir}/{args.exp_name}_best_val.pt",
                )

                # Log best validation model to wandb
                if wandb.run is not None:
                    wandb.save(f"{args.model_dir}/{args.exp_name}_best_val.pt")

            # Also save best model based on training loss for backward compatibility
            if epoch_loss < best_train_loss:
                best_train_loss = epoch_loss
                logger.info(f"New best training loss -> {best_train_loss}.")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "train_loss": best_train_loss,
                        "val_loss": val_loss,
                        "controlnet_state_dict": controlnet_state_dict,
                    },
                    f"{args.model_dir}/{args.exp_name}_best.pt",
                )

                # Log best model to wandb
                if wandb.run is not None:
                    wandb.save(f"{args.model_dir}/{args.exp_name}_best.pt")

        torch.cuda.empty_cache()

    # Close wandb run if it was initialized
    if rank == 0 and wandb.run is not None:
        wandb.finish()

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
