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

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

import monai
from monai.data import DataLoader, partition_dataset
from monai.transforms import Compose
from tqdm import tqdm
from monai.utils import first

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .utils import define_instance


def load_latents(latents_dir: str) -> list:
    latent_files = sorted(Path(latents_dir).glob("*_latent.pt"))
    return [{"image": str(f)} for f in latent_files]


def prepare_data(train_files, device, cache_rate, num_workers=2, batch_size=1):
    """
    Prepare training data.

    Args:
        train_files (list): List of training files.
        device (torch.device): Device to use for training.
        cache_rate (float): Cache rate for dataset.
        num_workers (int): Number of workers for data loading.
        batch_size (int): Mini-batch size.

    Returns:
        DataLoader: Data loader for training.
    """
    train_transforms = Compose(
        [
            # Custom loader for .pt files
            monai.transforms.Lambdad(keys=["image"], func=lambda x: torch.load(x)),
            monai.transforms.EnsureTyped(keys=["image"], dtype=torch.float32),
        ]
    )

    train_ds = monai.data.CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    return DataLoader(train_ds, num_workers=6, batch_size=batch_size, shuffle=True)


def load_unet(
    args: argparse.Namespace, device: torch.device, logger: logging.Logger
) -> torch.nn.Module:
    """
    Load the UNet model.

    Args:
        args (argparse.Namespace): Configuration arguments.
        device (torch.device): Device to load the model on.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.nn.Module: Loaded UNet model.
    """
    unet = define_instance(args, "diffusion_unet_def").to(device)
    unet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(unet)

    if dist.is_initialized():
        unet = DistributedDataParallel(
            unet, device_ids=[device], find_unused_parameters=True
        )

    if not args.trained_unet_path or args.trained_unet_path == "None":
        logger.info("Training from scratch.")
    else:
        checkpoint_unet = torch.load(args.trained_unet_path, map_location=device)
        if dist.is_initialized():
            unet.module.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        else:
            unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        logger.info(f"Pretrained checkpoint {args.trained_unet_path} loaded.")

    return unet


def calculate_scale_factor(
    train_loader: DataLoader, device: torch.device, logger: logging.Logger
) -> torch.Tensor:
    """
    Calculate the scaling factor for the dataset.

    Args:
        train_loader (DataLoader): Data loader for training.
        device (torch.device): Device to use for calculation.
        logger (logging.Logger): Logger for logging information.

    Returns:
        torch.Tensor: Calculated scaling factor.
    """
    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    logger.info(f"Scaling factor set to {scale_factor}.")

    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    logger.info(f"scale_factor -> {scale_factor}.")
    return scale_factor


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model (torch.nn.Module): Model to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Created optimizer.
    """
    return torch.optim.Adam(params=model.parameters(), lr=lr)


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.PolynomialLR:
    """
    Create learning rate scheduler.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        total_steps (int): Total number of training steps.

    Returns:
        torch.optim.lr_scheduler.PolynomialLR: Created learning rate scheduler.
    """
    return torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=total_steps, power=2.0
    )


def train_one_epoch(
    epoch,
    unet,
    train_loader,
    optimizer,
    lr_scheduler,
    loss_pt,
    scaler,
    scale_factor,
    noise_scheduler,
    num_images_per_batch,
    num_train_timesteps,
    device,
    logger,
    local_rank,
    amp=True,
):
    """
    Train the model for one epoch.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        train_loader (DataLoader): Data loader for training.
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_scheduler (torch.optim.lr_scheduler.PolynomialLR): Learning rate scheduler.
        loss_pt (torch.nn.L1Loss): Loss function.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        scale_factor (torch.Tensor): Scaling factor.
        noise_scheduler (torch.nn.Module): Noise scheduler.
        num_images_per_batch (int): Number of images per batch.
        num_train_timesteps (int): Number of training timesteps.
        device (torch.device): Device to use for training.
        logger (logging.Logger): Logger for logging information.
        local_rank (int): Local rank for distributed training.
        amp (bool): Use automatic mixed precision training.

    Returns:
        torch.Tensor: Training loss for the epoch.
    """
    if local_rank == 0:
        logger.info(f"Epoch {epoch + 1}, lr {optimizer.param_groups[0]['lr']}")

    loss_torch = torch.zeros(2, dtype=torch.float, device=device)
    unet.train()

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=local_rank != 0) as pbar:
        for train_data in pbar:
            optimizer.zero_grad(set_to_none=True)
            latents = train_data["image"].squeeze(1).to(device) * scale_factor

            with autocast("cuda", enabled=amp):
                noise = torch.randn_like(latents, device=device)
                timesteps = torch.randint(
                    0, num_train_timesteps, (latents.shape[0],), device=device
                )
                noisy_latent = noise_scheduler.add_noise(latents, noise, timesteps)
                noise_pred = unet(noisy_latent, timesteps)
                loss = loss_pt(noise_pred.float(), noise.float())

            if amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            lr_scheduler.step()
            loss_torch[0] += loss.item()
            loss_torch[1] += 1.0

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "lr": f'{optimizer.param_groups[0]["lr"]:.2e}',
                }
            )

    return loss_torch


def save_checkpoint(
    epoch,
    unet,
    loss_torch_epoch,
    num_train_timesteps,
    scale_factor,
    ckpt_folder,
    args,
    save_latest=True,
):
    """
    Save checkpoint.

    Args:
        epoch (int): Current epoch number.
        unet (torch.nn.Module): UNet model.
        loss_torch_epoch (float): Training loss for the epoch.
        num_train_timesteps (int): Number of training timesteps.
        scale_factor (torch.Tensor): Scaling factor.
        ckpt_folder (str): Checkpoint folder path.
        args (argparse.Namespace): Configuration arguments.
    """
    unet_state_dict = (
        unet.module.state_dict() if dist.is_initialized() else unet.state_dict()
    )
    checkpoint = {
        "epoch": epoch + 1,
        "loss": loss_torch_epoch,
        "num_train_timesteps": num_train_timesteps,
        "scale_factor": scale_factor,
        "unet_state_dict": unet_state_dict,
    }

    # Save latest checkpoint
    if save_latest:
        torch.save(checkpoint, f"{ckpt_folder}/{args.model_filename}")

    # Save periodic checkpoint
    save_path = Path(ckpt_folder) / "models"
    save_path.mkdir(exist_ok=True)
    torch.save(checkpoint, save_path / f"model_epoch_{epoch + 1}.pt")


def log_metrics(epoch, loss_torch_epoch, optimizer, wandb_run):
    metrics = {
        "train/loss": loss_torch_epoch,
        "train/learning_rate": optimizer.param_groups[0]["lr"],
        "epoch": epoch + 1,
    }
    wandb_run.log(metrics)


def diff_model_train(
    env_config_path,
    model_config_path,
    model_def_path,
    num_gpus=1,
    amp=True,
    start_epoch=0,
    wandb_run=None,
):
    args = load_config(env_config_path, model_config_path, model_def_path)
    local_rank, world_size, device = initialize_distributed(num_gpus)
    logger = setup_logging("training")
    logger.info(f"Using {device} of {world_size}")

    if local_rank == 0:
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    train_files = load_latents("./outputs/latents")

    if dist.is_initialized():
        train_files = partition_dataset(
            data=train_files,
            shuffle=True,
            num_partitions=dist.get_world_size(),
            even_divisible=True,
        )[local_rank]

    train_loader = prepare_data(
        train_files,
        device,
        args.diffusion_unet_train["cache_rate"],
        batch_size=args.diffusion_unet_train["batch_size"],
    )

    unet = load_unet(args, device, logger)
    noise_scheduler = define_instance(args, "noise_scheduler")
    scale_factor = calculate_scale_factor(train_loader, device, logger)
    optimizer = create_optimizer(unet, args.diffusion_unet_train["lr"])

    total_steps = (
        args.diffusion_unet_train["n_epochs"] * len(train_loader.dataset)
    ) / args.diffusion_unet_train["batch_size"]
    lr_scheduler = create_lr_scheduler(optimizer, total_steps)
    loss_pt = torch.nn.L1Loss()
    scaler = GradScaler("cuda")

    if wandb_run:
        wandb_run.config.update(
            {
                "batch_size": args.diffusion_unet_train["batch_size"],
                "learning_rate": args.diffusion_unet_train["lr"],
                "num_epochs": args.diffusion_unet_train["n_epochs"],
                "num_timesteps": args.noise_scheduler["num_train_timesteps"],
                "start_epoch": start_epoch,
            }
        )

    for epoch in range(start_epoch, args.diffusion_unet_train["n_epochs"]):
        loss_torch = train_one_epoch(
            epoch,
            unet,
            train_loader,
            optimizer,
            lr_scheduler,
            loss_pt,
            scaler,
            scale_factor,
            noise_scheduler,
            args.diffusion_unet_train["batch_size"],
            args.noise_scheduler["num_train_timesteps"],
            device,
            logger,
            local_rank,
            amp=amp,
        )

        if torch.cuda.device_count() == 1 or local_rank == 0:
            loss_torch = loss_torch.tolist()
            loss_torch_epoch = loss_torch[0] / loss_torch[1]

            if wandb_run:
                log_metrics(epoch, loss_torch_epoch, optimizer, wandb_run)

            save_checkpoint(
                epoch,
                unet,
                loss_torch_epoch,
                args.noise_scheduler["num_train_timesteps"],
                scale_factor,
                args.model_dir,
                args,
            )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument(
        "--env_config",
        type=str,
        default="./configs/environment_maisi_diff_model_train.json",
        help="Path to environment configuration file",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="./configs/config_maisi_diff_model_train.json",
        help="Path to model training/inference configuration",
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./configs/config_maisi.json",
        help="Path to model definition file",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--no_amp",
        dest="amp",
        action="store_false",
        help="Disable automatic mixed precision training",
    )

    args = parser.parse_args()
    diff_model_train(
        args.env_config, args.model_config, args.model_def, args.num_gpus, args.amp
    )
