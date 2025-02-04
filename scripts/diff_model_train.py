from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import monai
import torch
from monai.data import DataLoader
from monai.transforms import Compose
from monai.utils import first
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from scripts.diff_model_setting import setup_logging
from scripts.utils import define_instance


def load_latents(latents_dir: str) -> list:
    latent_files = sorted(Path(latents_dir).glob("*_latent.pt"))
    return [{"image": str(f)} for f in latent_files]


def load_config(config_path):
    if isinstance(config_path, dict):
        config = config_path
    else:
        with open(config_path) as f:
            config = json.load(f)

    # Merge configs with priority handling
    merged_config = {}
    for section in ["main", "model_config", "env_config", "vae_def"]:
        if section in config:
            merged_config.update(config[section])

    return argparse.Namespace(**merged_config)


def prepare_data(train_files, device, cache_rate, num_workers=2, batch_size=1):
    train_transforms = Compose(
        [
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
    unet = define_instance(args, "diffusion_unet_def").to(device)

    if not args.trained_unet_path or args.trained_unet_path == "None":
        logger.info("Training from scratch.")
    else:
        checkpoint_unet = torch.load(args.trained_unet_path, map_location=device)
        unet.load_state_dict(checkpoint_unet["unet_state_dict"], strict=True)
        logger.info(f"Pretrained checkpoint {args.trained_unet_path} loaded.")

    return unet


def calculate_scale_factor(
    train_loader: DataLoader, device: torch.device, logger: logging.Logger
) -> torch.Tensor:
    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    logger.info(f"Scaling factor set to {scale_factor}.")
    return scale_factor


def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(params=model.parameters(), lr=lr)


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.PolynomialLR:
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
    num_train_timesteps,
    device,
    logger,
    amp=True,
):
    logger.info(f"Epoch {epoch + 1}, lr {optimizer.param_groups[0]['lr']}")
    loss_torch = torch.zeros(2, dtype=torch.float, device=device)
    unet.train()

    with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as pbar:
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
    run_dir,
    args,
    save_latest=True,
):
    checkpoint = {
        "epoch": epoch + 1,
        "loss": loss_torch_epoch,
        "num_train_timesteps": num_train_timesteps,
        "scale_factor": scale_factor,
        "unet_state_dict": unet.state_dict(),
    }

    if save_latest:
        torch.save(checkpoint, f"{run_dir}/models/{args.model_filename}")
        # print(f"Saving latest to:", f"{run_dir}/models/{args.model_filename}")

    save_path = Path(run_dir) / "models"
    save_path.mkdir(exist_ok=True)
    torch.save(checkpoint, save_path / f"model_epoch_{epoch + 1}.pt")
    # print(f"Saving to: ", save_path / f"model_epoch_{epoch + 1}.pt")


def log_metrics(epoch, loss_torch_epoch, optimizer, wandb_run):
    metrics = {
        "train/loss": loss_torch_epoch,
        "train/learning_rate": optimizer.param_groups[0]["lr"],
        "epoch": epoch + 1,
    }
    wandb_run.log(metrics)


def diff_model_train(config_path, run_dir, amp=True, start_epoch=0, wandb_run=None):
    args = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging("training")
    logger.info(f"Using device: {device}")

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    train_files = load_latents("./outputs/latents")
    train_loader = prepare_data(
        train_files,
        # train_files[0:1000],
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
            args.noise_scheduler["num_train_timesteps"],
            device,
            logger,
            amp=amp,
        )

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
            run_dir,
            args,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Training")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config_diff_model_train.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--no_amp",
        dest="amp",
        action="store_false",
        help="Disable automatic mixed precision training",
    )

    args = parser.parse_args()
    diff_model_train(args.config, args.amp)
