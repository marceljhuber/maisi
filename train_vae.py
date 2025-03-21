import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from monai.config import print_config
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler
from tqdm import tqdm

import wandb
from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.utils import KL_loss
from scripts.utils_data import (
    set_random_seeds,
    setup_transforms,
    setup_dataloaders,
    split_train_val_by_patient,
    list_image_files,
)
from scripts.utils_plot import visualize_2d


def setup_models(config, device):
    """Initialize autoencoder and discriminator models."""
    autoencoder = AutoencoderKlMaisi(**config["model"]["autoencoder"])
    discriminator = PatchDiscriminator(**config["model"]["discriminator"])

    # First move to device, then set precision
    autoencoder = autoencoder.to(device)
    discriminator = discriminator.to(device)

    # If using mixed precision, models should start in float32
    autoencoder = autoencoder.float()
    discriminator = discriminator.float()

    return autoencoder, discriminator


def setup_optimizers(autoencoder, discriminator, config):
    """Setup optimizers and schedulers."""
    optimizer_g = torch.optim.Adam(
        autoencoder.parameters(),
        lr=config["training"]["learning_rate"],
        eps=1e-06 if config["training"]["amp"] else 1e-08,
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=config["training"]["learning_rate"],
        eps=1e-06 if config["training"]["amp"] else 1e-08,
    )

    def warmup_rule(epoch):
        return 1.0

        # # Warmup phase: Start with small learning rate to stabilize training
        # # For 40 epochs total, we use first ~10% (4 epochs) for initial warmup
        # if epoch < 4:
        #     return 0.01  # Initial learning rate: 1% of final rate
        #
        # # Intermediate phase: Gradually increase learning rate
        # # Use next ~15% (6 epochs) for smoother transition
        # elif epoch < 10:
        #     return 0.1  # Intermediate rate: 10% of final rate
        #
        # # Final phase: Use full learning rate for remaining epochs
        # # Remaining ~75% (30 epochs) use full learning rate for optimal training
        # else:
        #     return 1.0  # Full learning rate multiplier

    scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
    scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)

    return optimizer_g, optimizer_d, scheduler_g, scheduler_d


def setup_losses(config, device):
    """Setup loss functions."""
    recon_loss = MSELoss() if config["training"]["recon_loss"] == "l2" else L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    perceptual_loss = (
        PerceptualLoss(
            spatial_dims=2, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
        )
        .eval()
        .to(device)
    )

    return recon_loss, adv_loss, perceptual_loss


def train_step(
    batch,
    autoencoder,
    discriminator,
    optimizer_g,
    optimizer_d,
    losses,
    config,
    device,
    scaler_g=None,
    scaler_d=None,
):
    """Perform single training step."""
    recon_loss, adv_loss, perceptual_loss = losses
    images = batch["image"].to(device).contiguous()

    # Train Generator
    optimizer_g.zero_grad(set_to_none=True)
    with autocast(enabled=config["training"]["amp"]):
        reconstruction, z_mu, z_sigma = autoencoder(images)
        recons_loss = recon_loss(reconstruction, images)
        kl_loss = KL_loss(z_mu, z_sigma)
        p_loss = perceptual_loss(reconstruction.float(), images.float())

        logits_fake = discriminator(reconstruction.contiguous().float())[-1]
        generator_loss = adv_loss(
            logits_fake, target_is_real=True, for_discriminator=False
        )

        loss_g = (
            recons_loss
            + config["training"]["kl_weight"] * kl_loss
            + config["training"]["perceptual_weight"] * p_loss
            + config["training"]["adv_weight"] * generator_loss
        )

    if config["training"]["amp"]:
        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()
    else:
        loss_g.backward()
        optimizer_g.step()

    # Train Discriminator
    optimizer_d.zero_grad(set_to_none=True)
    with autocast(enabled=config["training"]["amp"]):
        logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = adv_loss(
            logits_fake, target_is_real=False, for_discriminator=True
        )
        logits_real = discriminator(images.contiguous().detach())[-1]
        loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        loss_d = (loss_d_fake + loss_d_real) * 0.5

    if config["training"]["amp"]:
        scaler_d.scale(loss_d).backward()
        scaler_d.step(optimizer_d)
        scaler_d.update()
    else:
        loss_d.backward()
        optimizer_d.step()

    return {
        "recons_loss": recons_loss.item(),
        "kl_loss": kl_loss.item(),
        "p_loss": p_loss.item(),
        "gen_loss": generator_loss.item(),  # Adversarial part only
        "total_g_loss": loss_g.item(),  # Total generator loss
        "disc_loss": loss_d.item(),
    }


def validate(autoencoder, discriminator, val_loader, losses, config, device):
    """Perform validation."""
    autoencoder.eval()
    discriminator.eval()
    val_losses = {
        "recons_loss": 0,
        "kl_loss": 0,
        "p_loss": 0,
        "gen_loss": 0,
        "total_g_loss": 0,
        "disc_loss": 0,
    }
    recon_loss, adv_loss, perceptual_loss = losses

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device).float()

            with autocast(enabled=config["training"]["amp"]):
                reconstruction, z_mu, z_sigma = autoencoder(images)

                # Regular losses
                recons_loss = recon_loss(reconstruction, images)
                kl_loss = KL_loss(z_mu, z_sigma)
                p_loss = perceptual_loss(reconstruction, images)

                val_losses["recons_loss"] += recons_loss.item()
                val_losses["kl_loss"] += kl_loss.item()
                val_losses["p_loss"] += p_loss.item()

                # Generator loss
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(
                    logits_fake, target_is_real=True, for_discriminator=False
                )
                val_losses["gen_loss"] += generator_loss.item()

                # Total generator loss
                total_g_loss = (
                    recons_loss
                    + config["training"]["kl_weight"] * kl_loss
                    + config["training"]["perceptual_weight"] * p_loss
                    + config["training"]["adv_weight"] * generator_loss
                )
                val_losses["total_g_loss"] += total_g_loss.item()

                # Discriminator loss
                loss_d_fake = adv_loss(
                    logits_fake, target_is_real=False, for_discriminator=True
                )
                logits_real = discriminator(images.contiguous())[-1]
                loss_d_real = adv_loss(
                    logits_real, target_is_real=True, for_discriminator=True
                )
                val_losses["disc_loss"] += (loss_d_fake + loss_d_real).item() * 0.5

    # Average the losses
    for k in val_losses:
        val_losses[k] /= len(val_loader)

    return val_losses


def save_checkpoint(state, filename, is_best=False):
    """Save model checkpoint."""
    if is_best:
        best_filename = str(filename).replace(".pt", "_best.pt")
        torch.save(state, best_filename)
    else:
        torch.save(state, filename)


def log_metrics(losses, epoch, phase="train"):
    """Prepare metrics for wandb without logging."""
    return {f"{phase}/{k}": v for k, v in losses.items()}


def save_reconstruction_plot(original, reconstruction, epoch, save_dir):
    """Save reconstruction plot and return wandb image."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original, cmap="gray")
    ax1.set_title("Original")
    ax2.imshow(reconstruction, cmap="gray")
    ax2.set_title("Reconstruction")

    save_path = save_dir / f"reconstruction_epoch_{epoch:03d}.png"
    plt.savefig(save_path)
    plt.close()

    return {"reconstructions": wandb.Image(str(save_path))}


def main():
    # Parse arguments and load config
    parser = argparse.ArgumentParser(description="Train VAE-GAN model")
    parser.add_argument("--config", type=str, default="./configs/config.json")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    # Initialize wandb
    wandb.init(
        project="vae-gan-training",
        config=config,
        name=f"{config['main']['jobname']}_{datetime.now().strftime('%Y%m%d_%H%M')}",
    )

    # Setup
    set_random_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = Path(f"./runs/{config['main']['jobname']}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = run_dir / "reconstructions"
    recon_dir.mkdir(exist_ok=True)

    # Setup data
    image_files = list_image_files(config["data"]["image_dir"])
    train_images, val_images = split_train_val_by_patient(image_files)
    print(f"Found {len(train_images)} train images.")
    train_transform, val_transform = setup_transforms()
    train_transform = val_transform  # No train transform #TODO
    train_loader, val_loader = setup_dataloaders(
        train_images, val_images, train_transform, val_transform, config
    )

    # Setup models and training components
    autoencoder, discriminator = setup_models(config, device)
    optimizer_g, optimizer_d, scheduler_g, scheduler_d = setup_optimizers(
        autoencoder, discriminator, config
    )
    losses = setup_losses(config, device)

    # Setup AMP if enabled
    if config["training"]["amp"]:
        scaler_g = GradScaler(init_scale=2.0**8)
        scaler_d = GradScaler(init_scale=2.0**8)
    else:
        scaler_g = scaler_d = None

    # Training loop
    best_val_loss = float("inf")
    start_epoch = 0

    for epoch in range(start_epoch, config["training"]["epochs"]):
        # Training phase
        autoencoder.train()
        discriminator.train()
        train_losses = {
            "recons_loss": 0,
            "kl_loss": 0,
            "p_loss": 0,
            "gen_loss": 0,
            "total_g_loss": 0,
            "disc_loss": 0,
        }

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            step_losses = train_step(
                batch,
                autoencoder,
                discriminator,
                optimizer_g,
                optimizer_d,
                losses,
                config,
                device,
                scaler_g,
                scaler_d,
            )

            for k in train_losses:
                train_losses[k] += step_losses[k]

        # Average losses
        for k in train_losses:
            train_losses[k] /= len(train_loader)

        # Prepare all metrics
        metrics = {}

        # Add training metrics
        metrics.update(log_metrics(train_losses, epoch, "train"))

        # Validation phase
        val_losses = validate(
            autoencoder, discriminator, val_loader, losses, config, device
        )
        metrics.update(log_metrics(val_losses, epoch, "val"))

        # Save visualization
        with torch.no_grad():
            val_batch = next(iter(val_loader))
            val_images = val_batch["image"].to(device).float()

            with autocast(enabled=config["training"]["amp"]):
                reconstruction, _, _ = autoencoder(val_images[:1])
                reconstruction = reconstruction.float()

            vis_original = visualize_2d(val_images[0].cpu())
            vis_recon = visualize_2d(reconstruction[0].cpu())
            metrics.update(
                save_reconstruction_plot(vis_original, vis_recon, epoch, recon_dir)
            )

        # Add epoch number
        metrics["epoch"] = epoch

        # Single wandb log call per epoch
        wandb.log(metrics)

        # Check if best model
        val_total_loss = (
            val_losses["recons_loss"]
            + config["training"]["kl_weight"] * val_losses["kl_loss"]
            + config["training"]["perceptual_weight"] * val_losses["p_loss"]
        )

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "autoencoder_state_dict": autoencoder.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "scheduler_g_state_dict": scheduler_g.state_dict(),
                    "scheduler_d_state_dict": scheduler_d.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                run_dir / "model.pt",
                is_best=True,
            )

            wandb.run.summary["best_val_loss"] = best_val_loss
            wandb.run.summary["best_epoch"] = epoch

        # Regular checkpoint save
        if epoch % config["training"]["save_interval"] == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "autoencoder_state_dict": autoencoder.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_g_state_dict": optimizer_g.state_dict(),
                    "optimizer_d_state_dict": optimizer_d.state_dict(),
                    "scheduler_g_state_dict": scheduler_g.state_dict(),
                    "scheduler_d_state_dict": scheduler_d.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": config,
                },
                run_dir / f"{config['main']['jobname']}_{epoch}.pt",
            )

        # Step schedulers
        scheduler_g.step()
        scheduler_d.step()

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print_config()
    main()
