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

# from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

from scripts.diff_model_setting import setup_logging
from scripts.utils import define_instance
from scripts.sample import ldm_conditional_sample_one_image
import os
from networks.autoencoderkl_maisi import AutoencoderKlMaisi


########################################################################################################################
def load_latents(latents_dir: str) -> list:
    latent_files = sorted(Path(latents_dir).glob("*_latent.pt"))
    return [{"image": str(f)} for f in latent_files]


########################################################################################################################
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


########################################################################################################################
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


########################################################################################################################
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


########################################################################################################################
def calculate_scale_factor(
    train_loader: DataLoader, device: torch.device, logger: logging.Logger
) -> torch.Tensor:
    check_data = first(train_loader)
    z = check_data["image"].to(device)
    scale_factor = 1 / torch.std(z)
    logger.info(f"Scaling factor set to {scale_factor}.")
    return scale_factor


########################################################################################################################
def create_optimizer(model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(params=model.parameters(), lr=lr)


########################################################################################################################
def create_lr_scheduler(
    optimizer: torch.optim.Optimizer, total_steps: int
) -> torch.optim.lr_scheduler.PolynomialLR:
    return torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=total_steps, power=2.0
    )


########################################################################################################################
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


########################################################################################################################
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
    torch.save(
        checkpoint, save_path / f"{run_dir.split('/')[1].split('_')[0]}_{epoch}.pt"
    )


########################################################################################################################
def log_metrics(epoch, loss_torch_epoch, optimizer, wandb_run):
    metrics = {
        "train/loss": loss_torch_epoch,
        "train/learning_rate": optimizer.param_groups[0]["lr"],
        "epoch": epoch + 1,
    }
    wandb_run.log(metrics)


########################################################################################################################
def generate_validation_images(
    autoencoder,
    diffusion_unet,
    noise_scheduler,
    scale_factor,
    device,
    latent_shape,
    output_dir,
    num_images=8,
    num_inference_steps=1000,
):
    """
    Generate a fixed set of validation images using predefined seeds.

    Args:
        autoencoder: The trained autoencoder model
        diffusion_unet: The trained diffusion UNet model
        noise_scheduler: The noise scheduler
        scale_factor: Scaling factor for the latents
        device: The device to run inference on
        latent_shape: Shape of the latent space
        output_dir: Directory to save the generated images
        num_images: Number of images to generate (default: 20)
        num_inference_steps: Number of diffusion steps (default: 1000)
    """
    import os
    from datetime import datetime
    import torch
    from monai.transforms import SaveImage

    # Store current RNG state
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state()

    # Create validation directory
    val_dir = os.path.join(output_dir)
    os.makedirs(val_dir, exist_ok=True)

    # Set fixed seeds for validation
    base_seed = 42
    validation_seeds = [base_seed + i for i in range(num_images)]

    generated_images = []

    for seed_idx, seed in enumerate(validation_seeds):
        # Set seed for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        try:
            # Generate image using the ldm_conditional_sample_one_image function
            synthetic_image = ldm_conditional_sample_one_image(
                autoencoder=autoencoder,
                diffusion_unet=diffusion_unet,
                noise_scheduler=noise_scheduler,
                scale_factor=scale_factor,
                device=device,
                latent_shape=latent_shape,
                noise_factor=1.0,
                num_inference_steps=num_inference_steps,
            )

            # Denormalize from [-1,1] to [0,1]
            synthetic_image = (synthetic_image + 1) / 2.0

            # Convert to uint8 range [0,255]
            synthetic_image = (synthetic_image * 255).type(torch.uint8)

            # Save the generated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_saver = SaveImage(
                output_dir=val_dir,
                output_postfix=f"val_{seed_idx:02d}_seed{seed}_{timestamp}",
                output_ext=".png",
                separate_folder=False,
            )
            img_saver(synthetic_image[0])

            # For return value, keep in normalized form
            generated_images.append((synthetic_image.float() / 255) * 2 - 1)

        except Exception as e:
            print(f"Error generating validation image with seed {seed}: {str(e)}")
            continue

    # Restore original RNG state
    torch.set_rng_state(rng_state)
    torch.cuda.set_rng_state(cuda_rng_state)

    return generated_images


########################################################################################################################
def save_validation_images_after_epoch(
    epoch, run_dir, models_dict, config, wandb_run=None
):
    """
    Generate and save validation images after each epoch.

    Args:
        epoch: Current epoch number
        run_dir: Directory for the current training run
        models_dict: Dictionary containing the models
        config: Configuration dictionary
        wandb_run: Optional wandb run object for logging
    """
    val_dir = os.path.join(run_dir, "validation_images", f"epoch_{epoch}")
    os.makedirs(val_dir, exist_ok=True)

    latent_shape = [
        4,
        64,
        64,
    ]

    # Generate validation images
    generated_images = generate_validation_images(
        autoencoder=models_dict["autoencoder"],
        diffusion_unet=models_dict["diffusion_unet"],
        noise_scheduler=models_dict["noise_scheduler"],
        scale_factor=models_dict["scale_factor"],
        device=models_dict["device"],
        latent_shape=latent_shape,
        output_dir=val_dir,
        num_images=2,
        num_inference_steps=config.get("num_inference_steps", 1000),
    )

    # Log to wandb if available
    if wandb_run is not None:
        for idx, img in enumerate(generated_images):
            # Convert tensor to numpy and normalize for wandb
            img_np = img.cpu().numpy()[0]  # Remove batch dimension
            img_np = (img_np + 1) / 2  # Convert from [-1, 1] to [0, 1]
            wandb_run.log(
                {f"validation_image_{idx}": wandb.Image(img_np), "epoch": epoch}
            )

    return val_dir


########################################################################################################################
def diff_model_train(
    config, run_dir, amp=True, start_epoch=0, wandb_run=None, config_path=None
):
    args = load_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logging("training")
    logger.info(f"Using device: {device}")

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    train_files = load_latents(args.latents_path)
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

    ####################################################################################################################
    # Load config
    with open(config_path) as f:
        config = json.load(f)

    model_config = config["model"]["autoencoder"]

    # Load model
    autoencoder = AutoencoderKlMaisi(**model_config).to(device)
    checkpoint = torch.load(
        config["main"]["trained_autoencoder_path"],
        map_location=device,
        weights_only=True,
    )
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    autoencoder.eval()
    ####################################################################################################################

    # Create a models dictionary
    models_dict = {
        "autoencoder": autoencoder,
        "diffusion_unet": unet,
        "noise_scheduler": noise_scheduler,
        "scale_factor": scale_factor,
        "device": device,
    }

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

        # After each epoch, generate and save validation images
        val_dir = save_validation_images_after_epoch(
            epoch=epoch,
            run_dir=run_dir,
            models_dict=models_dict,
            config=config,
            wandb_run=wandb_run,
        )

        print(f"Validation images for epoch {epoch} saved to: {val_dir}")


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
