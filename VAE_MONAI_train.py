import argparse
import glob
import json
import os
import random
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from monai.config import print_config
from monai.data import DataLoader
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from scripts.utils import KL_loss, dynamic_infer
from scripts.utils_plot import (
    visualize_2d,
)

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print_config()

###################################################################################################
# RANDOM SEEDS
###################################################################################################
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # CUDA
torch.cuda.manual_seed_all(seed)  # multiple GPUs
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
###################################################################################################


###################################################################################################
# UTILS
###################################################################################################
def list_image_files(directory_path):
    # Define common image file extensions
    image_extensions = (".jpg", ".jpeg", ".png")

    # Use glob to find all files in directory and subdirectories
    files = glob.glob(os.path.join(directory_path, "**", "*.*"), recursive=True)

    # Filter files for image extensions
    image_files = [file for file in files if file.lower().endswith(image_extensions)]

    return image_files


# Specify the directory path to search
directory_path = "/optima/exchange/mhuber/KermanyV3_resized/train"

# Get list of image files
image_files = list_image_files(directory_path)


def split_train_val_by_patient(image_names, train_ratio=0.9):
    # Extract unique patient IDs
    patient_ids = set(name.split("-")[1] for name in image_names)

    # Random split of patient IDs
    num_train = int(len(patient_ids) * train_ratio)
    train_patients = set(random.sample(list(patient_ids), num_train))

    # Split images based on patient IDs
    train_images = [img for img in image_names if img.split("-")[1] in train_patients]
    val_images = [img for img in image_names if img.split("-")[1] not in train_patients]

    return train_images, val_images


train_imgs, val_imgs = split_train_val_by_patient(image_files)

# Print the image file paths
if image_files:
    print(f"Found {len(image_files)} image(s)")
    print(
        f"Split into {len(train_imgs)} train images and {len(val_imgs)} valid images."
    )
else:
    print("No image files found.")
###################################################################################################


###################################################################################################
# GENERAL CONFIG
###################################################################################################
config = {
    "latent_dim": 128,
    "channels": (32, 64, 128, 256),  # Encoder/Decoder channels
    "strides": (2, 2, 2, 2),  # Encoder/Decoder strides
    "disc_channels": 64,  # Base channels for discriminator
    "disc_layers": 3,  # Number of layers in discriminator
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 100,
    "num_workers": 4,
    "kl_weight": 0.001,  # Weight for KL divergence loss
    "perceptual_weight": 0.1,  # Weight for perceptual loss
    "adv_weight": 0.01,  # Weight for adversarial loss
    "log_interval": 10,  # Steps between logging
    "save_interval": 5,  # Epochs between saving checkpoints
    "amp": True,  # Use Automatic Mixed Precision
    "train_images": train_imgs,  # List of training image paths
    "val_images": val_imgs,  # List of validation image paths
    #    'dataset': GrayscaleDataset,
    #    'transform': train_transform_2,
    "JOBNAME": "VAEGAN_AUG_Z_W",
}
###################################################################################################


###################################################################################################
# READ ENVIRONMENT SETTINGS
###################################################################################################
args = argparse.Namespace()

environment_file = "backup/configs_old/environment_maisi_vae_train.json"
env_dict = json.load(open(environment_file, "r"))
for k, v in env_dict.items():
    setattr(args, k, v)
    print(f"{k}: {v}")

# model path
timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
args.run_dir = f"./runs/{timestamp}_{config['JOBNAME']}"

# Create reconstructions directory
recon_dir = os.path.join(args.run_dir, "reconstructions")
os.makedirs(recon_dir, exist_ok=True)

# Path(args.run_dir).mkdir(parents=True, exist_ok=True)
trained_g_path = os.path.join(args.run_dir, "autoencoder.pt")
trained_d_path = os.path.join(args.run_dir, "discriminator.pt")
print(f"Trained model will be saved as {trained_g_path} and {trained_d_path}.")

# initialize tensorboard writer
Path(args.tfevent_path).mkdir(parents=True, exist_ok=True)
tensorboard_path = os.path.join(args.tfevent_path, "autoencoder")
Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
tensorboard_writer = SummaryWriter(tensorboard_path)
print(f"Tensorboard event will be saved as {tensorboard_path}.\n")
###################################################################################################


###################################################################################################
# READ CONFIGS
###################################################################################################
config_file = "backup/configs_old/config_maisi.json"
config_dict = json.load(open(config_file, "r"))
for k, v in config_dict.items():
    setattr(args, k, v)

# check the format of inference inputs
config_train_file = "backup/configs_old/config_maisi_vae_train_v3.json"
config_train_dict = json.load(open(config_train_file, "r"))
for k, v in config_train_dict["data_option"].items():
    setattr(args, k, v)
    print(f"{k}: {v}")
for k, v in config_train_dict["autoencoder_train"].items():
    setattr(args, k, v)
    print(f"{k}: {v}")

print("Network definition and training hyperparameters have been loaded.")
###################################################################################################


###################################################################################################
# MODEL VAE
###################################################################################################
from networks.autoencoderkl_maisi import AutoencoderKlMaisi

autoencoder = AutoencoderKlMaisi(
    spatial_dims=2,  # 2 for 2D images
    in_channels=1,  # 1 for grayscale
    out_channels=1,  # 1 for grayscale
    latent_channels=4,  # Replace with your latent_channels
    num_channels=[64, 128, 256],
    num_res_blocks=[2, 2, 2],
    norm_num_groups=32,
    norm_eps=1e-6,
    attention_levels=[False, False, False],
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
    use_checkpointing=False,
    use_convtranspose=False,
    norm_float16=True,
    num_splits=1,  # 1 from the maisi notebook
    dim_split=1,
).to(device)
###################################################################################################


###################################################################################################
# MODEL DISCRIMINATOR
###################################################################################################
from monai.networks.nets import PatchDiscriminator

discriminator_norm = "INSTANCE"
discriminator = PatchDiscriminator(
    spatial_dims=args.spatial_dims,
    num_layers_d=3,
    channels=32,
    in_channels=1,
    out_channels=1,
    norm=discriminator_norm,
).to(device)


###################################################################################################


###################################################################################################
# DATASET
###################################################################################################
class GrayscaleDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return {"image": image}


class PreloadedGrayscaleDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.transform = transform
        self.images = []

        for path in tqdm(image_paths, desc="Loading images"):
            img = Image.open(path).convert("L")
            if transform:
                img = transform(img)
            self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx]}


###################################################################################################


###################################################################################################
# TRAINING CONFIG
###################################################################################################
# config loss and loss weight
if args.recon_loss == "l2":
    loss_intensity = MSELoss()
    print("Use l2 loss")
else:
    loss_intensity = L1Loss(reduction="mean")
    print("Use l1 loss")
loss_adv = PatchAdversarialLoss(criterion="least_squares")

loss_perceptual = (
    PerceptualLoss(
        spatial_dims=2, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2
    )
    .eval()
    .to(device)
)

# config optimizer and lr scheduler
optimizer_g = torch.optim.Adam(
    params=autoencoder.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08
)
optimizer_d = torch.optim.Adam(
    params=discriminator.parameters(), lr=args.lr, eps=1e-06 if args.amp else 1e-08
)


# please adjust the learning rate warmup rule based on your dataset and n_epochs
def warmup_rule(epoch):
    # learning rate warmup rule
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.1
    else:
        return 1.0


scheduler_g = lr_scheduler.LambdaLR(optimizer_g, lr_lambda=warmup_rule)
scheduler_d = lr_scheduler.LambdaLR(optimizer_d, lr_lambda=warmup_rule)

# set AMP scaler
if args.amp:
    # test use mean reduction for everything
    scaler_g = GradScaler(init_scale=2.0**8, growth_factor=1.5)
    scaler_d = GradScaler(init_scale=2.0**8, growth_factor=1.5)
###################################################################################################


###################################################################################################
# TRAINING SETUP
###################################################################################################
class SpeckleNoise:
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std

    def __call__(self, img):
        noise = torch.randn_like(img) * self.noise_std
        return img * (1 + noise)


# Setup transforms
train_transform_aug_zscore = transforms.Compose(
    [
        transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        SpeckleNoise(0.1),
        transforms.Normalize(mean=[0.2100], std=[0.0300]),
    ]
)
val_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2100], std=[0.0300]),
    ]
)

config["transform"] = train_transform_aug_zscore
config["dataset"] = GrayscaleDataset

# Create datasets and dataloaders
dataset_train = config["dataset"](config["train_images"], transform=config["transform"])
dataset_val = config["dataset"](config["val_images"], transform=val_transform)

dataloader_train = DataLoader(
    dataset_train,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
)
###################################################################################################


###################################################################################################
# TRAINING
###################################################################################################
# Initialize variables
val_interval = args.val_interval
best_val_recon_epoch_loss = 10000000.0
total_step = 0
start_epoch = 0
max_epochs = args.n_epochs

# Setup validation inferer
val_inferer = (
    SlidingWindowInferer(
        roi_size=args.val_sliding_window_patch_size,
        sw_batch_size=1,
        progress=False,
        overlap=0.0,
        device=device,  # changed to gpu from cpu
        sw_device=device,
    )
    if args.val_sliding_window_patch_size
    else SimpleInferer()
)


def loss_weighted_sum(losses):
    return (
        losses["recons_loss"]
        + args.kl_weight * losses["kl_loss"]
        + args.perceptual_weight * losses["p_loss"]
    )


# Training and validation loops
for epoch in range(start_epoch, max_epochs):
    print("lr:", scheduler_g.get_lr())
    autoencoder.train()
    discriminator.train()
    train_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}

    for batch in tqdm(dataloader_train):
        images = batch["image"].to(device).contiguous()
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            # Train Generator
            reconstruction, z_mu, z_sigma = autoencoder(images)
            losses = {
                "recons_loss": loss_intensity(reconstruction, images),
                "kl_loss": KL_loss(z_mu, z_sigma),
                "p_loss": loss_perceptual(reconstruction.float(), images.float()),
            }
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = loss_adv(
                logits_fake, target_is_real=True, for_discriminator=False
            )
            loss_g = loss_weighted_sum(losses) + args.adv_weight * generator_loss

            if args.amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                optimizer_g.step()

            # Train Discriminator
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = loss_adv(
                logits_fake, target_is_real=False, for_discriminator=True
            )
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = loss_adv(
                logits_real, target_is_real=True, for_discriminator=True
            )
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            if args.amp:
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                loss_d.backward()
                optimizer_d.step()

        # Log training loss
        total_step += 1
        for loss_name, loss_value in losses.items():
            tensorboard_writer.add_scalar(
                f"train_{loss_name}_iter", loss_value.item(), total_step
            )
            train_epoch_losses[loss_name] += loss_value.item()
        tensorboard_writer.add_scalar("train_adv_loss_iter", generator_loss, total_step)
        tensorboard_writer.add_scalar("train_fake_loss_iter", loss_d_fake, total_step)
        tensorboard_writer.add_scalar("train_real_loss_iter", loss_d_real, total_step)

    scheduler_g.step()
    scheduler_d.step()
    for key in train_epoch_losses:
        train_epoch_losses[key] /= len(dataloader_train)
    formatted_losses = {k: f"{v:.6f}" for k, v in train_epoch_losses.items()}
    print(
        f"Epoch {epoch} train_vae_loss {loss_weighted_sum(train_epoch_losses):.6f}: {formatted_losses}."
    )
    for loss_name, loss_value in train_epoch_losses.items():
        tensorboard_writer.add_scalar(f"train_{loss_name}_epoch", loss_value, epoch)
    torch.save(autoencoder.state_dict(), trained_g_path)
    torch.save(discriminator.state_dict(), trained_d_path)

    # Validation
    if epoch % val_interval == 0:
        autoencoder.eval()
        val_epoch_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0}
        val_loader_iter = iter(dataloader_val)
        for batch in dataloader_val:
            with torch.no_grad():
                with autocast(enabled=args.amp):
                    images = batch["image"].to(device)  # Move to device here
                    reconstruction, _, _ = dynamic_infer(
                        val_inferer, autoencoder, images
                    )
                    val_epoch_losses["recons_loss"] += loss_intensity(
                        reconstruction, images.to(device)
                    ).item()
                    val_epoch_losses["kl_loss"] += KL_loss(z_mu, z_sigma).item()
                    val_epoch_losses["p_loss"] += loss_perceptual(
                        reconstruction, images.to(device)
                    ).item()

        for key in val_epoch_losses:
            val_epoch_losses[key] /= len(dataloader_val)

        val_loss_g = loss_weighted_sum(val_epoch_losses)
        formatted_losses = {k: f"{v:.6f}" for k, v in val_epoch_losses.items()}
        print(f"Epoch {epoch} val_vae_loss {val_loss_g:.6f}: {formatted_losses}.")

        if val_loss_g < best_val_recon_epoch_loss:
            best_val_recon_epoch_loss = val_loss_g
            trained_g_path_epoch = f"{trained_g_path[:-3]}_epoch{epoch}.pt"
            torch.save(autoencoder.state_dict(), trained_g_path_epoch)
            print("    Got best val vae loss.")
            print("    Save trained autoencoder to", trained_g_path_epoch)

        for loss_name, loss_value in val_epoch_losses.items():
            tensorboard_writer.add_scalar(loss_name, loss_value, epoch)

        # Monitor reconstruction
        scale_factor_sample = 1.0 / z_mu.flatten().std()
        tensorboard_writer.add_scalar(
            "val_one_sample_scale_factor", scale_factor_sample, epoch
        )

        vis_image = visualize_2d(images)
        vis_recon = visualize_2d(reconstruction)

        tensorboard_writer.add_image(
            "val_orig_img", torch.from_numpy(vis_image)[None], epoch
        )
        tensorboard_writer.add_image(
            "val_recon_img", torch.from_numpy(vis_recon)[None], epoch
        )

        # Save comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(vis_image, cmap="gray")
        ax1.set_title("Original")
        ax2.imshow(vis_recon, cmap="gray")
        ax2.set_title("Reconstruction")
        plt.savefig(
            f"{args.run_dir}/reconstructions/reconstruction_epoch_{epoch:03d}.png"
        )
        plt.close()
###################################################################################################


###################################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Train VAE-GAN model")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--jobname", type=str, default="VAEGAN", help="Name for this training run"
    )
    return parser.parse_args()


def setup_training_dirs(jobname, checkpoint_path=None):
    """Sets up training directories and handles checkpoint loading"""
    # Create timestamp for unique run directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    run_dir = Path(f"./runs/{timestamp}_{jobname}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    recon_dir = run_dir / "reconstructions"
    recon_dir.mkdir(exist_ok=True)

    # Set up model paths
    trained_g_path = run_dir / "autoencoder.pt"
    trained_d_path = run_dir / "discriminator.pt"

    # Initialize starting epoch
    start_epoch = 0
    if checkpoint_path is not None:
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        # Extract epoch number from checkpoint filename
        checkpoint_name = Path(checkpoint_path).name
        if "epoch" in checkpoint_name:
            try:
                start_epoch = int(checkpoint_name.split("epoch")[-1].split(".")[0])
            except ValueError:
                print(
                    "Could not parse epoch number from checkpoint filename, starting from epoch 0"
                )

    return start_epoch, run_dir, trained_g_path, trained_d_path


def load_config(config_path):
    """Loads and validates configuration from JSON file"""
    with open(config_path) as f:
        config = json.load(f)

    required_keys = ["training", "model", "data"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Config file missing required section: {key}")

    return config


def load_checkpoints(autoencoder, discriminator, checkpoint_path):
    """Loads model weights from checkpoint"""
    if checkpoint_path is None:
        return

    checkpoint = torch.load(checkpoint_path)
    if "autoencoder" in checkpoint:
        autoencoder.load_state_dict(checkpoint["autoencoder"])
    if "discriminator" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminator"])
    print(f"Loaded checkpoint from {checkpoint_path}")


def save_checkpoint(epoch, autoencoder, discriminator, path, is_best=False):
    """Saves model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "autoencoder": autoencoder.state_dict(),
        "discriminator": discriminator.state_dict(),
    }

    # Save regular checkpoint
    torch.save(checkpoint, path)

    # If this is the best model so far, save a copy
    if is_best:
        best_path = path.parent / f"best_model_epoch{epoch}.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best model checkpoint to {best_path}")


###################################################################################################


###################################################################################################
def train_epoch(
    autoencoder, discriminator, optimizer_g, optimizer_d, train_loader, config, device
):
    """Performs one epoch of training"""
    autoencoder.train()
    discriminator.train()
    train_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0, "total": 0}

    for batch in tqdm(train_loader):
        images = batch["image"].to(device).contiguous()
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)

        with autocast(enabled=config["training"]["amp"]):
            # Train Generator
            reconstruction, z_mu, z_sigma = autoencoder(images)
            losses = {
                "recons_loss": loss_intensity(reconstruction, images),
                "kl_loss": KL_loss(z_mu, z_sigma),
                "p_loss": loss_perceptual(reconstruction.float(), images.float()),
            }
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = loss_adv(
                logits_fake, target_is_real=True, for_discriminator=False
            )
            loss_g = (
                losses["recons_loss"]
                + config["training"]["kl_weight"] * losses["kl_loss"]
                + config["training"]["perceptual_weight"] * losses["p_loss"]
                + config["training"]["adv_weight"] * generator_loss
            )

            if config["training"]["amp"]:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(optimizer_g)
                scaler_g.step(optimizer_g)
                scaler_g.update()
            else:
                loss_g.backward()
                optimizer_g.step()

            # Train Discriminator
            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = loss_adv(
                logits_fake, target_is_real=False, for_discriminator=True
            )
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = loss_adv(
                logits_real, target_is_real=True, for_discriminator=True
            )
            loss_d = (loss_d_fake + loss_d_real) * 0.5

            if config["training"]["amp"]:
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()
            else:
                loss_d.backward()
                optimizer_d.step()

        # Update running losses
        for loss_name, loss_value in losses.items():
            train_losses[loss_name] += loss_value.item()
        train_losses["total"] += loss_g.item()

    # Average losses over dataset
    for key in train_losses:
        train_losses[key] /= len(train_loader)

    return train_losses


###################################################################################################


###################################################################################################
def validate_epoch(autoencoder, discriminator, val_loader, config, device):
    """Performs validation for one epoch"""
    autoencoder.eval()
    val_losses = {"recons_loss": 0, "kl_loss": 0, "p_loss": 0, "total": 0}

    with torch.no_grad():
        for batch in val_loader:
            with autocast(enabled=config["training"]["amp"]):
                images = batch["image"].to(device)
                reconstruction, z_mu, z_sigma = autoencoder(images)

                # Calculate losses
                recons_loss = loss_intensity(reconstruction, images).item()
                kl_loss = KL_loss(z_mu, z_sigma).item()
                p_loss = loss_perceptual(reconstruction, images).item()

                # Update running losses
                val_losses["recons_loss"] += recons_loss
                val_losses["kl_loss"] += kl_loss
                val_losses["p_loss"] += p_loss
                val_losses["total"] += (
                    recons_loss
                    + config["training"]["kl_weight"] * kl_loss
                    + config["training"]["perceptual_weight"] * p_loss
                )

    # Average losses over dataset
    for key in val_losses:
        val_losses[key] /= len(val_loader)

    return val_losses


###################################################################################################


###################################################################################################
def log_validation_metrics(
    val_losses, writer, epoch, run_dir, autoencoder, val_loader, device
):
    """
    Logs validation metrics to tensorboard and saves visualization

    Args:
        val_losses: Dictionary containing validation losses
        writer: Tensorboard writer
        epoch: Current epoch number
        run_dir: Directory to save outputs
        autoencoder: The autoencoder model
        val_loader: Validation data loader
        device: Torch device to use
    """
    # Log losses
    for loss_name, loss_value in val_losses.items():
        writer.add_scalar(f"val_{loss_name}", loss_value, epoch)

    # Get a single batch from validation loader for visualization
    val_iter = iter(val_loader)
    val_batch = next(val_iter)
    val_images = val_batch["image"]

    # Generate and save reconstructions
    with torch.no_grad():
        # Get reconstruction for first image
        reconstruction, z_mu, z_sigma = autoencoder(val_images[:1].to(device))

        # Convert to CPU for visualization
        val_img = val_images[0].cpu()
        recon_img = reconstruction[0].cpu()

        # Create visualizations
        vis_original = visualize_2d(val_img)
        vis_recon = visualize_2d(recon_img)

        # Log to tensorboard
        writer.add_image("val_original", torch.from_numpy(vis_original)[None], epoch)
        writer.add_image("val_reconstruction", torch.from_numpy(vis_recon)[None], epoch)

        # Save comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(vis_original, cmap="gray")
        ax1.set_title("Original")
        ax2.imshow(vis_recon, cmap="gray")
        ax2.set_title("Reconstruction")
        plt.savefig(f"{run_dir}/reconstructions/reconstruction_epoch_{epoch:03d}.png")
        plt.close()


###################################################################################################


###################################################################################################
# MAIN METHOD
###################################################################################################
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup directories and get starting epoch
    start_epoch, run_dir, trained_g_path, trained_d_path = setup_training_dirs(
        args.jobname, args.checkpoint
    )

    # Save config copy to run directory
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    # Setup data loaders
    train_loader = DataLoader(
        dataset_train,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    # Setup models
    autoencoder = AutoencoderKlMaisi(**config["model"]["autoencoder"]).to(device)
    discriminator = PatchDiscriminator(**config["model"]["discriminator"]).to(device)

    # Load checkpoint if specified
    load_checkpoints(autoencoder, discriminator, args.checkpoint)

    # Setup optimizers
    optimizer_g = torch.optim.Adam(
        params=autoencoder.parameters(),
        lr=config["training"]["learning_rate"],
        eps=1e-06 if config["training"]["amp"] else 1e-08,
    )
    optimizer_d = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=config["training"]["learning_rate"],
        eps=1e-06 if config["training"]["amp"] else 1e-08,
    )

    # Setup tensorboard
    tensorboard_path = run_dir / "tensorboard"
    tensorboard_path.mkdir(exist_ok=True)
    writer = SummaryWriter(tensorboard_path)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(start_epoch, config["training"]["epochs"]):
        print(f"Epoch {epoch}/{config['training']['epochs']}")

        # Training epoch
        train_losses = train_epoch(
            autoencoder,
            discriminator,
            optimizer_g,
            optimizer_d,
            train_loader,
            config,
            device,
        )

        # Log training metrics
        for loss_name, loss_value in train_losses.items():
            writer.add_scalar(f"train_{loss_name}", loss_value, epoch)

        # Save checkpoint
        save_checkpoint(epoch, autoencoder, discriminator, trained_g_path)

        # Validation
        if epoch % config["training"]["val_interval"] == 0:
            val_losses = validate_epoch(
                autoencoder, discriminator, val_loader, config, device
            )

            # Save best model
            if val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                save_checkpoint(
                    epoch, autoencoder, discriminator, trained_g_path, is_best=True
                )

            # Log validation metrics
            log_validation_metrics(val_losses, writer, epoch, run_dir)
###################################################################################################
