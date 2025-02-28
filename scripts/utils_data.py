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
import wandb
from PIL import Image
from monai.config import print_config
from monai.data import DataLoader
from monai.losses.adversarial_loss import PatchAdversarialLoss
from monai.losses.perceptual import PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss, MSELoss
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_image_files(directory_path):
    """Find all image files in directory and subdirectories."""
    image_extensions = (".jpg", ".jpeg", ".png")
    files = glob.glob(os.path.join(directory_path, "**", "*.*"), recursive=True)
    return [file for file in files if file.lower().endswith(image_extensions)]


def split_train_val_by_patient(image_names, train_ratio=0.9):
    """Split dataset into training and validation sets by patient ID."""
    patient_ids = set(name.split("-")[1] for name in image_names)
    num_train = int(len(patient_ids) * train_ratio)
    train_patients = set(random.sample(list(patient_ids), num_train))

    train_images = [img for img in image_names if img.split("-")[1] in train_patients]
    val_images = [img for img in image_names if img.split("-")[1] not in train_patients]

    return train_images, val_images


class SpeckleNoise:
    """Add speckle noise to images."""

    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, x):
        noise = torch.randn_like(x) * self.noise_level
        noisy = x + noise * x
        # Clip values to maintain [-1, 1] range
        return torch.clamp(noisy, -1, 1)


class GrayscaleDataset(Dataset):
    """Dataset for grayscale images."""

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


class GrayscaleDatasetLabels(Dataset):
    """Dataset for grayscale images with one-hot encoded labels."""

    def __init__(self, image_paths, transform=None, num_classes=5):
        self.image_paths = image_paths
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")

        def extract_class_idx(image_path):
            """Extract class index safely from the path, not the filename"""
            try:
                # Use just the parent directory name instead of the full stem
                parent_dir = Path(image_path).parent.name
                digits = [c for c in parent_dir if c.isdigit()]
                if digits:
                    return int(digits[0])
                # Fallback to 0 if no digits found
                return 0
            except (IndexError, ValueError):
                # Return a safe default value if extraction fails
                return 0

        # Replace the class_idx assignment with:
        class_idx = extract_class_idx(image_path)

        # Create one-hot encoded label with 5 dimensions
        label = torch.zeros(self.num_classes)
        label[class_idx] = 1  # Set the corresponding class index to 1

        # Also ensure that class_idx is within valid range before using it:
        if class_idx < label.size(0):  # Check if index is valid
            label[class_idx] = 1  # Set the corresponding class index to 1
        else:
            # Handle out-of-bounds index
            print(
                f"Warning: Class index {class_idx} out of bounds for label tensor with size {label.size(0)}"
            )
            # Either resize the tensor or use a default class
            label[0] = self.num_classes - 1  # Use combined class as default

        if self.transform:
            image = self.transform(image)

        # Reshape label to [num_classes, H, W] for each pixel
        H, W = image.shape[1:]  # Assuming image is [C, H, W]
        label = label.view(-1, 1, 1).repeat(1, H, W)

        return {"image": image, "label": label}


class LatentDataset(Dataset):
    """Dataset for loading latent tensors from .pt files with class labels."""

    def __init__(self, latent_paths):
        self.latent_paths = latent_paths
        self.class_mapping = {"CNV": 0, "DME": 1, "DRUSEN": 2, "NORMAL": 3}

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx):
        latent_path = self.latent_paths[idx]

        # Load the latent tensor
        latent = torch.load(latent_path, weights_only=True)

        # Extract class from filename (part before first dash)
        filename = Path(latent_path).stem  # Get filename without extension
        class_name = filename.split("-")[0]
        class_idx = self.class_mapping.get(class_name, 0)  # Default to 0 if not found

        # Create one-hot encoded label
        label = torch.zeros(4)
        label[class_idx] = 1

        return {
            "latent": latent,
            "label": label,
            "class_idx": class_idx,
            "patient_id": filename.split("-")[1],  # Extract patient ID
        }


def list_latent_files(directory_path):
    """Find all .pt files in directory and subdirectories."""
    return glob.glob(os.path.join(directory_path, "**", "*_latent.pt"), recursive=True)


def split_latents_by_patient(latent_files, train_ratio=0.8):
    """Split dataset by patient ID to avoid patient leakage between train and validation."""
    # Extract unique patient IDs
    patient_ids = set()
    for file_path in latent_files:
        filename = Path(file_path).stem
        try:
            patient_id = filename.split("-")[1]
            patient_ids.add(patient_id)
        except IndexError:
            continue

    patient_ids = list(patient_ids)
    random.shuffle(patient_ids)

    # Split patient IDs
    split_idx = int(len(patient_ids) * train_ratio)
    train_patients = set(patient_ids[:split_idx])
    val_patients = set(patient_ids[split_idx:])

    # Assign files based on patient ID
    train_files = []
    val_files = []

    for file_path in latent_files:
        filename = Path(file_path).stem
        try:
            patient_id = filename.split("-")[1]
            if patient_id in train_patients:
                train_files.append(file_path)
            elif patient_id in val_patients:
                val_files.append(file_path)
        except IndexError:
            continue  # Skip files that don't follow the naming convention

    return train_files, val_files


def create_latent_dataloaders(
    latent_dir, batch_size=40, num_workers=8, train_ratio=0.9
):
    """Create train and validation dataloaders for latent tensors."""
    # Set random seeds for reproducibility
    set_random_seeds()

    # List all latent files
    latent_files = list_latent_files(latent_dir)
    if not latent_files:
        raise ValueError(f"No latent files found in {latent_dir}")

    # Split by patient ID
    train_files, val_files = split_latents_by_patient(latent_files, train_ratio)

    print(
        f"Found {len(train_files)} training and {len(val_files)} validation latent files"
    )
    print(
        f"Training samples from {len(set(Path(f).stem.split('-')[1] for f in train_files))} patients"
    )
    print(
        f"Validation samples from {len(set(Path(f).stem.split('-')[1] for f in val_files))} patients"
    )

    # Create datasets
    train_dataset = LatentDataset(train_files)
    val_dataset = LatentDataset(val_files)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


def setup_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            SpeckleNoise(0.1),
            transforms.Lambda(lambda x: 2 * x - 1),  # Scale to [-1, 1]
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2 * x - 1),  # Scale to [-1, 1]
        ]
    )

    return train_transform, val_transform


def setup_dataloaders(train_images, val_images, train_transform, val_transform, config):
    """Setup training and validation dataloaders."""
    train_dataset = GrayscaleDataset(train_images, transform=train_transform)
    val_dataset = GrayscaleDataset(val_images, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    return train_loader, val_loader


def setup_training(config):
    """Setup all training components."""
    # Set random seeds
    set_random_seeds()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup directories
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_")
    run_dir = Path(f"./runs/{config['main']['jobname']}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = run_dir / "reconstructions"
    recon_dir.mkdir(exist_ok=True)

    # Setup data
    image_files = list_image_files(config["data"]["image_dir"])

    # Split by patient ID instead of random split
    train_images, val_images = split_train_val_by_patient(image_files, train_ratio=0.9)

    print(
        f"Found {len(train_images)} train images and {len(val_images)} validation images."
    )

    # Setup transforms
    train_transform, val_transform = setup_transforms()

    # Setup dataloaders
    train_dataset = GrayscaleDatasetLabels(train_images, transform=val_transform)
    val_dataset = GrayscaleDatasetLabels(val_images, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )

    return device, run_dir, recon_dir, train_loader, val_loader
