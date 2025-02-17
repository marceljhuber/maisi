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
