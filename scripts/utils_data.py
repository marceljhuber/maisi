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
import pandas as pd


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


########################################################################################################################
# Dataset Variations
########################################################################################################################
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


########################################################################################################################
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


########################################################################################################################
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


########################################################################################################################
class ClusterDataset(Dataset):
    """Dataset for loading images with cluster labels."""

    def __init__(self, latent_paths, cluster_labels, transform=None):
        """
        Initialize the dataset.

        Args:
            latent_paths (list): List of image file paths
            cluster_labels (list): List of cluster labels corresponding to file paths
            transform (callable, optional): Optional transform to be applied on the images
        """
        self.latent_paths = latent_paths
        self.cluster_labels = cluster_labels

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx):
        # Load latent
        latent_path = self.latent_paths[idx]
        latent = torch.load(latent_path, weights_only=True)

        # Get label
        label = self.cluster_labels[idx]

        return latent, label


########################################################################################################################


def list_latent_files(directory_path):
    """Find all .pt files in directory and subdirectories."""
    return glob.glob(os.path.join(directory_path, "**", "*_latent.pt"), recursive=True)


def split_data_by_patient(file_paths, train_ratio=0.8):
    """Split dataset by patient ID to avoid patient leakage between train and validation."""
    # Extract unique patient IDs
    patient_ids = set()
    for file_path in file_paths:
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

    for file_path in file_paths:
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


def split_data_by_patient(file_paths, cluster_labels=None, train_ratio=0.9):
    """
    Split data by patient ID to prevent patient data leakage between train and validation.

    Args:
        file_paths (list): List of image file paths
        cluster_labels (list): List of cluster labels
        train_ratio (float): Ratio of patients to include in training set

    Returns:
        tuple: (train_files, train_labels, val_files, val_labels)
    """
    # Extract patient IDs from filenames
    # Assuming format like: .../CNV-1016042-101.jpeg where 1016042 is the patient ID
    patient_ids = []
    for file_path in file_paths:
        path = Path(file_path)
        # Extract patient ID from filename using the pattern shown in the example
        try:
            patient_id = path.stem.split("-")[1]
            patient_ids.append(patient_id)
        except IndexError:
            # If the filename format doesn't match, use the filename as ID
            patient_ids.append(path.stem)

    # Get unique patient IDs
    unique_patients = list(set(patient_ids))

    # Shuffle and split patients
    random.shuffle(unique_patients)
    split_idx = int(len(unique_patients) * train_ratio)
    train_patients = set(unique_patients[:split_idx])
    val_patients = set(unique_patients[split_idx:])

    # Split data by patient
    train_files, train_labels = [], []
    val_files, val_labels = [], []

    if cluster_labels is not None:
        for i, (file_path, label) in enumerate(zip(file_paths, cluster_labels)):
            patient_id = patient_ids[i]
            if patient_id in train_patients:
                train_files.append(file_path)
                train_labels.append(label)
            else:
                val_files.append(file_path)
                val_labels.append(label)

        return train_files, train_labels, val_files, val_labels
    else:
        for file_path in file_paths:
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


########################################################################################################################
# Data Loader Variations
########################################################################################################################
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
    train_files, val_files = split_data_by_patient(latent_files, train_ratio)

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


########################################################################################################################
def create_cluster_dataloaders(
    csv_path, batch_size=40, num_workers=8, train_ratio=0.9, transform=None
):
    """
    Create train and validation dataloaders from a CSV file containing file paths and cluster labels.

    Args:
        csv_path (str): Path to the CSV file with 'filepath' and 'cluster' columns
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        train_ratio (float): Ratio of patients to include in training set
        transform (callable, optional): Transform to apply to the images

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Set random seeds for reproducibility
    set_random_seeds()

    # Read CSV file
    df = pd.read_csv(csv_path)

    # Check if required columns exist
    if "latent_path" not in df.columns or "cluster" not in df.columns:
        raise ValueError("CSV must contain 'filepath' and 'cluster' columns")

    # Extract file paths and cluster labels
    file_paths = df["latent_path"].tolist()
    cluster_labels = df["cluster"].tolist()

    # Verify that all files exist
    missing_files = [f for f in file_paths if not os.path.exists(f)]
    if missing_files:
        print(
            f"Warning: {len(missing_files)} files not found. First few: {missing_files[:5]}"
        )
        # Filter out missing files
        valid_indices = [i for i, f in enumerate(file_paths) if os.path.exists(f)]
        file_paths = [file_paths[i] for i in valid_indices]
        cluster_labels = [cluster_labels[i] for i in valid_indices]

    # Split data by patient ID
    train_files, train_labels, val_files, val_labels = split_data_by_patient(
        file_paths, cluster_labels, train_ratio
    )

    print(f"Found {len(train_files)} training and {len(val_files)} validation samples")

    # Count unique patients in each split
    train_patients = set(
        [
            Path(f).stem.split("-")[1] if "-" in Path(f).stem else Path(f).stem
            for f in train_files
        ]
    )
    val_patients = set(
        [
            Path(f).stem.split("-")[1] if "-" in Path(f).stem else Path(f).stem
            for f in val_files
        ]
    )

    print(f"Training samples from {len(train_patients)} patients")
    print(f"Validation samples from {len(val_patients)} patients")

    # Count samples per cluster
    train_cluster_counts = {}
    for label in train_labels:
        train_cluster_counts[label] = train_cluster_counts.get(label, 0) + 1

    val_cluster_counts = {}
    for label in val_labels:
        val_cluster_counts[label] = val_cluster_counts.get(label, 0) + 1

    print("Training samples per cluster:")
    for cluster, count in sorted(train_cluster_counts.items()):
        print(f"  Cluster {cluster}: {count} samples")

    print("Validation samples per cluster:")
    for cluster, count in sorted(val_cluster_counts.items()):
        print(f"  Cluster {cluster}: {count} samples")

    print(f"train_files: ", len(train_files))
    # Create datasets
    train_dataset = ClusterDataset(train_files, train_labels, transform)
    val_dataset = ClusterDataset(val_files, val_labels, transform)

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


########################################################################################################################
def create_oct_dataloaders(
    data_dir, batch_size=40, num_workers=8, train_ratio=0.9, transform=None
):
    """
    Create train and validation dataloaders from a directory containing
    precomputed OCT tensors (.pt files) and their reference mask images (.png files).

    Args:
        data_dir (str): Path to the directory containing OCT tensors and reference images
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        train_ratio (float): Ratio of patients to include in training set
        transform (callable, optional): Transform to apply to the data

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Set random seeds for reproducibility
    set_random_seeds()

    # Get all OCT tensor paths
    oct_files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith("_oct_latent.pt")]
    )
    oct_paths = [os.path.join(data_dir, f) for f in oct_files]

    # Get corresponding reference mask paths
    ref_paths = []
    valid_oct_paths = []

    for oct_path in oct_paths:
        ref_path = oct_path.replace("_oct_latent.pt", "_ref.png")
        if os.path.exists(ref_path):
            ref_paths.append(ref_path)
            valid_oct_paths.append(oct_path)
        else:
            print(
                f"Warning: Reference image not found for {os.path.basename(oct_path)}"
            )

    # Update oct_paths to only include those with matching ref files
    oct_paths = valid_oct_paths

    if len(oct_paths) == 0:
        raise ValueError("No valid OCT tensor-image pairs found in the directory")

    print(f"Found {len(oct_paths)} valid OCT tensor-image pairs")

    # Extract device types from filenames
    device_types = [os.path.basename(f).split("_")[0] for f in oct_paths]

    # Simple random split of data
    num_samples = len(oct_paths)
    num_train = int(num_samples * train_ratio)

    # Create a random permutation of indices
    indices = list(range(num_samples))
    random.shuffle(indices)

    # Split into train and validation sets
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # Get the actual paths
    train_oct_paths = [oct_paths[i] for i in train_indices]
    train_ref_paths = [ref_paths[i] for i in train_indices]
    val_oct_paths = [oct_paths[i] for i in val_indices]
    val_ref_paths = [ref_paths[i] for i in val_indices]

    print(
        f"Training set: {len(train_oct_paths)} pairs."
    )
    print(
        f"Validation set: {len(val_oct_paths)} pairs."
    )

    # Count samples per device type
    train_device_counts = {}
    for i in train_indices:
        device = device_types[i]
        train_device_counts[device] = train_device_counts.get(device, 0) + 1

    val_device_counts = {}
    for i in val_indices:
        device = device_types[i]
        val_device_counts[device] = val_device_counts.get(device, 0) + 1

    print("Training samples per device type:")
    for device, count in sorted(train_device_counts.items()):
        print(f"  {device}: {count} samples")

    print("Validation samples per device type:")
    for device, count in sorted(val_device_counts.items()):
        print(f"  {device}: {count} samples")

    # Create custom dataset for OCT tensor data
    class OCTMixedDataset(Dataset):
        def __init__(self, oct_paths, ref_paths, transform=None):
            self.oct_paths = oct_paths
            self.ref_paths = ref_paths
            self.transform = transform

        def __len__(self):
            return len(self.oct_paths)

        def __getitem__(self, idx):
            # Load precomputed OCT tensor
            oct_tensor = torch.load(self.oct_paths[idx], weights_only=True)

            # Load reference mask image as grayscale
            ref_img = Image.open(self.ref_paths[idx]).convert("L")

            # Convert reference image to tensor and ensure it's normalized between 0 and 1
            ref_array = np.array(ref_img)
            ref_tensor = (
                torch.tensor(ref_array, dtype=torch.float32).unsqueeze(0) / 255.0
            )

            # Apply transforms if specified
            if self.transform:
                # Create a dict with both tensors so they can be transformed together
                sample = {"image": oct_tensor, "mask": ref_tensor}
                sample = self.transform(sample)
                oct_tensor = sample["image"]
                ref_tensor = sample["mask"]

            return oct_tensor, ref_tensor

    # Create datasets
    train_dataset = OCTMixedDataset(train_oct_paths, train_ref_paths, transform)
    val_dataset = OCTMixedDataset(val_oct_paths, val_ref_paths, transform)

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


########################################################################################################################


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


def split_grayscale_to_channels(
    grayscale_img, values=[0, 85, 127, 170, 255], tolerance=0
):
    """
    Splits a grayscale image tensor into 5 binary channels, one for each specified value.

    Args:
        grayscale_img (torch.Tensor): Grayscale image tensor of shape [batch_size, 1, H, W] or [1, H, W] or [H, W]
        values (list): Pixel values to create channels for. Default: [0, 85, 127, 170, 255]
        tolerance (int): Tolerance for considering a pixel as belonging to a value. Default: 0

    Returns:
        torch.Tensor: Binary channels of shape [batch_size, len(values), H, W]
    """
    import torch

    # Ensure grayscale_img is a tensor
    if not isinstance(grayscale_img, torch.Tensor):
        grayscale_img = torch.tensor(grayscale_img)

    # Handle different input dimensions
    original_shape = grayscale_img.shape
    if len(original_shape) == 2:  # [H, W]
        grayscale_img = grayscale_img.unsqueeze(0).unsqueeze(
            0
        )  # Add batch and channel dims
    elif len(original_shape) == 3:  # [1, H, W] or [B, 1, W]
        if original_shape[0] == 1:  # [1, H, W]
            grayscale_img = grayscale_img.unsqueeze(0)  # Add batch dim
        else:  # [B, H, W]
            grayscale_img = grayscale_img.unsqueeze(1)  # Add channel dim

    batch_size = grayscale_img.shape[0]
    height, width = grayscale_img.shape[2], grayscale_img.shape[3]

    # Create output tensor for the 5 channels
    channels = torch.zeros(
        (batch_size, len(values), height, width), device=grayscale_img.device
    )

    # For each value, create a binary mask
    for i, value in enumerate(values):
        # Create binary mask where pixels are within tolerance of the target value
        lower_bound = value - tolerance
        upper_bound = value + tolerance
        mask = (grayscale_img >= lower_bound) & (grayscale_img <= upper_bound)
        channels[:, i : i + 1, :, :] = mask.float()

    return channels
