#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ControlNet inference script with proper image encoding/decoding integration."""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torchvision import transforms

# Import modules from codebase
from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.sample import ReconModel
from scripts.utils import define_instance
from scripts.utils_data import split_grayscale_to_channels


class InferenceConfig:
    """Configuration container for ControlNet inference."""

    def __init__(self, config_path: str):
        """Initialize configuration from JSON file."""
        self.config_path = Path(config_path)
        self._config_data = self._load_config()
        self.args = self._create_args_namespace()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with self.config_path.open("r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Error loading config from {self.config_path}: {e}")

    def _create_args_namespace(self) -> argparse.Namespace:
        """Create argument namespace from config dictionary."""
        args = argparse.Namespace()

        # Merge config sections into args
        for section in ["environment", "model_def", "training", "model"]:
            if section in self._config_data:
                for key, value in self._config_data[section].items():
                    setattr(args, key, value)

        # Set model path attributes
        env_config = self._config_data.get("environment", {})
        for path_attr in [
            "trained_autoencoder_path",
            "trained_diffusion_path",
            "trained_controlnet_path",
        ]:
            if path_attr in env_config:
                setattr(args, path_attr, env_config[path_attr])

        return args

    @property
    def autoencoder_config(self) -> Dict[str, Any]:
        """Get autoencoder model configuration."""
        return self._config_data.get("model", {}).get("autoencoder", {})


class ImageProcessor:
    """Handles image loading and encoding to latent space with proper normalization."""

    def __init__(self, device: torch.device, logger: logging.Logger):
        """Initialize image processor."""
        self.device = device
        self.logger = logger

    def load_and_preprocess_image(
        self, image_path: Path, target_size: Tuple[int, int] = (256, 256)
    ) -> torch.Tensor:
        """Load and preprocess image with consistent normalization."""
        # Load image and convert to grayscale
        image = Image.open(image_path).convert("L")

        # Resize to target size
        image = image.resize(target_size, Image.Resampling.LANCZOS)

        # Define transform pipeline matching your autoencoder script
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts to [0, 1]
                transforms.Normalize([0.5], [0.5]),  # Converts to [-1, 1]
            ]
        )

        # Apply transform and add batch dimension
        image_tensor = (
            transform(image).unsqueeze(0).to(self.device, dtype=torch.float32)
        )

        self.logger.info(
            f"Loaded image {image_path.name} with shape: {image_tensor.shape}"
        )
        return image_tensor

    def encode_to_latent(
        self,
        image_tensor: torch.Tensor,
        autoencoder: torch.nn.Module,
        scale_factor: float,
    ) -> torch.Tensor:
        """Encode image tensor to latent space using proper sampling."""
        with torch.no_grad():
            with autocast("cuda", enabled=self.device.type == "cuda"):
                # Encode the image
                z_mu, z_sigma = autoencoder.encode(image_tensor)

                # Sample from the latent distribution (matching your autoencoder script)
                latent = autoencoder.sampling(z_mu, z_sigma)

                # Apply scale factor
                if scale_factor != 1.0:
                    scale_tensor = torch.as_tensor(
                        scale_factor, device=latent.device, dtype=latent.dtype
                    )
                    latent = latent * scale_tensor

                self.logger.info(f"Encoded to latent shape: {latent.shape}")
                return latent

    def load_and_encode_image(
        self, image_path: Path, autoencoder: torch.nn.Module, scale_factor: float
    ) -> torch.Tensor:
        """Complete pipeline: load, preprocess, and encode image."""
        # Load and preprocess image
        image_tensor = self.load_and_preprocess_image(image_path)

        # Encode to latent space
        latent = self.encode_to_latent(image_tensor, autoencoder, scale_factor)

        return latent

    def find_image_files(self, image_dir: Path) -> List[Path]:
        """Find all image files in directory."""
        if not image_dir.is_dir():
            raise ValueError(f"Image path must be a directory: {image_dir}")

        # Look for common image formats
        image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.tiff", "*.tif"]:
            image_files.extend(image_dir.glob(ext))
            image_files.extend(image_dir.glob(ext.upper()))

        image_files = sorted(image_files)

        if not image_files:
            raise ValueError(f"No image files found in {image_dir}")

        self.logger.info(f"Found {len(image_files)} input images")
        return image_files


class ModelManager:
    """Manages loading and initialization of ControlNet models."""

    def __init__(
        self, config: InferenceConfig, device: torch.device, logger: logging.Logger
    ):
        """Initialize model manager."""
        self.config = config
        self.device = device
        self.logger = logger
        self.scale_factor = 1.0

    def load_all_models(
        self,
    ) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, Any, float]:
        """Load and prepare all required models."""
        self.logger.info("Loading models...")

        autoencoder = self._load_autoencoder()
        unet = self._load_unet()
        controlnet = self._load_controlnet()
        noise_scheduler = define_instance(self.config.args, "noise_scheduler")

        # Set all models to evaluation mode
        for model in [autoencoder, unet, controlnet]:
            model.eval()

        return autoencoder, unet, controlnet, noise_scheduler, self.scale_factor

    def _load_autoencoder(self) -> torch.nn.Module:
        """Load autoencoder model from checkpoint."""
        checkpoint_path = Path(self.config.args.trained_autoencoder_path)
        self._validate_checkpoint_exists(checkpoint_path, "autoencoder")

        # Create and load model
        autoencoder = AutoencoderKlMaisi(**self.config.autoencoder_config).to(
            self.device
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )

        # Handle different checkpoint formats (matching your autoencoder script)
        if "autoencoder_state_dict" in checkpoint:
            autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
        elif "model" in checkpoint:
            autoencoder.load_state_dict(checkpoint["model"])
        elif "state_dict" in checkpoint:
            autoencoder.load_state_dict(checkpoint["state_dict"])
        else:
            raise ValueError("Cannot find model weights in checkpoint")

        self.logger.info(f"Loaded autoencoder from {checkpoint_path}")
        return autoencoder

    def _load_unet(self) -> torch.nn.Module:
        """Load UNet model from checkpoint."""
        checkpoint_path = Path(self.config.args.trained_diffusion_path)
        self._validate_checkpoint_exists(checkpoint_path, "UNet")

        # Create and load model
        unet = define_instance(self.config.args, "diffusion_unet_def").to(self.device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        unet.load_state_dict(checkpoint["unet_state_dict"])

        # Store scale factor
        self.scale_factor = checkpoint.get("scale_factor", 1.0)

        self.logger.info(f"Loaded UNet from {checkpoint_path}")
        self.logger.info(f"Using scale_factor: {self.scale_factor}")
        return unet

    def _load_controlnet(self) -> torch.nn.Module:
        """Load ControlNet model from checkpoint."""
        checkpoint_path = Path(self.config.args.trained_controlnet_path)
        self._validate_checkpoint_exists(checkpoint_path, "ControlNet")

        # Create and load model
        controlnet = define_instance(self.config.args, "controlnet_def").to(self.device)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=True
        )
        controlnet.load_state_dict(checkpoint["controlnet_state_dict"])

        self.logger.info(f"Loaded ControlNet from {checkpoint_path}")
        return controlnet

    def _validate_checkpoint_exists(self, path: Path, model_name: str) -> None:
        """Validate that checkpoint file exists."""
        if not path.exists():
            raise FileNotFoundError(f"{model_name} checkpoint not found: {path}")


class ImageGenerator:
    """Handles image generation using ControlNet and UNet models."""

    def __init__(self, device: torch.device, seed: int, logger: logging.Logger):
        """Initialize image generator."""
        self.device = device
        self.seed = seed
        self.logger = logger
        self._setup_deterministic_behavior()

    def _setup_deterministic_behavior(self) -> None:
        """Configure PyTorch for deterministic behavior."""
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    def denoise_with_controlnet(
        self,
        unet: torch.nn.Module,
        controlnet: torch.nn.Module,
        noise_scheduler: Any,
        condition: torch.Tensor,
        initial_latent: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Denoise latent using ControlNet guidance."""
        self._set_random_seeds()

        latents = initial_latent.clone()
        condition = condition.clone()
        timesteps = noise_scheduler.timesteps

        progress_iter = (
            tqdm(timesteps, desc="ControlNet denoising") if verbose else timesteps
        )

        for timestep in progress_iter:
            timestep_tensor = torch.tensor([timestep], device=self.device)

            with torch.no_grad():
                with autocast("cuda", enabled=self.device.type == "cuda"):
                    # Get ControlNet outputs
                    down_samples, mid_sample = controlnet(
                        x=latents,
                        timesteps=timestep_tensor,
                        controlnet_cond=condition,
                    )

                    # Get noise prediction from UNet
                    noise_pred = unet(
                        x=latents,
                        timesteps=timestep_tensor,
                        down_block_additional_residuals=down_samples,
                        mid_block_additional_residual=mid_sample,
                    )

                    # Update latents
                    latents, _ = noise_scheduler.step(
                        noise_pred,
                        timestep,
                        latents,  # , eta=1.0
                        # todo
                    )

        return latents

    def denoise_with_unet_only(
        self,
        unet: torch.nn.Module,
        noise_scheduler: Any,
        initial_latent: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Denoise latent using UNet only (no ControlNet)."""
        self._set_random_seeds()

        latents = initial_latent.clone()
        timesteps = noise_scheduler.timesteps

        progress_iter = (
            tqdm(timesteps, desc="UNet-only denoising") if verbose else timesteps
        )

        for timestep in progress_iter:
            timestep_tensor = torch.tensor([timestep], device=self.device)

            with torch.no_grad():
                with autocast("cuda", enabled=self.device.type == "cuda"):
                    noise_pred = unet(x=latents, timesteps=timestep_tensor)
                    latents, _ = noise_scheduler.step(noise_pred, timestep, latents)

        return latents

    def _set_random_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)


class OutputManager:
    """Manages output file operations and visualization."""

    def __init__(self, output_dir: Path, logger: logging.Logger):
        """Initialize output manager."""
        self.output_dir = output_dir
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def decode_and_save_image(
        self,
        latent: torch.Tensor,
        autoencoder: torch.nn.Module,
        scale_factor: float,
        filename: str,
    ) -> Path:
        """Decode latent to image and save with proper normalization."""
        with torch.no_grad():
            with autocast("cuda", enabled=latent.device.type == "cuda"):
                # Reverse scale factor if applied
                if scale_factor != 1.0:
                    scale_tensor = torch.as_tensor(
                        scale_factor, device=latent.device, dtype=latent.dtype
                    )
                    latent_scaled = latent / scale_tensor
                else:
                    latent_scaled = latent

                # Decode the latent
                reconstructed = autoencoder.decode(latent_scaled)

        # Convert from [-1, 1] to [0, 1] range with better contrast handling
        reconstructed = reconstructed.squeeze().cpu()

        # Apply histogram stretching for better contrast
        min_val = reconstructed.min()
        max_val = reconstructed.max()

        if max_val > min_val:
            # Stretch to full [0, 1] range
            image_normalized = (reconstructed - min_val) / (max_val - min_val)
        else:
            # Fallback to standard normalization
            image_normalized = torch.clamp((reconstructed + 1.0) / 2.0, 0.0, 1.0)

        # Convert to numpy
        image_np = image_normalized.numpy()

        # Save using PIL with enhanced contrast
        output_path = self.output_dir / filename
        if len(image_np.shape) == 2:  # Grayscale
            # Apply gamma correction for better visibility
            gamma = 0.8  # Slightly brighten
            image_np = np.power(image_np, gamma)
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8), mode="L")
        else:  # RGB
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

        image_pil.save(output_path, quality=95)

        return output_path

    def save_image_tensor(self, image_tensor: torch.Tensor, filename: str) -> Path:
        """Save image tensor directly (for original images)."""
        # Convert from [-1, 1] to [0, 1] range
        image_normalized = torch.clamp((image_tensor + 1.0) / 2.0, 0.0, 1.0)

        # Convert to numpy and save
        image_np = image_normalized.squeeze().cpu().numpy()

        output_path = self.output_dir / filename
        if len(image_np.shape) == 2:  # Grayscale
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8), mode="L")
        else:  # RGB
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

        image_pil.save(output_path, quality=95)

        return output_path

    def create_comparison_grid(self, results: List[Dict[str, Any]]) -> Path:
        """Create 5-column comparison grid including input image."""
        self.logger.info("Creating 5-column comparison grid...")

        # Filter out results with no generated images
        valid_results = [
            r
            for r in results
            if all(
                [r["controlnet_images"], r["empty_mask_images"], r["unet_only_images"]]
            )
        ]

        if not valid_results:
            self.logger.error("No valid results to create comparison grid")
            return self._create_empty_grid()

        num_masks = len(valid_results)
        fig, axes = plt.subplots(num_masks, 5, figsize=(25, 5 * num_masks))

        # Reshape axes for consistent indexing
        if num_masks == 1:
            axes = axes.reshape(1, -1)

        # Column titles
        col_titles = [
            "Input Image",
            "Mask",
            "ControlNet Output",
            "Empty Mask Output",
            "UNet Only",
        ]
        for i, title in enumerate(col_titles):
            fig.text(
                0.1 + 0.2 * i, 0.98, title, ha="center", fontsize=14, weight="bold"
            )

        # Fill grid
        for i, result in enumerate(valid_results):
            # Column 0: Input image
            self._add_image_to_grid(
                axes[i, 0],
                result["input_image_path"],
                f"Input: {result['input_image_path'].name}",
                cmap="gray",
            )

            # Column 1: Original mask
            self._add_image_to_grid(
                axes[i, 1],
                result["mask_path"],
                f"Mask: {result['mask_path'].name}",
                cmap="gray",
            )

            # Column 2: ControlNet output
            self._add_image_to_grid(
                axes[i, 2], result["controlnet_images"][0], "ControlNet", cmap="gray"
            )

            # Column 3: Empty mask output
            self._add_image_to_grid(
                axes[i, 3], result["empty_mask_images"][0], "Empty Mask", cmap="gray"
            )

            # Column 4: UNet only output
            self._add_image_to_grid(
                axes[i, 4], result["unet_only_images"][0], "UNet Only", cmap="gray"
            )

        # Save grid
        grid_path = self.output_dir / "comparison_grid_5col.png"
        plt.tight_layout()
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Saved 5-column comparison grid to {grid_path}")
        return grid_path

    def _create_empty_grid(self) -> Path:
        """Create empty grid file when no valid results."""
        empty_grid_path = self.output_dir / "comparison_grid_5col_empty.png"
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(
            0.5,
            0.5,
            "No valid results generated",
            ha="center",
            va="center",
            fontsize=16,
        )
        ax.axis("off")
        plt.savefig(empty_grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return empty_grid_path

    def _add_image_to_grid(
        self, ax, image_path: Path, title: str, cmap: str = None
    ) -> None:
        """Add image to grid subplot."""
        image = Image.open(image_path).convert("L" if cmap == "gray" else "RGB")
        image_np = np.array(image) / 255.0

        ax.imshow(image_np, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")


class ControlNetInference:
    """Main ControlNet inference pipeline with proper image encoding/decoding."""

    def __init__(
        self,
        config_path: str,
        mask_path: str,
        input_image_path: str,
        output_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """Initialize ControlNet inference pipeline."""
        self.mask_path = Path(mask_path)
        self.input_image_path = Path(input_image_path)
        self.seed = seed

        # Setup components
        self.logger = self._setup_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            output_dir = f"./outputs/controlnet_inference_fixed_{timestamp}"

        self.output_manager = OutputManager(Path(output_dir), self.logger)
        self.logger.info(f"Output directory: {self.output_manager.output_dir}")

        # Initialize configuration and models
        self.config = InferenceConfig(config_path)
        self.model_manager = ModelManager(self.config, self.device, self.logger)
        self.image_generator = ImageGenerator(self.device, seed, self.logger)
        self.image_processor = ImageProcessor(self.device, self.logger)

        # Load models
        self.models = self.model_manager.load_all_models()

        # Create latents directory
        self.latents_dir = self.output_manager.output_dir / "encoded_latents"
        self.latents_dir.mkdir(exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("controlnet.inference")
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        # Console handler
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler("controlnet_inference.log")

        formatter = logging.Formatter(
            "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def find_mask_files(self) -> List[Path]:
        """Find all PNG mask files in the mask directory."""
        if not self.mask_path.is_dir():
            raise ValueError(f"Mask path must be a directory: {self.mask_path}")

        mask_files = sorted(self.mask_path.glob("*.png"))

        if not mask_files:
            raise ValueError(f"No PNG mask files found in {self.mask_path}")

        self.logger.info(f"Found {len(mask_files)} mask files")
        return mask_files

    def encode_input_images(self) -> Tuple[List[torch.Tensor], List[Path]]:
        """Encode input images to latent space with proper preprocessing."""
        autoencoder, _, _, _, scale_factor = self.models

        # Find input images
        input_images = self.image_processor.find_image_files(self.input_image_path)

        encoded_latents = []
        for i, image_path in enumerate(input_images):
            self.logger.info(
                f"Encoding input image {i+1}/{len(input_images)}: {image_path.name}"
            )

            # Encode image to latent space using proper pipeline
            latent = self.image_processor.load_and_encode_image(
                image_path, autoencoder, scale_factor
            )

            # Save encoded latent
            torch.save(
                latent, self.latents_dir / f"latent_{i:03d}_{image_path.stem}.pt"
            )
            encoded_latents.append(latent)

        return encoded_latents, input_images

    def process_single_mask(
        self,
        mask_path: Path,
        input_latents: List[torch.Tensor],
        input_image_paths: List[Path],
        mask_idx: int,
        num_samples: int = 1,
    ) -> Dict[str, Any]:
        """Process a single mask with all generation methods using proper encoding/decoding."""
        autoencoder, unet, controlnet, noise_scheduler, scale_factor = self.models

        # Load and preprocess mask
        mask_tensor = self.image_processor.load_and_preprocess_image(mask_path)
        mask_channels = split_grayscale_to_channels(mask_tensor)

        # Create empty mask (all zeros) with same dimensions
        empty_mask = torch.zeros_like(mask_tensor)
        empty_mask_channels = split_grayscale_to_channels(empty_mask)

        mask_name = mask_path.stem

        # Storage for results
        controlnet_images = []
        empty_mask_images = []
        unet_only_images = []

        # Use the corresponding input image for this mask
        input_latent = input_latents[mask_idx % len(input_latents)]
        input_image_path = input_image_paths[mask_idx % len(input_image_paths)]

        for i in range(num_samples):
            try:
                # Generate with ControlNet (original mask)
                self.logger.info(f"Generating ControlNet output {i + 1}/{num_samples}")
                controlnet_latent = self.image_generator.denoise_with_controlnet(
                    unet,
                    controlnet,
                    noise_scheduler,
                    mask_channels,
                    input_latent.clone(),
                    verbose=True,
                )
                controlnet_path = self.output_manager.decode_and_save_image(
                    controlnet_latent,
                    autoencoder,
                    scale_factor,
                    f"{mask_name}_controlnet_m{mask_idx:02d}_s{i:03d}.png",
                )
                controlnet_images.append(controlnet_path)

                # Generate with ControlNet (empty mask)
                self.logger.info(f"Generating empty mask output {i + 1}/{num_samples}")
                empty_mask_latent = self.image_generator.denoise_with_controlnet(
                    unet,
                    controlnet,
                    noise_scheduler,
                    empty_mask_channels,
                    input_latent.clone(),
                    verbose=True,
                )
                empty_mask_path = self.output_manager.decode_and_save_image(
                    empty_mask_latent,
                    autoencoder,
                    scale_factor,
                    f"{mask_name}_empty_mask_m{mask_idx:02d}_s{i:03d}.png",
                )
                empty_mask_images.append(empty_mask_path)

                # Generate with UNet only
                self.logger.info(f"Generating UNet-only output {i + 1}/{num_samples}")
                unet_latent = self.image_generator.denoise_with_unet_only(
                    unet, noise_scheduler, input_latent.clone(), verbose=True
                )
                unet_path = self.output_manager.decode_and_save_image(
                    unet_latent,
                    autoencoder,
                    scale_factor,
                    f"{mask_name}_unet_only_m{mask_idx:02d}_s{i:03d}.png",
                )
                unet_only_images.append(unet_path)

                # Clear GPU memory
                torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"Error processing sample {i + 1}: {e}")
                continue

        return {
            "mask_path": mask_path,
            "input_image_path": input_image_path,
            "controlnet_images": controlnet_images,
            "empty_mask_images": empty_mask_images,
            "unet_only_images": unet_only_images,
        }

    def run(self) -> None:
        """Run the complete inference pipeline with proper image handling."""
        # Find mask files and encode input images
        mask_files = self.find_mask_files()
        input_latents, input_image_paths = self.encode_input_images()

        if len(input_latents) == 0:
            raise ValueError("No input images found to encode")

        # Process all masks
        all_results = []
        start_time = time.time()

        for i, mask_file in enumerate(mask_files):
            self.logger.info(
                f"\nProcessing mask {i+1}/{len(mask_files)}: {mask_file.name}"
            )
            result = self.process_single_mask(
                mask_file, input_latents, input_image_paths, i, num_samples=1
            )
            all_results.append(result)

        # Create comparison grid
        total_time = time.time() - start_time
        self.logger.info(f"\nProcessed {len(all_results)} masks in {total_time:.2f}s")

        self.output_manager.create_comparison_grid(all_results)
        self.logger.info(f"Results saved to: {self.output_manager.output_dir}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ControlNet Inference with proper image encoding/decoding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/config_CONTROLNET_hungary.json",
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default="/home/mhuber/Thesis/data/RETOUCH/retouch_masks",
        help="Directory containing mask files",
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        default="/home/mhuber/Thesis/data/RETOUCH/input_images",
        help="Directory containing input images to encode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (creates timestamped dir if not specified)",
    )
    parser.add_argument(
        "--seed", type=int, default=324568, help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    try:
        args = parse_arguments()

        pipeline = ControlNetInference(
            config_path=args.config_path,
            mask_path=args.mask_path,
            input_image_path=args.input_image_path,
            output_dir=args.output_dir,
            seed=args.seed,
        )

        pipeline.run()

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
