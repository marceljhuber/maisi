#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ControlNet inference script with comparison grid generation.

This module provides functionality for running inference with ControlNet models
on image masks, including empty mask baselines. It generates a comparison grid
showing mask, ControlNet output, empty mask output, and UNet-only output.

Example usage:
    python controlnet_inference.py --config_path=./configs/my_config.json
                                   --mask_path=./masks_dir
                                   --output_dir=./outputs
"""

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
from torch.amp import autocast
from tqdm import tqdm

# Import modules from codebase
from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.sample import ReconModel
from scripts.utils import define_instance
from scripts.utils_data import split_grayscale_to_channels


class InferenceConfig:
    """Configuration container for ControlNet inference."""

    def __init__(self, config_path: str):
        """Initialize configuration from JSON file.

        Args:
            config_path: Path to configuration JSON file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            json.JSONDecodeError: If config file has invalid JSON.
        """
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


class ModelManager:
    """Manages loading and initialization of ControlNet models."""

    def __init__(
        self, config: InferenceConfig, device: torch.device, logger: logging.Logger
    ):
        """Initialize model manager.

        Args:
            config: Configuration object.
            device: PyTorch device.
            logger: Logger instance.
        """
        self.config = config
        self.device = device
        self.logger = logger
        self.scale_factor = 1.0

    def load_all_models(
        self,
    ) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, Any, float]:
        """Load and prepare all required models.

        Returns:
            Tuple of (autoencoder, unet, controlnet, noise_scheduler, scale_factor).
        """
        self.logger.info("Loading models...")

        autoencoder = self._load_autoencoder()
        unet = self._load_unet()
        controlnet = self._load_controlnet()
        noise_scheduler = define_instance(self.config.args, "noise_scheduler")

        # Set all models to evaluation mode and convert to float32
        models = [autoencoder, unet, controlnet]
        for model in models:
            model.eval()
            self._convert_model_to_float32(model)

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
        autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])

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

    def _convert_model_to_float32(self, model: torch.nn.Module) -> None:
        """Convert model parameters to float32 for consistent precision."""
        for param in model.parameters():
            param.data = param.data.to(torch.float32)


class ImageGenerator:
    """Handles image generation using ControlNet and UNet models."""

    def __init__(self, device: torch.device, seed: int, logger: logging.Logger):
        """Initialize image generator.

        Args:
            device: PyTorch device.
            seed: Random seed for reproducibility.
            logger: Logger instance.
        """
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
        """Denoise latent using ControlNet guidance.

        Args:
            unet: UNet model.
            controlnet: ControlNet model.
            noise_scheduler: Noise scheduler.
            condition: Conditioning tensor (mask).
            initial_latent: Initial noise latent.
            verbose: Whether to show progress.

        Returns:
            Denoised latent tensor.
        """
        self._set_random_seeds()

        latents = initial_latent.clone().to(dtype=torch.float32)
        condition = condition.to(dtype=torch.float32)
        timesteps = noise_scheduler.timesteps

        progress_iter = (
            tqdm(timesteps, desc="ControlNet denoising") if verbose else timesteps
        )

        for timestep in progress_iter:
            timestep_tensor = torch.tensor([timestep], device=self.device)

            with torch.no_grad():
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
                latents, _ = noise_scheduler.step(noise_pred, timestep, latents)

        return latents

    def denoise_with_unet_only(
        self,
        unet: torch.nn.Module,
        noise_scheduler: Any,
        initial_latent: torch.Tensor,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Denoise latent using UNet only (no ControlNet).

        Args:
            unet: UNet model.
            noise_scheduler: Noise scheduler.
            initial_latent: Initial noise latent.
            verbose: Whether to show progress.

        Returns:
            Denoised latent tensor.
        """
        self._set_random_seeds()

        latents = initial_latent.clone().to(dtype=torch.float32)
        timesteps = noise_scheduler.timesteps

        progress_iter = (
            tqdm(timesteps, desc="UNet-only denoising") if verbose else timesteps
        )

        for timestep in progress_iter:
            timestep_tensor = torch.tensor([timestep], device=self.device)

            with torch.no_grad():
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
        """Initialize output manager.

        Args:
            output_dir: Output directory path.
            logger: Logger instance.
        """
        self.output_dir = output_dir
        self.logger = logger
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_image(
        self, image_tensor: torch.Tensor, filename: str, is_grayscale: bool = True
    ) -> Path:
        """Save image tensor to file.

        Args:
            image_tensor: Image tensor to save.
            filename: Output filename.
            is_grayscale: Whether to save as grayscale.

        Returns:
            Path to saved image.
        """
        # Convert tensor to numpy
        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Normalize to [0, 1] range
        image_np = np.clip(image_np, 0, 1)

        output_path = self.output_dir / filename

        # Create and save figure
        fig, ax = plt.subplots(figsize=(8, 8))

        if is_grayscale or image_np.shape[2] == 1:
            if image_np.shape[2] == 1:
                image_np = image_np[:, :, 0]
            ax.imshow(image_np, cmap="gray")
        else:
            ax.imshow(image_np)

        ax.axis("off")
        plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        return output_path

    def create_comparison_grid(self, results: List[Dict[str, Any]]) -> Path:
        """Create 4-column comparison grid.

        Args:
            results: List of result dictionaries containing image paths.

        Returns:
            Path to saved grid image.
        """
        self.logger.info("Creating 4-column comparison grid...")

        num_masks = len(results)
        fig, axes = plt.subplots(num_masks, 4, figsize=(20, 5 * num_masks))

        # Reshape axes for consistent indexing
        if num_masks == 1:
            axes = axes.reshape(1, -1)

        # Column titles
        col_titles = ["Mask", "ControlNet Output", "Empty Mask Output", "UNet Only"]
        for i, title in enumerate(col_titles):
            fig.text(
                0.125 + 0.25 * i, 0.98, title, ha="center", fontsize=14, weight="bold"
            )

        # Fill grid
        for i, result in enumerate(results):
            # Column 0: Original mask
            self._add_image_to_grid(
                axes[i, 0],
                result["mask_path"],
                f"Mask: {result['mask_path'].name}",
                cmap="gray",
            )

            # Column 1: ControlNet output
            self._add_image_to_grid(
                axes[i, 1], result["controlnet_images"][0], "ControlNet", cmap="gray"
            )

            # Column 2: Empty mask output
            self._add_image_to_grid(
                axes[i, 2], result["empty_mask_images"][0], "Empty Mask", cmap="gray"
            )

            # Column 3: UNet only output
            self._add_image_to_grid(
                axes[i, 3], result["unet_only_images"][0], "UNet Only", cmap="gray"
            )

        # Save grid
        grid_path = self.output_dir / "comparison_grid_4col.png"
        plt.tight_layout()
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Saved 4-column comparison grid to {grid_path}")
        return grid_path

    def _add_image_to_grid(
        self, ax, image_path: Path, title: str, cmap: str = None
    ) -> None:
        """Add image to grid subplot."""
        image = Image.open(image_path).convert("L" if cmap == "gray" else "RGB")
        image_np = np.array(image)

        if cmap == "gray":
            image_np = image_np / 255.0

        ax.imshow(image_np, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")


class ControlNetInference:
    """Main ControlNet inference pipeline with 4-column comparison output."""

    def __init__(
        self,
        config_path: str,
        mask_path: str,
        output_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """Initialize ControlNet inference pipeline.

        Args:
            config_path: Path to configuration JSON file.
            mask_path: Directory containing mask files.
            output_dir: Output directory (creates timestamped dir if None).
            seed: Random seed for reproducibility.
        """
        self.mask_path = Path(mask_path)
        self.seed = seed

        # Setup components
        self.logger = self._setup_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            output_dir = f"./outputs/controlnet_inference_{timestamp}"

        self.output_manager = OutputManager(Path(output_dir), self.logger)
        self.logger.info(f"Output directory: {self.output_manager.output_dir}")

        # Initialize configuration and models
        self.config = InferenceConfig(config_path)
        self.model_manager = ModelManager(self.config, self.device, self.logger)
        self.image_generator = ImageGenerator(self.device, seed, self.logger)

        # Load models
        self.models = self.model_manager.load_all_models()

        # Create noise directory
        self.noise_dir = self.output_manager.output_dir / "noise_vectors"
        self.noise_dir.mkdir(exist_ok=True)

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

    def generate_noise_vectors(
        self, num_samples: int, latent_shape: Tuple[int, ...]
    ) -> List[torch.Tensor]:
        """Generate or load noise vectors for sampling."""
        existing_vectors = sorted(self.noise_dir.glob("noise_*.pt"))

        if len(existing_vectors) >= num_samples:
            self.logger.info(f"Using {num_samples} existing noise vectors")
            return [
                torch.load(v, map_location=self.device)
                for v in existing_vectors[:num_samples]
            ]

        # Generate new noise vectors
        torch.manual_seed(self.seed)
        self.logger.info(
            f"Generating {num_samples} noise vectors with seed {self.seed}"
        )

        noise_vectors = []
        for i in range(num_samples):
            noise = torch.randn(latent_shape, device=self.device, dtype=torch.float32)
            torch.save(noise, self.noise_dir / f"noise_{i:03d}.pt")
            noise_vectors.append(noise)

        return noise_vectors

    def process_single_mask(
        self,
        mask_path: Path,
        noise_vectors: List[torch.Tensor],
        mask_idx: int,
        num_samples: int = 1,
    ) -> Dict[str, Any]:
        """Process a single mask with all generation methods.

        Args:
            mask_path: Path to mask file.
            noise_vectors: List of noise vectors.
            mask_idx: Index of current mask.
            num_samples: Number of samples to generate.

        Returns:
            Dictionary containing paths to generated images.
        """
        autoencoder, unet, controlnet, noise_scheduler, scale_factor = self.models
        recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(
            self.device
        )

        # Load and prepare mask
        mask = Image.open(mask_path).convert("L")
        mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        mask_tensor = (
            mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
        )
        mask_channels = split_grayscale_to_channels(mask_tensor)

        # Create empty mask (all zeros)
        empty_mask = torch.zeros_like(mask_tensor)
        empty_mask = split_grayscale_to_channels(empty_mask)

        mask_name = mask_path.stem

        # Storage for results
        controlnet_images = []
        empty_mask_images = []
        unet_only_images = []

        for i, noise in enumerate(noise_vectors[:num_samples]):
            noise = noise.to(dtype=torch.float32)

            try:
                # Generate with ControlNet (original mask)
                self.logger.info(f"Generating ControlNet output {i+1}/{num_samples}")
                controlnet_latent = self.image_generator.denoise_with_controlnet(
                    unet,
                    controlnet,
                    noise_scheduler,
                    mask_channels,
                    noise,
                    verbose=True,
                )
                controlnet_image = self._decode_and_normalize(
                    recon_model, controlnet_latent
                )
                controlnet_path = self.output_manager.save_image(
                    controlnet_image,
                    f"{mask_name}_controlnet_m{mask_idx:02d}_s{i:03d}.png",
                )
                controlnet_images.append(controlnet_path)

                # Generate with ControlNet (empty mask)
                self.logger.info(f"Generating empty mask output {i+1}/{num_samples}")
                empty_mask_latent = self.image_generator.denoise_with_controlnet(
                    unet, controlnet, noise_scheduler, empty_mask, noise, verbose=True
                )
                empty_mask_image = self._decode_and_normalize(
                    recon_model, empty_mask_latent
                )
                empty_mask_path = self.output_manager.save_image(
                    empty_mask_image,
                    f"{mask_name}_empty_mask_m{mask_idx:02d}_s{i:03d}.png",
                )
                empty_mask_images.append(empty_mask_path)

                # Generate with UNet only
                self.logger.info(f"Generating UNet-only output {i+1}/{num_samples}")
                unet_latent = self.image_generator.denoise_with_unet_only(
                    unet, noise_scheduler, noise, verbose=True
                )
                unet_image = self._decode_and_normalize(recon_model, unet_latent)
                unet_path = self.output_manager.save_image(
                    unet_image, f"{mask_name}_unet_only_m{mask_idx:02d}_s{i:03d}.png"
                )
                unet_only_images.append(unet_path)

                # Clear GPU memory
                torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"Error processing sample {i+1}: {e}")
                continue

        return {
            "mask_path": mask_path,
            "controlnet_images": controlnet_images,
            "empty_mask_images": empty_mask_images,
            "unet_only_images": unet_only_images,
        }

    def _decode_and_normalize(
        self, recon_model: ReconModel, latent: torch.Tensor
    ) -> torch.Tensor:
        """Decode latent and normalize to [0,1] range."""
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
            image = recon_model(latent).to(dtype=torch.float32)

        # Normalize from [-1, 1] to [0, 1]
        image = torch.clip(image, -1.0, 1.0).cpu()
        image = (image + 1.0) / 2.0

        return image

    def run(self) -> None:
        """Run the complete inference pipeline."""
        # Find mask files and generate noise vectors
        mask_files = self.find_mask_files()
        noise_vectors = self.generate_noise_vectors(
            num_samples=1, latent_shape=(1, 4, 64, 64)
        )

        # Process all masks
        all_results = []
        start_time = time.time()

        for i, mask_file in enumerate(mask_files):
            self.logger.info(
                f"\nProcessing mask {i+1}/{len(mask_files)}: {mask_file.name}"
            )
            result = self.process_single_mask(
                mask_file, noise_vectors, i, num_samples=1
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
        description="ControlNet Inference with 4-column comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/config_CONTROLNET_germany.json",
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        # default="/home/user/Thesis/data/retouch_masks/new_mask_dir",
        default="/home/mhuber/Thesis/data/RETOUCH/retouch_masks",
        help="Directory containing mask files",
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
