#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ControlNet Inference Script.

This module provides functionality for running inference with ControlNet models
on image masks. It contains a modular, optimized implementation that supports
reproducible results with deterministic processing.

Typical usage example:
  python controlnet_inference.py --config_path=./configs/my_config.json
                                 --mask_path=./masks_dir
                                 --output_dir=./outputs
"""

import argparse
import os
import sys
import time
import json
import logging
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.amp import autocast
from tqdm import tqdm
from PIL import Image

# Import modules from codebase
from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.sample import ReconModel
from scripts.utils_data import split_grayscale_to_channels
from scripts.utils import define_instance


class ControlNetInference:
    """Main class for ControlNet inference pipeline.

    This class encapsulates the complete workflow for running inference with
    ControlNet models on image masks. It handles configuration, model loading,
    inference processes, and result visualization in an organized way.

    Attributes:
        config_path: Path to the configuration JSON file.
        mask_path: Directory containing mask files.
        output_dir: Directory for saving output files.
        seed: Random seed for reproducibility.
        logger: Configured logging instance.
        device: PyTorch device (CUDA or CPU).
        config: Loaded configuration dictionary.
        args: Namespace of arguments parsed from config.
        models: Tuple containing all loaded models.
    """

    def __init__(
            self,
            config_path: str,
            mask_path: str,
            output_dir: Optional[str] = None,
            seed: int = 42
    ):
        """Initializes the ControlNet inference pipeline.

        Args:
            config_path: Path to config JSON file.
            mask_path: Path to directory containing mask files.
            output_dir: Directory for output files. If None, creates a timestamped dir.
            seed: Random seed for reproducibility.
        """
        self.config_path = config_path
        self.mask_path = Path(mask_path)
        self.seed = seed

        # Set up logger
        self.logger = self._setup_logger()

        # Set deterministic behavior for reproducibility
        self._set_deterministic()

        # Set device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Create output directory with timestamp if not specified
        if output_dir is None:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            self.output_dir = Path("./outputs/retouch") / timestamp
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir}")

        # Load configuration and create args namespace
        self.config = self._load_config()
        self.args = self._create_args()

        # Initialize models
        self.logger.info("Loading models...")
        self.models = self._load_and_prepare_models()

        # Create noise directory
        self.noise_dir = self.output_dir / "noise_vectors"
        self.noise_dir.mkdir(exist_ok=True)

    def _setup_logger(self) -> logging.Logger:
        """Sets up logging configuration.

        Returns:
            A configured Logger instance.
        """
        logger = logging.getLogger("maisi.controlnet.inference")
        logger.setLevel(logging.INFO)

        # Avoid duplicate handlers if logger already exists
        if logger.handlers:
            return logger

        # Create console and file handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler("controlnet_inference.log")

        # Create formatter
        formatter = logging.Formatter(
            "[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Add formatters to handlers
        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

        return logger

    def _set_deterministic(self) -> None:
        """Enables deterministic behavior for PyTorch.

        Sets appropriate flags to ensure reproducible results across runs
        with the same seed.
        """
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from JSON file.

        Returns:
            Dictionary containing the loaded configuration.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            json.JSONDecodeError: If config file has invalid JSON format.
        """
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Error loading config: {e}")
            raise

    def _create_args(self) -> Namespace:
        """Creates argument namespace from config dictionary.

        Converts the JSON configuration into a Namespace object for
        compatibility with existing code.

        Returns:
            Namespace containing all configuration parameters.
        """
        args = Namespace()

        # Merge config sections into args
        for section in ["environment", "model_def", "training", "model"]:
            if section in self.config:
                for k, v in self.config[section].items():
                    setattr(args, k, v)

        # Set model path attributes
        for path_attr in ["trained_autoencoder_path", "trained_diffusion_path", "trained_controlnet_path"]:
            setattr(args, path_attr, self.config["environment"][path_attr])

        return args

    def _load_and_prepare_models(self) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, Any, float]:
        """Loads and prepares all required models.

        Loads the autoencoder, UNet, and ControlNet models from checkpoints,
        initializes the noise scheduler, and converts all models to float32
        for consistent precision.

        Returns:
            Tuple containing (autoencoder, unet, controlnet, noise_scheduler, scale_factor).

        Raises:
            ValueError: If any model checkpoint isn't found.
        """
        # Load autoencoder
        autoencoder = self._load_autoencoder()

        # Load UNet
        unet = self._load_unet()

        # Load ControlNet
        controlnet = self._load_controlnet()

        # Initialize noise scheduler
        noise_scheduler = define_instance(self.args, "noise_scheduler")

        # Get scale factor
        scale_factor = self.scale_factor

        # Set all models to evaluation mode
        autoencoder.eval()
        unet.eval()
        controlnet.eval()

        # Convert models to float32 for consistent precision
        return self._convert_models_to_float32((autoencoder, unet, controlnet, noise_scheduler, scale_factor))

    def _load_autoencoder(self) -> torch.nn.Module:
        """Loads the autoencoder model from checkpoint.

        Returns:
            Loaded autoencoder model.

        Raises:
            ValueError: If autoencoder checkpoint isn't found.
        """
        autoencoder_path = self.args.trained_autoencoder_path
        if not os.path.exists(autoencoder_path):
            raise ValueError(f"Autoencoder checkpoint not found: {autoencoder_path}")

        # Create model from config
        model_config = self.config["model"]["autoencoder"]
        autoencoder = AutoencoderKlMaisi(**model_config).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(
            autoencoder_path,
            map_location=self.device,
            weights_only=True,
        )
        autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])

        self.logger.info(f"Loaded autoencoder from {autoencoder_path}")
        return autoencoder

    def _load_unet(self) -> torch.nn.Module:
        """Loads the UNet model from checkpoint.

        Returns:
            Loaded UNet model and scale factor.

        Raises:
            ValueError: If UNet checkpoint isn't found.
        """
        diffusion_path = self.args.trained_diffusion_path
        if not os.path.exists(diffusion_path):
            raise ValueError(f"Diffusion model checkpoint not found: {diffusion_path}")

        # Create UNet model
        unet = define_instance(self.args, "diffusion_unet_def").to(self.device)

        # Load checkpoint
        diffusion_ckpt = torch.load(
            diffusion_path,
            map_location=self.device,
            weights_only=False,
        )
        unet.load_state_dict(diffusion_ckpt["unet_state_dict"])

        # Store scale factor for later use
        self.scale_factor = diffusion_ckpt.get("scale_factor", 1.0)

        self.logger.info(f"Loaded diffusion model from {diffusion_path}")
        self.logger.info(f"Using scale_factor: {self.scale_factor}")

        return unet

    def _load_controlnet(self) -> torch.nn.Module:
        """Loads the ControlNet model from checkpoint.

        Returns:
            Loaded ControlNet model.

        Raises:
            ValueError: If ControlNet checkpoint isn't found.
        """
        controlnet_path = self.args.trained_controlnet_path
        if not os.path.exists(controlnet_path):
            raise ValueError(f"ControlNet checkpoint not found: {controlnet_path}")

        # Create ControlNet model
        controlnet = define_instance(self.args, "controlnet_def").to(self.device)

        # Load checkpoint
        controlnet_ckpt = torch.load(controlnet_path, map_location=self.device, weights_only=True)
        controlnet.load_state_dict(controlnet_ckpt["controlnet_state_dict"])

        self.logger.info(f"Loaded ControlNet from {controlnet_path}")
        return controlnet

    def _convert_models_to_float32(self, models: Tuple) -> Tuple:
        """Converts model parameters to float32 for consistent precision.

        Args:
            models: Tuple of (autoencoder, unet, controlnet, noise_scheduler, scale_factor).

        Returns:
            Same tuple with models converted to float32.
        """
        autoencoder, unet, controlnet, noise_scheduler, scale_factor = models

        # Convert model parameters to float32
        for model in [autoencoder, unet, controlnet]:
            for param in model.parameters():
                param.data = param.data.to(torch.float32)

        return autoencoder, unet, controlnet, noise_scheduler, scale_factor

    def find_mask_files(self) -> List[Path]:
        """Finds all valid mask files in the mask directory.

        Returns:
            List of paths to valid mask files.

        Raises:
            ValueError: If mask path isn't a directory or no mask files are found.
        """
        if not self.mask_path.is_dir():
            raise ValueError(f"Mask path must be a directory: {self.mask_path}")

        mask_files = sorted(self.mask_path.glob("*.png"))

        if not mask_files:
            raise ValueError(f"No mask files found in {self.mask_path}")

        self.logger.info(f"Found {len(mask_files)} mask files in {self.mask_path}")
        return mask_files

    def generate_noise_vectors(self, num_samples: int, latent_shape: Tuple[int, ...]) -> List[torch.Tensor]:
        """Generates or loads random noise vectors for diffusion sampling.

        Args:
            num_samples: Number of noise vectors to generate.
            latent_shape: Shape of the latent noise vectors.

        Returns:
            List of noise latent tensors.
        """
        # Check if noise vectors already exist
        existing_vectors = sorted(self.noise_dir.glob("noise_*.pt"))

        if len(existing_vectors) >= num_samples:
            self.logger.info(f"Using {num_samples} existing noise vectors from {self.noise_dir}")
            return [torch.load(v, map_location=self.device) for v in existing_vectors[:num_samples]]
        else:
            # Set random seed for reproducibility
            torch.manual_seed(self.seed)
            self.logger.info(f"Generating {num_samples} new noise vectors with seed {self.seed}")

            noise_vectors = []
            for i in range(num_samples):
                noise = torch.randn(latent_shape, device=self.device, dtype=torch.float32)
                torch.save(noise, self.noise_dir / f"noise_{i:03d}.pt")
                noise_vectors.append(noise)

            return noise_vectors

    def denoise_with_controlnet(
            self,
            unet: torch.nn.Module,
            controlnet: torch.nn.Module,
            noise_scheduler: Any,
            condition: torch.Tensor,
            initial_latent: torch.Tensor,
            verbose: bool = False
    ) -> torch.Tensor:
        """Denoises a sample using ControlNet and UNet with deterministic results.

        Args:
            unet: UNet model.
            controlnet: ControlNet model.
            noise_scheduler: Noise scheduler.
            condition: Condition/mask tensor.
            initial_latent: Initial noise latent.
            verbose: Whether to show progress bar.

        Returns:
            Denoised latent tensor.
        """
        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Create a copy of the input latent to avoid modifying the original
        latents = initial_latent.clone().to(dtype=torch.float32)

        # Ensure condition is float32
        condition = condition.to(dtype=torch.float32)

        # Get all timesteps from the noise scheduler
        timesteps = noise_scheduler.timesteps

        # Setup progress tracking
        progress_iter = tqdm(enumerate(timesteps), total=len(timesteps)) if verbose else enumerate(timesteps)

        # Denoise step by step
        for i, t in progress_iter:
            # Get current timestep as tensor
            current_timestep = torch.tensor([t], device=self.device)

            # Process through ControlNet and UNet with consistent precision
            with torch.no_grad():
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=latents,
                    timesteps=current_timestep,
                    controlnet_cond=condition,
                )

                # Pass the ControlNet outputs to UNet
                noise_pred = unet(
                    x=latents,
                    timesteps=current_timestep,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

                # Update latent
                latents, _ = noise_scheduler.step(noise_pred, t, latents)

            # Clear intermediate tensors
            del noise_pred, down_block_res_samples, mid_block_res_sample

        return latents

    def denoise_without_controlnet(
            self,
            unet: torch.nn.Module,
            noise_scheduler: Any,
            initial_latent: torch.Tensor,
            verbose: bool = False
    ) -> torch.Tensor:
        """Denoises a sample using only UNet (no ControlNet) with deterministic results.

        Args:
            unet: UNet model.
            noise_scheduler: Noise scheduler.
            initial_latent: Initial noise latent.
            verbose: Whether to show progress bar.

        Returns:
            Denoised latent tensor.
        """
        # Set seed for reproducibility
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Create a copy of the input latent to avoid modifying the original
        latents = initial_latent.clone().to(dtype=torch.float32)

        # Get all timesteps from the noise scheduler
        timesteps = noise_scheduler.timesteps

        # Setup progress tracking
        progress_iter = tqdm(enumerate(timesteps), total=len(timesteps)) if verbose else enumerate(timesteps)

        # Denoise step by step
        for i, t in progress_iter:
            # Get current timestep as tensor
            current_timestep = torch.tensor([t], device=self.device)

            # Process through UNet with consistent precision
            with torch.no_grad():
                # Get noise prediction directly from UNet
                noise_pred = unet(
                    x=latents,
                    timesteps=current_timestep,
                )

                # Update latent
                latents, _ = noise_scheduler.step(noise_pred, t, latents)

            # Clear intermediate tensors
            del noise_pred

        return latents

    def save_generated_image(
            self,
            generated_image: torch.Tensor,
            prefix: str,
            sample_idx: int,
            mask_idx: int = 0,
            is_grayscale: bool = False
    ) -> Path:
        """Saves a generated image to disk.

        Args:
            generated_image: Generated image tensor.
            prefix: Prefix for the filename.
            sample_idx: Sample index.
            mask_idx: Mask index to prevent overwriting.
            is_grayscale: Whether to save as grayscale.

        Returns:
            Path to the saved image.
        """
        # Convert tensor to numpy for matplotlib
        gen_img_np = generated_image.squeeze(0).permute(1, 2, 0).numpy()

        # Normalize to 0-1 range if needed
        if gen_img_np.min() < 0 or gen_img_np.max() > 1:
            gen_img_np = (gen_img_np - gen_img_np.min()) / (gen_img_np.max() - gen_img_np.min())

        # Define output path with mask index to prevent overwriting
        output_path = self.output_dir / f"{prefix}_mask{mask_idx:02d}_sample{sample_idx:03d}.png"

        # Create figure and save image with appropriate colormap
        fig = plt.figure(figsize=(8, 8))

        # Use grayscale colormap if specified
        if is_grayscale or gen_img_np.shape[2] == 1:
            # If single channel, use first channel
            if gen_img_np.shape[2] == 1:
                gen_img_np = gen_img_np[:, :, 0]
            plt.imshow(gen_img_np, cmap="gray")
        else:
            plt.imshow(gen_img_np)

        plt.axis("off")
        plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        return output_path

    def process_mask(
            self,
            mask_path: Path,
            noise_vectors: List[torch.Tensor],
            mask_idx: int,
            num_samples: int = 1
    ) -> Dict[str, Any]:
        """Processes a mask with both ControlNet and UNet.

        Generates images for a given mask using both ControlNet-guided
        and unguided (UNet only) inference.

        Args:
            mask_path: Path to the mask file.
            noise_vectors: List of noise vectors to use.
            mask_idx: Index of the current mask to prevent overwriting.
            num_samples: Number of samples to generate per mask.

        Returns:
            Dictionary containing the results with paths to saved images.
        """
        autoencoder, unet, controlnet, noise_scheduler, scale_factor = self.models

        # Create reconstruction model
        recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(self.device)

        # Load and prepare mask
        mask = Image.open(mask_path).convert("L")
        mask_tensor = torch.from_numpy(np.array(mask)).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)

        # Split grayscale mask to channels if needed
        mask_channels = split_grayscale_to_channels(mask_tensor)

        # Get mask prefix for naming output files
        mask_prefix = mask_path.stem

        # Generate images
        with_controlnet_images = []
        without_controlnet_images = []

        for i, noise in enumerate(noise_vectors[:num_samples]):
            self.logger.info(f"Processing noise vector {i+1}/{num_samples}")

            try:
                # Ensure noise vector is float32
                noise = noise.to(dtype=torch.float32)

                # First, generate image without ControlNet
                self.logger.info(f"Generating original (no ControlNet) for vector {i+1}")
                original_latent = self.denoise_without_controlnet(
                    unet=unet,
                    noise_scheduler=noise_scheduler,
                    initial_latent=noise,
                    verbose=True
                )

                # Decode the latent with autocast
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
                    original_image = recon_model(original_latent)
                    original_image = original_image.to(dtype=torch.float32)

                # Normalize to 0-1 range
                b_min, b_max = -1.0, 1.0
                original_image = torch.clip(original_image, b_min, b_max).cpu()
                original_image = (original_image - b_min) / (b_max - b_min)

                # Save the original image
                original_path = self.save_generated_image(
                    generated_image=original_image,
                    prefix="original",
                    sample_idx=i,
                    mask_idx=mask_idx,
                    is_grayscale=True  # Save as grayscale if appropriate
                )

                without_controlnet_images.append(original_path)
                self.logger.info(f"Saved original image to {original_path}")

                # Clear memory
                del original_latent, original_image
                torch.cuda.empty_cache()

                # Now generate with ControlNet
                self.logger.info(f"Generating ControlNet version for vector {i+1} with mask {mask_prefix}")
                controlled_latent = self.denoise_with_controlnet(
                    unet=unet,
                    controlnet=controlnet,
                    noise_scheduler=noise_scheduler,
                    condition=mask_channels,
                    initial_latent=noise,
                    verbose=True
                )

                # Decode the latent with autocast
                with torch.no_grad(), autocast(device_type="cuda", dtype=torch.float32):
                    controlled_image = recon_model(controlled_latent)
                    controlled_image = controlled_image.to(dtype=torch.float32)

                # Normalize to 0-1 range
                controlled_image = torch.clip(controlled_image, b_min, b_max).cpu()
                controlled_image = (controlled_image - b_min) / (b_max - b_min)

                # Save the controlled image
                controlled_path = self.save_generated_image(
                    generated_image=controlled_image,
                    prefix=mask_prefix,
                    sample_idx=i,
                    mask_idx=mask_idx,
                    is_grayscale=True  # Save as grayscale if appropriate
                )

                with_controlnet_images.append(controlled_path)
                self.logger.info(f"Saved ControlNet image to {controlled_path}")

                # Clear memory
                del controlled_latent, controlled_image
                torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"Error processing vector {i+1}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

        return {
            "mask_path": mask_path,
            "controlnet_images": with_controlnet_images,
            "original_images": without_controlnet_images
        }

    def create_comparison_grid(self, results: List[Dict[str, Any]]) -> Path:
        """Creates a visualization grid comparing all processed masks.

        Args:
            results: List of dictionaries containing mask paths and generated images.

        Returns:
            Path to the saved grid image.
        """
        self.logger.info("Creating comparison grid...")

        num_masks = len(results)
        fig, axes = plt.subplots(num_masks, 3, figsize=(15, 5 * num_masks))

        # If only one mask, reshape axes for proper indexing
        if num_masks == 1:
            axes = axes.reshape(1, -1)

        # Column titles
        col_titles = ["Mask", "ControlNet Output", "Original (No ControlNet)"]
        for i, title in enumerate(col_titles):
            fig.text(0.15 + 0.35 * i, 0.98, title, ha='center', fontsize=14)

        # Fill the grid
        for i, result in enumerate(results):
            # First column: mask
            mask_img = Image.open(result["mask_path"]).convert("L")
            mask_np = np.array(mask_img) / 255.0
            axes[i, 0].imshow(mask_np, cmap='gray')  # Always use grayscale for masks
            mask_name = result["mask_path"].name
            axes[i, 0].set_title(f"Mask: {mask_name}")
            axes[i, 0].axis('off')

            # Second column: ControlNet output
            ctrl_image = Image.open(result["controlnet_images"][0])
            ctrl_np = np.array(ctrl_image)
            axes[i, 1].imshow(ctrl_np, cmap='gray')  # Use grayscale for ControlNet output
            axes[i, 1].set_title(f"ControlNet")
            axes[i, 1].axis('off')

            # Third column: Original image (no ControlNet)
            orig_image = Image.open(result["original_images"][0])
            orig_np = np.array(orig_image)
            axes[i, 2].imshow(orig_np, cmap='gray')  # Use grayscale for original output
            axes[i, 2].set_title(f"UNet Only")
            axes[i, 2].axis('off')

        # Save the grid
        grid_path = self.output_dir / "complete_comparison_grid.png"
        plt.tight_layout()
        plt.savefig(grid_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"Saved complete comparison grid to {grid_path}")
        return grid_path

    def run(self) -> None:
        """Runs the complete ControlNet inference pipeline.

        This method orchestrates the entire inference process:
        1. Finds all mask files
        2. Generates noise vectors
        3. Processes each mask with both UNet and ControlNet
        4. Creates visualization grid
        """
        # Find mask files
        mask_files = self.find_mask_files()

        # Generate noise vectors
        noise_vectors = self.generate_noise_vectors(
            num_samples=1,  # Only need one noise vector per mask
            latent_shape=(1, 4, 64, 64)
        )

        # Process each mask
        all_results = []
        start_time = time.time()

        for i, mask_file in enumerate(mask_files):
            self.logger.info(f"\nProcessing mask: {mask_file}")
            result = self.process_mask(
                mask_path=mask_file,
                noise_vectors=noise_vectors,
                mask_idx=i,  # Pass mask index to prevent overwriting
                num_samples=1  # Just one sample per mask
            )
            all_results.append(result)

        total_time = time.time() - start_time
        self.logger.info(f"\nProcessed {len(all_results)} masks in {total_time:.2f} seconds")

        # Create comparison grid
        self.create_comparison_grid(all_results)
        self.logger.info(f"Output directory: {self.output_dir}")


def parse_args() -> argparse.Namespace:
    """Parses command line arguments.

    Returns:
        Namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="ControlNet Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to config JSON file",
        default="./configs/config_CONTROLNET_germany.json"
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        help="Path to directory containing mask files",
        default="/home/user/Thesis/data/retouch_masks/new_mask_dir"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=324568,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main() -> None:
    """Main function that coordinates the ControlNet inference process."""
    try:
        # Parse command line arguments
        args = parse_args()

        # Create inference pipeline
        pipeline = ControlNetInference(
            config_path=args.config_path,
            mask_path=args.mask_path,
            output_dir=args.output_dir,
            seed=args.seed
        )

        # Run inference
        pipeline.run()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()