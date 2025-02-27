###################################################################################################
# IMPORTS
###################################################################################################
import argparse
import json
import os
import random
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import torch
from torch.cuda import amp
import torch.multiprocessing as mp
import torch.nn.functional as F
from monai.config import print_config
from monai.transforms import SaveImage
from monai.utils import set_determinism

from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from scripts.utils import define_instance

print_config()


###################################################################################################
# RANDOM SEEDS
###################################################################################################
def set_seeds(seed=42):
    set_determinism(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multiple GPUs
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False  # Set to False for performance
    torch.backends.cudnn.benchmark = True  # Set to True for performance


###################################################################################################
# CONFIG
###################################################################################################
def load_configs(config_path="./configs/config_INFERENCE_v2.json"):
    """
    Load configurations from a single JSON file into an argparse.Namespace object
    """
    args = argparse.Namespace()

    # Load the consolidated config file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Process paths section
    for k, v in config["paths"].items():
        setattr(args, k, v)
        print(f"{k}: {v}")

    # Process model configuration
    for k, v in config["model_config"].items():
        setattr(args, k, v)

    # Process inference configuration
    for k, v in config["inference"]["diffusion_unet_inference"].items():
        setattr(args, k, v)

    # Process training configuration
    for k, v in config["training"]["diffusion_unet_train"].items():
        setattr(args, k, v)

    # Process scheduler configurations
    for scheduler_type, scheduler_config in config["schedulers"].items():
        setattr(args, scheduler_type, scheduler_config)

    # Set model definitions
    for model_key in [
        "autoencoder_def",
        "diffusion_unet_def",
    ]:
        if model_key in config["model_config"]:
            # Resolve @ references
            model_config = config["model_config"][model_key].copy()
            for k, v in model_config.items():
                if isinstance(v, str) and v.startswith("@"):
                    ref_key = v[1:]
                    model_config[k] = config["model_config"][ref_key]
            setattr(args, model_key, model_config)

    # Calculate and set latent shape
    args.latent_shape = [
        args.latent_channels,
        args.dim[0] // 4,  # Using dim from inference config
        args.dim[1] // 4,
    ]

    # Add batch generation parameters with defaults if not present in config
    batch_generation = config.get("batch_generation", {})
    args.batch_size = batch_generation.get("batch_size", 64)
    args.num_workers = batch_generation.get("num_workers", 4)
    args.prefetch_factor = batch_generation.get("prefetch_factor", 2)
    args.pin_memory = batch_generation.get("pin_memory", True)
    args.save_interval = batch_generation.get("save_interval", 100)
    args.checkpoint_interval = batch_generation.get("checkpoint_interval", 1000)

    # Disable mixed precision to avoid data type issues
    args.use_mixed_precision = False

    return args


###################################################################################################
# MODEL LOADING
###################################################################################################
def load_autoencoder(trained_autoencoder_path, device):
    model_config = {
        "spatial_dims": 2,
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": 4,
        "num_channels": [64, 128, 256],
        "num_res_blocks": [2, 2, 2],
        "norm_num_groups": 32,
        "norm_eps": 1e-6,
        "attention_levels": [False, False, False],
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
        "use_checkpointing": False,
        "use_convtranspose": False,
        "norm_float16": False,  # Set to False to use float32
        "num_splits": 1,
        "dim_split": 1,
    }

    autoencoder = AutoencoderKlMaisi(**model_config).to(device)
    checkpoint = torch.load(trained_autoencoder_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])

    # Force model to use float32
    autoencoder = autoencoder.float()
    return autoencoder


def load_models(args, device):
    # Load autoencoder
    autoencoder = load_autoencoder(args.trained_autoencoder_path, device)
    autoencoder.eval()

    # Load diffusion model
    diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
    checkpoint_diffusion_unet = torch.load(
        args.trained_diffusion_path, weights_only=False
    )
    diffusion_unet.load_state_dict(
        checkpoint_diffusion_unet["unet_state_dict"], strict=True
    )
    diffusion_unet.eval()

    # Force model to use float32
    diffusion_unet = diffusion_unet.float()

    # Get scale factor and ensure it's float32
    scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device).float()

    # Load noise scheduler
    noise_scheduler = define_instance(args, "noise_scheduler")

    return autoencoder, diffusion_unet, noise_scheduler, scale_factor


###################################################################################################
# OPTIMIZED BATCH INFERENCE
###################################################################################################
class ReconModel(torch.nn.Module):
    """
    A PyTorch module for reconstructing images from latent representations.
    """

    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        """
        Decode the input latent representation to an image.
        Ensure z is float32 before decoding.
        """
        # Ensure z is in float32 format
        z = z.float()
        return self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)


def initialize_noise_latents(batch_size, latent_shape, device):
    """
    Initialize random noise latents for batch image generation.
    Use float32 to match model weights.
    """
    return torch.randn([batch_size] + list(latent_shape), device=device)


def process_images(synthetic_images):
    """Process synthetic images to the correct format and orientation"""
    # PNG image intensity range
    a_min = 0
    a_max = 255
    # autoencoder output intensity range
    b_min = -1.0
    b_max = 1.0

    # Clip values
    synthetic_images = torch.clip(synthetic_images, b_min, b_max).cpu()

    # Project output to [0, 1]
    synthetic_images = (synthetic_images - b_min) / (b_max - b_min)

    # Project output to [0, 255]
    synthetic_images = synthetic_images * (a_max - a_min) + a_min

    # Rotate image to correct orientation (rotate 90 degrees clockwise to fix 270 degree rotation)
    # For a tensor with shape [batch, channel, height, width]
    synthetic_images = synthetic_images.transpose(2, 3).flip(2)

    return synthetic_images


def save_images_parallel(images, output_dir, image_output_ext, start_idx):
    """Save images in parallel using ThreadPoolExecutor"""

    def save_single_image(args):
        idx, img = args
        output_postfix = (
            f"{start_idx + idx:06d}_{datetime.now().strftime('%m%d_%H%M%S')}"
        )

        img_saver = SaveImage(
            output_dir=output_dir,
            output_postfix=output_postfix,
            output_ext=image_output_ext,
            separate_folder=False,
        )

        img_saver(img)
        return os.path.join(output_dir, output_postfix + image_output_ext)

    with ThreadPoolExecutor(max_workers=min(os.cpu_count(), 16)) as executor:
        filenames = list(executor.map(save_single_image, enumerate(images)))

    return filenames


def generate_images_in_batches(
    args,
    autoencoder,
    diffusion_unet,
    noise_scheduler,
    scale_factor,
    device,
    total_images=100000,
):
    """
    Generate a large number of images in batches with optimized GPU usage.
    """
    batch_size = args.batch_size
    latent_shape = args.latent_shape
    output_dir = args.output_dir
    num_inference_steps = args.num_inference_steps
    image_output_ext = args.image_output_ext

    # Create reconstruction model
    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(
        device
    )

    # Initialize timesteps for the diffusion process
    noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    timesteps = noise_scheduler.timesteps

    # We don't need a GradScaler for inference-only code
    # as it's only used for training with mixed precision

    # Keep track of filenames
    all_filenames = []

    # Generate images in batches
    num_batches = (total_images + batch_size - 1) // batch_size
    with tqdm(total=total_images, desc="Generating images") as pbar:
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, total_images - batch_idx * batch_size)
            if current_batch_size <= 0:
                break

            # Generate a batch of images
            with torch.no_grad():
                # Mixed precision context
                # Initialize random noise (always using float32 for consistency)
                latents = (
                    initialize_noise_latents(current_batch_size, latent_shape, device)
                    * 1.0
                )  # noise_factor = 1.0

                # Run diffusion process
                for t in timesteps:
                    t_tensor = torch.tensor([t] * current_batch_size, device=device)
                    noise_pred = diffusion_unet(latents, t_tensor)
                    latents, _ = noise_scheduler.step(noise_pred, t, latents)

                    # Explicitly free memory
                    del noise_pred

                # Decode latents to images
                synthetic_images = recon_model(latents)

                # Initialize random noise
                latents = (
                    initialize_noise_latents(current_batch_size, latent_shape, device)
                    * 1.0
                )  # noise_factor = 1.0

                # Run diffusion process
                for t in timesteps:
                    t_tensor = torch.tensor([t] * current_batch_size, device=device)
                    noise_pred = diffusion_unet(latents, t_tensor)
                    latents, _ = noise_scheduler.step(noise_pred, t, latents)

                    # Explicitly free memory
                    del noise_pred

                # Decode latents to images
                synthetic_images = recon_model(latents)

                # Process images
                processed_images = process_images(synthetic_images)

                # Free memory
                del latents, synthetic_images
                torch.cuda.empty_cache()

            # Save images in parallel
            image_idx_start = batch_idx * batch_size
            filenames = save_images_parallel(
                processed_images, output_dir, image_output_ext, image_idx_start
            )
            all_filenames.extend(filenames)

            # Update progress bar
            pbar.update(current_batch_size)

            # Checkpoint progress periodically
            if (batch_idx + 1) % args.checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    output_dir, f"generation_checkpoint_{batch_idx}.json"
                )
                with open(checkpoint_path, "w") as f:
                    json.dump(
                        {
                            "completed_batches": batch_idx + 1,
                            "total_images_generated": (batch_idx + 1) * batch_size,
                        },
                        f,
                    )

    return all_filenames


###################################################################################################
# MAIN FUNCTION
###################################################################################################
def main():
    # Load configuration
    args = load_configs()

    # Set up seeds for reproducibility
    random_seed = getattr(args, "random_seed", 42)
    set_seeds(random_seed)

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        print("CUDA is not available, using CPU")

    # Enable TF32 precision on Ampere GPUs for increased performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load models
    print("Loading models...")
    autoencoder, diffusion_unet, noise_scheduler, scale_factor = load_models(
        args, device
    )

    # Log time
    start_time = time.time()

    # Generate images
    print(f"Starting batch generation of {args.num_output_samples} images...")
    output_filenames = generate_images_in_batches(
        args,
        autoencoder,
        diffusion_unet,
        noise_scheduler,
        scale_factor,
        device,
        total_images=args.num_output_samples,
    )

    # Calculate time taken
    end_time = time.time()
    total_time = end_time - start_time
    images_per_second = args.num_output_samples / total_time
    print(f"Generation completed in {total_time:.2f} seconds")
    print(f"Average generation speed: {images_per_second:.2f} images/second")
    print(f"Generated {len(output_filenames)} images in {args.output_dir}")


if __name__ == "__main__":
    # Enable easier debugging of CUDA errors
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Set to "1" for debugging

    # Handle process-specifics for multi-GPU setups
    mp.set_start_method("spawn", force=True)

    main()
