# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import numpy as np
from monai.data import decollate_batch, MetaTensor
from monai.networks.utils import copy_model_state
from monai.transforms import SaveImage
from monai.utils import RankFilter
import nibabel as nib
from PIL import Image
import random

from .sample import ldm_conditional_sample_one_image_controlnet
from .utils import define_instance, prepare_maisi_controlnet_json_dataloader, setup_ddp

from scripts.utils_data import setup_training
from networks.autoencoderkl_maisi import AutoencoderKlMaisi


def save_as_png(image_tensor, output_path):
    """
    Save a tensor as a PNG image.

    Args:
        image_tensor: The image tensor to save
        output_path: Path to save the PNG image
    """
    # Convert tensor to numpy and normalize to 0-255 range
    img_np = image_tensor.cpu().numpy()

    # Normalize to 0-255 range
    img_np = np.clip(img_np, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)

    # If 3D volume, save middle slice
    if len(img_np.shape) == 3:
        middle_slice_idx = img_np.shape[0] // 2
        img_np = img_np[middle_slice_idx, :, :]

    # Create PIL image and save
    img_pil = Image.fromarray(img_np)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img_pil.save(output_path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="maisi.controlnet.infer")
    parser.add_argument(
        "--config_path",
        default="./configs/config_CONTROLNET_v1.json",
        help="config json file that stores controlnet settings",
    )
    args = parser.parse_args()

    args.gpus = 1

    # Step 0: configuration
    logger = logging.getLogger("maisi.controlnet.infer")
    # whether to use distributed data parallel
    use_ddp = args.gpus > 1
    if use_ddp:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = setup_ddp(rank, world_size)
        logger.addFilter(RankFilter())
    else:
        rank = 0
        world_size = 1
        device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"World_size: {world_size}")

    with open(args.config_path, "r") as f:
        config = json.load(f)

    env_dict = config["environment"]
    model_def_dict = config["model_def"]
    training_dict = config["training"]

    # with open(args.environment_file, "r") as env_file:
    #     env_dict = json.load(env_file)
    # with open(args.config_file, "r") as config_file:
    #     config_dict = json.load(config_file)
    # with open(args.training_config, "r") as training_config_file:
    #     training_config_dict = json.load(training_config_file)

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in model_def_dict.items():
        setattr(args, k, v)
    for k, v in training_dict.items():
        setattr(args, k, v)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: set data loader
    # Use the setup_training function from utils_data instead

    # Get device, run directories, and data loaders
    device, run_dir, recon_dir, train_loader, val_loader = setup_training(config)

    # Step 2: define AE, diffusion model and controlnet
    # define AE
    # load trained autoencoder model
    if args.trained_autoencoder_path is not None:
        if not os.path.exists(args.trained_autoencoder_path):
            raise ValueError("Please download the autoencoder checkpoint.")

        model_config = config["model"]["autoencoder"]

        # Load model
        autoencoder = AutoencoderKlMaisi(**model_config).to(device)
        checkpoint = torch.load(
            config["environment"]["trained_autoencoder_path"],
            map_location=device,
            weights_only=True,
        )
        autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
        autoencoder.eval()
    else:
        logger.info("trained autoencoder model is not loaded.")

    # define diffusion Model
    unet = define_instance(args, "diffusion_unet_def").to(device)
    # load trained diffusion model
    if args.trained_diffusion_path is not None:
        if not os.path.exists(args.trained_diffusion_path):
            raise ValueError("Please download the trained diffusion unet checkpoint.")
        diffusion_model_ckpt = torch.load(
            args.trained_diffusion_path, map_location=device
        )
        unet.load_state_dict(diffusion_model_ckpt["unet_state_dict"])
        # load scale factor from diffusion model checkpoint
        scale_factor = diffusion_model_ckpt["scale_factor"]
        logger.info(f"Load trained diffusion model from {args.trained_diffusion_path}.")
        logger.info(f"loaded scale_factor from diffusion model ckpt -> {scale_factor}.")
    else:
        logger.info("trained diffusion model is not loaded.")
        scale_factor = 1.0
        logger.info(f"set scale_factor -> {scale_factor}.")

    # define ControlNet
    controlnet = define_instance(args, "controlnet_def").to(device)
    # copy weights from the DM to the controlnet
    copy_model_state(controlnet, unet.state_dict())
    # load trained controlnet model if it is provided
    if args.trained_controlnet_path is not None:
        if not os.path.exists(args.trained_controlnet_path):
            raise ValueError("Please download the trained ControlNet checkpoint.")
        controlnet.load_state_dict(
            torch.load(args.trained_controlnet_path, map_location=device)[
                "controlnet_state_dict"
            ]
        )
        logger.info(
            f"load trained controlnet model from {args.trained_controlnet_path}"
        )
    else:
        logger.info("trained controlnet is not loaded.")

    noise_scheduler = define_instance(args, "noise_scheduler")

    # Step 3: inference
    autoencoder.eval()
    controlnet.eval()
    unet.eval()

    # output_filenames = []

    for batch_idx, batch in enumerate(val_loader):
        print(f"Starting batch {batch_idx}/{len(val_loader)}.")
        # The GrayscaleDatasetLabels class already provides labels in format [batch_size, num_classes, H, W]
        # These labels are already one-hot encoded in the dataset
        labels = batch["label"].to(device)

        # Ensure the labels have the correct shape [batch_size, 4, 256, 256]
        batch_size = labels.shape[0]
        if labels.shape[1] != 4 or labels.shape[2] != 256 or labels.shape[3] != 256:
            # Resize or reshape if necessary
            resized_labels = torch.zeros((batch_size, 4, 256, 256), device=device)

            # Copy available classes (up to 4)
            num_classes = min(labels.shape[1], 4)
            resized_labels[:, :num_classes] = labels[
                :, :num_classes, : labels.shape[2], : labels.shape[3]
            ]

            labels = resized_labels

        print(f"Label:", labels.shape)

        def find_ones_dimension(tensor):
            """Returns index (0-3) of dimension that contains all 1s in tensor of shape (n,4,256,256)"""
            sums = tensor.sum(dim=(0, 2, 3))
            expected_sum = tensor.shape[0] * 256 * 256
            return torch.where(sums == expected_sum)[0].item()

        label = find_ones_dimension(labels)

        # Handle the new batch structure from GrayscaleDatasetLabels
        # The new dataset might not provide these tensors, so we'll create defaults

        # Handle dimensions from the new dataset format
        if "dim" in batch:
            dim = batch["dim"]
            output_size = (dim[0].item(), dim[1].item(), dim[2].item())
        else:
            # Use image dimensions if available, otherwise use default size
            if "image" in batch:
                img_shape = batch["image"].shape
                if len(img_shape) >= 3:
                    # For 2D images: use [C, H, W] but add a depth dimension
                    output_size = (img_shape[-2], img_shape[-1], 1)
                else:
                    # Default size if can't determine from image
                    output_size = (256, 256, 1)
            else:
                # Default size
                output_size = (256, 256, 1)

        # check if output_size and out_spacing are valid.
        # check_input(None, None, None, output_size, None)

        latent_shape = [4, 64, 64]

        # generate a single synthetic image using a latent diffusion model with controlnet.
        synthetic_image, _ = ldm_conditional_sample_one_image_controlnet(
            autoencoder=autoencoder,
            diffusion_unet=unet,
            controlnet=controlnet,
            noise_scheduler=noise_scheduler,
            scale_factor=scale_factor,
            device=device,
            combine_label_or=labels,
            latent_shape=latent_shape,
            noise_factor=1,
            num_inference_steps=100,
        )

        # Rotate image to correct orientation (rotate 90 degrees clockwise to fix 270 degree rotation)
        # For a tensor with shape [batch, channel, height, width]
        synthetic_image = synthetic_image.transpose(2, 3).flip(2)

        # Modified: Save images as PNG
        timestamp = datetime.now().strftime("%Y%m%d_%H%M_%f")
        timestamp += f"_{label}"

        # Save the generated image
        img_saver = SaveImage(
            output_dir=args.output_dir,
            output_postfix=timestamp,
            output_ext=".png",
            separate_folder=False,
        )
        img_saver(synthetic_image[0])

        # output_filenames.append(synthetic_image_filename)

        # print(f"    For i in range:", synthetic_images.shape[0])
        # for i in range(synthetic_images.shape[0]):
        #     # Get class index for this sample (if available)
        #     class_idx = i % 4  # Default to i mod 4
        #     if "class_idx" in batch and i < len(batch["class_idx"]):
        #         class_idx = batch["class_idx"][i].item()
        #
        #     # Get patient ID (if available)
        #     patient_id = f"patient{batch_idx}_{i}"
        #     if "patient_id" in batch and i < len(batch["patient_id"]):
        #         patient_id = str(batch["patient_id"][i])
        #
        #     # Save synthetic image as PNG with class and patient info
        #     image_filename = f"class{class_idx}-{patient_id}_{timestamp}_image.png"
        #     image_path = os.path.join(args.output_dir, image_filename)
        #
        #     # Save label image as PNG
        #     label_filename = f"class{class_idx}-{patient_id}_{timestamp}_label.png"
        #     label_path = os.path.join(args.output_dir, label_filename)
        #     save_as_png(labels[i], label_path)
        #
        #     logger.info(f"Saved sample {i} from batch {batch_idx} to {image_path}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d][%(levelname)5s](%(name)s) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
