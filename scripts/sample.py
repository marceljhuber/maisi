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

import json
import logging
import os
import time
from datetime import datetime

import monai
import torch
from monai.transforms import Compose, SaveImage
from monai.utils import set_determinism
from tqdm import tqdm


class ReconModel(torch.nn.Module):
    """
    A PyTorch module for reconstructing images from latent representations.

    Attributes:
        autoencoder: The autoencoder model used for decoding.
        scale_factor: Scaling factor applied to the input before decoding.
    """

    def __init__(self, autoencoder, scale_factor):
        super().__init__()
        self.autoencoder = autoencoder
        self.scale_factor = scale_factor

    def forward(self, z):
        """
        Decode the input latent representation to an image.

        Args:
            z (torch.Tensor): The input latent representation.

        Returns:
            torch.Tensor: The reconstructed image.
        """
        recon_pt_nda = self.autoencoder.decode_stage_2_outputs(z / self.scale_factor)
        return recon_pt_nda


def initialize_noise_latents(latent_shape, device):
    """
    Initialize random noise latents for image generation with float16.

    Args:
        latent_shape (tuple): The shape of the latent space.
        device (torch.device): The device to create the tensor on.

    Returns:
        torch.Tensor: Initialized noise latents.
    """
    return (
        torch.randn(
            [
                1,
            ]
            + list(latent_shape)
        )
        .half()
        .to(device)
    )


def ldm_conditional_sample_one_image(
    autoencoder,
    diffusion_unet,
    noise_scheduler,
    scale_factor,
    device,
    latent_shape,
    noise_factor,
    num_inference_steps,
):
    """
    Generate a single synthetic image using latent diffusion model without ControlNet.
    """
    # PNG image intensity range
    a_min = 0
    a_max = 255
    # autoencoder output intensity range
    b_min = -1.0
    b_max = 1

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(
        device
    )

    with torch.no_grad(), torch.amp.autocast("cuda"):
        # Generate random noise
        latents = initialize_noise_latents(latent_shape, device) * noise_factor

        # Synthesize latents
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        # for t in tqdm(noise_scheduler.timesteps, ncols=110):
        for t in noise_scheduler.timesteps:  # TODO tqdm
            timesteps = torch.Tensor((t,)).to(device)
            # Just use UNet without ControlNet conditioning
            noise_pred = diffusion_unet(latents, timesteps)
            latents, _ = noise_scheduler.step(noise_pred, t, latents)

        del noise_pred
        torch.cuda.empty_cache()

        # Decode latents to images
        synthetic_images = recon_model(latents)

        ################################################################################################################
        synthetic_images = torch.clip(synthetic_images, b_min, b_max).cpu()
        # project output to [0, 1]
        synthetic_images = (synthetic_images - b_min) / (b_max - b_min)
        # project output to [-1, 1]
        synthetic_images = synthetic_images * (a_max - a_min) + a_min
        torch.cuda.empty_cache()
        ################################################################################################################

    return synthetic_images


def ldm_conditional_sample_one_image_controlnet(
    autoencoder,
    diffusion_unet,
    controlnet,
    noise_scheduler,
    scale_factor,
    device,
    combine_label_or,
    latent_shape,
    noise_factor,
    num_inference_steps=1000,
):
    """
    Generate a single synthetic image using a latent diffusion model with controlnet.

    Args:
        autoencoder (nn.Module): The autoencoder model.
        diffusion_unet (nn.Module): The diffusion U-Net model.
        controlnet (nn.Module): The controlnet model.
        noise_scheduler: The noise scheduler for the diffusion process.
        scale_factor (float): Scaling factor for the latent space.
        device (torch.device): The device to run the computation on.
        combine_label_or (torch.Tensor): The combined label tensor.
        latent_shape (tuple): The shape of the latent space.
        output_size (tuple): The desired output size of the image.
        noise_factor (float): Factor to scale the initial noise.
        num_inference_steps (int): Number of inference steps for the diffusion process.

    Returns:
        tuple: A tuple containing the synthetic image and its corresponding label.
    """
    # PNG image intensity range
    a_min = 0
    a_max = 255
    # autoencoder output intensity range
    b_min = -1.0
    b_max = 1

    torch.cuda.empty_cache()

    recon_model = ReconModel(autoencoder=autoencoder, scale_factor=scale_factor).to(
        device
    )
    # Add at various points to track memory
    # print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    # print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    with torch.no_grad(), torch.amp.autocast("cuda"):
        # generate segmentation mask
        combine_label = combine_label_or.to(device)

        # controlnet_cond_vis = binarize_labels(combine_label.as_tensor().long()).half()
        controlnet_cond_vis = combine_label
        # print(f"controlnet_cond_vis.shape:", controlnet_cond_vis.shape)

        # Generate random noise
        latents = initialize_noise_latents(latent_shape, device) * noise_factor

        # synthesize latents
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
        for step_idx, t in enumerate(noise_scheduler.timesteps):
            # Get controlnet output
            down_block_res_samples, mid_block_res_sample = controlnet(
                x=latents,
                timesteps=torch.Tensor((t,)).to(device),
                controlnet_cond=controlnet_cond_vis,
            )
            latent_model_input = latents
            noise_pred = diffusion_unet(
                x=latent_model_input,
                timesteps=torch.Tensor((t,)).to(device),
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
            latents, _ = noise_scheduler.step(noise_pred, t, latents)
        del noise_pred
        torch.cuda.empty_cache()

        # decode latents to synthesized images
        synthetic_images = recon_model(latents)
        synthetic_images = torch.clip(synthetic_images, b_min, b_max).cpu()

        # post processing:
        # project output to [0, 1]
        synthetic_images = (synthetic_images - b_min) / (b_max - b_min)
        # project output to [-1, 1]
        synthetic_images = synthetic_images * (a_max - a_min) + a_min
        # regularize background intensities
        torch.cuda.empty_cache()

    return synthetic_images, combine_label


class LDMSampler:
    """
    A sampler class for generating synthetic medical images and masks using latent diffusion models.

    Attributes:
        Various attributes related to model configuration, input parameters, and generation settings.
    """

    def __init__(
        self,
        body_region,
        anatomy_list,
        all_mask_files_json,
        all_anatomy_size_condtions_json,
        all_mask_files_base_dir,
        label_dict_json,
        label_dict_remap_json,
        autoencoder,
        diffusion_unet,
        controlnet,
        noise_scheduler,
        scale_factor,
        mask_generation_autoencoder,
        mask_generation_diffusion_unet,
        mask_generation_scale_factor,
        mask_generation_noise_scheduler,
        device,
        latent_shape,
        mask_generation_latent_shape,
        output_size,
        output_dir,
        controllable_anatomy_size,
        image_output_ext,
        label_output_ext=".nii.gz",
        real_img_median_statistics=None,
        spacing=[1, 1, 1],
        num_inference_steps=None,
        mask_generation_num_inference_steps=None,
        random_seed=None,
        autoencoder_sliding_window_infer_size=[96, 96, 96],
        autoencoder_sliding_window_infer_overlap=0.6667,
    ) -> None:
        """
        Initialize the LDMSampler with various parameters and models.

        Args:
            Various parameters related to model configuration, input settings, and output specifications.
        """
        if random_seed is not None:
            set_determinism(seed=random_seed)

        # intialize variables
        self.data_root = all_mask_files_base_dir
        self.autoencoder = autoencoder
        self.diffusion_unet = diffusion_unet
        # self.controlnet = controlnet
        self.noise_scheduler = noise_scheduler
        self.scale_factor = scale_factor
        self.device = device
        self.latent_shape = latent_shape
        # self.output_size = output_size
        self.output_dir = output_dir
        self.noise_factor = 1.0
        self.image_output_ext = image_output_ext
        # self.label_output_ext = label_output_ext
        # Set the default value for number of inference steps to 1000
        self.num_inference_steps = (
            num_inference_steps if num_inference_steps is not None else 1000
        )
        self.mask_generation_num_inference_steps = (
            mask_generation_num_inference_steps
            if mask_generation_num_inference_steps is not None
            else 1000
        )

        with open(real_img_median_statistics, "r") as json_file:
            self.median_statistics = json.load(json_file)

        # networks
        self.autoencoder.eval()
        self.diffusion_unet.eval()
        # self.controlnet.eval()

        self.val_transforms = Compose(
            [
                monai.transforms.LoadImaged(keys=["image"]),
                monai.transforms.EnsureChannelFirstd(keys=["image"]),
                monai.transforms.ScaleIntensityd(keys=["image"], minv=0, maxv=255),
                monai.transforms.EnsureTyped(keys=["image"], dtype=torch.uint8),
            ]
        )
        logging.info("Standard grayscale image transformer initialized.")

        with open("./configs/image_median_statistics.json", "r") as f:
            statistics = json.load(f)
            self.oct_stats = statistics["oct"]

    def sample_multiple_images(self, num_img):
        """
        Generate multiple synthetic images using VAE and diffusion models.

        Args:
            num_img (int): Number of images to generate.
        Returns:
            list: List of paths to generated images.
        """
        output_filenames = []

        for _ in tqdm(range(num_img)):
            logging.info("---- Starting image generation... ----")
            start_time = time.time()

            # Generate image
            to_generate = True
            while to_generate:
                try:
                    # Sample latent using diffusion model
                    # synthetic_image = self.sample_one_image()
                    synthetic_image = ldm_conditional_sample_one_image(
                        autoencoder=self.autoencoder,
                        diffusion_unet=self.diffusion_unet,
                        noise_scheduler=self.noise_scheduler,
                        scale_factor=self.scale_factor,
                        device=self.device,
                        latent_shape=self.latent_shape,
                        noise_factor=self.noise_factor,
                        num_inference_steps=self.num_inference_steps,
                    )
                    # print(f"synthetic_image.shape:", synthetic_image.shape)

                    # Rotate image to correct orientation (rotate 90 degrees clockwise to fix 270 degree rotation)
                    # For a tensor with shape [batch, channel, height, width]
                    synthetic_image = synthetic_image.transpose(2, 3).flip(2)

                    # Save image
                    output_postfix = datetime.now().strftime("%m%d_%H%M%S")

                    # Save the generated image
                    img_saver = SaveImage(
                        output_dir=self.output_dir,
                        output_postfix=output_postfix,
                        output_ext=self.image_output_ext,
                        separate_folder=False,
                    )
                    img_saver(synthetic_image[0])

                    # Get the full path of saved image
                    synthetic_image_filename = os.path.join(
                        self.output_dir,
                        output_postfix + self.image_output_ext,
                    )

                    output_filenames.append(synthetic_image_filename)
                    to_generate = False

                    end_time = time.time()
                    logging.info(
                        f"---- Image generation time: {end_time - start_time} seconds ----"
                    )

                except Exception as e:
                    logging.error(f"Error during image generation: {str(e)}")

                torch.cuda.empty_cache()

        return output_filenames
