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
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from monai.networks.utils import copy_model_state
from monai.utils import RankFilter
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from monai.data.meta_tensor import MetaTensor

from .utils import (
    binarize_labels,
    define_instance,
    prepare_maisi_controlnet_json_dataloader,
    setup_ddp,
)

from scripts.utils_data import setup_training, create_latent_dataloaders
from scripts.utils_data import setup_training


def main():
    parser = argparse.ArgumentParser(description="maisi.controlnet.training")
    parser.add_argument(
        "--config_path",
        default="./configs/config_CONTROLNET_v2.json",
        help="config json file that stores controlnet settings",
    )
    # parser.add_argument(
    #     "-e",
    #     "--environment-file",
    #     default="./configs_old/environment_maisi_controlnet_train.json",
    #     help="environment json file that stores environment path",
    # )
    # parser.add_argument(
    #     "-c",
    #     "--config-file",
    #     default="./configs_old/config_maisi.json",
    #     help="config json file that stores network hyper-parameters",
    # )
    # parser.add_argument(
    #     "-t",
    #     "--training-config",
    #     default="./configs_old/config_maisi_controlnet_train.json",
    #     help="config json file that stores training hyper-parameters",
    # )
    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of gpus per node"
    )
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = json.load(f)

    # Step 0: configuration
    logger = logging.getLogger("maisi.controlnet.training")
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

    env_dict = config["environment"]
    model_def_dict = config["model_def"]
    training_dict = config["training"]

    for k, v in env_dict.items():
        setattr(args, k, v)
    for k, v in model_def_dict.items():
        setattr(args, k, v)
    for k, v in training_dict.items():
        setattr(args, k, v)

    # Step 1: set data loader
    # device, run_dir, recon_dir, train_loader, val_loader = setup_training(config)
    print(f"latent_dir:", args.latent_dir)
    train_loader, val_loader = create_latent_dataloaders(args.latent_dir)

    # Step 2: define diffusion model and controlnet
    # define diffusion Model
    unet = define_instance(args, "diffusion_unet_def").to(device)
    # load trained diffusion model
    if args.trained_diffusion_path is not None:
        if not os.path.exists(args.trained_diffusion_path):
            raise ValueError("Please download the trained diffusion unet checkpoint.")

        diffusion_model_ckpt = torch.load(
            args.trained_diffusion_path,
            map_location=device,
            weights_only=False,
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
        logger.info("train controlnet model from scratch.")
    # we freeze the parameters of the diffusion model.
    for p in unet.parameters():
        p.requires_grad = False

    noise_scheduler = define_instance(args, "noise_scheduler")

    if use_ddp:
        controlnet = DDP(
            controlnet,
            device_ids=[device],
            output_device=rank,
            find_unused_parameters=True,
        )

    # Step 3: training config
    weighted_loss = args.controlnet_train["weighted_loss"]
    weighted_loss_label = args.controlnet_train["weighted_loss_label"]
    optimizer = torch.optim.AdamW(
        params=controlnet.parameters(), lr=args.controlnet_train["lr"]
    )
    total_steps = (
        args.controlnet_train["n_epochs"] * len(train_loader.dataset)
    ) / args.controlnet_train["batch_size"]
    logger.info(f"total number of training steps: {total_steps}.")

    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=total_steps, power=2.0
    )

    # Step 4: training
    n_epochs = args.controlnet_train["n_epochs"]
    scaler = GradScaler("cuda")
    total_step = 0
    best_loss = 1e4

    if weighted_loss > 1.0:
        logger.info(
            f"apply weighted loss = {weighted_loss} on labels: {weighted_loss_label}"
        )

    controlnet.train()
    unet.eval()
    prev_time = time.time()
    for epoch in range(n_epochs):
        epoch_loss_ = 0
        for step, batch in enumerate(train_loader):
            # get image embedding and label mask and scale image embedding by the provided scale_factor
            # inputs = batch["image"].to(device) * scale_factor
            # labels = batch["label"].to(device)
            inputs = batch["latent"].squeeze(1).to(device) * scale_factor
            labels = batch["label"].to(device)

            labels = labels.unsqueeze(-1).unsqueeze(-1)  # Now shape [40, 4, 1, 1]
            labels = F.interpolate(
                labels, size=(256, 256), mode="bilinear", align_corners=False
            )  # Now shape [40, 4, 256, 256, 1]

            # labels = torch.zeros_like(inputs, dtype=torch.float32)
            # labels[1] = labels_[-1]
            # labels = batch["label"].to(device)
            # print("===" * 20)
            # print(f"inputs.shape:", inputs.shape)
            # print(f"labels.shape:", labels.shape)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=True):
                # generate random noise
                noise_shape = list(inputs.shape)
                noise = torch.randn(noise_shape, dtype=inputs.dtype).to(device)
                # print(f"noise.shape:", noise.shape)

                # controlnet_cond = binarize_labels(
                #     labels.to(
                #         torch.uint8
                #     )  # Remove the as_tensor() call since labels is already a tensor
                # ).float()
                controlnet_cond = labels.float()
                # print(f"controlnet_cond.shape:", controlnet_cond.shape)
                # print("===" * 20)

                # create timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (inputs.shape[0],),
                    device=device,
                ).long()

                # create noisy latent
                noisy_latent = noise_scheduler.add_noise(
                    original_samples=inputs, noise=noise, timesteps=timesteps
                )

                # get controlnet output
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=noisy_latent, timesteps=timesteps, controlnet_cond=controlnet_cond
                )
                # get noise prediction from diffusion unet
                noise_pred = unet(
                    x=noisy_latent,
                    timesteps=timesteps,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )

            ##########################################################################
            if weighted_loss > 1.0:
                weights = torch.ones_like(inputs).to(inputs.device)

                # Create a mask in float32 format
                roi_mask = torch.zeros_like(inputs[:, :1], dtype=torch.float32)

                interpolate_label = F.interpolate(
                    labels, size=inputs.shape[2:], mode="nearest"
                )

                # For each target label, add to the mask
                for label in weighted_loss_label:
                    for channel in range(interpolate_label.shape[1]):
                        # Create mask for this channel/label combination
                        channel_mask = (
                            interpolate_label[:, channel : channel + 1] == label
                        ).float()
                        # Add masks instead of OR operation
                        roi_mask = roi_mask + channel_mask

                # Convert to binary mask (any positive value becomes 1)
                roi_mask = (roi_mask > 0).float()

                # Apply the mask to weights
                weights = weights.masked_fill(
                    roi_mask.repeat(1, inputs.shape[1], 1, 1) > 0, weighted_loss
                )

                loss = (
                    F.l1_loss(noise_pred.float(), noise.float(), reduction="none")
                    * weights
                ).mean()
            else:
                loss = F.l1_loss(noise_pred.float(), noise.float())
            ##########################################################################

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            total_step += 1

            if rank == 0:
                batches_done = step + 1
                batches_left = len(train_loader) - batches_done
                time_left = timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                if (step - 1) % 500 == 0:
                    logger.info(
                        "\r[Epoch %d/%d] [Batch %d/%d] [LR: %.8f] [loss: %.4f] ETA: %s "
                        % (
                            epoch + 1,
                            n_epochs,
                            step + 1,
                            len(train_loader),
                            lr_scheduler.get_last_lr()[0],
                            loss.detach().cpu().item(),
                            time_left,
                        )
                    )
            epoch_loss_ += loss.detach()

        epoch_loss = epoch_loss_ / (step + 1)

        if use_ddp:
            dist.barrier()
            dist.all_reduce(epoch_loss, op=torch.distributed.ReduceOp.AVG)

        if rank == 0:
            # save controlnet only on master GPU (rank 0)
            controlnet_state_dict = (
                controlnet.module.state_dict()
                if world_size > 1
                else controlnet.state_dict()
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "controlnet_state_dict": controlnet_state_dict,
                },
                f"{args.model_dir}/{args.exp_name}_current.pt",
            )

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                logger.info(f"best loss -> {best_loss}.")
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "loss": best_loss,
                        "controlnet_state_dict": controlnet_state_dict,
                    },
                    f"{args.model_dir}/{args.exp_name}_best.pt",
                )

        torch.cuda.empty_cache()
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
