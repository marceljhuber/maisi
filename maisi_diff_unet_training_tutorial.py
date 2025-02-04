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

from scripts.diff_model_setting import setup_logging
from scripts.diff_model_train import diff_model_train

logger = setup_logging("notebook")


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


def process_image(image_path, output_dir, prefix):
    """Process and save image in standardized format."""
    img = Image.open(image_path).convert("L")
    img = img.resize((256, 256))
    filename = os.path.basename(image_path)
    new_filename = f"{prefix}_{filename}"
    save_path = os.path.join(output_dir, new_filename)
    img.save(save_path)
    return save_path


def log_metrics(losses, epoch, phase="train"):
    """Prepare metrics for wandb without logging."""
    return {f"{phase}/{k}": v for k, v in losses.items()}


def save_prediction_plot(original, prediction, epoch, save_dir):
    """Save prediction plot and return wandb image."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(original, cmap="gray")
    ax1.set_title("Original")
    ax2.imshow(prediction, cmap="gray")
    ax2.set_title("Prediction")
    save_path = Path(save_dir) / f"prediction_epoch_{epoch:03d}.png"
    plt.savefig(save_path)
    plt.close()
    return {"predictions": wandb.Image(str(save_path))}


def setup_training_dirs(config_name, checkpoint_path=None):
    """Sets up training directories and handles checkpoint loading."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"./runs/{timestamp}_{config_name}"
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    model_dir = Path(run_dir) / "models"
    log_dir = Path(run_dir) / "logs"
    output_dir = Path(run_dir) / "predictions"

    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    model_save_path = model_dir / "diffusion_model.pt"
    start_epoch = 0

    if checkpoint_path is not None and checkpoint_path != "None":
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        checkpoint_name = os.path.basename(checkpoint_path)
        if "epoch" in checkpoint_name:
            try:
                start_epoch = int(checkpoint_name.split("epoch")[-1].split(".")[0])
            except ValueError:
                logger.warning("Could not parse epoch number, starting from epoch 0")

    return start_epoch, run_dir, model_save_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/config_DIFF_v1.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="DIFFUSION",
        help="Name for this training run",
    )
    return parser.parse_args()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--config", type=str, default="./configs/config_DIFF_v1.json")
    parser.add_argument("--config_name", type=str, default="DIFFUSION")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint file"
    )
    args = parser.parse_args()

    # Setup
    set_random_seeds()
    start_epoch, run_dir, model_save_path = setup_training_dirs(
        args.config_name, args.checkpoint
    )

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    config["env_config"]["trained_unet_path"] = config["main"]["trained_unet_path"]

    # # Get list of image files
    # image_files = list_image_files(config["main"]["image_dir"])
    # train_imgs, val_imgs = split_train_val_by_patient(image_files)
    #
    # # Create training dataset structure
    # work_dir = os.path.abspath("./temp_work_dir")
    # if not os.path.isdir(work_dir):
    #     os.makedirs(work_dir)
    #
    # dataroot_dir = os.path.join(work_dir, "processed_data")
    # if not os.path.isdir(dataroot_dir):
    #     os.makedirs(dataroot_dir)
    #
    # # Process images
    # processed_train_imgs = []
    # processed_val_imgs = []
    #
    # for img_path in train_imgs:
    #     processed_path = process_image(img_path, dataroot_dir, "train")
    #     processed_train_imgs.append({"image": processed_path})
    #
    # for img_path in val_imgs:
    #     processed_path = process_image(img_path, dataroot_dir, "val")
    #     processed_val_imgs.append({"image": processed_path})
    #
    # # Create datalist
    # datalist = {"training": processed_train_imgs, "validation": processed_val_imgs}
    # datalist_file = os.path.join(work_dir, "datalist.json")
    # with open(datalist_file, "w") as f:
    #     json.dump(datalist, f)

    # Get list of image files directly
    image_files = list_image_files(config["main"]["image_dir"])
    train_imgs, val_imgs = split_train_val_by_patient(image_files)

    # Create datalist directly with image paths
    datalist = {
        "training": [{"image": path} for path in train_imgs],
        "validation": [{"image": path} for path in val_imgs],
    }

    datalist_file = os.path.join(run_dir, "datalist.json")
    with open(datalist_file, "w") as f:
        json.dump(datalist, f)

    # Initialize wandb
    wandb.init(
        project="diffusion-training",
        config=config,
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    print(f"Training directory set up at: {run_dir}")
    if args.checkpoint:
        print(f"Resuming training from checkpoint: {args.checkpoint}")
        print(f"Starting from epoch: {start_epoch}")

    # Save configurations
    env_config_path = os.path.join(run_dir, "environment.json")
    model_config_path = os.path.join(run_dir, "model_config.json")
    model_def_path = os.path.join(run_dir, "model_def.json")

    with open(env_config_path, "w") as f:
        json.dump(config["env_config"], f, indent=4)
    with open(model_config_path, "w") as f:
        json.dump(config["model_config"], f, indent=4)
    with open(model_def_path, "w") as f:
        json.dump(config["model_def"], f, indent=4)

    # Start training
    logger.info("Training the model...")
    diff_model_train(
        env_config_path,
        model_config_path,
        model_def_path,
        num_gpus=1,
        amp=True,
        start_epoch=start_epoch,
        wandb_run=wandb.run,
    )

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
