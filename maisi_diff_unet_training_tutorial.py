import argparse
import glob
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.diff_model_setting import setup_logging

logger = setup_logging("notebook")

###################################################################################################
# RANDOM SEEDS
###################################################################################################
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # CUDA
torch.cuda.manual_seed_all(seed)  # multiple GPUs
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


###################################################################################################
# UTILS
###################################################################################################
def list_image_files(directory_path):
    # Define common image file extensions
    image_extensions = (".jpg", ".jpeg", ".png")

    # Use glob to find all files in directory and subdirectories
    files = glob.glob(os.path.join(directory_path, "**", "*.*"), recursive=True)

    # Filter files for image extensions
    image_files = [file for file in files if file.lower().endswith(image_extensions)]

    return image_files


def split_train_val_by_patient(image_names, train_ratio=0.9):
    # Extract unique patient IDs
    patient_ids = set(name.split("-")[1] for name in image_names)

    # Random split of patient IDs
    num_train = int(len(patient_ids) * train_ratio)
    train_patients = set(random.sample(list(patient_ids), num_train))

    # Split images based on patient IDs
    train_images = [img for img in image_names if img.split("-")[1] in train_patients]
    val_images = [img for img in image_names if img.split("-")[1] not in train_patients]

    return train_images, val_images


###################################################################################################
# PREPARE DATASET
###################################################################################################
# Get list of image files
directory_path = "/optima/exchange/mhuber/KermanyV3_resized/train"
image_files = list_image_files(directory_path)
train_imgs, val_imgs = split_train_val_by_patient(image_files)

# Create training dataset structure
work_dir = os.path.abspath("./temp_work_dir")
if not os.path.isdir(work_dir):
    os.makedirs(work_dir)

dataroot_dir = os.path.join(work_dir, "processed_data")
if not os.path.isdir(dataroot_dir):
    os.makedirs(dataroot_dir)


# Process and save images in standardized format
def process_image(image_path, output_dir, prefix):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((256, 256))  # Resize to 256x256

    # Create standardized filename
    filename = os.path.basename(image_path)
    new_filename = f"{prefix}_{filename}"
    save_path = os.path.join(output_dir, new_filename)

    # Save processed image
    img.save(save_path)
    return save_path


# Process training and validation images
processed_train_imgs = []
processed_val_imgs = []

for img_path in train_imgs:
    processed_path = process_image(img_path, dataroot_dir, "train")
    processed_train_imgs.append({"image": processed_path})

for img_path in val_imgs:
    processed_path = process_image(img_path, dataroot_dir, "val")
    processed_val_imgs.append({"image": processed_path})

# Create datalist
datalist = {"training": processed_train_imgs, "validation": processed_val_imgs}

datalist_file = os.path.join(work_dir, "datalist.json")
with open(datalist_file, "w") as f:
    json.dump(datalist, f)

###################################################################################################
# SETUP
###################################################################################################
with open("configs/config_DIFF_v1.json", "r") as f:
    config = json.load(f)

model_config = config["model_config"]
env_config = config["env_config"]
model_def = config["model_def"]

# Save configurations
os.makedirs(env_config["model_dir"], exist_ok=True)
os.makedirs(env_config["output_dir"], exist_ok=True)
os.makedirs(env_config["embedding_base_dir"], exist_ok=True)
os.makedirs(env_config["log_dir"], exist_ok=True)

env_config_filepath = os.path.join(work_dir, "environment.json")
model_config_filepath = os.path.join(work_dir, "model_config.json")
model_def_filepath = os.path.join(work_dir, "model_def.json")

with open(env_config_filepath, "w") as f:
    json.dump(env_config, f, sort_keys=True, indent=4)
with open(model_config_filepath, "w") as f:
    json.dump(model_config, f, sort_keys=True, indent=4)
with open(model_def_filepath, "w") as f:
    json.dump(model_def, f, sort_keys=True, indent=4)

print(f"Dumped all config settings to JSON-files.")
###################################################################################################


###################################################################################################
# MAIN METHOD
###################################################################################################
# Add argument parsing at the start of your script
def parse_args():
    parser = argparse.ArgumentParser(description="Train diffusion model")
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


###################################################################################################


###################################################################################################
def setup_training_dirs(config_name, checkpoint_path=None):
    """
    Sets up training directories and handles checkpoint loading

    Args:
        config_name: Name of the configuration/run
        checkpoint_path: Optional path to checkpoint to resume from

    Returns:
        start_epoch: Epoch to start training from
        run_dir: Directory for this training run
        model_save_path: Path where model checkpoints will be saved
    """
    # Create runs directory structure
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    run_dir = f"./runs/{timestamp}_{config_name}"
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    model_dir = os.path.join(run_dir, "models")
    log_dir = os.path.join(run_dir, "logs")
    output_dir = os.path.join(run_dir, "predictions")

    Path(model_dir).mkdir(exist_ok=True)
    Path(log_dir).mkdir(exist_ok=True)
    Path(output_dir).mkdir(exist_ok=True)

    # Set up model save path
    model_save_path = os.path.join(model_dir, "diffusion_model.pt")

    # Handle checkpoint loading
    start_epoch = 0
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        # Extract epoch number from checkpoint filename
        checkpoint_name = os.path.basename(checkpoint_path)
        if "epoch" in checkpoint_name:
            try:
                start_epoch = int(checkpoint_name.split("epoch")[-1].split(".")[0])
            except ValueError:
                print(
                    "Could not parse epoch number from checkpoint filename, starting from epoch 0"
                )

    return start_epoch, run_dir, model_save_path


###################################################################################################


###################################################################################################
# MAIN METHOD
###################################################################################################
logger.info("Training the model...")


# Modify the main training section
if __name__ == "__main__":
    args = parse_args()

    # Setup directories and get starting epoch
    start_epoch, run_dir, model_save_path = setup_training_dirs(
        args.config_name, args.checkpoint
    )

    # Create configs_old with updated paths
    env_config = create_env_config(run_dir, args.checkpoint)

    # Save configurations to new run directory
    env_config_filepath = os.path.join(run_dir, "environment.json")
    model_config_filepath = os.path.join(run_dir, "model_config.json")
    model_def_filepath = os.path.join(run_dir, "model_def.json")

    with open(env_config_filepath, "w") as f:
        json.dump(env_config, f, sort_keys=True, indent=4)
    with open(model_config_filepath, "w") as f:
        json.dump(model_config, f, sort_keys=True, indent=4)
    with open(model_def_filepath, "w") as f:
        json.dump(model_def, f, sort_keys=True, indent=4)

    print(f"Training directory set up at: {run_dir}")
    if args.checkpoint:
        print(f"Resuming training from checkpoint: {args.checkpoint}")
        print(f"Starting from epoch: {start_epoch}")

    from scripts.diff_model_train import diff_model_train

    # Start training
    logger.info("Training the model...")
    diff_model_train(
        env_config_filepath,
        model_config_filepath,
        model_def_filepath,
        num_gpus=1,
        amp=True,
        start_epoch=start_epoch,  # Pass starting epoch to training function
    )
