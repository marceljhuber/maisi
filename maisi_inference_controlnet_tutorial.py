#!/usr/bin/env python
# coding: utf-8

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0

import copy
import json
import os
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

import torch
from monai.config import print_config

from scripts.utils_data import (
    set_random_seeds,
)


# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train and infer ControlNet model")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/config_CONTROLNET_v2.json",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to generate during inference",
    )
    parser.add_argument(
        "--label", type=int, default=0, help="Label to use for inference"
    )
    return parser.parse_args()


# Parse arguments
args = parse_args()
print_config()


########################################################################################################################
# Step 1: Training Config Preparation
########################################################################################################################
config_path = args.config_path

with open(config_path, "r") as f:
    config = json.load(f)

# Prepare training
set_random_seeds()

# Load environment configuration, model configuration and model definition
env_config = config["environment"]
train_config = config["training"]
model_def = config["model_def"]

env_config_out = copy.deepcopy(env_config)
train_config_out = copy.deepcopy(train_config)
model_def_out = copy.deepcopy(model_def)

# Make sure batch size is an integer
if isinstance(train_config_out["batch_size"], str):
    try:
        train_config_out["batch_size"] = int(train_config_out["batch_size"])
    except ValueError:
        train_config_out["batch_size"] = 1
        print(f"WARNING: Could not convert batch_size to integer, setting to 1")

if isinstance(train_config_out["controlnet_train"]["batch_size"], str):
    # If it's a string reference like '@batch_size', resolve it
    if train_config_out["controlnet_train"]["batch_size"] == "@batch_size":
        train_config_out["controlnet_train"]["batch_size"] = train_config_out[
            "batch_size"
        ]
    else:
        try:
            train_config_out["controlnet_train"]["batch_size"] = int(
                train_config_out["controlnet_train"]["batch_size"]
            )
        except ValueError:
            train_config_out["controlnet_train"]["batch_size"] = 1
            print(
                f"WARNING: Could not convert controlnet_train.batch_size to integer, setting to 1"
            )

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup directories
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
run_dir = Path(f"./runs/{config['main']['jobname']}_{timestamp}")
run_dir.mkdir(parents=True, exist_ok=True)
recon_dir = run_dir / "reconstructions"
recon_dir.mkdir(exist_ok=True)

# Set up directories based on configurations
env_config_out["model_dir"] = os.path.join(run_dir, env_config_out["model_dir"])
env_config_out["output_dir"] = os.path.join(run_dir, env_config_out["output_dir"])
env_config_out["tfevent_path"] = os.path.join(run_dir, env_config_out["tfevent_path"])
env_config_out["trained_controlnet_path"] = None
env_config_out["exp_name"] = "tutorial_training_example"


# Create necessary directories
os.makedirs(env_config_out["model_dir"], exist_ok=True)
os.makedirs(env_config_out["output_dir"], exist_ok=True)
os.makedirs(env_config_out["tfevent_path"], exist_ok=True)


# Update training configuration for demo
# train_config_out["controlnet_train"]["n_epochs"] = max_epochs
# We disable weighted_loss for dummy data, which is used to apply more penalty
# to the region of interest (e.g., tumors). When weighted_loss=1,
# we treat all regions equally in loss computation.
train_config_out["controlnet_train"]["weighted_loss"] = 1
# We also set weighted_loss_label to None, which indicates the list of label indices that
# we want to apply more penalty during training.
train_config_out["controlnet_train"]["weighted_loss_label"] = [None]

# Dump the configs to the run_dir
env_config_filepath = os.path.join(run_dir, "config_environment.json")
train_config_filepath = os.path.join(run_dir, "config_train.json")
model_def_filepath = os.path.join(run_dir, "config_model_def.json")

with open(env_config_filepath, "w") as f:
    json.dump(env_config_out, f, sort_keys=True, indent=4)

with open(train_config_filepath, "w") as f:
    json.dump(train_config_out, f, sort_keys=True, indent=4)

with open(model_def_filepath, "w") as f:
    json.dump(model_def_out, f, sort_keys=True, indent=4)

num_gpus = 1


########################################################################################################################
def run_torchrun(module, module_args, num_gpus=1):
    num_nodes = 1
    torchrun_command = [
        "torchrun",
        "--nproc_per_node",
        str(num_gpus),
        "--nnodes",
        str(num_nodes),
        "-m",
        module,
    ] + module_args

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    process = subprocess.Popen(
        torchrun_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    try:
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                print(output.strip())
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stdout, stderr = process.communicate()
        print(stdout)
        if stderr:
            print(stderr)
    return


########################################################################################################################
# Step 3: Train the Model
########################################################################################################################
# Run inference
module = "scripts.infer_controlnet"
module_args = [
    "--config_path",
    config_path,
    "--num_images",
    str(args.num_images),
    "--label",
    str(args.label),
]

run_torchrun(module, module_args, num_gpus=num_gpus)
########################################################################################################################
