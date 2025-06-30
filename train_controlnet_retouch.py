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

import copy
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import torch
from monai.config import print_config

from scripts.utils_data import (
    set_random_seeds,
)

print_config()

########################################################################################################################
# Step 1: Training Config Preparation
########################################################################################################################
config_path = "./configs/config_CONTROLNET_hungary.json"

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
# Step 3: Train the Model
########################################################################################################################
import importlib
import sys

# Set environment variable to limit OpenMP threads
os.environ["OMP_NUM_THREADS"] = "1"

# Import the module
sys.path.append(".")
# train_module = importlib.import_module("scripts.train_controlnet_retouch")
train_module = importlib.import_module("scripts.train_controlnet_retouch_newobjective")

# Set up sys.argv to simulate command line arguments
# sys.argv = ["train_controlnet_retouch.py", "--config", config_path]
sys.argv = ["train_controlnet_retouch_newobjective.py", "--config", config_path]

# Call the main function
print("Training the model...")
train_module.main()