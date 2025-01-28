###################################################################################################
# IMPORTS
###################################################################################################
import argparse
import json
import os
import random
import tempfile

import monai
import numpy as np
import torch
from monai.config import print_config
from monai.transforms import LoadImage, Orientation
from monai.utils import set_determinism

from scripts.sample import LDMSampler
from scripts.utils import define_instance
from scripts.utils_plot import find_label_center_loc, get_xyz_plot, show_image

print_config()
###################################################################################################
# RANDOM SEEDS
###################################################################################################
seed = 42
set_determinism(seed=seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # CUDA
torch.cuda.manual_seed_all(seed)  # multiple GPUs
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###################################################################################################
# PATHS
###################################################################################################
directory = os.environ.get("MONAI_DATA_DIRECTORY")
if directory is not None:
    os.makedirs(directory, exist_ok=True)
root_dir = tempfile.mkdtemp() if directory is None else directory

autoencoder_path = "./models/autoencoder_epoch17.pt"
diffusion_path = "./models/checkpoint_epoch_5.pt"
###################################################################################################


###################################################################################################
# CONFIG
###################################################################################################
def load_configs(config_path="./configs/config_INFERENCE_v1.json"):
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
        if "datasets/" in str(v):
            v = os.path.join(root_dir, v)
        print(f"{k}: {v}")
    print("Global config variables have been loaded.")

    # print(f"args.latent_channels:", args.latent_channels)
    print(f"config.latent_channels:", config["model_config"]["latent_channels"])
    # Process model configuration
    for k, v in config["model_config"].items():
        setattr(args, k, v)

    # Process inference configuration
    for k, v in config["inference"]["diffusion_unet_inference"].items():
        setattr(args, k, v)
        print(f"{k}: {v}")

    # Process training configuration
    for k, v in config["training"]["diffusion_unet_train"].items():
        setattr(args, k, v)

    # Process scheduler configurations
    for scheduler_type, scheduler_config in config["schedulers"].items():
        setattr(args, scheduler_type, scheduler_config)

    # Process mask generation configuration
    for k, v in config["mask_generation"].items():
        setattr(args, k, v)

    # Set model definitions
    for model_key in [
        "autoencoder_def",
        "controlnet_def",
        "diffusion_unet_def",
        "mask_generation_autoencoder_def",
        "mask_generation_diffusion_def",
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
    print(f"args.latent_channels:", args.latent_channels)
    args.latent_shape = [
        args.latent_channels,
        args.dim[0] // 4,  # Using dim from inference config
        args.dim[1] // 4,
    ]
    print(f"latent_shape: {args.latent_shape}")

    print("Network definition and inference inputs have been loaded.")
    return args


# Load all configs
args = load_configs()
###################################################################################################


###################################################################################################
# INITIALIZATIONnoise_scheduler = define_instance(args, "noise_scheduler")
mask_generation_noise_scheduler = define_instance(
    args, "mask_generation_noise_scheduler"
)

device = torch.device("cuda")

autoencoder = define_instance(args, "autoencoder_def").to(device)
checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
autoencoder.load_state_dict(checkpoint_autoencoder)

diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
checkpoint_diffusion_unet = torch.load(args.trained_diffusion_path, weights_only=False)
diffusion_unet.load_state_dict(
    checkpoint_diffusion_unet["unet_state_dict"], strict=True
)
scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)

controlnet = define_instance(args, "controlnet_def").to(device)
checkpoint_controlnet = torch.load(args.trained_controlnet_path, weights_only=False)
monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
controlnet.load_state_dict(checkpoint_controlnet["controlnet_state_dict"], strict=True)

mask_generation_autoencoder = define_instance(
    args, "mask_generation_autoencoder_def"
).to(device)
checkpoint_mask_generation_autoencoder = torch.load(
    args.trained_mask_generation_autoencoder_path, weights_only=True
)
mask_generation_autoencoder.load_state_dict(checkpoint_mask_generation_autoencoder)

mask_generation_diffusion_unet = define_instance(
    args, "mask_generation_diffusion_def"
).to(device)
checkpoint_mask_generation_diffusion_unet = torch.load(
    args.trained_mask_generation_diffusion_path, weights_only=True
)
mask_generation_diffusion_unet.load_state_dict(
    checkpoint_mask_generation_diffusion_unet["unet_state_dict"]
)
mask_generation_scale_factor = checkpoint_mask_generation_diffusion_unet["scale_factor"]

print("All the trained model weights have been loaded.")
###################################################################################################
noise_scheduler = define_instance(args, "noise_scheduler")
mask_generation_noise_scheduler = define_instance(
    args, "mask_generation_noise_scheduler"
)

device = torch.device("cuda")

autoencoder = define_instance(args, "autoencoder_def").to(device)
checkpoint_autoencoder = torch.load(args.trained_autoencoder_path, weights_only=True)
autoencoder.load_state_dict(checkpoint_autoencoder)

diffusion_unet = define_instance(args, "diffusion_unet_def").to(device)
checkpoint_diffusion_unet = torch.load(args.trained_diffusion_path, weights_only=False)
diffusion_unet.load_state_dict(
    checkpoint_diffusion_unet["unet_state_dict"], strict=True
)
scale_factor = checkpoint_diffusion_unet["scale_factor"].to(device)

# controlnet = define_instance(args, "controlnet_def").to(device)
# checkpoint_controlnet = torch.load(args.trained_controlnet_path, weights_only=False)
# monai.networks.utils.copy_model_state(controlnet, diffusion_unet.state_dict())
# controlnet.load_state_dict(checkpoint_controlnet["controlnet_state_dict"], strict=True)

# mask_generation_autoencoder = define_instance(args, "mask_generation_autoencoder_def").to(device)
# checkpoint_mask_generation_autoencoder = torch.load(args.trained_mask_generation_autoencoder_path, weights_only=True)
# mask_generation_autoencoder.load_state_dict(checkpoint_mask_generation_autoencoder)

# mask_generation_diffusion_unet = define_instance(args, "mask_generation_diffusion_def").to(device)
# checkpoint_mask_generation_diffusion_unet = torch.load(args.trained_mask_generation_diffusion_path, weights_only=True)
# mask_generation_diffusion_unet.load_state_dict(checkpoint_mask_generation_diffusion_unet["unet_state_dict"])
# mask_generation_scale_factor = checkpoint_mask_generation_diffusion_unet["scale_factor"]

print("All the trained model weights have been loaded.")
###################################################################################################


###################################################################################################
# LDM SAMPLER
###################################################################################################
ldm_sampler = LDMSampler(
    args.body_region,
    args.anatomy_list,
    args.all_mask_files_json,
    args.all_anatomy_size_conditions_json,
    args.all_mask_files_base_dir,
    args.label_dict_json,
    args.label_dict_remap_json,
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
    args.mask_generation_latent_shape,
    args.output_size,
    args.output_dir,
    args.controllable_anatomy_size,
    image_output_ext=args.image_output_ext,
    label_output_ext=args.label_output_ext,
    spacing=args.spacing,
    num_inference_steps=args.num_inference_steps,
    mask_generation_num_inference_steps=args.mask_generation_num_inference_steps,
    random_seed=args.random_seed,
    autoencoder_sliding_window_infer_size=args.autoencoder_sliding_window_infer_size,
    autoencoder_sliding_window_infer_overlap=args.autoencoder_sliding_window_infer_overlap,
)
###################################################################################################


###################################################################################################
# INFERENCE
###################################################################################################
print(f"The generated image/mask pairs will be saved in {args.output_dir}.")
output_filenames = ldm_sampler.sample_multiple_images(args.num_output_samples)
print("MAISI image/mask generation finished")
###################################################################################################


###################################################################################################
# VISUALIZE THE RESULTS
###################################################################################################
visualize_image_filename = output_filenames[0][0]
visualize_mask_filename = output_filenames[0][1]
print(f"Visualizing {visualize_image_filename} and {visualize_mask_filename}...")

# load image/mask pairs
loader = LoadImage(image_only=True, ensure_channel_first=True)
orientation = Orientation(axcodes="RAS")
image_volume = orientation(loader(visualize_image_filename))
mask_volume = orientation(loader(visualize_mask_filename)).to(torch.uint8)

# visualize for CT HU intensity between [-200, 500]
image_volume = torch.clip(image_volume, -200, 500)
image_volume = image_volume - torch.min(image_volume)
image_volume = image_volume / torch.max(image_volume)

# create a random color map for mask visualization
colorize = torch.clip(
    torch.cat([torch.zeros(3, 1, 1, 1), torch.randn(3, 200, 1, 1)], 1), 0, 1
)
target_class_index = 1

# find center voxel location for 2D slice visualization
center_loc_axis = find_label_center_loc(
    torch.flip(mask_volume[0, ...] == target_class_index, [-3, -2, -1])
)

# visualization
vis_mask = get_xyz_plot(
    mask_volume,
    center_loc_axis,
    mask_bool=True,
    n_label=201,
    colorize=colorize,
    target_class_index=target_class_index,
)
show_image(vis_mask, title="mask")

vis_image = get_xyz_plot(image_volume, center_loc_axis, mask_bool=False)
show_image(vis_image, title="image")
###################################################################################################
