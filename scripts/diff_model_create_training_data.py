import json
import sys
import os

from torchvision.transforms import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
import nibabel as nib
import numpy as np
from monai.transforms import Compose
import monai
from networks.autoencoderkl_maisi import AutoencoderKlMaisi
import random
from tqdm import tqdm
from PIL import Image


def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_transforms():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2 * x - 1),  # Scale to [-1, 1]
        ]
    )


def get_image_files(directory):
    extensions = (".png", ".jpg", ".jpeg")
    return sorted(f for ext in extensions for f in Path(directory).rglob(f"*{ext}"))


def process_images(
    input_dir, output_dir, autoencoder_path, skip_existing=True, seed=42
):
    if not Path(input_dir).exists():
        raise FileNotFoundError(f"Input directory {input_dir} not found")

    set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = create_transforms()

    # Load config
    with open("./configs/config_VAE_norm_v1.json") as f:
        config = json.load(f)

    model_config = config["model"]["autoencoder"]
    # model_config = {
    #     "spatial_dims": 2,
    #     "in_channels": 1,
    #     "out_channels": 1,
    #     "latent_channels": 4,
    #     "num_channels": [64, 128, 256],
    #     "num_res_blocks": [2, 2, 2],
    #     "norm_num_groups": 32,
    #     "norm_eps": 1e-6,
    #     "attention_levels": [False, False, False],
    #     "with_encoder_nonlocal_attn": False,
    #     "with_decoder_nonlocal_attn": False,
    # }

    # Load model
    autoencoder = AutoencoderKlMaisi(**model_config).to(device)
    checkpoint = torch.load(autoencoder_path, map_location=device, weights_only=True)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    autoencoder.eval()

    # Create output dir
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    files = get_image_files(input_dir)
    with tqdm(files, desc="Converting images to latents") as pbar:
        for filepath in pbar:
            out_filename = out_dir / f"{filepath.stem}_latent.pt"

            if skip_existing and out_filename.exists():
                continue

            pbar.set_description(f"Processing {filepath.name}")

            # data = {"image": str(filepath)}
            # image = transforms(data)["image"]

            image = Image.open(str(filepath)).convert("L")
            image = transforms(image)

            with torch.no_grad(), torch.amp.autocast("cuda"):
                latent, _ = autoencoder.encode(image.unsqueeze(0).to(device))
                torch.save(latent.cpu(), out_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--autoencoder_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_skip", action="store_false", dest="skip_existing")
    args = parser.parse_args()

    process_images(
        args.input_dir,
        args.output_dir,
        args.autoencoder_path,
        args.skip_existing,
        args.seed,
    )
