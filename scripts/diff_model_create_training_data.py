import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from tqdm import tqdm
from PIL import Image

from utils_data import list_image_files, set_random_seeds, setup_transforms


def process_images(
    input_dir, output_dir, autoencoder_path, skip_existing=True, seed=42
):
    if not Path(input_dir).exists():
        raise FileNotFoundError(f"Input directory {input_dir} not found")

    set_random_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, transforms = setup_transforms()

    # Load config
    with open("./configs/config_VAE_norm_v1.json") as f:
        config = json.load(f)

    model_config = config["model"]["autoencoder"]

    # Load model
    autoencoder = AutoencoderKlMaisi(**model_config).to(device)
    checkpoint = torch.load(autoencoder_path, map_location=device, weights_only=True)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    autoencoder.eval()

    # Create output dir
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process images
    files = list_image_files(input_dir)
    with tqdm(files, desc="Converting images to latents") as pbar:
        for filepath in pbar:
            # Convert string path to Path object
            filepath = Path(filepath)
            out_filename = out_dir / f"{filepath.stem}_latent.pt"

            if skip_existing and out_filename.exists():
                continue

            pbar.set_description(f"Processing {filepath.name}")

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
