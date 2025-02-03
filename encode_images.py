import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from networks.autoencoderkl_maisi import AutoencoderKlMaisi
from tqdm import tqdm


def load_configs(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    args = argparse.Namespace()
    for k, v in config["model_config"].items():
        setattr(args, k, v)
    for k, v in config["paths"].items():
        setattr(args, k, v)
    args.dim = config["inference"]["diffusion_unet_inference"]["dim"]
    args.latent_shape = [args.latent_channels, args.dim[0] // 4, args.dim[1] // 4]
    return args


def load_autoencoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = {
        "autoencoder": {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 4,
            "num_channels": [64, 128, 256],
            "num_res_blocks": [2, 2, 2],
            "norm_num_groups": 32,
            "norm_eps": 1e-6,
            "attention_levels": [False, False, False],
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
            "use_checkpointing": False,
            "use_convtranspose": False,
            "norm_float16": True,
            "num_splits": 1,
            "dim_split": 1,
        }
    }
    autoencoder = AutoencoderKlMaisi(**model["autoencoder"]).to(device)
    checkpoint = torch.load(args.trained_autoencoder_path, map_location=device)
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    return autoencoder


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2100], std=[0.0300]),
        ]
    )
    img = Image.open(image_path).convert("L")
    return transform(img).unsqueeze(0)


def get_all_images(directory):
    image_paths = []
    for ext in [".jpg", ".jpeg", ".png"]:
        image_paths.extend(Path(directory).rglob(f"*{ext}"))
    return sorted(image_paths)


def encode_images(autoencoder, image_paths):
    device = next(autoencoder.parameters()).device
    encoded_vectors = []
    file_paths = []

    for img_path in tqdm(image_paths, desc="Encoding images"):
        try:
            x = preprocess_image(str(img_path)).to(device)
            with torch.no_grad():
                with torch.amp.autocast(device.type):
                    z = autoencoder.encode(x)[0]  # Get mean from encoder output
                    encoded_vectors.append(z.cpu().numpy())
                    file_paths.append(str(img_path))
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    return np.array(encoded_vectors), file_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Directory containing images"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to config file"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    config = load_configs(args.config_path)
    autoencoder = load_autoencoder(config)
    autoencoder.eval()

    # Get all images
    image_paths = get_all_images(args.input_dir)
    print(f"Found {len(image_paths)} images")

    # Encode images
    encoded_vectors, file_paths = encode_images(autoencoder, image_paths)

    # Calculate statistics
    stats = {
        "mean": encoded_vectors.mean(axis=0),
        "std": encoded_vectors.std(axis=0),
        "min": encoded_vectors.min(axis=0),
        "max": encoded_vectors.max(axis=0),
    }

    # Save results
    np.save(os.path.join(args.output_dir, "encoded_vectors.npy"), encoded_vectors)
    with open(os.path.join(args.output_dir, "file_paths.json"), "w") as f:
        json.dump(file_paths, f, indent=2)
    np.savez(os.path.join(args.output_dir, "statistics.npz"), **stats)

    # Print statistics
    print("\nLatent Space Statistics:")
    print(f"Shape of encoded vectors: {encoded_vectors.shape}")
    print(f"Mean: {stats['mean'].mean():.4f}")
    print(f"Std: {stats['std'].mean():.4f}")
    print(f"Min: {stats['min'].min():.4f}")
    print(f"Max: {stats['max'].max():.4f}")


if __name__ == "__main__":
    main()
