import numpy as np
import os
import torch
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3
import argparse
from tqdm import tqdm


class InceptionV3Features:
    def __init__(self, device="cuda"):
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model = inception_v3(pretrained=True).to(self.device)
        self.model.eval()
        # Remove the final classifier layer to get features
        self.model.fc = torch.nn.Identity()
        # Define image transformation
        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_features(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(img)
            return features.cpu().numpy().reshape(-1)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None


def calculate_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet Distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    # Calculate sqrt of product between covariances
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("Warning: Non-finite values in sqrtm calculation.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Check and correct imaginary values
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    trace_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * trace_covmean


def get_all_image_paths(directory):
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def main():
    parser = argparse.ArgumentParser(
        description="Calculate FID score between real and synthetic images"
    )
    parser.add_argument(
        "--real", type=str, required=True, help="Directory containing real images"
    )
    parser.add_argument(
        "--synthetic",
        type=str,
        required=True,
        help="Directory containing synthetic images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of images to process in each batch",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to use from each set",
    )
    args = parser.parse_args()

    print(
        f"Using device: {args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'}"
    )

    real_paths = get_all_image_paths(args.real)
    synthetic_paths = get_all_image_paths(args.synthetic)

    if args.max_images is not None:
        real_paths = real_paths[: args.max_images]
        synthetic_paths = synthetic_paths[: args.max_images]

    print(
        f"Found {len(real_paths)} real images and {len(synthetic_paths)} synthetic images"
    )

    # It's recommended to use the same number of real and synthetic images
    num_images = min(len(real_paths), len(synthetic_paths))
    real_paths = real_paths[:num_images]
    synthetic_paths = synthetic_paths[:num_images]

    print(f"Using {num_images} images from each set")

    # Initialize InceptionV3 model
    inception = InceptionV3Features(device=args.device)

    # Extract features from real images
    print("Extracting features from real images...")
    real_features = []
    for path in tqdm(real_paths):
        features = inception.get_features(path)
        if features is not None:
            real_features.append(features)

    # Extract features from synthetic images
    print("Extracting features from synthetic images...")
    synthetic_features = []
    for path in tqdm(synthetic_paths):
        features = inception.get_features(path)
        if features is not None:
            synthetic_features.append(features)

    # Convert to numpy arrays
    real_features = np.array(real_features)
    synthetic_features = np.array(synthetic_features)

    print(
        f"Calculating statistics for {real_features.shape[0]} real and {synthetic_features.shape[0]} synthetic features"
    )

    # Calculate mean and covariance statistics
    mu_real, sigma_real = calculate_statistics(real_features)
    mu_synthetic, sigma_synthetic = calculate_statistics(synthetic_features)

    # Calculate FID score
    fid_value = calculate_fid(mu_real, sigma_real, mu_synthetic, sigma_synthetic)
    print(f"FID Score: {fid_value:.4f}")

    return fid_value


if __name__ == "__main__":
    main()
