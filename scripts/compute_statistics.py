import numpy as np
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def compute_statistics(images_dir):
    Path("./configs").mkdir(parents=True, exist_ok=True)
    print(f"Scanning directory and subdirectories: {images_dir}")

    # Find all image files
    image_files = []
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
        image_files.extend(Path(images_dir).rglob(f"*{ext}"))
        image_files.extend(Path(images_dir).rglob(f"*{ext.upper()}"))

    print(f"Image files found: {len(image_files)}")
    if not image_files:
        raise ValueError("No valid image files found in directory")

    # Initialize Welford's algorithm variables
    mean = 0.0
    M2 = 0.0
    count = 0
    all_medians = []
    all_values = []

    # Process images in batches
    batch_size = 300
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i : i + batch_size]

        for img_path in batch_files:
            try:
                img = np.array(Image.open(img_path).convert("L")).astype(np.float32)

                # Update Welford's algorithm
                for pixel in img.flat:
                    count += 1
                    delta = pixel - mean
                    mean += delta / count
                    delta2 = pixel - mean
                    M2 += delta * delta2

                # Store median
                all_medians.append(np.median(img))

                # Store random sample of values for percentiles
                sample_indices = np.random.choice(
                    img.size, min(1000, img.size), replace=False
                )
                all_values.extend(img.flat[sample_indices])

            except Exception as e:
                print(f"Error loading {img_path}: {str(e)}")

    # Compute final statistics
    variance = M2 / (count - 1) if count > 1 else 0
    std_val = np.sqrt(variance)

    stats = {
        "oct": {
            "min_median": float(np.min(all_medians)),
            "max_median": float(np.max(all_medians)),
            "percentile_0_5": float(np.percentile(all_values, 0.5)),
            "percentile_99_5": float(np.percentile(all_values, 99.5)),
            "sigma_6_low": float(mean - 6 * std_val),
            "sigma_6_high": float(mean + 6 * std_val),
            "sigma_12_low": float(mean - 12 * std_val),
            "sigma_12_high": float(mean + 12 * std_val),
        }
    }

    # Print some debug info
    print(f"\nDebug information:")
    print(f"Total pixels processed: {count}")
    print(f"Mean: {mean}")
    print(f"Standard deviation: {std_val}")

    # Save statistics
    with open("./configs/image_median_statistics.json", "w") as f:
        json.dump(stats, f, indent=4)

    print("\nStatistics saved to ./configs/image_median_statistics.json")
    print("Statistics:", stats)

    return stats


if __name__ == "__main__":
    stats = compute_statistics("/home/mhuber/Thesis/data/KermanyV3_resized/train")
