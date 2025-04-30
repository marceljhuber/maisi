import os
import argparse
import shutil
from pathlib import Path
import re
from PIL import Image
import numpy as np


def process_image_pairs(source_dirs, target_dir, resize_width, resize_height):
    """
    Process OCT image pairs from source directories and their subdirectories,
    resize them, and copy to target directory.

    Args:
        source_dirs: List of source directory paths
        target_dir: Target directory path (str)
        resize_width: Width to resize images to
        resize_height: Height to resize images to
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Counter for naming output files
    counter = 0

    for source_dir in source_dirs:
        # Walk through directory and all subdirectories
        for root, dirs, files in os.walk(source_dir):
            # Find all OCT images in the current directory
            oct_files = sorted([f for f in files if f.endswith("_oct.png")])

            # Extract device type from the path (cirrus, spectralis, or topcon)
            device_type = None
            path_str = str(root).lower()
            if "cirrus" in path_str:
                device_type = "cirrus"
            elif "spectralis" in path_str:
                device_type = "spectralis"
            elif "topcon" in path_str:
                device_type = "topcon"
            else:
                device_type = "unknown"

            for oct_filename in oct_files:
                # Get the base name (e.g., bscan_000)
                base_name = oct_filename.replace("_oct.png", "")

                # Construct the reference filename
                ref_filename = f"{base_name}_reference.png"

                # Check if the reference file exists
                ref_path = os.path.join(root, ref_filename)
                if not os.path.exists(ref_path):
                    print(
                        f"Warning: Reference file {ref_filename} not found for {oct_filename} in {root}, skipping..."
                    )
                    continue

                # Load images
                oct_path = os.path.join(root, oct_filename)
                oct_img = Image.open(oct_path)
                ref_img = Image.open(ref_path)

                # Resize images using nearest neighbor interpolation
                oct_img_resized = oct_img.resize(
                    (resize_width, resize_height), Image.NEAREST
                )
                ref_img_resized = ref_img.resize(
                    (resize_width, resize_height), Image.NEAREST
                )

                # Create new filenames
                new_oct_filename = f"{device_type}_{counter:05d}_oct.png"
                new_ref_filename = f"{device_type}_{counter:05d}_ref.png"

                # Save resized images to target directory
                oct_img_resized.save(os.path.join(target_dir, new_oct_filename))
                ref_img_resized.save(os.path.join(target_dir, new_ref_filename))

                print(f"Processed: {oct_path} -> {new_oct_filename}")
                print(f"Processed: {ref_path} -> {new_ref_filename}")

                # Increment counter
                counter += 1

    return counter


def main():
    parser = argparse.ArgumentParser(description="Process OCT image pairs")
    parser.add_argument(
        "--width", type=int, default=256, help="Width to resize images to"
    )
    parser.add_argument(
        "--height", type=int, default=256, help="Height to resize images to"
    )

    args = parser.parse_args()

    target_dir = "/home/mhuber/Thesis/data/RETOUCH_TRAIN_SPECTRALIS"

    source_dirs = [
        # "/home/mhuber/Thesis/data/RETOUCH_PNG/RETOUCH-TrainingSet-Cirrus",
        "/home/mhuber/Thesis/data/RETOUCH_PNG/RETOUCH-TrainingSet-Spectralis",
        # "/home/mhuber/Thesis/data/RETOUCH_PNG/RETOUCH-TrainingSet-Topcon",
    ]

    process_image_pairs(source_dirs, target_dir, args.width, args.height)


if __name__ == "__main__":
    main()
