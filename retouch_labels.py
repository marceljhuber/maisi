import os
import numpy as np
from PIL import Image
import argparse


def count_unique_pixels(directory):
    """
    Process all images ending with '_ref.png' in the specified directory.
    Convert each to grayscale and count the unique pixel values across all images.

    Args:
        directory (str): Path to the directory containing images

    Returns:
        list: List of unique pixel values across all processed images
    """
    # Keep track of all unique values found
    all_unique_values = set()
    processed_count = 0

    # Get all files in the directory
    try:
        files = os.listdir(directory)
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return list(all_unique_values)

    # Filter for files that end with '_ref.png'
    ref_images = [f for f in files if f.endswith("_ref.png")]

    if not ref_images:
        print(f"No '_ref.png' files found in directory '{directory}'.")
        return list(all_unique_values)

    # Process each image
    for image_file in ref_images:
        file_path = os.path.join(directory, image_file)

        try:
            # Open and convert to grayscale
            with Image.open(file_path) as img:
                grayscale_img = img.convert("L")

                # Convert to numpy array for easier processing
                img_array = np.array(grayscale_img)

                # Get unique pixel values for this image
                unique_values = set(np.unique(img_array).tolist())

                # Add to our overall set of unique values
                all_unique_values.update(unique_values)
                processed_count += 1

        except Exception as e:
            print(f"Error processing '{image_file}': {str(e)}")

    # Convert to sorted list for nicer output
    result = sorted(list(all_unique_values))

    print(f"Processed {processed_count} images ending with '_ref.png'")
    print(f"Unique pixel values across all images: {result}")
    print(f"Total number of unique values: {len(result)}")

    return result


def main():

    directory = "/home/mhuber/Thesis/data/RETOUCH_TRAIN_SPECTRALIS"

    # Process the images
    count_unique_pixels(directory)


if __name__ == "__main__":
    main()
