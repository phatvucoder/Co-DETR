import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm

def compute_mean_std(image_paths, batch_size=100):
    """Compute mean and std by reading images in batches to save memory."""
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    num_pixels = 0

    with tqdm(total=len(image_paths), desc="Processing Images", unit="img") as pbar:
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]  # Only load one batch of images
            for img_path in batch_paths:
                img = cv2.imread(img_path)  # Read the image
                if img is None:
                    print(f"Warning: Skipping {img_path}")
                    pbar.update(1)
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = img.astype(np.float64) / 255.0  # Normalize to [0,1]
                
                pixel_sum += img.sum(axis=(0, 1))  # Sum of each color channel
                pixel_sq_sum += (img ** 2).sum(axis=(0, 1))  # Sum of squares of each color channel
                num_pixels += img.shape[0] * img.shape[1]  # Total number of pixels
                
                pbar.update(1)

    mean = pixel_sum / num_pixels  # Compute mean for each channel
    std = np.sqrt((pixel_sq_sum / num_pixels) - (mean ** 2))  # Compute std for each channel

    return mean * 255, std * 255  # Convert back to scale [0, 255]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mean and std efficiently for large datasets.")
    parser.add_argument("folders", nargs="+", help="Paths to dataset folders (e.g., train/ val/ test/)")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for processing images")
    args = parser.parse_args()

    # Gather image paths from all input folders
    image_paths = []
    for folder in args.folders:
        if not os.path.exists(folder):
            print(f"Warning: Folder '{folder}' not found, skipping...")
            continue
        image_paths.extend([os.path.join(folder, img) for img in os.listdir(folder) if img.endswith((".jpg", ".png", ".jpeg"))])

    if not image_paths:
        raise ValueError("No images found in the provided folders.")

    # Calculate mean and std
    mean, std = compute_mean_std(image_paths, batch_size=args.batch_size)

    # Result
    print("\n🔹 Mean:", mean)
    print("🔹 Std:", std)
    print("\n🔥 Config format:")
    print(f"img_norm_cfg = dict(\n\tmean={list(mean)},\n\tstd={list(std)},\n\tto_rgb=True\n)")