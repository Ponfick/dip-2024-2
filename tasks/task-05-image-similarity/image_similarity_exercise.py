# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np
from PIL import Image

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

    diff = i1 - i2
    mse = np.mean(diff ** 2)

    if mse == 0:
        return float('inf')
    max_pixel_value = 1.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        
    C1 = 1e-4
    C2 = 1e-4
    mu1 = np.mean(i1)
    mu2 = np.mean(i2)
    sigma1_sq = np.var(i1)
    sigma2_sq = np.var(i2)
    sigma12 = np.mean((i1 - mu1) * (i2 - mu2))
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim = numerator / denominator

    
    mean_i1 = np.mean(i1)
    mean_i2 = np.mean(i2)
    numerator = np.sum((i1 - mean_i1) * (i2 - mean_i2))
    denominator = np.sqrt(np.sum((i1 - mean_i1) ** 2) * np.sum((i2 - mean_i2) ** 2))
    if denominator == 0:
        return 0
    npcc = numerator / denominator

    print("Comparison Results:")
    print("MSE: ", mse)
    print("PSNR: ", psnr)
    print("SSIM: ", ssim)
    print("NPCC: ", npcc)
    
    pass


def main():
    # Load images
    image1_path = "img/lena.png"
    image2_path = "img/baboon.png"
    image1 = Image.open(image1_path).convert('L')  # Convert to grayscale
    image2 = Image.open(image2_path).convert('L')
    i1 = np.array(image1) / 255.0  # Normalize to [0, 1]
    i2 = np.array(image2) / 255.0

    compare = compare_images(i1, i2)


if __name__ == "__main__":
    main()