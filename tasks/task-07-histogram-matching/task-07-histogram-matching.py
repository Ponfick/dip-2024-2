# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
from skimage.exposure import match_histograms

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    
    reference = source_img
    image = reference_img

    matched = match_histograms(image, reference, channel_axis=-1)
    matched_img = np.clip(matched, 0, 255).astype(np.uint8)
    return matched_img
    pass

def main():

    source_img = cv.imread("tasks\\task-07-histogram-matching\\source.jpg", cv.IMREAD_COLOR)
    reference_img = cv.imread('C:\\Users\\hans_\\OneDrive\\Documents\\DIP\\dip-2024-2\\tasks\\task-07-histogram-matching\\reference.jpg', cv.IMREAD_COLOR)
    
    results = match_histograms_rgb(source_img, reference_img)
    
if __name__ == "__main__":
    main()