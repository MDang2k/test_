import os
import nibabel as nib
import tensorflow as tf
from scipy import ndimage
import numpy as np


def read_nifti_file(filepath):
    """Read and load volume"""
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """Normalize the volume"""
    min = -1000
    max = 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""

    desired_depth = 64
    desired_width = 128
    desired_height = 128

    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


def process_scan(path):
    """Read and resize volume"""
    volume = read_nifti_file(path)
    volume = normalize(volume)
    volume = resize_volume(volume)
    return volume


CN_scan_paths = [
    os.path.join(os.getcwd(), "C:/Users/Potato/Documents/ADNI/Train/CN", x)
    for x in os.listdir("C:/Users/Potato/Documents/ADNI/Train/CN")
]

AD_scan_paths = [
    os.path.join(os.getcwd(), "C:/Users/Potato/Documents/ADNI/Train/AD", x)
    for x in os.listdir("C:/Users/Potato/Documents/ADNI/Train/AD")
]

print("CN scans: " + str(len(CN_scan_paths)))
print("AD scans: " + str(len(AD_scan_paths)))
