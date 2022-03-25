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
        # Set the desired depth
        desired_depth = 64
        desired_width = 128
        desired_height = 128
        # Get current depth
        current_depth = img.shape[-1]
        current_width = img.shape[0]
        current_height = img.shape[1]
        # Compute depth factor
        depth = current_depth / desired_depth
        width = current_width / desired_width
        height = current_height / desired_height
        depth_factor = 1 / depth
        width_factor = 1 / width
        height_factor = 1 / height
        # Rotate
        img = ndimage.rotate(img, 90, reshape=False)
        # Resize across z-axis
        img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
        return img


def process_scan(path):
        """Read and resize volume"""
        volume = read_nifti_file(path)
        volume = normalize(volume)
        volume = resize_volume(volume)
        return volume
    
def read_data(dir_path):

        CN_path = os.path.join(dir_path, "CN")
        AD_path = os.path.join(dir_path, "AD")
    
        CN_scan_paths = [
            os.path.join(CN_path, x)
            for x in os.listdir(CN_path)
        ]

        AD_scan_paths = [
            os.path.join(AD_path, x)
            for x in os.listdir(AD_path)
        ]


        #scan for all images in data folder
        AD_scans = np.array([process_scan(path) for path in AD_scan_paths])
        CN_scans = np.array([process_scan(path) for path in CN_scan_paths])

        print("CN scans: " + str(len(CN_scan_paths)))
        print("AD scans: " + str(len(AD_scan_paths)))

        #assign labels for 2 types of data
        AD_labels = np.array([1 for _ in range(len(AD_scans))])
        CN_labels = np.array([0 for _ in range(len(CN_scans))])

        #divide into train an validation set
        x_train = np.concatenate((AD_scans[:2], CN_scans[:1]), axis=0)
        y_train = np.concatenate((AD_labels[:2], CN_labels[:1]), axis=0)
        x_val = np.concatenate((AD_scans[2:], CN_scans[1:]), axis=0)
        y_val = np.concatenate((AD_labels[2:], CN_labels[1:]), axis=0)
        print(
            "Number of samples in train and validation are %d and %d."
            % (x_train.shape[0], x_val.shape[0])
        )

        return x_train, y_train, x_val, y_val
        
    
