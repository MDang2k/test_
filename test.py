import os
import numpy as np
from src.dataset import MRIdataset
from src.model import get_model


IMAGE_SIZE = 224
DATA_DIR = './data'
DATA_DIR_TEST_IMAGES = os.path.join(DATA_DIR, 'test')
DATA_DIR_TEST_RESULTS = os.path.join(DATA_DIR, 'sample_submission.csv')
DATA_DIR_TRAIN_LABEL = os.path.join(DATA_DIR, 'train_sample.csv')

TRAINED_MODEL_PATH="models/epoch_score"    #place your saved checkpoint file here.

CN_scan_paths = [
    os.path.join(os.getcwd(), "./data/CN", x)
    for x in os.listdir("./data/CN")
]

AD_scan_paths = [
    os.path.join(os.getcwd(), "./data/AD", x)
    for x in os.listdir("./data/AD")
]