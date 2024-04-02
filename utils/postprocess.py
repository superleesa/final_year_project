from typing import List, Literal

from src.toenet.test import load_checkpoint
import numpy as np
import utils.metrics as metrics
from torch.utils.data import DataLoader
from utils.preprocess import SIEDataset
import torch
import random
import os
import cv2
from torchvision.transforms import v2

CHANNEL_SIZE = 255

def validate_transform(images):
    images = images.cpu().numpy()
    images = np.transpose(images, axes=[0, 2, 3, 1]).astype('float32')
    images = np.clip(images * CHANNEL_SIZE, 0.0, CHANNEL_SIZE) # normalize back to 0~255
    return images