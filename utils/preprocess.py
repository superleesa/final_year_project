from typing import List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import torch
import os
from pathlib import Path
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, sand_dust_images: List[Image.Image], image_names: List[str], ground_truth_images: Optional[List[Image.Image]] = None):
        self.sand_dust_images = sand_dust_images
        self.ground_truth_images = ground_truth_images
        self.image_names = image_names
        self.W_THRESHOLD = 440
        self.H_THRESHOLD = 330
        self.is_paired = ground_truth_images is not None

    def __len__(self):
        return len(self.sand_dust_images)

    def preprocess(self, image: Image) -> torch.Tensor:
        image = resize_image(image, self.W_THRESHOLD, self.H_THRESHOLD)
        image = np.asarray(image) / 255.0
        image = np.transpose(image, axes=[2, 0, 1]).astype('float32')  # to C, H, W
        image = torch.from_numpy(image).float()
        return image

    def __getitem__(self, idx) -> Union[Tuple[torch.Tensor, torch.Tensor, str], Tuple[torch.Tensor, str]]:
        sand_dust_image = self.preprocess(self.sand_dust_images[idx])
        image_name = self.image_names[idx]

        if self.is_paired:
            ground_truth_image = self.preprocess(self.ground_truth_images[idx])
            return sand_dust_image, ground_truth_image, image_name
        else:
            return sand_dust_image, image_name

def crop_image(image, w_threshold, h_threshold):
    assert image.width >= w_threshold and image.height >= h_threshold, "to crop, image size must be bigger than or equal to the threshold values"

    # choose top and right randomly -> bottom and left automatically determined
    top = random.randint(0, image.height - h_threshold)  # inclusive
    left = random.randint(0, image.width - w_threshold)

    bottom = top + h_threshold
    right = left + w_threshold

    return image.crop((left, top, right, bottom))


def is_image_smaller_than_threshold(image, w_threshold, h_threshold) -> bool:
    return image.width < w_threshold or image.height < h_threshold


def stretch_image(image, w_threshold, h_threshold):
    aspect_ratio = h_threshold / w_threshold

    if h_threshold - image.height < 0:
        resize_based_on_width = True
    elif w_threshold - image.width < 0:
        resize_based_on_width = False
    else:
        # resize based on whichever the difference is smaller
        resize_based_on_width = np.argmin([w_threshold - image.width, h_threshold - image.height])

    if resize_based_on_width:
        new_w = w_threshold
        new_h = int(new_w * aspect_ratio)
    else:
        new_h = h_threshold
        new_w = int(new_h / aspect_ratio)

    return image.resize((new_w, new_h))


def resize_images(images, w_threshold, h_threshold):
    resized_images = []
    for image in tqdm(images):
        if is_image_smaller_than_threshold(image, w_threshold, h_threshold):
            image = stretch_image(image, w_threshold, h_threshold)

        new_image = crop_image(image, w_threshold, h_threshold)
        resized_images.append(new_image)

    return resized_images

def resize_image(image: Image, w_threshold: int, h_threshold: int) -> Image:
    if is_image_smaller_than_threshold(image, w_threshold, h_threshold):
        image = stretch_image(image, w_threshold, h_threshold)

    return crop_image(image, w_threshold, h_threshold)

def load_images_in_a_directory(directory_path):
    images = []
    image_names = []
    print(os.listdir(directory_path))
    for filename in os.listdir(directory_path):
        if filename.endswith("png") or filename.endswith("jpg"):
            image_path = os.path.join(directory_path, filename)
            image = Image.open(image_path)
            images.append(image)
            image_names.append(filename)

    return images, image_names

def create_dataset(dataset_dir: str, save_dir: str, isPaired: bool):
    '''"
    dataset_dir: directory of the dataset (e.g. "Data/Synthetic_images/")
    save_dir: directory to save the output (format: "Data/output/base_toenet_on_sie")
    isUnpaired: whether it is paired Dataset or not.
    '''
    save_images = "all"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if isPaired:
        gt_path = os.path.join(dataset_dir, "Ground_truth")
        gt_images, gt_image_names = load_images_in_a_directory(gt_path)
        gt_images = [gt_image for gt_image, _ in sorted(zip(gt_images, gt_image_names), key=lambda x: x[1])]

        noisy_path = os.path.join(dataset_dir, "Sand_dust_images")
        noisy_images, noisy_image_names = load_images_in_a_directory(noisy_path)
        noisy_images = [noisy_image for noisy_image, _ in sorted(zip(noisy_images, noisy_image_names), key=lambda x: x[1])]
        denoised_image_file_names = [f"{index}_denoised" for index in range(len(noisy_images))]
        dataset = ImageDataset(noisy_images, gt_images, denoised_image_file_names)
    else:
        noisy_path = os.path.join(dataset_dir, "Sand_dust_images")
        noisy_images, noisy_image_names = load_images_in_a_directory(noisy_path)
        noisy_images = [noisy_image for noisy_image, _ in sorted(zip(noisy_images, noisy_image_names), key=lambda x: x[1])]
        denoised_image_file_names = [f"{index}_denoised" for index in range(len(noisy_images))]
        dataset = ImageDataset(noisy_images, denoised_image_file_names)

    return dataset
