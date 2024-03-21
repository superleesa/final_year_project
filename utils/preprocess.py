from typing import List

from PIL.Image import Image
import numpy as np
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import torch


class SIEDataset(Dataset):
    # load all images at once
    # we use float32 for the image data type
    def __init__(self, sand_dust_images: List[Image], ground_truth_images: List[Image], image_names: List[str], w_threshold: int, h_threshold: int):
        assert len(sand_dust_images) == len(ground_truth_images), "the number of sand dust images and ground truth images must be the same"
        self.sand_dust_images = sand_dust_images
        self.ground_truth_images = ground_truth_images
        self.image_names = image_names
        self.w_threshold = w_threshold
        self.h_threshold = h_threshold

    def __len__(self):
        return len(self.sand_dust_images)

    def preprocess(self, image: Image) -> torch.Tensor:
        image = resize_image(image, self.w_threshold, self.h_threshold)
        image = np.asarray(image) / 255.0
        image = np.transpose(image, axes=[2, 0, 1]).astype('float32')  # to C, H, W
        image = torch.from_numpy(image).float()
        return image

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, str]:
        sand_dust_image = self.preprocess(self.sand_dust_images[idx])
        ground_truth_image = self.preprocess(self.ground_truth_images[idx])
        image_name = self.image_names[idx]

        return sand_dust_image, ground_truth_image, image_name

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