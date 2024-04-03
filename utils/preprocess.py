from typing import List, Optional, Tuple, Union
import numpy as np
from torch.utils.data import Dataset
import torch
import os
from torchvision.transforms import v2
import random
import cv2


# dataset directory constants
NOISY_IMAGE_DIR_NAME = "noisy"
GROUND_TRUTH_IMAGE_DIR_NAME = "ground_truth"
CLEAR_IMAGE_DIR_NAME = "clear"


# image size constants
W_THRESHOLD = 440
H_THRESHOLD = 330

train_transform = v2.Compose([
    v2.Lambda(lambda image: np.transpose(image, axes=[2, 0, 1]).astype('float32')),
    v2.ToTensor(),
    v2.RandomResizedCrop((W_THRESHOLD, H_THRESHOLD)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(degrees=90),
])

eval_transform = v2.Compose([
    v2.Lambda(lambda image: np.transpose(image, axes=[2, 0, 1]).astype('float32')),
    v2.ToTensor(),
    v2.Resize((W_THRESHOLD, H_THRESHOLD)),
])


class PairedDataset(Dataset):
    def __init__(self, sand_dust_images: List[np.ndarray],
                 ground_truth_images: List[np.ndarray],
                 output_image_names: List[str],
                 transformer: v2.Transform.Compose) -> None:
        """
        output_image_names: should only be the name of the image without the extension and the path
        """
        self.sand_dust_images = sand_dust_images
        self.ground_truth_images = ground_truth_images
        self.output_image_names = output_image_names

        self.transformer = transformer

    def __len__(self) -> int:
        return len(self.sand_dust_images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sand_dust_image = self.transformer(self.sand_dust_images[idx])
        ground_truth_image = self.transformer(self.ground_truth_images[idx])
        return sand_dust_image, ground_truth_image, self.output_image_names[idx]


class UnpairedDataset:
    """This class must be instantiated for each epoch to change pairs."""
    def __init__(self, sand_dust_images: List[np.ndarray],
                 clear_images: List[np.array],
                 output_image_names: List[str],
                 transformer: v2.Transform.Compose) -> None:
        self.sand_dust_images = sand_dust_images
        self.clear_images = clear_images
        self.output_image_names = output_image_names

        self.transformer = transformer

        # generate random pairs
        random.shuffle(self.sand_dust_images)
        random.shuffle(self.clear_images)

    def __len__(self) -> int:
        return len(self.sand_dust_images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sand_dust_image = self.transformer(self.sand_dust_images[idx].copy())
        clear_image = self.transformer(self.sand_dust_images[idx].copy())
        return sand_dust_image, clear_image, self.output_image_names[idx]


def load_images_in_a_directory(directory_path: str) -> tuple[List[np.ndarray], List[str]]:
    images = []
    image_names = []
    print(os.listdir(directory_path))
    for filename in os.listdir(directory_path):
        if filename.endswith("png") or filename.endswith("jpg"):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)

            images.append(image)
            image_names.append(filename)

    return images, image_names


def sort_image_by_filenames(images: List[np.ndarray], image_names: List[str]) -> List[np.ndarray]:
    return [image for image, _ in sorted(zip(images, image_names), key=lambda x: x[1])]

def create_paired_dataset(dataset_dir: str, is_train=False, num_datasets: int = 1) -> List[PairedDataset]:
    """
    dataset_dir: directory of the dataset (e.g. "Data/Synthetic_images/")
    """
    noisy_path = os.path.join(dataset_dir, NOISY_IMAGE_DIR_NAME)
    noisy_images, noisy_image_names = load_images_in_a_directory(noisy_path)
    noisy_images = sort_image_by_filenames(noisy_images, noisy_image_names)

    gt_path = os.path.join(dataset_dir, GROUND_TRUTH_IMAGE_DIR_NAME)
    gt_images, gt_image_names = load_images_in_a_directory(gt_path)
    gt_images = [gt_image for gt_image, _ in sorted(zip(gt_images, gt_image_names), key=lambda x: x[1])]

    denoised_image_file_names = [f"{index}_denoised" for index in range(len(noisy_images))]

    return [PairedDataset(noisy_images, gt_images, denoised_image_file_names, transformer) for _ in range(num_datasets)]

def create_unpaired_datasets(dataset_dir: str, num_datasets: int = 1, is_train=False) -> List[UnpairedDataset]:
    """
    dataset_dir: directory of the dataset (e.g. "Data/Synthetic_images/")
    """
    noisy_path = os.path.join(dataset_dir, CLEAR_IMAGE_DIR_NAME)
    noisy_images, noisy_image_names = load_images_in_a_directory(noisy_path)
    noisy_images = sort_image_by_filenames(noisy_images, noisy_image_names)

    clear_path = os.path.join(dataset_dir, CLEAR_IMAGE_DIR_NAME)
    clear_images, clear_image_names = load_images_in_a_directory(clear_path)
    clear_images = sort_image_by_filenames(clear_images, clear_image_names)

    denoised_image_file_names = [f"{index}_denoised" for index in range(len(noisy_images))]

    transformer = train_transform if is_train else eval_transform
    return [UnpairedDataset(noisy_images, clear_images, denoised_image_file_names, transformer) for _ in range(num_datasets)]
