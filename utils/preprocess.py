from typing import List, Optional, Tuple, Union
import numpy as np
from utils.transforms import (
    train_paired_transform,
    train_unpaired_transform,
    eval_transform,
    CoupledCompose,
)
from torch.utils.data import Dataset
import torch
import os
from torchvision.transforms import v2
import random
import cv2
import random

# dataset directory constants
NOISY_IMAGE_DIR_NAME = "noisy"
GROUND_TRUTH_IMAGE_DIR_NAME = "ground_truth"
CLEAR_IMAGE_DIR_NAME = "clear"


class PairedDataset(Dataset):
    def __init__(
        self,
        sand_dust_images: List[np.ndarray],
        ground_truth_images: List[np.ndarray],
        transformer: CoupledCompose | v2.Compose,
    ) -> None:
        self.sand_dust_images = sand_dust_images
        self.ground_truth_images = ground_truth_images

        self.transformer = transformer

    def __len__(self) -> int:
        return len(self.sand_dust_images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.transformer, CoupledCompose):
            sand_dust_image, ground_truth_image = self.transformer(
                self.sand_dust_images[idx].copy(), self.ground_truth_images[idx].copy()
            )
        elif isinstance(self.transformer, v2.Compose):
            # copy to ensure arrays are different across Datasets
            sand_dust_image = self.transformer(self.sand_dust_images[idx].copy())
            ground_truth_image = self.transformer(self.ground_truth_images[idx].copy())
        return sand_dust_image, ground_truth_image


class EvaluationDataset(Dataset):
    def __init__(
        self,
        sand_dust_images: List[np.ndarray],
        ground_truth_images: List[np.ndarray],
        output_image_names: List[str],
        transformer: v2.Compose,
    ) -> None:
        """
        output_image_names: should only be the name of the image without the extension and the path
        """
        self.sand_dust_images = sand_dust_images
        self.ground_truth_images = ground_truth_images
        self.output_image_names = output_image_names

        self.transformer = transformer

    def __len__(self) -> int:
        return len(self.sand_dust_images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
        sand_dust_image = self.transformer(self.sand_dust_images[idx].copy())
        ground_truth_image = self.transformer(self.ground_truth_images[idx].copy())
        return sand_dust_image, ground_truth_image, self.output_image_names[idx]


class UnpairedDataset:
    """This class must be instantiated for each epoch to change pairs."""

    def __init__(
        self,
        sand_dust_images: List[np.ndarray],
        clear_images: List[np.ndarray],
        transformer: v2.Compose,
    ) -> None:
        self.sand_dust_images = sand_dust_images
        self.clear_images = clear_images

        self.transformer = transformer
        # generate random pairs
        random.shuffle(self.sand_dust_images)
        random.shuffle(self.clear_images)

    def __len__(self) -> int:
        # length of sand-dust-images and clear images do not need to be the same
        # -> ensure minimum of the two is selected to avoid index range over error
        return min(len(self.sand_dust_images), len(self.clear_images))

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        sand_dust_image = self.transformer(self.sand_dust_images[idx].copy())
        clear_image = self.transformer(self.clear_images[idx].copy())
        return sand_dust_image, clear_image


def load_dataset(
    directory_path: str,
) -> tuple[List[np.ndarray], List[str]]:
    """
    Finds all pngs/jpgs in the specified directory and load them using cv2.
    """
    # FIXME: in future versions, we should not return image names as it's redundant
    images = []
    image_names = []
    for filename in os.listdir(directory_path):
        if filename.endswith("png") or filename.endswith("jpg"):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)

            images.append(image)
            image_names.append(filename)

    return images, image_names


def sort_image_by_filenames(
    images: List[np.ndarray], image_names: List[str]
) -> List[np.ndarray]:
    return [image for image, _ in sorted(zip(images, image_names), key=lambda x: x[1])]


def create_paired_datasets(
    dataset_dir: str, num_datasets: int = 1
) -> List[PairedDataset]:
    """
    dataset_dir: directory of the dataset (e.g. "Data/Synthetic_images/")
    """
    noisy_path = os.path.join(dataset_dir, NOISY_IMAGE_DIR_NAME)
    noisy_images, noisy_image_names = load_dataset(noisy_path)
    noisy_images = sort_image_by_filenames(noisy_images, noisy_image_names)

    gt_path = os.path.join(dataset_dir, GROUND_TRUTH_IMAGE_DIR_NAME)
    gt_images, gt_image_names = load_dataset(gt_path)
    gt_images = [
        gt_image
        for gt_image, _ in sorted(zip(gt_images, gt_image_names), key=lambda x: x[1])
    ]

    return [
        PairedDataset(noisy_images, gt_images, train_paired_transform)
        for _ in range(num_datasets)
    ]


def create_unpaired_datasets(
    dataset_dir: str, num_datasets: int = 1
) -> List[UnpairedDataset]:
    """
    dataset_dir: directory of the dataset (e.g. "Data/Synthetic_images/")
    Do not use thsi funciton when evaluating
    """
    noisy_path = os.path.join(dataset_dir, NOISY_IMAGE_DIR_NAME)
    noisy_images, noisy_image_names = load_dataset(noisy_path)
    noisy_images = sort_image_by_filenames(noisy_images, noisy_image_names)

    clear_path = os.path.join(dataset_dir, CLEAR_IMAGE_DIR_NAME)
    clear_images, clear_image_names = load_dataset(clear_path)
    clear_images = sort_image_by_filenames(clear_images, clear_image_names)

    return [
        UnpairedDataset(noisy_images, clear_images, train_unpaired_transform)
        for _ in range(num_datasets)
    ]


def create_evaluation_dataset(dataset_dir: str) -> EvaluationDataset:
    """
    dataset_dir: directory of the dataset (e.g. "Data/Synthetic_images/")
    """
    noisy_path = os.path.join(dataset_dir, NOISY_IMAGE_DIR_NAME)
    noisy_images, noisy_image_names = load_dataset(noisy_path)
    noisy_images = sort_image_by_filenames(noisy_images, noisy_image_names)

    gt_path = os.path.join(dataset_dir, GROUND_TRUTH_IMAGE_DIR_NAME)
    gt_images, gt_image_names = load_dataset(gt_path)
    gt_images = [
        gt_image
        for gt_image, _ in sorted(zip(gt_images, gt_image_names), key=lambda x: x[1])
    ]

    denoised_image_file_names = [
        f"{index}_denoised" for index in range(len(noisy_images))
    ]

    return EvaluationDataset(
        noisy_images, gt_images, denoised_image_file_names, eval_transform
    )


def split_indices_randomely(
    num_images: int, train_ratio: float
) -> tuple[list[int], list[int]]:
    """
    Prereq: ensure the sepcified directory only contains relevant images
    """
    indices = list(range(num_images))
    random.shuffle(indices)
    thres = int(num_images * train_ratio)
    return indices[:thres], indices[thres:]


def train_val_split(
    images: List[np.ndarray],
    train_ratio: float | None = None,
) -> tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Split specified dataset into train and validation data, using specified indices.
    """
    num_images = len(images)
    train_indices, validation_indices = split_indices_randomely(num_images, train_ratio)

    loaded_train_images = [images[i] for i in train_indices]
    loaded_validation_images = [images[i] for i in validation_indices]
    return (
        loaded_train_images,
        loaded_validation_images,
    )


def train_val_split_paired(
    x_images: List[np.ndarray],
    y_images: List[np.ndarray],
    train_ratio: float | None = None,
) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Split specified dataset into train and validation data, using specified indices.
    """
    assert len(x_images) == len(
        y_images
    ), "in paired training, the number of sand dust images and their ground truth images must be the same"
    num_images = len(x_images)
    train_indices, validation_indices = split_indices_randomely(num_images, train_ratio)

    x_train_images = [x_images[i] for i in train_indices]
    y_train_images = [y_images[i] for i in train_indices]
    x_validation_images = [x_images[i] for i in validation_indices]
    y_validation_images = [y_images[i] for i in validation_indices]
    return x_train_images, y_train_images, x_validation_images, y_validation_images


def create_train_and_validation_paired_datasets(
    dataset_dir: str, num_datasets: int = 1, train_ratio: float = 0.8
) -> tuple[list[PairedDataset], list[PairedDataset]]:
    """
    Create both train and valiation dataset at the same time, ensuring there is no data leakage.
    This function is for paired datasets.
    """

    noisy_dataset_path = os.path.join(dataset_dir, NOISY_IMAGE_DIR_NAME)
    gt_path = os.path.join(dataset_dir, GROUND_TRUTH_IMAGE_DIR_NAME)
    noisy_images, _ = load_dataset(noisy_dataset_path)
    gt_images, _ = load_dataset(gt_path)

    (
        noisy_train_images,
        gt_train_images,
        noisy_validation_images,
        gt_validation_images,
    ) = train_val_split_paired(noisy_images, gt_images, train_ratio=train_ratio)

    # handling ground truth data
    train_dataset = [
        PairedDataset(noisy_train_images, gt_train_images, train_paired_transform)
        for _ in range(num_datasets)
    ]
    validation_dataset = [
        PairedDataset(
            noisy_validation_images, gt_validation_images, train_paired_transform
        )
        for _ in range(num_datasets)
    ]

    return train_dataset, validation_dataset


def create_train_and_validation_unpaired_datasets(
    dataset_dir: str, num_datasets: int = 1, train_ratio: float = 0.8
) -> tuple[list[UnpairedDataset], list[UnpairedDataset]]:
    """
    Create both train and valiation dataset at the same time, ensuring there is no data leakage.
    This function is for unpaired datasets.
    """
    # handling noisy dataset
    noisy_path = os.path.join(dataset_dir, NOISY_IMAGE_DIR_NAME)
    noisy_images, _ = load_dataset(noisy_path)
    (
        noisy_train_images,
        noisy_validation_images,
    ) = train_val_split(noisy_images, train_ratio=train_ratio)

    # handling clear dataset
    clear_path = os.path.join(dataset_dir, CLEAR_IMAGE_DIR_NAME)
    clear_images, _ = load_dataset(clear_path)
    (
        clear_train_images,
        clear_validation_images,
    ) = train_val_split(clear_images, train_ratio=train_ratio)

    # creating training and validation datasets
    train_dataset = [
        UnpairedDataset(noisy_train_images, clear_train_images, train_unpaired_transform)
        for _ in range(num_datasets)
    ]
    validation_dataset = [
        UnpairedDataset(
            noisy_validation_images, clear_validation_images, train_unpaired_transform
        )
        for _ in range(num_datasets)
    ]

    return train_dataset, validation_dataset
