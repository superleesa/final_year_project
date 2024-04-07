from typing import List, Literal
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.utils import load_checkpoint
import numpy as np
import utils.metrics as metrics
from torch.utils.data import DataLoader
import torch
import random
import os
import cv2

SaveImageType = Literal["all", "sample", "no"]
MAX_PIXEL_VALUE = 255


def save_image(image: np.ndarray, image_name: str, save_dir: str) -> None:
    save_path = os.path.join(save_dir, image_name + ".jpg")
    cv2.imwrite(save_path, image)


def evaluate(dataloader: DataLoader, save_dir: "str", checkpoint_path: str, save_images: SaveImageType = "sample") -> \
tuple[np.ndarray, np.ndarray]:
    """
    Ensure all images have the same size
    """
    assert save_images in ["all", "sample", "no"], "save_images parameter must be one of all, sample, or no"

    model = load_checkpoint(checkpoint_path, True)  # True for GPU
    model.eval()

    psnr_output_batches: List[np.ndarray] = []
    ssim__output_batches: List[np.ndarray] = []

    for sand_dust_images, ground_truth_images, image_names in dataloader:
        with torch.no_grad():
            sand_dust_images = sand_dust_images.cuda()
            denoised_images = model(sand_dust_images)
            ground_truth_images = torch.clip(ground_truth_images.cuda() * MAX_PIXEL_VALUE, min=0.0,
                                             max=MAX_PIXEL_VALUE)  # normalize back to 0~255
            denoised_images = torch.clip(denoised_images * MAX_PIXEL_VALUE, min=0.0,
                                         max=MAX_PIXEL_VALUE)  # normalize back to 0~255

            psnr_per_sample = metrics.psnr_per_sample(denoised_images, ground_truth_images).cpu().numpy()
            ssim_per_sample = metrics.ssim_per_sample(denoised_images, ground_truth_images).cpu().numpy()
            psnr_output_batches.append(psnr_per_sample)
            ssim__output_batches.append(ssim_per_sample)

            if save_images == "all":
                denoised_images = np.transpose(denoised_images.cpu().numpy(), axes=[0, 2, 3, 1]).astype('float32')
                for denoised_image, image_name in zip(denoised_images, image_names):
                    save_image(denoised_image, image_name, save_dir)

            elif save_images == "sample":
                image_idx = random.randint(0, len(denoised_images) - 1)
                denoised_image, image_name = denoised_images[image_idx], image_names[image_idx]
                denoised_image = np.transpose(denoised_image.cpu().numpy(), axes=[1, 2, 0]).astype('float32')
                save_image(denoised_image, image_name, save_dir)

    return np.concatenate(psnr_output_batches), np.concatenate(ssim__output_batches)