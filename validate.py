from typing import List, Literal

from src.toenet.test import load_checkpoint
import numpy as np
import utils.metrics as metrics
from torch.utils.data import DataLoader
from utils.preprocess import SIEDataset
import torch
import random
import os
from PIL import Image


SaveImageType = Literal["all", "sample", "no"]


def save_image(image: np.ndarray, image_name: str, save_dir: str) -> None:
    save_path = os.path.join(save_dir, image_name+".png")
    image = Image.fromarray(image)
    image.save(save_dir)


def validate(dataloader: DataLoader, save_dir: "str", save_images: SaveImageType = "sample") -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure all images have the same size
    """
    assert save_images in ["all", "sample", "no"], "save_images parameter must be one of all, sample, or no"

    checkpoint_dir = './src/toenet/checkpoint/'
    # note: model is already in gpu
    model, _, _ = load_checkpoint(checkpoint_dir, 1)  # 1 for GPU
    model.eval()

    psnr_output_batches: List[np.ndarray] = []
    ssim__output_batches = List[np.ndarray] = []

    for sand_dust_images, ground_truth_images, image_names in range(len(dataloader)):
        with torch.no_grad():
            sand_dust_images = sand_dust_images.cuda()
            denoised_images = model(sand_dust_images)

            ground_truth_images = ground_truth_images.cuda()
            mse_per_sample = metrics.mse_per_sample(denoised_images, ground_truth_images)
            psnr_per_sample = metrics.psnr_per_sample(mse_per_sample).cpu().numpy()
            ssim_per_sample = metrics.ssim_per_sample(denoised_images, ground_truth_images).cpu().numpy()

            psnr_output_batches.append(psnr_per_sample)
            ssim__output_batches.append(ssim_per_sample)

            if save_images == "all":
                denoised_images = denoised_images.cpu().numpy()
                denoised_images = np.transpose(denoised_images, axes=[0, 2, 3, 1]).astype('float32')
                denoised_images = np.clip(denoised_images*255, 0.0, 255.0)  # normalize back to 0~255
                map(lambda denoised_image, image_path: save_image(denoised_image, image_name, save_dir), zip(denoised_images, image_names))
            
            elif save_images == "sample":
                image_idx = random.randint(0, len(denoised_images)-1)
                denoised_image, image_name = denoised_images[image_idx], image_names[image_idx]
                save_image(denoised_image, image_name, save_dir)


    return np.concatenate(psnr_output_batches), np.concatenate(ssim__output_batches)
