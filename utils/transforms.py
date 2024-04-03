import torch
from torchvision.transforms import v2
import random
import matplotlib.pyplot as plt 
import numpy as np 

# image size constants
W_THRESHOLD = 440
H_THRESHOLD = 330

class ToTensorTransform():
    def __call__(self, noisy_image: torch.Tensor, ground_truth_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ground_truth_image = v2.functional.to_tensor(ground_truth_image)
        noisy_image = v2.functional.to_tensor(noisy_image)
        return noisy_image, ground_truth_image

class RandomResizedCropTransform():
    def __call__(self, noisy_image: torch.Tensor, ground_truth_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ground_truth_image = v2.functional.resize(ground_truth_image, size=(H_THRESHOLD, W_THRESHOLD))
        noisy_image = v2.functional.resize(noisy_image, size=(H_THRESHOLD, W_THRESHOLD))

        # Random crop
        i, j, h, w = v2.RandomCrop.get_params(
            ground_truth_image, output_size=(H_THRESHOLD, W_THRESHOLD))
        ground_truth_image = v2.functional.crop(ground_truth_image, i, j, h, w)
        noisy_image = v2.functional.crop(noisy_image, i, j, h, w)
        return noisy_image, ground_truth_image

class RandomHorizontalFlipTransform():
    def __call__(self, noisy_image: torch.Tensor, ground_truth_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            ground_truth_image = v2.functional.hflip(ground_truth_image)
            noisy_image = v2.functional.hflip(noisy_image)
        return noisy_image, ground_truth_image


class RandomVerticalFlipTransform:
    def __call__(self, noisy_image: torch.Tensor, ground_truth_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            ground_truth_image = v2.functional.vflip(ground_truth_image)
            noisy_image = v2.functional.vflip(noisy_image)
        return noisy_image, ground_truth_image


class CoupledCompose:
    def __init__(self, transformers: list) -> None:
        self.transformers = transformers
        
    def __call__(self, noisy_image, ground_truth_image) -> tuple[torch.Tensor, torch.Tensor]:
        for transformer in self.transformers:
            noisy_image, ground_truth_image = transformer(noisy_image, ground_truth_image)

            # image_noisy_image = noisy_image.numpy()
            # image_noisy_image = np.transpose(image_noisy_image, (1, 2, 0))
            # plt.imshow(image_noisy_image)
            # plt.show()
            # image_ground_truth_image = ground_truth_image.numpy()
            # image_ground_truth_image = np.transpose(image_ground_truth_image, (1, 2, 0))
            # plt.imshow(image_ground_truth_image)
            # plt.show()

        return noisy_image, ground_truth_image


train_unpaired_transform = v2.Compose([
    v2.ToTensor(),
    v2.RandomResizedCrop((H_THRESHOLD, W_THRESHOLD)),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
])

train_paired_transform = CoupledCompose([
    ToTensorTransform(),
    RandomResizedCropTransform(),
    RandomHorizontalFlipTransform(),
    RandomVerticalFlipTransform(),
])

eval_transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((H_THRESHOLD, W_THRESHOLD)),
])