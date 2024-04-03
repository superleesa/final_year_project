import numpy as np
import torch

CHANNEL_SIZE = 255

def postprocess(images: torch.Tensor) -> list[np.ndarray]:
    images = images.cpu().numpy()
    # images = np.transpose(images, axes=[0, 2, 3, 1]).astype('float32')
    images = np.transpose(images, axes=[1, 2, 0]).astype('float32')
    images = np.clip(images * CHANNEL_SIZE, 0.0, CHANNEL_SIZE) # normalize back to 0~255
    return images