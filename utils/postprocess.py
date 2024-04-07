import numpy as np
import torch

MAX_PIXEL_VALUE = 255

def postprocess(images: torch.Tensor) -> list[np.ndarray]:
    images = images.cpu().numpy()
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        
    images = np.transpose(images, axes=[0, 2, 3, 1]).astype('float32')
    images = np.clip(images * MAX_PIXEL_VALUE, 0.0, MAX_PIXEL_VALUE)  # normalize back to 0~255
    return images


def postprocess_tensor(images: torch.Tensor) -> torch.Tensor:
    if len(images.shape) == 3:
        images = images.unsqueeze(0)

    images = torch.permute(images, dims=[0, 2, 3, 1]).float()
    images = torch.clip(images * MAX_PIXEL_VALUE, min=0.0, max=MAX_PIXEL_VALUE)  # normalize back to 0~255
    return images
