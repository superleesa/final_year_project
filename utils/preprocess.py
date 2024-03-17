import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import os
import random
from tqdm import tqdm


def crop_image(image, w_threshold, h_threshold):
    assert image.width >= w_threshold and image.height >= h_threshold, "to crop, image size must be bigger than or equal to the threshold values"

    # choose top and right randomly -> bottom and left automallycally determined
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