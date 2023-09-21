import matplotlib
import numpy as np
from PIL import Image
import glob, os

matplotlib.use('agg')

def prepare_hazy_image(file_name):
    img_pil = crop_image(get_image(file_name, -1)[0], d=32)
    #print(np.array(img_pil).shape)(896, 1184, 3)
    #img_pil = load(file_name)
    return pil_to_np(img_pil)

def crop_image(img, d=32):
    """
    Make dimensions divisible by d
    :param pil img:
    :param d:
    :return:
    """
    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)
    bbox = [
        int((img.size[0] - new_size[0]) / 2),
        int((img.size[1] - new_size[1]) / 2),
        int((img.size[0] + new_size[0]) / 2),
        int((img.size[1] + new_size[1]) / 2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a specific size.
    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)
    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0] != -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)
    #    3*460*620
    #    print(np.shape(img_np))
    return img, img_np

def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.
    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.
def save_image(name, image_np, output_path="output/new_nyu/normal/"):
    p = np_to_pil(image_np)
    p.save(output_path + "{}.jpg".format(name))
#    p.save(output_path + "{}.jpg".format(name))

def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.
    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def resize():
    for files in glob.glob('data/*'):
        opfile = r'./resize/'
        if (os.path.isdir(opfile)==False):
            os.mkdir(opfile)
        hazy_add = files
        name = hazy_add[(hazy_add.index('\\') + 1):hazy_add.index('.')]
        print(name)
        hazy_img = prepare_hazy_image(hazy_add)
        save_image(name + "", np.clip(hazy_img, 0, 1), "resize/")


if __name__ == "__main__":

    resize()
