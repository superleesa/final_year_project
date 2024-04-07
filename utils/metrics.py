import torch
from torchmetrics.functional.image import structural_similarity_index_measure
from skimage.io import imread
from skimage.transform import resize
from torch.nn.functional import mse_loss
from torchmetrics.functional.image import peak_signal_noise_ratio

def mse_per_sample(predicted, true):
    return mse_loss(predicted, true)

def psnr_per_sample(predicted, true):
    return peak_signal_noise_ratio(predicted, true, data_range=255, reduction="none", dim=(1, 2, 3))

def ssim_per_sample(predicted, true):
    return structural_similarity_index_measure(predicted, true, reduction="none")

def loe(predicted_path, true_path):
    I = imread(predicted_path) 
    Ie = imread(true_path) 

    N, M, _ = I.shape

    L = torch.max(torch.tensor(I), dim=2).values
    Le = torch.max(torch.tensor(Ie), dim=2).values

    r = 50 / min(M, N)
    Md = round(M * r)
    Nd = round(N * r)
    Ld = torch.tensor(resize(L.numpy(), (Nd, Md)))
    Led = torch.tensor(resize(Le.numpy(), (Nd, Md)))

    RD = torch.zeros((Nd, Md))

    for y in range(Md):
        for x in range(Nd):
            E = torch.logical_xor(Ld[x, y] >= Ld, Led[x, y] >= Led)
            RD[x, y] = torch.sum(E).item()

    LOE = torch.sum(RD) / (Md * Nd)
    return LOE.item()