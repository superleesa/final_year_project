import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage.io import imread
from skimage.transform import resize

def mse_per_sample(predicted, true):
    num_piexels_per_sample = (predicted.size[1]*predicted.size[2]*predicted.size[3])
    return ((predicted - true)**2).sum(0) / num_piexels_per_sample

def pnsr_per_sample(mse_per_sample):
    intensity_max = torch.tensor(1.0)
    return torch.log10(intensity_max / mse_per_sample)

def ssim (predicted, true):
    ssim_metric = StructuralSimilarityIndexMeasure()
    return ssim_metric(predicted, true)

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
