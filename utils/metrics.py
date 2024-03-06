import torch


def mse_per_sample(predicted, true):
    num_piexels_per_sample = (predicted.size[1]*predicted.size[2]*predicted.size[3])
    return ((predicted - true)**2).sum(0) / num_piexels_per_sample

def pnsr_per_sample(mse_per_sample):
    intensity_max = torch.tensor(1.0)
    return torch.log10(intensity_max / mse_per_sample)