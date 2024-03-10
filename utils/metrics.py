import torch
import numpy
import scipy.signal
import scipy.ndimage

def mse_per_sample(predicted, true):
    num_piexels_per_sample = (predicted.size[1]*predicted.size[2]*predicted.size[3])
    return ((predicted - true)**2).sum(0) / num_piexels_per_sample

def pnsr_per_sample(mse_per_sample):
    intensity_max = torch.tensor(1.0)
    return torch.log10(intensity_max / mse_per_sample)

# Original Paper: H. R. Sheikh and A. C. Bovik, "Image Information and Visual Quality"., IEEE Transactions on Image Processing
# Reference: https://github.com/aizvorski/video-quality/blob/master/vifp.py
# THIS IS A COMPUTATIONALLY SIMPLER DERIVATIVE OF THE ALGORITHM PRESENTED IN THE PAPER
def vifp_per_sample(predicted, true):
    sigma_nsq=2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):
       
        N = 2**(4-scale+1) + 1
        sd = N/5.0

        if (scale > 1):
            true = scipy.ndimage.gaussian_filter(true, sd)
            predicted = scipy.ndimage.gaussian_filter(predicted, sd)
            true = true[::2, ::2]
            predicted = predicted[::2, ::2]
                
        mu1 = scipy.ndimage.gaussian_filter(true, sd)
        mu2 = scipy.ndimage.gaussian_filter(predicted, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(true * true, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(predicted * predicted, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(true * predicted, sd) - mu1_mu2
        
        sigma1_sq[sigma1_sq<0] = 0
        sigma2_sq[sigma2_sq<0] = 0
        
        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        
        g[sigma1_sq<eps] = 0
        sv_sq[sigma1_sq<eps] = sigma2_sq[sigma1_sq<eps]
        sigma1_sq[sigma1_sq<eps] = 0
        
        g[sigma2_sq<eps] = 0
        sv_sq[sigma2_sq<eps] = 0
        
        sv_sq[g<0] = sigma2_sq[g<0]
        g[g<0] = 0
        sv_sq[sv_sq<=eps] = eps
        
        num += numpy.sum(numpy.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += numpy.sum(numpy.log10(1 + sigma1_sq / sigma_nsq))
        
    vifp = num/den

    if numpy.isnan(vifp):
        return 1.0
    else:
        return vifp
