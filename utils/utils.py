import torch
import math
import numpy as np
from scipy.fft import fft2, fftshift
import scipy.stats as stats

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def num2str_deciaml(x):
    s = str(x)
    c = ''
    for i in range(len(s)):
        if s[i] == '0':
            c = c + 'z'
        elif s[i] == '.':
            c = c + 'p'
        elif s[i] == '-':
            c = c + 'n'
        else:
            c = c + s[i]

    return c

def tensor2nump(x):
    return x.cpu().detach().numpy()

def make_tensor(*args):
    return [torch.from_numpy(arg).float().to(device) for arg in args]

def make_image(mat):
    for i in range(mat.shape[0]):
        mat[i, ...] /= np.max(np.abs(mat[i, ...]))

    return mat


# def compute_energy_spectrum(image, max_k=100):
#     npix = image.shape[0]
#     fourier_image = np.fft.fftn(image)
#     fourier_amplitudes = np.abs(fourier_image) ** 2
#
#     kfreq = np.fft.fftfreq(npix) * npix
#     kfreq2D = np.meshgrid(kfreq, kfreq)
#     knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)
#
#     knrm = knrm.flatten()
#     fourier_amplitudes = fourier_amplitudes.flatten()
#
#     kbins = np.arange(0.5, npix // 2 + 1, 1.)
#     kvals = 0.5 * (kbins[1:] + kbins[:-1])
#     Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
#                                          statistic="mean",
#                                          bins=kbins)
#     Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
#     return kvals[:max_k], Abins[:max_k]






