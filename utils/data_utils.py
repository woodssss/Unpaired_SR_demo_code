import torch
import math
import numpy as np
import pywt
from scipy import sparse
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
import xarray
import cv2
import scipy.stats as stats
from utils.distance import compute_KL, compute_w2

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")

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

def make_image_one(mat):
    return mat / np.max(np.abs(mat))


def generate_batch_pink_noise(bs, N):
    """
    Generate a batch of pink noise with shape [bs, N, N, 1].

    Parameters:
        bs (int): Batch size.
        N (int): Dimensions of the square matrix (N x N).

    Returns:
        numpy.ndarray: Pink noise array with shape [bs, N, N, 1].
    """
    pink_noise_batch = []
    for _ in range(bs):
        # Frequency grid
        fx = np.fft.fftfreq(N).reshape(-1, 1)
        fy = np.fft.fftfreq(N).reshape(1, -1)

        # Frequency magnitude
        f_magnitude = np.sqrt(fx ** 2 + fy ** 2)
        f_magnitude[0, 0] = 1e-10  # Avoid division by zero

        # Generate random phase and amplitude
        phase = np.random.uniform(0, 2 * np.pi, (N, N))
        amplitude = np.random.normal(size=(N, N)) + 1j * np.random.normal(size=(N, N))

        # Create spectrum with 1/f scaling
        spectrum = amplitude / f_magnitude
        spectrum *= np.exp(1j * phase)

        # Inverse FFT to spatial domain
        pink_noise = np.fft.ifft2(spectrum).real

        # Normalize
        pink_noise -= pink_noise.mean()
        pink_noise /= pink_noise.std()

        # Append to batch
        pink_noise_batch.append(pink_noise[..., np.newaxis])

    return np.array(pink_noise_batch)


def generate_batch_brown_noise(bs, N, scale=1.0):
    """
    Generate a batch of Brown noise with shape [bs, N, N, 1].

    Parameters:
        bs (int): Batch size.
        N (int): Dimensions of the square matrix (N x N).
        scale (float): Scaling factor for the noise intensity.

    Returns:
        numpy.ndarray: Brown noise array with shape [bs, N, N, 1].
    """
    brown_noise_batch = []
    for _ in range(bs):
        # Frequency grid
        fx = np.fft.fftfreq(N).reshape(-1, 1)
        fy = np.fft.fftfreq(N).reshape(1, -1)

        # Frequency magnitude with Brown noise scaling (1/f^2)
        f_magnitude = np.sqrt(fx ** 2 + fy ** 2)
        f_magnitude[0, 0] = 1e-6  # Avoid division by zero
        f_magnitude = f_magnitude ** 2  # Brown noise has 1/f^2 scaling

        # Generate random phase and amplitude
        phase = np.random.uniform(0, 2 * np.pi, (N, N))
        amplitude = np.random.normal(size=(N, N)) + 1j * np.random.normal(size=(N, N))

        # Create spectrum with 1/f^2 scaling
        spectrum = amplitude / f_magnitude
        spectrum *= np.exp(1j * phase)

        # Inverse FFT to spatial domain
        brown_noise = np.fft.ifft2(spectrum).real

        # Normalize and scale
        brown_noise -= brown_noise.mean()
        brown_noise /= brown_noise.std()
        brown_noise *= scale  # Apply scaling factor

        # Append to batch
        brown_noise_batch.append(brown_noise[..., np.newaxis])

    return np.array(brown_noise_batch)

def prepare_wave_one(mat, wavelet='haar'):
    w_0_ls = []

    N = mat.shape[0]

    #print('prepare wave')
    for i in tqdm(range(N)):

        #coeffs2 = pywt.dwt2(mat[i, ..., 0], wavelet)  # 'haar' wavelet is used here

        coeffs2 = pywt.wavedec2(mat[i, ..., 0], wavelet, level=1, mode='periodization')  # 'haar' wavelet is used here
        cA2, (cH2, cV2, cD2) = coeffs2
        mat2 = np.concatenate([cA2[None, ..., None], cH2[None, ..., None], cV2[None, ..., None], cD2[None, ..., None]],
                              axis=-1)

        w_0_ls.append(mat2)

    mat_w_0 = np.concatenate(w_0_ls, axis=0)
    ##################################################

    return mat_w_0

def prepare_wave(mat, wavelet='haar'):
    w_ng_ls = []
    w_0_ls = []
    w_1_ls = []
    w_2_ls = []

    N = mat.shape[0]

    #print('prepare wave zzz')
    for i in tqdm(range(N)):

        coeffs2 = pywt.dwt2(mat[i, ..., 0], wavelet)  # 'haar' wavelet is used here
        cA2, (cH2, cV2, cD2) = coeffs2
        mat2 = np.concatenate([cA2[None, ..., None], cH2[None, ..., None], cV2[None, ..., None], cD2[None, ..., None]],
                              axis=-1)


        coeffs1 = pywt.dwt2(cA2, wavelet)  # 'haar' wavelet is used here
        cA1, (cH1, cV1, cD1) = coeffs1
        mat1 = np.concatenate([cA1[None, ..., None], cH1[None, ..., None], cV1[None, ..., None], cD1[None, ..., None]],
                              axis=-1)


        coeffs0 = pywt.dwt2(cA1, wavelet)  # 'haar' wavelet is used here
        cA0, (cH0, cV0, cD0) = coeffs0
        mat0 = np.concatenate([cA0[None, ..., None], cH0[None, ..., None], cV0[None, ..., None], cD0[None, ..., None]],
                              axis=-1)

        coeffsng = pywt.dwt2(cA0, wavelet)  # 'haar' wavelet is used here
        cAng, (cHng, cVng, cDng) = coeffsng
        matng = np.concatenate([cAng[None, ..., None], cHng[None, ..., None], cVng[None, ..., None], cDng[None, ..., None]],
                              axis=-1)

        w_ng_ls.append(matng)
        w_0_ls.append(mat0)
        w_1_ls.append(mat1)
        w_2_ls.append(mat2)

    mat_w_2 = np.concatenate(w_2_ls, axis=0)
    mat_w_1 = np.concatenate(w_1_ls, axis=0)
    mat_w_0 = np.concatenate(w_0_ls, axis=0)
    mat_w_ng = np.concatenate(w_ng_ls, axis=0)
    ##################################################

    return mat_w_0, mat_w_1, mat_w_2

def prepare_cv_one(kernel, blur, mat):
    N = mat.shape[0]
    Nx = mat.shape[1]

    ##### preparation for up ##################
    u_2 = np.zeros((N, int(Nx / 2), int(Nx / 2), 1))

    for i in tqdm(range(N)):
        tmp_2 = cv2.resize(mat[i, ..., 0], (int(Nx / 2), int(Nx / 2)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_2 = cv2.blur(tmp_2, (kernel, kernel))

        u_2[i, ..., 0] = tmp_2

    return u_2

def prepare_cv_data(kernel, blur, mat):
    N = mat.shape[0]
    Nx = mat.shape[1]

    ##### preparation for up ##################
    u_0 = np.zeros((N, int(Nx / 8), int(Nx / 8), 1))  ## last dim 0 is ref and 1 is interp
    u_1 = np.zeros((N, int(Nx / 4), int(Nx / 4), 1))
    u_2 = np.zeros((N, int(Nx / 2), int(Nx / 2), 1))

    print('prepare up')
    for i in tqdm(range(N)):

        tmp_0 = cv2.resize(mat[i, ..., 0], (int(Nx / 8), int(Nx / 8)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_0 = cv2.blur(tmp_0, (kernel, kernel))

        u_0[i, ..., 0] = tmp_0

        tmp_1 = cv2.resize(mat[i, ..., 0], (int(Nx / 4), int(Nx / 4)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_1 = cv2.blur(tmp_1, (kernel, kernel))

        u_1[i, ..., 0] = tmp_1

        tmp_2 = cv2.resize(mat[i, ..., 0], (int(Nx / 2), int(Nx / 2)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_2 = cv2.blur(tmp_2, (kernel, kernel))

        u_2[i, ..., 0] = tmp_2

    return u_0, u_1, u_2

def prepare_evo_cv_data(kernel, blur, mat):
    N = mat.shape[0]
    step = mat.shape[1]
    Nx = mat.shape[2]

    ##### preparation for up ##################
    u_0 = np.zeros((N, step, int(Nx / 8), int(Nx / 8), 1))  ## last dim 0 is ref and 1 is interp

    for i in tqdm(range(N)):
        for j in range(step):
            tmp_0 = cv2.resize(mat[i, j, ..., 0], (int(Nx / 8), int(Nx / 8)), interpolation=cv2.INTER_LINEAR)
            if blur:
                tmp_0 = cv2.blur(tmp_0, (kernel, kernel))

            u_0[i, j, ..., 0] = tmp_0
    return u_0


def prepare_up_cv(L, points_x_0, points_x_1, points_x_2, points_x, mat, kernel = 2, blur = None):
    N = mat.shape[0]
    Nx = mat.shape[1]

    ##### preparation for up ##################
    mat_u_0 = np.zeros((N, int(Nx / 4), int(Nx / 4), 2))
    mat_u_1 = np.zeros((N, int(Nx / 2), int(Nx / 2), 2))
    mat_u_2 = np.zeros((N, Nx, Nx, 2))
    mat_u_d = np.zeros((N, Nx, Nx, 2))

    print('prepare up')
    for i in tqdm(range(N)):

        tmp_0 = cv2.resize(mat[i, ...], (int(Nx / 8), int(Nx / 8)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_0 = cv2.blur(tmp_0, (kernel, kernel))

        tmp_1 = cv2.resize(mat[i, ...], (int(Nx / 4), int(Nx / 4)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_1 = cv2.blur(tmp_1, (kernel, kernel))

        tmp_2 = cv2.resize(mat[i, ...], (int(Nx / 2), int(Nx / 2)), interpolation=cv2.INTER_LINEAR)
        if blur:
            tmp_2 = cv2.blur(tmp_2, (kernel, kernel))

        u0 = interp_pbc_2d(points_x_1, points_x_0, L, tmp_0)
        u1 = interp_pbc_2d(points_x_2, points_x_1, L, tmp_1)
        u2 = interp_pbc_2d(points_x, points_x_2, L, tmp_2)
        ud = interp_pbc_2d(points_x, points_x_0, L, tmp_0)

        mat_u_0[i, ..., 0] = make_image_one(tmp_1)
        mat_u_0[i, ..., 1] = make_image_one(u0)
        #

        mat_u_1[i, ..., 0] = make_image_one(tmp_2)
        mat_u_1[i, ..., 1] = make_image_one(u1)

        mat_u_2[i, ..., 0] = make_image_one(mat[i, ..., 0])
        mat_u_2[i, ..., 1] = make_image_one(u2)

        mat_u_d[i, ..., 0] = make_image_one(mat[i, ..., 0])
        mat_u_d[i, ..., 1] = make_image_one(ud)

    return mat_u_0, mat_u_1, mat_u_2, mat_u_d

def prepare_up_skip(L, points_x_0, points_x_1, points_x_2, points_x, mat):
    N = mat.shape[0]
    Nx = mat.shape[1]

    ##### preparation for up ##################
    mat_u_0 = np.zeros((N, int(Nx / 4), int(Nx / 4), 2))  ## last dim 0 is ref and 1 is interp
    mat_u_1 = np.zeros((N, int(Nx / 2), int(Nx / 2), 2))
    mat_u_2 = np.zeros((N, Nx, Nx, 2))
    mat_u_d = np.zeros((N, Nx, Nx, 2))

    print('prepare up')
    for i in tqdm(range(N)):
        tmp_0 = mat[i, ::8, ::8, 0]
        tmp_1 = mat[i, ::4, ::4, 0]
        tmp_2 = mat[i, ::2, ::2, 0]

        u0 = interp_pbc_2d(points_x_1, points_x_0, L, tmp_0)
        u1 = interp_pbc_2d(points_x_2, points_x_1, L, tmp_1)
        u2 = interp_pbc_2d(points_x, points_x_2, L, tmp_2)
        ud = interp_pbc_2d(points_x, points_x_0, L, tmp_0)

        mat_u_0[i, ..., 0] = tmp_1
        mat_u_0[i, ..., 1] = u0
        #

        mat_u_1[i, ..., 0] = tmp_2
        mat_u_1[i, ..., 1] = u1

        mat_u_2[i, ..., 0] = mat[i, ..., 0]
        mat_u_2[i, ..., 1] = u2

        mat_u_d[i, ..., 0] = mat[i, ..., 0]
        mat_u_d[i, ..., 1] = ud

    return mat_u_0, mat_u_1, mat_u_2, mat_u_d


def interp_pbc_1d(x_new, x, L, f):
    x = np.concatenate([x, np.ones((1))*L], axis=0)
    f = np.concatenate([f, np.ones((1)) * f[0]], axis=0)
    func = CubicSpline(x, f)

    f = func(x_new)
    return f

def interp_pbc_2d(x_new, x, L, f):
    Nx = x.shape[0]
    Nx_new = x_new.shape[0]

    f_f_1 = np.zeros((Nx, Nx_new))
    f_f_2 = np.zeros((Nx_new, Nx_new))

    for i in range(Nx):
        f_f_1[i, :] = interp_pbc_1d(x_new, x, L, f[i, :])

    for j in range(Nx_new):
        f_f_2[:, j] = interp_pbc_1d(x_new, x, L, f_f_1[:, j])

    return f_f_2

def interp_pbc_2d_batch(x_new, x, L, f_mat):
    bs = f_mat.shape[0]
    Nx_new = x_new.shape[0]

    f_new_mat = np.zeros((bs, Nx_new, Nx_new, 1))

    for i in range(bs):
        f_new_mat[i, ..., 0] = interp_pbc_2d(x_new, x, L, f_mat[i, ..., 0])

    return f_new_mat


def get_grid(Nx, L):
 dx = L/Nx
 points = np.linspace(0, L - dx, Nx)
 return points


def batch_iwt(Nx, ca, coeff):
    bs = ca.shape[0]
    res = np.zeros((bs, Nx, Nx, 1))
    for i in range(bs):
        tmp_coeff = (ca[i, ..., 0], (coeff[i, ..., 0], coeff[i, ..., 1], coeff[i, ..., 2]))
        res[i, ..., 0] = pywt.idwt2(tmp_coeff, wavelet='haar')

    return res

def recover_wave(Nf, ca, coeff):
    return batch_iwt(Nf, ca, coeff)

def down_sample_cv(mat, Nx, kernel=2, blur=None):
    bs = mat.shape[0]
    down = np.zeros((bs, Nx, Nx, 1))
    for i in range(bs):
        down[i, ..., 0] = cv2.resize(mat[i, ...], (Nx, Nx), interpolation=cv2.INTER_LINEAR)
        if blur:
            down[i, ..., 0] = cv2.blur(down[i, ..., 0], (kernel, kernel))

    return down

def down_sample_evo_cv(mat, Nx, kernel=2, blur=None):
    bs = mat.shape[0]
    T = mat.shape[1]
    down = np.zeros((bs, T, Nx, Nx, 1))
    for i in range(bs):
        for j in range(T):
            down[i, j, ..., 0] = cv2.resize(mat[i, j, ...], (Nx, Nx), interpolation=cv2.INTER_LINEAR)
            if blur:
                down[i, j, ..., 0] = cv2.blur(down[i, j, ..., 0], (kernel, kernel))

    return down

def get_relative_l2_error(pd, ref):
    return np.linalg.norm(pd[..., 0] - ref[..., 0]) / np.linalg.norm(ref[..., 0])

def compute_energy_spectrum_one(image, max_k):
    npix = image.shape[0]
    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image) ** 2
    #fourier_amplitudes = np.abs(fourier_image)

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0] ** 2 + kfreq2D[1] ** 2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.05, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic="mean",
                                         bins=kbins)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    return kvals[:max_k], Abins[:max_k]

def compute_energy_spectrum_average(mat, max_k = 80):
    bs = mat.shape[0]

    E_k = np.zeros((max_k))

    for i in range(bs):
        k_vec, E_k_tmp = compute_energy_spectrum_one(mat[i, ..., 0], max_k)

        E_k += E_k_tmp

    return k_vec, E_k/bs

def compute_TVD(pd, ref):
    return np.mean(np.sum(np.abs(pd[..., 0] - ref[..., 0]), axis=(1,2)) / np.sum(np.abs(ref[..., 0]), axis=(1,2)))



def compute_melr(E_pred, E_ref, max_k=80, weighted=True, power=1):
    # Total number of modes (k)
    num_modes = max_k

    # Unweighted or Weighted weights
    if weighted:
        weights = E_ref / np.sum(E_ref)  # Weighted case: energy-based weights
    else:
        weights = np.ones(num_modes) / num_modes   # Unweighted case: equal weights

    # Compute MELR
    vec = np.abs(np.log(E_pred / E_ref))
    #vec = (np.log(E_pred / E_ref))
    melr = np.sum(weights**power * np.abs(np.log(E_pred / E_ref)))

    # print(E_ref[-20:], E_ref[:20])
    # zxc

    return vec, melr

def rbf_kernel(X, Y, sigma):
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    distances = XX + YY.T - 2 * np.dot(X, Y.T)
    return np.exp(-distances / (2 * sigma ** 2))

def compute_mmd(X, Y, sigma=0.01):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two samples X and Y using an RBF kernel.

    Parameters:
        X (np.ndarray): First sample, shape (n, d), where n is the number of points and d is the dimensionality.
        Y (np.ndarray): Second sample, shape (m, d), where m is the number of points.
        sigma (float): Bandwidth of the RBF kernel.

    Returns:
        float: The MMD statistic.
    """

    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    # Compute the kernels
    K_XX = rbf_kernel(X, X, sigma)
    K_YY = rbf_kernel(Y, Y, sigma)
    K_XY = rbf_kernel(X, Y, sigma)

    # Compute the MMD statistic
    m = X.shape[0]
    n = Y.shape[0]

    term_XX = np.sum(K_XX) / (m * m)
    term_YY = np.sum(K_YY) / (n * n)
    term_XY = 2 * np.sum(K_XY) / (m * n)

    mmd = term_XX + term_YY - term_XY
    return mmd

def compute_cov_rmse(true_data, pred_data):
    bs = true_data.shape[0]
    # Step 1: Flatten the 32x32 matrices into vectors of shape [100, 1024]
    true_data_flattened = true_data.reshape(bs, -1)  # shape [100, 1024]
    pred_data_flattened = pred_data.reshape(bs, -1)  # shape [100, 1024]

    # Step 2: Compute covariance matrices
    true_cov = np.cov(true_data_flattened, rowvar=False)  # shape [1024, 1024]
    pred_cov = np.cov(pred_data_flattened, rowvar=False)  # shape [1024, 1024]

    # Step 3: Compute element-wise squared difference
    cov_diff = (true_cov - pred_cov) ** 2

    # Step 4: Compute mean of squared differences
    mean_squared_diff = np.mean(cov_diff)

    # Step 5: Compute square root of mean squared difference (covRMSE)
    cov_rmse = np.sqrt(mean_squared_diff)

    return cov_rmse

def compute_all_error(pd, ref, max_k):
    RMSE = get_relative_l2_error(pd, ref)
    #covRMSE = compute_cov_rmse(pd, ref)
    MMD = compute_mmd(pd, ref, sigma=0.01)
    TVD = compute_TVD(pd, ref)

    k_vec, E_pd = compute_energy_spectrum_average(pd, max_k)
    k_vec, E_ref = compute_energy_spectrum_average(ref, max_k)

    log_vec, melr_u = compute_melr(E_pd, E_ref, max_k, weighted=False)
    log_vec, melr_w = compute_melr(E_pd, E_ref, max_k, weighted=True, power=1)

    return RMSE, MMD, TVD, melr_u, melr_w, k_vec, log_vec

def compute_all_error_2(pd, ref, max_k):
    RMSE = get_relative_l2_error(pd, ref)
    #covRMSE = compute_cov_rmse(pd, ref)
    MMD = compute_mmd(pd, ref, sigma=0.01)
    TVD = compute_TVD(pd, ref)

    k_vec, E_pd = compute_energy_spectrum_average(pd, max_k)
    k_vec, E_ref = compute_energy_spectrum_average(ref, max_k)

    log_vec, melr_u = compute_melr(E_pd, E_ref, max_k, weighted=False)
    log_vec, melr_w = compute_melr(E_pd, E_ref, max_k, weighted=True, power=1)

    w2 = compute_w2(pd, ref)
    kl = compute_KL(pd, ref)

    return RMSE, MMD, TVD, melr_u, melr_w, w2, kl

def find_inter_t(X, Y, get_perturbed_x, marginal_prob_fn, t1_ls, cmin=0.25, cmax=4, metric= 'spectrum', weight=True, max_k = 16, sigma=0.1):
    X = torch.from_numpy(X).float().to(device)
    Y = torch.from_numpy(Y).float().to(device)

    min_value = float('inf')
    opt_t1 = 100
    opt_t2 = 100
    for t1 in t1_ls:
        t2_ls = np.linspace(cmin*t1, np.minimum(cmax*t1, 1), 20)
        for t2 in t2_ls:
            #print(t1, t2)
            latent_x = get_perturbed_x(X, marginal_prob_fn, t=t1)
            latent_y = get_perturbed_x(Y, marginal_prob_fn, t=t2)

            if metric == 'spectrum':
                k_vec, E_x =compute_energy_spectrum_average(latent_x.detach().cpu().numpy(), max_k=max_k)
                _, E_y = compute_energy_spectrum_average(latent_y.detach().cpu().numpy(), max_k=max_k)

                _, dist = compute_melr(E_x, E_y, max_k=max_k, weighted=weight, power=2)

            else:
                dist = compute_mmd(latent_x.detach().cpu().numpy(), latent_y.detach().cpu().numpy(), sigma=sigma)

            if dist < min_value:
                min_value = dist
                opt_t1, opt_t2 = t1, t2

    return min_value, opt_t1, opt_t2

def find_inter_t_ode(X, Y, ode_solver, marginal_prob_fn, get_sde_forward_fn, mdl_c, mdl_d, t1_ls, t2_ls, metric= 'spectrum', weight=True, max_k = 16, sigma=0.1):
    X = torch.from_numpy(X).float().to(device)
    Y = torch.from_numpy(Y).float().to(device)

    min_value = float('inf')
    opt_t1 = 100
    opt_t2 = 100
    for t1 in t1_ls:
        for t2 in t2_ls:
            print(t1, t2)
            latent_x = ode_solver(mdl_c, marginal_prob_fn, get_sde_forward_fn, X, forward=1,
                        eps=1e-4, T=t1)
            latent_y = ode_solver(mdl_d, marginal_prob_fn, get_sde_forward_fn, Y, forward=1,
                                  eps=1e-4, T=t1)

            if metric == 'spectrum':
                k_vec, E_x =compute_energy_spectrum_average(latent_x.detach().cpu().numpy(), max_k=max_k)
                _, E_y = compute_energy_spectrum_average(latent_y.detach().cpu().numpy(), max_k=max_k)

                _, dist = compute_melr(E_x, E_y, max_k=max_k, weighted=weight, power=2)

            elif metric== 'mmd':
                dist = compute_mmd(latent_x.detach().cpu().numpy(), latent_y.detach().cpu().numpy(), sigma=sigma)
            elif metric == 'W2':
                dist = compute_w2(latent_x.detach().cpu().numpy(), latent_y.detach().cpu().numpy())
            elif metric == 'KL':
                dist = compute_KL(latent_x.detach().cpu().numpy(), latent_y.detach().cpu().numpy())


            if dist < min_value:
                min_value = dist
                opt_t1, opt_t2 = t1, t2

    return min_value, opt_t1, opt_t2

def compute_TVD_vector(pd, ref):
    return np.sum(np.abs(pd[..., 0] - ref[..., 0]), axis=(1,2)) / np.sum(np.abs(ref[..., 0]), axis=(1,2))

def get_relative_l2_vector(pd, ref):
    return np.sqrt(np.sum(np.square(pd[..., 0] - ref[..., 0]), axis=(1,2)) / np.sum(np.square(ref[..., 0]), axis=(1,2)))

def get_MELR_vector(pd, ref, max_k = 80):
    bs = pd.shape[0]
    res_u = np.zeros((bs,))
    res_w = np.zeros((bs,))
    for i in range(bs):
        k_vec, E_pd = compute_energy_spectrum_average(pd[[i], ...], max_k)
        k_vec, E_ref = compute_energy_spectrum_average(ref[[i], ...], max_k)
        log_vec, tmp_melr_u = compute_melr(E_pd, E_ref, max_k, weighted=False)
        log_vec, tmp_melr_w = compute_melr(E_pd, E_ref, max_k, weighted=True, power=1)
        res_u[i] = tmp_melr_u
        res_w[i] = tmp_melr_w

    return res_u, res_w


