import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
import time
from scipy import integrate
from tqdm import tqdm
import pywt
from matplotlib import cm
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")
def mylogger(filename, content):
    with open(filename, 'a') as fw:
        print(content, file=fw)

def get_sde_forward(x, t, sigma):
    # out put mean and difssuion coeff
    drift = 0 * x
    diffusion = sigma ** t
    return drift, diffusion


def marginal_prob(x, t, sigma):
    t = torch.tensor(t).to(torch.float32).to(device)
    mean = x
    std = torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))
    return mean, std


def get_perturbed_x(x, marginal_prob, t, eps=1e-5):
  random_t = t * torch.ones(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  mean, std = marginal_prob(x, random_t)
  perturbed_x = mean + z * std[:, None, None, None]
  return perturbed_x

def loss_score_t(model, x, marginal_prob, eps=1e-5):
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  mean, std = marginal_prob(x, random_t)
  perturbed_x = mean + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)

  loss = torch.mean(torch.sum((score + z)**2, dim=(1,2,3)))
  return loss


def ode_solver(score_model, marginal_prob, get_sde_forward, init_x, forward, atol=1e-6, rtol=1e-6, device=device, eps=1e-3, T=1):

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = get_sde_forward(x, torch.tensor(t))[1].cpu().numpy()
        std = marginal_prob(x, t)[1].cpu().numpy()
        return -0.5 * (g ** 2) / std * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    if forward ==1:
        # forward
        res = integrate.solve_ivp(ode_func, (eps, T), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    else:
        res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device, dtype=torch.float32).reshape(shape)

    return x

def EM(score_model, marginal_prob, get_sde_forward, init_x, N, device='cuda', eps=1e-3, T=1):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    batch_size = init_x.shape[0]
    time_steps = torch.linspace(T, eps, N, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step

            g = get_sde_forward(x, time_step)[1]
            std = marginal_prob(x, time_step)[1]

            mean_x = x + (g ** 2)/std * score_model(x, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x)
            # Do not include any noise in the last sampling step.
    return mean_x


def pc_sampler(score_model, marginal_prob, get_sde_forward, init_x, N, snr=0.16, device='cuda', eps=1e-3, T=1):
    batch_size = init_x.shape[0]
    time_steps = np.linspace(T, eps, N)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step

            g = get_sde_forward(x, time_step)[1]
            std = marginal_prob(x, time_step)[1]

            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step) / std
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            # Predictor step (Euler-Maruyama)
            x_mean = x + (g ** 2) / std * score_model(x, batch_time_step) * step_size
            x = x_mean + math.sqrt(g ** 2 * step_size) * torch.randn_like(x)

            # The last step does not include any noise
        return x_mean


################# for conditional ##############################
def loss_score_t_cond(model, x, cond, marginal_prob, eps=1e-5):
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)
  mean, std = marginal_prob(x, random_t)
  perturbed_x = mean + z * std[:, None, None, None]
  #print('pp', perturbed_x.shape, cond.shape)
  score = model(perturbed_x, cond, random_t)

  loss = torch.mean(torch.sum((score + z)**2, dim=(1,2,3)))
  return loss

def ode_solver_cond(score_model, marginal_prob, get_sde_forward, init_x, cond, forward=2, atol=1e-6, rtol=1e-6, device='cuda', eps=1e-3, T=1):

    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, cond, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t
        g = get_sde_forward(x, torch.tensor(t))[1].cpu().numpy()
        std = marginal_prob(x, t)[1].cpu().numpy()
        return -0.5 * (g ** 2) / std * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    if forward ==1:
        # forward
        res = integrate.solve_ivp(ode_func, (eps, T), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    else:
        res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

    return x

def EM_cond(score_model, marginal_prob, get_sde_forward, init_x, cond, N, device=device, eps=1e-3, T=1):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
      batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    batch_size = init_x.shape[0]
    time_steps = torch.linspace(T, eps, N, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x

    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step

            g = get_sde_forward(x, time_step)[1]
            std = marginal_prob(x, time_step)[1]

            mean_x = x + (g ** 2)/std * score_model(x, cond, batch_time_step) * step_size
            x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x)
            # Do not include any noise in the last sampling step.
    return mean_x


def pc_sampler_cond(score_model, marginal_prob, get_sde_forward, init_x, cond, N, snr=0.16, device=device, eps=1e-3, T=1):
    batch_size = init_x.shape[0]
    time_steps = np.linspace(T, eps, N)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for time_step in time_steps:
            batch_time_step = torch.ones(batch_size, device=device) * time_step

            g = get_sde_forward(x, time_step)[1]
            std = marginal_prob(x, time_step)[1]

            # Corrector step (Langevin MCMC)
            grad = score_model(x, cond, batch_time_step) / std
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            # Predictor step (Euler-Maruyama)
            x_mean = x + (g ** 2) / std * score_model(x, cond, batch_time_step) * step_size
            x = x_mean + math.sqrt(g ** 2 * step_size) * torch.randn_like(x)

            # The last step does not include any noise
        return x_mean

def gen_sample_cond(model, cond, Nx, bs, dim, marginal_prob_fn, get_sde_forward_fn, method, N=1000, eps = 1e-5, T=1):
    noise = torch.randn(bs, Nx, Nx, dim).to(device)
    t = torch.ones(bs, device=device) * T
    noise *= marginal_prob_fn(noise, t)[1][:, None, None, None]

    print('step is ', N)

    #print('ss', a.shape, noise.shape)
    if method == 'ode_solver_cond':
        sample = ode_solver_cond(model, marginal_prob_fn, get_sde_forward_fn, noise, cond, forward=2, eps=eps, T=T)
    elif method == 'pc_sampler_cond':
        sample = pc_sampler_cond(model, marginal_prob_fn, get_sde_forward_fn, noise, cond, N=N, eps=eps, T=T)
    elif method == 'EM_cond':
        sample = EM_cond(model, marginal_prob_fn, get_sde_forward_fn, noise, cond, N=N, eps=eps, T=T)

    return sample

#
#
# def ode_solver_cond(score_model,
#                 marginal_prob_std,
#                 diffusion_coeff,
#                 init_x,
#                 a,
#                 forward,
#                 atol=1e-5,
#                 rtol=1e-5,
#                 device='cuda',
#                 eps=1e-4,
#                 T = 1):
#     """Generate samples from score-based models with black-box ODE solvers.
#
#     Args:
#       score_model: A PyTorch model that represents the time-dependent score-based model.
#       marginal_prob_std: A function that returns the standard deviation
#         of the perturbation kernel.
#       diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
#       batch_size: The number of samplers to generate by calling this function once.
#       atol: Tolerance of absolute errors.
#       rtol: Tolerance of relative errors.
#       device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
#       forward: 1 means forward and 2 means backward
#       z: The latent code that governs the final sample. If None, we start from p_1;
#         otherwise, we start from the given z.
#       eps: The smallest time step for numerical stability.
#     """
#     shape = init_x.shape
#
#     def score_eval_wrapper(sample, time_steps):
#         """A wrapper of the score-based model for use by the ODE solver."""
#         sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
#         time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
#         with torch.no_grad():
#             # print(sample.dtype, a.dtype, time_steps.dtype)
#             # zxc
#             score = score_model(sample, a, time_steps)
#         return score.cpu().numpy().reshape((-1,)).astype(np.float64)
#
#     def ode_func(t, x):
#         """The ODE function for use by the ODE solver."""
#         time_steps = np.ones((shape[0],)) * t
#         g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
#         std = marginal_prob_std(t).cpu().numpy()
#         return -0.5 * (g ** 2) / std * score_eval_wrapper(x, time_steps)
#
#     # Run the black-box ODE solver.
#     if forward ==1:
#         # forward
#         res = integrate.solve_ivp(ode_func, (eps, T), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
#     else:
#         res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
#     print(f"Number of function evaluations: {res.nfev}")
#     x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
#
#     return x
#
#
# def pc_sampler_cond(score_model,
#                marginal_prob_std,
#                diffusion_coeff,
#                init_x,
#                a,
#                num_steps,
#                snr=0.16,
#                device='cuda',
#                eps=1e-3,
#                T= 1):
#     """Generate samples from score-based models with Predictor-Corrector method.
#
#     Args:
#       score_model: A PyTorch model that represents the time-dependent score-based model.
#       marginal_prob_std: A function that gives the standard deviation
#         of the perturbation kernel.
#       diffusion_coeff: A function that gives the diffusion coefficient
#         of the SDE.
#       batch_size: The number of samplers to generate by calling this function once.
#       num_steps: The number of sampling steps.
#         Equivalent to the number of discretized time steps.
#       device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
#       eps: The smallest time step for numerical stability.
#
#     Returns:
#       Samples.
#     """
#     batch_size = a.shape[0]
#     time_steps = np.linspace(T, eps, num_steps)
#     step_size = time_steps[0] - time_steps[1]
#     x = init_x
#     with torch.no_grad():
#         for time_step in time_steps:
#             batch_time_step = torch.ones(batch_size, device=device) * time_step
#             std = marginal_prob_std(batch_time_step)[:, None, None, None]
#             # Corrector step (Langevin MCMC)
#             grad = score_model(x, a, batch_time_step)/std
#             grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
#             noise_norm = np.sqrt(np.prod(x.shape[1:]))
#             langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
#             x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)
#
#             # Predictor step (Euler-Maruyama)
#             g = diffusion_coeff(batch_time_step)
#             x_mean = x + (g ** 2)[:, None, None, None]/std * score_model(x, a, batch_time_step) * step_size
#             x = x_mean + torch.sqrt(g ** 2 * step_size)[:, None, None, None] * torch.randn_like(x)
#
#             # The last step does not include any noise
#         return x_mean
#
#
# def EM_cond(score_model,
#         marginal_prob_std,
#         diffusion_coeff,
#         init_x,
#         a,
#         num_steps,
#         device='cuda',
#         eps=1e-3,
#         T=1):
#     """Generate samples from score-based models with the Euler-Maruyama solver.
#
#     Args:
#       score_model: A PyTorch model that represents the time-dependent score-based model.
#       marginal_prob_std: A function that gives the standard deviation of
#         the perturbation kernel.
#       diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
#       batch_size: The number of samplers to generate by calling this function once.
#       num_steps: The number of sampling steps.
#         Equivalent to the number of discretized time steps.
#       device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
#       eps: The smallest time step for numerical stability.
#
#     Returns:
#       Samples.
#     """
#     batch_size = a.shape[0]
#     time_steps = torch.linspace(T, eps, num_steps, device=device)
#     step_size = time_steps[0] - time_steps[1]
#     x = init_x
#     with torch.no_grad():
#         for time_step in time_steps:
#             batch_time_step = torch.ones(batch_size, device=device) * time_step
#             std = marginal_prob_std(batch_time_step)[:, None, None, None]
#             g = diffusion_coeff(batch_time_step)
#             mean_x = x + (g ** 2)[:, None, None, None]/std * score_model(x, a, batch_time_step) * step_size
#             x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
#             # Do not include any noise in the last sampling step.
#     return mean_x
#
# def gen_sample(model, Nx, bs, marginal_prob_std_fn, diffusion_coeff_fn, ode_solver, forward, T, method, init_x=None):
#     if init_x!=None:
#         init_x = init_x
#     else:
#         noise = torch.randn(bs, Nx, Nx, 1).to(device)
#         init_x = noise
#         t = torch.ones(bs, device=device)
#         noise *= marginal_prob_std_fn(t)[:, None, None, None]
#
#     if method == 'ode_solver':
#         sample = ode_solver(model, marginal_prob_std_fn, diffusion_coeff_fn, init_x, forward=forward, eps=1e-5, T=T)
#     elif method == 'EM':
#         sample = EM(model, marginal_prob_std_fn, diffusion_coeff_fn, init_x, num_steps=1000, eps=1e-5)
#
#     return sample
#
# def gen_sample_cond(model, a, Nx, bs, dim, marginal_prob_std_fn, diffusion_coeff_fn, method, eps = 1e-5, T=1):
#     noise = torch.randn(bs, Nx, Nx, dim).to(device)
#     t = torch.ones(bs, device=device) * T
#     noise *= marginal_prob_std_fn(t)[:, None, None, None]
#
#     #print('ss', a.shape, noise.shape)
#     if method == 'ode_solver_cond':
#         sample = ode_solver_cond(model, marginal_prob_std_fn, diffusion_coeff_fn, noise, a, forward=2, eps=eps, T=T)
#     elif method == 'pc_sampler_cond':
#         sample = pc_sampler_cond(model, marginal_prob_std_fn, diffusion_coeff_fn, noise, a, num_steps=2000, eps=eps, T=T)
#     elif method == 'EM_cond':
#         sample = EM_cond(model, marginal_prob_std_fn, diffusion_coeff_fn, noise, a, num_steps=2000, eps=eps, T=T)
#
#     return sample
#
# def gen_sample_cond_t(model, a, Nx, bs, dim, marginal_prob_std_fn, diffusion_coeff_fn, method, latent, t, eps = 1e-5):
#     #print('ss', a.shape, noise.shape)
#     if method == 'ode_solver_cond':
#         sample = ode_solver_cond(model, marginal_prob_std_fn, diffusion_coeff_fn, latent, a, forward=2, eps=eps, T=t)
#     elif method == 'pc_sampler_cond':
#         sample = pc_sampler_cond(model, marginal_prob_std_fn, diffusion_coeff_fn, latent, a, num_steps=2000, eps=eps, T=t)
#     elif method == 'EM_cond':
#         sample = EM_cond(model, marginal_prob_std_fn, diffusion_coeff_fn, latent, a, num_steps=2000, eps=eps, T=t)
#
#     return sample

