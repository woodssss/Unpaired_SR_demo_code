import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import integrate
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")
def mylogger(filename, content):
    with open(filename, 'a') as fw:
        print(content, file=fw)

def get_sde_forward(x, t, beta_0, beta_T):
    # out put mean and difssuion coeff
    beta_t = beta_0 + t * (beta_T - beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    discount = 1.0 - torch.exp(-2.0 * beta_0 * t - (beta_T - beta_0) * t ** 2)
    diffusion = torch.sqrt(beta_t * discount)
    return drift, diffusion


def marginal_prob(x, t, beta_0, beta_T, device=device):
    t = torch.tensor(t, device=device)
    #t = t.to(device)
    log_mean_coeff = (-0.25 * t ** 2 * (beta_T - beta_0)
                      - 0.5 * t * beta_0)
    mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
    std = 1.0 - torch.exp(2.0 * log_mean_coeff)
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


def ode_solver(score_model,
                marginal_prob,
                get_sde_forward,
                init_x,
                forward,
                atol=1e-6,
                rtol=1e-6,
                device=device,
                eps=1e-3,
                T=1):
    shape = init_x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper of the score-based model for use by the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def get_sde_forward_eval_wrapper(sample, time_steps):
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            drfit, g = get_sde_forward(sample, time_steps)
        return drfit.cpu().numpy().reshape((-1,)).astype(np.float64), g.cpu().numpy().reshape((-1,)).astype(np.float64)

    def marginal_prob_eval_wrapper(sample, time_steps):
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            mean, std = marginal_prob(sample, time_steps)
        return mean.cpu().numpy().reshape((-1,)).astype(np.float64), std.cpu().numpy()

    def ode_func(t, x):
        """The ODE function for use by the ODE solver."""
        time_steps = np.ones((shape[0],)) * t

        drift, g =get_sde_forward_eval_wrapper(x, time_steps)
        mean, std = marginal_prob_eval_wrapper(x, time_steps)

        return drift - 0.5 * (g[0] ** 2) / std[0] * score_eval_wrapper(x, time_steps)

    # Run the black-box ODE solver.
    if forward ==1:
        # forward
        res = integrate.solve_ivp(ode_func, (eps, T), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    else:
        res = integrate.solve_ivp(ode_func, (T, eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
    print(f"Number of function evaluations: {res.nfev}")
    x = torch.tensor(res.y[:, -1], device=device, dtype=torch.float32).reshape(shape)

    return x