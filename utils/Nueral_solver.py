import torch
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
import argparse
from utils.data_utils import *
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import ott
from ott.tools import plot, sinkhorn_divergence
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

def transfer_OT(src_train, target_train, src_eval):
    @jax.jit
    def sinkhorn_loss(x: jax.Array, y: jax.Array, epsilon: float = 0.1) -> jax.Array:
        """Computes transport between (x, a) and (y, b) via Sinkhorn algorithm."""
        # We assume equal weights for all points.
        a = jnp.ones(len(x)) / len(x)
        b = jnp.ones(len(y)) / len(y)

        sdiv = sinkhorn_divergence.sinkhorn_divergence(
            pointcloud.PointCloud, x, y, epsilon=epsilon, a=a, b=b
        )

        return sdiv[0]

    momentum = ott.solvers.linear.acceleration.Momentum(value=.5)

    # Defining the geometry.
    geom = pointcloud.PointCloud(src_train,
                                 target_train,
                                 epsilon=0.001)

    # Computing the potentials.
    out = sinkhorn.Sinkhorn(max_iterations=1000,
                            momentum=momentum,
                            parallel_dual_updates=True)(
        linear_problem.LinearProblem(geom))

    dual_potentials = out.to_dual_potentials()

    target_pred = dual_potentials.transport(src_eval)

    return target_pred


def transfer_NOT(T, X, z_size):
    X = X if torch.is_tensor(X) else torch.from_numpy(X).float().to(device)
    X = X.permute(0, 3, 1, 2)
    Nx_c = X.shape[-1]
    Z = torch.randn(X.shape[0], z_size, 1, Nx_c, Nx_c, dtype=torch.float32).to(device)
    # print(X.shape, Z.shape)
    # zxc
    with torch.no_grad():
        XZ = torch.cat([X[:,None].repeat(1,z_size,1,1,1), Z], dim=2).to(device)

    with torch.no_grad():
        pd = T(XZ.flatten(start_dim=0, end_dim=1)).permute(1, 2, 3, 0).reshape(1, Nx_c, Nx_c, XZ.shape[0], z_size).permute(3, 4, 0, 1, 2).to('cpu')

    return (pd[:, 0, ...].permute(0, 2, 3, 1)).detach().cpu().numpy()

def transfer_DDIB(ode_solver, marginal_prob_fn, get_sde_forward_fn, model_coarse_gen, model_down_gen, X, eps, t1, t2):
    #ic = torch.from_numpy(X).float().to(device)
    ic = X if torch.is_tensor(X) else torch.from_numpy(X).float().to(device)

    latent = ode_solver(model_coarse_gen, marginal_prob_fn, get_sde_forward_fn, ic, forward=1,
                        eps=eps, T=t1)

    sample_down = ode_solver(model_down_gen, marginal_prob_fn, get_sde_forward_fn, latent, forward=2, eps=eps, T=t2)

    return sample_down.detach().cpu().numpy()


def transfer_sdit(get_perturbed_x, ode_solver, marginal_prob_fn, get_sde_forward_fn, model_down_gen, X, eps, t1, t2):
    ic = X if torch.is_tensor(X) else torch.from_numpy(X).float().to(device)

    latent = get_perturbed_x(ic, marginal_prob_fn, t=t1)

    sample_down = ode_solver(model_down_gen, marginal_prob_fn, get_sde_forward_fn, latent, forward=2, eps=eps, T=t2)

    return sample_down.detach().cpu().numpy()

def Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_c, point_x, L, Nx, model_up, X, eps, t, method='ode_solver_cond', N=1000):
    ### X [bs, 16, 16, 1]
    ### first interp to [bs, 32, 32, 1] then normal
    bs = X.shape[0]
    X_np = X.detach().cpu().numpy() if torch.is_tensor(X) else X
    X_interp_np = interp_pbc_2d_batch(point_x, point_x_c, L, X_np)
    X_interp = torch.from_numpy(X_interp_np).float().to(device)
    X_sample = gen_sample_cond(model_up, X_interp, Nx, bs, 1, marginal_prob_fn, get_sde_forward_fn, method=method, eps=eps, T=t, N=N)
    return X_sample

####################### 256x256 #####################################################################
# def Sup_up_256(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, point_x_2, point_x, L, model_up_0, model_up_1, model_up_2, X, eps, t):
#     X_64 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, L, 64, model_up_0, X, eps, t)
#     X_128 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_1, point_x_2, L, 128, model_up_1, X_64, eps, t)
#     X_256 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_2, point_x, L, 256, model_up_2, X_128, eps, t)
#     return X_64.detach().cpu().numpy(), X_128.detach().cpu().numpy(), X_256.detach().cpu().numpy()

def direct_super(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, point_x_2, point_x, L, eps, t, X, model_up_0, model_up_1=None, model_up_2=None, method1='ode_solver_cond', method2='ode_solver_cond', method3='ode_solver_cond', eps1=1e-5, eps2=1e-5, eps3=1e-5, N=1000):
    ### given ic, do adapt + fno evo + sup
    out_ls = []
    X_64 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, L, 64, model_up_0, X, eps1, t, method=method1, N = N)
    X_64 = X_64.detach().cpu().numpy()
    tmp_nor = np.max(np.abs(X_64), axis=(1, 2, 3), keepdims=True)
    X_64 /= tmp_nor
    out_ls.append(X_64)

    X_64 = torch.from_numpy(X_64).float().to(device)

    if model_up_1:
        print('Sup 64x64->128x128')
        X_128 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_1, point_x_2, L, 128, model_up_1, X_64,
                           eps2, t, method=method2, N = N)
        X_128 = X_128.detach().cpu().numpy()
        tmp_nor = np.max(np.abs(X_128), axis=(1, 2, 3), keepdims=True)
        X_128 /= tmp_nor
        out_ls.append(X_128)
        X_128 = torch.from_numpy(X_128).float().to(device)

        if model_up_2:
            print('Sup 128x128->256x256')
            X_256 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_2, point_x, L, 256, model_up_2,
                               X_128, eps3, t, method=method3, N = N)
            X_256 = X_256.detach().cpu().numpy()
            tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
            X_256 /= tmp_nor
            out_ls.append(X_256)

    return out_ls



def transfer_sdit_super(get_perturbed_x, ode_solver, marginal_prob_fn_tr, get_sde_forward_fn_tr, gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, point_x_2, point_x, L, eps, t1, t2, t, X, model_down_gen, model_up_0=None, model_up_1=None, model_up_2=None, method1='ode_solver_cond', method2='ode_solver_cond', method3='ode_solver_cond', eps1=1e-5, eps2=1e-5, eps3=1e-5, N = 1000):
    ### given ic, do adapt + fno evo + sup
    out_ls = []

    ### adaption
    Y = transfer_sdit(get_perturbed_x, ode_solver, marginal_prob_fn_tr, get_sde_forward_fn_tr, model_down_gen, X, eps, t1, t2)

    out_ls.append(Y)

    if model_up_0:
        norf = np.max(np.abs(Y), axis=(1, 2, 3), keepdims=True)
        Y_nor = Y / norf
        Y_nor = torch.from_numpy(Y_nor).float().to(device)

        X_64 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, L, 64, model_up_0, Y_nor, eps1,
                          t, method=method1, N = N)
        X_64 = X_64.detach().cpu().numpy()
        tmp_nor = np.max(np.abs(X_64), axis=(1, 2, 3), keepdims=True)
        X_64 /= tmp_nor
        out_ls.append(X_64)

        X_64 = torch.from_numpy(X_64).float().to(device)

        if model_up_1:
            print('Sup 64x64->128x128')
            X_128 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_1, point_x_2, L, 128, model_up_1, X_64,
                               eps2, t, method=method2, N = N)
            X_128 = X_128.detach().cpu().numpy()
            tmp_nor = np.max(np.abs(X_128), axis=(1, 2, 3), keepdims=True)
            X_128 /= tmp_nor
            out_ls.append(X_128)
            X_128 = torch.from_numpy(X_128).float().to(device)

            if model_up_2:
                print('Sup 128x128->256x256')
                X_256 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_2, point_x, L, 256, model_up_2,
                                   X_128, eps3, t, method=method3, N = N)
                X_256 = X_256.detach().cpu().numpy()
                tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
                X_256 /= tmp_nor
                out_ls.append(X_256)

    return out_ls

def transfer_ddib_super(ode_solver, marginal_prob_fn_tr, get_sde_forward_fn_tr, gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, point_x_2, point_x, L, eps, t1, t2, t, X, model_coarse_gen, model_down_gen, model_up_0=None, model_up_1=None, model_up_2=None, method1='ode_solver_cond', method2='ode_solver_cond', method3='ode_solver_cond', eps1=1e-5, eps2=1e-5, eps3=1e-5, N = 1000):
    ### given ic, do adapt + fno evo + sup
    out_ls = []

    ### adaption
    Y = transfer_DDIB(ode_solver, marginal_prob_fn_tr, get_sde_forward_fn_tr, model_coarse_gen, model_down_gen, X, eps, t1, t2)

    out_ls.append(Y)

    if model_up_0:
        norf = np.max(np.abs(Y), axis=(1, 2, 3), keepdims=True)
        Y_nor = Y / norf
        Y_nor = torch.from_numpy(Y_nor).float().to(device)

        X_64 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, L, 64, model_up_0, Y_nor, eps1,
                          t, method=method1, N = N)
        X_64 = X_64.detach().cpu().numpy()
        tmp_nor = np.max(np.abs(X_64), axis=(1, 2, 3), keepdims=True)
        X_64 /= tmp_nor
        out_ls.append(X_64)

        X_64 = torch.from_numpy(X_64).float().to(device)

        if model_up_1:
            print('Sup 64x64->128x128')
            X_128 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_1, point_x_2, L, 128, model_up_1, X_64, eps1,
                          t, method=method1, N = N)
            X_128 = X_128.detach().cpu().numpy()
            tmp_nor = np.max(np.abs(X_128), axis=(1, 2, 3), keepdims=True)
            X_128 /= tmp_nor
            out_ls.append(X_128)
            X_128 = torch.from_numpy(X_128).float().to(device)

            if model_up_2:
                print('Sup 128x128->256x256')
                X_256 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_2, point_x, L, 256, model_up_2,
                                   X_128, eps3, t, method=method3, N = N)
                X_256 = X_256.detach().cpu().numpy()
                tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
                X_256 /= tmp_nor
                out_ls.append(X_256)

    return out_ls

def transfer_not_super(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, point_x_2, point_x, L, eps, t, X, model_T, model_up_0=None, model_up_1=None, model_up_2=None, method1='ode_solver_cond', method2='ode_solver_cond', method3='ode_solver_cond', eps1=1e-5, eps2=1e-5, eps3=1e-5, N = 1000):
    ### given ic, do adapt + fno evo + sup
    out_ls = []

    ### adaption
    Y = transfer_NOT(model_T, X, z_size=1)

    out_ls.append(Y)

    if model_up_0:
        norf = np.max(np.abs(Y), axis=(1, 2, 3), keepdims=True)
        Y_nor = Y / norf
        Y_nor = torch.from_numpy(Y_nor).float().to(device)

        X_64 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, L, 64,
                          model_up_0, Y_nor, eps1,
                          t, method=method1, N = N)
        X_64 = X_64.detach().cpu().numpy()
        tmp_nor = np.max(np.abs(X_64), axis=(1, 2, 3), keepdims=True)
        X_64 /= tmp_nor
        out_ls.append(X_64)

        X_64 = torch.from_numpy(X_64).float().to(device)

        if model_up_1:
            print('Sup 64x64->128x128')
            X_128 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_1, point_x_2, L, 128,
                               model_up_1, X_64, eps1,
                               t, method=method1, N = N)
            X_128 = X_128.detach().cpu().numpy()
            tmp_nor = np.max(np.abs(X_128), axis=(1, 2, 3), keepdims=True)
            X_128 /= tmp_nor
            out_ls.append(X_128)
            X_128 = torch.from_numpy(X_128).float().to(device)

            if model_up_2:
                print('Sup 128x128->256x256')
                X_256 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_2, point_x, L, 256,
                                   model_up_2,
                                   X_128, eps3, t, method=method3, N = N)
                X_256 = X_256.detach().cpu().numpy()
                tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
                X_256 /= tmp_nor
                out_ls.append(X_256)

    return out_ls

#### hhhhhhhhhhh
def transfer_ot_super(src_train, target_train, gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, point_x_2, point_x, L, eps, t, X, model_up_0=None, model_up_1=None, model_up_2=None, method1='ode_solver_cond', method2='ode_solver_cond', method3='ode_solver_cond', eps1=1e-5, eps2=1e-5, eps3=1e-5, N = 1000):
    ### prepare src_eval
    src_eval = X.detach().cpu().numpy().reshape(X.shape[0], -1)

    out_ls = []

    print('start ot')

    ### adaption
    Y = transfer_OT(src_train, target_train, src_eval)
    Y = Y.reshape(Y.shape[0], 32, 32)[..., None].__array__()

    print('finish ot')

    print(Y.dtype)

    out_ls.append(Y)

    if model_up_0:
        norf = np.max(np.abs(Y), axis=(1, 2, 3), keepdims=True)
        Y_nor = Y / norf
        Y_nor = torch.from_numpy(Y_nor).float().to(device)

        X_64 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, L, 64,
                          model_up_0, Y_nor, eps1,
                          t, method=method1, N = N)
        X_64 = X_64.detach().cpu().numpy()
        tmp_nor = np.max(np.abs(X_64), axis=(1, 2, 3), keepdims=True)
        X_64 /= tmp_nor
        out_ls.append(X_64)

        X_64 = torch.from_numpy(X_64).float().to(device)

        if model_up_1:
            print('Sup 64x64->128x128')
            X_128 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_1, point_x_2, L, 128,
                               model_up_1, X_64, eps1,
                               t, method=method1, N = N)
            X_128 = X_128.detach().cpu().numpy()
            tmp_nor = np.max(np.abs(X_128), axis=(1, 2, 3), keepdims=True)
            X_128 /= tmp_nor
            out_ls.append(X_128)
            X_128 = torch.from_numpy(X_128).float().to(device)

            if model_up_2:
                print('Sup 128x128->256x256')
                X_256 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_2, point_x, L, 256,
                                   model_up_2,
                                   X_128, eps3, t, method=method3, N = N)
                X_256 = X_256.detach().cpu().numpy()
                tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
                X_256 /= tmp_nor
                out_ls.append(X_256)

    return out_ls



### direct sup
def transfer_sdit_superd(get_perturbed_x, ode_solver, marginal_prob_fn_tr, get_sde_forward_fn_tr, gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, point_x_2, point_x, L, eps, t1, t2, t, X, model_down_gen, model_up_dr, method1='ode_solver_cond', eps1=1e-5, N = 1000):
    ### given ic, do adapt + fno evo + sup
    out_ls = []

    ### adaption
    Y = transfer_sdit(get_perturbed_x, ode_solver, marginal_prob_fn_tr, get_sde_forward_fn_tr, model_down_gen, X, eps, t1, t2)

    out_ls.append(Y)

    norf = np.max(np.abs(Y), axis=(1, 2, 3), keepdims=True)
    Y_nor = Y / norf
    Y_nor = torch.from_numpy(Y_nor).float().to(device)

    X_256 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x, L, 256, model_up_dr,
                      Y_nor, eps1, t, method=method1, N=N)

    X_256 = X_256.detach().cpu().numpy()
    tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
    X_256 /= tmp_nor
    out_ls.append(X_256)

    return out_ls

def transfer_ddib_superd(ode_solver, marginal_prob_fn_tr, get_sde_forward_fn_tr, gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, point_x_2, point_x, L, eps, t1, t2, t, X, model_coarse_gen, model_down_gen, model_up_dr, method1='ode_solver_cond', eps1=1e-5, N = 1000):
    ### given ic, do adapt + fno evo + sup
    out_ls = []

    ### adaption
    Y = transfer_DDIB(ode_solver, marginal_prob_fn_tr, get_sde_forward_fn_tr, model_coarse_gen, model_down_gen, X, eps, t1, t2)

    out_ls.append(Y)

    norf = np.max(np.abs(Y), axis=(1, 2, 3), keepdims=True)
    Y_nor = Y / norf
    Y_nor = torch.from_numpy(Y_nor).float().to(device)

    X_256 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x, L, 256, model_up_dr,
                       Y_nor, eps1, t, method=method1, N=N)

    X_256 = X_256.detach().cpu().numpy()
    tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
    X_256 /= tmp_nor
    out_ls.append(X_256)

    return out_ls

def transfer_not_superd(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, point_x_2, point_x, L, eps, t, X, model_T, model_up_dr, method1='ode_solver_cond', eps1=1e-5, N = 1000):
    ### given ic, do adapt + fno evo + sup
    out_ls = []

    ### adaption
    Y = transfer_NOT(model_T, X, z_size=1)

    out_ls.append(Y)

    norf = np.max(np.abs(Y), axis=(1, 2, 3), keepdims=True)
    Y_nor = Y / norf
    Y_nor = torch.from_numpy(Y_nor).float().to(device)

    X_256 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x, L, 256, model_up_dr,
                       Y_nor, eps1, t, method=method1, N=N)

    X_256 = X_256.detach().cpu().numpy()
    tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
    X_256 /= tmp_nor
    out_ls.append(X_256)

    return out_ls

def direct_superd(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x_1, point_x_2, point_x, L, eps, t, X, model_up_dr, method1='ode_solver_cond', eps1=1e-5, N=1000):
    ### given ic, do adapt + fno evo + sup
    out_ls = []

    X_256 = Sup_up_one(gen_sample_cond, marginal_prob_fn, get_sde_forward_fn, point_x_0, point_x, L, 256, model_up_dr,
                       X, eps1, t, method=method1, N=N)

    X_256 = X_256.detach().cpu().numpy()
    tmp_nor = np.max(np.abs(X_256), axis=(1, 2, 3), keepdims=True)
    X_256 /= tmp_nor
    out_ls.append(X_256)

    return out_ls



