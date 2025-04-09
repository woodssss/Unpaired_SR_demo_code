import numpy as np
import torch
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else
                      "cpu")

def mylogger(filename, content):
    with open(filename, 'a') as fw:
        print(content, file=fw)

def weak_kernel_cost(X, T_XZ, gamma):
    """
    Input
    --------
    X : tensor, shape (bs, dim) or (bs, n_ch, w, h)
    T_XZ : tensor, shape (bs, z_size, dim) or (bs, z_size, n_ch, w, h)
    gamma : float

    Output
    --------
    cost : tensor, shape ()
    """
    X = X.flatten(start_dim=1)
    T_XZ = T_XZ.flatten(start_dim=2)
    z_size = T_XZ.size(1)

    l2_dist = (X[:, None] - T_XZ).norm(dim=2).mean()
    kvar = .5 * torch.cdist(T_XZ, T_XZ).mean() * z_size / (z_size - 1)
    return l2_dist - 0.5 * gamma * kvar

def hide_z(batch):
    "Converts batch B x Z x C x H x W -> BZ x C x H x W"
    return batch.reshape(batch.shape[0]*batch.shape[1], *batch.shape[2:])

def restore_z(batch, batch_size):
    "Converts batch BZ x C x H x W -> B x Z x C x H x W"
    return batch.reshape(batch_size, -1, *batch.shape[1:])

def get_sample(iteridx, data_loader):
    try:
        return next(iteridx)[0]
    except StopIteration:
        iteridx = iter(data_loader)
        return next(iteridx)[0]


class DataLoaderSampler:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iteridx = iter(data_loader)

    def get_sample(self):
        try:
            return next(self.iteridx)[0]
        except StopIteration:
            self.iteridx = iter(self.data_loader)
            return next(self.iteridx)[0]

def plot_images_batch(X, C, Y, mat, nrows, bs, xx, yy):
    fig, ax = plt.subplots(nrows + 3, bs, figsize=(bs * 2, (nrows+2)*2))

    for i in range(bs):
        ax[0, i].contourf(xx, yy, X[i, 0, ...], 36, cmap='jet')

    for i in range(bs):
        ax[1, i].contourf(xx, yy, C[i, 0, ...], 36, cmap='jet')

    for j in range(2, nrows+2):
        for i in range(bs):
            ax[j, i].contourf(xx, yy, mat[i, j-2, 0, ...], 36, cmap='jet')

    for i in range(bs):
        ax[-1, i].contourf(xx, yy, Y[i, 0, ...], 36, cmap='jet')

    fig.tight_layout(pad=0.1)