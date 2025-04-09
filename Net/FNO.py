import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

### FNO ###
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input.to(torch.float32), weights.to(torch.float32))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FourierLayer2D(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FourierLayer2D, self).__init__()
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv1d(width, width, 1)
        self.width = width

    def forward(self, x):
        # x shape [bs, width, Nx, Nx]
        bs, N = x.shape[0], x.shape[-1]
        y1 = self.conv0(x)
        y2 = self.w0(x.view(bs, self.width, -1)).view(bs, self.width, N, N)
        y = y1 + y2
        y = F.gelu(y)
        out = y
        return out

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, nl, width):
        super(FNO2d, self).__init__()
        self.fc0 = nn.Linear(2, width)

        self.layers_ls = nn.ModuleList()

        for i in range(nl):
            self.layers_ls.append(FourierLayer2D(modes1, modes2, width))

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x shape [bs, 1, Nx, Nx]
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for layer in self.layers_ls:
            x = layer(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x

class FNO2d_evo(nn.Module):
    def __init__(self, in_dim, out_dim, modes1, modes2, nl, nr):
        super(FNO2d_evo, self).__init__()
        self.fc0 = nn.Linear(in_dim, nr)

        self.layers_ls = nn.ModuleList()

        for i in range(nl):
            self.layers_ls.append(FourierLayer2D(modes1, modes2, nr))

        self.fc1 = nn.Linear(nr, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x):
        # x shape [bs, t, Nx, Nx]
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        for layer in self.layers_ls:
            x = layer(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x
