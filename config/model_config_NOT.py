import torch
import torch.nn as nn
from Net.FNO import FNO2d

device = torch.device("cuda" if torch.cuda.is_available() else
                      "cpu")

torch.manual_seed(25)
############## OT models #####################################
nl = 6
model_T = FNO2d(16, 16, nl, 16).to(device)

model_f = nn.Sequential(
    nn.Conv2d(1, 128, kernel_size=5, padding=2),
    nn.ReLU(True),
    nn.AvgPool2d(2), #  128 x 8 x 8
    nn.Conv2d(128, 128, kernel_size=5, padding=2),
    nn.ReLU(True),
    nn.AvgPool2d(2), #  256 x 4 x 4
    nn.Conv2d(128, 128, kernel_size=5, padding=2),
    nn.ReLU(True),
    nn.AvgPool2d(2), #  512 x 2 x 2
    nn.Conv2d(128, 128, kernel_size=3, padding=1),
    nn.ReLU(True),
    nn.AvgPool2d(2), #  512 x 1 x 1
    nn.Conv2d(128, 1, kernel_size=1, padding=0),
    nn.Flatten(1),
).to(device)


BATCH_SIZE = 25
T_iters = 10
max_iters = 2000 + 1
z_size = 2
gamma = 0.1
save_iters = 500