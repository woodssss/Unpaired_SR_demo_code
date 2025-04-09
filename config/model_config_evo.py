import torch
import torch.nn as nn
from Net.FNO import FNO2d_evo
#from config.config_evo_wave import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(25)

T = 5
nl = 4
model_evo = FNO2d_evo(1, T, 12, 12, nl, 64).to(device)

training_config = {
        'lr': 1e-4,
        'batch_size': 25,
        'epochs': 101,
        'log_freq': 100,
    }
