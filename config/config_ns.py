import numpy as np
import os
from utils.utils import *
from utils.data_utils import *
import functools


### modify dataset
L = 1
### gird
Nx = 256
m = 8
Nx_c = int(Nx/m)

points_x = get_grid(Nx, L)
points_x_0 = get_grid(int(Nx/m), L)
points_x_1 = get_grid(int(Nx/4), L)
points_x_2 = get_grid(int(Nx/2), L)

xx_0, yy_0 = np.meshgrid(points_x_0, points_x_0)
xx_1, yy_1 = np.meshgrid(points_x_1, points_x_1)
xx_2, yy_2 = np.meshgrid(points_x_2, points_x_2)
xx, yy = np.meshgrid(points_x, points_x)
##### number of training ############
## two dataset: set 1: coarse and their correct version for gen and test
## set 2: gen and sup

N_gen = 5000
N_sup = 5000

# N_train = 9000
# N_sup = 2000
N_train = N_gen

N_train_sup = int(N_sup * 0.95)
N_test_sup = int(N_sup * 0.05)

N_test = 100

cwd=os.getcwd()

Ori_data_high = cwd + '/raw_data/NS_data_high.npy'
Ori_data_low = cwd + '/raw_data/NS_data_low.npy'
Ori_data_test = cwd + '/raw_data/NS_data_test.npy'

#Gen_data_ak = cwd + '/data/Wave_gen_data_all_kind_N_' + num2str_deciaml(N_gen) + '.npy'
Gen_data = cwd + '/data/NS_gen_data_N_' + num2str_deciaml(N_gen) + '.npy'
Super_data = cwd + '/data/NS_super_data_N_' + num2str_deciaml(N_sup) + '.npy'
Test_data = cwd + '/data/NS_test_data_N_' + num2str_deciaml(N_test) + '.npy'