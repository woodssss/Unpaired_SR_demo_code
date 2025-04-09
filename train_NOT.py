import functools
import torch
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
import argparse
from utils.NOT_utils import *
from config.model_config_NOT import *

device = torch.device("cuda" if torch.cuda.is_available() else
                      "cpu")

parser = argparse.ArgumentParser(description='Generate')
parser.add_argument('-type', '--type', type=str, metavar='', help='type')
args = parser.parse_args()

if __name__ == "__main__":

    if args.type:
        print('User defined problem')
        type = args.type
    else:
        print('Not define problem type, use default generate coarse at 32x32')
        type = 'ns'

    if type == 'wave':
        from config.config_wave import *
    if type == 'euler':
        from config.config_euler import *
    if type == 'ns':
        from config.config_ns import *

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "cpu")

    with open(Gen_data, 'rb') as ss:
        c = np.load(ss)
        cv = np.load(ss)

    mat_1, mat_2 = c[..., 0][:, None, ...], cv[..., 0][:, None, ...]


    save_name = 'NOT_'+type

    chkpts_base_name = cwd + '/mdls/' + save_name
    log_name = cwd + '/logs/' + save_name + '_log.txt'

    data_tensor_1 = torch.tensor(mat_1, dtype=torch.float32)
    data_tensor_2 = torch.tensor(mat_2, dtype=torch.float32)

    dataset_1 = TensorDataset(data_tensor_1)
    dataset_2 = TensorDataset(data_tensor_2)

    data_loader_1 = DataLoader(dataset_1, batch_size=BATCH_SIZE, shuffle=True)
    data_loader_2 = DataLoader(dataset_2, batch_size=BATCH_SIZE, shuffle=True)

    data_1_sampler = DataLoaderSampler(data_loader_1)
    data_2_sampler = DataLoaderSampler(data_loader_2)

    # Load test datasets
    cwd = os.getcwd()

    with open(Test_data, 'rb') as ss:
        c = np.load(ss)
        cv = np.load(ss)

    mat_1, mat_2 = c[..., 0][:, None, ...], cv[..., 0][:, None, ...]
    mat_c = c[..., 0][:, None, ...]

    # print(cv_skip.shape, coarse_0.shape)
    # zxc

    X_test = torch.tensor(mat_1, dtype=torch.float32)
    Y_test = torch.tensor(mat_2, dtype=torch.float32)
    Z_test = torch.randn(mat_1.shape[0], z_size, 1, 32, 32, dtype=torch.float32)

    C_test = torch.tensor(mat_c, dtype=torch.float32)

    with torch.no_grad():
        XZ_test = torch.cat([X_test[:, None].repeat(1, z_size, 1, 1, 1), Z_test], dim=2).to(device)

    ########################################################################################

    ### define models
    T = model_T.to(device)
    f = model_f.to(device)

    T_optimizer = torch.optim.Adam(T.parameters(), lr=1e-4)
    T_scheduler = torch.optim.lr_scheduler.StepLR(T_optimizer, step_size=1000, gamma=0.95)

    f_optimizer = torch.optim.Adam(f.parameters(), lr=1e-4)
    f_scheduler = torch.optim.lr_scheduler.StepLR(f_optimizer, step_size=1000, gamma=0.95)

    ########## start training #################
    COST = weak_kernel_cost
    for step in tqdm(range(max_iters)):
        T.train(True)
        f.eval()
        for t_iter in range(T_iters):
            # Sample X, Z
            X = data_1_sampler.get_sample().to(device)  # (bs, 3, 16, 16)
            Z = torch.randn(BATCH_SIZE, z_size, 1, 32, 32, device=device)  # (bs, z_size, 1, 16, 16)

            XZ = torch.cat([X[:, None].repeat(1, z_size, 1, 1, 1), Z], dim=2).to(device)  # (bs, z_size, 1+1, 16, 16)

            T_XZ = restore_z(T(hide_z(XZ)), BATCH_SIZE)  # (bs, z_size, 1, 16, 16)

            # Compute the loss for T
            T_loss = (COST(X, T_XZ, gamma).mean() - f(hide_z(T_XZ)).mean()).to(device)
            T_optimizer.zero_grad()
            T_loss.backward()
            T_optimizer.step()
            T_scheduler.step()

        # f optimization
        T.eval()
        f.train(True)
        # Sample X, Y, Z
        X = data_1_sampler.get_sample().to(device)  # (bs, 3, 16, 16)
        Z = torch.randn(BATCH_SIZE, z_size, 1, 32, 32, device=device)  # (bs, z_size, 1, 16, 16)
        Y = data_2_sampler.get_sample().to(device)  # (bs, 3, 16, 16)

        # Get T_XZ
        XZ = torch.cat([X[:, None].repeat(1, z_size, 1, 1, 1), Z], dim=2)  # (bs, z_size, 3+1, 16, 16)
        T_XZ = restore_z(T(hide_z(XZ)), BATCH_SIZE)  # (bs, z_size, 3, 16, 16)

        # Compute the loss for f
        f_loss = - f(Y).mean() + f(hide_z(T_XZ)).mean()
        f_optimizer.zero_grad()
        f_loss.backward()
        f_optimizer.step()
        f_scheduler.step()

        if step % save_iters == 0 and step > 0:
            print("Step", step)
            print("gamma", gamma)

            # The code for plotting the results
            with torch.no_grad():
                T_XZ_test = T(XZ_test.flatten(start_dim=0, end_dim=1)
                              ).permute(1, 2, 3, 0).reshape(1, 32, 32, XZ_test.shape[0], z_size).permute(3, 4, 0, 1,
                                                                                                         2).to('cpu')

            # print(X_test.shape, T_XZ_test.shape, Y_test.shape)
            # zxc

            error0 = torch.linalg.norm(C_test - Y_test) / torch.linalg.norm(Y_test)
            error1 = torch.linalg.norm(T_XZ_test[:, 0, ...] - Y_test) / torch.linalg.norm(Y_test)
            error2 = torch.linalg.norm(T_XZ_test[:, 1, ...] - Y_test) / torch.linalg.norm(Y_test)

            content = 'at step: %d, coarse error: %3f, first item error: %3f, second item error: %3f' % (
                step, error0, error1, error2)
            mylogger(log_name, content)
            print(content)

            plot_images_batch(X_test.detach().cpu().numpy(), C_test.detach().cpu().numpy(),
                              Y_test.detach().cpu().numpy(), T_XZ_test.detach().cpu().numpy(), z_size, 5, xx_0, yy_0)
            plt.show()

            chkpts_name = chkpts_base_name + '_iters_' + str(step) + '_ckpt.pt'
            torch.save(T, chkpts_name)