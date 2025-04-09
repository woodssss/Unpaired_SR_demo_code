import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import argparse
from utils.train_ve_utils import *
from Net import Unet_cond

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Super resolution')
parser.add_argument('-type', '--type', type=str, metavar='', help='type of problem')
parser.add_argument('-flag', '--flag', type=int, metavar='', help='flag of sup')
args = parser.parse_args()

if __name__ == "__main__":
    if args.type:
        print('User defined problem')
        type = args.type
        flag = args.flag
    else:
        print('Not define problem type, use default poentential vorticity generate skip point at 32x32')
        type = 'ns'
        flag = 0

    print(type, ' Problem with flag ', flag)

    if type == 'wave':
        from config.config_wave import *
    if type == 'euler':
        from config.config_euler import *
    if type == 'ns':
        from config.config_ns import *


    with open(Super_data, 'rb') as ss:

        up1_0 = np.load(ss)
        up1_1 = np.load(ss)
        up1_2 = np.load(ss)

        ref = np.load(ss)


    if flag == 0:
        up_0, tup_0 = up1_0[:N_train_sup, ...], up1_0[N_train_sup:, ...]
        from config.model_config_sup_0 import *

        mat_up = up_0
        tmat_up = tup_0
        train = torch.from_numpy(mat_up).float()
        bs = 10
        test_cond = tmat_up[..., [1]]
        test_cond = torch.from_numpy(test_cond).float()
        test_real = tmat_up[..., [0]]
        N_c = int(Nx / 4)
        xx_c, yy_c = xx_1, yy_1
        xx_f, yy_f = xx_2, yy_2
        save_epoch = training_config['log_freq']
        batch_size = training_config['batch_size']
        n_epochs = training_config['epochs']
        lr = training_config['lr']
    if flag == 1:
        up_1, tup_1 = up1_1[:N_train_sup, ...], up1_1[N_train_sup:, ...]
        from config.model_config_sup_1 import *

        mat_up = up_1
        tmat_up = tup_1
        train = torch.from_numpy(mat_up).float()
        bs = 10
        test_cond = tmat_up[..., [1]]
        test_cond = torch.from_numpy(test_cond).float()
        test_real = tmat_up[..., [0]]
        N_c = int(Nx / 2)
        xx_c, yy_c = xx_2, yy_2
        xx_f, yy_f = xx, yy
        save_epoch = training_config['log_freq']
        batch_size = training_config['batch_size']
        n_epochs = training_config['epochs']
        lr = training_config['lr']
    elif flag == 2:
        up_2, tup_2 = up1_2[:N_train_sup, ...], up1_2[N_train_sup:, ...]
        from config.model_config_sup_2 import *

        mat_up = up_2
        tmat_up = tup_2
        train = torch.from_numpy(mat_up).float()
        bs = 10
        test_cond = tmat_up[..., [1]]
        test_cond = torch.from_numpy(test_cond).float()
        test_real = tmat_up[..., [0]]
        N_c = int(Nx)
        xx_c, yy_c = xx, yy
        xx_f, yy_f = xx, yy
        save_epoch = training_config['log_freq']
        batch_size = training_config['batch_size']
        n_epochs = training_config['epochs']
        lr = training_config['lr']

    model = Unet_cond(**model_config).to(device)

    save_name = 'Sup_' + type + '_flag_'  + str(flag)

    dataset = TensorDataset(train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cwd = os.getcwd()

    log_name = cwd + '/logs/' + save_name + '_log.txt'
    chkpts_base_name = cwd + '/mdls/' + save_name

    content = log_name
    mylogger(log_name, content)
    print(content)

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.95)

    for epoch in tqdm(range(n_epochs)):
        avg_loss = 0.
        num_items = 0
        for x in data_loader:
            a, x = x[0][..., [1]].to(device), x[0][..., [0]].to(device)

            loss = my_loss_func(model, x, a, marginal_prob_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        if epoch % save_epoch == 0 and epoch>0:
        #if epoch % 2 == 0 and epoch > 0:
            # Print the averaged training loss so far.
            content = 'at epoch: %d, Average Loss: %3f' % (
                epoch, avg_loss / num_items)
            mylogger(log_name, content)
            print(content)

            # ### test sampling
            # bs = 10
            # cond = test_cond[:bs, ...].to(device)
            #
            # noise = torch.randn(bs, N_c, N_c, 1).to(device)
            # t = torch.ones(bs, device=device)
            # init_x = noise * marginal_prob_fn(noise, t)[1][:, None, None, None]
            #
            # sample_1 = ode_sampler(model, marginal_prob_fn, get_sde_forward_fn, init_x, cond, eps=1e-5)
            # sample_1 = sample_1.detach().cpu().numpy()
            #
            # sample_real = test_real[:bs, ...]
            #
            # ### compare coefficent
            # error1 = get_relative_l2_error(sample_1, sample_real)
            #
            # content = 'coeff error at epoch: %d, error ode is: %3f' % (
            #     epoch, error1)
            # mylogger(log_name, content)
            # print(content)

            # ######### plot #########################
            # bs = 4
            # nrows = 3
            # fig1, ax = plt.subplots(nrows, bs, figsize=(bs * 2, nrows * 2))
            # for i in range(bs):
            #     ax[0, i].contourf(xx_c, yy_c, cond[i, ..., 0].detach().cpu().numpy(), 36, cmap=cm.jet)
            #     ax[1, i].contourf(xx_c, yy_c, sample_1[i, ..., 0], 36, cmap=cm.jet)
            #     ax[2, i].contourf(xx_c, yy_c, sample_real[i, ..., 0], 36, cmap=cm.jet)
            # plt.show()
            # ###########################################################################
            if epoch>0:
                chkpts_model_name = chkpts_base_name + '_epoch_' + str(epoch) + '_ckpt.pt'
                torch.save(model, chkpts_model_name)