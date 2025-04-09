import functools
import torch
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset
import argparse
from Net import Unet

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")
parser = argparse.ArgumentParser(description='Generate')
parser.add_argument('-sde', '--sde', type=str, metavar='', help='sde')
parser.add_argument('-type', '--type', type=str, metavar='', help='type')
parser.add_argument('-flag', '--flag', type=int, metavar='', help='flag')
args = parser.parse_args()

if __name__ == "__main__":

    if args.sde:
        print('User defined problem')
        flag = args.flag
        sde = args.sde
        type = args.type
    else:
        print('Not define problem type, use default generate coarse at 32x32')
        flag = 1
        sde = 'vp'
        type = 'ns'

    if sde == 've':
        from config.model_config_vesde import *
    else:
        from config.model_config_vpsde import *

    if type == 'wave':
        from config.config_wave import *
    if type == 'euler':
        from config.config_euler import *
    if type == 'ns':
        from config.config_ns import *

    with open(Gen_data, 'rb') as ss:
        c = np.load(ss)
        cv1 = np.load(ss)

    model = Unet(**model_config).to(device)

    if flag == 0:
        print('Gen coarse')
        train = torch.from_numpy(c).float()
    elif flag == 1:
        print('Gen cv1')
        train = torch.from_numpy(cv1).float()

    save_name = 'Gen_'+ type + '_' + sde + '_flag_'  + str(flag)

    ### define training detail
    log_freq = training_config['log_freq']
    batch_size = training_config['batch_size']
    n_epochs = training_config['epochs']

    my_loss_func = loss_score_t

    dataset = TensorDataset(train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_params = sum(p.numel() for p in model.parameters())

    cwd = os.getcwd()

    log_name = cwd + '/logs/' + save_name + '_log.txt'
    chkpts_base_name = cwd + '/mdls/' + save_name

    optimizer = Adam(model.parameters(), lr=training_config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.98)


    for epoch in tqdm(range(n_epochs)):
        avg_loss = 0.
        num_items = 0
        model.train()
        for x in data_loader:
            x = x[0][..., [0]].to(device)
            loss = my_loss_func(model, x, marginal_prob_fn)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        if epoch % log_freq == 0 and epoch > 0:
            content = 'at epoch: %d, Average Loss: %3f' % (
                epoch, avg_loss / num_items)
            mylogger(log_name, content)
            print(content)

            ### test sampling ###########################
            bs = 8
            mat = train[:bs, ...]
            noise = torch.randn(bs, Nx_c, Nx_c, 1).to(device)
            t = torch.ones(bs, device=device)

            model.eval()
            if sde == 've':
                noise *= marginal_prob_fn(noise, t)[1][:, None, None, None]
            sample_f = ode_solver(model, marginal_prob_fn, get_sde_forward_fn, noise, forward=2, eps=1e-5)

            fig1, ax = plt.subplots(2, bs, figsize=(bs * 2, 4))
            for i in range(bs):
                ax[0, i].contourf(xx_0, yy_0, sample_f[i, ..., 0].detach().cpu().numpy(), 36, cmap=cm.jet)
                ax[1, i].contourf(xx_0, yy_0, train[i, ..., 0].detach().cpu().numpy(), 36, cmap=cm.jet)
            plt.show()

            ###########################
            chkpts_model_name = chkpts_base_name + '_epoch_' + str(epoch) + '_ckpt.pt'
            torch.save(model, chkpts_model_name)