from utils.train_ve_utils import *
import functools

######## training strategy for score model #################
### define SDE coefficients for VP SDE ##########################
model_config = {
    "ch": 64,
    "ch_mul": [1, 2, 2, 2],
    "att_channels": [0, 1, 0, 0],
    "groups": 8,
    "dropout": 0.1,
    "scale_shift": True,
}

training_config = {
        'sigma': 25,
        'timesteps': 400,
        'lr': 2e-4,
        'warmup': 50,
        'batch_size': 64,
        'epochs': 501,
        'log_freq': 100,
        'num_workers': 2,
        'use_ema': True
    }


sigma = training_config['sigma']
marginal_prob_fn = functools.partial(marginal_prob, sigma=sigma)
get_sde_forward_fn  = functools.partial(get_sde_forward, sigma=sigma)
my_loss_func = loss_score_t
sampler = ode_solver

# save_name_c_gen = 'Gen_c_vp'
# save_name_cv1_gen = 'Gen_cv1_vp'
# save_name_cv2_gen = 'Gen_cv2_vp'
####################################################