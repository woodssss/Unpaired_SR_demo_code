import functools
import torch
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import ott
from ott.tools import plot, sinkhorn_divergence
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import pickle



import os
from torch.utils.data import DataLoader, TensorDataset
import argparse
from utils.NOT_utils import *
from config.model_config_NOT import *

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
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

    with open(Gen_data, 'rb') as ss:
        c = np.load(ss)
        cv = np.load(ss)


    c = c.reshape(c.shape[0], -1)
    cv = cv.reshape(cv.shape[0], -1)

    print(c.shape, cv.shape)


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
    geom = pointcloud.PointCloud(c,
                                 cv,
                                 epsilon=0.001)

    # Computing the potentials.
    out = sinkhorn.Sinkhorn(max_iterations=1000,
                            momentum=momentum,
                            parallel_dual_updates=True)(
        linear_problem.LinearProblem(geom))
    dual_potentials = out.to_dual_potentials()


    save_name = 'GOT_' + type
    chkpts_base_name = cwd + '/mdls/' + save_name + '.pkl'


    # to_save = jax.tree_util.tree_map(lambda x: jnp.array(x), dual_potentials)
    #
    # with open(chkpts_base_name, 'wb') as f:
    #     pickle.dump(to_save, f)

    with open(chkpts_base_name, 'wb') as f:
        pickle.dump(out, f)