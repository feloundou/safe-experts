import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import gym
import safety_gym
import time
import random
import os.path as osp
from torch.optim import Adam

from torch.distributions.categorical import Categorical

from neural_nets import ActorCritic, ValorDiscriminator, VDB

import wandb
import wandb.plot as wplot

from utils import PureVALORBuffer
from utils import mpi_fork, proc_id, num_procs, EpochLogger, \
    setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars


def blind_valor(env_fn,
                disc=ValorDiscriminator,

                dc_kwargs=dict(), seed=0,
                episodes_per_epoch=40,
                epochs=50,
                dc_lr=5e-4,
                train_dc_iters=10,
                train_dc_interv=1,
                max_ep_len=20, logger_kwargs=dict(),
                config_name='standard',
                splitN=8,
                save_freq=10, replay_buffers=[]):
    # W&B Logging
    wandb.login()

    composite_name = 'new_valor_' + config_name
    wandb.init(project="LearningCurves", group="Blind VALOR Expert", name=composite_name)

    assert replay_buffers != [], "Replay buffers must be set"

    # Note: hooking the context filter to the generator is key
    # Self-organizing maps might be useful here.

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create discriminator and monitor it
    # max_con_dim = len(replay_buffers)
    max_con_dim = 10  # init context dimension.
    # the discriminator should not know exactly how many contexts there are
    discrim = disc(input_dim=obs_dim[0], context_dim=max_con_dim, **dc_kwargs)

    # reward_discrim = label_disc(input_dim=obs_dim[0], context_dim=max_con_dim, **dc_kwargs)

    reward_criterion = criterion = nn.MSELoss()

    # Note that later, having a decoder might help

    # Set up model saving
    logger.setup_pytorch_saver([discrim])

    # Sync params across processes
    sync_params(discrim)

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buffer = PureVALORBuffer(max_con_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv,
                             N=splitN)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [discrim.pi])
    logger.log('\nNumber of parameters: \t d: %d\n' % var_counts)

    # Optimizers
    discrim_optimizer = Adam(discrim.pi.parameters(), lr=dc_lr)

    def update(e):
        obs, act = [torch.Tensor(x) for x in buffer.retrieve_all()]

        # Discriminator
        print('Discriminator Update!')
        con, s_diff = [torch.Tensor(x) for x in buffer.retrieve_dc_buff()]

        _, logp_dc, _ = discrim(s_diff, con)
        d_l_old = -logp_dc.mean()

        discriminator_metrics = {'discriminator loss': d_l_old}

        wandb.log(discriminator_metrics)

        # Discriminator train
        for _ in range(train_dc_iters):
            _, logp_dc, _ = discrim(s_diff, con)
            d_loss = -logp_dc.mean()  # Tyna remove the mean and give per time step reward
            discrim_optimizer.zero_grad()
            d_loss.backward()
            mpi_avg_grads(discrim.pi)
            discrim_optimizer.step()

        label, loggt_dc, logp_dc, gt = discrim(s_diff, con, classes=True)

        print("LABELS: ", label)
        print("GROUND TRUTH: ", gt)

        dc_l_new = -loggt_dc.mean()

        logger.store(LossDC=d_l_old,
                     DeltaLossDC=(dc_l_new - d_l_old))

    start_time = time.time()
    # context_dist = Categorical(logits=torch.Tensor(np.ones(con_dim)))
    total_t, ep_len = 0, 0

    for epoch in range(epochs):
        discrim.eval()
        faux_con_dim = random.randrange(1, max_con_dim)  # Crucial Step , choose from 1 to max_con_dim
        print("local episodes:", local_episodes_per_epoch)
        for ep in range(local_episodes_per_epoch):

            # Now how do we get data? Here, the discriminator must be able to see data from different
            # modes all at once.

            # t = random.randrange(0, faux_con_dim) # want to randomize draws for now
            # c = torch.tensor(t)
            # print("context sample: ", c)
            # c_onehot = F.one_hot(c, faux_con_dim).squeeze().float()

            # print("Guessing the number of dimensions: ", faux_con_dim)

            # for _ in range(10):
            # for _ in range(max_ep_len):
            for _ in range(max_ep_len):
                # draw sample from replay buffer (try just one at a time for now)
                # TODO: Review sampling method

                j = random.randrange(0, faux_con_dim)
                c = torch.tensor(j)
                c_onehot = F.one_hot(c, max_con_dim).squeeze().float()

                # j can be much larger than the number of experts available (in this context)
                # so reduce the dimension of j by some method
                k = j % len(replay_buffers)
                # print("j is ", j)
                # print("k is", k)
                # print("----------")

                sample_rb = replay_buffers[k].sample(1)

                o = sample_rb['obs']
                a = sample_rb['act']

                concat_obs = torch.cat([torch.Tensor(o.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)

                ep_len += 1
                buffer.store(c, concat_obs.squeeze(), a)
                # instead of doing average over sequence dimension,
                # do not need to average reward, just give reward now
                terminal = (ep_len == max_ep_len)
                if terminal:  # This does not make sense in this context
                    dc_diff = torch.Tensor(buffer.calc_diff()).unsqueeze(0)
                    con = torch.Tensor([float(c)]).unsqueeze(0)
                    _, loggt, _ = discrim(dc_diff, con)
                    #
                    buffer.finish_path(loggt.detach().numpy())
                    ep_len = 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [discrim], None)

        def con_dim_update():
            pass

        # Update
        discrim.train()

        # update models
        update(epoch)

        # Log
        logger.log_tabular('Epoch', epoch)
        # logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('LossDC', average_only=True)
        logger.log_tabular('DeltaLossDC', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

    # After training, evaluate the final discriminator

    wandb.finish()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--episodes-per-epoch', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='valor-anonymous-expert')
    parser.add_argument('--con', type=int, default=10)
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    blind_valor(lambda: gym.make(args.env),
                disc=ValorDiscriminator, dc_kwargs=dict(hidden_dims=[args.hid] * args.l),
                seed=args.seed, episodes_per_epoch=args.episodes_per_epoch,
                epochs=args.epochs,
                logger_kwargs=logger_kwargs)
