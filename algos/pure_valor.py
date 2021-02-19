# Main entrance of GAIL
import numpy as np
import torch
import torch.nn.functional as F
import gym
import safety_gym
import time
import random
import os.path as osp
from torch.optim import Adam

from torch.distributions.categorical import Categorical

from neural_nets import ActorCritic, ValorDiscriminator

import wandb

from utils import PureVALORBuffer
from utils import mpi_fork, proc_id, num_procs, EpochLogger,\
     setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars

def pure_valor(env_fn,
          disc=ValorDiscriminator, dc_kwargs=dict(), seed=0,
          episodes_per_epoch=40,
          epochs=50,
          dc_lr=5e-4,
          train_dc_iters=10,
          train_dc_interv=1,
          max_ep_len=20, logger_kwargs=dict(),
          config_name='standard',
          save_freq=10, replay_buffers=[]):
    # W&B Logging
    wandb.login()

    composite_name = 'new_valor_penalized_' + config_name
    wandb.init(project="LearningCurves", group="Pure VALOR Expert", name=composite_name)

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

    # Model    # Create discriminator and monitor it
    con_dim = len(replay_buffers)
    discrim = disc(input_dim=obs_dim[0], context_dim=con_dim, **dc_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver([discrim])

    # Sync params across processes
    sync_params(discrim)

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buffer = PureVALORBuffer(con_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [discrim.pi])
    logger.log('\nNumber of parameters: \t d: %d\n' % var_counts)

    # Optimizers
    discrim_optimizer = Adam(discrim.pi.parameters(), lr=dc_lr)

    def update(e):
        obs, act = [torch.Tensor(x) for x in buffer.retrieve_all()]
        # print("fetched obs from buffer for update")   # print(obs)

        # Discriminator
        print('Discriminator Update!')
        con, s_diff = [torch.Tensor(x) for x in buffer.retrieve_dc_buff()]

        _, logp_dc, _ = discrim(s_diff, con)
        d_l_old = -logp_dc.mean()

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

        # Confusion matrix
        class_names = ["cyan", "marigold", "rose"]

        # wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,
        # y_true = gt,
        # preds = label,
        # class_names = class_names)})

        dc_l_new = -loggt_dc.mean()

        logger.store(LossDC=d_l_old,
                     DeltaLossDC=(dc_l_new - d_l_old))

    start_time = time.time()

    context_dist = Categorical(logits=torch.Tensor(np.ones(con_dim)))
    print("context distribution:", context_dist)
    total_t = 0
    ep_len = 0

    for epoch in range(epochs):
         # ac.eval()
         discrim.eval()
         print("local episodes:", local_episodes_per_epoch)
         for ep in range(local_episodes_per_epoch):
              t = random.randrange(0, len(replay_buffers)) # want to randomize draws for now
              c = torch.tensor(t)
              print("context sample: ", c)
              c_onehot = F.one_hot(c, con_dim).squeeze().float()

              for _ in range(max_ep_len):
                   # draw sample from replay buffer (try just one at a time for now)
                   # TODO: Review sampling method

                   sample_rb = replay_buffers[t].sample(1)
                   o = sample_rb['obs']
                   a = sample_rb['act']
                   next_o = sample_rb['next_obs']

                   # get change in state
                   o_diff = torch.tensor(next_o - o)

                   # concat_obs = torch.cat([torch.Tensor(o_diff.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)
                   concat_obs = torch.cat([torch.Tensor(o.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)

                   ep_len += 1
                   buffer.store(c, concat_obs.squeeze(), a)
                        # look at the bug, should take the second output instead of log_p ///
                        # instead of doing average over sequence dimension, do not need to average reward,
                        # just give reward now
                        # with pure VAE, do not have to calculate advantages
                   terminal = (ep_len == max_ep_len)
                   if terminal:
                       dc_diff = torch.Tensor(buffer.calc_diff()).unsqueeze(0)
                       con = torch.Tensor([float(c)]).unsqueeze(0)
                       _, loggt, _ = discrim(dc_diff, con)

                       buffer.finish_path(loggt.detach().numpy())
                       ep_len = 0

         if (epoch % save_freq == 0) or (epoch == epochs - 1):
              logger.save_state({'env': env}, [discrim], None)

         # Update
         discrim.train()

         # update models
         update(epoch)

         # Log
         logger.log_tabular('Epoch', epoch)
         # logger.log_tabular('EpLen', average_only=True)
         # logger.log_tabular('VVals', with_min_and_max=True)
         logger.log_tabular('LossDC', average_only=True)
         logger.log_tabular('DeltaLossDC', average_only=True)
         logger.log_tabular('Time', time.time() - start_time)
         logger.dump_tabular()

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

     pure_valor(lambda: gym.make(args.env),
                disc=ValorDiscriminator, dc_kwargs=dict(hidden_dims=[args.hid] * args.l),
                seed=args.seed, episodes_per_epoch=args.episodes_per_epoch,
                epochs=args.epochs,
                logger_kwargs=logger_kwargs)
