# Main entrance of GAIL
import numpy as np
import torch
import torch.nn.functional as F
import gym
import safety_gym
import time
import random
import os.path as osp
from torch import nn
from torch.optim import Adam

from torch.distributions.categorical import Categorical

from neural_nets import ActorCritic, ValorDiscriminator, VDB, GaussianReward

import wandb
import wandb.plot as wplot

from utils import PureVALORBuffer
from utils import mpi_fork, proc_id, num_procs, EpochLogger, \
    setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars


def value_valor(env_fn,
                disc=ValorDiscriminator,
                label_disc=GaussianReward,
                ac_kwargs=dict(),
                dc_kwargs=dict(),
                seed=0,
                episodes_per_epoch=40,
                epochs=50,
                dc_lr=5e-4,
                train_dc_iters=10,
                train_dc_interv=1,
                max_ep_len=20, logger_kwargs=dict(),
                config_name='standard',
                splitN=8,
                save_freq=10, replay_buffers=[], memories=[]):
    # W&B Logging
    wandb.login()

    composite_name = 'new_valor_' + config_name
    wandb.init(project="LearningCurves", group="Guess VALOR Expert", name=composite_name)

    assert replay_buffers != [], "Replay buffers must be set"

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
    # label_nn = label_disc(input_dim=obs_dim)

    reward_nn = label_disc(env.observation_space.shape[0],  activation=nn.LeakyReLU, **ac_kwargs)

    reward_criterion = criterion = nn.MSELoss()

    # Set up model saving
    logger.setup_pytorch_saver([discrim])

    # Sync params across processes
    sync_params(discrim)

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buffer = PureVALORBuffer(con_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv,
                             N=splitN)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [discrim.pi])
    logger.log('\nNumber of parameters: \t d: %d\n' % var_counts)

    # Optimizers
    discrim_optimizer = Adam(discrim.pi.parameters(), lr=dc_lr)
    reward_optimizer = Adam(reward_nn.parameters(), lr=dc_lr)

    def compute_loss_reward(obs, ret):
        guessed_reward = reward_nn(obs)
        print("Guessed Reward over the episode: ", guessed_reward.sum())
        print("Actual Reward over the episode: ", ret.sum())
        v_loss = ((guessed_reward - ret) ** 2).mean()
        print()
        return v_loss

    def update(e):
        obs, act, rew = [torch.Tensor(x) for x in buffer.retrieve_all()]
        simple_obs = torch.split(obs, [60,3], dim=1 )


        # print("fetched obs from buffer for update")   # print(obs)

        # Discriminator
        print('Discriminator Update!')

        # print("Buffer states! ", obs)
        # print("State buffer shape: ", obs.shape)
        # # print("Simple obs: ", simple_obs)
        # print("split obs shape: ", torch.Tensor(simple_obs[0]).shape)
        # # print("Buffer rewards! ", rew)
        # print("Rewards buffer shape: ", rew.shape)
        # print("Wow")
        # print(torch.Tensor(simple_obs[0]))
        # print("All DONE!")

        con, s_diff = [torch.Tensor(x) for x in buffer.retrieve_dc_buff()]

        # Train policy with multiple steps of gradient descent
        for _ in range(10):
            reward_optimizer.zero_grad()
            # split concatenated loss, take away the expert label for now
            loss_v = compute_loss_reward(s_diff, rew)
            loss_v.backward()
            mpi_avg_grads(reward_nn)
            reward_optimizer.step()

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
    context_dist = Categorical(logits=torch.Tensor(np.ones(con_dim)))
    total_t, ep_len, total_r = 0, 0, 0

    for epoch in range(epochs):
        discrim.eval()
        print("local episodes:", local_episodes_per_epoch)
        for ep in range(local_episodes_per_epoch):
            t = random.randrange(0, len(replay_buffers))  # want to randomize draws for now
            c = torch.tensor(t)
            print("context sample: ", c)
            c_onehot = F.one_hot(c, con_dim).squeeze().float()

            for _ in range(max_ep_len):
                # draw sample from replay buffer (try just one at a time for now)
                # TODO: Review sampling method

                sample_rb = replay_buffers[t].sample(1)
                o = sample_rb['obs']
                a = sample_rb['act']
                rewards = sample_rb['rew']

                # Sample memory also
                mem_observations, mem_actions, mem_rewards, mem_costs = memories[t].sample()


                # print("Comparing the outputs")
                # print("replay buffer observations: ", o.shape)
                # print("memory observations: ", mem_observations.shape)

                episode_lengths = torch.tensor([len(episode) for episode in memories[t]])
                # print("episode lengths: ", episode_lengths)
                episode_limits = torch.cat([torch.tensor([0]), torch.cumsum(episode_lengths, dim=-1)])
                # print("episode limits: ", episode_limits)

                N = np.sum([len(episode) for episode in memories[t]])

                T = max_ep_len  # simulator.max_ep_len

                # grouped_observations = torch.zeros(N)
                grouped_observations = []
                # print("N: ", N)


                for start, end in zip(episode_limits[:-1], episode_limits[1:]):

                    # print("start: ", start)
                    # print("end: ", end)
                    grouped_observations.append(mem_observations[start:end])
                    # grouped_observations[start:end] = mem_observations[start:end]

                # print("grouped observations length: ", len(grouped_observations))
                # print("first episode in grouped obs: ", len(grouped_observations[0]))

                # next_o = sample_rb['next_obs']

                # get change in state
                # o_diff = torch.tensor(next_o - o)



                total_r += rewards[0]

                reward_pred = reward_nn(torch.tensor(o).float())
                reward_loss = reward_criterion(reward_pred, torch.tensor(rewards))

                test_reward_pred = reward_nn(torch.tensor(grouped_observations[0]).float())
                test_reward_loss = reward_criterion(test_reward_pred, torch.tensor(mem_rewards))
                print("test reward loss: ", test_reward_loss)

                # print("Reward Loss:", reward_loss)
                # print("reward predictions: ", reward_pred)
                # print("actual rewards: ", rewards)

                # var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v])
                # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

                # concat_obs = torch.cat([torch.Tensor(o_diff.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)
                concat_obs = torch.cat([torch.Tensor(o.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)

                ep_len += 1
                buffer.store(c, concat_obs.squeeze(), a, rewards)

                # instead of doing average over sequence dimension, do not need to average reward,
                # just give reward now
                terminal = (ep_len == max_ep_len)
                if terminal:
                    print("episode reward: ", total_r)
                    dc_diff = torch.Tensor(buffer.calc_diff()).unsqueeze(0)
                    con = torch.Tensor([float(c)]).unsqueeze(0)
                    _, loggt, _ = discrim(dc_diff, con)

                    buffer.finish_path(loggt.detach().numpy())
                    ep_len, total_r = 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [discrim], None)

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
    #
    # # Run eval
    # print("RUNNING FINAL EVAL")
    #
    # ground_truth = []
    # predictions = []
    # ep_len = 0
    #
    # for _ in range(50):
    #     discrim.eval()
    #     for ep in range(local_episodes_per_epoch):
    #         t = random.randrange(0, len(replay_buffers))  # want to randomize draws for now
    #         c = torch.tensor(t)
    #         print("context sample: ", c)
    #         c_onehot = F.one_hot(c, con_dim).squeeze().float()
    #
    #         for _ in range(max_ep_len):
    #
    #             sample_rb = replay_buffers[t].sample(1)
    #             o = sample_rb['obs']
    #             a = sample_rb['act']
    #
    #             concat_obs = torch.cat([torch.Tensor(o.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)
    #
    #             ep_len += 1
    #             buffer.store(c, concat_obs.squeeze(), a)
    #             terminal = (ep_len == max_ep_len)
    #             if terminal:
    #                 dc_diff = torch.Tensor(buffer.calc_diff()).unsqueeze(0)
    #                 con = torch.Tensor([float(c)]).unsqueeze(0)
    #                 label, loggt_dc, logp_dc, gt = discrim(dc_diff, con, classes=True)
    #
    #                 ground_truth.append(int(gt[0][0]))
    #                 predictions.append(int(label[0]))
    #
    #                 buffer.finish_path(loggt_dc.detach().numpy())
    #                 ep_len = 0
    #
    #     # Update
    #     discrim.train()
    #     obs, act = [torch.Tensor(x) for x in buffer.retrieve_all()]
    #
    # # Confusion matrix
    # class_names = ["cyan", "marigold", "rose"]
    # wandb.log({"confusion_mat": wplot.confusion_matrix(
    #     y_true=np.array(ground_truth), preds=np.array(predictions), class_names=class_names)})

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

    value_valor(lambda: gym.make(args.env),
                disc=ValorDiscriminator, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                dc_kwargs=dict(hidden_dims=[args.hid] * args.l),
                seed=args.seed, episodes_per_epoch=args.episodes_per_epoch,
                epochs=args.epochs,
                logger_kwargs=logger_kwargs)
