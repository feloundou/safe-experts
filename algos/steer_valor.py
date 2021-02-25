# Main entrance of GAIL
import numpy as np
import gym
import safety_gym
import time, random, torch, wandb

import wandb.plot as wplot

import os.path as osp
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from neural_nets import ActorCritic, ValorDiscriminator, VDB, MLPContextLabeler, GaussianReward

from utils import PureVALORBuffer, mpi_fork, proc_id, num_procs, EpochLogger, \
    setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars


def steer_valor(env_fn,
                disc=ValorDiscriminator,
                con_labeler=MLPContextLabeler,
                reward_labeler=GaussianReward,
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
    wandb.init(project="LearningCurves", group="Steer VALOR Expert", name=composite_name)

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

    # con_encoder = con_labeler(env.observation_space.shape[0], context_dim = con_dim, activation=nn.LeakyReLU, **ac_kwargs)
    # con_encoder = con_labeler(env.observation_space.shape[0]*1000, context_dim=con_dim, activation=nn.LeakyReLU, **ac_kwargs)

    con_encoder = con_labeler(env.observation_space.shape[0], context_dim=con_dim, activation=nn.LeakyReLU,
                              **ac_kwargs)

    con_decoder = reward_labeler((env.observation_space.shape[0] + con_dim), activation=nn.LeakyReLU, **ac_kwargs  )  # TODO: Try nn.Tanh here

    # Set up model saving
    logger.setup_pytorch_saver([discrim])

    # Sync params across processes
    sync_params(discrim)
    sync_params(con_encoder)

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buffer = PureVALORBuffer(con_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv,
                             N=splitN)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [discrim.pi])
    logger.log('\nNumber of parameters: \t d: %d\n' % var_counts)

    # Optimizers
    discrim_optimizer = Adam(discrim.pi.parameters(), lr=dc_lr)
    encoder_optimizer = Adam(con_encoder.parameters(), lr=dc_lr)
    decoder_optimizer = Adam(con_decoder.parameters(), lr=dc_lr)

    def update(e):
        obs, act, rew = [torch.Tensor(x) for x in buffer.retrieve_all()]
        shaped_rewards = rew.reshape(local_episodes_per_epoch, max_ep_len)

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
    total_t, ep_len, total_r = 0, 0, 0

    def compute_loss_context(obs, ret):
        guessed_reward = con_decoder(obs)
        v_loss = F.mse_loss(guessed_reward, ret)
        print("Guessed Reward | Actual Reward | Value Loss \t %8.4f | %8.4f | %8.4f " % (guessed_reward.sum(), ret.sum(), v_loss))
        print("------------")

        return v_loss

    for epoch in range(epochs):
        discrim.eval()
        print("local episodes:", local_episodes_per_epoch)
        for ep in range(local_episodes_per_epoch):
            # Randomize among experts and sample replay buffers
            t = random.randrange(0, len(replay_buffers))

            # Sample memory also
            mem_observations, mem_actions, mem_rewards, mem_costs = memories[t].sample()

            episode_lengths = torch.tensor([len(episode) for episode in memories[t]])
            episode_limits = torch.cat([torch.tensor([0]), torch.cumsum(episode_lengths, dim=-1)])

            N = np.sum([len(episode) for episode in memories[t]])
            T = max_ep_len  # simulator.max_ep_len

            grouped_observations, grouped_rewards, grouped_actions = [], [], []   # grouped_observations = torch.zeros(N)

            for start, end in zip(episode_limits[:-1], episode_limits[1:]):
                grouped_observations.append(mem_observations[start:end])
                grouped_actions.append(mem_actions[start:end])
                grouped_rewards.append(mem_rewards[start:end])

            true_context = torch.tensor(t)

            # TODO: Feed the s_diff to this con_encoder
            init_diff = grouped_observations[0][1] - grouped_observations[0][0]
            print("GROUPED OBS: ", init_diff.shape)
            # c = con_encoder.label_context(torch.flatten(torch.Tensor(grouped_observations[0])))
            c = con_encoder.label_context(torch.flatten(torch.Tensor(init_diff)))
            print("Real Label \ Fake Label:  \t %d \ %d " % (true_context, c))
            # print("The fake context sample: ", c)
            c_onehot = F.one_hot(c, con_dim).squeeze().float()
            concat_obs = torch.cat([torch.Tensor(grouped_observations[0]), c_onehot.expand(1000, -1)], 1)  # for now only taking the first trajectory

            ep_rewards = []

            for st in range(max_ep_len):
                # draw trajectory from the memory (just one trajectory for now)
                # TODO: Review sampling method
                rewards = grouped_rewards[0][st]   # TODO: remove the 0s to complete the loop, change 0 to ep
                actions = grouped_actions[0][st]
                total_r += rewards
                ep_rewards.append(rewards)

                ep_len += 1
                buffer.store(c, concat_obs[st].squeeze(), actions, rewards)

                # instead of doing average over sequence dimension,# do not need to average reward, just give reward now
                terminal = (ep_len == max_ep_len)

                if terminal:
                    print("episode reward: ", total_r)
                    dc_diff = torch.Tensor(buffer.calc_diff()).unsqueeze(0)
                    con = torch.Tensor([float(c)]).unsqueeze(0)
                    _, loggt, _ = discrim(dc_diff, con)

                    dc_diff_concat = torch.cat([torch.Tensor(dc_diff[0]), c_onehot.expand(998, -1)], 1)
                    # use the concat obs and one-hot to predict episode reward (total r).
                    # the labeler gets the difference between true episode reward and the predicted reward as a loss

                    for step in range(5):
                        decoder_optimizer.zero_grad()
                        encoder_optimizer.zero_grad()
                        # split concatenated loss, take away the expert label for now
                        # TODO: Batch this operation
                        # guess the reward based on state differences and episode reward (may want to change this to step by step)
                        decode_loss = compute_loss_context(dc_diff_concat[step], ep_rewards[step+1])
                        # decode_loss = F.mse_loss(con_decoder(dc_diff_concat[step]), total_r)
                        decode_loss.backward()
                        # print("Decoder Loss: ", decode_loss)
                        mpi_avg_grads(con_decoder)
                        decoder_optimizer.step()

                        # use value loss (how good is the decoder at guessing step reward) as reward for the encoder
                        encode_loss = compute_loss_context(dc_diff_concat[step], ep_rewards[step+1])
                        encode_loss.backward()
                        mpi_avg_grads(con_encoder)
                        encoder_optimizer.step()

                    buffer.finish_path(loggt.detach().numpy())
                    ep_len, total_r, ep_rewards = 0, 0, []

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
    print("RUNNING FINAL EVAL")
    ground_truth, predictions = [], []

    for _ in range(50):
        discrim.eval()
        for ep in range(local_episodes_per_epoch):
            t = random.randrange(0, len(replay_buffers))  # want to randomize draws for now

            # Sample memory also
            mem_observations, mem_actions, mem_rewards, mem_costs = memories[t].sample()

            episode_lengths = torch.tensor([len(episode) for episode in memories[t]])
            episode_limits = torch.cat([torch.tensor([0]), torch.cumsum(episode_lengths, dim=-1)])

            N = np.sum([len(episode) for episode in memories[t]])
            T = max_ep_len  # simulator.max_ep_len

            grouped_observations, grouped_rewards, grouped_actions = [], [], []  # grouped_observations = torch.zeros(N)

            for start, end in zip(episode_limits[:-1], episode_limits[1:]):
                grouped_observations.append(mem_observations[start:end])
                grouped_actions.append(mem_actions[start:end])
                grouped_rewards.append(mem_rewards[start:end])

            true_context = torch.tensor(t)

            init_diff = grouped_observations[0][1] - grouped_observations[0][0]

            c = con_encoder.label_context(torch.flatten(torch.Tensor(init_diff)))
            print("Real Label \ Fake Label:  \t %d \ %d " % (true_context, c))
            c_onehot = F.one_hot(c, con_dim).squeeze().float()

            # append labels for plotting
            ground_truth.append(true_context)
            predictions.append(c)


    # Confusion matrix
    class_names = ["1", "2", "3"]
    wandb.log({"confusion_matrix": wplot.confusion_matrix(
        y_true=np.array(ground_truth), preds=np.array(predictions), class_names=class_names)})


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

    steer_valor(lambda: gym.make(args.env),
                disc=ValorDiscriminator, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                dc_kwargs=dict(hidden_dims=[args.hid] * args.l),
                seed=args.seed, episodes_per_epoch=args.episodes_per_epoch,
                epochs=args.epochs,
                logger_kwargs=logger_kwargs)
