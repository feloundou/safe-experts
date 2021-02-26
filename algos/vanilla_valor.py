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

from neural_nets import ActorCritic, ValorDiscriminator, \
    VDB, MLPContextLabeler, GaussianReward, ValorActorCritic, VanillaVAE, VAELOR

from utils import PureVALORBuffer, mpi_fork, proc_id, num_procs, EpochLogger, \
    setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars


####################################################3

def vanilla_valor(env_fn,
                  disc=ValorDiscriminator,

                  con_labeler=MLPContextLabeler,
                  # vae = VanillaVAE,
                  vae=VAELOR,
                  reward_labeler=GaussianReward,
                  actor_critic=ValorActorCritic,
                  ac_kwargs=dict(), dc_kwargs=dict(),
                  seed=0,
                  episodes_per_epoch=40,
                  epochs=50,
                  dc_lr=5e-4, pi_lr=3e-4,
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
    # discrim = disc(input_dim=obs_dim[0], context_dim=con_dim, **dc_kwargs)
    input_dim = obs_dim[0] + con_dim
    ac = actor_critic(input_dim, env.action_space, **ac_kwargs)

    context_decoder = con_labeler(env.observation_space.shape[0], context_dim=con_dim, activation=nn.LeakyReLU,
                                  **ac_kwargs)
    context_encoder = reward_labeler((env.observation_space.shape[0] + con_dim), activation=nn.LeakyReLU,
                                     **ac_kwargs)  # TODO: Try nn.Tanh here
    # valor_vae = vae(in_dims=env.observation_space.shape[0], latent_dim=3, out_dim=2)

    valor_vae = vae(input_dim=env.observation_space.shape[0], latent_dim=3)

    # Set up model saving
    logger.setup_pytorch_saver([context_decoder])

    # Sync params across processes
    sync_params(ac)
    sync_params(valor_vae)
    sync_params(context_decoder)

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buffer = PureVALORBuffer(con_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv,
                             N=splitN)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi])
    logger.log('\nNumber of parameters: \t d: %d\n' % var_counts)

    # Optimizers
    vae_optimizer = Adam(valor_vae.parameters(), lr=pi_lr)
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    # encoder_optimizer = Adam(context_encoder.parameters(), lr=dc_lr)
    decoder_optimizer = Adam(context_decoder.parameters(), lr=dc_lr)

    vae_criterion = nn.MSELoss()

    def latent_loss(z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    start_time = time.time()
    context_dist = Categorical(logits=torch.Tensor(np.ones(con_dim)))
    total_t, ep_len, total_r = 0, 0, 0

    for epoch in range(epochs):
        print("local episodes:", local_episodes_per_epoch)
        expert_states, expert_actions, = None, None

        for k in range(len(replay_buffers)):
            if expert_states is None:
                expert_states, expert_actions, _, _ = memories[k].sample()
            else:
                states, actions, _, _ = memories[k].sample()
                expert_states = torch.cat([expert_states, torch.Tensor(states)])
                expert_actions = torch.cat([expert_actions, torch.Tensor(actions)])

        N = expert_states.shape[0]
        print("N: ", N)

        # Shuffle the data (make sure the NN is not simply learning how long episodes are)
        expert_states = expert_states[torch.randperm(expert_states.size()[0])]
        expert_actions = expert_actions[torch.randperm(expert_actions.size()[0])]

        for ep in range(local_episodes_per_epoch):
            # Randomize among experts and sample replay buffers
            t = random.randrange(0, len(replay_buffers))
            true_context = torch.tensor(t)

            # for st in range(max_ep_len):
            s_diff_data = None
            step_batch_size=2

            for st in range(step_batch_size):
                s_diff = expert_states[st + 1] - expert_states[st]
                a = expert_actions[st]

                if s_diff_data is None:
                    s_diff_data = torch.Tensor(s_diff).unsqueeze(0)
                else:
                    s_diff_data = torch.cat([torch.Tensor(s_diff_data), torch.Tensor(s_diff.unsqueeze(0))])

                # print("collecting some data", s_diff_data.shape)

                # # Generate some label
                # c = context_decoder.label_context(torch.flatten(torch.Tensor(s_diff)))
                # c_onehot = F.one_hot(c, con_dim).squeeze().float()
                #
                # # Attach it to generated one-hot vector
                # concat_obs = torch.cat([torch.Tensor(s_diff), c_onehot])

            # test_vae_output = valor_vae(s_diff)

            # Train the VAE
            train_vae_iters=20
            for _ in range(train_vae_iters):
                vae_optimizer.zero_grad()
                dec = valor_vae(s_diff_data)
                # print("Decoded Values: ", dec)
                # print("decoder shape: ", dec.shape)
                ll = latent_loss(valor_vae.z_mean, valor_vae.z_sigma)

                loss = vae_criterion(dec, s_diff_data) + ll

                # print("Latent Loss: ", ll)
                # print("VAE Loss: ", vae_criterion(dec, s_diff_data))
                loss.backward()
                vae_optimizer.step()
                # print("Loss Data: ", loss.data)
                l = loss.data.item()
                ep_len += 1

                # discriminator_metrics = {'Step Loss': decoder_loss}
                # wandb.log(discriminator_metrics)

        print("VAELoss: ", l)

        logger.store(VAELoss=l)

        # logger.store(LossPi=loss_pi)

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [context_decoder], None)

        # Log
        logger.log_tabular('Epoch', epoch)
        # logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VAELoss', average_only=True)
        # logger.log_tabular('DeltaLossDC', average_only=True)
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

    vanilla_valor(lambda: gym.make(args.env),
                  disc=ValorDiscriminator, ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
                  dc_kwargs=dict(hidden_dims=[args.hid] * args.l),
                  seed=args.seed, episodes_per_epoch=args.episodes_per_epoch,
                  epochs=args.epochs,
                  logger_kwargs=logger_kwargs)
