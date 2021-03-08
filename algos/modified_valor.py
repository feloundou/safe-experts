# Main entrance of GAIL
import numpy as np
import torch
import torch.nn.functional as F
import gym
import safety_gym
import time
import os.path as osp

from torch.distributions.categorical import Categorical

from neural_nets import ActorCritic, ValorDiscriminator, ModValorDiscriminator

import wandb

from utils import VALORBuffer, PureVALORBuffer
from utils import mpi_fork, proc_id, num_procs, EpochLogger,\
     setup_pytorch_for_mpi, sync_params, mpi_avg_grads, count_vars,  mpi_sum, MemoryBatch

def valor_mod(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(),
          disc=ModValorDiscriminator, dc_kwargs=dict(), seed=0,
          episodes_per_epoch=40,
          epochs=50, pi_lr=3e-4, vf_lr=1e-3, dc_lr=5e-4,
          train_pi_iters=1, train_v_iters=80,
          train_dc_iters=100,
          train_dc_interv=1,
          penalty_init=1.,
          penalty_lr=5e-3,
          clip_ratio=0.2,
          train_batch_size=50,
          eval_batch_size=100,
          train_valor_iters=50,
          max_ep_len=1000, logger_kwargs=dict(),
              # con_dim=5,
              con_dim=2,
              config_name='standard',
# splitN=max_ep_len-1,
          save_freq=10, k=1, memories=[]):
    # W&B Logging
    wandb.login()

    composite_name = 'new_valor_penalized_' + config_name
    wandb.init(project="LearningCurves", group="VALOR Expert", name=composite_name)

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

    ac_kwargs['action_space'] = env.action_space

    print("Running Modified VALOR")

    # Model    # Create actor-critic modules and discriminator and monitor them
    ac = actor_critic(input_dim=obs_dim[0] + con_dim, **ac_kwargs)
    discrim = disc(input_dim=obs_dim[0], context_dim=con_dim, **dc_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver([discrim])

    # Sync params across processes
    sync_params(ac)
    sync_params(discrim)

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())

    splitN = 999
    # buffer = VALORBuffer(con_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv)
    buffer = PureVALORBuffer(con_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv,
                             N=splitN)
    # Count variables
    var_counts = tuple(count_vars(module) for module in  [ac.pi, ac.v, discrim.pi])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t d: %d\n' % var_counts)

    # Optimizers
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)
    discrim_optimizer = torch.optim.Adam(discrim.pi.parameters(), lr=dc_lr)
    # new_discrim_optimizer = torch.optim.Adam(discrim.pi.parameters(), lr=dc_lr)

    start_time = time.time()

    # Prepare data
    mem = MemoryBatch(memories)
    transition_states, pure_states, transition_actions, expert_ids = mem.collate()

    o, r, d, ep_ret, ep_cost, ep_len, cum_reward, cum_cost = env.reset(), 0, False, 0, 0, 0, 0, 0
    context_dist = Categorical(logits=torch.Tensor(np.ones(con_dim)))
    print("context distribution:", context_dist)
    total_t = 0

    # Initialize penalty
    cur_penalty = np.log(max(np.exp(penalty_init) - 1, 1e-8))

    for epoch in range(epochs):
        ac.eval()
        discrim.eval()

        # Select state transitions and actions at random indexes
        batch_indexes = torch.randint(len(transition_states), (train_batch_size,))
        raw_states_batch, delta_states_batch, actions_batch, sampled_experts = \
            pure_states[batch_indexes], transition_states[batch_indexes], transition_actions[batch_indexes], expert_ids[
                batch_indexes]

        print("new diff state shape: ", delta_states_batch.shape)
        # c_tensor = [context_dist.sample() for i in range(train_batch_size)]
        o_tensor = context_dist.sample_n(train_batch_size)

        print("C Tensor: ", o_tensor)
        o_onehot = F.one_hot(o_tensor, con_dim).squeeze().float()
        # print("one hot: ", o_onehot)

        # New attempt to train here
        for _ in range(train_valor_iters):
            _, logp_dc_new, log_real = discrim(delta_states_batch, o_tensor)
            # _, logp_dc, _ = discrim(s_diff, con)
            # new_d_loss = -logp_dc_new.mean()
            new_d_loss = -log_real.mean()

            discrim_optimizer.zero_grad()
            new_d_loss.backward()
            mpi_avg_grads(discrim.pi)
            discrim_optimizer.step()

        # print("new Discrim loss: ", new_d_loss)
        logger.store(DiscrimLoss=new_d_loss)


        # print("Newest attempt logp dc", logp_dc_new)

        #
        # for _ in range(local_episodes_per_epoch):
        #     c = context_dist.sample()
        #     print("context sample: ", c)
        #     c_onehot = F.one_hot(c, con_dim).squeeze().float()
        #
        #     for _ in range(max_ep_len):
        #         concat_obs = torch.cat([torch.Tensor(o.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)
        #
        #         ep_ret += r
        #         ep_len += 1
        #         total_t += 1
        #
        #         buffer.store(c, concat_obs.squeeze(), 0, 0)
        #
        #         terminal = d or (ep_len == max_ep_len)
        #         if terminal:
        #             dc_diff = torch.Tensor(buffer.calc_diff()).unsqueeze(0)
        #             con = torch.Tensor([float(c)]).unsqueeze(0)
        #             # print("context going into discrim:", con)
        #             _, log_p, _ = discrim(dc_diff, con)
        #
        #             buffer.finish_path(log_p.detach().numpy())
        #             logger.store(EpRet=ep_ret, EpCost=ep_cost, EpLen=ep_len)
        #
        #             o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
        #
        # # Update
        # discrim.train()
        #
        # # update models
        # # Update!
        # obs, act, rew = [torch.Tensor(x) for x in buffer.retrieve_all()]
        #
        # # Discriminator
        # print('Discriminator Update!')
        #
        # con, s_diff = [torch.Tensor(x) for x in buffer.retrieve_dc_buff()]
        # print("s diff shape: ", s_diff.shape)
        # # print("mem state diff: ", )
        # # _, logp_dc, _ = discrim(s_diff, con)
        # _, logp_dc, _ = discrim(delta_states_batch, con)
        # print("LOG P DC: ", logp_dc)
        # d_l_old = -logp_dc.mean()
        #
        # # Discriminator train
        # for _ in range(train_dc_iters):
        #     _, logp_dc, _ = discrim(s_diff, con)
        #     d_loss = -logp_dc.mean()
        #     discrim_optimizer.zero_grad()
        #     d_loss.backward()
        #     mpi_avg_grads(discrim.pi)
        #     discrim_optimizer.step()
        #
        # _, logp_dc, _ = discrim(s_diff, con)
        # dc_l_new = -logp_dc.mean()
    # else:
    #     d_l_old = 0
    #     dc_l_new = 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [ac, discrim], None)

        # Log
        logger.log_tabular('Epoch', epoch)
        # logger.log_tabular('EpRet', with_min_and_max=True)
        # logger.log_tabular('EpLen', average_only=True)
        # logger.log_tabular('VVals', with_min_and_max=True)
        # logger.log_tabular('TotalEnvInteracts', total_t)
        # logger.log_tabular('LossPi', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        # logger.log_tabular('LossV', average_only=True)
        # logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('DiscrimLoss', average_only=True)
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
    # parser.add_argument('--episodes-per-epoch', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--exp_name', type=str, default='valor-anonymous-expert')
    parser.add_argument('--con', type=int, default=5)
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    valor_mod(lambda: gym.make(args.env), actor_critic=ActorCritic,
                    ac_kwargs=dict(hidden_dims=[args.hid] * args.l),
          disc=ModValorDiscriminator, dc_kwargs=dict(hidden_dims=[args.hid]*args.l),
           seed=args.seed, episodes_per_epoch=args.episodes_per_epoch, epochs=args.epochs,
          logger_kwargs=logger_kwargs, con_dim=args.con)
