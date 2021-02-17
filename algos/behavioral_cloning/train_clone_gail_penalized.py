
# Main entrance of GAIL
import numpy as np
import torch
import torch.nn.functional as F
import gym
import safety_gym
import time
import os.path as osp

import wandb

from neural_nets import Discriminator, ActorCritic, count_vars

from utils import BufferStudent, BufferTeacher
from utils import mpi_fork, proc_id, num_procs, EpochLogger,\
    average_gradients, sync_all_params, setup_pytorch_for_mpi, sync_params, mpi_avg_grads


def gail_penalized(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(),
         disc=Discriminator,
         dc_kwargs=dict(), seed=0,
         episodes_per_epoch=40,
         epochs=500,
         gamma=0.99, lam=0.97,
         # Cost constraints / penalties:
         cost_lim=25,
         penalty_init=1.,
         penalty_lr=5e-3,
         clip_ratio=0.2,
         pi_lr=3e-3, vf_lr=3e-3, dc_lr=5e-4,
         train_v_iters=80, train_pi_iters=80, train_dc_iters=80,
         max_ep_len=1000, logger_kwargs=dict(), config_name = 'standard', save_freq=10):
    # W&B Logging
    wandb.login()

    composite_name = 'new_gail_penalized_' + config_name
    wandb.init(project="LearningCurves", group="GAIL Clone", name=composite_name)

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    l_lam = 0  # balance two loss terms


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

    # Models  # Create actor-critic and discriminator modules
    ac = actor_critic(input_dim=obs_dim[0], **ac_kwargs)
    discrim = disc(input_dim=obs_dim[0], **dc_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver([ac, discrim])

    # Sync params across processes
    sync_params(ac)
    sync_params(discrim)


    # Load expert policy here
    expert = actor_critic(input_dim=obs_dim[0], **ac_kwargs)
    # expert_name = "expert_torch_save.pt"
    expert_name = "model.pt"
    # expert = torch.load(osp.join(logger_kwargs['output_dir'],'pyt_save' , expert_name))
    # expert = torch.load('/home/tyna/Documents/openai/research-project/data/anonymous-expert/anonymous-expert_s0/pyt_save/model.pt')
    expert = torch.load(
        '/home/tyna/Documents/openai/research-project/data/test-pen-ppo/test-pen-ppo_s0/pyt_save/model.pt')

    print('RUNNING GAIL')

    # Buffers
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buff_s = BufferStudent(obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len)
    buff_t = BufferTeacher(obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.pi, ac.v, discrim.pi])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t d: %d\n' % var_counts)


    # Optimizers
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)
    discrim_optimizer = torch.optim.Adam(discrim.pi.parameters(), lr=dc_lr)

    # # Parameters Sync
    # sync_all_params(ac.parameters())
    # sync_all_params(disc.parameters())

    # Set up function for computing PPO policy loss
    def compute_loss_pi(obs, act, adv, logp_old):
        # Policy loss # policy gradient term + entropy term
        # Policy loss with clipping (without clipping, loss_pi = -(logp*adv).mean()).
        # TODO: Think about removing clipping
        _, logp, _ = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        return loss_pi


    def penalty_update(cur_penalty):
        cur_cost = logger.get_stats('EpCostS')[0]
        cur_rew = logger.get_stats('EpRetS')[0]

        # Penalty update
        cur_penalty = max(0, cur_penalty + penalty_lr * (cur_cost - cost_lim))
        return cur_penalty

    def update(e):
        obs_s, act, adv, ret, log_pi_old = [torch.Tensor(x) for x in buff_s.retrieve_all()]
        obs_t, _ = [torch.Tensor(x) for x in buff_t.retrieve_all()]

        # Policy
        _, logp, _ = ac.pi(obs_s, act)
        entropy = (-logp).mean()

        # Policy loss   # policy gradient term + entropy term
        # loss_pi = -(logp * adv).mean() - l_lam * entropy

        # Train policy
        if e > 10:
            # Train policy with multiple steps of gradient descent
            for _ in range(train_pi_iters):
                pi_optimizer.zero_grad()
                loss_pi = compute_loss_pi(obs, act, adv, ret)
                loss_pi.backward()
                mpi_avg_grads(ac.pi)
                pi_optimizer.step()

        # Value function
        v = ac.v(obs_s)
        v_l_old = F.mse_loss(v, ret)

        for _ in range(train_v_iters):
            v = ac.v(obs_s)
            v_loss = F.mse_loss(v, ret)

            # Value function train
            vf_optimizer.zero_grad()
            v_loss.backward()
            mpi_avg_grads(ac.v)  # average gradients across MPI processes
            vf_optimizer.step()

        # Discriminator
        gt1 = torch.ones(obs_s.size()[0], dtype=torch.int)
        gt2 = torch.zeros(obs_t.size()[0], dtype=torch.int)
        _, logp_student, _ = discrim(obs_s, gt=gt1)
        _, logp_teacher, _ = discrim(obs_t, gt=gt2)
        discrim_l_old = - logp_student.mean() - logp_teacher.mean()

        for _ in range(train_dc_iters):
            _, logp_student, _ = discrim(obs_s, gt=gt1)
            _, logp_teacher, _ = discrim(obs_t, gt=gt2)
            dc_loss = - logp_student.mean() - logp_teacher.mean()

            # Discriminator train
            discrim_optimizer.zero_grad()
            dc_loss.backward()
            # average_gradients(discrim_optimizer.param_groups)
            mpi_avg_grads(discrim.pi)
            discrim_optimizer.step()


        _, logp_student, _ = discrim(obs_s, gt=gt1)
        _, logp_teacher, _ = discrim(obs_t, gt=gt2)
        dc_loss_new = - logp_student.mean() - logp_teacher.mean()

        # Log the changes
        _, logp, _, v = ac(obs, act)
        entropy_new = (-logp).mean()
        pi_loss_new = -(logp * adv).mean() - l_lam * entropy
        v_loss_new = F.mse_loss(v, ret)
        kl = (log_pi_old - logp).mean()
        logger.store(
            # LossPi=loss_pi,
            LossV=v_l_old, LossDC=discrim_l_old,
                     # DeltaLossPi=(pi_loss_new - loss_pi),
                     DeltaLossV=(v_loss_new - v_l_old), DeltaLossDC=(dc_loss_new - discrim_l_old),
                     DeltaEnt=(entropy_new - entropy),
                     Entropy=entropy, KL=kl)

    start_time = time.time()
    o, r, sdr, d, ep_ret, ep_cost, ep_sdr, ep_len = env.reset(), 0, 0, False, 0, 0, 0, 0
    total_t = 0
    ep_len_t = 0

    # Initialize penalty
    cur_penalty = np.log(max(np.exp(penalty_init) - 1, 1e-8))

    for epoch in range(epochs):
        ac.eval()
        discrim.eval()
        # We recognize the probability term of index [0] correspond to the teacher's policy

        # Student's policy rollout
        for _ in range(local_episodes_per_epoch):
            for _ in range(max_ep_len):
                obs = torch.Tensor(o.reshape(1, -1))
                a, _, lopg_t, v_t = ac(obs)

                buff_s.store(o, a.detach().numpy(), r, sdr, v_t.item(), lopg_t.detach().numpy())
                logger.store(VVals=v_t)

                o, r, d, info = env.step(a.detach().numpy()[0])
                # print("INFO: ", info)
                c = info.get("cost")

                _, sdr, _ = discrim(torch.Tensor(o.reshape(1, -1)), gt=torch.Tensor([0]))
                if sdr < -4:  # Truncate rewards
                    sdr = -4

                ep_ret += r
                ep_cost += c
                ep_sdr += sdr
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    buff_s.end_episode()
                    logger.store(EpRetS=ep_ret, EpCostS= ep_cost, EpLenS=ep_len, EpSdrS=ep_sdr)
                    print("Student Episode Return: \t", ep_ret)
                    o, r, sdr, d, ep_ret, ep_cost, ep_sdr, ep_len = env.reset(), 0, 0, False, 0, 0, 0, 0

        # Teacher's policy rollout
        for _ in range(local_episodes_per_epoch):
            for _ in range(max_ep_len):
                # obs =
                a, _, _, _ = expert(torch.Tensor(o.reshape(1, -1)))

                buff_t.store(o, a.detach().numpy(), r)

                o, r, d, info = env.step(a.detach().numpy()[0])
                c = info.get("cost")
                ep_ret += r
                ep_cost += c
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    buff_t.end_episode()
                    logger.store(EpRetT=ep_ret, EpCostT=ep_cost, EpLenT=ep_len)
                    print("Teacher Episode Return: \t", ep_ret)
                    o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [ac, discrim], None)

        # Update
        ac.train()
        discrim.train()

        # update penalty
        cur_penalty = penalty_update(cur_penalty)

        # update networks
        update(epoch)

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRetS', average_only=True)
        logger.log_tabular('EpSdrS', average_only=True)
        logger.log_tabular('EpLenS', average_only=True)
        logger.log_tabular('EpRetT', average_only=True)
        logger.log_tabular('EpLenT', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', total_t)
        # logger.log_tabular('LossPi', average_only=True)
        # logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('LossDC', average_only=True)
        logger.log_tabular('DeltaLossDC', average_only=True)
        # logger.log_tabular('Entropy', average_only=True)
        # logger.log_tabular('DeltaEnt', average_only=True)
        # logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.97)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--episodes-per-epoch', type=int, default=5)
    # parser.add_argument('--episodes-per-epoch', type=int, default=40)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='valor-anonymous-expert')
    parser.add_argument('--con', type=int, default=5)
    args = parser.parse_args()

    mpi_fork(args.cpu)

    from utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    gail_penalized(lambda: gym.make(args.env), actor_critic=ActorCritic, ac_kwargs=dict(hidden_dims=[args.hid] * args.l),
         disc=Discriminator, dc_kwargs=dict(hidden_dims=[args.hid] * args.l), gamma=args.gamma, lam=args.lam,
         seed=args.seed, episodes_per_epoch=args.episodes_per_epoch, epochs=args.epochs, logger_kwargs=logger_kwargs)
