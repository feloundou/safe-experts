# Main entrance of GAIL
import numpy as np
import torch
import torch.nn.functional as F
import gym
import safety_gym
import time
import os.path as osp

from torch.distributions.categorical import Categorical

from neural_nets import Discriminator, ActorCritic, ValorDiscriminator, count_vars

from utils import BufferS, BufferT, BufferActor, VALORBuffer
from utils import mpi_fork, proc_id, num_procs, EpochLogger,\
    average_gradients, sync_all_params



def ppo_penalized(env_fn,
            actor_critic=ActorCritic,
            ac_kwargs=dict(),
            seed=0,
            episodes_per_epoch=40,
            epochs=500,
            gamma=0.99,
            lam=0.97,
            pi_lr=3e-4,
            vf_lr=1e-3,
            train_v_iters=80,
            max_ep_len=1000,
            logger_kwargs=dict(),
            save_freq=10):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac_kwargs['action_space'] = env.action_space

    # Models
    ac = actor_critic(input_dim=obs_dim[0], **ac_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    # Buffers
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buf = BufferActor(obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.policy, ac.value_f])
    print("POLICY GRADIENT")
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # Optimizers
    train_pi = torch.optim.Adam(ac.policy.parameters(), lr=pi_lr)
    train_v = torch.optim.Adam(ac.value_f.parameters(), lr=vf_lr)

    # Parameters Sync
    sync_all_params(ac.parameters())

    def update(e):
        obs, act, adv, ret, lgp_old = [torch.Tensor(x) for x in buf.retrieve_all()]

        # Policy
        _, lgp, _ = ac.policy(obs, act)
        entropy = (-lgp).mean()

        # Policy loss # policy gradient term + entropy term
        pi_loss = -(lgp * adv).mean()

        # Train policy
        train_pi.zero_grad()
        pi_loss.backward()
        average_gradients(train_pi.param_groups)
        train_pi.step()

        # Value function
        v = ac.value_f(obs)
        v_l_old = F.mse_loss(v, ret)
        for _ in range(train_v_iters):
            v = ac.value_f(obs)
            v_loss = F.mse_loss(v, ret)

            # Value function train
            train_v.zero_grad()
            v_loss.backward()
            average_gradients(train_v.param_groups)
            train_v.step()

        # Log the changes
        _, lgp, _, v = ac(obs, act)
        entropy_new = (-lgp).mean()
        pi_loss_new = -(lgp * adv).mean()
        v_loss_new = F.mse_loss(v, ret)
        kl = (lgp_old - lgp).mean()
        logger.store(LossPi=pi_loss, LossV=v_l_old, DeltaLossPi=(pi_loss_new - pi_loss),
                     DeltaLossV=(v_loss_new - v_l_old), Entropy=entropy, KL=kl)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_t = 0

    for epoch in range(epochs):
        ac.eval()
        # Policy rollout
        for _ in range(local_episodes_per_epoch):
            for _ in range(max_ep_len):
                obs = torch.Tensor(o.reshape(1, -1))
                a, _, lopg_t, v_t = ac(obs)

                buf.store(o, a.detach().numpy(), r, v_t.item(), lopg_t.detach().numpy())
                logger.store(VVals=v_t)

                o, r, d, _ = env.step(a.detach().numpy()[0])
                ep_ret += r
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    buf.end_episode()
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            # logger._torch_save(ac, fname="expert_torch_save.pt")
            # logger._torch_save(ac, fname="model.pt")
            logger.save_state({'env': env}, None, None)

        # Update
        ac.train()

        update(epoch)

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', total_t)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()







def gail_penalized(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(),
         disc=Discriminator,
         dc_kwargs=dict(), seed=0,
         episodes_per_epoch=40,
         epochs=500,
         gamma=0.99, lam=0.97,
         pi_lr=3e-3, vf_lr=3e-3, dc_lr=5e-4, train_v_iters=80, train_dc_iters=80,
         max_ep_len=1000, logger_kwargs=dict(), save_freq=10):
    l_lam = 0  # balance two loss term

    print("starting now")

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac_kwargs['action_space'] = env.action_space

    # Models
    ac = actor_critic(input_dim=obs_dim[0], **ac_kwargs)
    disc = disc(input_dim=obs_dim[0], **dc_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver([ac, disc])

    # TODO: Load expert policy here
    expert = actor_critic(input_dim=obs_dim[0], **ac_kwargs)
    # expert_name = "expert_torch_save.pt"
    expert_name = "model.pt"
    # expert = torch.load(osp.join(logger_kwargs['output_dir'],'pyt_save' , expert_name))
    expert = torch.load('/home/tyna/Documents/openai/research-project/data/anonymous-expert/anonymous-expert_s0/pyt_save/model.pt')

    print('RUNNING GAIL')

    # Buffers
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buff_s = BufferS(obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len)
    buff_t = BufferT(obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len)

    # Count variables
    var_counts = tuple(count_vars(module) for module in [ac.policy, ac.value_f, disc.policy])
    print("GAIL")
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t d: %d\n' % var_counts)


    # Optimizers
    train_pi = torch.optim.Adam(ac.policy.parameters(), lr=pi_lr)
    train_v = torch.optim.Adam(ac.value_f.parameters(), lr=vf_lr)
    train_dc = torch.optim.Adam(disc.policy.parameters(), lr=dc_lr)

    # Parameters Sync
    sync_all_params(ac.parameters())
    sync_all_params(disc.parameters())

    def update(e):
        obs_s, act, adv, ret, lgp_old = [torch.Tensor(x) for x in buff_s.retrieve_all()]
        obs_t, _ = [torch.Tensor(x) for x in buff_t.retrieve_all()]

        # Policy
        _, lgp, _ = ac.policy(obs_s, act)
        entropy = (-lgp).mean()

        # Policy loss
        # policy gradient term + entropy term
        pi_loss = -(lgp * adv).mean() - l_lam * entropy

        # Train policy
        if e > 10:
            train_pi.zero_grad()
            pi_loss.backward()
            average_gradients(train_pi.param_groups)
            train_pi.step()

        # Value function
        v = ac.value_f(obs_s)
        v_l_old = F.mse_loss(v, ret)

        for _ in range(train_v_iters):
            v = ac.value_f(obs_s)
            v_loss = F.mse_loss(v, ret)

            # Value function train
            train_v.zero_grad()
            v_loss.backward()
            average_gradients(train_v.param_groups)
            train_v.step()

        # Discriminator
        gt1 = torch.ones(obs_s.size()[0], dtype=torch.int)
        gt2 = torch.zeros(obs_t.size()[0], dtype=torch.int)
        _, lgp_s, _ = disc(obs_s, gt=gt1)
        _, lgp_t, _ = disc(obs_t, gt=gt2)
        dc_loss_old = - lgp_s.mean() - lgp_t.mean()

        for _ in range(train_dc_iters):
            _, lgp_s, _ = disc(obs_s, gt=gt1)
            _, lgp_t, _ = disc(obs_t, gt=gt2)
            dc_loss = - lgp_s.mean() - lgp_t.mean()

            # Discriminator train
            train_dc.zero_grad()
            dc_loss.backward()
            average_gradients(train_dc.param_groups)
            train_dc.step()

        _, lgp_s, _ = disc(obs_s, gt=gt1)
        _, lgp_t, _ = disc(obs_t, gt=gt2)
        dc_loss_new = - lgp_s.mean() - lgp_t.mean()

        # Log the changes
        _, lgp, _, v = ac(obs, act)
        entropy_new = (-lgp).mean()
        pi_loss_new = -(lgp * adv).mean() - l_lam * entropy
        v_loss_new = F.mse_loss(v, ret)
        kl = (lgp_old - lgp).mean()
        logger.store(LossPi=pi_loss, LossV=v_l_old, LossDC=dc_loss_old, DeltaLossPi=(pi_loss_new - pi_loss),
                     DeltaLossV=(v_loss_new - v_l_old), DeltaLossDC=(dc_loss_new - dc_loss_old),
                     DeltaEnt=(entropy_new - entropy),
                     Entropy=entropy, KL=kl)

    start_time = time.time()
    o, r, sdr, d, ep_ret, ep_sdr, ep_len = env.reset(), 0, 0, False, 0, 0, 0
    total_t = 0

    ep_len_t = 0
    for epoch in range(epochs):
        ac.eval()
        disc.eval()
        # We recognize the probability term of index [0] correspond to the teacher's policy

        # Student's policy rollout
        for _ in range(local_episodes_per_epoch):
            for _ in range(max_ep_len):
                obs = torch.Tensor(o.reshape(1, -1))
                a, _, lopg_t, v_t = ac(obs)

                buff_s.store(o, a.detach().numpy(), r, sdr, v_t.item(), lopg_t.detach().numpy())
                logger.store(VVals=v_t)

                o, r, d, _ = env.step(a.detach().numpy()[0])
                _, sdr, _ = disc(torch.Tensor(o.reshape(1, -1)), gt=torch.Tensor([0]))
                if sdr < -4:  # Truncate rewards
                    sdr = -4
                ep_ret += r
                ep_sdr += sdr
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    buff_s.end_episode()
                    logger.store(EpRetS=ep_ret, EpLenS=ep_len, EpSdrS=ep_sdr)
                    print("Student Episode Return: \t", ep_ret)
                    o, r, sdr, d, ep_ret, ep_sdr, ep_len = env.reset(), 0, 0, False, 0, 0, 0

        # Teacher's policy rollout
        for _ in range(local_episodes_per_epoch):
            for _ in range(max_ep_len):
                obs = torch.Tensor(o.reshape(1, -1))
                a, _, _, _ = expert(obs)

                buff_t.store(o, a.detach().numpy(), r)

                o, r, d, _ = env.step(a.detach().numpy()[0])
                ep_ret += r
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    buff_t.end_episode()
                    logger.store(EpRetT=ep_ret, EpLenT=ep_len)
                    print("Teacher Episode Return: \t", ep_ret)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [ac, disc], None)

        # Update
        ac.train()
        disc.train()

        update(epoch)

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRetS', with_min_and_max=True)
        logger.log_tabular('EpSdrS', with_min_and_max=True)
        logger.log_tabular('EpLenS', average_only=True)
        logger.log_tabular('EpRetT', with_min_and_max=True)
        logger.log_tabular('EpLenT', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', total_t)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('LossDC', average_only=True)
        logger.log_tabular('DeltaLossDC', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('DeltaEnt', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()


def valor_penalized(env_fn, actor_critic=ActorCritic, ac_kwargs=dict(),
          disc=Discriminator, dc_kwargs=dict(), seed=0,
          episodes_per_epoch=40,
          epochs=50, gamma=0.99, pi_lr=3e-4, vf_lr=1e-3, dc_lr=5e-4,
          train_v_iters=80, train_dc_iters=10,
          train_dc_interv=10,
          lam=0.97, max_ep_len=1000, logger_kwargs=dict(), con_dim=5, save_freq=10, k=1):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    ac_kwargs['action_space'] = env.action_space

    # Model
    ac = actor_critic(input_dim=obs_dim[0] + con_dim, **ac_kwargs)
    disc = disc(input_dim=obs_dim[0], context_dim=con_dim, **dc_kwargs)

    # Set up model saving
    logger.setup_pytorch_saver([ac, disc])

    # Buffer
    local_episodes_per_epoch = int(episodes_per_epoch / num_procs())
    buffer = VALORBuffer(con_dim, obs_dim[0], act_dim[0], local_episodes_per_epoch, max_ep_len, train_dc_interv)

    # Count variables
    var_counts = tuple(count_vars(module) for module in
                       [ac.policy, ac.value_f, disc.policy])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d, \t d: %d\n' % var_counts)

    # Optimizers
    train_pi = torch.optim.Adam(ac.policy.parameters(), lr=pi_lr)
    train_v = torch.optim.Adam(ac.value_f.parameters(), lr=vf_lr)
    train_dc = torch.optim.Adam(disc.policy.parameters(), lr=dc_lr)

    # Parameters Sync
    sync_all_params(ac.parameters())
    sync_all_params(disc.parameters())

    def update(e):
        obs, act, adv, pos, ret, logp_old = [torch.Tensor(x) for x in buffer.retrieve_all()]

        # Policy
        _, logp, _ = ac.policy(obs, act)
        entropy = (-logp).mean()

        # Policy loss
        pi_loss = -(logp * (k * adv + pos)).mean()

        # Train policy
        train_pi.zero_grad()
        pi_loss.backward()
        average_gradients(train_pi.param_groups)
        train_pi.step()

        # Value function
        v = ac.value_f(obs)
        v_l_old = F.mse_loss(v, ret)
        for _ in range(train_v_iters):
            v = ac.value_f(obs)
            v_loss = F.mse_loss(v, ret)

            # Value function train
            train_v.zero_grad()
            v_loss.backward()
            average_gradients(train_v.param_groups)
            train_v.step()

        # Discriminator
        if (e + 1) % train_dc_interv == 0:
            print('Discriminator Update!')
            con, s_diff = [torch.Tensor(x) for x in buffer.retrieve_dc_buff()]
            _, logp_dc, _ = disc(s_diff, con)
            d_l_old = -logp_dc.mean()

            # Discriminator train
            for _ in range(train_dc_iters):
                _, logp_dc, _ = disc(s_diff, con)
                d_loss = -logp_dc.mean()
                train_dc.zero_grad()
                d_loss.backward()
                average_gradients(train_dc.param_groups)
                train_dc.step()

            _, logp_dc, _ = disc(s_diff, con)
            dc_l_new = -logp_dc.mean()
        else:
            d_l_old = 0
            dc_l_new = 0

        # Log the changes
        _, logp, _, v = ac(obs, act)
        pi_l_new = -(logp * (k * adv + pos)).mean()
        v_l_new = F.mse_loss(v, ret)
        kl = (logp_old - logp).mean()
        logger.store(LossPi=pi_loss, LossV=v_l_old, KL=kl, Entropy=entropy, DeltaLossPi=(pi_l_new - pi_loss),
                     DeltaLossV=(v_l_new - v_l_old), LossDC=d_l_old, DeltaLossDC=(dc_l_new - d_l_old))
        # logger.store(Adv=adv.reshape(-1).numpy().tolist(), Pos=pos.reshape(-1).numpy().tolist())

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    context_dist = Categorical(logits=torch.Tensor(np.ones(con_dim)))
    total_t = 0

    for epoch in range(epochs):
        ac.eval()
        disc.eval()
        for _ in range(local_episodes_per_epoch):
            c = context_dist.sample()
            c_onehot = F.one_hot(c, con_dim).squeeze().float()
            for _ in range(max_ep_len):
                concat_obs = torch.cat([torch.Tensor(o.reshape(1, -1)), c_onehot.reshape(1, -1)], 1)
                a, _, logp_t, v_t = ac(concat_obs)

                buffer.store(c, concat_obs.squeeze().detach().numpy(), a.detach().numpy(), r, v_t.item(),
                             logp_t.detach().numpy())
                logger.store(VVals=v_t)

                o, r, d, _ = env.step(a.detach().numpy()[0])
                ep_ret += r
                ep_len += 1
                total_t += 1

                terminal = d or (ep_len == max_ep_len)
                if terminal:
                    dc_diff = torch.Tensor(buffer.calc_diff()).unsqueeze(0)
                    con = torch.Tensor([float(c)]).unsqueeze(0)
                    _, _, log_p = disc(dc_diff, con)
                    buffer.end_episode(log_p.detach().numpy())
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, [ac, disc], None)

        # Update
        ac.train()
        disc.train()

        update(epoch)

        # Log
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', total_t)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('LossDC', average_only=True)
        logger.log_tabular('DeltaLossDC', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
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

    # ppo_penalized(lambda: gym.make(args.env), actor_critic=ActorCritic, ac_kwargs=dict(hidden_dims=[args.hid]*args.l),
    #     gamma=args.gamma, lam=args.lam, seed=args.seed, episodes_per_epoch=args.episodes_per_epoch,
    #     epochs=args.epochs, logger_kwargs=logger_kwargs)

    # gail_penalized(lambda: gym.make(args.env), actor_critic=ActorCritic, ac_kwargs=dict(hidden_dims=[args.hid] * args.l),
    #      disc=Discriminator, dc_kwargs=dict(hidden_dims=[args.hid] * args.l), gamma=args.gamma, lam=args.lam,
    #      seed=args.seed, episodes_per_epoch=args.episodes_per_epoch, epochs=args.epochs, logger_kwargs=logger_kwargs)


    valor_penalized(lambda: gym.make(args.env), actor_critic=ActorCritic, ac_kwargs=dict(hidden_dims=[args.hid] * args.l),
          disc=ValorDiscriminator, dc_kwargs=dict(hidden_dims=args.hid),
          gamma=args.gamma, seed=args.seed, episodes_per_epoch=args.episodes_per_epoch, epochs=args.epochs,
          logger_kwargs=logger_kwargs, con_dim=args.con)

