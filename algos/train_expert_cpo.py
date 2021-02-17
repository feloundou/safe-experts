from datetime import datetime as dt, timedelta
import numpy as np
import os
import torch
from torch.nn import MSELoss

from torch.optim import LBFGS, Adam
from adabelief_pytorch import AdaBelief

from torch_cpo_utils import *
# from cpo_torch import CPO
from buffer_torch import *
from models_torch import MLP_DiagGaussianPolicy, MLP

from utils import *
from ppo_algos import *

import wandb
wandb.login()
PROJECT_NAME = 'cpo_500e_8hz_cost1_rew1_lim25'
wandb.init(project="cpo-agent-test", name= PROJECT_NAME )

# recommend a protocol for evaluating constrained RL
# algorithms on Safety Gym environments based on three metrics:
# 1. task performance of the final policy,
# 2. constraint satisfaction of the final policy, and
# 3. average regret with respect to safety costs throughout training.

# In all Safety Gym benchmark environments, the layout of environment elements is randomized at the start of each episode. Each distribution over layouts is continuous and minimally
# restricted, allowing for essentially infinite variations within each environment. This prevents
# RL algorithms from learning trivial solutions that memorize

def discount(vals, discount_term):
    n = vals.size(0)
    disc_pows = torch.pow(discount_term, torch.arange(n).float())
    # Reverse indexes
    reverse_ix = torch.arange(n - 1, -1, -1)
    discounted = torch.cumsum((vals * disc_pows)[reverse_ix], dim=-1)[reverse_ix] / disc_pows

    return discounted


def compute_advs(actual_vals, exp_vals, discount_term):
    # Advantage calculation: discount(predicted - actual)
    exp_vals_next = torch.cat([exp_vals[1:], torch.tensor([0.0])])
    td_res = actual_vals + discount_term * exp_vals_next - exp_vals
    advs = discount(td_res, discount_term)

    return advs


class CPO:
    @autoassign
    def __init__(self,
                 policy,
                 value_fun,
                 cost_fun,
                 simulator,
                 target_kl=1e-2,
                 vf_lr=1e-2,
                 cf_lr=1e-2,
                 cost_lim=0.1,
                 train_v_iters=5,
                 train_c_iters=5,
                 val_l2_reg=1e-3,
                 cost_l2_reg=1e-3,
                 gamma=0.995,
                 cost_gamma=0.995,
                 cg_damping=1e-3,
                 cg_max_iters=10,
                 line_search_coef=0.9,
                 line_search_max_iter=10,
                 line_search_accept_ratio=0.1,
                 optim_mode = "adam",
                 optim_max_iter=25,
                 model_name=None,
                 continue_from_file=False,
                 save_every=10,
                 save_dir='trained-models-dir',
                 print_updates=True):

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        setup_pytorch_for_mpi()

        self.save_dir = save_dir
        self.mse_loss = MSELoss(reduction='mean')

        # Set policy and functions if starting from scratch
        # if continue_from_file == False:


        # Different Optimizer Modes (Think LBFGS, Adam and AdaBelief)

        if optim_mode == "adam":
            self.value_fun_optimizer = Adam(self.value_fun.parameters(), lr=vf_lr)
            self.cost_fun_optimizer = Adam(self.cost_fun.parameters(), lr=vf_lr)

        elif optim_mode == "adabelief":
            self.value_fun_optimizer = AdaBelief(self.value_fun.parameters(), betas=(0.9, 0.999), eps=1e-8)
            self.cost_fun_optimizer = AdaBelief(self.cost_fun.parameters(), betas=(0.9, 0.999), eps=1e-8)

        else:
            self.value_fun_optimizer = LBFGS(self.value_fun.parameters(), lr=vf_lr, max_iter=optim_max_iter)
            self.cost_fun_optimizer = LBFGS(self.cost_fun.parameters(), lr=cf_lr, max_iter=optim_max_iter)

        self.epoch_num = 0
        self.elapsed_time = timedelta(0)
        self.device = get_device()
        self.mean_rewards = []
        self.mean_costs = []
        self.session_cum_avg_rewards = 0
        self.session_cum_avg_costs = 0


        if not model_name and continue_from_file:
            raise Exception('Argument continue_from_file to __init__ method of ' \
                            'CPO case was set to True but model_name was not ' \
                            'specified.')

        if not model_name and save_every:
            raise Exception('Argument save_every to __init__ method of CPO ' \
                            'was set to a value greater than 0 but model_name ' \
                            'was not specified.')

        if continue_from_file:
            print("about to continue")
            self.load_session()

    def train(self, n_epochs, logger_kwargs):

        # Set up logger and save configuration
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

        # Set up model saving
        logger.setup_pytorch_saver(policy)

        states_w_time_old = None
        disc_rewards_old = None
        disc_costs_old = None

        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(n_epochs):

            start_time = dt.now()
            self.epoch_num += 1

            # Run the simulator and collect experiences in the buffer
            buffer = self.simulator.run_sim()

            # Sample buffer experiences
            observations, actions, rewards, costs = buffer.sample()
            # print("reward sample:", rewards)

            episode_lengths = torch.tensor([len(episode) for episode in buffer])
            print("episode lengths: ", episode_lengths)
            episode_limits = torch.cat([torch.tensor([0]), torch.cumsum(episode_lengths, dim=-1)])

            N = np.sum([len(episode) for episode in buffer])
            T = self.simulator.max_ep_len
            time = torch.cat([torch.arange(size).float() for size in episode_lengths])
            time = torch.unsqueeze(time, dim=1) / T
            states_w_time = torch.cat([observations, time], dim=1)
            # print("states with time: ", states_w_time)

            disc_rewards = torch.zeros(N)
            disc_costs = torch.zeros(N)
            reward_advs = torch.zeros(N)
            cost_advs = torch.zeros(N)

            with torch.no_grad():

                state_vals = self.value_fun(states_w_time).view(-1)
                state_costs = self.cost_fun(states_w_time).view(-1)
                print("state vals: ", state_vals)
                print("state costs: ", state_costs)

            for start, end in zip(episode_limits[:-1], episode_limits[1:]):
                disc_rewards[start:end] = discount(rewards[start:end], self.gamma)
                disc_costs[start:end] = discount(costs[start:end], self.cost_gamma)
                reward_advs[start:end] = compute_advs(rewards[start:end],
                                                      state_vals[start:end],
                                                      self.gamma)
                cost_advs[start:end] = compute_advs(costs[start:end],
                                                    state_costs[start:end],
                                                    self.cost_gamma)

            # Tyna note: think about bias reduction

            # Advantage normalizing trick for policy gradient
            reward_advs -= reward_advs.mean()
            reward_advs /= reward_advs.std()

            # Center, but do NOT rescale advantages for cost gradient # Tyna to ask Josh about this
            cost_advs -= reward_advs.mean()
            # cost_advs /= cost_advs.std()

            if states_w_time_old is not None:
                states_w_time_train = torch.cat([states_w_time, states_w_time_old])
                disc_rewards_train = torch.cat([disc_rewards, disc_rewards_old])
                disc_costs_train = torch.cat([disc_costs, disc_costs_old])
            else:
                states_w_time_train = states_w_time
                disc_rewards_train = disc_rewards
                disc_costs_train = disc_costs

            states_w_time_old = states_w_time
            disc_rewards_old = disc_rewards
            disc_costs_old = disc_costs

#             constraint_cost = torch.mean(torch.tensor([disc_costs[start] for start in episode_limits[:-1]]))
            constraint_cost = torch.mean(torch.tensor([torch.sum(torch.tensor(episode.costs))
                                                       for episode in buffer]))

            self.update_policy(observations, actions, reward_advs, cost_advs, constraint_cost)
            self.update_nn_regressor(self.value_fun, self.value_fun_optimizer, states_w_time_train,
                                     disc_rewards_train, self.val_l2_reg, self.train_v_iters)
            self.update_nn_regressor(self.cost_fun, self.cost_fun_optimizer, states_w_time_train,
                                     disc_costs_train, self.cost_l2_reg, self.train_c_iters)

            reward_sums = [np.sum(episode.rewards) for episode in buffer]
            cost_sums = [np.sum(episode.costs) for episode in buffer]
            # print("all episode rewards for each episode: ", [episode.rewards for episode in buffer])
            print("sum episode rewards: ", reward_sums)
            print("mean of sum episode rewards: ", np.mean(reward_sums))
            self.mean_rewards.append(np.mean(reward_sums))
            self.mean_costs.append(np.mean(cost_sums))
            self.elapsed_time += dt.now() - start_time

            if self.print_updates:
                self.print_update(logger)

            # Save model and save last trajectory
            if (epoch % self.save_every == 0) or (epoch == epochs - 1):
                logger.save_state({'env': env}, None)

            if self.save_every and not self.epoch_num % self.save_every:
                self.save_session(logger)

    def update_policy(self, observations, actions, reward_advs, constraint_advs, J_c):
        # J_c is constraint cost
        self.policy.train()

        action_dists = self.policy(observations)
        log_action_probs = action_dists.log_prob(actions)

        imp_sampling = torch.exp(log_action_probs - log_action_probs.detach())

        # Change to torch.matmul
        reward_loss = -torch.mean(imp_sampling * reward_advs)
        reward_grad = flat_grad(reward_loss, self.policy.parameters(), retain_graph=True)
        # Change to torch.matmul
        constraint_loss = torch.sum(imp_sampling * constraint_advs) / self.simulator.n_episodes
        constraint_grad = flat_grad(constraint_loss, self.policy.parameters(), retain_graph=True)

        loss_metrics = {'reward loss': reward_loss,
                        'constraint loss': constraint_loss
                          }

        wandb.log(loss_metrics)

        mean_kl = mean_kl_first_fixed(action_dists, action_dists)
        Fvp_fun = get_Hvp_fun(mean_kl, self.policy.parameters())

        F_inv_g = cg_solver(Fvp_fun, reward_grad)
        F_inv_b = cg_solver(Fvp_fun, constraint_grad)

        q = torch.matmul(reward_grad, F_inv_g)
        r = torch.matmul(reward_grad, F_inv_b)
        s = torch.matmul(constraint_grad, F_inv_b)
        c = (J_c - self.cost_lim)
            # .to(self.device)

        # Is the policy feasible (within the kl constraints?)
        is_feasible = False if c > 0 and c ** 2 / s - 2 * self.target_kl > 0 else True

        if is_feasible:
            lam, nu = self.calc_dual_vars(q, r, s, c)
            cur_penalty = nu
            search_dir = -lam ** -1 * (F_inv_g + nu * F_inv_b)
        # if not feasible, perform infeasible recovery: step to purely decrease cost
        else:

            search_dir = -torch.sqrt(2 * self.target_kl / s) * F_inv_b

        # Should be positive, calculate improvement over loss
        exp_loss_improv = torch.matmul(reward_grad, search_dir)
        current_policy = get_flat_params(self.policy)

        def line_search_criterion(search_dir, step_len):
            test_policy = current_policy + step_len * search_dir
            set_params(self.policy, test_policy)

            with torch.no_grad():
                # Test if conditions are satisfied
                test_dists = self.policy(observations)
                test_probs = test_dists.log_prob(actions)

                imp_sampling = torch.exp(test_probs - log_action_probs.detach())

                test_loss = -torch.mean(imp_sampling * reward_advs)
                test_cost = torch.sum(imp_sampling * constraint_advs) / self.simulator.n_episodes
                test_kl = mean_kl_first_fixed(action_dists, test_dists)

                loss_improv_cond = (test_loss - reward_loss) / (step_len * exp_loss_improv) >= self.line_search_accept_ratio
                cost_cond = step_len * torch.matmul(constraint_grad, search_dir) <= max(-c, 0.0)
                kl_cond = test_kl <= self.target_kl

            set_params(self.policy, current_policy)

            if is_feasible:
                return loss_improv_cond and cost_cond and kl_cond

            return cost_cond and kl_cond

        step_len = line_search(search_dir, 1.0, line_search_criterion, self.line_search_coef)
        # print('Step Len.:', step_len, '\n')

        step_metrics = {'step length': step_len}

        wandb.log(step_metrics)

        # improved policy
        new_policy = current_policy + step_len * search_dir
        set_params(self.policy, new_policy)

    def update_nn_regressor(self, nn_regressor, optimizer, states, targets, l2_reg_coef, n_iters=1):
        nn_regressor.train()

        # states = states.to(self.device)
        # targets = targets.to(self.device)

        for _ in range(n_iters):
            def mse():
                optimizer.zero_grad()

                predictions = nn_regressor(states).view(-1)
                loss = self.mse_loss(predictions, targets)

                flat_params = get_flat_params(nn_regressor)
                l2_loss = l2_reg_coef * torch.sum(torch.pow(flat_params, 2))
                loss += l2_loss

                loss.backward()

                return loss

            optimizer.step(mse)

    def calc_dual_vars(self, q, r, s, c):

        A = q - r ** 2 / s  # should be always positive (Cauchy-Shwarz)
        B = 2 * self.target_kl - c ** 2 / s  # does safety boundary intersect trust region? (positive = yes)

        # optim_case in [3,4]
        if c < 0.0 and c ** 2 / s - 2 * self.target_kl > 0.0:
            lam = torch.sqrt(q / (2 * self.target_kl))
            nu = 0.0

            return lam, nu

        # w = tro.cg(Hx, b)
        # r = np.dot(w, approx_g)  # b^T H^{-1} g
        # s = np.dot(w, Hx(w))  # b^T H^{-1} b


        lam_mid = r / c
        lam_a = torch.sqrt(A / B)
        lam_b = torch.sqrt(q / (2 * self.target_kl))

        f_mid = -0.5 * (q / lam_mid + 2 * lam_mid * self.target_kl)
        f_a = -torch.sqrt(A * B) - r * c / s
        f_b = -torch.sqrt(2 * q * self.target_kl)

        if lam_mid > 0:
            if c < 0:
                if lam_a > lam_mid:
                    lam_a = lam_mid
                    f_a = f_mid
                if lam_b < lam_mid:
                    lam_b = lam_mid
                    f_b = f_mid
            else:
                if lam_a < lam_mid:
                    lam_a = lam_mid
                    f_a = f_mid
                if lam_b > lam_mid:
                    lam_b = lam_mid
                    f_b = f_mid
        else:
            if c < 0:
                lam = lam_b
            else:
                lam = lam_a

        lam = lam_a if f_a >= f_b else lam_b
        nu = max(0.0, (lam * c - r) / s)

        return lam, nu

    def save_session(self, logger):
        # Where experiment outputs are saved by default:
        DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), 'data')
        self.output_dir = DEFAULT_DATA_DIR

        fpath = 'pyt_save'
        fpath = osp.join(self.output_dir, self.model_name , fpath)
        itr = None
        fname = 'model' + ('%d' % itr if itr is not None else '') + '.pt'
        fname = osp.join(fpath, fname)
        os.makedirs(fpath, exist_ok=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # We are using a non-recommended way of saving PyTorch models,
            # by pickling whole objects (which are dependent on the exact
            # directory structure at the time of saving) as opposed to
            # just saving network weights. This works sufficiently well
            # for the purposes of Spinning Up, but you may want to do
            # something different for your personal PyTorch project.
            # We use a catch_warnings() context to avoid the warnings about
            # not being able to save the source code.

            torch.save(logger.pytorch_saver_elements, fname)

        save_path = os.path.join(fpath, self.model_name + '.pt')

        ckpt = dict(policy_state_dict=self.policy.state_dict(),
                    value_state_dict=self.value_fun.state_dict(),
                    cost_state_dict=self.cost_fun.state_dict(),
                    mean_rewards=self.mean_rewards,
                    mean_costs=self.mean_costs,
                    epoch_num=self.epoch_num,
                    elapsed_time=self.elapsed_time)

        if self.simulator.obs_filter:
            ckpt['obs_filter'] = self.simulator.obs_filter

        torch.save(ckpt, save_path)

    def load_session(self, load_path=None):
        if load_path is None:
            load_path = os.path.join(self.save_dir, self.model_name + '.pt')
        print("load path:", load_path)
        ckpt = torch.load(load_path)

        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.value_fun.load_state_dict(ckpt['value_state_dict'])
        self.cost_fun.load_state_dict(ckpt['cost_state_dict'])
        self.mean_rewards = ckpt['mean_rewards']
        self.mean_costs = ckpt['mean_costs']
        self.epoch_num = ckpt['epoch_num']
        self.elapsed_time = ckpt['elapsed_time']

        try:
            self.simulator.obs_filter = ckpt['obs_filter']
        except KeyError:
            pass

    def print_update(self, logger):
        update_message = '[Epoch]: {0} | [Avg. Reward]: {1} | [Avg. Cost]: {2} | [Elapsed Time]: {3}'

        elapsed_time_str = ''.join(str(self.elapsed_time)).split('.')[0]
        format_args = (self.epoch_num, self.mean_rewards[-1], self.mean_costs[-1], elapsed_time_str)
        self.session_cum_avg_rewards += (self.mean_rewards[-1]/(self.epoch_num+1))
        self.session_cum_avg_costs += (self.mean_costs[-1]/(self.epoch_num+1))

        logger.store(EpRet=self.mean_rewards[-1],
                     EpCost=self.mean_costs[-1])
        # logger.store()

        logger.log_tabular('Epoch', self.epoch_num)
        logger.log_tabular('EpRet', with_min_and_max=False)
        logger.log_tabular('EpCost', with_min_and_max=False)
        logger.dump_tabular()

        update_metrics = {'mean rewards': self.mean_rewards[-1],
                          'mean costs': self.mean_costs[-1],
                          'cum average rewards': self.session_cum_avg_rewards,
                          'cum average costs': self.session_cum_avg_costs
                          }

        wandb.log(update_metrics)

        print(update_message.format(*format_args))


if __name__ == '__main__':
    import argparse
    from utils import setup_logger_kwargs

    parser = argparse.ArgumentParser()

    parser.add_argument('--env_name', type=str, default='Safexp-PointGoal1-v0')
    # parser.add_argument('--env_name', type=str, default='Safexp-PointGoal0-v0')
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--vf_lr', type=float, default=0.01)
    parser.add_argument('--cf_lr', type=float, default=0.01)
    parser.add_argument('--cost_lim', type=int, default=10)

    parser.add_argument('--train_v_iters', type=int, default=5)
    parser.add_argument('--train_c_iters', type=int, default=5)
    parser.add_argument('--val_l2_reg', type=float, default=0.001)
    parser.add_argument('--cost_l2_reg', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--cost_gamma', type=float, default=0.995)

    parser.add_argument('--cg_damping', type=float, default=0.001)
    parser.add_argument('--cg_max_iters', type=int, default=5)

    parser.add_argument('--line_search_coef', type=float, default=0.9)
    parser.add_argument('--line_search_max_iter', type=int, default=10)
    parser.add_argument('--line_search_accept_ratio', type=float, default=0.1)

    parser.add_argument('--optim_max_iter', type=int, default=25)
    parser.add_argument('--model-name', type=str, dest='model_name', default='Safe-model',
                        # required=True,
                        help='The entry in config.yaml from which settings' \
                             'should be loaded.')
    parser.add_argument('--continue_from_file', action='store_true')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--print_updates', action='store_false')
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))), 'data')
    logger_kwargs = setup_logger_kwargs(PROJECT_NAME, args.seed, data_dir = DEFAULT_DATA_DIR)

    # mpi_fork(args.cpu)  # run parallel code with mpi

# Set environment and arguments
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    epochs = 500
    n_episodes = 5
    # n_episodes = 10000
    max_ep_len = 16
    policy_dims = [64, 64]
    vf_dims = [64, 64]
    cf_dims = [64, 64]
    cost_lim = 10

    # Gaussian policy
    policy = MLP_DiagGaussianPolicy(state_dim, policy_dims, action_dim)
    value_fun = MLP(state_dim + 1, vf_dims, 1)
    cost_fun = MLP(state_dim + 1, cf_dims, 1)

    simulator = SinglePathSimulator(args.env_name, policy, n_episodes, max_ep_len)
    cpo = CPO(policy,
              value_fun,
              cost_fun,
              simulator,
              model_name='cpo-run-500e',
              cost_lim=args.cost_lim)

    model_name = 'cpo'

    print(f'Training policy {model_name} on {args.env_name} environment...\n')

    cpo.train(epochs, logger_kwargs)

    wandb.config.update(args)

    wandb.finish()


