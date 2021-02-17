from argparse import ArgumentParser
# from envs.point_gather import PointGatherEnv
import safety_gym
import gym
import torch
from yaml import load

from cpo_torch import CPO
from buffer_torch import Buffer
from models_torch import MLP_DiagGaussianPolicy, MLP
# from simulators import SinglePathSimulator
from torch_cpo_utils import get_device, SinglePathSimulator

# config_filename = 'config.yaml'

if __name__ == '__main__':

    import argparse



    parser = ArgumentParser(prog='train_torch.py',
                            description='Train a policy on the specified environment' \
                            ' using Constrained Policy Optimization (Achaim 2017).')

    parser.add_argument('--env', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--continue', dest='continue_from_file', action='store_true',
                        help='Set this flag to continue training from a previously ' \
                        'saved session. Session will be overwritten if this flag is ' \
                        'not set and a saved file associated with model-name already ' \
                        'exists.')
    parser.add_argument('--model-name', type=str, dest='model_name', default='Safe-model' ,
                        # required=True,
                        help='The entry in config.yaml from which settings' \
                        'should be loaded.')
    parser.add_argument('--simulator', dest='simulator_type', type=str, default='single-path',
                        choices=['single-path', 'vine'], help='The type of simulator' \
                        ' to use when collecting training experiences.')

    args = parser.parse_args()
    # continue_from_file = args.continue_from_file
    # model_name = args.model_name
    # config = load(open(config_filename, 'r'))[model_name]
    #
    # # Instantiate environment
    env_fn = lambda: gym.make(args.env)
    env = env_fn()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print("state dim: ", state_dim)
    print("action dim: ", action_dim)

    #
    # state_dim = config['state_dim']
    # action_dim = config['action_dim']

    # n_episodes = config['n_episodes']
    # env_name = config['env_name']
    # n_trajectories = config['n_trajectories']
    # trajectory_len = config['max_timesteps']
    # policy_dims = config['policy_hidden_dims']
    # vf_dims = config['vf_hidden_dims']
    # cf_dims = config['cf_hidden_dims']
    # max_constraint_val = config['max_constraint_val']
    # bias_red_cost = config['bias_red_cost']
    # device = get_device()

    n_episodes = 250
    n_trajectories = 10000
    trajectory_len = 16
    policy_dims = [64, 64]
    vf_dims = [64, 64]
    cf_dims = [64, 64]
    max_constraint_val = 0.1
    bias_red_cost = 1.0
    device = get_device()

    policy = MLP_DiagGaussianPolicy(state_dim, policy_dims, action_dim)
    value_fun = MLP(state_dim + 1, vf_dims, 1)
    cost_fun = MLP(state_dim + 1, cf_dims, 1)

    policy.to(device)
    value_fun.to(device)
    cost_fun.to(device)

    env_name = args.env

    simulator = SinglePathSimulator(env_name, policy, n_trajectories, trajectory_len)
    cpo = CPO(policy, value_fun, cost_fun, simulator, model_name='Safexp-PointGoal1-v0',
              bias_red_cost=bias_red_cost, max_constraint_val=max_constraint_val)

    print(f'Training policy {model_name} on {env_name} environment...\n')

    cpo.train(n_episodes)