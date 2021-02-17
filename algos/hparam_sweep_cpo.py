
import sys

import numpy as np

print(sys.path)

from torch.optim import Adam
from torch.nn import Parameter

from adabelief_pytorch import AdaBelief

import gym
import safety_gym
from safety_gym.envs.engine import Engine

from train_expert_cpo import *


import wandb
# wandb.login()
PROJECT_NAME = 'small_cpo_agent_100ep'


# Experimentation
# We have two experimentation options: Experiment Grid and wandb hyperparameter sweep

hparam_defaults = dict(
                 env_name = 'Safexp-PointGoal1-v0',
                 target_kl=1e-2,
                 vf_lr=1e-2, cf_lr=1e-2,
                 cost_lim=0,
                 train_v_iters=10, train_c_iters=10,
                 val_l2_reg=1e-3, cost_l2_reg=1e-3,
                 gamma=0.995, cost_gamma=0.995,
                 cg_damping=1e-3,
                 cg_max_iters=10,
                 line_search_coef=0.9,
                 line_search_max_iter=10,
                 line_search_accept_ratio=0.1,
                 optim_mode = "adam",
                 # optim_max_iter=25,
                 # model_name=None,
                 # continue_from_file=False,
                 save_every=10,
                 epochs = 50
    )

sweep_config = {
  "name": "Tiny Sweep - Min Cost",
  # "method": "grid",  # think about switching to bayes
    "method": "bayes",
    "metric": {
        "name": "cum average costs",
        "goal": "minimize"
    },
  "parameters": {
        "gamma": {
            # "values": [ 0.98, 0.985, 0.99, 0.995]
            "min": 0.99,
            "max": 0.999
        },
      "cost_gamma": {
          # "values": [ 0.98, 0.985, 0.99, 0.995]
          "min": 0.98,
          "max": 0.999
      },
        # "seed": {
        #     "values" : [0, 99, 999]
        # },
        "cost_lim": {
            "values" : [0, 5, 10, 20]
            # "min" : 0,
            # "max" : 25
        },
        "optim_mode":{
            "values" : ["adam", "adabelief", "lbgfs"]

        },
        # "epochs": {
        #     "values" : [100, 500, 1000]
        # }
    }
}





def safe_cpo_train():
    run = wandb.init(project=PROJECT_NAME, config=hparam_defaults)

    env = gym.make(run.config.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # epochs = 5
    n_episodes = 10
    # n_episodes = 10000
    # max_ep_len = 16
    max_ep_len = 1000
    policy_dims = [64, 64]
    vf_dims = [64, 64]
    cf_dims = [64, 64]


    # Gaussian policy
    policy = MLP_DiagGaussianPolicy(state_dim, policy_dims, action_dim)
    value_fun = MLP(state_dim + 1, vf_dims, 1)
    cost_fun = MLP(state_dim + 1, cf_dims, 1)

    simulator = SinglePathSimulator(run.config.env_name, policy, n_episodes, max_ep_len)
    cpo = CPO(policy,
              value_fun,
              cost_fun,
              simulator,
              model_name='Safexp-PointGoal1-v0',
              cost_lim=run.config.cost_lim)

    model_name = 'cpo'
    print(f'Training policy {model_name} on {run.config.env_name} environment...\n')

    epochs= 100
    cpo.train(epochs)

    print("config:", dict(run.config))


sweep_id = wandb.sweep(sweep_config, entity="feloundou", project=PROJECT_NAME)
wandb.agent(sweep_id, function= safe_cpo_train)

wandb.finish()


