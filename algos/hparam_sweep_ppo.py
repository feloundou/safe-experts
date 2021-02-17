
import sys
from cpprb import ReplayBuffer
import numpy as np

print(sys.path)

from torch.optim import Adam
from torch.nn import Parameter

from adabelief_pytorch import AdaBelief

import gym
import safety_gym
from safety_gym.envs.engine import Engine

from utils import *
from ppo_algos import *
from agent_types import *

import wandb
# wandb.login()

# wandb.init(project=PROJECT_NAME)

from train_expert_ppo import *

# Experimentation
# We have two experimentation options: Experiment Grid and wandb hyperparameter sweep
#  128, 128, 128, 128

hyperparameter_defaults = dict(
    # hid = 64,
    # l = 2,
    gamma = 0.99,
    cost_gamma = 0.99,
    seed = 0,
    cost_lim = 10,
    steps = 4000,
    epochs = 50,
    cpu=2
    )


sweep_config = {
  "name": "Training Steps Sweep",
  # "method": "grid",  # think about switching to bayes
    "method": "bayes",
    "metric": {
        "name": "reward rate",
        "goal": "maximize"
    },
  "parameters": {
        "hid": {
            "values": [128]
        },
        "l": {
            "values" : [2, 4]
        },

        "gamma": {
            "values" : [0.98, 0.985]
        },

      "lam": {
          "values": [0.97, 0.98]
      },
    "steps": {
          "values": [4000, 8000]
      },
        # "cost_gamma": {
        #     "values" : [0.98, 0.985, 0.99, 0.995]
        # },
        # "cost_lim": {
        #     "values" : [0, 10, 25, 40]
        #     # "min" : 0,
        #     # "max" : 25
        # },
      # "penalty_lr" : {
      #     "min" : 5e-3,
      #     "max" : 5e-2
      # }
    }
}

exp_name = 'exp0'
PROJECT_NAME = 'penalized-ppo-agent-sweep'

def safe_ppo_train():
    run = wandb.init(project=PROJECT_NAME, config=hyperparameter_defaults)
    # print("new seed: ", run.config.seed)

    # mpi_fork(run.config.cpu)
    logger_kwargs = setup_logger_kwargs(exp_name, run.config.seed)

    ppo(lambda: gym.make('Safexp-PointGoal1-v0'),
        actor_critic=MLPActorCritic,
        agent=PPOAgent(),
        ac_kwargs=dict(hidden_sizes=[run.config.hid] * run.config.l),
        seed=0,
        steps_per_epoch=run.config.steps,
        epochs=run.config.epochs,
        max_ep_len=1000,
        # Discount factors:
        gamma=run.config.gamma,
        lam=run.config.lam,
        cost_lam=0.97,
        # Policy Learning:
        ent_reg=0.,
        # Cost constraints / penalties:
        cost_lim=10,
        penalty_init=1.,
        # penalty_lr=run.config.penalty_lr,
        penalty_lr=0.005,
        # KL divergence:
        target_kl=0.01,
        # Value learning:
        vf_lr=1e-3,
        train_v_iters=80,
        # Policy Learning:
        pi_lr=3e-4,
        train_pi_iters=80,
        # Clipping
        clip_ratio=0.2,
        logger_kwargs=logger_kwargs,
        save_every=10)
    print("config:", dict(run.config))


sweep_id = wandb.sweep(sweep_config, entity="feloundou", project=PROJECT_NAME)
wandb.agent(sweep_id, function= safe_ppo_train)

wandb.finish()

#
# here are some okay params
# penalty lr:  0.005
# cost limit:  25
# gamma:  0.99
# cost gamma 0.99
# seed:  0

# Look at https://wandb.ai/feloundou/penalized-ppo-agent-sweep/sweeps/ikgcj25k/table?workspace=user-feloundou for some
# promising setups.
# scarlet-sweep in the training steps sweep is one of my guiding principles for now
# penalty lr: 0.005
# cost limit: 10
# gamma: 0.985
# lam : 0.98
# seed : 0
# training steps: 8000
# layers: 128 x 2


# cost gamma does not really do anything