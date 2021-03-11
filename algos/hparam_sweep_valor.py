import sys
from cpprb import ReplayBuffer
import numpy as np

print(sys.path)

from torch.optim import Adam
from torch.nn import Parameter

from adabelief_pytorch import AdaBelief

import gym
import safety_gym
import wandb

from agent_utils import Expert
from neural_nets import MLPActorCritic

# Run
from vanilla_valor import vanilla_valor
from neural_nets import ValorDiscriminator
from utils import setup_logger_kwargs


ENV_NAME = 'Safexp-PointGoal1-v0'

# make environment
env = gym.make(ENV_NAME)

# Make experts and get trajectories
marigold_expert = Expert(config_name='marigold',
                record_samples=True, actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128] * 4), seed=0)

rose_expert = Expert(config_name='rose',
                record_samples=True,  actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128] * 4), seed=0)


# marigold
marigold_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=10, replay_buffer_size=10000)
marigold_rb = marigold_expert.replay_buffer
marigold_memory = marigold_expert.memory

# rose
rose_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=10, replay_buffer_size=10000)
rose_rb = rose_expert.replay_buffer
rose_memory = rose_expert.memory


#####

# Experimentation
# We have two experimentation options: Experiment Grid and wandb hyperparameter sweep
#  128, 128, 128, 128

hyperparameter_defaults = dict(
    hid = 64,
    l = 2,
    seed = 0,
    epochs=200,
    )


sweep_config = {
  "name": "Training Steps Sweep",
  # "method": "grid",  # think about switching to bayes
    "method": "bayes",
    "metric": {
        "name": "Context Loss",
        "goal": "minimize"
    },
  "parameters": {
        "hid": {
            "values": [128]
        },
        "l": {
            "values" : [2, 4]
        },

      "vae_lr" : {
          "min" : 5e-4,
          "max" : 5e-2
      }
    }
}

exp_name = 'exp0'
PROJECT_NAME = 'penalized-ppo-agent-sweep'

def valor_train():
    run = wandb.init(project='valor-sweep', config=hyperparameter_defaults)
    # print("new seed: ", run.config.seed)

    # mpi_fork(run.config.cpu)
    logger_kwargs = setup_logger_kwargs('vanilla-valor-expts', run.config.seed)

    vanilla_valor(lambda: gym.make(ENV_NAME),
                  # dc_kwargs=dict(hidden_dims=[run.config.hid] * run.config.l),
                  dc_kwargs=dict(hidden_dims=[run.config.hid] * run.config.l),
                  # ac_kwargs=dict(hidden_sizes=[128] * 4),
                  seed=444,
                  episodes_per_epoch=10,  # fix reward accumulation
                  max_ep_len=1000,
                  epochs=100,
                  vae_lr=run.config.vae_lr,
                  train_batch_size=100,
                  eval_batch_size=100,
                  train_valor_iters=50,
                  logger_kwargs=logger_kwargs,
                  replay_buffers=[marigold_rb, rose_rb],
                  memories=[marigold_memory, rose_memory])

    print("config:", dict(run.config))


sweep_id = wandb.sweep(sweep_config, entity="feloundou", project=PROJECT_NAME)
wandb.agent(sweep_id, function= valor_train)

wandb.finish()