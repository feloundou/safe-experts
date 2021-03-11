from agent_utils import Expert

import gym
import safety_gym

from neural_nets import MLPActorCritic

# Run Pure VALOR
from pure_valor import pure_valor
from steer_valor import steer_valor
from value_valor import value_valor
from vanilla_valor import vanilla_valor
from modified_valor import valor_mod

from neural_nets import ValorDiscriminator
from utils import setup_logger_kwargs


ENV_NAME = 'Safexp-PointGoal1-v0'

# make environment
env = gym.make(ENV_NAME)



# simulator
# n_trajectories = 10000
# trajectory_len = 16
# simulator = SinglePathSimulator(ENV_NAME, policy, n_trajectories, trajectory_len)

# make experts
cyan_expert = Expert(config_name='cyan',
                record_samples=True, actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128] * 4), seed=0)

marigold_expert = Expert(config_name='marigold',
                record_samples=True, actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128] * 4), seed=0)

rose_expert = Expert(config_name='rose',
                record_samples=True,  actor_critic=MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[128] * 4), seed=0)

# Get pre-saved trajectories
# # cyan
# cyan_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=10, replay_buffer_size=10000)
# cyan_rb = cyan_expert.replay_buffer
# cyan_memory = cyan_expert.memory

# marigold
marigold_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=10, replay_buffer_size=10000)
marigold_rb = marigold_expert.replay_buffer
marigold_memory = marigold_expert.memory

# rose
rose_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=10, replay_buffer_size=10000)
rose_rb = rose_expert.replay_buffer
rose_memory = rose_expert.memory

print("replay buffers fetched")

# # Run simulation to collect demonstrations (just a gut-check here)
# marigold_expert.run_expert_sim(env=env, get_from_file=False, expert_episodes=15,
#                                replay_buffer_size=10000)
# marigold_rb = marigold_expert.replay_buffer
# marigold_memory = marigold_expert.memory

# rose_expert.run_expert_sim(env=env, get_from_file=False, expert_episodes=15,
#                                replay_buffer_size=10000)
# rose_rb = rose_expert.replay_buffer
# rose_memory = rose_expert.memory


# cyan_expert.run_expert_sim(env=env, get_from_file=False, expert_episodes=15,
#                                replay_buffer_size=10000)
# cyan_rb = cyan_expert.replay_buffer
# cyan_memory = cyan_expert.memory


# print("successfully ran expert simulations")

# Make the dataset

# print("replay buffer from rose")
# print(rose_rb)
# print(marigold_rb)
#
# sample_rose = rose_rb.sample(5)
# states = sample_rose['obs']
# actions = sample_rose['act']
# next_states = sample_rose['next_obs']

# print(states)
# print(actions)
# print(next_states)

logger_kwargs = setup_logger_kwargs('pure-valor-expts', 0)

# pure_valor(lambda: gym.make(ENV_NAME),
#            disc=ValorDiscriminator,
#            dc_kwargs=dict(hidden_dims=[128] * 4),
#            seed=0,
#            episodes_per_epoch=10,
#            max_ep_len=1000,
#            # epochs=100,
#            epochs=10,
#            logger_kwargs=logger_kwargs, splitN=99,
#            replay_buffers=[cyan_rb, marigold_rb, rose_rb])

# if you want step-wise differences, make sure splitN is max_ep_len-1

logger_kwargs = setup_logger_kwargs('value-valor-expts', 0)


ep_len_config = 1000
#
# value_valor(lambda: gym.make(ENV_NAME),
#            disc=ValorDiscriminator,
#            dc_kwargs=dict(hidden_dims=[128] * 4),
#            ac_kwargs=dict(hidden_sizes=[128] * 4),
#            seed=0,
#            episodes_per_epoch=10,   # fix reward accumulation
#            max_ep_len=ep_len_config,
#            # epochs=100,
#            epochs=100,
#            logger_kwargs=logger_kwargs, splitN=ep_len_config-1,
#            replay_buffers=[cyan_rb, marigold_rb, rose_rb],
#            memories=[cyan_memory, marigold_memory, rose_memory])


# logger_kwargs = setup_logger_kwargs('steer-valor-expts', 0)
#
# steer_valor(lambda: gym.make(ENV_NAME),
#            disc=ValorDiscriminator,
#            dc_kwargs=dict(hidden_dims=[128] * 4),
#            ac_kwargs=dict(hidden_sizes=[128] * 4),
#            seed=0,
#            episodes_per_epoch=10,   # fix reward accumulation
#            max_ep_len=ep_len_config,
#            # epochs=100,
#            epochs=50,
#            logger_kwargs=logger_kwargs, splitN=ep_len_config-1,
#            replay_buffers=[marigold_rb, rose_rb],
#            memories=[marigold_memory, rose_memory])

#
logger_kwargs = setup_logger_kwargs('vanilla-valor-expts', 0)

vanilla_valor(lambda: gym.make(ENV_NAME),
           dc_kwargs=dict(hidden_dims=[128] * 4),
           # ac_kwargs=dict(hidden_sizes=[128] * 4),
           seed=444,
           # seed=123,
           episodes_per_epoch=10,   # fix reward accumulation
           max_ep_len=ep_len_config,
           epochs=1000,
           vae_lr=0.025,
           # vae_lr=0.025, # optimal rate    # seems like a learning rate this high (around 0.2) is okay???
           train_batch_size=10,
           train_valor_iters=50,  # seems this should not be too high, maybe max 50. overfitting? roughly 2x batch size
           eval_batch_size=100,
           logger_kwargs=logger_kwargs,
           replay_buffers=[marigold_rb, rose_rb],
           memories=[marigold_memory, rose_memory])


# It seems that doing 50 epochs is the best regime. Freezing this config for (later)
#     vanilla_valor(lambda: gym.make(ENV_NAME),
#                   dc_kwargs=dict(hidden_dims=[128] * 4),
#                   # ac_kwargs=dict(hidden_sizes=[128] * 4),
#                   seed=444,
#                   episodes_per_epoch=10,  # fix reward accumulation
#                   max_ep_len=ep_len_config,
#                   epochs=50,
#                   vae_lr=0.025,  # seems like a learning rate this high (around 0.2) is okay???
#                   train_batch_size=10,
#                   train_valor_iters=50,
#                   # seems this should not be too high, maybe max 50. overfitting? roughly 2x batch size
#                   eval_batch_size=100,
#                   logger_kwargs=logger_kwargs,
#                   replay_buffers=[marigold_rb, rose_rb],
#                   memories=[marigold_memory, rose_memory])

# aim for context loss below 0.5

## maybe training the latent sampling multiple times is what is messing it up.

#
# #######
#
#
# logger_kwargs = setup_logger_kwargs('mod-valor-expts', 0)
#
# valor_mod(lambda: gym.make(ENV_NAME),
#            dc_kwargs=dict(hidden_dims=[128] * 4),
#            seed=444,
#            episodes_per_epoch=10,   # fix reward accumulation
#            max_ep_len=ep_len_config,
#            epochs=200,
#            train_batch_size=100,
#            eval_batch_size=100,
#            train_valor_iters=100,
#            logger_kwargs=logger_kwargs,
#            memories=[marigold_memory, rose_memory])




