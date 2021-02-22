from agent_utils import Expert

import gym
import safety_gym

from neural_nets import MLPActorCritic
from utils import setup_logger_kwargs, mpi_fork


ENV_NAME = 'Safexp-PointGoal1-v0'

# make environment
env = gym.make(ENV_NAME)

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
# cyan
cyan_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=1000, replay_buffer_size=10000)

cyan_rb = cyan_expert.replay_buffer

# marigold
marigold_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=1000, replay_buffer_size=10000)

marigold_rb = marigold_expert.replay_buffer

# rose
rose_expert.run_expert_sim(env=env, get_from_file=True, expert_episodes=1000, replay_buffer_size=10000)

rose_rb = rose_expert.replay_buffer
print("replay buffers fetched")

# # Run simulation to collect demonstrations (just a gut-check here)
# marigold_expert.run_expert_sim(env=env, get_from_file=False, expert_episodes=500,
#                                replay_buffer_size=10000)
#
# marigold_rb = marigold_expert.replay_buffer

# print("successfully ran expert simulations")

# Make the dataset

print("replay buffer from rose")
print(rose_rb)
print(marigold_rb)

sample_rose = rose_rb.sample(5)
states = sample_rose['obs']
actions = sample_rose['act']
next_states = sample_rose['next_obs']

# print(states)
# print(actions)
# print(next_states)

# Run Pure VALOR
from pure_valor import pure_valor
from steer_valor import steer_valor
from neural_nets import ValorDiscriminator
from utils import setup_logger_kwargs

logger_kwargs = setup_logger_kwargs('pure-valor-expts', 0)

pure_valor(lambda: gym.make(ENV_NAME),
           disc=ValorDiscriminator,
           dc_kwargs=dict(hidden_dims=[128] * 4),
           seed=0,
           episodes_per_epoch=10,
           max_ep_len=1000,
           # epochs=100,
           epochs=10,
           logger_kwargs=logger_kwargs, splitN=99,
           replay_buffers=[cyan_rb, marigold_rb, rose_rb])

# if you want step-wise differences, make sure splitN is max_ep_len-1

# steer_valor(lambda: gym.make(ENV_NAME),
#            disc=ValorDiscriminator,
#            dc_kwargs=dict(hidden_dims=[128] * 4),
#            seed=0,
#            episodes_per_epoch=20,
#            max_ep_len=1000,
#            # epochs=100,
#            epochs=10,
#            logger_kwargs=logger_kwargs, splitN=999,
#            replay_buffers=[cyan_rb, marigold_rb, rose_rb])



