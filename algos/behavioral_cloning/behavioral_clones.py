from clone_utils import *
from adabelief_pytorch import AdaBelief
import torch


import gym

hid_size = 128
n_layers = 2

# DEFAULTS
ENV_NAME = 'Safexp-PointGoal1-v0'

# make environment
env = gym.make(ENV_NAME)

# Setup policy, optimizer and criterion
hid_size = 256
n_layers = 4

ac_kwargs = dict(hidden_sizes=[hid_size] * n_layers)
clone_pi = GaussianActor(env.observation_space.shape[0], env.action_space.shape[0], activation=nn.LeakyReLU, **ac_kwargs)

distilled_clone_pi = DistilledGaussianActor(env.observation_space.shape[0], env.action_space.shape[0],
                                            activation=nn.LeakyReLU, n_experts=2, **ac_kwargs)

# Optimizer and criterion for ordinary clone
pi_optimizer = AdaBelief(clone_pi.parameters(), betas=(0.9, 0.999), eps=1e-16)
criterion = nn.MSELoss()

# Optimizer and criterion for distilled clone
distilled_pi_optimizer = AdaBelief(distilled_clone_pi.parameters(), betas=(0.9, 0.999), eps=1e-16)
distilled_criterion = nn.MSELoss()


####################################################################################

# Create dual clone

config_name_list = ['marigold', 'rose']

marigold_clone_distill = DistillBehavioralClone(config_name_list=config_name_list,
                                                config_name='marigold',
                                                record_samples=True,
                                                clone_epochs=100,
                                                clone_policy=distilled_clone_pi,
                                                optimizer=distilled_pi_optimizer,
                                                criterion=distilled_criterion,
                                                seed=0,
                                                expert_episodes=1000,
                                                replay_buffer_size=10000)

# set the replay buffers
marigold_clone_distill.set_multiple_replay_buffers(env=env)

# train the clones using the replay buffers
# first version of the code
# marigold_clone_distill.dualtrain_clone1(env=env, train_iters=100, batch_size=100)

# attempts at policy distillation
marigold_clone_distill.dualtrain_clone2(env=env, train_iters=200, batch_size=100,
                                        exp_name='distilltest_rose_marigold_clone_[arbitrary]')


# Run episode simulations
# marigold_clone_distill.run_clone_sim(env, record_clone=True, num_episodes=100, render=False, input_vector=[1,0])
# marigold_clone_distill.run_clone_sim(env, record_clone=True, num_episodes=100, render=False, input_vector=[0, 1])
marigold_clone_distill.run_clone_sim(env, record_clone=True, num_episodes=100, render=False, input_vector=[0.8, 0.2])

wandb.finish()
#
# PROJECT_NAME = 'distillppo_tests'
# wandb.init(project=PROJECT_NAME,name=marigold_clone_distill.fname)
# marigold_clone_distill.run_clone_sim(env, record_clone=True, num_episodes=100, render=False, input_vector=[0, 1])
# # print("done")


####################################################################################3
# Create simple clone
# marigold_clone = BehavioralClone(config_name='marigold', record_samples=True, clone_epochs=200,
#                                  clone_policy=clone_pi, optimizer=pi_optimizer, criterion=criterion,
#                                  seed=0, expert_episodes=1000, replay_buffer_size=10000)

# # Get expert samples (prerecorded)
# marigold_clone.set_replay_buffer(env=env, get_from_file=True)
#
# # Get expert samples (not prerecorded)
# # marigold_clone.set_replay_buffer(env=env, get_from_file=False)
#
# # Train the clone
# marigold_clone.train_clone(env=env, batch_size=100, train_iters=100, eval_episodes=5, eval_every=5,
#                            eval_sample_efficiency=True, print_every=10, save_every=10)

# Run episode simulations
# marigold_clone.run_clone_sim(env, record_clone=True, num_episodes=100, render=False)

#
# ###################################################################################3
# # Create clone
# clone = BehavioralClone(config_name='cyan', record_samples=True, clone_epochs=200,
#                         clone_policy=clone_pi, optimizer=pi_optimizer, criterion=criterion,
#                         seed=0, expert_episodes=500, replay_buffer_size=10000)
#
# # # Get expert samples (prerecorded)
# # clone.set_replay_buffer(env=env, get_from_file=True)
#
# # Get expert samples (not prerecorded)
# clone.set_replay_buffer(env=env, get_from_file=False)
#
# # Train the clone
# clone.train_clone(env=env, batch_size=100, train_iters=100, eval_episodes=5, eval_every=5,
#                            eval_sample_efficiency=True, print_every=10, save_every=10)
#
# # Run episode simulations
# clone.run_clone_sim(env, record_clone=True, num_episodes=100, render=False)
#









#
# # EVALUATE against the experts
#
# # Run episode simulations
# marigold_clone_distill.run_clone_sim(env, record_clone=True, num_episodes=100, render=False)
#
# # Run the experts in the same project
#
# file_names = ['ppo_penalized_' + name + '_128x4' for name in config_name_list]
# print(file_names)
#
# # Logging
#
# wandb.login()
# PROJECT_NAME = 'distillppo_tests'
# wandb.init(project=PROJECT_NAME, name='expert_' + file_names[0])
#
# # Run Marigold
# _, expert_pi = load_policy_and_env(osp.join(marigold_clone_distill._root_data_path,
#                                             file_names[0], file_names[0] + '_s0/'),
#                                                 'last', False)
#
# expert_rewards, expert_costs = run_policy(env,
#                                               expert_pi,
#                                               max_ep_len=0,
#                                               num_episodes=100,
#                                               render= False,
#                                               record=False,
#                                               # record_name='expert_' + file_names[0],
#                                               # record_project= PROJECT_NAME,
#                                               data_path= marigold_clone_distill._expert_path,
#                                               config_name= marigold_clone_distill.config_name,
#                                               max_len_rb=  marigold_clone_distill.replay_buffer_size,
#                                               benchmark=True,
#                                               log_prefix='')
#
# wandb.finish()
#
#
# # Run Rose
# wandb.init(project=PROJECT_NAME, name='expert_' + file_names[1])
# _, expert_pi = load_policy_and_env(osp.join(marigold_clone_distill._root_data_path,
#                                             file_names[1], file_names[1] + '_s0/'),
#                                                 'last', False)
#
# expert_rewards, expert_costs = run_policy(env,
#                                               expert_pi,
#                                               max_ep_len=0,
#                                               num_episodes=100,
#                                               render= False,
#                                               record=False,
#                                               # record_name='expert_' + file_names[0],
#                                               # record_project= PROJECT_NAME,
#                                               data_path= marigold_clone_distill._expert_path,
#                                               config_name= marigold_clone_distill.config_name,
#                                               max_len_rb=  marigold_clone_distill.replay_buffer_size,
#                                               benchmark=True,
#                                               log_prefix='')