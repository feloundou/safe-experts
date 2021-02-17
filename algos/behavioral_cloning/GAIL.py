import torch
import torch.nn as nn
import os
import gym

import argparse
import math
from torch.distributions import Normal
from utils import *

import pickle
import argparse
import numpy as np
from collections import deque

import torch.optim as optim



def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action

def get_entropy(mu, std):
    dist = Normal(mu, std)
    entropy = dist.entropy().mean()
    return entropy

def log_prob_density(x, mu, std):
    log_prob_density = -(x - mu).pow(2) / (2 * std.pow(2)) \
                     - 0.5 * math.log(2 * math.pi)
    return log_prob_density.sum(1, keepdim=True)

def get_reward(discrim, state, action):
    state = torch.Tensor(state)
    action = torch.Tensor(action)
    state_action = torch.cat([state, action])
    with torch.no_grad():
        return -math.log(discrim(state_action)[0].item())

def save_checkpoint(state, filename):
    torch.save(state, filename)


import torch
import numpy as np
# from utils.utils import get_entropy, log_prob_density


def train_discrim(discrim, memory, discrim_optim, demonstrations, args):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])

    states = torch.Tensor(states)
    actions = torch.Tensor(actions)

    criterion = torch.nn.BCELoss()

    for _ in range(args.discrim_update_num):
        learner = discrim(torch.cat([states, actions], dim=1))
        demonstrations = torch.Tensor(demonstrations)
        expert = discrim(demonstrations)

        discrim_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
                       criterion(expert, torch.zeros((demonstrations.shape[0], 1)))

        discrim_optim.zero_grad()
        discrim_loss.backward()
        discrim_optim.step()

    expert_acc = ((discrim(demonstrations) < 0.5).float()).mean()
    learner_acc = ((discrim(torch.cat([states, actions], dim=1)) > 0.5).float()).mean()

    return expert_acc, learner_acc


def train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])

    old_values = critic(torch.Tensor(states))
    returns, advants = get_gae(rewards, masks, old_values, args)

    mu, std = actor(torch.Tensor(states))
    old_policy = log_prob_density(torch.Tensor(actions), mu, std)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for _ in range(args.actor_critic_update_num):
        np.random.shuffle(arr)

        for i in range(n // args.batch_size):
            batch_index = arr[args.batch_size * i: args.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)

            inputs = torch.Tensor(states)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()

            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -args.clip_param,
                                         args.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
                                                  old_policy.detach(), actions_samples,
                                                  batch_index)
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - args.clip_param,
                                        1.0 + args.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            critic_optim.zero_grad()
            loss.backward(retain_graph=True)
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()


def get_gae(rewards, masks, values, args):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (args.gamma * running_returns * masks[t])
        returns[t] = running_returns

        running_delta = rewards[t] + (args.gamma * previous_value * masks[t]) - \
                        values.data[t]
        previous_value = values.data[t]

        running_advants = running_delta + (args.gamma * args.lamda * \
                                           running_advants * masks[t])
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    mu, std = actor(states)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    entropy = get_entropy(mu, std)

    return surrogate_loss, ratio, entropy

class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, args):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, num_outputs)

        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std


class Critic(nn.Module):
    def __init__(self, num_inputs, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)

        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v


class Discriminator(nn.Module):
    def __init__(self, num_inputs, args):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)

        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        prob = torch.sigmoid(self.fc3(x))
        return prob





# parser = argparse.ArgumentParser()
# parser.add_argument('--env', type=str, default="Hopper-v2",
#                     help='name of Mujoco environement')
# parser.add_argument('--iter', type=int, default=5,
#                     help='number of episodes to play')
# parser.add_argument("--load_model", type=str, default='ppo_max.tar',
#                     help="if you test pretrained file, write filename in save_model folder")
#
# args = parser.parse_args()

if __name__ == "__main__":

    env = gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    running_state = ZFilter((num_inputs,), clip=5)

    print('state size:', num_inputs)
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions, args)
    critic = Critic(num_inputs, args)
    discrim = Discriminator(num_inputs + num_actions, args)

    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate,
                              weight_decay=args.l2_rate)
    discrim_optim = optim.Adam(discrim.parameters(), lr=args.learning_rate)

    # load demonstrations
    expert_demo, _ = pickle.load(open('./expert_demo/expert_demo.p', "rb"))
    demonstrations = np.array(expert_demo)
    print("demonstrations.shape", demonstrations.shape)

    writer = SummaryWriter(args.logdir)

    if args.load_model is not None:
        saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        ckpt = torch.load(saved_ckpt_path)

        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        discrim.load_state_dict(ckpt['discrim'])

        running_state.rs.n = ckpt['z_filter_n']
        running_state.rs.mean = ckpt['z_filter_m']
        running_state.rs.sum_square = ckpt['z_filter_s']

        print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

    episodes = 0
    train_discrim_flag = True

    for iter in range(args.max_iter_num):
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []

        while steps < args.total_sample_size:
            state = env.reset()
            score = 0

            state = running_state(state)

            for _ in range(10000):
                if args.render:
                    env.render()

                steps += 1

                mu, std = actor(torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                next_state, reward, done, _ = env.step(action)
                irl_reward = get_reward(discrim, state, action)

                if done:
                    mask = 0
                else:
                    mask = 1

                memory.append([state, action, irl_reward, mask])

                next_state = running_state(next_state)
                state = next_state

                score += reward

                if done:
                    break

            episodes += 1
            scores.append(score)

        score_avg = np.mean(scores)
        print('{}:: {} episode score is {:.2f}'.format(iter, episodes, score_avg))
        writer.add_scalar('log/score', float(score_avg), iter)

        actor.train(), critic.train(), discrim.train()
        if train_discrim_flag:
            expert_acc, learner_acc = train_discrim(discrim, memory, discrim_optim, demonstrations, args)
            print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
            if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
                train_discrim_flag = False
        train_actor_critic(actor, critic, memory, actor_optim, critic_optim, args)

        if iter % 100:
            score_avg = int(score_avg)

            model_path = os.path.join(os.getcwd(), 'save_model')
            if not os.path.isdir(model_path):
                os.makedirs(model_path)

            ckpt_path = os.path.join(model_path, 'ckpt_' + str(score_avg) + '.pth.tar')

            save_checkpoint({
                'actor': actor.state_dict(),
                'critic': critic.state_dict(),
                'discrim': discrim.state_dict(),
                'z_filter_n': running_state.rs.n,
                'z_filter_m': running_state.rs.mean,
                'z_filter_s': running_state.rs.sum_square,
                'args': args,
                'score': score_avg
            }, filename=ckpt_path)