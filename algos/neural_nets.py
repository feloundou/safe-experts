import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

from torch.nn import Parameter

import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from utils import *


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()

        # log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # self.log_std = nn.Parameter(-0.5 * torch.ones(act_dim))
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # print("test list")
        # print(list(hidden_sizes))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation, action_dim):
        super(GaussianPolicy, self).__init__()
        # print("Gaussian policy used.")
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))
        self.mu = MLP(layers=[input_dim] + list(hidden_dims) + [action_dim], activation=activation,
                      output_activation=output_activation)

    def forward(self, x, act=None):
        policy = Normal(self.mu(x), self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)


        if act is not None:
            logp = policy.log_prob(act).sum(dim=1)

            # print("gaussian action: ", act)
            # print("pi:", pi)
            # print("log pi:", logp_pi)
            # print("logp:", logp)

        else:
            logp = None

        return pi, logp, logp_pi

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.



class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function critics
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
        self.vc = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            vc = self.vc(obs)
            # pen = self.pen(obs)

        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]



class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space, hidden_dims=(64, 64), activation=torch.tanh, output_activation=None,
                 policy=None):
        super(ActorCritic, self).__init__()

        if policy is None:
            if isinstance(action_space, Box):
                self.pi = GaussianPolicy(input_dim, hidden_dims, activation, output_activation,
                                             action_space.shape[0])
            elif isinstance(action_space, Discrete):
                self.pi = CategoricalPolicy(input_dim, hidden_dims, activation, output_activation, action_space.n)
        else:
            self.pi = policy(input_dim, hidden_dims, activation, output_activation, action_space)

        self.v = MLP(layers=[input_dim] + list(hidden_dims) + [1], activation=activation, output_squeeze=True)

    def forward(self, x, a=None):
        pi, logp, logp_pi = self.pi(x, a)
        v = self.v(x)

        return pi, logp, logp_pi, v

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        # log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        self.shared_net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mu_net = nn.Linear(hidden_sizes[-1], act_dim)
        self.var_net = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, x):
        mu = self.mu_net(F.leaky_relu(self.shared_net(x)))
        std = self.var_net(F.leaky_relu(self.shared_net(x)))
        return Normal(loc=mu, scale=std).rsample()



class DistilledGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, n_experts):
        super().__init__()
        obs_dim_aug = obs_dim + n_experts
        self.shared_net = mlp([obs_dim_aug] + list(hidden_sizes), activation)

        self.mu_net = nn.Linear(hidden_sizes[-1], act_dim)
        self.var_net = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, x):

        out = F.leaky_relu(self.shared_net(x))
        mu = self.mu_net(out)
        std = self.var_net(out)

        return Normal(loc=mu, scale=std).rsample()

class MLPDiscriminator(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        discrim_dim = obs_dim + act_dim
        self.discrim_net = mlp([discrim_dim] + list(hidden_sizes) + [1], activation)


    def forward(self, obs):
        prob = torch.sigmoid(self.discrim_net(obs))
        return prob


class BiclassificationPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation):
        super(BiclassificationPolicy, self).__init__()

        self.output_dim = 2
        self.logits = MLP(layers=[input_dim] + list(hidden_dims) + [self.output_dim], activation=activation)

    def forward(self, x, label=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        l = policy.sample()
        logp_l = policy.log_prob(l).squeeze()
        if label is not None:
            logp = policy.log_prob(label).squeeze()
        else:
            logp = None

        return l, logp, logp_l

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 64), activation=torch.relu, output_activation=torch.softmax):
        super(Discriminator, self).__init__()

        self.pi = BiclassificationPolicy(input_dim, hidden_dims, activation, output_activation)

    def forward(self, state, gt=None):
        label, loggt, logp = self.pi(state, gt)
        return label, loggt, logp


class VDB(nn.Module):
    # def __init__(self, num_inputs, args):
    def __init__(self, obs_space, act_space, hidden_sizes, activation=nn.Tanh):
        super(VDB, self).__init__()
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        discrim_dim = obs_dim + act_dim
        z_size = 128

        # self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        self.fc1 = nn.Linear(discrim_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], z_size)
        self.fc3 = nn.Linear(hidden_sizes[0], z_size)
        self.fc4 = nn.Linear(z_size, hidden_sizes[0])
        self.fc5 = nn.Linear(hidden_sizes[0], 1)

        self.fc5.weight.data.mul_(0.1)
        self.fc5.bias.data.mul_(0.0)

    def encoder(self, x):
        h = torch.tanh(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def discriminator(self, z):
        h = torch.tanh(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        prob = self.discriminator(z)
        return prob, mu, logvar


class MLP(nn.Module):
    def __init__(self, layers, activation=torch.tanh, output_activation=None,
                 output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze

        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x



# class CategoricalPolicy(nn.Module):
#     def __init__(self, input_dim, hidden_dims, activation, output_activation, action_dim):
#         super(CategoricalPolicy, self).__init__()
#
#         print("Categorical policy used.")
#         self.logits = MLP(layers=[input_dim] + list(hidden_dims) + [action_dim], activation=activation)
#
#     def forward(self, x, a=None):
#         logits = self.logits(x)
#         policy = Categorical(logits=logits)
#         pi = policy.sample()
#         logp_pi = policy.log_prob(pi).squeeze()
#         if a is not None:
#             logp = policy.log_prob(a).squeeze()
#         else:
#             logp = None
#
#         return pi, logp, logp_pi



# class BLSTMPolicy(nn.Module):
#     def __init__(self, input_dim, hidden_dims, activation, output_activation, con_dim):
#         super(BLSTMPolicy, self).__init__()
#         self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dims//2, batch_first=True, bidirectional=True)
#         self.linear = nn.Linear(hidden_dims, con_dim)
#         nn.init.zeros_(self.linear.bias)
#
#     def forward(self, seq, gt=None):
#         inter_states, _ = self.lstm(seq)
#         print("inter states: ", inter_states)
#         logit_seq = self.linear(inter_states)
#         print("LOGIT SEQ")
#         print(logit_seq)
#         self.logits = torch.mean(logit_seq, dim=1)
#         policy = Categorical(logits=self.logits)
#         label = policy.sample()
#         logp = policy.log_prob(label).squeeze()
#         if gt is not None:
#             loggt = policy.log_prob(gt).squeeze()
#         else:
#             loggt = None
#
#         return label, loggt, logp


class ValorFFNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation, con_dim):
        super(ValorFFNNPolicy, self).__init__()

        self.context_net = mlp([input_dim] + list(hidden_dims), activation)
        self.linear = nn.Linear(hidden_dims[-1], con_dim)

    def forward(self, seq, gt=None, classes=False):
        inter_states = self.context_net(seq)
        logit_seq = self.linear(inter_states)
        self.logits = torch.mean(logit_seq, dim=1)
        policy = Categorical(logits=self.logits)
        label = policy.sample()
        # print("LABEL: ", label)
        logp = policy.log_prob(label).squeeze()

        if gt is not None:
            # print('GROUND TRUTH: ', gt)
            # ground_truth_ids = gt.argmax(axis=1)
            # print("ground truth ids", ground_truth_ids)
            loggt = policy.log_prob(gt).squeeze()
        else:
            loggt = None

        if classes is False:
            return label, loggt, logp
        else:
            return label, loggt, logp, gt


class ValorDiscriminator(nn.Module):
    def __init__(self, input_dim, context_dim, activation=nn.Softmax,
                 output_activation=nn.Softmax, hidden_dims=64):

        super(ValorDiscriminator, self).__init__()
        self.context_dim = context_dim

        # self.pi = BLSTMPolicy(input_dim, hidden_dims, activation=torch.softmax,
        # output_activation=torch.softmax, con_dim=self.context_dim)
        self.pi = ValorFFNNPolicy(input_dim, hidden_dims, activation=nn.Tanh,
                                  output_activation=nn.Tanh, con_dim=self.context_dim)

    def forward(self, seq, gt=None, classes=False):
        if classes is False:
            pred, loggt, logp = self.pi(seq, gt, classes)
            return pred, loggt, logp

        else:
            pred, loggt, logp, gt = self.pi(seq, gt, classes)
            return pred, loggt, logp, gt



