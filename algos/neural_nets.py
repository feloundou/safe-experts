import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

import torch

from torch.distributions import Independent, OneHotCategorical
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

# from utils import *



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

        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)  # Last axis sum needed for Torch Normal distribution


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

        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]



class ValorActorCritic(nn.Module):

    def __init__(self, input_dim, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = input_dim
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

        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]




class GaussianReward(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()

        self.shared_net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mu_net = nn.Linear(hidden_sizes[-1], 1)
        self.var_net = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, x):

        out = F.leaky_relu(self.shared_net(x))
        mu = self.mu_net(out)
        std = self.var_net(out)
        return Normal(loc=mu, scale=std).rsample()



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


class Reward_VDB(nn.Module):
    # def __init__(self, num_inputs, args):
    def __init__(self, obs_space, act_space, hidden_sizes, activation=nn.Tanh):
        super().__init__()
        obs_dim = obs_space.shape[0]
        act_dim = act_space.shape[0]
        discrim_dim = obs_dim + act_dim

        # self.mu_net = mlp([discrim_dim] + list(hidden_sizes) + [1], activation)

        z_size = 128

        self.shared_net = mlp([discrim_dim] + list(hidden_sizes), activation)
        self.mu_net = nn.Linear(hidden_sizes[-1], z_size)
        self.var_net = nn.Linear(hidden_sizes[-1], z_size)

        # self.fc1 = nn.Linear(num_inputs, args.hidden_size)
        # self.fc1 = nn.Linear(discrim_dim, hidden_sizes[0])
        # self.fc2 = nn.Linear(hidden_sizes[0], z_size)
        # self.fc3 = nn.Linear(hidden_sizes[0], z_size)
        self.fc4 = nn.Linear(z_size, hidden_sizes[0])
        self.fc5 = nn.Linear(hidden_sizes[0], 1)

        self.fc5.weight.data.mul_(0.1)
        self.fc5.bias.data.mul_(0.0)

    def encoder(self, x):
        h = torch.tanh(self.shared_net(x))
        return self.mu_net(h), self.var_net(h)

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


class VDB(nn.Module):
    # def __init__(self, num_inputs, args):
    def __init__(self, obs_space, act_space, hidden_sizes, activation=nn.Tanh):
        super().__init__()
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



class ValorFFNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation, con_dim):
        super(ValorFFNNPolicy, self).__init__()

        self.context_net = mlp([input_dim] + list(hidden_dims), activation)
        self.linear = nn.Linear(hidden_dims[-1], con_dim)

    def forward(self, seq, gt=None, classes=False):
        # inter_states = self.context_net(seq)
        # # print("seq: ", seq)
        # # print("gt: ", gt)
        # logit_seq = self.linear(inter_states)
        logit_seq = self.context_net(seq)
        self.logits = torch.mean(logit_seq, dim=1)
        policy = Categorical(logits=self.logits)
        label = policy.sample()
        # print("LABEL: ", label)
        logp = policy.log_prob(label).squeeze()

        if gt is not None:
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

        self.pi = ValorFFNNPolicy(input_dim, hidden_dims, activation=nn.Tanh,
                                  output_activation=nn.Tanh, con_dim=self.context_dim)

    def forward(self, seq, gt=None, classes=False):
        if classes is False:
            pred, loggt, logp = self.pi(seq, gt, classes)
            return pred, loggt, logp

        else:
            pred, loggt, logp, gt = self.pi(seq, gt, classes)
            return pred, loggt, logp, gt




class ModValorFFNNPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation, con_dim):
        super(ModValorFFNNPolicy, self).__init__()

        self.context_net = mlp([input_dim] + list(hidden_dims) + [con_dim], activation)


    def forward(self, seq, gt=None):

        logit_seq = self.context_net(seq)
        self.logits = torch.mean(logit_seq, dim=1)
        policy = Categorical(logits=self.logits)
        label = policy.sample()
        print("LABEL: ", label)
        logp = policy.log_prob(label).squeeze()

        if gt is not None:
            loggt = policy.log_prob(gt).squeeze()
        else:
            loggt = None

            return label, loggt, logp




class ModValorDiscriminator(nn.Module):
    def __init__(self, input_dim, context_dim, activation=nn.Softmax,
                 output_activation=nn.Softmax, hidden_dims=64):

        super(ModValorDiscriminator, self).__init__()
        self.context_dim = context_dim

        self.pi = ModValorFFNNPolicy(input_dim, hidden_dims, activation=nn.Tanh,
                                  output_activation=nn.Tanh, con_dim=self.context_dim)

    def forward(self, seq, gt=None):
            pred, loggt, logp = self.pi(seq, gt)
            return pred, loggt, logp


class DiagGaussianLayer(nn.Module):
    '''
    Implements a layer that outputs a Gaussian distribution with a diagonal
    covariance matrix
    Attributes
    ----------
    log_std : torch.FloatTensor
        the log square root of the diagonal elements of the covariance matrix
    Methods
    -------
    __call__(mean)
        takes as input a mean vector and outputs a Gaussian distribution with
        diagonal covariance matrix defined by log_std
    '''

    def __init__(self, output_dim=None, log_std=None):
        nn.Module.__init__(self)

        self.log_std = log_std

        if log_std is None:
            self.log_std = Parameter(torch.zeros(output_dim), requires_grad=True)

    def __call__(self, mean):
        std = torch.exp(self.log_std)
        normal_dist = Independent(Normal(loc=mean, scale=std), 1)

        return normal_dist


def build_layers(input_dim, hidden_dims, output_dim,
                 activation=nn.Tanh, output_activation=nn.Identity):
    '''
    Returns a list of Linear and Tanh layers with the specified layer sizes
    Parameters
    ----------
    input_dim : int
        the input dimension of the first linear layer
    hidden_dims : list
        a list of type int specifying the sizes of the hidden layers
    output_dim : int
        the output dimension of the final layer in the list
    Returns
    -------
    layers : list
        a list of Linear layers, each one followed by a Tanh layer, excluding the
        final layer
    '''

    layer_sizes = [input_dim] + hidden_dims + [output_dim]
    layers = []

    for i in range(len(layer_sizes) - 1):
        act = activation if i < len(layer_sizes) - 2 else output_activation # Tyna note
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias=True))

        if i != len(layer_sizes) - 2:
            layers.append(nn.Tanh())

    return layers



def MLP_DiagGaussianPolicy(state_dim, hidden_dims, action_dim,
                           log_std=None):
    '''
    Build a multilayer perceptron with a DiagGaussianLayer at the output layer
    Parameters
    ----------
    state_dim : int
        the input size of the network
    hidden_dims : list
        a list of type int specifying the sizes of the hidden layers
    action_dim : int
        the dimensionality of the Gaussian distribution to be outputted by the
        policy
    log_std : torch.FloatTensor
        the log square root of the diagonal elements of the covariance matrix
        (will be set to a vector of zeros if none is specified)
    Returns
    -------
    policy : torch.nn.Sequential
        a pytorch sequential model that outputs a Gaussian distribution
    '''

    layers = build_layers(state_dim, hidden_dims, action_dim)
    layers[-1].weight.data *= 0.1
    layers[-1].bias.data *= 0.0
    layers.append(DiagGaussianLayer(action_dim, log_std))
    policy = nn.Sequential(*layers)

    return policy


#############################################################################

class OneHotCategoricalActor(Actor):

    def __init__(self, obs_dim, con_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [con_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return OneHotCategorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class VAE_Encoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super(VAE_Encoder, self).__init__()
        # self.linear1 = nn.Linear(in_dim, hidden)
        # self.linear2 = nn.Linear(hidden, out_dim)

        hidden_sizes = [100]
        self.logits_net = mlp([in_dim] + hidden_sizes + [out_dim], activation=nn.Tanh)

    def forward(self, x):
        # y = F.relu(self.linear1(x))
        # z = F.relu(self.linear2(y))
        # y = F.tanh(self.linear1(x))   ### TODO: Confirm if this tanh vs relu was the biggest ISSUE! UGH
        # z = F.tanh(self.linear2(y))

        z = self.logits_net(x)
        return z


class VAE_Decoder(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(VAE_Decoder, self).__init__()

        act_dim = 2
        hidden_sizes = [128]*4
        activation = nn.Tanh

        ## actor
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([D_in] + list(hidden_sizes) + [act_dim], activation)
        ##
        self.pi = MLPGaussianActor(D_in, act_dim, hidden_sizes, activation)

    def forward(self, x):
        return self.pi._distribution(x)



class VAELOR(torch.nn.Module):

    def __init__(self, obs_dim, latent_dim):
        super(VAELOR, self).__init__()
        """
        Given input tensor, forward prop to get context samples, raw state and state differences. 
        Returns latent variable
        :param obs_dim: state dimension
        :param latent_length: latent vector length
        """
        act_dim = 2
        link_layer = 400
        self.encoder = VAE_Encoder(obs_dim, 100, link_layer) # original
        self.lmbd = Lambda(input_dim=link_layer, latent_length=latent_dim)
        self.decoder = VAE_Decoder(obs_dim + latent_dim, 100, act_dim)

    def forward(self, state, delta_state, action, latent_labels = None):
        delta_state_enc = self.encoder(delta_state) # original

        # latent_v = self.lmbd(delta_state_enc)  # original
        latent_v_dist = self.lmbd(delta_state_enc)
        if latent_labels is None:
            latent_labels= latent_v_dist.sample()

        # print("Latent V Sample: ", latent_labels[:2])
        concat_state = torch.cat([state, latent_labels], dim=1)
        action_dist = self.decoder(concat_state)
        # return action_dist, latent_labels
        return action_dist, latent_labels, latent_v_dist


    def compute_latent_loss(self, X, Delta_X, A, context_sample):
        """
        Given input tensor, forward propagate, compute the loss, and backward propagate.
        Represents the lifecycle of a single iteration
        :param X: Input tensor
        :return: total loss, reconstruction loss, kl-divergence loss and original input
        """
        # decoded_action, latent_labels, logp_action = self(X, Delta_X)
        action_dist, latent_labels, latent_labels_dist = self(X, Delta_X, A)
        sampled_action = action_dist.sample()

        # get latent labels for checking accuracy
        context_loss = -latent_labels_dist.log_prob(context_sample)  # this is the correct version  ##
        # context_loss = -latent_labels_dist.log_prob(latent_labels)
        # recon_loss = -action_dist.log_prob(sampled_action).sum(axis=-1)
        recon_loss = -action_dist.log_prob(A).sum(axis=-1)
        loss = recon_loss * context_loss          # loss = recon_loss + context_loss

        # print("Expert Action: \t", A[:2])
        # print("Learner Action: \t", sampled_action[:2])
        # print("Total Valor Loss: \t ", loss[:2])
        # print("context loss: \t", context_loss[:2])
        # print("recon loss: \t", recon_loss[:2])
        return loss, recon_loss, context_loss, X, latent_labels


class Lambda(nn.Module):
    """Lambda module converts output of encoder to latent vector
    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, input_dim, latent_length):
        super(Lambda, self).__init__()
        self._latent_net = nn.Linear(input_dim, latent_length)   ## old

        hidden_sizes = [500]
        con_dim = 2

        self.lambda_pi = OneHotCategoricalActor(input_dim, con_dim, hidden_sizes, activation=nn.Tanh)


    def forward(self, cell_output):
        return self.lambda_pi._distribution(cell_output)










########3

class MLPContextLabeler(Actor):

    # def __init__(self, obs_dim, context_dim, hidden_sizes, activation):
    def __init__(self, input_dim, context_dim, hidden_sizes, activation):
        super().__init__()

        # log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        # self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [context_dim], activation)

        self.logits_net = mlp([input_dim] + list(hidden_sizes) + [context_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)


    def _log_prob_from_distribution(self, pi, con):
        return pi.log_prob(con)

    def label_context(self, obs):
        with torch.no_grad():
            pi = self._distribution(obs)
            con = pi.sample()
            # print("Drawn context: ", con)
            # logp_con = self._log_prob_from_distribution(pi, con)

        # return logp_con.numpy()
        return con


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
