from il_utils import *
from torch.optim import Adam


from fullyconnected_models import *

class Approximation():
    '''
    Base function approximation object.
    This defines a Pytorch-based function approximation object that
    wraps key functionality useful for reinforcement learning, including
    decaying learning rates, model checkpointing, loss scaling, gradient
    clipping, target networks, and tensorboard logging. This enables
    increased code reusability and simpler Agent implementations.
    Args:
            model (:torch.nn.Module:): A Pytorch module representing the model
                used to approximate the function. This could be a convolution
                network, a fully connected network, or any other Pytorch-compatible
                model.
            optimizer (:torch.optim.Optimizer:): A optimizer initialized with the
                model parameters, e.g. SGD, Adam, RMSprop, etc.
            checkpointer: (:all.approximation.checkpointer.Checkpointer): A Checkpointer object
                that periodically saves the model and its parameters to the disk. Default:
                A PeriodicCheckpointer that saves the model once every 200 updates.
            clip_grad: (float, optional): If non-zero, clips the norm of the
                gradient to this value in order prevent large updates and
                improve stability.
                See torch.nn.utils.clip_grad.
            loss_scaling: (float, optional): Multiplies the loss by this value before
                performing a backwards pass. Useful when used with multi-headed networks
                with shared feature layers.
            name: (str, optional): The name of the function approximator used for logging.
            lr_scheduler: (:torch.optim.lr_scheduler._LRScheduler:, optional): A learning
                rate scheduler initialized with the given optimizer. step() will be called
                after every update.
            target: (:all.approximation.target.TargetNetwork, optional): A target network object
                to be used during optimization. A target network updates more slowly than
                the base model that is being optimizing, allowing for a more stable
                optimization target.
    '''

    def __init__(
            self,
            model,
            optimizer,
            checkpointer=None,
            clip_grad=0,
            loss_scaling=1,
            name='approximation',
            lr_scheduler=None,
            target=None,
    ):
        self.model = model
        self.device = next(model.parameters()).device
        # self._target = target or TrivialTarget()
        self._target = target
        self._lr_scheduler = lr_scheduler
        self._target.init(model)
        self._optimizer = optimizer
        self._loss_scaling = loss_scaling
        self._cache = []
        self._clip_grad = clip_grad
        self._writer = get_writer()
        self._name = name

        if checkpointer is None:
            checkpointer = PeriodicCheckpointer(DEFAULT_CHECKPOINT_FREQUENCY)
        self._checkpointer = checkpointer
        self._checkpointer.init(
            self.model,
            self._writer.log_dir,
            name
        )

    def __call__(self, *inputs):
        '''
        Run a forward pass of the model.
        '''
        return self.model(*inputs)

    def no_grad(self, *inputs):
        '''Run a forward pass of the model in no_grad mode.'''
        with torch.no_grad():
            return self.model(*inputs)

    def eval(self, *inputs):
        '''
        Run a forward pass of the model in eval mode with no_grad.
        The model is returned to its previous mode afer the forward pass is made.
        '''
        with torch.no_grad():
            # check current mode
            mode = self.model.training
            # switch to eval mode
            self.model.eval()
            # run forward pass
            result = self.model(*inputs)
            # change to original mode
            self.model.train(mode)
            return result

    def target(self, *inputs):
        '''Run a forward pass of the target network.'''
        return self._target(*inputs)

    def reinforce(self, loss=None):
        if loss is not None:
            self._optimizer.zero_grad()
            loss = self._loss_scaling * loss
            self._writer.add_scalar("loss/" + self._name, loss.detach())
            loss.backward()
        self.step()
        return self

    def step(self):
        '''Given that a backward pass has been made, run an optimization step.'''
        if self._clip_grad != 0:
            utils.clip_grad_norm_(self.model.parameters(), self._clip_grad)
        self._optimizer.step()
        self._target.update()
        if self._lr_scheduler:
            self._writer.add_scalar(
                "schedule/" + self._name + '/lr', self._optimizer.param_groups[0]['lr'])
            self._lr_scheduler.step()
        self._checkpointer()
        return self

    def zero_grad(self):
        self._optimizer.zero_grad()
        return self

class FeatureNetwork(Approximation):
    '''
    A special type of Approximation that accumulates gradients before backpropagating them.
    This is useful when features are shared between network heads.
    The __call__ function caches the computation graph and detaches the output.
    Then, various functions approximators may backpropagate to the output.
    The reinforce() function will then backpropagate the accumulated gradients on the output
    through the original computation graph.
    '''

    def __init__(self, model, optimizer=None, name='feature', **kwargs):
        model = FeatureModule(model)
        super().__init__(model, optimizer, name=name, **kwargs)
        self._cache = []
        self._out = []

    def __call__(self, states):
        '''
        Run a forward pass of the model and return the detached output.
        Args:
            state (all.environment.State): An environment State
        Returns:
            all.environment.State: An enviornment State with the computed features
        '''
        features = self.model(states)
        graphs = features.raw
        features._raw = graphs.detach()
        features._raw.requires_grad = True
        self._enqueue(graphs, features._raw)
        return features

    def reinforce(self):
        '''
        Backward pass of the model.
        '''
        self._optimizer.zero_grad()
        graphs, grads = self._dequeue()
        graphs.backward(grads)
        self.step()

    def _enqueue(self, features, out):
        self._cache.append(features)
        self._out.append(out)

    def _dequeue(self):
        graphs = []
        grads = []
        for graph, out in zip(self._cache, self._out):
            if out.grad is not None:
                graphs.append(graph)
                grads.append(out.grad)
        self._cache = []
        self._out = []
        return torch.cat(graphs), torch.cat(grads)


class FeatureModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, states):
        features = self.model(states.features.float())
        return State(
            features,
            mask=states.mask,
            info=states.info
        )


class Agent(ABC):
    """
    Abstract agent class
    """

    @abstractmethod
    def act(self, state, reward=None):
        """
        Select an action for evaluation.
        If the agent has a replay-buffer, state and reward are stored.
        Args:
            state (rlil.environment.State): The environment state at the current timestep.
            reward (torch.Tensor): The reward from the previous timestep.
        Returns:
            rllib.Action: The action to take at the current timestep.
        """

    @abstractmethod
    def make_lazy_agent(self, evaluation=False):
        """
        Return a LazyAgent object for sampling or evaluation.
        Args:
            evaluation (bool, optional): If evaluation==True, the returned
            object act greedily. Defaults to False.
        Returns:
            LazyAgent: The LazyAgent object for Sampler.
        """
        pass

    def train(self):
        """
        Update internal parameters
        """
        pass

    def load(self, dirname):
        """
        Load pretrained agent.
        Args:
            dirname (str): Directory where the agent saved
        """
        pass


class LazyAgent(ABC):
    """
    Agent class for Sampler.
    """

    def __init__(self,
                 evaluation=False,
                 store_samples=True):
        self._states = None
        self._actions = None
        self._evaluation = evaluation
        self._store_samples = store_samples
        self.replay_buffer = None
        # for N step replay buffer
        self._n_step, self._discount_factor = get_n_step()
        if self._evaluation:
            self._n_step = 1  # disable Nstep buffer when evaluation mode

    def set_replay_buffer(self, env):
        self.replay_buffer = ExperienceReplayBuffer(
            1e7, env, n_step=self._n_step,
            discount_factor=self._discount_factor)

    def act(self, states, reward):
        """
        In the act function, the lazy_agent put a sample
        (last_state, last_action, reward, states) into self.replay_buffer.
        Then, it outputs a corresponding action.
        """
        if self._store_samples:
            assert self.replay_buffer is not None, \
                "Call self.set_replay_buffer(env) at lazy_agent initialization."
            samples = Samples(self._states, self._actions, reward, states)
            self.replay_buffer.store(samples)

    def compute_priorities(self, samples):
        """
        Compute priorities of the given samples.
        This method is useful for Apex implementation.
        Args:
            samples (rlil.utils.Samples)
        """
        return None

class PPO(Agent):
    """
    Proximal Policy Optimization (PPO).
    PPO is an actor-critic style policy gradient algorithm that allows for the reuse of samples
    by using importance weighting. This often increases sample efficiency compared to algorithms
    such as A2C. To avoid overfitting, PPO uses a special "clipped" objective that prevents
    the algorithm from changing the current policy too quickly.
    Args:
        feature_nw (FeatureNetwork): Shared feature layers.
        v (VNetwork): Value head which approximates the state-value function.
        policy (StochasticPolicy): Policy head which outputs an action distribution.
        entropy_loss_scaling (float): Contribution of the entropy loss to the total policy loss.
        epochs (int): Number of times to reuse each sample.
        lam (float): The Generalized Advantage Estimate (GAE) decay parameter.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        minibatches (int): The number of minibatches to split each batch into.
        epochs (int): Number of times to reuse each sample.
    """

    def __init__(
            self,
            feature_nw,
            v,
            policy,
            entropy_loss_scaling=0.01,
            epsilon=0.2,
            replay_start_size=5000,
            minibatches=4,
            epochs=4,
    ):
        # objects
        self.feature_nw = feature_nw
        self.v = v
        self.policy = policy
        self.replay_buffer = get_replay_buffer()
        self.writer = get_writer()
        self.device = get_device()
        # hyperparameters
        self.entropy_loss_scaling = entropy_loss_scaling
        self.epsilon = epsilon
        self.minibatches = minibatches
        self.epochs = epochs
        self.replay_start_size = replay_start_size
        # private
        self._states = None
        self._actions = None

    def act(self, states, rewards=None):
        if rewards is not None:
            samples = Samples(self._states, self._actions, rewards, states)
            self.replay_buffer.store(samples)
        self._states = states
        self._actions = Action(self.policy.no_grad(
            self.feature_nw.no_grad(states.to(self.device))).sample()).to("cpu")
        return self._actions

    def train(self):
        if self.should_train():
            states, actions, rewards, next_states, _, _ = \
                self.replay_buffer.get_all_transitions()

            # compute gae
            features = self.feature_nw.target(states)
            values = self.v.target(features)
            next_values = self.v.target(self.feature_nw.target(next_states))
            advantages = self.replay_buffer.compute_gae(
                rewards, values, next_values, next_states.mask)

            # compute target values
            # actions.raw is used since .features clip the actions
            pi_0 = self.policy.no_grad(features).log_prob(actions.raw)
            targets = values + advantages

            # train for several epochs
            for _ in range(self.epochs):
                # randomly permute the indexes to generate minibatches
                minibatch_size = int(len(states) / self.minibatches)
                indexes = torch.randperm(len(states))
                for n in range(self.minibatches):
                    # load the indexes for the minibatch
                    first = n * minibatch_size
                    last = first + minibatch_size
                    i = indexes[first:last]

                    # perform a single training step
                    self._train_minibatch(
                        states[i], actions[i], pi_0[i], advantages[i], targets[i])
                    self.writer.train_steps += 1

            # clear buffer for on-policy training
            self.replay_buffer.clear()

    def _train_minibatch(self, states, actions, pi_0, advantages, targets):
        # forward pass
        features = self.feature_nw(states)
        values = self.v(features)
        distribution = self.policy(features)
        pi_i = distribution.log_prob(actions.raw)

        # compute losses
        value_loss = mse_loss(values, targets).mean()
        policy_gradient_loss = self._clipped_policy_gradient_loss(
            pi_0, pi_i, advantages)
        entropy_loss = -distribution.entropy().mean()
        policy_loss = policy_gradient_loss + \
            self.entropy_loss_scaling * entropy_loss

        # backward pass
        self.policy.reinforce(policy_loss)
        self.v.reinforce(value_loss)
        self.feature_nw.reinforce()

        # debugging
        self.writer.add_scalar('loss/policy_gradient',
                               policy_gradient_loss.detach())
        self.writer.add_scalar('loss/entropy',
                               entropy_loss.detach())

    def _clipped_policy_gradient_loss(self, pi_0, pi_i, advantages):
        ratios = torch.exp(pi_i - pi_0)
        # debugging
        self.writer.add_scalar('loss/ratios/max', ratios.max())
        self.writer.add_scalar('loss/ratios/min', ratios.min())
        surr1 = ratios * advantages
        epsilon = self.epsilon
        surr2 = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages
        return -torch.min(surr1, surr2).mean()

    def should_train(self):
        return len(self.replay_buffer) > self.replay_start_size

    def make_lazy_agent(self, evaluation=False, store_samples=True):
        policy_model = deepcopy(self.policy.model)
        feature_model = deepcopy(self.feature_nw.model)
        return PPOLazyAgent(policy_model.to("cpu"),
                            feature_model.to("cpu"),
                            evaluation=evaluation,
                            store_samples=store_samples)

    def load(self, dirname):
        for filename in os.listdir(dirname):
            if filename == 'policy.pt':
                self.policy.model = torch.load(os.path.join(
                    dirname, filename), map_location=self.device)
            if filename in ('feature.pt'):
                self.feature_nw.model = torch.load(os.path.join(dirname, filename),
                                                   map_location=self.device)
            if filename in ('v.pt'):
                self.v.model = torch.load(os.path.join(dirname, filename),
                                          map_location=self.device)


class PPOLazyAgent(LazyAgent):
    """
    Agent class for sampler.
    """

    def __init__(self, policy_model, feature_model, *args, **kwargs):
        self._feature_model = feature_model
        self._policy_model = policy_model
        super().__init__(*args, **kwargs)
        if self._evaluation:
            self._feature_model.eval()
            self._policy_model.eval()

    def act(self, states, reward):
        super().act(states, reward)
        self._states = states
        with torch.no_grad():
            if self._evaluation:
                outputs = self._policy_model(self._feature_model(states),
                                             return_mean=True)
            else:
                outputs = self._policy_model(
                    self._feature_model(states)).sample()
            self._actions = Action(outputs).to("cpu")
        return self._actions


def ppo_continuous(
        # Common settings
        discount_factor=0.98,
        # Adam optimizer settings
        lr=3e-4,  # Adam learning rate
        eps=1e-5,  # Adam stability
        # Loss scaling
        entropy_loss_scaling=0.0,
        value_loss_scaling=0.5,
        # Replay Buffer settings
        replay_start_size=5000,
        # Training settings
        clip_grad=0.5,
        epsilon=0.2,
        minibatches=4,
        epochs=2,
        # GAE settings
        lam=0.95,
):
    """
    PPO continuous control preset.
    Args:
        discount_factor (float): Discount factor for future rewards.
        lr (float): Learning rate for the Adam optimizer.
        eps (float): Stability parameters for the Adam optimizer.
        entropy_loss_scaling (float):
            Coefficient for the entropy term in the total loss.
        value_loss_scaling (float): Coefficient for the value function loss.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        clip_grad (float):
            The maximum magnitude of the gradient for any given parameter.
            Set to 0 to disable.
        epsilon (float):
            Epsilon value in the clipped PPO objective function.
        minibatches (int): The number of minibatches to split each batch into.
        lam (float): The Generalized Advantage Estimate (GAE) decay parameter.
    """
    def _ppo(env):
        enable_on_policy_mode()

        device = get_device()
        feature_model, value_model, policy_model = fc_actor_critic(env)
        feature_model.to(device)
        value_model.to(device)
        policy_model.to(device)

        feature_optimizer = Adam(
            feature_model.parameters(), lr=lr, eps=eps
        )
        value_optimizer = Adam(value_model.parameters(), lr=lr, eps=eps)
        policy_optimizer = Adam(policy_model.parameters(), lr=lr, eps=eps)

        feature_nw = FeatureNetwork(
            feature_model,
            feature_optimizer,
            clip_grad=clip_grad,
        )
        v = VNetwork(
            value_model,
            value_optimizer,
            loss_scaling=value_loss_scaling,
            clip_grad=clip_grad,
        )
        policy = GaussianPolicy(
            policy_model,
            policy_optimizer,
            env.action_space,
            clip_grad=clip_grad,
        )

        replay_buffer = ExperienceReplayBuffer(1e7, env)
        replay_buffer = GaeWrapper(replay_buffer, discount_factor, lam)
        set_replay_buffer(replay_buffer)

        return PPO(
            feature_nw,
            v,
            policy,
            epsilon=epsilon,
            replay_start_size=replay_start_size,
            minibatches=minibatches,
            entropy_loss_scaling=entropy_loss_scaling,
        )

    return _ppo


__all__ = ["ppo_continuous"]