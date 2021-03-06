import json, joblib, subprocess, sys, os
import joblib
import shutil
import numpy as np
import torch
import os.path as osp, time, atexit, os
import warnings
import torch.nn.functional as F
<<<<<<< HEAD
import math
from ppo_algos import *
=======
from neural_nets import *
>>>>>>> parent of b84a2cf... gail and vail implementations
from torch import nn

from mpi4py import MPI
import numpy as np

import string



# Default neural network backend for each algo
# (Must be either 'tf1' or 'pytorch')
DEFAULT_BACKEND = {
    'vpg': 'pytorch',
    'trpo': 'tf1',
    'ppo': 'pytorch',
    'ddpg': 'pytorch',
    'td3': 'pytorch',
    'sac': 'pytorch'
}

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching
# experiments.
WAIT_BEFORE_LAUNCH = 5

# EPS
EPS = 1e-8


class Policy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Policy, self).__init__()
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, action_dim)
        self.linear_var = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.leaky_relu(x, 0.001)
        x = self.linear_2(x)
        x = F.leaky_relu(x, 0.001)
        x_mu = self.linear_mu(x)
        x_var = self.linear_var(x)
        return x_mu, x_var

def setup_pytorch_for_mpi():
    """
    Avoid slowdowns caused by each separate process's PyTorch using
    more than its fair share of CPU resources.
    """
    #print('Proc %d: Reporting original number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)
    if torch.get_num_threads()==1:
        return
    fair_num_threads = max(int(torch.get_num_threads() / num_procs()), 1)
    torch.set_num_threads(fair_num_threads)
    #print('Proc %d: Reporting new number of Torch threads as %d.'%(proc_id(), torch.get_num_threads()), flush=True)

def mpi_avg_grads(module):
    """ Average contents of gradient buffers across MPI processes. """
    if num_procs()==1:
        return
    for p in module.parameters():
        p_grad_numpy = p.grad.numpy()   # numpy view of tensor data
        avg_p_grad = mpi_avg(p.grad)
        p_grad_numpy[:] = avg_p_grad[:]


def average_gradients(param_groups):
    for param_group in param_groups:
        for p in param_group['params']:
            if p.requires_grad:
                p.grad.data.copy_(torch.Tensor(mpi_avg(p.grad.data.numpy())))


def sync_params(module):
    """ Sync all parameters of module across all MPI processes. """
    if num_procs()==1:
        return
    for p in module.parameters():
        p_numpy = p.data.numpy()
        broadcast(p_numpy)


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k,v in obj.items()}

        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)

        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]

        elif hasattr(obj,'__name__') and not('lambda' in obj.__name__):
            return convert_json(obj.__name__)

        elif hasattr(obj,'__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k,v in obj.__dict__.items()}
            return {str(obj): obj_dict}

        return str(obj)

def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.
    Also, terminates the original process that launched it.
    Taken almost without modification from the Baselines function of the
    `same name`_.
    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py
    Args:
        n (int): Number of process to split into.
        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n <= 1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        print("mpi args: ", args)
        print("sys args types partial", sys.argv)
        # print("sys args types full", sys.argv[0])
        print("testing paths: ", os.path.abspath(sys.argv[0]))
        # print("env: ", env)
        subprocess.check_call(args, env=env)
        sys.exit()


def msg(m, string=''):
    print(('Message from %d: %s \t ' % (MPI.COMM_WORLD.Get_rank(), string)) + str(m))


def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()


def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()


def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.
    Args:
        x: An array containing samples of the scalar to produce statistics
            for.
        with_min_and_max (bool): If true, return min and max of x in
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean) ** 2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

# Utils


def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()


def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()

"""
Conjugate gradient
"""


def cg(Ax, b, cg_iters=10):
    x = np.zeros_like(b)
    r = b.copy()  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = r.copy()
    r_dot_old = np.dot(r, r)
    for _ in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r, r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x, [x0, x1,  x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]



class Logger:
    """
    A general-purpose logger.
    Makes it easy to save diagnostics, hyperparameter configurations, the
    state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='progress.txt', exp_name=None):
        """
        Initialize a Logger.
        Args:
            output_dir (string): A directory for saving results to. If
                ``None``, defaults to a temp directory of the form
                ``/tmp/experiments/somerandomnumber``.
            output_fname (string): Name for the tab-separated-value file
                containing metrics logged throughout a training run.
                Defaults to ``progress.txt``.
            exp_name (string): Experiment name. If you run multiple training
                runs and give them all the same ``exp_name``, the plotter
                will know to group them. (Use case: if you run the same
                hyperparameter configuration with multiple random seeds, you
                should give them all the same ``exp_name``.)
        """
        if proc_id() == 0:
            self.output_dir = output_dir or "/tmp/experiments/%i" % int(time.time())
            if osp.exists(self.output_dir):
                print("Warning: Log dir %s already exists! Storing info there anyway." % self.output_dir)
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(osp.join(self.output_dir, output_fname), 'w')
            print("here is the output file: ", self.output_file)

            atexit.register(self.output_file.close)
            print(colorize("Logging data to %s" % self.output_file.name, 'green', bold=True))
        else:
            self.output_dir = None
            self.output_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    def log(self, msg, color='green'):
        """Print a colorized message to stdout."""
        if proc_id() == 0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.
        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration" % key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.
        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible).
        Example use:
        .. code-block:: python
            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if proc_id() == 0:
            output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
            print(colorize('Saving config:\n', color='cyan', bold=True))
            print(output)
            with open(osp.join(self.output_dir, "config.json"), 'w') as out:
                out.write(output)

    def save_state(self, state_dict, models, itr=None):
        """
        Saves the state of an experiment.
        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function. This function
        will save whatever is in ``state_dict``---usually just a copy of the
        environment---and the most recent parameters for the model you
        previously set up saving for with ``setup_tf_saver``.
        Call with any frequency you prefer. If you only want to maintain a
        single state and overwrite it at each call with the most recent
        version, leave ``itr=None``. If you want to keep all of the states you
        save, provide unique (increasing) values for 'itr'.
        Args:
            state_dict (dict): Dictionary containing essential elements to
                describe the current state of training.
            itr: An int, or None. Current iteration of training.
        """
        if proc_id() == 0:
            fname = 'vars.pkl' if itr is None else 'vars%d.pkl' % itr
            try:
                joblib.dump(state_dict, osp.join(self.output_dir, fname))
            except:
                self.log('Warning: could not pickle state_dict.', color='red')
            # if hasattr(self, 'pytorch_saver_elements'):
            # if models is not None:
            #     for m in models:
            #         self._pytorch_simple_save(str(m), itr)
            # else:
            self._pytorch_simple_save(itr)

    def setup_pytorch_saver(self, what_to_save):
        """
        Set up easy model saving for a single PyTorch model.
        Because PyTorch saving and loading is especially painless, this is
        very minimal; we just need references to whatever we would like to
        pickle. This is integrated into the logger because the logger
        knows where the user would like to save information about this
        training run.
        Args:
            what_to_save: Any PyTorch model or serializable object containing
                PyTorch models.
        """
        # print("here is what to save")
        # print(what_to_save)
        self.pytorch_saver_elements = what_to_save

    def _pytorch_simple_save(self, fname=None, itr=None):
        """
        Saves the PyTorch model (or models).
        """
        if proc_id() == 0:
            assert hasattr(self, 'pytorch_saver_elements'), \
                "First have to setup saving with self.setup_pytorch_saver"
            fpath = 'pyt_save'
            fpath = osp.join(self.output_dir, fpath)
            if fname is None:
                fname = 'model' + ('%d' % itr if itr is not None else '') + '.pt'
                fname = osp.join(fpath, fname)
            os.makedirs(fpath, exist_ok=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # We are using a non-recommended way of saving PyTorch models,
                # by pickling whole objects (which are dependent on the exact
                # directory structure at the time of saving) as opposed to
                # just saving network weights. This works sufficiently well
                # for the purposes of Spinning Up, but you may want to do
                # something different for your personal PyTorch project.
                # We use a catch_warnings() context to avoid the warnings about
                # not being able to save the source code.
                # print("what are the elements here")
                # print(self.pytorch_saver_elements)
                # print(fname)
                torch.save(self.pytorch_saver_elements, fname)

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.
        Writes both to stdout, and to the output file.
        """
        if proc_id() == 0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = '%' + '%d' % max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len
            print("-" * n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g" % val if hasattr(val, "__float__") else val
                print(fmt % (key, valstr))
                vals.append(val)
            print("-" * n_slashes, flush=True)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers) + "\n")
                self.output_file.write("\t".join(map(str, vals)) + "\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.
    Typical use case: there is some quantity which is calculated many times
    throughout an epoch, and at the end of the epoch, you would like to
    report the average / std / min / max value of that quantity.
    With an EpochLogger, each time the quantity is calculated, you would
    use
    .. code-block:: python
        epoch_logger.store(NameOfQuantity=quantity_value)
    to load it into the EpochLogger's state. Then at the end of the epoch, you
    would use
    .. code-block:: python
        epoch_logger.log_tabular(NameOfQuantity, **options)
    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.
        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.
        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.
            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.
            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.
            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
            stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)

            super().log_tabular(key if average_only else 'Average' + key, stats[0])
            if not (average_only):
                super().log_tabular('Std' + key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max' + key, stats[3])
                super().log_tabular('Min' + key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
        return mpi_statistics_scalar(vals)


DIV_LINE_WIDTH = 80


def raw_numpy(raw):
    # return raw: np.array and done: np.array
    _mask = torch.ones(len(raw), dtype=torch.bool) # Tyna change this mask eventually
    done= ~_mask
    return raw, done


def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class Samples:
    def __init__(self, states=None, actions=None, rewards=None,
                 next_states=None, weights=None, indexes=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        # self.weights = weights
        # self.indexes = indexes
        self._keys = [self.states, self.actions, self.rewards,
                      self.next_states \
            # , self.weights, self.indexes
                      ]

    def __iter__(self):
        return iter(self._keys)


def samples_to_np(samples):

    np_states, np_dones = raw_numpy(samples.states)

    np_actions = samples.actions
    np_rewards = samples.rewards
    np_next_states, np_next_dones = samples.next_states
    return np_states, np_rewards, np_actions, np_next_states, np_dones, np_next_dones


def samples_from_cpprb(npsamples, device=None):
    """
    Convert samples generated by cpprb.ReplayBuffer.sample() into
    State, Action, rewards, State.
    Return Samples object.
    Args:
        npsamples (dict of nparrays):
            Samples generated by cpprb.ReplayBuffer.sample()
        device (optional): The device where the outputs are loaded.
    Returns:
        Samples(State, Action, torch.FloatTensor, State)
    """
    # device = self.device if device is None else device

    states = npsamples["obs"]
    actions = npsamples["act"]
    rewards = torch.tensor(npsamples["rew"], dtype=torch.float32).squeeze()
    next_states = npsamples["next_obs"], npsamples["done"]

    return Samples(states, actions, rewards, next_states)


class RLNetwork(nn.Module):
    """
    Wraps a network such that States can be given as input.
    """

    def __init__(self, model, _=None):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, state):
        return self.model(state.features.float()) * state.mask.float().unsqueeze(-1)

    def to(self, device):
        self.device = device
        return super().to(device)


# class GaussianPolicyNetwork(RLNetwork):
#     def __init__(self, model, space):
#         super().__init__(model)
#         self._action_dim = space.shape[0]
#
#     def forward(self, state, return_mean=False):
#         outputs = super().forward(state)
#         means = outputs[:, :self._action_dim]
#
#         if return_mean:
#             return means
#
#         logvars = outputs[:, self._action_dim:]
#         std = logvars.exp_()
#         return Independent(Normal(means, std), 1)
#
#     def to(self, device):
#         return super().to(device)


# Train
class MyModel:
    def __init__(self):
        self._weights = 0

    def get_action(self, obs):
        # Implement action selection
        return 0

    def abs_TD_error(self, sample):
        # Implement absolute TD error
        return np.zeros(sample["obs"].shape[0])

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, w):
        self._weights = w

    def train(self, sample):
        # Implement model update
        pass


def setup_logger_kwargs(exp_name, seed=None, data_dir=None, datestamp=False):
    """
    Sets up the output_dir for a logger and returns a dict for logger kwargs.
    If no seed is given and datestamp is false,
    ::
        output_dir = data_dir/exp_name
    If a seed is given and datestamp is false,
    ::
        output_dir = data_dir/exp_name/exp_name_s[seed]
    If datestamp is true, amend to
    ::
        output_dir = data_dir/YY-MM-DD_exp_name/YY-MM-DD_HH-MM-SS_exp_name_s[seed]
    You can force datestamp=True by setting ``FORCE_DATESTAMP=True`` in
    ``spinup/user_config.py``.
    Args:
        exp_name (string): Name for experiment.
        seed (int): Seed for random number generators used by experiment.
        data_dir (string): Path to folder where results should be saved.
            Default is the ``DEFAULT_DATA_DIR`` in ``spinup/user_config.py``.
        datestamp (bool): Whether to include a date and timestamp in the
            name of the save directory.
    Returns:
        logger_kwargs, a dict containing output_dir and exp_name.
    """

    # Datestamp forcing
    datestamp = datestamp or FORCE_DATESTAMP

    # Make base path
    ymd_time = time.strftime("%Y-%m-%d_") if datestamp else ''
    relpath = ''.join([ymd_time, exp_name])

    if seed is not None:
        # Make a seed-specific subfolder in the experiment directory.
        if datestamp:
            hms_time = time.strftime("%Y-%m-%d_%H-%M-%S")
            subfolder = ''.join([hms_time, '-', exp_name, '_s', str(seed)])
        else:
            subfolder = ''.join([exp_name, '_s', str(seed)])
        relpath = osp.join(relpath, subfolder)
        print("relative path: ", relpath)

    print("default data dir: ", DEFAULT_DATA_DIR)

    data_dir = data_dir or DEFAULT_DATA_DIR
    logger_kwargs = dict(output_dir=osp.join(data_dir, relpath),
                         exp_name=exp_name)
    return logger_kwargs


def all_bools(vals):
    return all([isinstance(v, bool) for v in vals])


def valid_str(v):
    """
    Convert a value or values to a string which could go in a filepath.
    Partly based on `this gist`_.
    .. _`this gist`: https://gist.github.com/seanh/93666
    """
    if hasattr(v, '__name__'):
        return valid_str(v.__name__)

    if isinstance(v, tuple) or isinstance(v, list):
        return '-'.join([valid_str(x) for x in v])

    # Valid characters are '-', '_', and alphanumeric. Replace invalid chars
    # with '-'.
    str_v = str(v).lower()
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    str_v = ''.join(c if c in valid_chars else '-' for c in str_v)
    return str_v


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting with the environment,
    and using Generalized Advantage Estimation (GAE-Lambda) for calculating the advantages of state-action pairs.

    Generalized Advantage Estimation: Forks the different runs across cores, message passes
    using mpi_fork, calculates the average of advantage, then descends.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, cost_gamma=0.99, cost_lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)

        self.cadv_buf = np.zeros(size, dtype=np.float32)    # cost advantage
        self.cost_buf = np.zeros(size, dtype=np.float32)    # costs
        self.cret_buf = np.zeros(size, dtype=np.float32)    # cost return
        self.cval_buf = np.zeros(size, dtype=np.float32)    # cost value

        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    # def store(self, obs, act, rew, val, cost, cval, logp, done, next_obs):
    def store(self, obs, act, rew, val, cost, cval, logp, next_obs):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val

        self.cost_buf[self.ptr] = cost
        self.cval_buf[self.ptr] = cval
        self.logp_buf[self.ptr] = logp

        # self.rb.add(obs=obs, act=act, rew=rew, next_obs=next_obs, done=done)
        # self.rb.add(obs=obs, act=act, rew=rew, next_obs=next_obs)

        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        :param last_cval:
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        # print("Path slice")
        # print(path_slice)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        costs = np.append(self.cost_buf[path_slice], last_cval)
        cvals = np.append(self.cval_buf[path_slice], last_cval)

        # implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]

        # print("Advantage buffer")
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        # print(self.adv_buf)
        # computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        # implement GAE-Lambda advantage for costs
        cdeltas = costs[:-1] + self.gamma * cvals[1:] - cvals[:-1]
        self.cadv_buf[path_slice] = discount_cumsum(cdeltas, self.cost_gamma * self.cost_lam)
        # computes rewards-to-go
        self.cret_buf[path_slice] = discount_cumsum(costs, self.cost_gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next four lines implement the advantage normalization trick, for rewards and costs
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std

        cadv_mean, _ = mpi_statistics_scalar(self.cadv_buf)
        # Center, but do NOT rescale advantages for cost gradient # Tyna Note
        self.cadv_buf -= cadv_mean

        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, cadv=self.cadv_buf,
                    cret=self.cret_buf)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


class CostPOBuffer:

    def __init__(self,
                     obs_dim,
                     act_dim,
                     size,
                     gamma=0.99,
                     lam=0.95,
                     cost_gamma=0.99,
                     cost_lam=0.95):

        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)

        self.cadv_buf = np.zeros(size, dtype=np.float32)  # cost advantage
        self.cost_buf = np.zeros(size, dtype=np.float32)  # costs
        self.cret_buf = np.zeros(size, dtype=np.float32)  # cost return
        self.cval_buf = np.zeros(size, dtype=np.float32)  # cost value

        self.logp_buf = np.zeros(size, dtype=np.float32)
        # self.pi_info_bufs = {k: np.zeros([size] + list(v), dtype=np.float32)
        #                      for k,v in pi_info_shapes.items()}
        # self.sorted_pi_info_keys = keys_as_sorted_list(self.pi_info_bufs)
        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size




    def store(self, obs, act, rew, val, cost, cval, logp, pi_info):
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.cost_buf[self.ptr] = cost
        self.cval_buf[self.ptr] = cval
        self.logp_buf[self.ptr] = logp
        # for k in self.sorted_pi_info_keys:
        #     self.pi_info_bufs[k][self.ptr] = pi_info[k]
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        costs = np.append(self.cost_buf[path_slice], last_cval)
        cvals = np.append(self.cval_buf[path_slice], last_cval)
        cdeltas = costs[:-1] + self.gamma * cvals[1:] - cvals[:-1]
        self.cadv_buf[path_slice] = discount_cumsum(cdeltas, self.cost_gamma * self.cost_lam)
        self.cret_buf[path_slice] = discount_cumsum(costs, self.cost_gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0

        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, cadv=self.cadv_buf,
                    cret=self.cret_buf)

        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}

    # def sample(self, *args, **kwargs):
    #     return self.buffer.sample(*args, **kwargs)

"""
Conjugate gradient
"""

def cg(Ax, b, cg_iters=10):
    x = np.zeros_like(b)
    r = b.copy() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = r.copy()
    r_dot_old = np.dot(r,r)
    for _ in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r,r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x




# from https://github.com/joschu/modular_rl
# http://www.johndcook.com/blog/standard_deviation/

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM)/self._n
            self._S[...] = self._S + (x - oldM)*(x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S/(self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape


class ZFilter:
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)

        if self.demean:
            x = x - self.rs.mean

        if self.destd:
            x = x / (self.rs.std + 1e-8)

        if self.clip:
            x = np.clip(x, -self.clip, self.clip)

        return x




def fc_q(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
    )


def fc_v(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
    )


def fc_deterministic_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0]),
    )


def fc_deterministic_noisy_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.NoisyFactorizedLinear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.NoisyFactorizedLinear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.NoisyFactorizedLinear(hidden2, env.action_space.shape[0]),
    )


def fc_soft_policy(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2),
    )


def fc_actor_critic(env, hidden1=400, hidden2=300):
    features = nn.Sequential(
        nn.Linear(env.state_space.shape[0], hidden1),
        nn.LeakyReLU(),
    )

    v = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1)
    )

    policy = nn.Sequential(
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, env.action_space.shape[0] * 2)
    )

    return features, v, policy


def fc_discriminator(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] + env.action_space.shape[0],
                  hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1),
        nn.Sigmoid())


def fc_reward(env, hidden1=400, hidden2=300):
    return nn.Sequential(
        nn.Linear(env.state_space.shape[0] +
                  env.action_space.shape[0], hidden1),
        nn.LeakyReLU(),
        nn.Linear(hidden1, hidden2),
        nn.LeakyReLU(),
        nn.Linear(hidden2, 1)
    )



def sync_all_params(param, root=0):
    data = torch.nn.utils.parameters_to_vector(param).detach().numpy()
    broadcast(data, root)
    torch.nn.utils.vector_to_parameters(torch.from_numpy(data), param)





class Buffer(object):
    def __init__(self, obs_dim, act_dim, batch_size, ep_len):
        self.max_batch = batch_size
        self.max_volume = batch_size * ep_len
        self.obs_dim = obs_dim

        self.obs = np.zeros((self.max_volume, obs_dim))
        self.act = np.zeros((self.max_volume, act_dim))
        self.rew = np.zeros(self.max_volume)
        self.end = np.zeros(batch_size + 1) # The first term will always be 0 / boundries of trajectories

        self.ptr = 0
        self.eps = 0

    def store(self, obs, act, rew, sdr, val, lgp):
        raise NotImplementedError

    def end_episode(self, last_val=0):
        raise NotImplementedError

    def retrieve_all(self):
        raise NotImplementedError

# Buffer for training an expert
class BufferActor(Buffer):
    def __init__(self, obs_dim, act_dim, batch_size, ep_len, gamma=0.99, lam=0.95):
        super(BufferActor, self).__init__(obs_dim, act_dim, batch_size, ep_len)

        self.ret = np.zeros(self.max_volume)
        self.val = np.zeros(self.max_volume)
        self.adv = np.zeros(self.max_volume)
        self.lgp = np.zeros(self.max_volume) # Log prob of selected actions, used for entropy estimation

        self.gamma = gamma
        self.lam = lam

    def store(self, obs, act, rew, val, lgp):
        assert self.ptr < self.max_volume
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.lgp[self.ptr] = lgp
        self.ptr += 1

    def finish_path(self, last_val=0):
        ep_slice = slice(int(self.end[self.eps]), self.ptr)
        rewards = np.append(self.rew[ep_slice], last_val)
        values = np.append(self.val[ep_slice], last_val)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        returns = scipy.signal.lfilter([1], [1, float(-self.gamma)], rewards[::-1], axis=0)[::-1]
        self.ret[ep_slice] = returns[:-1]
        self.adv[ep_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma * self.lam)], deltas[::-1], axis=0)[::-1]

        self.eps += 1
        self.end[self.eps] = self.ptr

    # def finish_path(self, last_val=0):
    #     path_slice = slice(self.path_start_idx, self.ptr)
    #     rews = np.append(self.rew_buf[path_slice], last_val)
    #     vals = np.append(self.val_buf[path_slice], last_val)
    #     deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    #     self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
    #     self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
    #
    #     self.path_start_idx = self.ptr

    def retrieve_all(self):
        assert self.eps == self.max_batch
        occup_slice = slice(0, self.ptr)
        self.ptr = 0
        self.eps = 0

        adv_mean, adv_std = mpi_statistics_scalar(self.adv[occup_slice])
        self.adv[occup_slice] = (self.adv[occup_slice] - adv_mean) / adv_std
        return [self.obs[occup_slice], self.act[occup_slice], self.adv[occup_slice],
            self.ret[occup_slice], self.lgp[occup_slice]]

class BufferStudent(Buffer):
    def __init__(self, obs_dim, act_dim, batch_size, ep_len, gamma=0.99, lam=0.95):
        super(BufferStudent, self).__init__(obs_dim, act_dim, batch_size, ep_len)

        self.sdr = np.zeros(self.max_volume) # Pseudo reward, the log prob
        self.ret = np.zeros(self.max_volume) # Discounted return based on self.sdr
        self.val = np.zeros(self.max_volume)
        self.adv = np.zeros(self.max_volume)
        self.lgp = np.zeros(self.max_volume) # Log prob of selected actions, used for entropy estimation

        self.gamma = gamma
        self.lam = lam

    def store(self, obs, act, rew, sdr, val, lgp):
        assert self.ptr < self.max_volume
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.sdr[self.ptr] = sdr
        self.val[self.ptr] = val
        self.lgp[self.ptr] = lgp
        self.ptr += 1

    def end_episode(self, last_val=0):
        ep_slice = slice(int(self.end[self.eps]), self.ptr)
        rewards = np.append(self.sdr[ep_slice], last_val)
        values = np.append(self.val[ep_slice], last_val)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        returns = scipy.signal.lfilter([1], [1, float(-self.gamma)], rewards[::-1], axis=0)[::-1]
        self.ret[ep_slice] = returns[:-1]
        self.adv[ep_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma * self.lam)], deltas[::-1], axis=0)[::-1]

        self.eps += 1
        self.end[self.eps] = self.ptr

    def retrieve_all(self):
        assert self.eps == self.max_batch
        occup_slice = slice(0, self.ptr)
        self.ptr = 0
        self.eps = 0

        adv_mean, adv_std = mpi_statistics_scalar(self.adv[occup_slice])
        self.adv[occup_slice] = (self.adv[occup_slice] - adv_mean) / adv_std
        return [self.obs[occup_slice], self.act[occup_slice], self.adv[occup_slice],
            self.ret[occup_slice], self.lgp[occup_slice]]

class BufferTeacher(Buffer):
    def __init__(self, obs_dim, act_dim, batch_size, ep_len, gamma=0.99, lam=0.95):
        super(BufferTeacher, self).__init__(obs_dim, act_dim, batch_size, ep_len)

    def store(self, obs, act, rew):
        assert self.ptr < self.max_volume
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew

        self.ptr += 1

    def end_episode(self, last_val=0):
        self.eps += 1
        self.end[self.eps] = self.ptr

    def retrieve_all(self):
        assert self.eps == self.max_batch
        occup_slice = slice(0, self.ptr)
        self.ptr = 0
        self.eps = 0

        return [self.obs[occup_slice], self.act[occup_slice]]



class VALORBuffer(object):
    def __init__(self, con_dim, obs_dim, act_dim, batch_size, ep_len, dc_interv, gamma=0.99, lam=0.95, N=11):
        self.max_batch = batch_size
        self.dc_interv = dc_interv
        self.max_s = batch_size * ep_len
        self.obs_dim = obs_dim
        self.obs = np.zeros((self.max_s, obs_dim + con_dim))
        self.act = np.zeros((self.max_s, act_dim))
        self.rew = np.zeros(self.max_s)
        self.ret = np.zeros(self.max_s)
        self.adv = np.zeros(self.max_s)
        self.pos = np.zeros(self.max_s)
        self.lgt = np.zeros(self.max_s)
        self.val = np.zeros(self.max_s)
        self.end = np.zeros(batch_size + 1) # The first will always be 0
        self.ptr = 0
        self.eps = 0
        self.dc_eps = 0

        self.N = 11

        self.con = np.zeros(self.max_batch * self.dc_interv)
        self.dcbuf = np.zeros((self.max_batch * self.dc_interv, self.N-1, obs_dim))

        self.gamma = gamma
        self.lam = lam

    def store(self, con, obs, act, rew, val, lgt):
        assert self.ptr < self.max_s
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.con[self.eps] = con
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.lgt[self.ptr] = lgt
        self.ptr += 1

    def calc_diff(self):
        # Store differences into a specific memory
        # TODO: convert this into vector operation
        start = int(self.end[self.eps])
        ep_l = self.ptr - start - 1
        for i in range(self.N-1):
            prev = int(i*ep_l/(self.N-1))
            succ = int((i+1)*ep_l/(self.N-1))
            self.dcbuf[self.eps, i] = self.obs[start + succ][:self.obs_dim] - self.obs[start + prev][:self.obs_dim]

        return self.dcbuf[self.eps]

    def finish_path(self, pret_pos, last_val=0): # pret_pos gives the log possibility of cheating the discriminator
        ep_slice = slice(int(self.end[self.eps]), self.ptr)
        rewards = np.append(self.rew[ep_slice], last_val)
        values = np.append(self.val[ep_slice], last_val)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        returns = scipy.signal.lfilter([1], [1, float(-self.gamma)], rewards[::-1], axis=0)[::-1]
        self.ret[ep_slice] = returns[:-1]
        self.adv[ep_slice] = scipy.signal.lfilter([1], [1, float(-self.gamma * self.lam)], deltas[::-1], axis=0)[::-1]
        self.pos[ep_slice] = pret_pos

        self.eps += 1
        self.dc_eps += 1
        self.end[self.eps] = self.ptr

    def retrieve_all(self):
        assert self.eps == self.max_batch
        occup_slice = slice(0, self.ptr)
        self.ptr = 0
        self.eps = 0
        adv_mean, adv_std = mpi_statistics_scalar(self.adv[occup_slice])
        pos_mean, pos_std = mpi_statistics_scalar(self.pos[occup_slice])
        self.adv[occup_slice] = (self.adv[occup_slice] - adv_mean) / adv_std
        self.pos[occup_slice] = (self.pos[occup_slice] - pos_mean) / pos_std
        return [self.obs[occup_slice], self.act[occup_slice], self.adv[occup_slice], self.pos[occup_slice],
            self.ret[occup_slice], self.lgt[occup_slice]]

    def retrieve_dc_buff(self):
        assert self.dc_eps == self.max_batch * self.dc_interv
        self.dc_eps = 0
        return [self.con, self.dcbuf]



# # Set up function for computing PPO policy loss
# def compute_loss_policy(obs, act, adv, logp_old):
#     # Policy loss # policy gradient term + entropy term
#     # Policy loss with clipping (without clipping, loss_pi = -(logp*adv).mean()).
#     # TODO: Think about removing clipping
#     _, logp, _ = ac.pi(obs, act)
#
#     ratio = torch.exp(logp - logp_old)
#     clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
#     loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    # return loss_pi