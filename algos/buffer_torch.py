from itertools import chain
import torch
import numpy as np


class Trajectory:
    def __init__(self):
        self.observations = []
        self.next_observations = []
        self.obs_diff = []
        self.actions = []
        self.rewards = []
        self.costs = []
        self.done = False


    def __len__(self):
        return len(self.observations)


class Buffer:
    def __init__(self, trajectories):
        self.trajectories = trajectories

    def sample(self, next=False):
        observations_diff = torch.cat([torch.stack(trajectory.obs_diff) for trajectory in self.trajectories])
        observations = torch.cat([torch.stack(trajectory.observations) for trajectory in self.trajectories])
        actions = torch.cat([torch.stack(trajectory.actions) for trajectory in self.trajectories])
        rewards = torch.cat([torch.tensor(trajectory.rewards) for trajectory in self.trajectories])
        costs = torch.cat([torch.tensor(trajectory.costs) for trajectory in self.trajectories])

        next_observations = torch.cat([torch.stack(trajectory.next_observations) for trajectory in self.trajectories])

        if next:
            return observations, actions, rewards, costs, next_observations, observations_diff
        else:
            return observations, actions, rewards, costs

    def __getitem__(self, i):
        return self.trajectories[i]