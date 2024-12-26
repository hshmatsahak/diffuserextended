from collections import namedtuple
import os
import numpy as np
import torch
import pdb

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer

Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch

class GoalDataset(SequenceDataset):

    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon - 1: observations[-1],
        }

class MidGoalDataset(SequenceDataset):
    
    def get_conditions(self, observations):
        '''
            condition on both the current observation and the last observation in the plan
        '''
        return {
            0: observations[0],
            self.horizon//2 : observations[self.horizon//2],
            self.horizon - 1: observations[-1],
        }


class ObstacleDataset(SequenceDataset):
    '''
    Adds two stationary obstacles to each trajectory.
    Obstacles are represented as additional features and included in the conditions under the 'obstacles' key.
    '''

    def __init__(self, *args, obstacle_dim=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.obstacle_dim = obstacle_dim
        print(self.n_episodes)
        self.obstacles = self.generate_obstacles(self.n_episodes)  # Generate obstacles for all episodes
        print(self.obstacles.shape)

    def generate_obstacles(self, n_episodes):
        '''
        Generate two stationary obstacles for each episode.
        Each obstacle is represented as a vector of size `obstacle_dim`.
        '''
        obstacle1 = np.random.uniform(-1.0, 1.0, size=(n_episodes, self.obstacle_dim)).astype(np.float32)
        print(obstacle1.shape)
        obstacle2 = np.random.uniform(-1.0, 1.0, size=(n_episodes, self.obstacle_dim)).astype(np.float32)
        return np.stack([obstacle1, obstacle2], axis=1)  # Shape: [n_episodes, 2, obstacle_dim]

    def get_conditions(self, observations, path_idx):
        '''
        Add obstacles to the conditions.
        '''
        base_conditions = super().get_conditions(observations)
        base_conditions.update({
            'obstacles': self.obstacles[path_idx],  # Add both obstacles as a single key
        })
        return base_conditions

    def __getitem__(self, idx):
        '''
        Return a batch with obstacles integrated into the conditions.
        '''
        # Base trajectory and conditions
        path_ind, start, end = self.indices[idx]
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        # Add obstacles to the trajectory as new features
        trajectory = np.concatenate([actions, observations], axis=-1)  # [horizon, action_dim + observation_dim]

        # Update conditions with obstacles
        conditions = self.get_conditions(observations, path_ind)

        # Return updated batch
        return Batch(trajectories=trajectory, conditions=conditions)
    

class PrecomputedDualTrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self, precomputed_file='/home/hsa/Documents/CSC2626/final/diffuser/dual_trajectory_dataset_10k.npz', action_dim=2, observation_dim=4, horizon=64, env='hopper-medium-replay',
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True):
        
        self.singledataset = SequenceDataset(env, horizon, normalizer, preprocess_fns, max_path_length, max_n_episodes, termination_penalty, use_padding)
        """
        Initialize the dataset from a precomputed file.
        """
        assert os.path.exists(precomputed_file), f"File not found: {precomputed_file}"
        data = np.load(precomputed_file, allow_pickle=True)
        self.trajectory1 = data['trajectory1']  # Shape: [N, horizon, action_dim + observation_dim]
        self.trajectory2 = data['trajectory2']  # Shape: [N, horizon, action_dim + observation_dim]
        self.conditions = data['conditions']    # Shape: [N]
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.horizon = horizon

        # Combine trajectories for batch compatibility
        self.combined_trajectories = self._combine_trajectories()

    def _combine_trajectories(self):
        """
        Combine trajectory1 and trajectory2 along the horizon axis.
        The combined format will allow for consistent batching.
        """
        combined = np.concatenate([self.trajectory1, self.trajectory2], axis=2)  # [N, 2*horizon, action_dim + observation_dim]
        return combined

    def __len__(self):
        return len(self.trajectory1)

    def __getitem__(self, idx):
        """
        Return a batch-like format compatible with training code.
        """
        combined_trajectory = self.combined_trajectories[idx]  # Shape: [2*horizon, action_dim + observation_dim]
        conditions = self.conditions[idx]                     # Conditions for both trajectories

        # Create a Batch object
        batch = Batch(trajectories=combined_trajectory, conditions=conditions)
        return batch

class ValueDataset(SequenceDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch
