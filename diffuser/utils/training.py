import os
import copy
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
        exp_name="maze2d",
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        # hshmat - num_workers was 1 originally
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

        self.exp_name = exp_name

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

            if self.step == 0 and self.sample_freq:
                if self.exp_name in ["maze2d", "maze2d_midgoal", "maze2d_condnet"]:     
                    self.render_reference(self.n_reference, False)
                elif self.exp_name == "maze2d_condnet_obstacle":
                    self.render_reference(self.n_reference, True)
                elif self.exp_name == "maze2d_dual":
                    self.render_reference_dual(self.n_reference)
                else:
                    raise ValueError(f"Unrecognized experiment name: {self.exp_name}. Please check the configuration.")

            if self.sample_freq and self.step % self.sample_freq == 0:
                if self.exp_name in ["maze2d", "maze2d_midgoal", "maze2d_condnet"]:     
                    self.render_samples(n_samples=self.n_samples)
                elif self.exp_name == "maze2d_condnet_obstacle":
                    self.render_samples_with_obstacles(n_samples=self.n_samples)
                elif self.exp_name == "maze2d_dual":
                    self.render_samples_dual(n_samples=self.n_samples)
                else:
                    raise ValueError(f"Unrecognized experiment name: {self.exp_name}. Please check the configuration.")

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10, obstacles=False):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        if obstacles:
            conditions = to_np(batch.conditions['obstacles'])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        if obstacles:
            self.renderer.composite(savepath, observations, conditions)
        else:
            self.renderer.composite(savepath, observations)

    def render_reference_dual(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories) 
        conditions = to_np(batch.conditions['start1'])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations1 = trajectories[:, :, self.dataset.action_dim:self.dataset.action_dim+self.dataset.observation_dim]
        observations1 = self.dataset.singledataset.normalizer.unnormalize(normed_observations1, 'observations')
        normed_observations2 = trajectories[:, :, 2*self.dataset.action_dim+self.dataset.observation_dim:]
        observations2 = self.dataset.singledataset.normalizer.unnormalize(normed_observations2, 'observations')

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.dual_composite(savepath, observations1, observations2)
    
    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            # print(batch.shape)
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)

    def render_samples_dual(self, batch_size=2, n_samples=2):
        """
        Renders samples from (ema) diffusion model, showing two trajectories per maze.
        """
        for i in range(batch_size):
            ## Get a single datapoint
            batch = self.dataloader_vis.__next__()
            
            # Get the conditions and move to device
            conditions = to_device(batch.conditions, 'cuda:0')

            # Repeat conditions `n_samples` times for diversity
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            # [n_samples x horizon x (action_dim + observation_dim)]
            samples = self.ema_model.conditional_sample(conditions)
            samples = to_np(samples)

            # Separate into trajectory1 and trajectory2
            observations1 = samples[:, :, self.dataset.action_dim:self.dataset.action_dim+self.dataset.observation_dim]
            observations2 = samples[:, :, 2*self.dataset.action_dim+self.dataset.observation_dim:]

            # Extract normalized conditions
            normed_conditions1 = to_np(batch.conditions['start1'])[:, None]
            normed_conditions2 = to_np(batch.conditions['start2'])[:, None]

            # Add conditions to trajectories
            observations1 = np.concatenate([
                np.repeat(normed_conditions1, n_samples, axis=0),
                observations1
            ], axis=1)

            observations2 = np.concatenate([
                np.repeat(normed_conditions2, n_samples, axis=0),
                observations2
            ], axis=1)

            # Unnormalize the trajectories
            observations1 = self.dataset.singledataset.normalizer.unnormalize(observations1, 'observations')
            observations2 = self.dataset.singledataset.normalizer.unnormalize(observations2, 'observations')

            # Combine and render both trajectories on the same maze
            savepath = os.path.join(self.logdir, f'dualsample-{self.step}-{i}.png')
            self.renderer.dual_composite(savepath, observations1, observations2)

    def render_samples_with_obstacles(self, batch_size=2, n_samples=2):
        """
        Renders samples from (ema) diffusion model with obstacle conditioning.
        """
        for i in range(batch_size):
            ## Get a single datapoint
            batch = self.dataloader_vis.__next__()

            # Get the conditions and preprocess `obstacles`
            conditions = to_device(batch.conditions, 'cuda:0')
            if 'obstacles' in conditions:
                # Flatten `obstacles` for compatibility with `einops.repeat`
                conditions['obstacles'] = einops.rearrange(
                    conditions['obstacles'], 'b n d -> b (n d)'
                )

            # Repeat conditions `n_samples` times for diversity
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d',
                repeat=n_samples,
            )

            # Restore `obstacles` shape to original `[batch, 2, dim]`
            if 'obstacles' in conditions:
                conditions['obstacles'] = einops.rearrange(
                    conditions['obstacles'], 'b (n d) -> b 1 n d',
                    n=2  # Assuming 2 obstacles
                )

            # [n_samples x horizon x (action_dim + observation_dim)]
            samples = self.ema_model.conditional_sample(conditions)
            samples = to_np(samples)

            # Separate trajectories and normalize
            normed_observations = samples[:, :, self.dataset.action_dim:]
            # normed_conditions = to_np(batch.conditions)[:, None]
            
            # # Concatenate conditions for rendering
            # normed_observations = np.concatenate([
            #     np.repeat(normed_conditions, n_samples, axis=0),
            #     normed_observations
            # ], axis=1)

            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            # Render and save
            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations, conditions['obstacles'])

