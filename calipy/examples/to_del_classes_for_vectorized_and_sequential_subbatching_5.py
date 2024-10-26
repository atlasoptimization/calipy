# This provides functionality for writing a model with a vectorizable flag where
# the model code actually does not branch depending on the flag. We investigate 
# here, how this can be used to perform subbatching without a model-rewrite
# We make our life a bit easier by only considering one batch dim for now.
# We feature nontrivial event_shape and functorch.dim based indexing.



import pyro
import pyro.distributions as dist
import torch
import contextlib
import itertools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functorch.dim import dims

import numpy as np
import random
import matplotlib.pyplot as plt

# i) Generate synthetic data

n_data_1 = 21
n_data_2 = 1
batch_shape = [n_data_1]
n_event = 2
n_event_dims = 1
n_data_total = n_data_1 + n_data_2
torch.manual_seed(42)
pyro.set_rng_seed(42)

mu_true = torch.zeros([1,2])
sigma_true = torch.tensor([[1,0],[0,0.01]])

extension_tensor = torch.ones([n_data_1,1])
# extension_tensor = torch.ones([n_data_1, n_data_2])
data_dist = pyro.distributions.MultivariateNormal(loc = mu_true * extension_tensor, covariance_matrix = sigma_true)
data = data_dist.sample()


# ii) Build dataloader

class SubbatchDataset(Dataset):
    def __init__(self, data, subsample_size, batch_shape=None, event_shape=None):
        self.data = data
        self.subsample_size = subsample_size

        # Determine batch_shape and event_shape
        if batch_shape is None:
            batch_dims = len(subsample_size)
            self.batch_shape = data.shape[:batch_dims]
        else:
            self.batch_shape = batch_shape

        if event_shape is None:
            self.event_shape = data.shape[len(self.batch_shape):]
        else:
            self.event_shape = event_shape

        # Compute number of blocks using ceiling division to include all data
        self.num_blocks = [
            (self.batch_shape[i] + subsample_size[i] - 1) // subsample_size[i]
            for i in range(len(subsample_size))
        ]
        self.block_indices = list(itertools.product(*[range(n) for n in self.num_blocks]))
        random.shuffle(self.block_indices)

    def __len__(self):
        return len(self.block_indices)

    def __getitem__(self, idx):
        block_idx = self.block_indices[idx]
        slices = []
        indices_ranges = []
        for i, (b, s) in enumerate(zip(block_idx, self.subsample_size)):
            start = b * s
            end = min(start + s, self.batch_shape[i])
            slices.append(slice(start, end))
            indices_ranges.append(torch.arange(start, end))
        # Include event dimensions in the slices
        slices.extend([slice(None)] * len(self.event_shape))
        data_block = self.data[tuple(slices)]
        meshgrid = torch.meshgrid(*indices_ranges, indexing='ij')
        indices = torch.stack(meshgrid, dim=-1).reshape(-1, len(self.subsample_size))
        return data_block, indices

# Usage example
subsample_sizes = [20]
subbatch_dataset = SubbatchDataset(data, subsample_sizes, batch_shape=batch_shape, event_shape=[2])
subbatch_dataloader = DataLoader(subbatch_dataset, batch_size=None)


index_list = []
datablock_list = []
mean_list = []
for data_block, index_tensor in subbatch_dataloader:
    print("Data block shape:", data_block.shape)
    print("Index tensor shape:", index_tensor.shape)
    print("data block:", data_block)
    print("Indices:", index_tensor)
    index_list.append(index_tensor)
    datablock_list.append(data_block)
    mean_list.append(torch.mean(data_block,0))
    print("----")
    
    
# iii) Support classes

class CalipyObservation:
    """
    Stores observations along with batch and event dimensions.
    Each observation has a unique name for reference during sampling.
    """
    def __init__(self, observations, plate_names, batch_shape, event_shape):
        self.observations = observations  # Should be a tensor
        self.batch_shape = batch_shape
        self.event_shape = event_shape
        self.plate_names = plate_names
        self.index_to_name = self._generate_index_to_name()

    def _generate_index_to_name(self):
        # Map indices to unique names
        indices = list(itertools.product(*[range(size) for size in self.batch_shape]))
        index_to_name = {}
        for idx in indices:
            idx_str = '_'.join(f"plate_{self.plate_names[i]}_sample_{idx[i]}" for i in range(len(self.plate_names)))
            index_to_name[idx] = idx_str
        return index_to_name

    def get_observation(self, idx):
        # Retrieve observation and its unique name
        obs = self.observations[idx]
        name = self.index_to_name[idx]
        return name, obs
    

class CalipySample:
    """
    Holds sampled data, preserving batch and event dimensions.
    Provides methods for data access and manipulation.
    """    
    
    def __init__(self, data, batch_shape, event_shape, vectorizable):
        self.data = data
        self.event_shape = event_shape
        self.batch_shape = batch_shape
        self.vectorizable = vectorizable

    def __iter__(self):
        if self.vectorizable:
            # Flatten the data and iterate over it
            return iter(self.data.reshape(-1))
        else:
            # Data is already a list of samples
            return iter(self.data)

    def __getitem__(self, idx):
        if self.vectorizable:
            return self.data[idx]
        else:
            return self.data[idx]

    def as_tensor(self):
        if self.vectorizable:
            return self.data
        else:
            return torch.stack(self.data).reshape(self.batch_shape + self.event_shape)
    
def calipy_sample(name, dist, plate_names, plate_sizes, vectorizable=True, obs=None, subsample_indices=None):
    """
    Flexible sampling function handling multiple plates and four cases based on obs and subsample_indices.

    Parameters:
    -----------
    name : str
        Base name for the sample site.
    dist : pyro.distributions.Distribution
        The distribution to sample from.
    plate_names : list of str
        Names of the plates (batch dimensions).
    plate_sizes : list of int
        Sizes of the plates (batch dimensions).
    vectorizable : bool, optional
        If True, uses vectorized sampling. If False, uses sequential sampling. Default is True.
    obs : CalipyObservation or None, optional
        Observations wrapped in CalipyObservation. If provided, sampling is conditioned on these observations.
    subsample_indices : list of torch.Tensor or None, optional
        Subsample indices for each plate dimension. If provided, sampling is performed over these indices.

    Returns:
    --------
    CalipySample
        The sampled data, preserving batch and event dimensions.
    """
    event_shape = dist.event_shape
    n_plates = len(plate_sizes)

    if vectorizable == True:
        # Vectorized sampling using pyro.plate
        with contextlib.ExitStack() as stack:
            # Determine dimensions for plates
            batch_dims_from_event = [-(n_plates + len(event_shape)) + i + 1 for i in range(n_plates)]
            batch_dims_from_right = [-(n_plates ) + i + 1 for i in range(n_plates)]
            current_obs = obs.observations if obs is not None else None

            # Handle multiple plates
            for i, (plate_name, plate_size, dim) in enumerate(zip(plate_names, plate_sizes, batch_dims_from_event)):
                subsample = subsample_indices[i] if subsample_indices is not None else None
                size = plate_size
                stack.enter_context(pyro.plate(plate_name, size=size, subsample=subsample, dim=dim))

                # Index observations if subsampling
                if current_obs is not None and subsample is not None:
                    current_obs = torch.index_select(current_obs, dim=batch_dims_from_right[i], index=subsample)

            # Sample data
            data = pyro.sample(name, dist, obs=current_obs)
            batch_shape = data.shape[:n_plates]
            return CalipySample(data, batch_shape, event_shape, vectorizable=True)
    elif vectorizable == False:
            # Non-vectorized sampling
            samples = []
            batch_ranges = []
    
            # Determine ranges for each plate
            if subsample_indices is not None:
                batch_ranges = [idx.tolist() for idx in subsample_indices]
            else:
                batch_ranges = [list(range(size)) for size in plate_sizes]
    
            # Iterate over all combinations of indices
            for idx in itertools.product(*batch_ranges):
                idx_dict = {plate_names[i]: idx[i] for i in range(n_plates)}
                # Use actual data indices for naming
                idx_str = '_'.join(f"{plate_names[i]}_{idx[i]}" for i in range(n_plates))
    
                # Get observation if available
                if obs is not None:
                    obs_name, obs_value = obs.get_observation(idx)
                    sample_name = f"{obs_name}"
                else:
                    obs_name, obs_value = None, None
                    sample_name = f"{name}_{idx_str}"
    
                # Sample with unique name based on actual data indices
                sample_value = pyro.sample(sample_name, dist, obs=obs_value)
                samples.append((idx, sample_value))
    
            # Construct tensor from samples
            # Need to sort samples based on indices to ensure correct tensor shape
            samples.sort(key=lambda x: x[0])
            sample_values = [s[1] for s in samples]
            batch_shapes = [len(r) for r in batch_ranges]
            data = torch.stack(sample_values).reshape(*batch_shapes, *event_shape)
            batch_shape = data.shape[:n_plates]
            return CalipySample(data, batch_shape, event_shape, vectorizable=False)
    
    
# iv) Test the classes

# # Plate names and sizes
plate_names = ['batch']
plate_sizes = [n_data_1]

obs_object = CalipyObservation(data, plate_names, batch_shape=[n_data_1], event_shape=[2])

# Subsample indices (if any)
subsample_indices = index_list[0]



sample_results = []
# Sample using calipy_sample and check results
for vectorizable in [True, False]:
    for obs in [None, obs_object]:
        for ssi in [None, subsample_indices]:
            
            sample_dist = pyro.distributions.MultivariateNormal(loc = mu_true, covariance_matrix = sigma_true)
            sample_result = calipy_sample('my_sample', sample_dist, plate_names, plate_sizes, vectorizable=vectorizable, obs=obs, subsample_indices=ssi)
            
            vflag = 1 if vectorizable == True else 0
            oflag = 1 if obs is not None else 0
            sflag = 1 if ssi is not None else 0
            
            sample_result.data
            print('vect_{}_obs_{}_ssi_{}_batch_shape'.format(vflag, oflag, sflag), sample_result.batch_shape )
            print('vect_{}_obs_{}_ssi_{}_event_shape'.format(vflag, oflag, sflag), sample_result.event_shape )
            print('vect_{}_obs_{}_ssi_{}_data_shape'.format(vflag, oflag, sflag), sample_result.data.shape )
            print('vect_{}_obs_{}_ssi_{}_data'.format(vflag, oflag, sflag), sample_result.data )
            sample_results.append(sample_result)
            
            
            
# v) Perform inference in different configurations

signature = [0,0,0] # vect, obs, ssi = None or values

































   