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
from calipy.core.utils import dim_assignment, generate_trivial_dims, context_plate_stack

import numpy as np
import random
import matplotlib.pyplot as plt

# i) Generate synthetic data

n_data_1 = 5
n_data_2 = 3
batch_shape = [n_data_1, n_data_2]
n_event = 2
n_event_dims = 1
n_data_total = n_data_1 * n_data_2
torch.manual_seed(42)
pyro.set_rng_seed(42)

# # Plate names and sizes
plate_names = ['batch_plate_1', 'batch_plate_2']
plate_sizes = [n_data_1, n_data_2]

mu_true = torch.zeros([1,1,2])
sigma_true = torch.tensor([[1,0],[0,0.01]])

extension_tensor = torch.ones([n_data_1,n_data_2,1])
# extension_tensor = torch.ones([n_data_1, n_data_2])
data_dist = pyro.distributions.MultivariateNormal(loc = mu_true * extension_tensor, covariance_matrix = sigma_true)
data = data_dist.sample()


# functorch dims
batch_dims = dim_assignment(dim_names = ['bd_1', 'bd_2'])
event_dims = dim_assignment(dim_names = ['ed_1'])
data_dims = batch_dims + event_dims


# ii) Build dataloader

class CalipyObservation:
    """
    Stores observations along with batch and event dimensions.
    Each observation has a unique name for reference during sampling.
    """
    def __init__(self, observations, batch_dims, event_dims, subsample_indices = None):

        self.subsample_indices = subsample_indices
        self.batch_dims = batch_dims.get_local_copy()
        self.event_dims = event_dims.get_local_copy()
        self.obs_dims = self.batch_dims + self.event_dims
        self.observations = observations[self.obs_dims]  # Should be a tensor
        # self.plate_names = plate_names
        self.index_to_name_dict = self._generate_index_to_name_dict()
        self.name_to_index_dict = self._generate_name_to_index_dict()
        self.name_to_obs_dict = self._generate_name_to_obs_dict()
        self.index_to_obs_dict = self._generate_index_to_obs_dict()
        

    def _generate_index_to_name_dict(self):
        # Map indices to unique names
        local_indices = list(itertools.product(*[range(size) for size in self.batch_shape]))
        global_indices = self.subsample_indices if self.subsample_indices is not None else local_indices
        index_to_name_dict = {}
        for idx in indices:
            # idx_str = '_'.join("plate_{}_sample_{}".format(self.plate_names[i], idx[i]) for i in range(len(self.plate_names)))
            idx_code = [idx[k].item() for k in range(len(indices.shape))]
            idx_str =  "sample_{}".format(idx_code)
            index_to_name_dict[idx] = idx_str
        return index_to_name_dict
    
    def _generate_name_to_index_dict(self):
        # Map unique names back to indices
        name_to_index_dict = {}
        for index, name in self.index_to_name_dict.items():
            name_to_index_dict[name] = index
        return name_to_index_dict
    
    def _generate_name_to_obs_dict(self):
        # Map observation names to observation values
        name_to_obs_dict = {}
        for obs_name, index in self.name_to_index_dict.items():
            name_to_obs_dict[obs_name] = self.observations[index,...]
        return name_to_obs_dict
    
    def _generate_index_to_obs_dict(self):
        # Map subsample indices to observation values
        pass
    
    def get_observation(self, idx):
        # Retrieve observation and its unique name
        obs = self.observations[idx]
        name = self.index_to_name_dict[idx]
        return name, obs

    def __repr__(self):
        repr_str = 'CalipyObservation object with observation of shape {}'.format(self.observations.shape)
        return repr_str

class SubbatchDataset(Dataset):
    def __init__(self, data, subsample_sizes, batch_shape=None, event_shape=None):
        self.data = data
        self.batch_shape = batch_shape
        self.subsample_sizes = subsample_sizes

        # Determine batch_shape and event_shape
        if batch_shape is None:
            batch_dims = len(subsample_sizes)
            self.batch_shape = data.shape[:batch_dims]
        else:
            self.batch_shape = batch_shape

        if event_shape is None:
            self.event_shape = data.shape[len(self.batch_shape):]
        else:
            self.event_shape = event_shape

        # Compute number of blocks using ceiling division to include all data
        self.num_blocks = [
            (self.batch_shape[i] + subsample_sizes[i] - 1) // subsample_sizes[i]
            for i in range(len(subsample_sizes))
        ]
        self.block_indices = list(itertools.product(*[range(n) for n in self.num_blocks]))
        random.shuffle(self.block_indices)

    def __len__(self):
        return len(self.block_indices)

    def __getitem__(self, idx):
        block_idx = self.block_indices[idx]
        slices = []
        indices_ranges = []
        for i, (b, s) in enumerate(zip(block_idx, self.subsample_sizes)):
            start = b * s
            end = min(start + s, self.batch_shape[i])
            slices.append(slice(start, end))
            indices_ranges.append(torch.arange(start, end))
        # Include event dimensions in the slices
        slices.extend([slice(None)] * len(self.event_shape))
        meshgrid = torch.meshgrid(*indices_ranges, indexing='ij')
        indices = torch.stack(meshgrid, dim=-1).reshape(-1, len(self.subsample_sizes))
        obs_block = CalipyObservation(self.data[tuple(slices)], self.batch_shape, self.event_shape, subsample_indices = indices)
        return obs_block, indices

# Usage example
subsample_sizes = [4, 2]
subbatch_dataset = SubbatchDataset(data, subsample_sizes, batch_shape=batch_shape, event_shape=[2])
subbatch_dataloader = DataLoader(subbatch_dataset, batch_size=None)


index_list = []
obs_block_list = []
mean_list = []
for obs_block, index_tensor in subbatch_dataloader:
    print("Obs block shape:", obs_block.observations.shape)
    print("Index tensor shape:", index_tensor.shape)
    print("obs block:", obs_block)
    print("Indices:", index_tensor)
    index_list.append(index_tensor)
    obs_block_list.append(obs_block)
    mean_list.append(torch.mean(obs_block.observations,0))
    print("----")
    
    
# iii) Support classes
    

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
    batch_shape = dist.batch_shape
    n_plates = len(plate_sizes)
    n_default = 3
    batch_shape_default = [n_default]*len(batch_shape)
    ssi = subsample_indices

    # cases [1,x,x] vectorizable
    if vectorizable == True:
        # Vectorized sampling using pyro.plate
        with contextlib.ExitStack() as stack:
            # Determine dimensions for plates
            batch_dims_from_event = [-(n_plates + len(event_shape)) + i + 1 for i in range(n_plates)]
            batch_dims_from_right = [-(n_plates ) + i + 1 for i in range(n_plates)]
            current_obs = obs.observations if obs is not None else None
            
            # case [0,0] (obs, ssi)
            if obs == None and ssi == None:
                pass
            
            # case [0,1] (obs, ssi)
            if obs == None and ssi is not None:
                pass
            
            
            # case [1,0] (obs, ssi)
            if obs is not None and ssi == None:
                pass
            
            # case [1,1] (obs, ssi)
            if obs is not None and ssi is not None:
                pass
            
            # Handle multiple plates
            for i, (plate_name, plate_size, dim) in enumerate(zip(plate_names, plate_sizes, batch_dims_from_event)):
                subsample = subsample_indices if subsample_indices is not None else None
                size = plate_size
                stack.enter_context(pyro.plate(plate_name, size=size, subsample=subsample, dim=dim))

                # # Not necessary anymore since obs are assumed to be subsampled already
                # # Index observations if subsampling
                # if current_obs is not None and subsample is not None:
                #     current_obs = torch.index_select(current_obs, dim=batch_dims_from_right[i], index=subsample.flatten())

            # Sample data
            data = pyro.sample(name, dist, obs=current_obs)
            batch_shape = data.shape[:n_plates]
            return CalipySample(data, batch_shape, event_shape, vectorizable=True)
        
    # cases [0,x,x] nonvectorizable
    elif vectorizable == False:
            
            # case [0,0] (obs, ssi)
            if obs == None and ssi == None:
                # Create new observations of shape batch_shape_default with ssi
                # a flattened list of product(range(n_default))
                batch_shape_obs = batch_shape_default
                ssi_lists = [list(range(bs)) for bs in batch_shape_obs]
                ssi_list = [torch.tensor(idx) for idx in itertools.product(*ssi_lists)]
                ssi_codes = [[ssi[k].item() for k in range(len(ssi.shape)+1)] for ssi in ssi_list]
                ssi_tensor = torch.vstack(ssi_list)
                obs_name_list = ["sample_{}".format(ssi_code) for ssi_code in ssi_codes] 
                obs_value_list = len(ssi_list)*[None]
            
            # case [0,1] (obs, ssi)
            if obs == None and ssi is not None:
                # Create len(ssi) new observations with given ssi's
                ssi_list = ssi
                ssi_codes = [[ssi[k].item() for k in range(len(ssi.shape)+1)] for ssi in ssi_list]
                ssi_tensor = torch.vstack(ssi_list)
                obs_name_list = ["sample_{}".format(ssi_code) for ssi_code in ssi_codes] 
                obs_value_list = len(ssi_list)*[None]
            
            # case [1,0] (obs, ssi)
            if obs is not None and ssi == None:
                #  Create obs with standard ssi derived from obs batch_shape
                batch_shape_obs = obs.shape[:len(batch_shape_default)]
                ssi_lists = [list(range(bs)) for bs in batch_shape_obs]
                ssi_list = [torch.tensor(idx) for idx in itertools.product(*ssi_lists)]
                ssi_codes = [[ssi[k].item() for k in range(len(ssi.shape)+1)] for ssi in ssi_list]
                ssi_tensor = torch.vstack(ssi_list)
                obs_name_list = ["sample_{}".format(ssi_code) for ssi_code in ssi_codes] 
                obs_value_list = [obs[ssi,...] for ssi in ssi_list]
            
            # case [1,1] (obs, ssi)
            if obs is not None and ssi is not None:
                # Create obs associated to given ssi's
                batch_shape_obs = obs.shape[:len(batch_shape_default)]
                ssi_list = ssi
                ssi_codes = [[ssi[k].item() for k in range(len(ssi.shape)+1)] for ssi in ssi_list]
                ssi_tensor = torch.vstack(ssi_list)
                obs_name_list = ["sample_{}".format(ssi_code) for ssi_code in ssi_codes] 
                obs_value_list = [obs[ssi,...] for ssi in ssi_list]
            
            # Iterate over the lists and sample
            n_samples = len(obs_name_list)
            samples = []
            plate_list = [pyro.plate(plate_names[plate], plate_sizes[plate], subsample = ssi_tensor[:,plate]) for plate in range(len(plate_names))]
            
            
    
            # Iterate over all combinations of indices
            samples = []
            for idx in itertools.product(*batch_index_lists):
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
            batch_shapes = [len(r) for r in batch_index_lists]
            data = torch.stack(sample_values).reshape(*batch_shapes, *event_shape)
            batch_shape = data.shape[:n_plates]
            return CalipySample(data, batch_shape, event_shape, vectorizable=False)
    
    
# iv) Test the classes

obs_object = CalipyObservation(data, plate_names, batch_shape=[n_data_1], event_shape=[2])

# Subsample indices (if any)
subsample_indices = index_list[0]

sample_results = []
# Sample using calipy_sample and check results
for vectorizable in [True, False]:
    for obs in [None, obs_block_list[0]]:
        for ssi in [None, subsample_indices]:
            
            sample_dist = pyro.distributions.MultivariateNormal(loc = mu_true, covariance_matrix = sigma_true)
            sample_result = calipy_sample('my_sample', sample_dist, plate_names, plate_sizes, vectorizable=vectorizable, obs=obs, subsample_indices=ssi)
            
            vflag = 1 if vectorizable == True else 0
            oflag = 1 if obs is not None else 0
            sflag = 1 if ssi is not None else 0
            
            print('vect_{}_obs_{}_ssi_{}_batch_shape'.format(vflag, oflag, sflag), sample_result.batch_shape )
            print('vect_{}_obs_{}_ssi_{}_event_shape'.format(vflag, oflag, sflag), sample_result.event_shape )
            print('vect_{}_obs_{}_ssi_{}_data_shape'.format(vflag, oflag, sflag), sample_result.data.shape )
            # print('vect_{}_obs_{}_ssi_{}_data'.format(vflag, oflag, sflag), sample_result.data )
            sample_results.append(sample_result)
            
            

def model(observations = None, vectorizable = True, subsample_indices = None):
    # observations is CalipyObservation object
    mu = pyro.param(name = 'mu', init_tensor = torch.tensor([5.0,5.0])) 
    sigma = pyro.param( name = 'sigma', init_tensor = 5*torch.eye(2), constraint = dist.constraints.positive_definite)
    
    plate_names = ['batch_dim_1']
    plate_sizes = [n_data_1]
    obs_dist = dist.MultivariateNormal(loc = mu, covariance_matrix = sigma)
    obs = calipy_sample('obs', obs_dist, plate_names, plate_sizes, vectorizable=vectorizable, obs = observations, subsample_indices = subsample_indices)
    
    return obs


# Example usage:
# Run the model with vectorizable=True
output_vectorized = model(vectorizable=True)
print("Vectorized Output Shape:", output_vectorized.data.shape)
print("Vectorized Output:", output_vectorized)

# Run the model with vectorizable=False
output_non_vectorized = model(vectorizable=False)
print("Non-Vectorized Output Shape:", output_non_vectorized.data.shape)
print("Non-Vectorized Output:", output_non_vectorized)

# Plot the shapes
calipy_observation = CalipyObservation(data, plate_names, batch_shape = batch_shape, event_shape = [2])
model_trace = pyro.poutine.trace(model)
spec_trace = model_trace.get_trace(calipy_observation, vectorizable = True, subsample_indices = None)
spec_trace.nodes
print(spec_trace.format_shapes())
            
            
# v) Perform inference in different configurations

# Set up guide
def guide(observations = None, vectorizable= True, subsample_indices = None):
    pass


adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# Set up svi
signature = [1,1,1] # vect, obs, ssi = None or values
# signature = [1,1,0]
# signature = [1,0,1]
# signature = [1,0,0]
# signature = [0,1,1] 
# signature = [0,1,0]
# signature = [0,0,1]
# signature = [0,0,0]


# Handle DataLoader case
loss_sequence = []
for epoch in range(1000):
    epoch_loss = 0
    for batch_data, subsample_indices in subbatch_dataloader:
        
        # Set up kwargs for svi step
        obs_svi_dict = {}
        obs_svi_dict['vectorizable'] = True if signature[0] ==1 else False
        obs_svi_dict['observations'] = batch_data if signature[1] == 1 else None
        obs_svi_dict['subsample_indices'] = subsample_indices if signature[2] == 1 else None
        
        # loss = svi.step(batch_data, True, subsample_indices)
        loss = svi.step(**obs_svi_dict)
        # loss = svi.step(observations = obs_svi_dict['observations'],
        #                 vectorizable = obs_svi_dict['observations'],
        #                 subsample_indices = obs_svi_dict['subsample_indices'])
        # loss = svi.step(obs_svi_dict['observations'],
        #                 obs_svi_dict['observations'],
        #                 obs_svi_dict['subsample_indices'])
        epoch_loss += loss
    
    epoch_loss /= len(subbatch_dataloader)
    loss_sequence.append(epoch_loss)

    if epoch % 100 == 0:
        print(f'epoch: {epoch} ; loss : {epoch_loss}')


"""
    5. Plots and illustrations
"""


# i) Print results

print('True mu = {}, True sigma = {} \n Inferred mu = {}, Inferred sigma = {} \n mean = {} \n mean of batch means = {}'
      .format(mu_true, sigma_true, 
              pyro.get_param_store()['mu'],
              pyro.get_param_store()['sigma'],
              torch.mean(data,0),
              torch.mean(torch.vstack(mean_list),0)))
































   