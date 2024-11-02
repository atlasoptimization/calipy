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
from calipy.core.utils import dim_assignment, generate_trivial_dims, context_plate_stack, DimTuple

import numpy as np
import random
import matplotlib.pyplot as plt

torch.manual_seed(42)
pyro.set_rng_seed(42)


# i) Generate synthetic data A and B

# Generate data_A corresponding e.g. to two temperature vals on a grid on a plate
n_data_1_A = 6
n_data_2_A = 4
batch_shape_A = [n_data_1_A, n_data_2_A]
n_event_A = 2
n_event_dims_A = 1
n_data_total_A = n_data_1_A * n_data_2_A


# # Plate names and sizes
plate_names_A = ['batch_plate_1_A', 'batch_plate_2_A']
plate_sizes_A = [n_data_1_A, n_data_2_A]

mu_true_A = torch.zeros([1,1,n_event_A])
sigma_true_A = torch.tensor([[1,0],[0,0.01]])

extension_tensor_A = torch.ones([n_data_1_A,n_data_2_A,1])
data_dist_A = pyro.distributions.MultivariateNormal(loc = mu_true_A * extension_tensor_A, covariance_matrix = sigma_true_A)
data_A = data_dist_A.sample()

# Generate data_B corresponding e.g. to three temperature vals on each location on a rod
n_data_1_B = 5
batch_shape_B = [n_data_1_B]
n_event_B = 3
n_event_dims_B = 1
n_data_total_B = n_data_1_B


# # Plate names and sizes
plate_names_B = ['batch_plate_1_B']
plate_sizes_B = [n_data_1_B]

mu_true_B = torch.zeros([1,n_event_B])
sigma_true_B = torch.tensor([[1,0,0],[0,0.1,0],[0,0,0.01]])

extension_tensor_B = torch.ones([n_data_1_B,1])
data_dist_B = pyro.distributions.MultivariateNormal(loc = mu_true_B * extension_tensor_B, covariance_matrix = sigma_true_B)
data_B = data_dist_B.sample()


# functorch dims
batch_dims_A = dim_assignment(dim_names = ['bd_1_A', 'bd_2_A'])
event_dims_A = dim_assignment(dim_names = ['ed_1_A'])
data_dims_A = batch_dims_A + event_dims_A

batch_dims_B = dim_assignment(dim_names = ['bd_1_B'])
event_dims_B = dim_assignment(dim_names = ['ed_1_B'])
data_dims_B = batch_dims_B + event_dims_B


# i) DataTuple Class
class DataTuple:
    """
    Custom class for holding tuples of various objects with explicit names.
    Provides easy access to each tensor and related metadata.
    """
    def __init__(self, names, values):
        if len(names) != len(values):
            raise ValueError("Length of names must match length of values.")
        self._data_dict = {name: value for name, value in zip(names, values)}

    def __getitem__(self, key):
        return self._data_dict[key]

    def keys(self):
        return self._data_dict.keys()

    def values(self):
        return self._data_dict.values()

    def items(self):
        return self._data_dict.items()
    
    def apply_elementwise(self, function):
        """ 
        Returns a new DataTuple with keys = self._data_dict.keys() and associated
        values = function(self._data_dict.values())
        """
        new_dict = {}
        key_list = []
        value_list = []
        
        for key, value in self._data_dict.items():
            new_key = key
            new_value =  function(value)
            
            new_dict[new_key] = new_value
            key_list.append(new_key)
            value_list.append(new_value)
            
        return DataTuple(key_list, value_list)
    
    def bind_dims(self, datatuple_dims):
        """ 
        Returns a new DataTuple of tensors with dimensions bound to the dims
        recorded in the DataTuple datatuple_dims.
        """
        
        for key, value in self._data_dict.items():
            if isinstance(value, torch.Tensor) or value is None:
                pass
            else:
                raise Exception('bind dims only available for tensors or None '\
                                'but tuple element is {}'.format(value.__class__))
        new_dict = {}
        key_list = []
        value_list = []
        
        for key, value in self._data_dict.items():
            new_key = key
            new_value =  value[datatuple_dims[key]]
            
            new_dict[new_key] = new_value
            key_list.append(new_key)
            value_list.append(new_value)
            
        return DataTuple(key_list, value_list)
                    

    def get_local_copy(self):
        """
        Returns a new DataTuple with local copies of all DimTuple instances.
        Non-DimTuple items remain unchanged.
        """
        local_copy_data = {}
        key_list = []
        value_list = []
        
        for key, value in self._data_dict.items():
            if isinstance(value, DimTuple):
                local_copy_data[key] = value.get_local_copy()
            else:
                local_copy_data[key] = value
            key_list.append(key)
            value_list.append(value)
            
        return DataTuple(key_list, value_list)

    def __add__(self, other):
        """ 
        Overloads the + operator to return a new DataTuple when adding two DataTuple objects.
        
        :param other: The DataTuple to add.
        :type other: DataTuple
        :return: A new DataTuple with elements from each tuple added elementwise.
        :rtype: DataTuple
        :raises ValueError: If both DataTuples do not have matching keys.
        """
        if not isinstance(other, DataTuple):
            return NotImplemented

        if self.keys() != other.keys():
            raise ValueError("Both DataTuples must have the same keys for elementwise addition.")

        combined_data = {}
        for key in self.keys():
            combined_data[key] = self[key] + other[key]

        return DataTuple(list(combined_data.keys()), list(combined_data.values()))

    def __repr__(self):
        repr_items = []
        for k, v in self._data_dict.items():
            if isinstance(v, torch.Tensor):
                repr_items.append(f"{k}: shape={v.shape}")
            else:
                repr_items.append(f"{k}: {v.__repr__()}")
        return f"DataTuple({', '.join(repr_items)})"
    
   

# ii) Build dataloader
class CalipyObservation:
    """
    Stores observations along with batch and event dimensions.
    Each observation has a unique name for reference during sampling.
    """
    def __init__(self, observations, batch_dims, event_dims, subsample_indices=None, vectorizable=True):

        # Metadata: keep it directly tied with observations
        self.entry_names = [key for key in observations.keys()]
        self.batch_dims = batch_dims.get_local_copy()
        self.event_dims = event_dims.get_local_copy()
        self.obs_dims = self.batch_dims + self.event_dims
        self.index_dims = DataTuple(self.entry_names, [dim_assignment(['id_{}'.format(key)],
                                    [len(self.obs_dims[key])]) for key in self.entry_names])
        self.ssi_dims = self.obs_dims + self.index_dims
        self.vectorizable = vectorizable
        
        # Handle tensor tuples for obs and ssi
        self.observations = observations
        self.observations_bound = observations.bind_dims(self.obs_dims)
        self.subsample_indices = subsample_indices
        self.subsample_indices_bound = subsample_indices.bind_dims(self.ssi_dims) if subsample_indices is not None else None

        # Initialize local and global indices for easy reference
        self.obs_local_indices = self._initialize_local_indices()
        self.obs_global_indices = self._initialize_global_indices()
        self.index_to_name_dict = self._generate_index_to_name_dict()

    def _initialize_local_indices(self):
        # Create local indices based on the current tensor shape
        key_list = []
        value_list = []
        for key, tensor in self.observations.items():
            key_list.append(key)
            index_tensor = torch.zeros(self.ssi_dims[key].sizes)
            index_list = list(itertools.product(*[range(size) for size in tensor.shape]))
            for index in index_list:
                index_tensor[index + (...,)] = torch.tensor(index)
            value_list.append(index_tensor)
        indices = DataTuple(key_list, value_list)
        return indices

    def _initialize_global_indices(self):
        # Initialize global indices based on subsampling, or default to local indices if none provided
        if self.subsample_indices is None:
            return self._initialize_local_indices()
        else:
            return self.subsample_indices  # This needs to be a meaningful subsampling view

    def _generate_index_to_name_dict(self):
        # Map indices to unique names
        index_to_name_dict = {}
        for key, indices in self.obs_global_indices.items():
            for idx in indices:
                idx_str = f"{key}_sample_{'_'.join(str(i) for i in idx)}"
                index_to_name_dict[(key, idx)] = idx_str
        return index_to_name_dict

    def get_entry(self, **batch_dims_spec):
        # Retrieve an observation by specifying batch dimensions explicitly
        indices = [batch_dims_spec[dim] for dim in self.batch_dims]
        obs_values = {key: tensor[tuple(indices)] if len(tensor) > 0 else None for key, tensor in self.observations.items()}
        obs_name = {key: self.index_to_name_dict[(key, tuple(indices))] for key in obs_values.keys()}
        return obs_name, obs_values

    def get_local_index(self, key, idx):
        return self.obs_local_indices[key][idx]

    def get_global_index(self, key, idx):
        return self.obs_global_indices[key][idx]

    def __repr__(self):
        repr_str = 'CalipyObservation object with observations: {}'\
            .format(', '.join(f"{k}: {v.shape}" for k, v in self.observations.items()))
        return repr_str

# Instantiate CalipyObservation
obs_name_list = ['T_grid', 'T_rod']
observations = DataTuple(obs_name_list, [data_A, data_B])
batch_dims = DataTuple(obs_name_list, [batch_dims_A, batch_dims_B])
event_dims = DataTuple(obs_name_list, [event_dims_A, event_dims_B])
obs_dims = batch_dims + event_dims

calipy_obs = CalipyObservation(observations, batch_dims, event_dims)
print(calipy_obs)



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
































   