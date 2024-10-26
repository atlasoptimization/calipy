# This provides functionality for writing a model with a vectorizable flag where
# the model code actually does not branch depending on the flag. We investigate 
# here, how this can be used to perform subbatching without a model-rewrite
# We make our life a bit easier by only considering one batch dim for now.
# We add however here support for multiple event_dims

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

# class SubbatchDataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         # self.input_data = None
#         self.output_data = data
    
        
#     def __getitem__(self, idx):
#         return (self.output_data[idx, ...], idx)
    
#     def __len__(self):
#         return data.shape[0]

# subsample_size = 20
# subbatch_dataset = SubbatchDataset(data)
# subbatch_dataloader = torch.utils.data.DataLoader(subbatch_dataset, batch_size = subsample_size, shuffle = True)


# # Iterate through the DataLoader
# for batch_output, index in subbatch_dataloader:
#     print(batch_output, index)
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


# The actual sampling + modelling functions

class CalipySample:
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

def sample(name, dist, plate_names, plate_sizes, vectorizable=True, observations = None, subsample_indices = None):
    event_shape = list(dist.event_shape)
    n_event_dims = len(event_shape)
    if vectorizable:
        n_plates = len(plate_sizes)
        # Vectorized sampling
        dims = [ (-n_plates -n_event_dims + 1 + i) for i in range(n_plates)]
        plate_nrs = [k for k in range(n_plates)]
        with contextlib.ExitStack() as stack:
            current_observations = observations
            for plate_nr, plate_name, plate_size, dim in zip(plate_nrs, plate_names, plate_sizes, dims):
                if current_observations is not None and subsample_indices is not None:
                    # Get subsample indices for the current plate dimension
                    obs_subsample_indices = torch.unique(subsample_indices[:, plate_nr])
                    stack.enter_context(pyro.plate(
                        plate_name,
                        plate_size,
                        subsample=obs_subsample_indices,
                        dim=dim
                    ))
                else:
                    # Use the full plate if indices are not provided
                    stack.enter_context(pyro.plate(plate_name, plate_size, dim=dim))
            data = pyro.sample(name, dist, obs = current_observations)
        return CalipySample(data, plate_sizes, event_shape, vectorizable)
    else:
        # Non-vectorized sampling
        ranges_prior = [range(size) for size in plate_sizes]
        ranges_subsample = [range(subsample_size) for subsample_size in subsample_sizes]
        ranges_obs = [range(size) for size in (observations.shape[0:-n_event_dims] if observations is not None else [])]
        # ranges_obs = subsample_indices
        ranges = ranges_obs if subsample_indices is not None else ranges_prior
        samples = []
        for idx in itertools.product(*ranges):
            idx_str = '_'.join(map(str, idx))
            subsample_index = subsample_indices[idx] if subsample_indices is not None else idx_str
            idx_str = '{}'.format(subsample_index)
            obs_or_None = observations[idx] if observations is not None else None
            x = pyro.sample(f"{name}_{subsample_index}", dist, obs = obs_or_None)
            samples.append(x)
        return CalipySample(samples, plate_sizes, event_shape, vectorizable)


def model(observations = None, vectorizable = True, subsample_indices = None):
    # observations is data_list
    mu = pyro.param(name = 'mu', init_tensor = torch.tensor([5.0,5.0])) 
    sigma = pyro.param( name = 'sigma', init_tensor = 5*torch.eye(2), constraint = dist.constraints.positive_definite)
    
    plate_names = ['batch_dim_1']
    plate_sizes = [n_data_1]
    obs_dist = dist.MultivariateNormal(loc = mu, covariance_matrix = sigma)
    obs = sample('obs', obs_dist, plate_names, plate_sizes, vectorizable=vectorizable, observations = observations, subsample_indices = subsample_indices)
    
    return obs


# # Example usage:
# # Run the model with vectorizable=True
# output_vectorized = model(vectorizable=True)
# print("Vectorized Output Shape:", output_vectorized.as_tensor().shape)
# print("Vectorized Output:", output_vectorized)

# # Run the model with vectorizable=False
# output_non_vectorized = model(vectorizable=False)
# print("Non-Vectorized Output Shape:", output_non_vectorized.as_tensor().shape)
# print("Non-Vectorized Output:", output_non_vectorized)


# Inference
vectorizable = True
# vectorizable = False
spec_model = lambda observations, subsample_indices :  model(observations = observations, vectorizable = vectorizable, subsample_indices = subsample_indices)
model_trace = pyro.poutine.trace(spec_model)
spec_trace = model_trace.get_trace(data, subsample_indices = None)
spec_trace.nodes
print(spec_trace.format_shapes())



"""
    4. Perform inference
"""


# i) Set up guide

def guide(observations = None, vectorizable= True):
    pass


adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(spec_model, guide, adam, elbo)


# iii) Perform svi

# Handle DataLoader case
loss_sequence = []
for epoch in range(1000):
    epoch_loss = 0
    for batch_data, subsample_indices in subbatch_dataloader:
        loss = svi.step(batch_data, subsample_indices)
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


# ii) Plot data

fig1 = plt.figure(num = 1, dpi = 300)
plt.hist(data.flatten().detach().numpy())
plt.title('Histogram of data')