# This provides functionality for writing a model with a vectorizable flag where
# the model code actually does not branch depending on the flag. We investigate 
# here, how this can be used to perform subbatching without a model-rewrite
# To make out life a bit easier, we go for a single output and ignore the torchdims
# module and also discrete distributions. This is remedied in the subsequent script.

# Multiple batch dims make our life a bit hard and we need some extra structure:
    # SubbatchDataset for delivering subbatches
    # Custom MultiBatchSampler that allows multidimensional batches 
    #   -> multidim random partitions of unity creates observation blocks
    # Sample statement in vectorized form: Go observation block by observation block
    # Sample statement in sequential form: Go observation by observation, would also work with flattened indices

# I DID NOT MANAGE TO DO THIS AND IT SEEMS TO COMPLICATED AND BARELY RELEVANT RIGHT NOW
# The problem is actually the weighting of the gradients, this is not done properly
# as pyro doesnt seem to handle subsubbatching well and the scaling of the grads
# in those nested contexts is wrong.

import pyro
import pyro.distributions as dist
import torch
import contextlib
import itertools
from itertools import product
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from functorch.dim import dims

import numpy as np
import random
import matplotlib.pyplot as plt

# i) Generate synthetic data

n_data_1 = 5
n_data_2 = 3
n_data_total = n_data_1 * n_data_2
batch_shape = [n_data_1, n_data_2]
event_shape = [1]
torch.manual_seed(1)
pyro.set_rng_seed(1)

mu_true = torch.zeros([1])
sigma_true = torch.ones([1])

extension_tensor = torch.ones([n_data_1, n_data_2])
data_dist = pyro.distributions.Normal(loc = mu_true * extension_tensor, scale = sigma_true)
data = data_dist.sample().unsqueeze(-1)

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
subsample_size = [2, 2]
dataset = SubbatchDataset(data, subsample_size, batch_shape=batch_shape, event_shape=[1])
subbatch_dataloader = DataLoader(dataset, batch_size=None)

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
    mean_list.append(torch.mean(data_block))
    print("----")




# The actual sampling + modelling functions

class SampleResult:
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
        
        
def sample(name, dist, plate_names, plate_sizes, vectorizable=True, observations=None, indices=None):
    n_event_dims = len(event_shape)
    if vectorizable:
        n_plates = len(plate_sizes)
        # Vectorized sampling
        dims = [(-n_plates + i) for i in range(n_plates)]
        plate_nrs = [k for k in range(n_plates)]
        with contextlib.ExitStack() as stack:
            current_observations = observations
            for plate_nr, plate_name, plate_size, dim in zip(plate_nrs, plate_names, plate_sizes, dims):
                if current_observations is not None and indices is not None:
                    # Get subsample indices for the current plate dimension
                    obs_subsample_indices = torch.unique(indices[:, plate_nr])
                    stack.enter_context(pyro.plate(
                        plate_name,
                        plate_size,
                        subsample=obs_subsample_indices,
                        dim=dim
                    ))
                else:
                    # Use the full plate if indices are not provided
                    stack.enter_context(pyro.plate(plate_name, plate_size, dim=dim))
            data = pyro.sample(name, dist, obs=current_observations)
        return SampleResult(data, plate_sizes, event_shape, vectorizable)
    else:
        # Non-vectorized sampling remains the same
        ranges_prior = [range(size) for size in plate_sizes]
        ranges_obs = [range(size) for size in (observations.shape[0:-n_event_dims] if observations is not None else [])]
        ranges = ranges_obs if observations is not None else ranges_prior
        samples = []
        for idx in itertools.product(*ranges):
            idx_str = '_'.join(map(str, idx))
            subsample_index = idx_str
            # subsample_index = indices[idx] if indices is not None else idx_str # Naming incorrect
            obs_or_None = observations[idx] if observations is not None else None
            x = pyro.sample(f"{name}_{subsample_index}", dist, obs=obs_or_None)
            samples.append(x)
        return SampleResult(samples, plate_sizes, event_shape, vectorizable)



def model(observations=None, indices=None, vectorizable=True):
    mu = pyro.param(name='mu', init_tensor=torch.tensor([5.0])) 
    sigma = pyro.param(name='sigma', init_tensor=torch.tensor([5.0]), constraint=dist.constraints.positive)
    
    plate_names = ['batch_dim_1', 'batch_dim_2']
    plate_sizes = [n_data_1, n_data_2]
    obs_dist = dist.Normal(loc=mu, scale=sigma).to_event(1)
    obs = sample('obs', obs_dist, plate_names, plate_sizes, vectorizable=vectorizable, observations=observations, indices=indices)
    return obs


# Example usage:
# Run the model with vectorizable=True
output_vectorized = model(vectorizable=True)
print("Vectorized Output Shape:", output_vectorized.as_tensor().shape)
print("Vectorized Output:", output_vectorized)

# Run the model with vectorizable=False
output_non_vectorized = model(vectorizable=False)
print("Non-Vectorized Output Shape:", output_non_vectorized.as_tensor().shape)
print("Non-Vectorized Output:", output_non_vectorized)


# Inference
vectorizable = True
# vectorizable = False
spec_model = lambda observations, indices :  model(observations = observations, indices = indices, vectorizable = vectorizable)
model_trace = pyro.poutine.trace(spec_model)
spec_trace = model_trace.get_trace(data, None)
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

# # This is pretty nice: works if model is called vectorizable or sequential
loss_sequence = []
for epoch in range(1000):
    epoch_loss = 0
    for batch_data, index_tensor in subbatch_dataloader:
        # The sequeezing here urts, makes indices and tensors have different shapes 5
        # batch_data = batch_data.squeeze()
        # index_tensor = index_tensor.squeeze()
        loss = svi.step(batch_data, index_tensor)
        epoch_loss += loss
    
    epoch_loss /= len(subbatch_dataloader)
    loss_sequence.append(epoch_loss)

    if epoch % 100 == 0:
        print(f'epoch: {epoch} ; loss : {epoch_loss}')



"""
    5. Plots and illustrations
"""


# i) Print results

print('True mu = {}, True sigma = {} \n Inferred mu = {:.3f}, Inferred sigma = {:.3f} \n mean = {:.3f}, \n mean of batch means = {}'
      .format(mu_true, sigma_true, 
              pyro.get_param_store()['mu'].item(),
              pyro.get_param_store()['sigma'].item(),
              torch.mean(data),
              torch.mean(torch.hstack(mean_list))))
# The issue can be seen here: The result should be the mean of the whole data but 
# instead it is the mean of the batch means.

# ii) Plot data

fig1 = plt.figure(num = 1, dpi = 300)
plt.hist(data.flatten().detach().numpy())
plt.title('Histogram of data')



# OLD STUFF


# Indexing
def indexfun(tuple_of_indices, vectorizable = True):
    """ Function to create a multiindex that can handle an arbitrary number of indices 
    (both integers and vectors), while preserving the shape of the original tensor.
    # Example usage 1 
    A = torch.randn([4,5])  # Tensor of shape [4, 5]
    i = torch.tensor([1])
    j = torch.tensor([3, 4])
    # Call indexfun to generate indices
    indices_A, symbol = indexfun((i, j))
    # Index the tensor
    result_A = A[indices_A]     # has shape [1,2]
    # Example usage 2
    B = torch.randn([4,5,6,7])  # Tensor of shape [4, 5, 6, 7]
    k = torch.tensor([1])
    l = torch.tensor([3, 4])
    m = torch.tensor([3])
    n = torch.tensor([0,1,2,3])
    # Call indexfun to generate indices
    indices_B, symbol = indexfun((k,l,m,n))
    # Index the tensor
    result_B = B[indices_B]     # has shape [1,2,1,4]
    """
    # Calculate the shape needed for each index to broadcast correctly
    if vectorizable == True:        
        idx = tuple_of_indices
        shape = [1] * len(idx)  # Start with all singleton dimensions
        broadcasted_indices = []
    
        for i, x in enumerate(idx):
            target_shape = list(shape)
            target_shape[i] = len(x)
            # Reshape the index to target shape
            x_reshaped = x.view(target_shape)
            # Expand to match the full broadcast size
            x_broadcast = x_reshaped.expand(*[len(idx[j]) if j != i else x_reshaped.shape[i] for j in range(len(idx))])
            broadcasted_indices.append(x_broadcast)
            indexsymbol = 'vectorized'
    else:
        broadcasted_indices = tuple_of_indices
        indexsymbol = broadcasted_indices
    
    # Convert list of broadcasted indices into a tuple for direct indexing
    return tuple(broadcasted_indices), indexsymbol       


