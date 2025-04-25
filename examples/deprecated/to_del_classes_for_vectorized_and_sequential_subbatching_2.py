# This provides functionality for writing a model with a vectorizable flag where
# the model code actually does not branch depending on the flag. We investigate 
# here, how this can be used to perform subbatching without a model-rewrite
# We make our life a bit easier by only considering one batch dim for now.
# We go for multiple batch dims in the subsequent script.

import pyro
import pyro.distributions as dist
import torch
import contextlib
import itertools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functorch.dim import dims

import numpy as np
import matplotlib.pyplot as plt

# i) Generate synthetic data

n_data_1 = 21
n_data_2 = 1
n_data_total = n_data_1 + n_data_2
torch.manual_seed(42)
pyro.set_rng_seed(42)

mu_true = torch.zeros([1])
sigma_true = torch.ones([1])

extension_tensor = torch.ones([n_data_1])
# extension_tensor = torch.ones([n_data_1, n_data_2])
data_dist = pyro.distributions.Normal(loc = mu_true * extension_tensor, scale = sigma_true)
data = data_dist.sample()

# ii) Build dataloader

class SubbatchDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        # self.input_data = None
        self.output_data = data
    
        
    def __getitem__(self, idx):
        return (self.output_data[idx, ...], idx)
    
    def __len__(self):
        return data.shape[0]

subsample_size = 20
subbatch_dataset = SubbatchDataset(data)
subbatch_dataloader = torch.utils.data.DataLoader(subbatch_dataset, batch_size = subsample_size, shuffle = True)


# Iterate through the DataLoader
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

class SampleResult:
    def __init__(self, data, plate_sizes, vectorizable):
        self.data = data
        self.plate_sizes = plate_sizes
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
            return torch.stack(self.data).reshape(self.plate_sizes)

def sample(name, dist, plate_names, plate_sizes, vectorizable=True, observations = None):
    if vectorizable:
        n_plates = len(plate_sizes)
        # Vectorized sampling
        dims = [ (-n_plates + i) for i in range(n_plates)]
        with contextlib.ExitStack() as stack:
            for plate_name, plate_size, dim in zip(plate_names, plate_sizes, dims):
                obs_subsample_size = observations.shape[0] if observations is not None else subsample_size
                stack.enter_context(pyro.plate(plate_name, plate_size, subsample_size = obs_subsample_size, dim=dim))
            data = pyro.sample(name, dist, obs = observations)
        return SampleResult(data, plate_sizes, vectorizable)
    else:
        # Non-vectorized sampling
        ranges_prior = [range(size) for size in plate_sizes]
        ranges_obs = [range(size) for size in (observations.shape if observations is not None else [])]
        ranges = ranges_obs if observations is not None else ranges_prior
        samples = []
        for idx in itertools.product(*ranges):
            idx_str = '_'.join(map(str, idx))
            obs_or_None = observations[idx] if observations is not None else None
            x = pyro.sample(f"{name}_{idx_str}", dist, obs = obs_or_None)
            samples.append(x)
        return SampleResult(samples, plate_sizes, vectorizable)


def model(observations = None, vectorizable = True):
    # observations is data_list
    mu = pyro.param(name = 'mu', init_tensor = torch.tensor([5.0])) 
    sigma = pyro.param( name = 'sigma', init_tensor = torch.tensor([5.0]), constraint = dist.constraints.positive)
    
    plate_names = ['batch_dim_1']
    plate_sizes = [n_data_1]
    obs = sample('obs', dist.Normal(loc = mu, scale = sigma), plate_names, plate_sizes, vectorizable=vectorizable, observations = observations)
    # obs_or_None = observations[indices] if observations is not None else None
    # obs = pyro.sample('observation_{}'.format(indexsymbol), obs_dist, obs = obs_or_None)
    
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
spec_model = lambda observations :  model(observations = observations, vectorizable = vectorizable)
model_trace = pyro.poutine.trace(spec_model)
spec_trace = model_trace.get_trace(data)
spec_trace.nodes
print(spec_trace.format_shapes())



"""
    4. Perform inference
"""


# i) Set up guide

def guide(observations = None, vectorizable= True):
    pass


adam = pyro.optim.Adam({"lr": 0.1})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(spec_model, guide, adam, elbo)


# iii) Perform svi

# # This is pretty nice: works if model is called vectorizable or sequential
# data_svi = data
# for step in range(1000):
#     loss = svi.step(data_svi)
#     if step % 100 == 0:
#         print('Loss = ' , loss)
# Handle DataLoader case
loss_sequence = []
for epoch in range(1000):
    epoch_loss = 0
    for batch_data, index in subbatch_dataloader:
        loss = svi.step(batch_data)
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