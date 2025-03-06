# This provides functionality for writing a model with a vectorizable flag where
# the model code actually does not branch depending on the flag. This allows
# in principle having a single model code that works for sequential and vectorizable
# data and also for subbatching.
# Subbatching is not included here but will be investigated in version_2 of this
# script.

import pyro
import pyro.distributions as dist
import torch
import contextlib
import itertools

import numpy as np
import matplotlib.pyplot as plt

# i) Generate synthetic data

n_data_1 = 5
n_data_2 = 3
n_data_total = n_data_1 + n_data_2
torch.manual_seed(1)
pyro.set_rng_seed(1)

mu_true = torch.zeros([1])
sigma_true = torch.ones([1])

extension_tensor = torch.ones([n_data_1, n_data_2])
data_dist = pyro.distributions.Normal(loc = mu_true * extension_tensor, scale = sigma_true)
data = data_dist.sample()


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
                stack.enter_context(pyro.plate(plate_name, plate_size, dim=dim))
            data = pyro.sample(name, dist, obs = observations)
        return SampleResult(data, plate_sizes, vectorizable)
    else:
        # Non-vectorized sampling
        ranges = [range(size) for size in plate_sizes]
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
    
    plate_names = ['dim1', 'dim2']
    plate_sizes = [5, 3]
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
# vectorizable = True
vectorizable = False
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


adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(spec_model, guide, adam, elbo)


# iii) Perform svi

# This is pretty nice: works if model is called vectorizable or sequential
data_svi = data
for step in range(1000):
    loss = svi.step(data_svi)
    if step % 100 == 0:
        print('Loss = ' , loss)



"""
    5. Plots and illustrations
"""


# i) Print results

print('True mu = {}, True sigma = {} \n Inferred mu = {:.3f}, Inferred sigma = {:.3f} \n mean = {:.3f}'
      .format(mu_true, sigma_true, 
              pyro.get_param_store()['mu'].item(),
              pyro.get_param_store()['sigma'].item(),
              torch.mean(data)))


# ii) Plot data

fig1 = plt.figure(num = 1, dpi = 300)
plt.hist(data.flatten().detach().numpy())
plt.title('Histogram of data')