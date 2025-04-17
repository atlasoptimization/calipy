#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a minimal demo of calipy functionality. An unknown parameter mu is 
observed noisily and has to be estimated from these observations. 

The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""

# Imports and definitions
import pyro
import matplotlib.pyplot as plt

from calipy.core.base import NodeStructure, CalipyProbModel
from calipy.core.effects import UnknownParameter, NoiseAddition
from calipy.core.utils import dim_assignment
from calipy.core.tensor import CalipyTensor

# Simulate data
n_meas = 20
mu_true, sigma_true = 0.0, 0.1
data = pyro.distributions.Normal(mu_true, sigma_true).sample([n_meas])

# Define dimensions
batch_dims = dim_assignment(['batch'], [n_meas])
single_dims = dim_assignment(['single'], [])

# Set up model nodes
mu_ns = NodeStructure(UnknownParameter)
mu_ns.set_dims(batch_dims=batch_dims, param_dims = single_dims)
mu_node = UnknownParameter(mu_ns, name='mu')
noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims=batch_dims, event_dims = single_dims)
noise_node = NoiseAddition(noise_ns, name='noise')

# Define probabilistic model
class DemoProbModel(CalipyProbModel):
    def model(self, input_vars = None, observations=None):
        mu = mu_node.forward()
        return noise_node.forward({'mean': mu, 'standard_deviation': sigma_true}, observations)

    def guide(self, input_vars = None, observations=None):
        pass

# Train model
demo_probmodel = DemoProbModel()
data_cp = CalipyTensor(data, dims=batch_dims)
optim_results = demo_probmodel.train(None, data_cp, optim_opts = {})

# Plot results
plt.plot(optim_results)
plt.xlabel('Epoch'); plt.ylabel('ELBO loss'); plt.title('Training Progress')
plt.show()