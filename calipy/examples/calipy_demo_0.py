#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 21:20:20 2025

@author: jemil
"""
import torch, pyro
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
optim_results = demo_probmodel.train(None, data_cp, optim_opts={
    'optimizer': pyro.optim.NAdam({"lr": 0.01}),
    'loss': pyro.infer.Trace_ELBO(),
    'n_steps': 1000
})

# Plot results
plt.plot(optim_results)
plt.xlabel('Epoch'); plt.ylabel('ELBO loss'); plt.title('Training Progress')
plt.show()