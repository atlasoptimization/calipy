#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to model a simple measurement process 
with a single unknown mean and known variance. The measurement procedure has
been used to collect a single datasets, that features n_meas samples. Inference
is performed to estimate the value of the underlying expected value.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Load and customize effects
    4. Build the probmodel
    5. Perform inference
    6. Analyse results and illustrate

The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    1. Imports and definitions
"""


# i) Imports

# base packages
import torch
import pyro

# calipy
from calipy.core.base import NodeStructure, CalipyProbModel
from calipy.core.effects import UnknownParameter, NoiseAddition
from calipy.core.utils import dim_assignment
from calipy.core.tensor import CalipyTensor


# ii) Definitions

n_meas = 20


"""
    2. Simulate some data
"""


# i) Set up sample distributions

mu_true = torch.tensor(0.0)
sigma_true = torch.tensor(0.1)


# ii) Sample from distributions

data_distribution = pyro.distributions.Normal(mu_true, sigma_true)
data = data_distribution.sample([n_meas])


"""
   1. Load and customize effects
"""


# i) Set up dimensions
batch_dims = dim_assignment(['bd_1'], dim_sizes = [n_meas])
event_dims = dim_assignment(['ed_1'], dim_sizes = [])
param_dims = dim_assignment(['pd_1'], dim_sizes = [])

# ii) Set up mu and noise
mu_ns = NodeStructure(UnknownParameter)
mu_ns.set_dims(batch_dims = batch_dims, param_dims = param_dims,)
mu_object = UnknownParameter(mu_ns, name = 'mu')

noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims = batch_dims, event_dims = event_dims)
noise_object = NoiseAddition(noise_ns, name = 'noise')
        

"""
    2. Build the probmodel
"""

# i) Define the probmodel class 
class DemoProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # integrate nodes
        self.mu_object = mu_object
        self.noise_object = noise_object 
        
    # Define model by forward passing
    def model(self, input_vars = None, observations = None):
        mu = self.mu_object.forward()       

        inputs = {'mean':mu, 'standard_deviation': sigma_true} 
        output = self.noise_object.forward(input_vars = inputs,
                                           observations = observations)
        return output
    
    # Define guide (trivial since no posteriors)
    def guide(self, input_vars = None, observations = None):
        pass
    
demo_probmodel = DemoProbModel()
    


"""
    3. Perform inference
"""
    
# i) Set up optimization
adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
n_steps = 1000
optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps}


# ii) Train the model
input_data = None
optim_results = demo_probmodel.train(input_data, data_cp, optim_opts)







data_cp = CalipyTensor(data, dims = batch_dims + event_dims)

















