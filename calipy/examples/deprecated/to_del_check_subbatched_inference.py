#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to perform subbatched inference on
a model a simple measurement process with unknown mean and known variance. The
measurement procedure has been used to collect a single dataset, that features
n_meas samples. Inference is performed to estimate the value of the underlying
expected value.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data, enable subbatching
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
import math
import matplotlib.pyplot as plt

# calipy
import calipy
from calipy.core.base import NodeStructure, CalipyProbModel
from calipy.core.effects import UnknownParameter, NoiseAddition
from calipy.core.utils import dim_assignment
from calipy.core.data import DataTuple, CalipyDict, CalipyIO, CalipyDataset, io_collate
from calipy.core.tensor import CalipyTensor, CalipyIndex
from calipy.core.funs import calipy_cat


# Todel
from torch.utils.data import Dataset, DataLoader
from calipy.core.utils import CalipyDim


# ii) Definitions

n_meas = 10
n_event = 1
n_subbatch = 9



"""
    2. Simulate some data, enable subbatching
"""


# i) Set up sample distributions

mu_true = torch.tensor(0.0)
sigma_true = torch.tensor(0.1)


# ii) Sample from distributions

data_distribution = pyro.distributions.Normal(mu_true, sigma_true)
data = data_distribution.sample([n_meas, n_event])

# The data now is a tensor of shape [n_meas] and reflects measurements being
# taken of a single object with a single measurement device.

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.


# iii) Enable subbatching
data_dims = dim_assignment(['bd_data', 'ed_data'], dim_sizes = [n_meas, n_event])
data_cp = CalipyTensor(data, data_dims, name = 'data')
dataset = CalipyDataset(input_data = None, output_data = data_cp)

dataloader = DataLoader(dataset, batch_size=n_subbatch, shuffle=True, collate_fn=io_collate)

# Iterate through the DataLoader
for batch_input, batch_output, batch_index in dataloader:
    print(batch_input, batch_output, batch_index)


"""
    3. Load and customize effects
"""


# i) Set up dimensions

batch_dims = dim_assignment(['bd_1'], dim_sizes = [n_meas])
event_dims = dim_assignment(['ed_1'], dim_sizes = [1])
param_dims = dim_assignment(['pd_1'], dim_sizes = [])


# ii) Set up dimensions for mean parameter mu

# Setting up requires correctly specifying a NodeStructure object. You can get 
# a template for the node_structure by calling generate_template() on the example
# node_structure delivered with the class description. Here, we call the example
# node structure, then set the dims; required dims that need to be provided can
# be found via help(mu_ns.set_dims).

# mu setup
mu_ns = NodeStructure(UnknownParameter)
mu_ns.set_dims(batch_dims = batch_dims, param_dims = param_dims,)
mu_object = UnknownParameter(mu_ns, name = 'mu')


# iii) Set up the dimensions for noise addition
# This requires again the batch shapes and event shapes. They are used to set
# up the dimensions in which the noise is i.i.d. and the dims in which it is
# copied. Again, required keys can be found via help(noise_ns.set_dims).
noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims = batch_dims, event_dims = event_dims)
noise_object = NoiseAddition(noise_ns, name = 'noise')
        


"""
    4. Build the probmodel
"""


# i) Define the probmodel class 

class DemoProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # integrate nodes
        self.mu_object = mu_object
        self.noise_object = noise_object 
        
    # Define model by forward passing
    def model(self, input_vars = None, observations = None, subsample_index = None):
        mu = self.mu_object.forward(subsample_index = subsample_index)    

        # Dictionary/DataTuple input is converted to CalipyIO internally. It
        # is also possible, to pass single element input_vars or observations;
        # these are also autowrapped.
        inputs = {'mean':mu, 'standard_deviation': sigma_true} 
        output = self.noise_object.forward(input_vars = inputs,
                                           observations = observations,
                                           subsample_index = subsample_index)
        
        return output
    
    # Define guide (trivial since no posteriors)
    def guide(self, input_vars = None, observations = None, subsample_index = None):
        pass
    
demo_probmodel = DemoProbModel()
    



"""
    5. Perform inference
"""
    

# i) Set up optimization

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
n_steps = 1000

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps}


# ii) Train the model
# When the dataloader argument is passed, it replaces the input_data and output_data
# args. Internally, an inference loop is started that cycles through batch_input,
# batch_output_batch_index of the iterable dataloader and passes them to svi.step()
input_data = None
data_cp = CalipyTensor(data, dims = batch_dims + event_dims)
optim_results = demo_probmodel.train(dataloader = dataloader, optim_opts = optim_opts)



"""
    6. Analyse results and illustrate
"""


# i)  Plot loss

plt.figure(1, dpi = 300)
plt.plot(optim_results)
plt.title('ELBO loss')
plt.xlabel('epoch')

# ii) Print  parameters

for param, value in pyro.get_param_store().items():
    print(param, '\n', value)
    
print('True values of mu = ', mu_true)
print('Results of taking empirical means for mu_1 = ', torch.mean(data))






















