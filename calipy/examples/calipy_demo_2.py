#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to model two measurement processes 
with the same variance but different means. The two measurement procedures have 
been used to collect two different datasets, both featuring a different amount
of samples. Inference is performed to estimate the value of the two underlying 
expected values and the single standard deviation.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Load and customize effects
    4. Build the probmodel
    5. Perform inference
    6. Analyse results and illustrate

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    1. Imports and definitions
"""


# i) Imports

# base packages
import torch
import pyro
import matplotlib.pyplot as plt

# calipy
import calipy
from calipy.core.base import NodeStructure, CalipyProbModel
from calipy.core.effects import UnknownParameter, NoiseAddition, UnknownVariance
from calipy.core.utils import dim_assignment
from calipy.core.data import DataTuple, CalipyDict
from calipy.core.tensor import CalipyTensor


# ii) Definitions

n_meas_1 = 20
n_meas_2 = 50



"""
    2. Simulate some data
"""


# i) Set up sample distributions

mu_true = torch.tensor([1.0, 1.5])
sigma_true = torch.tensor(0.1)


# ii) Sample from distributions

data_1_distribution = pyro.distributions.Normal(mu_true[0], sigma_true)
data_2_distribution = pyro.distributions.Normal(mu_true[1], sigma_true)
data_1 = data_1_distribution.sample([n_meas_1])
data_2 = data_2_distribution.sample([n_meas_2])
data = (data_1, data_2)

# The data now is a 2-tuple with shapes [n_meas_1] and [n_meas_2] and reflects
# measurements being taken of two different objects but with the same device.

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.



"""
    3. Load and customize effects
"""


# i) Set up dimensions
# The nodestructures require dimensions that are trivial for the parameter dim
# of mu_1, mu_2, and siigma but vary for the batch dimensions of mu_1 and mu_2 
# and are even undefined for sigma which takes on the role as standard deviation
# for both sets of observations.

param_dims_mu_1 = dim_assignment(['pd_mu1'], dim_sizes = [])
batch_dims_meas_1 = dim_assignment(['bd_meas1'], dim_sizes = [n_meas_1])
event_dims_meas_1 = dim_assignment(['ed_meas1'], dim_sizes = [])

param_dims_mu_2 = dim_assignment(['pd_mu2'], dim_sizes = [])
batch_dims_meas_2 = dim_assignment(['bd_meas2'], dim_sizes = [n_meas_2])
event_dims_meas_2 = dim_assignment(['ed_meas2'], dim_sizes = [])

param_dims_sigma = dim_assignment(['pd_sigma'], dim_sizes = [])
batch_dims_sigma = dim_assignment(['bd_sigma'], dim_sizes = [])


# ii) Set up nodestructures for mean parameters mu_1, mu_2
# Setting up requires correctly specifying a NodeStructure object. You can get 
# a template for the node_structure by calling NodeStructure(UnknownParameter)
# or more generally NodeStructure(Node) where node is the CalipyQuantity or the
# CalipyEffect class you want to designate the NodeStructure for.

# mu_1 setup
mu_1_ns = NodeStructure(UnknownParameter)
mu_1_ns.set_dims(batch_dims = batch_dims_meas_1, param_dims = param_dims_mu_1)
mu_1_object = UnknownParameter(mu_1_ns, name = 'mu_1')

# mu_2 setup
mu_2_ns = NodeStructure(UnknownParameter)
mu_2_ns.set_dims(batch_dims = batch_dims_meas_2, param_dims = param_dims_mu_2)
mu_2_object = UnknownParameter(mu_2_ns, name = 'mu_2')


# iii) Set up nodestructures for sigma
# Make it into a scalar so it automatically broadcasts with mu_1, mu_2
sigma_ns = NodeStructure(UnknownVariance)
sigma_ns.set_dims(batch_dims = batch_dims_sigma, param_dims = param_dims_sigma)
sigma_object = UnknownVariance(sigma_ns, name = 'sigma')


# iv) Set up the nodestructure for noise addition
# This is similar to what happened previously but notive that we have two noise
# objects with different dimensions and therefore different nodestructures.
noise_1_ns = NodeStructure(NoiseAddition)
noise_1_ns.set_dims(batch_dims = batch_dims_meas_1, event_dims = event_dims_meas_1)
noise_1_object = NoiseAddition(noise_1_ns)
        
noise_2_ns = NodeStructure(NoiseAddition)
noise_2_ns.set_dims(batch_dims = batch_dims_meas_2, event_dims = event_dims_meas_2)
noise_2_object = NoiseAddition(noise_2_ns) 



"""
    4. Build the probmodel
"""


# i) Define the probmodel class 

class DemoProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        # Integrate the nodes        
        self.mu_1_object = mu_1_object
        self.mu_2_object = mu_2_object
        self.sigma_object = sigma_object
        self.noise_1_object = noise_1_object 
        self.noise_2_object = noise_2_object
        
    # Define model by forward passing
    def model(self, input_vars = None, observations = CalipyDict({'meas_1' : None,
                                                                  'meas_2' : None})):
        mu_1 = self.mu_1_object.forward()
        mu_2 = self.mu_2_object.forward()
        sigma = self.sigma_object.forward()
        # The above works nicely because sigma broadcasts. however, in other cases
        # I would need to do quite a bit of gymnastics, first take sigma.tensor,
        # then expand it to the right sizes via sigma.tensor * torch.ones, then
        # wrap that result back into a CalipyTensor. Potential remedies: Better
        # Broadcasting, or allow things like multiplication with torch.ones to
        # act on calipyTensors and produce CalipyTensors.
        
        inputs_1 = {'mean':mu_1, 'standard_deviation': sigma}
        inputs_2 = {'mean':mu_2, 'standard_deviation': sigma}
        
        output_1 = self.noise_1_object.forward(inputs_1, observations = observations['meas_1'])
        output_2 = self.noise_2_object.forward(inputs_2, observations = observations['meas_2'])
        output = CalipyDict({'meas_1' : output_1, 'meas_2' : output_2})
        
        return output
    
    # Define the guide (trivial since no posteriors)    
    def guide(self, input_vars = None, observations = (None, None)):
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


# ii) Prepare observation data
# Note that now we need to provide a CalipyDict object as output_data; i.e. the
# observations inputted to the model because we distinnguish meas_1 and meas_2
# and our observations are not anymore any single tensor.
input_data = None
data_1_cp = CalipyTensor(data_1, dims = batch_dims_meas_1)
data_2_cp = CalipyTensor(data_2, dims = batch_dims_meas_2)
output_data = CalipyDict({ 'meas_1' : data_1_cp, 'meas_2' : data_2_cp})
optim_results = demo_probmodel.train(input_data, output_data, optim_opts = optim_opts)
    


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
    
print('True values of mu_1, mu_2, sigma = ', mu_true[0], mu_true[1], sigma_true)
print('Results of taking empirical means for mu_1, mu_2 = ', torch.mean(data[0]), torch.mean(data[1]))






















