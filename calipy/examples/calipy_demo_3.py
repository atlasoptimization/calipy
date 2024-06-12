#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to model a chain of production and
measurement process. Multiple copies of an product are produced; due to production
randomness they all feature different length parameters mu_prod. Subsequently,
these products are measured leading to another source of uncertainty that results
in noisy measurement obs. Inference is performed to estimate the value of the 
underlying mean and standard deviation of the production process.
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
from calipy.core.effects import UnknownParam, UnknownVar, NoiseAddition


# ii) Definitions

n_meas = 100
n_prod = 30



"""
    2. Simulate some data
"""


# i) Set up sample distributions

mu_prod_true = torch.tensor(0.5)
sigma_prod_true = torch.tensor(0.1)
sigma_meas_true = torch.tensor(0.01)



# ii) Sample from distributions

prod_distribution = pyro.distributions.Normal(mu_prod_true, sigma_prod_true)
prod_lengths = prod_distribution.sample([n_prod])

data_distribution = pyro.distributions.Normal(prod_lengths, sigma_meas_true)
data = data_distribution.sample([n_meas]).T

# The data now is a tensor of shape [n_el, n_meas] and reflects 5 products being
# produced with different length characteristics that are then subsequently measured
# 20 times

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.



"""
    3. Load and customize effects
"""


# i) Set up dimensions for mean parameter mu_prod = same for all measurements

# mu_prod setup
mu_prod_ns = NodeStructure()
mu_prod_ns.set_shape('batch_shape', (), 'Independent values')
mu_prod_ns.set_shape('event_shape', (), 'Repeated values')
mu_prod_object = UnknownParam(mu_prod_ns, name = 'mu')


# i) Set up dimensions for std parameter sigma

# sigma_el setup
sigma_prod_ns = NodeStructure()
sigma_prod_ns.set_shape('batch_shape', (), 'Independent values')
sigma_prod_ns.set_shape('event_shape', (n_prod,), 'Repeated values')
sigma_prod_object = UnknownVar(sigma_prod_ns, name = 'sigma')


# iii) Set up the dimensions for production randomness and noise

# Production randomness
prod_noise_ns = NodeStructure()
prod_noise_ns.set_plate_stack('noise_stack', [('prod_plate', n_prod, -1, 'independent noise 1')], 'Stack containing production randomness')
prod_noise_object = NoiseAddition(prod_noise_ns)

# Measurement randomness
meas_noise_ns = NodeStructure()
meas_noise_ns.set_plate_stack('noise_stack', [('meas_plate_1', n_prod, -2, 'plate denoting independence in row dim'),
                                              ('meas_plate_2', n_meas, -1, 'plate denoting independence in col dim')], 
                              'Stack containing measurement randomness')
meas_noise_object = NoiseAddition(meas_noise_ns)
        


"""
    4. Build the probmodel
"""


# i) Define the probmodel class 

class DemoProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # integrate nodes
        self.mu_prod_object = mu_prod_object
        self.sigma_prod_object = sigma_prod_object
        self.prod_noise_object = prod_noise_object 
        self.meas_noise_object = meas_noise_object 
        
    # Define model by forward passing
    def model(self, input_vars = None, observations = None):
        # parameter initialization
        mu_prod = self.mu_prod_object.forward()       
        sigma_prod = self.sigma_prod_object.forward()
        
        # sample lengths of shape [n_prod] via prod_lengths ~ N(mu_prod, sigma_prod)
        prod_lengths = self.prod_noise_object.forward(input_vars = (mu_prod, sigma_prod))
        
        # sample length measurements of shape [n_prod, n_meas] by adding noise
        # to copies of prod_lengths via meas ~ N(prod_lengths_extended, sigma_meas)
        extension_tensor = torch.ones([n_prod, n_meas])
        prod_lengths_extended = prod_lengths.unsqueeze(1) * extension_tensor
        output = self.meas_noise_object.forward((prod_lengths_extended, sigma_meas_true), observations = observations)     
        
        return output
    
    # Define guide (we ignore the latents here for now)
    def guide(self, input_vars = None, observations = None):
        pass
    
    # # Define guide (we use an autoguide to model the latents)
    # def guide(self, input_vars = None, observations = None):
    #     guide_obj = pyro.infer.autoguide.AutoNormal(self.model)
    #     guide_return = guide_obj(input_vars = None, observations = None)
    #     output = tuple(guide_return.values())
    #     return output
        
demo_probmodel = DemoProbModel()
    


"""
    5. Perform inference
"""
    

# i) Set up optimization

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
n_steps = 1000

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps}


# ii) 

input_data = None
output_data = data
optim_results = demo_probmodel.train(input_data, output_data, optim_opts)
    


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
    
print('True values of mu, sigma = ', mu_prod_true, sigma_prod_true)
print('Results of taking empirical means for mu = ', torch.mean(data))
print('Results of taking empirical std for sigma = ', torch.std(torch.mean(data, dim=1)))






















