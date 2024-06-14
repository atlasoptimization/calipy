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
from calipy.core.effects import UnknownParameter, UnknownVariance, NoiseAddition


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


# i) Set up dimensions for mean parameters mu_1, mu_2

# Setting up requires correctly specifying a NodeStructure object. You can get 
# a template for the node_structure by calling generate_template() on the example
# node_structure delivered with the class description.
# Here we modify the output of UnknownParam.example_node_structure.generate_template()

# mu_1 setup
mu_1_ns = NodeStructure()
mu_1_ns.set_shape('batch_shape', (), 'Independent values')
mu_1_ns.set_shape('event_shape', (n_meas_1,), 'Repeated values')
mu_1_object = UnknownParameter(mu_1_ns, name = 'mu_1')

# mu_2 setup
mu_2_ns = NodeStructure()
mu_2_ns.set_shape('batch_shape', (), 'Independent values')
mu_2_ns.set_shape('event_shape', (n_meas_2,), 'Repeated values')
mu_2_object = UnknownParameter(mu_2_ns, name = 'mu_2')


# ii) Set up dimensions for sigma
# Make it into a scalar so it automatically broadcasts with mu_1, mu_2
sigma_ns = NodeStructure()
sigma_ns.set_shape('batch_shape', (), 'Independent values')
sigma_ns.set_shape('event_shape', (), 'Repeated values')
sigma_object = UnknownVariance(sigma_ns, name = 'sigma')


# iii) Set up the dimensions for noise addition
# This requires not batch_shapes and event shapes but plate stacks instead - these
# quantities determine conditional independence for stochastic objects. In our 
# case, everything is independent since we prescribe i.i.d. noise.
# Here we modify the output of NoiseAddition.example_node_structure.generate_template()
noise_1_ns = NodeStructure()
noise_1_ns.set_plate_stack('noise_stack', [('batch_plate', n_meas_1, -1, 'independent noise 1')], 'Stack containing noise')
noise_1_object = NoiseAddition(noise_1_ns)
        
noise_2_ns = NodeStructure()
noise_2_ns.set_plate_stack('noise_stack', [('batch_plate', n_meas_2, -1, 'independent noise 2')], 'Stack containing noise')
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
    def model(self, input_vars = None, observations = (None, None)):
        mu_1 = self.mu_1_object.forward()
        mu_2 = self.mu_2_object.forward()
        sigma = self.sigma_object.forward()
        
        output_1 = self.noise_1_object.forward((mu_1, sigma), observations = observations[0])
        output_2 = self.noise_2_object.forward((mu_2, sigma), observations = observations[1])
        output = (output_1, output_2)
        
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
    
print('True values of mu_1, mu_2, sigma = ', mu_true[0], mu_true[1], sigma_true)
print('Results of taking empirical means for mu_1, mu_2 = ', torch.mean(data[0]), torch.mean(data[1]))






















