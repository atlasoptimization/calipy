#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to model a simple measurement process 
with a single unknown image observed multiple times. The measurement procedure has
been used to collect n images, that feature n_y x n_x pixels. Inference is performed 
to estimate the value of the underlying expected value.
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
import matplotlib.pyplot as plt

# calipy
import calipy
from calipy.core.tensor import CalipyTensor
from calipy.core.utils import dim_assignment
from calipy.core.base import NodeStructure, CalipyProbModel
from calipy.core.effects import UnknownParameter, NoiseAddition


# ii) Definitions

n_images = 20
n_x = 5
n_y = 10



"""
    2. Simulate some data
"""


# i) Set up sample distributions

mu_x = torch.linspace(0, 1, n_x)
mu_y = torch.linspace(0, 1, n_y)
mu_yy, mu_xx = torch.meshgrid(mu_y, mu_x, indexing = 'ij')
mu_true = mu_xx * mu_yy
sigma_true = torch.tensor(0.1)


# ii) Sample from distributions

data_distribution = pyro.distributions.Normal(mu_true, sigma_true)
data = data_distribution.sample([n_images])

# The data now is a tensor of shape [n_images, n_y, n_x] and reflects n_images
# measurements being being taken of a single image with a single measurement device.

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.



"""
    3. Load and customize effects
"""


# i) Set up dimensions

batch_dims = dim_assignment(['bd_1'], dim_sizes = [n_images])
event_dims = dim_assignment(['ed_1'], dim_sizes = [])
param_dims = dim_assignment(['pd_1', 'pd_2'], dim_sizes = [n_y, n_x])


# ii) Set up dimensions for mean parameter mu

# mu setup
mu_ns = NodeStructure(UnknownParameter)
mu_ns.set_dims(batch_dims = batch_dims, param_dims = param_dims)
mu_object = UnknownParameter(mu_ns, name = 'mu')


# iii) Set up the dimensions for noise addition

noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims = batch_dims + param_dims, event_dims = event_dims)
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
    5. Perform inference
"""
    

# i) Set up optimization

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
n_steps = 1000

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps}


# ii) Train the model

input_data = None
data_cp = CalipyTensor(data, dims = batch_dims + param_dims)
optim_results = demo_probmodel.train(input_data, data_cp, optim_opts = optim_opts)



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

mu_mean = torch.mean(data, dim = 0)
mu_calipy = pyro.get_param_store()['Node_1__param_mu'].detach()
diff = mu_mean - mu_calipy    

# iii) Create side-by-side plots

fig, axes = plt.subplots(1, 3, figsize=(10, 5))
# Plot first image
axes[0].imshow(mu_true, cmap='gray')
axes[0].set_title('True mu')
axes[0].axis('off')

# Plot second image
axes[1].imshow(mu_mean, cmap='gray')
axes[1].set_title('LS estimation')
axes[1].axis('off')

# Plot third image
axes[2].imshow(mu_calipy, cmap='gray')
axes[2].set_title('calipy estimation')
axes[2].axis('off')

# Show the plot
plt.tight_layout()
plt.show()








