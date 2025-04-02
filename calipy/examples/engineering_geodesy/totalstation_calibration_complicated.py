#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to model a complicated total station
measurement process that involves multiple drifts and correlated noise.
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
from calipy.core.base import NodeStructure, CalipyProbModel
from calipy.core.effects import UnknownParameter, NoiseAddition
from calipy.core.utils import dim_assignment
from calipy.core.tensor import CalipyTensor
from calipy.core.data import CalipyIO


# ii) Definitions

n_time = 100 # number of configurations
n_simu = 10  # number of full measurement setups
def set_seed(seed=42):
    torch.manual_seed(seed)
    pyro.set_rng_seed(seed)

set_seed(123)



"""
    2. Simulate some data
"""


# Standard atmosphere
T0 = 15
P0 = 1013.25

# i) Set up sample distributions

# Global instrument params
i_true = torch.tensor(0.01)
c_true = torch.tensor(0.01)
# sigma_0 = torch.tensor(0.001)

# Config specific params
x_true = torch.normal(0, 100, [n_simu, n_time, 3])
d_true = torch.norm(x_true, dim = 2)
d2_true = torch.norm(x_true[:,:,0:2], dim = 2)
phi_true = torch.sign(x_true[:,:,1]) * torch.arccos(x_true[:,:,0] / d2_true[:,:])
beta_true = torch.arccos(x_true[:,:,2] / d_true[:,:])

# 2 face versions
x_true_2f = x_true.unsqueeze(2).repeat(1, 1, 2, 1)
d_true_2f = d_true.unsqueeze(2).repeat(1, 1, 2)
phi_true_2f = phi_true.unsqueeze(2).repeat(1, 1, 2)
beta_true_2f = beta_true.unsqueeze(2).repeat(1, 1, 2)
                         
# Simulate time, temperature T, pressure P

time = torch.linspace(0,1, n_time)

# Squared Exponential Kernel
def sqexp(x, y, length_scale, variance):
    sqdist = (x[:, None] - y[None, :])**2
    return variance * torch.exp(-0.5 * sqdist / length_scale**2)

# Distributional parameters for P, T
mu_T = T0 * torch.ones([n_time])
mu_P = P0 * torch.ones([n_time])
cov_T = sqexp(time, time, 0.03, 2) + 0.01*torch.eye(n_time)
cov_P = sqexp(time, time, 0.2, 10) + 0.01*torch.eye(n_time)

list_T = [pyro.distributions.MultivariateNormal(loc = mu_T, covariance_matrix = cov_T).sample().reshape([1,n_time]) for k in range(n_simu)]
list_P = [pyro.distributions.MultivariateNormal(loc = mu_P, covariance_matrix = cov_P).sample().reshape([1,n_time]) for k in range(n_simu)]

T = torch.cat(list_T,dim = 0)
P = torch.cat(list_P,dim = 0)
Delta_T = T - T0
Delta_P = P - P0

# Distance drift time
coeff_time_1 = 0.005
coeff_time_2 = 5
delta_d_t = coeff_time_1 * torch.exp(-coeff_time_2*time)

# Distance drift meteo
coeff_meteo_p = 0.3
coeff_meteo_t = -1
delta_d_m = coeff_meteo_t * Delta_T + coeff_meteo_p * Delta_P

# Distribution param distance
mu_d = d_true + delta_d_m + delta_d_t
mu_d_2f = mu_d.unsqueeze(2).repeat(1, 1, 2, 1)

# Distribution param horizontal angle
face = torch.zeros(10, 100, 2)
face[:, :, 1] = 1
gamma_c = c_true / torch.cos(beta_true_2f)
gamma_i = i_true * torch.tan(beta_true_2f)

mu_phi_2f = (phi_true_2f + torch.pi * face 
    - gamma_c + 2* gamma_c * face 
    - gamma_i + 2* gamma_i * face)

# Distribution param vertical angle
mu_beta = beta_true
mu_beta_2f = mu_beta.unsqueeze(2).repeat(1, 1, 2)

sigma = 1e-3*torch.eye(3)

# ii) Sample from distributions
# CONTINUTE HERE
data = torch.zeros([n_simu, n_time,3])
for k in range(n_simu):
    mu_data = torch.vstack([mu_d_2f[k,:,:], mu_phi_2f[k,:,:], mu_beta_2f[k,:,:]])
    data_distribution = pyro.distributions.MultivariateNormal(loc = mu_data, covariance_matrix = sigma)
    data[k,:, :] = data_distribution.sample()

# The data now is a tensor of shape [2,2] and reflects biased measurements being
# taken by a total station impacted by axis errors.

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.



"""
    3. Load and customize effects
"""


# i) Set up dimensions

dim_1 = dim_assignment(['dim_1'], dim_sizes = [n_config])
dim_2 = dim_assignment(['dim_2'], dim_sizes = [2])
dim_3 = dim_assignment(['dim_3'], dim_sizes = [])

# ii) Set up dimensions parameters

# c setup
c_ns = NodeStructure(UnknownParameter)
c_ns.set_dims(batch_dims = dim_1 + dim_2, param_dims = dim_3)
c_object = UnknownParameter(c_ns, name = 'c', init_tensor = torch.tensor(0.1))

# i setup
i_ns = NodeStructure(UnknownParameter)
i_ns.set_dims(batch_dims = dim_1 + dim_2, param_dims = dim_3)
i_object = UnknownParameter(i_ns, name = 'i', init_tensor = torch.tensor(0.1))


# iii) Set up the dimensions for noise addition
noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims = dim_1 + dim_2, event_dims = dim_3)
noise_object = NoiseAddition(noise_ns, name = 'noise')




"""
    4. Build the probmodel
"""


# i) Define the probmodel class 

class DemoProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # integrate nodes
        self.c_object = c_object
        self.i_object = i_object
        self.noise_object = noise_object 
        
    # Define model by forward passing, input_vars = {'faces': face_tensor,
    #                                                 'beta' : beta_tensor}
    def model(self, input_vars, observations = None):
        # Input vars untangling
        face = input_vars.dict['faces']
        beta = input_vars.dict['beta']
        
        # Set up axis impacts
        c = self.c_object.forward()        
        i = self.i_object.forward()
        gamma_c = c / torch.cos(beta)        
        gamma_i = i * torch.tan(beta)

        mu_phi = (phi_true + torch.pi * face 
                  - gamma_c + 2* gamma_c * face 
                  - gamma_i + 2* gamma_i * face)

        inputs = {'mean': mu_phi, 'standard_deviation': sigma_true} 
        output = self.noise_object.forward(input_vars = inputs,
                                           observations = observations)
        
        return output
    
    # Define guide (trivial since no posteriors)
    def guide(self, input_vars, observations = None):
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

input_data = {'faces' : face, 'beta' : beta_true}
data_cp = CalipyTensor(data, dims = dim_1 + dim_2)
optim_results = demo_probmodel.train(input_data, data_cp, optim_opts = optim_opts)


# iii) Solve via handcrafted equations

# first measurement
gamma_c_ls = 0.5*(data[0,1] - data[0,0] - torch.pi)
c_ls = gamma_c_ls * torch.cos(beta_true[0])

# second measurement
gamma_c_hat = c_ls / torch.cos(beta_true[1])
gamma_i_ls = 0.5* ( data[1,1] - data[1,0] - torch.pi - 2*gamma_c_hat)
i_ls = gamma_i_ls/torch.tan(beta_true[1])


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
    
print('True values \n c : {} \n i : {}'.format(c_true, i_true))
print('Values estimated by least squares \n c : {} \n i : {}'.format(c_ls, i_ls))






















