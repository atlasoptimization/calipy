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
from calipy.core.funs import calipy_cat

# ii) Definitions

n_time = 15 # number of timesteps
n_config = 10  # number of full measurement setups
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
sigma_true = 1e-6*torch.eye(3)

# Config specific params
 # Every point random, point coords vary over configs and time
# x_true = torch.normal(0, 100, [n_config, n_time, 3])
# Fixed point per config, point coords vary over configs but stay constant on time
x_true = torch.normal(0,100, [n_config,1,3]) * torch.ones([1,n_time,1]) 

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
cov_T = sqexp(time, time, 0.03, 10) + 0.01*torch.eye(n_time)
cov_P = sqexp(time, time, 0.2, 40) + 0.01*torch.eye(n_time)

list_T = [pyro.distributions.MultivariateNormal(loc = mu_T, covariance_matrix = cov_T).sample().reshape([1,n_time]) for k in range(n_config)]
list_P = [pyro.distributions.MultivariateNormal(loc = mu_P, covariance_matrix = cov_P).sample().reshape([1,n_time]) for k in range(n_config)]

T = torch.cat(list_T,dim = 0)
P = torch.cat(list_P,dim = 0)
Delta_T = T - T0
Delta_P = P - P0

# Distance drift time
coeff_time_1 = 0.005
coeff_time_2 = 5
delta_d_t_true = coeff_time_1 * torch.exp(-coeff_time_2*time)

# Distance drift meteo
coeff_meteo_p = 0.3
coeff_meteo_t = -1
delta_d_m_true = 1e-6*d_true*(coeff_meteo_t * Delta_T + coeff_meteo_p * Delta_P)

# Distribution param distance
mu_d = d_true + delta_d_m_true + delta_d_t_true
mu_d_2f = mu_d.unsqueeze(2).repeat(1, 1, 2)

# Distribution param horizontal angle
face = torch.zeros(n_config, n_time, 2)
face[:, :, 1] = 1
gamma_c_true = c_true / torch.cos(beta_true_2f)
gamma_i_true = i_true * torch.tan(beta_true_2f)

mu_phi_2f = (phi_true_2f + torch.pi * face 
    - gamma_c_true + 2* gamma_c_true * face 
    - gamma_i_true + 2* gamma_i_true * face)

# Distribution param vertical angle
mu_beta = beta_true
mu_beta_2f = beta_true_2f



# ii) Sample from distributions

data = torch.zeros([n_config, n_time, 2, 3])
for k in range(n_config):
    mu_data = torch.vstack([mu_d_2f[k:k+1,:,:], mu_phi_2f[k:k+1,:,:], mu_beta_2f[k:k+1,:,:]])
    mu_data = mu_data.permute([1,2,0])
    data_distribution = pyro.distributions.MultivariateNormal(loc = mu_data,
                                                covariance_matrix = sigma_true)
    
    # Shape is [n_time, 2, 3] for a single sample
    data[k,:, :, :] = data_distribution.sample()

# The data now is a tensor of shape [10,100,2,3] and reflects biased measurements
# of 3D coordinates being made for 10 configurations over 100 timesteps in two
# faces. These measurements have been taken by a total station and are impacted
# by axis deviations, noise, meteoeffects and temporal drift.

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.



"""
    1. Load and customize effects
"""


# i) Set up dimensions

# Base dims data
dim_config = dim_assignment(['dim_config'], dim_sizes = [n_config])
dim_time = dim_assignment(['dim_time'], dim_sizes = [n_time])
dim_face = dim_assignment(['dim_face'], dim_sizes = [2])
dim_3d = dim_assignment(['dim_3d'], dim_sizes = [3])

# Base dims params
dim_trivial = dim_assignment(['dim_trivial'], dim_sizes = [])
dim_single = dim_assignment(['dim_single'], dim_sizes = [1])
dim_coeffs = dim_assignment(['dim_coeffs'], dim_sizes = [2])

# Composite dims
dims_sub = dim_config + dim_time + dim_face
dims_full = dim_config + dim_time + dim_face + dim_3d


# ii) Set up dimensions parameters

# c setup
c_ns = NodeStructure(LatentVariable)
c_ns.set_dims(batch_dims = dims_sub, param_dims = dim_single)
c_object = LatentVariable(c_ns, name = 'c', prior = normal_dist)

# i setup
i_ns = NodeStructure(LatentVariable)
i_ns.set_dims(batch_dims = dims_sub, param_dims = dim_single)
i_object = LatentVariable(i_ns, name = 'i', prior = normal_dist)

# coeffs time-drift
coeffs_t_ns = NodeStructure(UnknownParameter)
coeffs_t_ns.set_dims(batch_dims = dims_sub, param_dims = dim_coeffs)
coeffs_t_object = UnknownParameter(coeffs_t_ns, name = 'coeffs_t',
                                   init_tensor = torch.tensor([0.01, 1]))

# coeffs meteo-drift
coeffs_m_ns = NodeStructure(UnknownParameter)
coeffs_m_ns.set_dims(batch_dims = dims_sub, param_dims = dim_coeffs)
coeffs_m_object = UnknownParameter(coeffs_m_ns, name = 'coeffs_m',
                                   init_tensor = torch.tensor([-2.0, 1.0]))

# d setup
d_ns = NodeStructure(UnknownParameter)
d_ns.set_dims(batch_dims = dim_time + dim_face + dim_single,
              param_dims = dim_config)
d_object = UnknownParameter(d_ns, name = 'd', init_tensor = d_mean)

# phi setup 
phi_ns = NodeStructure(UnknownParameter)
phi_ns.set_dims(batch_dims = dim_time + dim_face + dim_single,
                param_dims = dim_config)
phi_object = UnknownParameter(phi_ns, name = 'phi', init_tensor = phi_mean)

# beta setup
beta_ns = NodeStructure(UnknownParameter)
beta_ns.set_dims(batch_dims = dim_time + dim_face + dim_single,
                 param_dims = dim_config)
beta_object = UnknownParameter(beta_ns, name = 'beta', init_tensor = beta_mean)


# iii) Set up the dimensions for noise addition

noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims = dims_full, event_dims = dim_trivial)
noise_object = NoiseAddition(noise_ns, name = 'noise')


# sigma
sigma_ns = NodeStructure(UnknownParameter)
sigma_ns.set_dims(batch_dims = dims_sub, param_dims = dim_coeffs)
sigma_object = UnknownParameter(sigma_ns, name = 'sigma',
                                   init_tensor = torch.ones([1]))



"""
    2. Build the probmodel
"""


# i) Define the probmodel class 

class DemoProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # integrate nodes
        self.c_object = c_object
        self.i_object = i_object
        self.d_object = d_object
        self.beta_object = beta_object
        self.phi_object = phi_object
        self.coeffs_t_object = coeffs_t_object
        self.coeffs_m_object = coeffs_m_object
        self.noise_object = noise_object 
        
    # Probabilistic model for the total station
    def model(self, input_vars, observations = None):
        
        
        # ii) Untangle input_vars
        
        # Input data considered known
        time = input_vars['time']
        temp = input_vars['temperature']
        press = input_vars['pressure']
        face = input_vars['faces']
        
        # Input data considered known sufficiently for effect computation
        approx_d = input_vars['approx_d']
        approx_beta = input_vars['approx_beta']
        approx_phi = input_vars['approx_phi']
        
        # Derived values
        Delta_T = temp - T0
        Delta_P = press - P0
        
        
        # iii) Set up unknown d, beta, phi
        
        d = self.d_object.forward().reorder(dim_config + dim_time 
                                            + dim_face + dim_single)
        phi = self.phi_object.forward().reorder(dim_config + dim_time 
                                                + dim_face + dim_single)
        beta = self.beta_object.forward().reorder(dim_config + dim_time 
                                                  + dim_face + dim_single)
        
        
        # iv) Set up axis impacts
        
        c = self.c_object.forward()        
        i = self.i_object.forward()
        gamma_c = c / torch.cos(approx_beta)        
        gamma_i = i * torch.tan(approx_beta)
        
        
        # v) Set up drifts
        
        # Temporal drift
        coeffs_t = self.coeffs_t_object.forward()
        delta_d_t = coeffs_t[:,:,:,0:1] * torch.exp(-coeffs_t[:,:,:,1:2]*time)
        
        # Meteo_drift
        coeffs_m = self.coeffs_m_object.forward()
        delta_d_m = 1e-6 * approx_d *(coeffs_m[:,:,:,0:1] * Delta_T 
                                      + coeffs_m[:,:,:,1:2] * Delta_P)
        
        # vi) Set up distributional quantities
        
        mu_d = d + delta_d_t.tensor + delta_d_m.tensor
        mu_beta = beta
        mu_phi = (phi + torch.pi * face 
                  - gamma_c + 2* gamma_c * face 
                  - gamma_i + 2* gamma_i * face)

        mu = calipy_cat((mu_d, mu_phi, mu_beta), dim = 3)

        # inputs = {'mean': mu, 'standard_deviation': sigma_true} 
        inputs = {'mean': mu, 'standard_deviation': torch.tensor(0.001)} 
        output = self.noise_object.forward(input_vars = inputs,
                                           observations = observations)
        
        return output
    
    # Define guide (trivial since no posteriors)
    def guide(self, input_vars, observations = None):
        c_post_mu = c_post_mu_object.forward
        c_post_sigma = c_post_sigma_object.forward
        i_post_mu = i_post_mu_object.forward
        i_post_sigma = i_post_sigma_object.forward
        
        c_posterior = calipy.core.dist.Normal
    
demo_probmodel = DemoProbModel()


# vii) demo run

input_data = {'time': time.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat([10,1,2,1]),
              'temperature' : T.unsqueeze(2).unsqueeze(3).repeat([1,1,2,1]),
              'pressure' : P.unsqueeze(2).unsqueeze(3).repeat([1,1,2,1]),
              'faces' : face.unsqueeze(3),
              'approx_d' : data[:,:,:,0:1],
              'approx_phi' : data[:,:,:,1:2],
              'approx_beta' : data[:,:,:,2:3]}
data_cp = CalipyTensor(data, dims = dim_config + dim_time + dim_face + dim_3d)

# Visualize
import torchviz
graphical_model = pyro.render_model(model = demo_probmodel.model,
                                    model_args= (input_data,),
                                    render_distributions=True, render_params=True)

output = demo_probmodel.model(input_data).value.tensor
comp_graph = torchviz.make_dot(output)
example_output = output.clone().detach()
example_diffs = example_output - data_cp




"""
    3. Perform inference
"""
    

# i) Set up optimization

adam = pyro.optim.NAdam({"lr": 0.003})
elbo = pyro.infer.Trace_ELBO()
n_steps = 300
n_steps_report = 5

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps,
              'n_steps_report' : n_steps_report}


# ii) Train the model

# input_data = {'faces' : face, 'beta' : beta_true}
# data_cp = CalipyTensor(data, dims = dim_config + dim_time + dim_face + dim_3d)
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
    
print('True values \n c : {} \n i : {}'.format(c_true, i_true))
# print('Values estimated by least squares \n c : {} \n i : {}'.format(c_ls, i_ls))


# Plot some values of observations over time

plt.figure(2, dpi = 300)
plt.plot(data[0,:,:,0], label = 'Distance observations point 0')
plt.xlabel('Time')
plt.ylabel('Obs value')
plt.title('Distance observations point 0')

h_angle_diffs = data[:,:,1,:] - data[:,:,0,:] - torch.pi
plt.figure(3, dpi = 300)
plt.plot(h_angle_diffs[0,:,1], label = '2 face h angle differences point 0')
plt.xlabel('Time')
plt.ylabel('Obs value')
plt.title('2 face h angle differences point 0')

plt.figure(4, dpi = 300)
plt.plot(h_angle_diffs[:,0,1], label = '2 face h angle differences time 0')
plt.xlabel('Configurations')
plt.ylabel('Obs value')
plt.title('2 face h angle differences time 0')



# Residual plots

# Initial residuals
initial_d_resids = torch.norm(example_diffs.tensor, dim = [2])[:,:,0]
initial_phi_resids = torch.norm(example_diffs.tensor, dim = [2])[:,:,1]
initial_beta_resids = torch.norm(example_diffs.tensor, dim = [2])[:,:,2]


fig, axes = plt.subplots(3, 1, figsize=(5, 10))

# Plot each image and capture the AxesImage object
im0 = axes[0].imshow(initial_d_resids, aspect='auto', cmap='viridis')
im1 = axes[1].imshow(initial_phi_resids, aspect='auto', cmap='viridis')
im2 = axes[2].imshow(initial_beta_resids, aspect='auto', cmap='viridis')

axes[0].set_title('Initial residuals for d')
axes[1].set_title('Initial residuals for phi')
axes[2].set_title('Initial residuals for beta')

# Add individual colorbars for each subplot
fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

# Remove axis ticks
for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

print('total norm residuals initial', torch.norm(example_diffs.tensor))


# Post optimization residuals
output_post = demo_probmodel.model(input_data).value.tensor
example_output_post = output_post.clone().detach()
example_diffs_post = example_output_post - data_cp

post_d_resids = torch.norm(example_diffs_post.tensor, dim = [2])[:,:,0]
post_phi_resids = torch.norm(example_diffs_post.tensor, dim = [2])[:,:,1]
post_beta_resids = torch.norm(example_diffs_post.tensor, dim = [2])[:,:,2]


fig, axes = plt.subplots(3, 1, figsize=(5, 10))

# Plot each image and capture the AxesImage object
im0 = axes[0].imshow(post_d_resids, aspect='auto', cmap='viridis')
im1 = axes[1].imshow(post_phi_resids, aspect='auto', cmap='viridis')
im2 = axes[2].imshow(post_beta_resids, aspect='auto', cmap='viridis')

axes[0].set_title('Postopt residuals for d')
axes[1].set_title('Postopt residuals for phi')
axes[2].set_title('Postopt residuals for beta')

# Add individual colorbars for each subplot
fig.colorbar(im0, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
fig.colorbar(im1, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
fig.colorbar(im2, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

# Remove axis ticks
for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

print('total norm residuals post optimization', torch.norm(example_diffs_post.tensor))


# plot inferred d, phi, beta - true d, phi, beta
inferred_d = pyro.get_param_store()['Node_5__param_d'].detach().numpy()
inferred_phi = pyro.get_param_store()['Node_6__param_phi'].detach().numpy()
inferred_beta = pyro.get_param_store()['Node_7__param_beta'].detach().numpy()


fig, axes = plt.subplots(3, 1, figsize=(5, 10))

# Plot each image and capture the AxesImage object
axes[0].plot(d_true[:,0] - inferred_d)
axes[1].plot(phi_true[:,0] - inferred_phi)
axes[2].plot(beta_true[:,0] - inferred_beta)

axes[0].set_title('Residuals for estimation of d')
axes[1].set_title('Residuals for estimation of phi')
axes[2].set_title('Residuals for estimation of beta')


plt.tight_layout()
plt.show()