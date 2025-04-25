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
coeff_time_1 = 0.01
coeff_time_2 = 1
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
    3. Load and customize effects
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
c_ns = NodeStructure(UnknownParameter)
c_ns.set_dims(batch_dims = dims_sub, param_dims = dim_single)
c_object = UnknownParameter(c_ns, name = 'c', init_tensor = torch.tensor([0.05]))

# i setup
i_ns = NodeStructure(UnknownParameter)
i_ns.set_dims(batch_dims = dims_sub, param_dims = dim_single)
i_object = UnknownParameter(i_ns, name = 'i', init_tensor = torch.tensor([0.05]))

# coeffs time-drift
coeffs_t_ns = NodeStructure(UnknownParameter)
coeffs_t_ns.set_dims(batch_dims = dims_sub, param_dims = dim_coeffs)
coeffs_t_object = UnknownParameter(coeffs_t_ns, name = 'coeffs_t',
                                   init_tensor = torch.tensor([0.1, 1]))

# coeffs meteo-drift
coeffs_m_ns = NodeStructure(UnknownParameter)
coeffs_m_ns.set_dims(batch_dims = dims_sub, param_dims = dim_coeffs)
coeffs_m_object = UnknownParameter(coeffs_m_ns, name = 'coeffs_m',
                                   init_tensor = torch.tensor([-2.0, 1.0]))

# d setup
d_ns = NodeStructure(UnknownParameter)
d_ns.set_dims(batch_dims = dim_time + dim_face + dim_single, param_dims = dim_config)
d_object = UnknownParameter(d_ns, name = 'd', init_tensor = torch.mean(data[:,:,:,0], dim = [1,2]))

# phi setup 
# phi is more problematic producing bigger residuals since the impacted observations
# are taken as initialization - thereby making mu_phi = init + effect initialize
# incorrectly event then everything else is chose correctly
phi_ns = NodeStructure(UnknownParameter)
phi_ns.set_dims(batch_dims = dim_time + dim_face + dim_single, param_dims = dim_config)
phi_object = UnknownParameter(phi_ns, name = 'phi', init_tensor = torch.mean(data[:,:,0,1], dim = [1]))
# If we would initialize as follows, then we would directly start with optimal configuration:
# phi_object = UnknownParameter(phi_ns, name = 'phi', init_tensor = phi_true_2f[:,0,0])

# beta setup
beta_ns = NodeStructure(UnknownParameter)
beta_ns.set_dims(batch_dims = dim_time + dim_face + dim_single, param_dims = dim_config)
beta_object = UnknownParameter(beta_ns, name = 'beta', init_tensor = torch.mean(data[:,:,:,2], dim = [1,2]))


# iii) Set up the dimensions for noise addition

noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims = dims_full, event_dims = dim_trivial)
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
        self.d_object = d_object
        self.beta_object = beta_object
        self.phi_object = phi_object
        self.coeffs_t_object = coeffs_t_object
        self.coeffs_m_object = coeffs_m_object
        self.noise_object = noise_object 
        
    # Define model by forward passing, input_vars = dict with keys ['time', 
    #   'temperature', 'pressure', 'faces', 'approx_d', 'approx_beta', 'approx_phi']
    def model(self, input_vars, observations = None):
        
        # ii) Untangle input_vars - these quantities are needed as inputs to effect
        # computation and for forward simulation.
        
        # Input data considered known
        time = input_vars['time']
        temp = input_vars['temperature']
        press = input_vars['pressure']
        face = input_vars['faces']
        
        # Input data considered known approximately (sufficient for effect computation)
        approx_d = input_vars['approx_d']
        approx_beta = input_vars['approx_beta']
        approx_phi = input_vars['approx_phi']
        
        # Derived values
        Delta_T = temp - T0
        Delta_P = press - P0
        
        
        # iii) Set up unknown d, beta, phi
        
        d = self.d_object.forward().reorder(dim_config + dim_time + dim_face + dim_single)
        phi = self.phi_object.forward().reorder(dim_config + dim_time + dim_face + dim_single)
        beta = self.beta_object.forward().reorder(dim_config + dim_time + dim_face + dim_single)
        
        
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
        pass
    
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
output_data_io = CalipyIO(data_cp)
input_data_io = CalipyIO(input_data)

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
    5. Perform inference
"""
    

# i) Set up optimization

n_steps = 350


# ii) Train the model

# Fetch optional arguments
adam = pyro.optim.NAdam({"lr": 0.003})
loss = pyro.infer.Trace_ELBO()
n_steps_report = 10

# Set optimizer and initialize training
svi = pyro.infer.SVI(demo_probmodel.model, demo_probmodel.guide, adam, loss)

epochs = []
loss_sequence = []
param_sequence_d = []
param_sequence_phi = []
param_sequence_beta = []
noisy_obs_sequence = []
d_resid_images = []
phi_resid_images = []
beta_resid_images = []
# noisy_obs_sequence_many = []

n_simu = 1
# Handle direct data input case
for step in range(n_steps):
    loss = svi.step(input_vars=input_data_io, observations=output_data_io)
    param_d_val = pyro.get_param_store()['Node_5__param_d'].clone().detach()
    param_phi_val = pyro.get_param_store()['Node_6__param_phi'].clone().detach()
    param_beta_val = pyro.get_param_store()['Node_7__param_beta'].clone().detach()
    
    sim_vals = demo_probmodel.model(input_data_io).value.tensor.clone().detach()
    # sim_vals_many = [demo_probmodel.model(input_data_io).value.tensor.clone().detach()[0,0] for k in range(n_simu)]
    
    if step % n_steps_report == 0:
        print(f'epoch: {step} ; loss : {loss}')
    
    
    epochs.append(step)
    loss_sequence.append(loss)
    param_sequence_d.append(param_d_val.numpy())
    param_sequence_phi.append(param_phi_val.numpy())
    param_sequence_beta.append(param_beta_val.numpy())
    noisy_obs_sequence.append(sim_vals)
    
    resid = sim_vals - data_cp
    d_resid_images.append(resid[:,:,0,0].tensor.squeeze())
    phi_resid_images.append(resid[:,:,0,1].tensor.squeeze())
    beta_resid_images.append(resid[:,:,0,2].tensor.squeeze())
    # noisy_obs_sequence_many.append(torch.stack(sim_vals_many).numpy())
    


"""
    6. Analyse results and illustrate
"""


# # i)  Plot loss

# plt.figure(1, dpi = 300)
# plt.plot(optim_results)
# plt.title('ELBO loss')
# plt.xlabel('epoch')

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




train_data = []
for k in range(n_steps):
    train_data.append([epochs[k], 
                       param_sequence_d[k]- d_true[:,0].numpy(),
                       param_sequence_phi[k]- phi_true[:,0].numpy(),
                       param_sequence_beta[k]- beta_true[:,0].numpy(),
                       loss_sequence[k],
                       noisy_obs_sequence[k].flatten()])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def make_training_video_no_colorbar(
    train_data,
    d_images, phi_images, beta_images,   # each: list of length n_steps, shape (Nconfig, Ntime)
    n_steps,
    # X-axis and Y-lims for line plots
    epoch_xlim=(0, None),
    elbo_ylim=(None, None),
    # We won't autoscale these line subplots for d, phi, beta
    resid_d_ylim=(None, None),
    resid_phi_ylim=(None, None),
    resid_beta_ylim=(None, None),
    # If you want to fix color ranges for images (but no colorbar):
    d_clim=(-0.1, 0.1),
    phi_clim=(-0.1, 0.1),
    beta_clim=(-0.1, 0.1),
    # Fixed bin edges for each histogram
    d_bin_edges=None,
    phi_bin_edges=None,
    beta_bin_edges=None,
    output_path="training_sim_grid.gif",
    fps=3,
    cmap='viridis'
):
    """
    3×4 grid with arrangement:
      [0,0] => ELBO (single line)      [0,1] => d param residual (10 lines)   [0,2] => d image    [0,3] => d histogram
      [1,0] => empty                   [1,1] => phi param residual (10 lines) [1,2] => phi image  [1,3] => phi histogram
      [2,0] => empty                   [2,1] => beta param residual (10 lines)[2,2] => beta image [2,3] => beta histogram

    - No colorbar
    - Each param residual subplot has multiple lines (one for each channel).
    - The images have axis ticks disabled.
    - The histograms update each frame (they use data from .ravel() of the images, or from separate arrays).
    - train_data[frame] = [epoch_k, d_k, phi_k, beta_k, elbo_k, ...]
      where d_k, phi_k, beta_k are shape (10,) if you have 10 channels.

    Modify as needed if you want fewer/more channels.
    """

    fig, axes = plt.subplots(3, 4, figsize=(12, 7), sharex=False, sharey=False)

    # Layout
    ax_elbo       = axes[0,0]
    ax_resid_d    = axes[0,1]
    ax_img_d      = axes[0,2]
    ax_hist_d     = axes[0,3]

    ax_empty1     = axes[1,0]
    ax_resid_phi  = axes[1,1]
    ax_img_phi    = axes[1,2]
    ax_hist_phi   = axes[1,3]

    ax_empty2     = axes[2,0]
    ax_resid_beta = axes[2,1]
    ax_img_beta   = axes[2,2]
    ax_hist_beta  = axes[2,3]

    # Hide empty subplots
    ax_empty1.set_visible(False)
    ax_empty2.set_visible(False)

    fig.suptitle("3×4 Training Animation", fontsize=14)

    # Titles
    ax_elbo.set_title("ELBO")
    ax_resid_d.set_title("Param d residual (10 lines)")
    ax_resid_phi.set_title("Param φ residual (10 lines)")
    ax_resid_beta.set_title("Param β residual (10 lines)")

    ax_img_d.set_title("d residual image")
    ax_img_phi.set_title("φ residual image")
    ax_img_beta.set_title("β residual image")

    ax_hist_d.set_title("Hist d")
    ax_hist_phi.set_title("Hist φ")
    ax_hist_beta.set_title("Hist β")
    ax_hist_d.set_ylim(0,50)
    ax_hist_phi.set_ylim(0, 50)
    ax_hist_beta.set_ylim(0, 50)

    # X-axes for line plots
    ax_elbo.set_xlabel("Epoch")
    ax_resid_d.set_xlabel("Epoch")
    ax_resid_phi.set_xlabel("Epoch")
    ax_resid_beta.set_xlabel("Epoch")

    # Fix the epoch range if user didn't specify
    if epoch_xlim[1] is None:
        epoch_xlim = (epoch_xlim[0], n_steps)
    ax_elbo.set_xlim(*epoch_xlim)
    ax_resid_d.set_xlim(*epoch_xlim)
    ax_resid_phi.set_xlim(*epoch_xlim)
    ax_resid_beta.set_xlim(*epoch_xlim)

    # Fix y-lims if specified
    if elbo_ylim != (None, None):
        ax_elbo.set_ylim(*elbo_ylim)
    if resid_d_ylim != (None, None):
        ax_resid_d.set_ylim(*resid_d_ylim)
    if resid_phi_ylim != (None, None):
        ax_resid_phi.set_ylim(*resid_phi_ylim)
    if resid_beta_ylim != (None, None):
        ax_resid_beta.set_ylim(*resid_beta_ylim)

    # Prepare images (no colorbar)
    im_d = ax_img_d.imshow(np.zeros_like(d_images[0]), cmap=cmap, aspect='auto',
                           vmin=d_clim[0], vmax=d_clim[1])
    im_phi = ax_img_phi.imshow(np.zeros_like(phi_images[0]), cmap=cmap, aspect='auto',
                               vmin=phi_clim[0], vmax=phi_clim[1])
    im_beta = ax_img_beta.imshow(np.zeros_like(beta_images[0]), cmap=cmap, aspect='auto',
                                 vmin=beta_clim[0], vmax=beta_clim[1])

    # Turn off axis ticks for image subplots
    ax_img_d.axis('off')
    ax_img_phi.axis('off')
    ax_img_beta.axis('off')

    # Hist bin edges
    if d_bin_edges is None:
        d_bin_edges = np.linspace(0, 1, 21)
    if phi_bin_edges is None:
        phi_bin_edges = np.linspace(0, 1, 21)
    if beta_bin_edges is None:
        beta_bin_edges = np.linspace(0, 1, 21)

    # Initialize empty hist
    _, _, patches_d    = ax_hist_d.hist([], bins=d_bin_edges, alpha=0.6, color="blue")
    _, _, patches_phi  = ax_hist_phi.hist([], bins=phi_bin_edges, alpha=0.6, color="blue")
    _, _, patches_beta = ax_hist_beta.hist([], bins=beta_bin_edges, alpha=0.6, color="blue")

    # Prepare ELBO single line
    line_elbo, = ax_elbo.plot([], [], 'b-', lw=2)
    epoch_vals = []
    elbo_vals  = []

    # Now we have multiple channels for d, phi, beta. Suppose each is shape (10,).
    # We'll create 10 lines for each param.
    num_channels = n_config  # adapt as needed

    # d-lines
    line_d = []
    d_vals = [[] for _ in range(num_channels)]  # d_vals[i] = the time series for channel i
    for i in range(num_channels):
        linei, = ax_resid_d.plot([], [], lw=1.5, label=f"d[{i}]")
        line_d.append(linei)

    # phi-lines
    line_phi = []
    phi_vals = [[] for _ in range(num_channels)]
    for i in range(num_channels):
        linei, = ax_resid_phi.plot([], [], lw=1.5, label=f"phi[{i}]")
        line_phi.append(linei)

    # beta-lines
    line_beta = []
    beta_vals = [[] for _ in range(num_channels)]
    for i in range(num_channels):
        linei, = ax_resid_beta.plot([], [], lw=1.5, label=f"beta[{i}]")
        line_beta.append(linei)

    def init():
        line_elbo.set_data([], [])

        # Clear the param lines
        for i in range(num_channels):
            line_d[i].set_data([], [])
            line_phi[i].set_data([], [])
            line_beta[i].set_data([], [])

        # Clear images
        im_d.set_data(np.zeros_like(d_images[0]))
        im_phi.set_data(np.zeros_like(phi_images[0]))
        im_beta.set_data(np.zeros_like(beta_images[0]))

        # Clear hist
        for p in patches_d:   p.set_height(0)
        for p in patches_phi: p.set_height(0)
        for p in patches_beta:p.set_height(0)

        return (
            line_elbo,
            *line_d, *line_phi, *line_beta,
            im_d, im_phi, im_beta,
            *patches_d, *patches_phi, *patches_beta
        )

    def update(frame):
        row = train_data[frame]
        # row = [epoch_k, d_k, phi_k, beta_k, elbo_k, ...]
        # each d_k, phi_k, beta_k is shape (10,)
        epoch_k   = row[0]
        d_k       = row[1]
        phi_k     = row[2]
        beta_k    = row[3]
        elbo_k    = row[4]

        # single line for elbo
        epoch_vals.append(epoch_k)
        elbo_vals.append(elbo_k)
        line_elbo.set_data(epoch_vals, elbo_vals)

        # multiple lines for d
        for i in range(num_channels):
            d_vals[i].append(d_k[i])  # each channel
            line_d[i].set_data(epoch_vals, d_vals[i])

        # multiple lines for phi
        for i in range(num_channels):
            phi_vals[i].append(phi_k[i])
            line_phi[i].set_data(epoch_vals, phi_vals[i])

        # multiple lines for beta
        for i in range(num_channels):
            beta_vals[i].append(beta_k[i])
            line_beta[i].set_data(epoch_vals, beta_vals[i])

        # Update images
        im_d.set_data(d_images[frame])
        im_phi.set_data(phi_images[frame])
        im_beta.set_data(beta_images[frame])

        # Hist update
        # If you have separate arrays, plug them in. Here we just use the images' data:
        d_resid_array   = d_images[frame].ravel()
        phi_resid_array = phi_images[frame].ravel()
        beta_resid_array= beta_images[frame].ravel()

        d_counts, _    = np.histogram(d_resid_array, bins=d_bin_edges)
        phi_counts, _  = np.histogram(phi_resid_array, bins=phi_bin_edges)
        beta_counts, _ = np.histogram(beta_resid_array, bins=beta_bin_edges)

        for c, p in zip(d_counts, patches_d):
            p.set_height(c)
        for c, p in zip(phi_counts, patches_phi):
            p.set_height(c)
        for c, p in zip(beta_counts, patches_beta):
            p.set_height(c)

        return (
            line_elbo,
            *line_d, *line_phi, *line_beta,
            im_d, im_phi, im_beta,
            *patches_d, *patches_phi, *patches_beta
        )

    anim = FuncAnimation(
        fig, update,
        frames=n_steps,
        init_func=init,
        blit=False,
        interval=200,
        repeat=False
    )

    plt.tight_layout()
    anim.save(output_path, writer="pillow", fps=fps, dpi=150)
    print(f"Animation saved to {output_path}")


make_training_video_no_colorbar(
    train_data=train_data,
    d_images=d_resid_images, phi_images=phi_resid_images, beta_images=beta_resid_images,
    n_steps=n_steps,
    epoch_xlim=(0, n_steps),   # x-axis for line plots
    elbo_ylim=(-1e5, 1e6),
    resid_d_ylim=(-0.01,0.01),
    resid_phi_ylim=(-0.2,0.2),
    resid_beta_ylim=(-0.01,0.01),
    d_clim=(-0.01,0.01),
    phi_clim=(-0.1,0.1),
    beta_clim=(-0.01,0.01),
    d_bin_edges=np.linspace(-0.01,0.01,21),
    phi_bin_edges=np.linspace(-0.05,0.05,21),
    beta_bin_edges=np.linspace(-0.01,0.01,21),
    output_path="training_sim_grid.gif",
    fps=15
)

    



# # -----------------------------------------------------------------
# # Example usage
# if __name__ == "__main__":
#     n_steps = 8
#     train_data = []
#     d_images   = []
#     phi_images = []
#     beta_images= []

#     for k in range(n_steps):
#         epoch_k = k
#         d_k     = 0.01*np.random.randn()
#         phi_k   = 0.02*np.random.randn()
#         beta_k  = 0.03*np.random.randn()
#         elbo_k  = 1000/(k+1)+50*np.random.randn()

#         # store in train_data
#         train_data.append([epoch_k, d_k, phi_k, beta_k, elbo_k])

#         # images
#         d_img   = np.random.rand(10, 8)*0.2 + 0.1
#         phi_img = np.random.rand(10, 8)*0.3 + 0.2
#         beta_img= np.random.rand(10, 8)*0.1 - 0.05

#         d_images.append(d_img)
#         phi_images.append(phi_img)
#         beta_images.append(beta_img)

#     make_training_video_no_colorbar(
#         train_data=train_data,
#         d_images=d_images, phi_images=phi_images, beta_images=beta_images,
#         n_steps=n_steps,
#         epoch_xlim=(0, n_steps),   # x-axis for line plots
#         elbo_ylim=(0, 1200),
#         resid_d_ylim=(-0.1,0.1),
#         resid_phi_ylim=(-0.2,0.2),
#         resid_beta_ylim=(-0.3,0.3),
#         d_clim=(0.0,0.3),
#         phi_clim=(0.2,0.5),
#         beta_clim=(-0.05,0.05),
#         d_bin_edges=np.linspace(0,0.3,21),
#         phi_bin_edges=np.linspace(0.2,0.5,21),
#         beta_bin_edges=np.linspace(-0.05,0.05,21),
#         output_path="training_sim_grid.gif",
#         fps=2
#     )
