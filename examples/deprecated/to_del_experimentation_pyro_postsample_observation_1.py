#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check if delta distribution can be used for post fact observation.
"""

# i) imports and definitons

import torch
import pyro
import matplotlib.pyplot as plt

n_samples = 100

# ii) genrate some data

mean_true = 5*torch.ones([n_samples])
sigma_true = 1*torch.ones([n_samples])

data = pyro.distributions.Normal(mean_true, sigma_true).sample()

# iii) build model

def model(observations = None):
    mu = pyro.param('mu', torch.ones([1]))*torch.ones([n_samples])
    
    #traditional
    # obs_dist = pyro.distributions.Normal(loc = mu, scale = sigma_true)
    # with pyro.plate('batch_plate', size = n_samples, dim = -1):
    #     obs = pyro.sample('obs', obs_dist, obs = observations)
    
    # with delta
    noise_dist = pyro.distributions.Normal(loc = torch.zeros([n_samples]), scale = sigma_true)
    with pyro.plate('batch_plate', size = n_samples, dim = -1):
        noise = pyro.sample('noise', noise_dist)
    
        noisy_mean = mu + noise
        # obs = pyro.sample('obs', pyro.distributions.Delta(noisy_mean), obs = observations)
        obs = pyro.sample('obs', pyro.distributions.Normal(noisy_mean, 0.01), obs = observations)
    return obs

# # traditional
# def guide(obs = None):
#     pass

# with delta
# guide = pyro.infer.autoguide.AutoNormal(model)

# def guide(observations = None):
#     mu_noise = pyro.param('mu_noise', torch.ones([1]))*torch.ones([n_samples])
#     noise_dist = pyro.distributions.Normal(loc = mu_noise, scale = sigma_true)
#     with pyro.plate('batch_plate', size = n_samples, dim = -1):
#         noise = pyro.sample('noise', noise_dist)

# Force posterior = prior (-> mu param learned but not post density)
def guide(observations = None):
    noise_dist = pyro.distributions.Normal(loc = torch.zeros([n_samples]), scale = sigma_true)
    with pyro.plate('batch_plate', size = n_samples, dim = -1):
        noise = pyro.sample('noise', noise_dist)
        return noise

# # Posterior noise = function of observation (-> actually trains the noise to be the observation. interesting but useless right now)
# def guide(observations = None):
#     coeff_noise = pyro.param('coeff_noise', 0*torch.ones([1]))*torch.ones([n_samples])
#     sigma_post = pyro.param('sigma_post', 1*torch.ones([1]), constraint = pyro.distributions.constraints.positive)*torch.ones([n_samples])
#     noise_dist = pyro.distributions.Normal(loc = coeff_noise * observations, scale = sigma_post+0.001)
#     with pyro.plate('batch_plate', size = n_samples, dim = -1):
#         noise = pyro.sample('noise', noise_dist)
#     return noise


# # Posterior noise = Normal(linear comb of obs and params, sigma_post)
# # That actually seems to convirge t the proper solution after (toooo) many steps
# def guide(observations = None):
#     mu = pyro.param('mu', torch.ones([1]))*torch.ones([n_samples])
#     coeffs = pyro.param('coeffs', 0*torch.ones([2]))
#     sigma_post = pyro.param('sigma_post', 1*torch.ones([1]), constraint = pyro.distributions.constraints.positive)*torch.ones([n_samples])
#     noise_dist = pyro.distributions.Normal(loc = coeffs[0]*mu + coeffs[1]*observations, scale = sigma_post+0.001)
#     with pyro.plate('batch_plate', size = n_samples, dim = -1):
#         noise = pyro.sample('noise', noise_dist)
#     return noise


# iv) inference

# specifying scalar options
learning_rate = 1*1e-1
num_epochs = 1000
adam_args = {"lr" : learning_rate}

# Setting up svi
optimizer = pyro.optim.NAdam(adam_args)
elbo_loss = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model = model, guide = guide, optim = optimizer, loss= elbo_loss)


# ii) Execute training

train_loss = []
for epoch in range(num_epochs):
    epoch_loss = svi.step(data)
    train_loss.append(epoch_loss)
    if epoch % 100 == 0:
        # log the data on tensorboard
        print("Epoch : {} train loss : {}".format(epoch, epoch_loss))

plt.plot(train_loss)
print('mu train', pyro.get_param_store()['mu'])
print('mu data', torch.mean(data))

for name, val in pyro.get_param_store().items():
    print(name, val)