#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimentation with pyro models.

Can submodels chained and what is the implication of the plates being defined in the
submodels then?
"""


# i) imports and definitions

import pyro
import torch
import matplotlib.pyplot as plt
from functools import partial

n_samples = 100


# ii) Generate data

mu_true = 1
sigma_true = 1
data = pyro.distributions.Normal(mu_true, sigma_true).sample([n_samples])


# iii) two models

def offset_model(batch_shape):
    offset = pyro.param('offset', init_tensor = torch.zeros([1]))
    return offset

def noise_model(batch_shape, mean, observations = None):
    noise_dist = pyro.distributions.Normal(loc = mean, scale = 1).expand(batch_shape)
    
    with pyro.plate('batch_plate', size = batch_shape[0], dim = -1):
        obs = pyro.sample('obs', noise_dist, obs = observations)
    return obs


def combined_model(batch_shape, observations = None):
    mean = offset_model(batch_shape)
    obs = noise_model(batch_shape, mean, observations)
    return obs


# iv) Guides and partialization

def combined_guide(batch_shape, observations = None):
    pass

fixed_batch_shape = [n_samples]
partial_model = partial(combined_model, batch_shape = fixed_batch_shape)
partial_guide = partial(combined_guide, batch_shape = fixed_batch_shape)


# v) inference

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(partial_model, partial_guide, adam, elbo)

loss_sequence = []
offset_sequence = []
for step in range(500):
    loss = svi.step(observations = data)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
    else:
        pass
    loss_sequence.append(loss)
    offset_sequence.append(pyro.get_param_store()['offset'].item())


# vi) Training process

fig, ax = plt.subplots(1,2, figsize = (10,5))
ax[0].plot(loss_sequence)
ax[0].set_title('Loss during computation')

ax[1].plot(offset_sequence)
ax[1].set_title('Estimated mu during computation')

