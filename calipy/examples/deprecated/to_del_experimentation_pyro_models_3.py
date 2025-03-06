#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimentation with pyro models.

Lets try it with classes instead of partial functions.
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
fixed_batch_shape = [n_samples]


# iii) two models

class OffsetModel():
    def __init__(self):  
        pass
    def forward(self, batch_shape):
        offset = pyro.param('offset', init_tensor = torch.zeros([1]))
        extension_tensor = torch.ones((batch_shape))
        output = extension_tensor * offset
        return output

class NoiseModel():
    def __init__(self):
        pass
    def forward(self, mean, batch_shape, observations = None):
        noise_dist = pyro.distributions.Normal(loc = mean, scale = 1).expand(batch_shape)
        with pyro.plate('batch_plate', size = batch_shape[0], dim = -1):
            obs = pyro.sample('obs', noise_dist, obs = observations)
        return obs
    
class CombinedModel():
    def __init__(self, batch_shape):
        self.offset_model_object = OffsetModel()
        self.noise_model_object = NoiseModel()
        self.batch_shape = batch_shape
    def forward(self, observations = None):
        mean = self.offset_model_object.forward(self.batch_shape)
        obs = self.noise_model_object.forward(mean, self.batch_shape, observations = observations)
        return obs

combined_model_object = CombinedModel(fixed_batch_shape)
combined_model_fn = combined_model_object.forward



# iv) Guides and partialization

def combined_guide_fn(observations = None):
    pass


# v) inference

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(combined_model_fn, combined_guide_fn, adam, elbo)

loss_sequence = []
offset_sequence = []

for step in range(500):
    loss = svi.step(data)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
        grads = {name: pyro.param(name).grad for name in pyro.get_param_store().keys()}
        print(grads)
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



# offset = offset_model(fixed_batch_shape)
# loss = (offset.forward() - torch.randn(fixed_batch_shape)).sum()
# loss.backward()
# loss
# pyro.param('offset').grad

# noise = noise_model(fixed_batch_shape)
# loss = (noise.forward(offset.forward()) - torch.randn(fixed_batch_shape)).sum()
# loss.backward()
# loss
# pyro.param('offset').grad

# model = combined_model(fixed_batch_shape)
# loss = (model.forward() - torch.randn(fixed_batch_shape)).sum()
# loss.backward()
# loss
# pyro.param('offset').grad









