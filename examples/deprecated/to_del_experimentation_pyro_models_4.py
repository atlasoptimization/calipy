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
    def __init__(self, batch_shape):  
        self.extension_tensor = torch.ones((batch_shape))
    def forward(self):
        self.offset = pyro.param('offset', init_tensor = torch.zeros([1]))
        output = self.extension_tensor * self.offset
        return output

class NoiseModel():
    def __init__(self, batch_shape):
        self.batch_shape= batch_shape
    def forward(self, mean, observations = None):
        self.noise_dist = pyro.distributions.Normal(loc = mean, scale = 1).expand(self.batch_shape)
        with pyro.plate('batch_plate', size = self.batch_shape[0], dim = -1):
            obs = pyro.sample('obs', self.noise_dist, obs = observations)
        return obs
    
class CombinedModel():
    def __init__(self, batch_shape):
        self.offset_model = OffsetModel(batch_shape)
        self.noise_model = NoiseModel(batch_shape)
    def forward(self, observations = None):
        mean = self.offset_model.forward()
        obs = self.noise_model.forward(mean, observations = observations)
        return obs
    
combined_model_object = CombinedModel(fixed_batch_shape)
combined_model_fn = combined_model_object.forward



# iv) Guides and partialization

def combined_guide_fn(observations = None):
    pass


# v) inference

# # Check manual optimization
# param_list = [tensor for name,tensor in pyro.get_param_store().items()]
# optimizer = torch.optim.Adam(param_list, lr=0.01)      
# loss_history = []
        
# for step in range(1000):
#     optimizer.zero_grad()  
#     loss = torch.norm(combined_model_fn() - data)
#     loss.backward()
#     optimizer.step()
#     loss_history.append(loss.item())
#     if step % 100 == 0:
#         print('epoch: {} ; loss : {}'.format(step, loss))

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(combined_model_fn, combined_guide_fn, adam, elbo)

loss_sequence = []
offset_sequence = []

for step in range(500):
    loss = svi.step(observations = data)
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









