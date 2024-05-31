#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimentation with pyro models. Tapemeasure example.

Lets try to do iterative plate construction.
"""


# i) imports and definitions

import pyro
import torch
import copy
import contextlib
import matplotlib.pyplot as plt
from functools import partial

n_tape = 5
n_meas = 20


# ii) Generate data

length_true = torch.ones([n_tape])

coin_flip = pyro.distributions.Bernoulli(probs = torch.tensor(0.5)).sample([n_tape])
bias_true = coin_flip * torch.tensor(0.1)      # measurement bias is either 0 cm or 1 cm
sigma_true = torch.tensor(0.01)                 # measurement standard deviation is 1 cm


# Sample from distributions

data_distribution = pyro.distributions.Normal(length_true + bias_true, sigma_true)
data = data_distribution.sample([n_meas]).T

# The data now has shape [n_tape, n_meas] and reflects n_tape tape measures each
# with a bias of either 0 cm or 1 cm being used to measure an object of length 1 m.


# ii) axiliary functions


def multi_unsqueeze(input_tensor, dims):
    for dim in sorted(dims):
        output_tensor = input_tensor.unsqueeze(dim)
    return output_tensor

# # Function to apply a function within all context managers
# def apply_in_contexts(plates, func, *args, **kwargs):
#     with contextlib.ExitStack() as stack:
#         # Enter all contexts
#         for plate in plates.values():
#             stack.enter_context(plate)
#         # Execute the function inside the nested contexts
#         result = func(*args, **kwargs)
#     return result

# iii) two models


class OffsetModel():
    def __init__(self, offset_shape_dict):  
        self.batch_shape = offset_shape_dict['batch_shape']
        self.event_shape = offset_shape_dict['event_shape']
        self.extension_tensor = multi_unsqueeze(torch.ones(self.event_shape), 
                                                dims = [0 for dim in self.batch_shape])
    def forward(self):
        self.offset = pyro.param('offset', init_tensor = multi_unsqueeze(torch.zeros(self.batch_shape), 
                                            dims = [len(self.batch_shape) for dim in self.event_shape]))
        output = self.extension_tensor * self.offset
        return output

class NoiseModel():
    def __init__(self, noise_shape_dict, noise_plate_dict):
        self.batch_shape = noise_shape_dict['batch_shape']
        self.event_shape = noise_shape_dict['event_shape']
        self.plate_dict = noise_plate_dict
    def forward(self, mean, observations = None):
        self.noise_dist = pyro.distributions.Normal(loc = mean, scale = sigma_true)
        with contextlib.ExitStack() as stack:
            # Dynamically add all plates to the stack
            for plate in self.plate_dict.values():
                stack.enter_context(plate)
            obs = pyro.sample('obs', self.noise_dist, obs = observations)
        return obs
    
class CombinedModel():
    def __init__(self, offset_shape_dict, noise_shape_dict, noise_plate_dict):
        self.offset_model = OffsetModel(offset_shape_dict)
        self.noise_model = NoiseModel(noise_shape_dict, noise_plate_dict)
    def forward(self, observations = None):
        mean = self.offset_model.forward()
        obs = self.noise_model.forward(mean, observations = observations)
        return obs
  
    
offset_shape_dict = {'batch_shape' : (n_tape,),
                     'event_shape' : (n_meas,)}
noise_shape_dict = {'batch_shape' : (n_tape, n_meas),
                     'event_shape' : (0,)}
noise_plate_dict = {'batch_plate_1': pyro.plate('batch_plate_1', size = noise_shape_dict['batch_shape'][0], dim = -2),
                    'batch_plate_2': pyro.plate('batch_plate_2', size = noise_shape_dict['batch_shape'][1], dim = -1)}

combined_model_object = CombinedModel(offset_shape_dict, noise_shape_dict, noise_plate_dict)
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
    loss = svi.step(observations = data)
    if step % 100 == 0:
        print('epoch: {} ; loss : {}'.format(step, loss))
        # grads = {name: pyro.param(name).grad for name in pyro.get_param_store().keys()}
        # print(grads)
    else:
        pass
    loss_sequence.append(loss)
    offset_sequence.append(copy.copy(pyro.get_param_store()['offset'].detach().squeeze().numpy()))


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









