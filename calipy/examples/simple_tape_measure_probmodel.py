#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to model a tape measure that is affected
by an additive bias. We will create the instrument model and the probmodel, then
train it on data to showcase the grammar and typical use cases of calipy.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Build the instrument model
    4. Build the probmodel
    5. Perform inference
    6. Analyse results and illustrate
In this example we will have n_tape different tape measures and n_meas different
measurements per tape measure. The bias mu of the measurements is supposed to
be an unknown constant that is different for each tape measure.
These tape measures are used to mease an object with a known length of 1 m, the
bias of the tape measures is to be inferred.

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
import contextlib
import matplotlib.pyplot as plt

# calipy
import calipy
from calipy.core.base import CalipyProbModel
from calipy.core.instruments import CalipyInstrument
from calipy.core.effects import CalipyEffect, CalipyQuantity
from calipy.core.utils import multi_unsqueeze


# ii) Definitions

n_tape = 5
n_meas = 20



"""
    2. Simulate some data
"""


# i) Set up sample distributions

length_true = torch.ones([n_tape])

coin_flip = pyro.distributions.Bernoulli(probs = torch.tensor(0.5)).sample([n_tape])
bias_true = coin_flip * torch.tensor(0.01)      # measurement bias is either 0 cm or 1 cm
sigma_true = torch.tensor(0.01)                 # measurement standard deviation is 1 cm


# ii) Sample from distributions

data_distribution = pyro.distributions.Normal(length_true + bias_true, sigma_true)
data = data_distribution.sample([n_meas]).T

# The data now has shape [n_tape, n_meas] and reflects n_tape tape measures each
# with a bias of either 0 cm or 1 cm being used to measure an object of length 1 m.

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.



"""
    3. Build the effect models
"""


# i) Deterministic offset


class DeterministicOffset(CalipyQuantity):
    def __init__(self, offset_shape_dict, **kwargs):  
        super().__init__(**kwargs)
        self.batch_shape = offset_shape_dict['batch_shape']
        self.event_shape = offset_shape_dict['event_shape']
        self.extension_tensor = multi_unsqueeze(torch.ones(self.event_shape), 
                                                dims = [0 for dim in self.batch_shape])
    def forward(self):
        self.offset = pyro.param('offset', init_tensor = multi_unsqueeze(torch.zeros(self.batch_shape), 
                                            dims = [len(self.batch_shape) for dim in self.event_shape]))
        output = self.extension_tensor * self.offset
        return output

# ii) Addition of noise

class WhiteNoise(CalipyEffect):
    def __init__(self, noise_shape_dict, noise_plate_dict, **kwargs):
        super().__init__(**kwargs)
        self.batch_shape = noise_shape_dict['batch_shape']
        self.event_shape = noise_shape_dict['event_shape']
        self.plate_dict = noise_plate_dict
    def forward(self, mean, observations = None):
        self.noise_dist = pyro.distributions.Normal(loc = mean, scale = sigma_true)
        # Add all plates to the context stack
        with contextlib.ExitStack() as stack:
            for plate in self.plate_dict.values():
                stack.enter_context(plate)
            obs = pyro.sample('obs', self.noise_dist, obs = observations)
        return obs
    


offset_shape_dict = {'batch_shape' : (n_tape,),
                     'event_shape' : (n_meas,)}
offset_name = 'offset_due_to_miscalibration'
offset_type = 'tape_measure_offset'
offset_info = {'description': 'Is a deterministic offset. Can be used for many things'}
offset_dict = {'name' : offset_name, 'type': offset_type, 'info': offset_info}


noise_shape_dict = {'batch_shape' : (n_tape, n_meas),
                     'event_shape' : (0,)}
noise_plate_dict = {'batch_plate_1': pyro.plate('batch_plate_1', size = noise_shape_dict['batch_shape'][0], dim = -2),
                    'batch_plate_2': pyro.plate('batch_plate_2', size = noise_shape_dict['batch_shape'][1], dim = -1)}
noise_name = 'noise_due_to_optical_effects'
noise_type = 'tape_measure_noise'
noise_info = {'description': 'Is a bunch of noise. Can be used for many things'}
noise_dict = {'name' : noise_name, 'type': noise_type, 'info': noise_info}


"""
    4. Build the instrument model
"""

# i) Instantiate probabilistic model

probmodel_TapeMeasure = CalipyProbModel('probmodel_TapeMeasure', 'example', {})


# i) Create TapeMeasure class

tm_type = 'tape_measure'
tm_name = 'tape_measure_nr_17'
tm_info = {'description': 'That weird one with the abrasions'}
tm_dict = {'name': tm_name, 'type': tm_type, 'info' : tm_info}

class TapeMeasure(CalipyInstrument):
        
    def __init__(self, offset_shape_dict, noise_shape_dict, noise_plate_dict, **kwargs):
        super().__init__(**kwargs)
        
        
        self.offset_model = DeterministicOffset(offset_shape_dict, **offset_dict)
        self.noise_model = WhiteNoise(noise_shape_dict, noise_plate_dict, **noise_dict)
        
    def forward(self, observations = None):
        mean = self.offset_model.forward()
        obs = self.noise_model.forward(mean, observations = observations)
        return obs


# Invoke tape_measure
tape_measure = TapeMeasure(offset_shape_dict, noise_shape_dict, noise_plate_dict, **tm_dict)
model_fn = tape_measure.forward

# illustrate
fig = tape_measure.render()
# call fig to plot inline


# iv) Guides and partialization

def guide_fn(observations = None):
    pass


"""
    5. Build probmodel & perform inference
"""

# Set up probmodel
tape_measure_probmodel = CalipyProbModel(model_name = 'Tape measure', info_dict = {})

# Integrate tape_measure
tape_measure_probmodel

# set up optimization
adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': 500}

# run and log optimization
optim_results = tape_measure_probmodel.train(model_fn, guide_fn, data, optim_opts)


    
    

"""
    6. Analyse results and illustrate
"""






























