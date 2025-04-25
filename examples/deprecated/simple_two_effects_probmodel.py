#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to model two effects that share some
quantities but produce different output datasets. We will create the effect models
and the probmodel aggregating them, then train it on two datasets to showcase 
the grammar and typical use cases of calipy.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Build the effect models
    4. Build the probmodel
    5. Perform inference
    6. Analyse results and illustrate
In this example we will have two different measurements that are being made. They
share the same variance but have different biases. This leads to two different
datasets of measurement data being collected and they both need to be trained on
jointly. Technically, the same can be achieved by combining the datasets into
a single one featuring a batch_shape of 2, but we want to explore how to do this
differently.

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

n_meas_1 = 30
n_meas_2 = 40



"""
    2. Simulate some data
"""


# i) Set up sample distributions

mu_true_1 = torch.tensor(1.0)
mu_true_2 = torch.tensor(2.0)
sigma_true = torch.tensor(0.1)                 # measurement standard deviation is 1 cm


# ii) Sample from distributions

data_distribution_1 = pyro.distributions.Normal(mu_true_1, sigma_true).expand([n_meas_1])
data_distribution_2 = pyro.distributions.Normal(mu_true_2, sigma_true).expand([n_meas_2])
data_1 = data_distribution_1.sample()
data_2 = data_distribution_2.sample()




"""
    3. Build the effect models
"""


# i) Deterministic offset

# Note: It is actually used as the std in the model below
class Variance(CalipyQuantity):
    def __init__(self, **kwargs):  
        super().__init__(**kwargs)

    def forward(self, variance_shape_dict):
        self.batch_shape = variance_shape_dict['batch_shape']
        self.event_shape = variance_shape_dict['event_shape']
        self.extension_tensor = multi_unsqueeze(torch.ones(self.event_shape), 
                                                dims = [0 for dim in self.batch_shape])
        self.variance = pyro.param('variance_{}'.format(self.node_nr), init_tensor = multi_unsqueeze(torch.ones(self.batch_shape), 
                                            dims = [len(self.batch_shape) for dim in self.event_shape]),
                                   constraint = pyro.distributions.constraints.positive)
        output = self.extension_tensor * self.variance
        return output


class Mean(CalipyQuantity):
    def __init__(self, **kwargs):  
        super().__init__(**kwargs)

    def forward(self, mean_shape_dict):
        self.batch_shape = mean_shape_dict['batch_shape']
        self.event_shape = mean_shape_dict['event_shape']
        self.extension_tensor = multi_unsqueeze(torch.ones(self.event_shape), 
                                                dims = [0 for dim in self.batch_shape])
        self.offset = pyro.param('mean_{}'.format(self.node_nr), init_tensor = multi_unsqueeze(torch.zeros(self.batch_shape), 
                                            dims = [len(self.batch_shape) for dim in self.event_shape]))
        output = self.extension_tensor * self.offset
        return output



# ii) Addition of noise

class AddWhiteNoise(CalipyEffect):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, mean, variance, obs_name, noise_shape_dict, noise_plate_dict, observations = None):
        self.batch_shape = noise_shape_dict['batch_shape']
        self.event_shape = noise_shape_dict['event_shape']
        self.plate_dict = noise_plate_dict
        self.noise_dist = pyro.distributions.Normal(loc = mean, scale = variance)
        # Add all plates to the context stack
        with contextlib.ExitStack() as stack:
            for plate in self.plate_dict.values():
                stack.enter_context(plate)
            obs = pyro.sample(obs_name, self.noise_dist, obs = observations)
        return obs
    


var_shape_dict_1 = {'batch_shape' : (),
                     'event_shape' : (n_meas_1,)}
var_shape_dict_2 = {'batch_shape' : (),
                     'event_shape' : (n_meas_2,)}
# var_name = 'deviations_due_to_cross_talk'
# var_type = 'variance_random_noise'
# var_info = {'description': 'Is noise variance. Can be used for many things'}
# var_dict = {'name' : var_name, 'type': var_type, 'info': var_info}

mean_shape_dict_1 = {'batch_shape' : (),
                     'event_shape' : (n_meas_1,)}
mean_shape_dict_2 = {'batch_shape' : (),
                     'event_shape' : (n_meas_2,)}
# mean_name = 'offset_due_to_miscalibration'
# mean_type = 'tape_measure_offset'
# mean_info = {'description': 'Is a deterministic offset. Can be used for many things'}
# mean_dict = {'name' : mean_name, 'type': mean_type, 'info': mean_info}


noise_shape_dict_1 = {'batch_shape' : (n_meas_1,),
                     'event_shape' : (0,)}
noise_plate_dict_1 = {'batch_plate_1': pyro.plate('batch_plate_1', size = noise_shape_dict_1['batch_shape'][0], dim = -1)}
noise_shape_dict_2 = {'batch_shape' : (n_meas_2,),
                     'event_shape' : (0,)}
noise_plate_dict_2 = {'batch_plate_1': pyro.plate('batch_plate_1', size = noise_shape_dict_2['batch_shape'][0], dim = -1)}
# noise_name = 'noise_due_to_optical_effects'
# noise_type = 'tape_measure_noise'
# noise_info = {'description': 'Is a bunch of noise. Can be used for many things'}
# noise_dict = {'name' : noise_name, 'type': noise_type, 'info': noise_info}


"""
    4. Build the instrument model
"""

# i) Instantiate probabilistic model

# probmodel_TapeMeasure = CalipyProbModel('probmodel_TapeMeasure', 'example', {})


# i) Create TapeMeasure class

# te_type = 'timeseries'
# te_name = 'timeseries_of_lab_experiment_12'
# te_info = {'description': 'The experiment, where someone messed up'}
# te_dict = {'name': ts_name, 'type': ts_type, 'info' : ts_info}

class TwoEffects(CalipyEffect):
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.variance = Variance()
        self.mean_1 = Mean()
        self.mean_2 = Mean()
        self.noise = AddWhiteNoise()
        
    def forward(self, input_vars, observations = None):
        # create subobservations that are None if no data given
        subobs_1 = observations[0] if observations is not None else None
        subobs_2 = observations[1] if observations is not None else None
        # twice the same variance
        variance_1 = self.variance.forward(var_shape_dict_1)
        variance_2 = self.variance.forward(var_shape_dict_2)
        # two different means
        mean_1 = self.mean_1.forward(mean_shape_dict_1)
        mean_2 = self.mean_2.forward(mean_shape_dict_2)
        # two noises with same variance but different shapes
        obs_1 = self.noise.forward(mean_1, variance_1, 'obs_1', noise_shape_dict_1, noise_plate_dict_1, observations = subobs_1)
        obs_2 = self.noise.forward(mean_2, variance_2, 'obs_2', noise_shape_dict_2, noise_plate_dict_2, observations = subobs_2)
                
        return (obs_1, obs_2)


# # Invoke tape_measure
# two_effects = TwoEffects()


# # illustrate
# input_vars = None
# fig = two_effects.render(input_vars)
# # call fig to plot inline


# iv) Guides and partialization

def guide_fn(input_vars, observations = None):
    pass


"""
    5. Build probmodel & perform inference
"""

# # Set up probmodel
# tape_measure_probmodel = CalipyProbModel(name = 'Tape measure')

# # set up optimization
# adam = pyro.optim.NAdam({"lr": 0.01})
# elbo = pyro.infer.Trace_ELBO()
# n_steps = 500

# optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps}

# # run and log optimization
# optim_results = tape_measure_probmodel.train(model_fn, guide_fn, data, optim_opts)


# set up optimization
adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
n_steps = 500

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps}

data = (data_1,data_2)

class TwoEffectsProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.two_effects = TwoEffects()
        self.output_data = data
        self.input_data = None
        
    def model(self, input_vars, observations = None):
        output = self.two_effects.forward(input_vars, observations = observations)
        return output
    def guide(self, input_vars, observations = None):
        output = guide_fn(input_vars, observations = observations)
        return output
    
two_effects_probmodel = TwoEffectsProbModel(name = 'concrete_probmodel')
    
# run and log optimization
input_data = None
output_data = data
optim_results = two_effects_probmodel.train(input_data, output_data, optim_opts)


    
    

"""
    6. Analyse results and illustrate
"""

# convergence
plt.plot(optim_results)

# model params
# alpha_est = pyro.get_param_store()['coeffs'].detach().numpy()
# sigma_est = pyro.get_param_store()['variance'].detach().numpy()
# print('coeffs true:', alpha_true)
# print('coeffs estimated:', alpha_est)
# print('sigma true:', sigma_true)
# print('sigma estimated:', sigma_est)


# data vs predictions
data_simulated = two_effects_probmodel.model(input_data)
fig, ax = plt.subplots(1,2, figsize = (10,5))
ax[0].plot(data[0])
ax[0].plot(data[1])
ax[0].set_title('true_data')

ax[1].plot(data_simulated[0].detach().numpy())
ax[1].plot(data_simulated[1].detach().numpy())
ax[1].set_title('simulated_data')



























