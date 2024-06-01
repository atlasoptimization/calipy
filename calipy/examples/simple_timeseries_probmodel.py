#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to model a timeseries featuring 
second order polynomial drift and noise of a constant level. We will create the
effect model and the probmodel, then train it on data to showcase the grammar 
and typical use cases of calipy.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Build the effect models
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

n_ts = 5
n_time = 10
time = torch.linspace(0,1,n_time)

n_coeff = 3 # second order polynomial drift


"""
    2. Simulate some data
"""


# i) Set up sample distributions

alpha_true = torch.linspace(0,1,n_ts).repeat([n_coeff,1]).T
sigma_true = torch.tensor(0.1)                 # measurement standard deviation is 1 cm


# ii) Sample from distributions
A_mat = torch.cat([time.unsqueeze(0)**k for k in range(n_coeff)], dim = 0).T
trends_true = (A_mat@(alpha_true.T)).T
data_distribution = pyro.distributions.Normal(trends_true, sigma_true)
data = data_distribution.sample()

# The data now has shape [n_tape, n_meas] and reflects n_tape tape measures each
# with a bias of either 0 cm or 1 cm being used to measure an object of length 1 m.

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.



"""
    3. Build the effect models
"""


# i) Deterministic offset

# Note: It is actually used as the std in the model below
class Variance(CalipyQuantity):
    def __init__(self, variance_shape_dict, **kwargs):  
        super().__init__(**kwargs)
        self.batch_shape = variance_shape_dict['batch_shape']
        self.event_shape = variance_shape_dict['event_shape']
        self.extension_tensor = multi_unsqueeze(torch.ones(self.event_shape), 
                                                dims = [0 for dim in self.batch_shape])
    def forward(self):
        self.variance = pyro.param('variance', init_tensor = multi_unsqueeze(torch.ones(self.batch_shape), 
                                            dims = [len(self.batch_shape) for dim in self.event_shape]),
                                   constraint = pyro.distributions.constraints.positive)
        output = self.extension_tensor * self.variance
        return output


class PolynomialTrend(CalipyEffect):
    def __init__(self, trend_shape_dict, **kwargs):  
        super().__init__(**kwargs)
        self.batch_shape = trend_shape_dict['batch_shape']
        self.event_shape = trend_shape_dict['event_shape']
        self.n_coeffs = trend_shape_dict['n_coeffs']
        
    def forward(self, input_vars):
        self.coeffs = pyro.param('coeffs', init_tensor = torch.ones(self.batch_shape + (self.n_coeffs,)))
        self.A_mat = torch.cat([input_vars.unsqueeze(-1)**k for k in range(self.n_coeffs)], dim = -1)

        output = torch.einsum('bjk, bk -> bj' , self.A_mat, self.coeffs )
        return output

# ii) Addition of noise

class WhiteNoise(CalipyEffect):
    def __init__(self, noise_shape_dict, noise_plate_dict, **kwargs):
        super().__init__(**kwargs)
        self.batch_shape = noise_shape_dict['batch_shape']
        self.event_shape = noise_shape_dict['event_shape']
        self.plate_dict = noise_plate_dict
    def forward(self, mean, variance, observations = None):
        self.noise_dist = pyro.distributions.Normal(loc = mean, scale = variance)
        # Add all plates to the context stack
        with contextlib.ExitStack() as stack:
            for plate in self.plate_dict.values():
                stack.enter_context(plate)
            obs = pyro.sample('obs', self.noise_dist, obs = observations)
        return obs
    


var_shape_dict = {'batch_shape' : (),
                     'event_shape' : (n_ts, n_time,)}
var_name = 'deviations_due_to_cross_talk'
var_type = 'variance_random_noise'
var_info = {'description': 'Is a deterministic offset. Can be used for many things'}
var_dict = {'name' : var_name, 'type': var_type, 'info': var_info}

trend_shape_dict = {'batch_shape' : (n_ts,),
                     'event_shape' : (n_time,),
                     'n_coeffs' : n_coeff}
trend_name = 'drift due to warmup'
trend_type = 'timeseries_trend'
trend_info = {'description': 'Is a deterministic offset. Can be used for many things'}
trend_dict = {'name' : trend_name, 'type': trend_type, 'info': trend_info}


noise_shape_dict = {'batch_shape' : (n_ts, n_time),
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

# probmodel_TapeMeasure = CalipyProbModel('probmodel_TapeMeasure', 'example', {})


# i) Create TapeMeasure class

ts_type = 'timeseries'
ts_name = 'timeseries_of_lab_experiment_12'
ts_info = {'description': 'The experiment, where someone messed up'}
ts_dict = {'name': ts_name, 'type': ts_type, 'info' : ts_info}

class TotalstationDrift(CalipyEffect):
        
    def __init__(self, trend_shape_dict, noise_shape_dict, noise_plate_dict, **kwargs):
        super().__init__(**kwargs)
        
        self.variance_model = Variance(var_shape_dict, **var_dict)
        self.polynomial_model = PolynomialTrend(trend_shape_dict, **trend_dict)
        self.noise_model = WhiteNoise(noise_shape_dict, noise_plate_dict, **noise_dict)
        
    def forward(self, input_vars, observations = None):
        variance = self.variance_model.forward()
        mean = self.polynomial_model.forward(input_vars)
        obs = self.noise_model.forward(mean, variance, observations = observations)
        return obs


# Invoke tape_measure
totalstation_drift = TotalstationDrift(trend_shape_dict, noise_shape_dict, noise_plate_dict, **ts_dict)
model_fn = totalstation_drift.forward
input_vars_concrete = time.repeat([5,1])

# illustrate
fig = totalstation_drift.render(input_vars_concrete)
# call fig to plot inline


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

class TotalstationDriftProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.totalstation_drift = totalstation_drift
        self.output_data = data
        self.input_data = input_vars_concrete
        
    def model(self, input_vars, observations = None):
        output = self.totalstation_drift.forward(input_vars, observations = observations)
        return output
    def guide(self, input_vars, observations = None):
        output = guide_fn(input_vars, observations = observations)
        return output
    
drift_probmodel = TotalstationDriftProbModel(name = 'concrete_probmodel')
    
# run and log optimization
input_data = input_vars_concrete
output_data = data
optim_results = drift_probmodel.train(input_data, output_data, optim_opts)


    
    

"""
    6. Analyse results and illustrate
"""

# convergence
plt.plot(optim_results)

# model params
alpha_est = pyro.get_param_store()['coeffs'].detach().numpy()
sigma_est = pyro.get_param_store()['variance'].detach().numpy()
print('coeffs true:', alpha_true)
print('coeffs estimated:', alpha_est)
print('sigma true:', sigma_true)
print('sigma estimated:', sigma_est)


# data vs predictions
data_simulated = drift_probmodel.model(input_data).detach().numpy()
fig, ax = plt.subplots(1,2, figsize = (10,5))
ax[0].plot(data.T)
ax[0].set_title('true_data')

ax[1].plot(data_simulated.T)
ax[1].set_title('simulated_data')



























