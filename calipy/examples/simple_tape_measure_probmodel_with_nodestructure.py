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
from collections import namedtuple

# calipy
import calipy
from calipy.core.base import CalipyProbModel
from calipy.core.instruments import CalipyInstrument
from calipy.core.effects import CalipyEffect, CalipyQuantity
from calipy.core.utils import multi_unsqueeze, context_plate_stack
from calipy.core.base import NodeStructure


# ii) Definitions

n_tape = 5
n_meas = 100



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
    
    # Initialize the class-level NodeStructure
    example_node_structure = NodeStructure()
    example_node_structure.set_shape('batch_shape', (10, ), 'Batch shape description')
    example_node_structure.set_shape('event_shape', (5, ), 'Event shape description')

    
    def __init__(self, node_structure, **kwargs):  
        super().__init__(**kwargs)
        self.node_structure = node_structure
        self.batch_shape = self.node_structure.shapes['batch_shape']
        self.event_shape = self.node_structure.shapes['event_shape']
        self.extension_tensor = multi_unsqueeze(torch.ones(self.event_shape), 
                                                dims = [0 for dim in self.batch_shape])
        
    def forward(self, input_vars = None, observations = None):
        self.offset = pyro.param('offset', init_tensor = multi_unsqueeze(torch.zeros(self.batch_shape), 
                                            dims = [len(self.batch_shape) for dim in self.event_shape]))
        output = self.extension_tensor * self.offset
        return output
    
    
# Some cool things i can do now:
# ns = DeterministicOffset.example_node_structure
# ns.generate_template()
# det = DeterministicOffset(ns)
# DeterministicOffset.check_node_structure(ns)
# shape_updates = {'batch_shape' : (33,)}
# DeterministicOffset.build_node_structure(ns, shape_updates, {})
# det.render()
# det.render_com_graph()


# ii) Addition of noise

class NoiseAddition(CalipyEffect):
    
    # Initialize the class-level NodeStructure
    example_node_structure = NodeStructure()
    # example_node_structure.set_shape('batch_shape', (10, 5), 'Batch shape description')
    # example_node_structure.set_shape('event_shape', (0, ), 'Event shape description')
    example_node_structure.set_plate_stack('2d_noise_stack', [('batch_plate_1', 10, -2, 'plate denoting independence in row dim'),
                                                              ('batch_plate_2', 5, -1, 'plate denoting independence in col dim')],
                                           'Plate stack for noise in 2 independent dims')
    input_vars = namedtuple('InputNoiseAddition', ['mean', 'standard_deviation'])

    def __init__(self, node_structure, **kwargs):
        super().__init__(**kwargs)
        self.node_structure = node_structure
        # self.batch_shape = node_structure.shapes['batch_shape']
        # self.event_shape = node_structure.shapes['event_shape']
        self.plate_stack = self.node_structure.plate_stacks['2d_noise_stack']
    def forward(self, input_vars, observations = None):
        """
        Input input_vars is named tuple of type InputNoiseAddition with fields mean
        and standard_deviation.
        """
        self.noise_dist = pyro.distributions.Normal(loc = input_vars.mean, scale = input_vars.standard_deviation)
        
        # Sample within independence context
        with context_plate_stack(self.plate_stack):
            obs = pyro.sample('obs', self.noise_dist, obs = observations)
        return obs
    
    
# ns = NoiseAddition.example_node_structure
# ns.shapes
# ns.plate_stacks
# ns.plates
# ns.description
# noise = NoiseAddition(ns)
# shape_updates = {'batch_shape' : (33,)}
# plate_stack_updates = {'2d_noise_stack' : [('batch_plate_1', 33, -1, 'new shape now')]}
# new_ns = ns.update(shape_updates, plate_stack_updates)
# other_new_ns = NoiseAddition.build_node_structure(ns, shape_updates, plate_stack_updates)
# other_new_ns.print_shapes_and_plates()







"""
    4. Build the instrument model
"""

# i) Set up actual dimensions

basic_node_structure_offset = DeterministicOffset.example_node_structure
shape_updates_offset =  {'batch_shape' : (n_tape,),
                         'event_shape' : (n_meas,)}
plate_stack_updates_offset = {}
node_structure_offset = DeterministicOffset.build_node_structure(basic_node_structure_offset, shape_updates_offset, plate_stack_updates_offset)

offset_name = 'offset_due_to_miscalibration'
offset_type = 'tape_measure_offset'
offset_info = {'description': 'Is a deterministic offset. Can be used for many things'}
offset_dict = {'name' : offset_name, 'type': offset_type, 'info': offset_info}

basic_node_structure_noise = NoiseAddition.example_node_structure
shape_updates_noise =  {'batch_shape' : (n_tape,n_meas),
                         'event_shape' : (0,)}
plate_stack_updates_noise = {'2d_noise_stack' : [('batch_plate_1', n_tape, -2, 'independent noise different tapes'),
                                                 ('batch_plate_2', n_meas, -1, 'independent noise different measurements')]}
node_structure_noise = NoiseAddition.build_node_structure(basic_node_structure_noise, shape_updates_noise, plate_stack_updates_noise)

noise_name = 'noise_due_to_optical_effects'
noise_type = 'tape_measure_noise'
noise_info = {'description': 'Is a bunch of noise. Can be used for many things'}
noise_dict = {'name' : noise_name, 'type': noise_type, 'info': noise_info}


# i) Create TapeMeasure class

tm_type = 'tape_measure'
tm_name = 'tape_measure_nr_17'
tm_info = {'description': 'That weird one with the abrasions'}
tm_dict = {'name': tm_name, 'type': tm_type, 'info' : tm_info}


class TapeMeasure(CalipyInstrument):
        
    def __init__(self, node_structure, **kwargs):
        super().__init__(**kwargs)
        
        
        self.offset_model = DeterministicOffset(node_structure_offset, **offset_dict)
        self.noise_model = NoiseAddition(node_structure_noise, **noise_dict)
        
    def forward(self, input_vars = None, observations = None):
        mean = self.offset_model.forward()
        ip = NoiseAddition.input_vars(mean, sigma_true)
        obs = self.noise_model.forward(ip, observations = observations)
        return obs


# Invoke tape_measure
tape_measure = TapeMeasure({}, **tm_dict)
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
n_steps = 1000

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps}

class TapeMeasureProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.tape_measure = tape_measure
        # self.output_data = data
        # self.input_data = None
        
    def model(self, input_vars = None, observations = None):
        output = tape_measure.forward(observations = observations)
        return output
    def guide(self, input_vars = None, observations = None):
        output = guide_fn(observations = observations)
        return output
    
tape_measure_probmodel = TapeMeasureProbModel(name = 'concrete_probmodel')
    
# run and log optimization
input_data = None
output_data = data
optim_results = tape_measure_probmodel.train(input_data, output_data, optim_opts)


    
    

"""
    6. Analyse results and illustrate
"""

plt.plot(optim_results)




























