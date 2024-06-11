#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to model two tape measures with the
same variance but different means. The two tape measures have been used to collect 
two different datasets, both featuring a different amount of samples.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Build the effect models
    4. Build the probmodel
    5. Perform inference
    6. Analyse results and illustrate
In this example we will have 2 different tape measures and n_meas_1 and n_meas_2
measurements for the respective tape measures. The bias mu of the measurements 
is supposed to be an unknown constant that is different for each tape measure.
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

n_tape = 2
n_meas_1 = 20
n_meas_2 = 50



"""
    2. Simulate some data
"""


# i) Set up sample distributions

length_true = torch.tensor(1.0)

coin_flip = pyro.distributions.Bernoulli(probs = torch.tensor(0.5)).sample([n_tape])
bias_true = coin_flip * torch.tensor(0.01)      # measurement bias is either 0 cm or 1 cm
sigma_true = torch.tensor(0.01)                 # measurement standard deviation is 1 cm


# ii) Sample from distributions

data_1_distribution = pyro.distributions.Normal(length_true + bias_true[0], sigma_true)
data_2_distribution = pyro.distributions.Normal(length_true + bias_true[1], sigma_true)
data_1 = data_1_distribution.sample([n_meas_1])
data_2 = data_2_distribution.sample([n_meas_2])
data = (data_1, data_2)

# The data now is a 2-tuple with shapes [1, n_meas_1] and [1, n_meas_2] and 
# reflects 2 tape measures each with a bias of either 0 cm or 1 cm being used 
# to measure an object of length 1 m.

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
        self.offset = pyro.param('{}__offset'.format(self.id), init_tensor = multi_unsqueeze(torch.zeros(self.batch_shape), 
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
    example_node_structure.set_plate_stack('noise_stack', [('batch_plate_1', 10, -2, 'plate denoting independence in row dim'),
                                                              ('batch_plate_2', 5, -1, 'plate denoting independence in col dim')],
                                           'Plate stack for noise ')
    # input_vars = namedtuple('InputNoiseAddition', ['mean', 'standard_deviation'])

    def __init__(self, node_structure, **kwargs):
        super().__init__(**kwargs)
        self.node_structure = node_structure
        # self.batch_shape = node_structure.shapes['batch_shape']
        # self.event_shape = node_structure.shapes['event_shape']
        self.plate_stack = self.node_structure.plate_stacks['noise_stack']
    def forward(self, input_vars, observations = None):
        """
        Input input_vars is named tuple of type InputNoiseAddition with fields mean
        and standard_deviation.
        """
        # self.noise_dist = pyro.distributions.Normal(loc = input_vars.mean, scale = input_vars.standard_deviation)
        self.noise_dist = pyro.distributions.Normal(loc = input_vars[0], scale = input_vars[1])
        
        # Sample within independence context
        with context_plate_stack(self.plate_stack):
            output = pyro.sample('{}_obs'.format(self.id), self.noise_dist, obs = observations)
        return output
    
    
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


# iii) Shape extension

# class ShapeExtension(CalipyEffect):
    
#     # Initialize class-level NodeStructure
#     example_node_structure = NodeStructure()
#     example_node_structure.set_shape('extension_shape', (10, 5), 'Shape of repetitions')
#     example_node_structure.set_shape('extended_shape', (-1, 10, 5), 'Shape after repetitions')
#     example_input_var = (torch.ones([3]),)
    
#     def __init__(self, node_structure, **kwargs):
#         super().__init__(**kwargs)
#         self.node_structure = node_structure

#     def forward(self, input_vars, observations = (None,)):
#         """
#         Input input_vars is tuple of (tensor_for repetition,).
#         """
        
#         input_dims = input_vars[0].shape
        
#         output = input_vars[0].expand()
        
        
#         return output



"""
    4. Build the instrument model
"""

# i) Set up actual dimensions

basic_node_structure_offset = DeterministicOffset.example_node_structure
shape_updates_offset =  {'batch_shape' : (),
                         'event_shape' : ()}
plate_stack_updates_offset = {}
node_structure_offset = DeterministicOffset.build_node_structure(basic_node_structure_offset, shape_updates_offset, plate_stack_updates_offset)

offset_name = 'offset_due_to_miscalibration'
offset_type = 'tape_measure_offset'
offset_info = {'description': 'Is a deterministic offset. Can be used for many things'}
offset_dict = {'name' : offset_name, 'type': offset_type, 'info': offset_info}



node_structure_noise_1 = NodeStructure()
node_structure_noise_1.set_plate_stack('noise_stack', [('batch_plate', n_meas_1, -1, 'independent noise tape 1')], 'Plate stack description')


node_structure_noise_2 = NodeStructure()
node_structure_noise_2.set_plate_stack('noise_stack', [('batch_plate', n_meas_2, -1, 'independent noise tape 2')], 'Plate stack description')


# shape_updates_noise_2 =  {'batch_shape' : (n_meas_2,),
#                          'event_shape' : (0,)}
# plate_stack_updates_noise_2 = {'2d_noise_stack_2' : [('batch_plate_2', n_meas_2, -1, 'independent noise tape 2')]}
# node_structure_noise_2 = NoiseAddition.build_node_structure(basic_node_structure_noise, shape_updates_noise_2, plate_stack_updates_noise_2)


noise_name = 'noise_due_to_optical_effects'
noise_type = 'tape_measure_noise'
noise_info = {'description': 'Is a bunch of noise. Can be used for many things'}
noise_dict = {'name' : noise_name, 'type': noise_type, 'info': noise_info}





# iv) Guides and partialization

def guide_fn(observations = None):
    pass


"""
    5. Build probmodel & perform inference
"""




# set up optimization
adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
n_steps = 1000

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps}

class TwoObservationsProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.mu_1_model = DeterministicOffset(node_structure_offset, **offset_dict)
        self.mu_2_model = DeterministicOffset(node_structure_offset, **offset_dict)
        self.noise_1_model = NoiseAddition(node_structure_noise_1, **noise_dict)
        self.noise_2_model = NoiseAddition(node_structure_noise_2, **noise_dict)        
        
    def model(self, input_vars = None, observations = (None, None)):
        mu_1 = self.mu_1_model.forward()
        mu_2 = self.mu_2_model.forward()
        
        output_1 = self.noise_1_model.forward((mu_1, sigma_true), observations = observations[0])
        output_2 = self.noise_2_model.forward((mu_2, sigma_true), observations = observations[1])
        
        output = (output_1, output_2)
        
        return output
    def guide(self, input_vars = None, observations = (None, None)):
        output = guide_fn(observations = observations)
        return output
    
two_observations_probmodel = TwoObservationsProbModel(name = 'concrete_probmodel')
    
# run and log optimization
input_data = None
output_data = data
optim_results = two_observations_probmodel.train(input_data, output_data, optim_opts)


    
    

"""
    6. Analyse results and illustrate
"""

plt.plot(optim_results)

for param, value in pyro.get_param_store().items():
    print(param, '\n', value)

print(bias_true)
print(torch.mean(data[0]), torch.mean(data[1]))






















