#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the CalipyEffect base class that is used for specifying
random and deterministic phenomena affecting measurements and provides a list
of basic effects as well as the functionality needed to integrate them into
the CalipyProbModel class for simulation and inference.

The classes are
    CalipyEffect: Base class from which all concrete effects inherit. Blueprint
        for effects that involve known parameters, unknown parameters, and random
        variables. Provides effect in form of differentiable forward map.
    
    CalipyQuantity: Base class from which all concrete quantities inherit. Blueprint
        for quantities, i.e. known parameters, unknown parameters, and random
        variables that are the building blocks for CalipyEffect objects.
  

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    CalipyEffect class ----------------------------------------------------
"""


# i) Imports

import pyro
import torch
from calipy.core.base import CalipyNode
from calipy.core.utils import multi_unsqueeze, context_plate_stack
from calipy.core.base import NodeStructure
from pyro.distributions import constraints
from abc import ABC, abstractmethod


# ii) Definitions


class CalipyEffect(CalipyNode):
    """
    The CalipyEffect class provides a comprehensive representation of a specific 
    effect. It is named, explained, and referenced in the effect description. The
    effect is incorporated as a differentiable function based on torch. This function
    can depend on known parameters, unknown parameters, and random variables. Known 
    parameters have to be provided during invocation of the effect. During training,
    unknown parameters and the posterior density of the random variables is inferred.
    This requires providing a unique name, a prior distribution, and a variational
    distribution for the random variables.
    """
    
    
    def __init__(self, type = None, name = None, info = None):
        
        # Basic infos
        super().__init__(node_type = type, node_name = name, info_dict = info)
        
        
        self._effect_model = None
        self._effect_guide = None
        

    

"""
    CalipyQuantity class ----------------------------------------------------
"""



class CalipyQuantity(CalipyNode):
    """
    The CalipyQuantity class provides a comprehensive representation of a specific 
    quantity used in the construction of a CalipyEffect object. This could be a
    known parameter, an unknown parameter, or a random variable. This quantity
    is named, explained, and referenced in the quantity description. Quantities
    are incorporated into the differentiable function that define the CalipyEffect
    forward pass. Each quantity is subservient to an effect and gets a unique id
    that reflects this, quantities are local and cannot be shared between effects.
    """
    
    def __init__(self, type = None, name = None, info = None):
        
        # Basic infos
        super().__init__(node_type = type, node_name = name, info_dict = info)
        

    



"""
    Classes of quantities
"""  


# i) Deterministic known parameter

class KnownParam(CalipyQuantity):
    pass


# ii) Deterministic unknown parameter

# Define a class of quantities that consists in unknown deterministic parameters
# that serve as model parameters to be inferred. This means that invocation of 
# the class produces objects that are interpretable as scalars, vectors, or matrices
# of constant values. These objects dont necessarily need to contain the same 
# constant in each of the dimensions. 
# The node_structure passed upon instantiation determines batch_shape (indicating 
# independent parameter values) and event_shape (indicating identical parameter
# values). E.g batch_shape = 5, event_shape = 20 produces a (5,20) tensor in which
# 5 independent constant values are repeated 20 times. This tensor can be used to
# e.g. construct a tensor mu to be passed into a distribution that represents 5
# different mean values that have been sampled 20 times each. 

class UnknownParam(CalipyQuantity):
    """ UnknownParam is a subclass of CalipyQuantity that produces an object whose
    forward() method produces a parameter that is subject to inference.

    :param node_structure: Instance of NodeStructure that determines the internal
        structure (shapes, plate_stacks, plates, aux_data) completely.
    :type node_structure: NodeStructure
    :return: Instance of the UnknownParam class built on the basis of node_structure
    :rtype: UnknownParam (subclass of CalipyQuantity subclass of CalipyNode)
    
    Example usage: Run line by line to investigate Class
        
    .. code-block:: python
    
        # Investigate 2D bias tensor -------------------------------------------
        #
        # i) Imports and definitions
        import calipy
        from calipy.core.effects import UnknownParam
        node_structure = UnknownParam.example_node_structure
        bias_object = UnknownParam(node_structure, name = 'tutorial')
        #
        # ii) Produce bias value
        bias = bias_object.forward()
        #
        # iii) Investigate object
        bias_object.dtype_chain
        bias_object.id
        bias_object.node_structure.description
        render_1 = bias_object.render()
        render_1
        render_2 = bias_object.render_comp_graph()
        render_2
    """
    
    
    # Initialize the class-level NodeStructure
    example_node_structure = NodeStructure()
    example_node_structure.set_shape('batch_shape', (10, ), 'Batch shape description')
    example_node_structure.set_shape('event_shape', (5, ), 'Event shape description')

    # Class initialization consists in passing args and building shapes
    def __init__(self, node_structure, constraint = constraints.real, **kwargs):  
        super().__init__(**kwargs)
        self.node_structure = node_structure
        self.batch_shape = self.node_structure.shapes['batch_shape']
        self.event_shape = self.node_structure.shapes['event_shape']
        self.extension_tensor = multi_unsqueeze(torch.ones(self.event_shape), 
                                                dims = [0 for dim in self.batch_shape])
    
    # Forward pass is initializing and passing parameter
    def forward(self, input_vars = None, observations = None):
        self.param = pyro.param('{}__param_{}'.format(self.id_short, self.name), init_tensor = multi_unsqueeze(torch.ones(self.batch_shape), 
                                            dims = [len(self.batch_shape) for dim in self.event_shape]))
        self.extended_param = self.extension_tensor * self.param
        return self.extended_param


# ii) - a) Subclass of UnknownParam for variances (featuring positivity constraint)

class UnknownVar(UnknownParam):
    def __init__(self, node_structure, **kwargs):  
        super().__init__(node_structure, constraint = constraints.positive, **kwargs)



# iii) Random variable

# class RandomVar(CalipyQuantity):
    
#     # Initialize the class-level NodeStructure
#     example_node_structure = NodeStructure()
#     example_node_structure.set_shape('batch_shape', (10, ), 'Batch shape description')
#     example_node_structure.set_shape('event_shape', (5, ), 'Event shape description')

    
#     def __init__(self, node_structure, distribution, distribution_args, **kwargs):  
#         super().__init__(**kwargs)
#         self.node_structure = node_structure
#         self.distribution = distribution
        
#         self.batch_shape = self.node_structure.shapes['batch_shape']
#         self.event_shape = self.node_structure.shapes['event_shape']
        
#     def forward(self, input_vars = None, observations = None):
#         self.distribution_args = distribution_args
        
#         return output


# iv) Gaussian process



# v) Neural net




"""
    Classes of simple effects
"""



# ii) Addition of noise

# Define a class of effects that consist in a noise distribution that can be 
# sampled to serve as observations used for inference. This means that invocation of 
# the class produces objects that are interpretable as noisy observations of scalars,
# vectors, or matrices. Neither the mean nor the standard deviation for the noise
# distribution need to be constant.
# The node_structure passed upon instantiation determines the shape of the noise
# being added to the mean. E.g batch_plate_1 with size 5, batch_plate_2 with 
# size 20 produces (5,20) tensors of independent noise when using the function
# call forward(input_vars = (mean, standard_deviation)).


class NoiseAddition(CalipyEffect):
    """ NoiseAddition is a subclass of CalipyEffect that produces an object whose
    forward() method emulates uncorrelated noise being added to an input. 

    :param node_structure: Instance of NodeStructure that determines the internal
        structure (shapes, plate_stacks, plates, aux_data) completely.
    :type node_structure: NodeStructure
    :return: Instance of the NoiseAddition class built on the basis of node_structure
    :rtype: NoiseAddition (subclass of CalipyEffect subclass of CalipyNode)
    
    Example usage: Run line by line to investigate Class
        
    .. code-block:: python
    
        # Investigate 2D noise ------------------------------------------------
        #
        # i) Imports and definitions
        import calipy
        from calipy.core.effects import NoiseAddition
        node_structure = NoiseAddition.example_node_structure
        noisy_meas_object = NoiseAddition(node_structure, name = 'tutorial')
        #
        # ii) Sample noise
        mean = torch.zeros([10,5])
        std = torch.ones([10,5])
        noisy_meas = noisy_meas_object.forward(input_vars = (mean, std))
        #
        # iii) Investigate object
        noisy_meas_object.dtype_chain
        noisy_meas_object.id
        noisy_meas_object.noise_dist
        noisy_meas_object.node_structure.description
        noisy_meas_object.plate_stack
        render_1 = noisy_meas_object.render((mean, std))
        render_1
        render_2 = noisy_meas_object.render_comp_graph((mean, std))
        render_2
    """
    
    
    # Initialize the class-level NodeStructure
    example_node_structure = NodeStructure()
    example_node_structure.set_plate_stack('noise_stack', [('batch_plate_1', 5, -2, 'plate denoting independence in row dim'),
                                                              ('batch_plate_2', 10, -1, 'plate denoting independence in col dim')],
                                           'Plate stack for noise ')

    def __init__(self, node_structure, **kwargs):
        super().__init__(**kwargs)
        self.node_structure = node_structure
        # self.batch_shape = node_structure.shapes['batch_shape']
        # self.event_shape = node_structure.shapes['event_shape']
        self.plate_stack = self.node_structure.plate_stacks['noise_stack']
    def forward(self, input_vars, observations = None):
        """
        Create noisy samples using input_vars = (mean, standard_deviation) with
        shapes as indicated in the node_structures plate_stack 'noise_stack' used
        for noisy_meas_object = NoiseAddition(node_structure).
        
        :param input vars: 2-tuple (mean, standard_deviation) of tensors with 
            equal (or at least broadcastable) shapes. 
        :type input_vars: 2-tuple of instances of torch.Tensor
        :return: Tensor representing simulation of a noisy measurement of the mean.
        :rtype: torch.Tensor
        """
        # self.noise_dist = pyro.distributions.Normal(loc = input_vars.mean, scale = input_vars.standard_deviation)
        self.noise_dist = pyro.distributions.Normal(loc = input_vars[0], scale = input_vars[1])
        
        # Sample within independence context
        with context_plate_stack(self.plate_stack):
            output = pyro.sample('{}_obs_{}'.format(self.id_short, self.name), self.noise_dist, obs = observations)
        return output
    
    

# i) OffsetDeterministic class 
# Define a class of errors that transform the input by adding deterministic 
# offsets. This means that invocation of the class produces objects that are 
# interpretable as scalars, vectors, or matrices of constant values. These offsets
# dont necessarily need to be constant in each dimension. E.g. 5 with different
# series of 20 measurements for each of which the offset remains constant, the
# corresponding OffsetDeterministic object would be a 5 x 20 tensor containing 5
# different constants repeated 20 times per row.
















