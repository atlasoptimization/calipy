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
from calipy.core.base import CalipyNode
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
        
    

    
    # # Abstract methods for model and guide that subclasses need to provide
    # @abstractmethod
    # def create_effect_model(self):
    #     # Subclasses must implement this method to create the specific effect model.
    #     pass
    
    # @abstractmethod
    # def create_effect_guide(self):
    #     # Subclasses must implement this method to create the specific effect guide.
    #     pass
    

    # # Fetch functions by lazy initialization from subclass definitions
    # def get_effect_model(self):
    #     # Lazy initialization for the effect model.
    #     if self._effect_model is None:
    #         self._effect_model = self.create_effect_model()
    #     return self._effect_model

    # def get_effect_guide(self):
    #     # Lazy initialization for the effect guide.
    #     if self._effect_guide is None:
    #         self._effect_guide = self.create_effect_guide()
    #     return self._effect_guide
    
    
    # # Apply effect and call guide    
    # def apply_effect(self, input_vars, data):
    #     # Apply the effect to the data. This method uses the effect model.
    #     model = self.get_effect_model()
    #     return model(input_vars, data)
    
    # def call_guide(self, input_vars, data):
    #     # Call the guide to sample from variational distribution
    #     guide = self.get_effect_guide()
    #     return guide(input_vars, data)
    
    # def __repr__(self):
    #     return f"{self.__class__.__name__}(name={self.name})"


    

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
    
    _quantity_counters = {}
    
    def __init__(self, type = None, name = None, info = None):
        
        # Basic infos
        super().__init__(node_type = type, node_name = name, info_dict = info)
        

        

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"



"""
    Classes of quantities
"""  


# i) Deterministic known parameter

class KnownParam(CalipyQuantity):
    pass


# ii) Deterministic unknown parameter

class UnknownParam(CalipyQuantity):
    pass


# iii) Random variable

class RandomVar(CalipyQuantity):
    pass


# iv) Gaussian process



# v) Neural net




"""
    Classes of simple effects
"""



# i) OffsetDeterministic class 
# Define a class of errors that transform the input by adding deterministic 
# offsets. This means that invocation of the class produces objects that are 
# interpretable as scalars, vectors, or matrices of constant values. These offsets
# dont necessarily need to be constant in each dimension. E.g. 5 with different
# series of 20 measurements for each of which the offset remains constant, the
# corresponding OffsetDeterministic object would be a 5 x 20 tensor containing 5
# different constants repeated 20 times per row.

# Name and Info
OffsetDeterministic_name = 'OffsetDeterministic'
OffsetDeterministic_info =  'Class of errors that transform the input by adding deterministic '\
    'but unknown offsets. Offsets can be different for different dimensions.'
    
# # Effect model
# OffsetDeterministic_model = 

# # Effect guide
# OffsetDeterministic_guide =

# # Effect details
# OffsetDeterministic_details = {}

# class OffsetDeterministic(CalipyEffect):
#     def __init__(self, offset_initialization):
#         super().__init__(OffsetDeterministic_name, OffsetDeterministic_info)
#         self.offset_initialization = offset_initialization
        

#     def create_effect_model(self):
#         # Creates a simple deterministic model applying an offset.

#         offset = pyro.param(self.id, self.offset_initialization)
        
#         def apply_offset(data):
#             data + offset
        
#         return apply_offset 
    
#     def create_effect_guide(self):
#         # Creates an empty guide
#         pass


#     def __repr__(self):
#         return "{}".format(self.id)
















