#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the CalipyInstrument base class that is useful for representing,
modifying, analyzing, and optimizing instrument models.

The classes are
    CalipyInstrument: Base class providing functionality for defining instruments
        and the data they produce. This is done by chaining effects to produce
        a forward model. CalipyInstrument objects can be used for simulation and
        inference and are the building blocks for CalipyProbModel objects.   
        

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    CalipyInstrument class ----------------------------------------------------
"""


# i) Imports

import pyro
from abc import ABC, abstractmethod
from calipy.core.utils import CalipyRegistry
from calipy.core.base import CalipyNode


class CalipyInstrument(CalipyNode):
    """
    The CalipyInstrument class provides a comprehensive representation of the 
    instruments and the effects occuring during measurement with the instrument.
    It contains several objects of type CalipyEffect (themselves containing objects
    of type CalipyQuantity) whose effect.apply_effect() methods are chained together
    in the forward methods to simulate the data generation process of the instrument.
    This is used for simulation and inference. For more information, see the separate
    documentation entries the CalipyProbModel class, for the subobjects, or the tutorial.   
    """
        
    
    def __init__(self, type = None, name = None, info = None):
        
        # Basic infos
        super().__init__(node_type = type, node_name = name, info_dict = info)
        
        
        

    


"""
    Instrument classes
"""


# # i) EmptyInstrument class: Catchall class for effects unassociated to any specific 
# # instrument
# type_EmptyInstrument = 'empty_instrument'
# name_EmptyInstrument = 'base'
# info_dict_EmptyInstrument = {}

# class EmptyInstrument(CalipyInstrument):
        
#     def __init__(self, instrument_name):
#         super().__init__(instrument_type = type_EmptyInstrument, 
#                          instrument_name = instrument_name, 
#                          info_dict = info_dict_EmptyInstrument)
        
#     def forward(self):
#         pass

# empty_instrument = EmptyInstrument(name_EmptyInstrument)
