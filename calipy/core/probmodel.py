#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the CalipyProbModel base class that is useful for representing,
modifying, analyzing, and optimizing instrument models based on observed data.

The classes are
    CalipyProbModel: Short for Calipy Probabilistic Model. Base class providing
    functionality for integrating instruments, effects, and data into one 
    CalipyProbModel object. Allows simulation, inference, and illustration of 
    deep instrument models.
  
The CalipyProbModel class provides a comprehensive representation of the interactions
between instruments and data. It contains several subobjects representing the
physical instrument, random and systematic effects originating from instrument
or environment, unknown parameters and variables, constraints, and the objective
function. All of these subobjects form a probabilistic model that can be sampled
and conditioned on measured data. For more information, see the separate
documentation entries the CalipyProbModel class, for the subobjects, or the tutorial.        
        

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    CalipyProbModel class ----------------------------------------------------
"""


import pyro
from calipy.core.utils import CalipyRegistry



# Probmodel class determines attributes and methods for summarizing and accessing
# instruments, components, effects, and data of the whole probabilistic model.

class CalipyProbModel():

    # i) Initialization
    
    def __init__(self, model_type = None, model_name = None, info_dict = {}):
        self.dtype = 'CalipyProbModel'
        self.type = model_type
        self.name = model_name
        self.info_dict = info_dict
        self.instrument_registry = CalipyRegistry()
        self.effects_registry = CalipyRegistry()
        self.quantity_registry = CalipyRegistry()

        self.id = "{}_{}".format(self.type, self.name)


# i) EmptyProbModel class: Catchall class for instruments unassociated to any specific 
# probmodel
type_EmptyProbModel = 'empty_probmodel'
name_EmptyProbModel = 'base'
info_dict_EmptyProbModel = {}

class EmptyProbModel(CalipyProbModel):
        
    def __init__(self, model_name):
        super().__init__(model_type = type_EmptyProbModel, 
                         model_name = model_name, 
                         info_dict = info_dict_EmptyProbModel)
        

empty_probmodel = EmptyProbModel(name_EmptyProbModel)
