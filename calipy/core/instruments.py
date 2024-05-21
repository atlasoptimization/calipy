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
from calipy.core.probmodel import empty_probmodel


class CalipyInstrument(ABC):
    """
    The CalipyInstrument class provides a comprehensive representation of the 
    instruments and the effects occuring during measurement with the instrument.
    It contains several objects of type CalipyEffect (themselves containing objects
    of type CalipyQuantity) whose effect.apply_effect() methods are chained together
    in the forward methods to simulate the data generation process of the instrument.
    This is used for simulation and inference. For more information, see the separate
    documentation entries the CalipyProbModel class, for the subobjects, or the tutorial.   
    """
        
    def __init__(self, instrument_type = None, instrument_name = None, info_dict = {}, **kwargs):
        if instrument_name is None:
            raise ValueError(f"{self.__class__.__name__} requires a 'instrument_name' argument")

        # basic data integration
        self.dtype = 'CalipyInstrument'
        self.type = instrument_type
        self.name = instrument_name
        self.info_dict = info_dict
        
        # assign to superior instance
        self.superior_instance = kwargs.get('superior_instance', empty_probmodel)
        self.superior_instance_dtype = self.superior_instance.dtype
        self.superior_instance_type = self.superior_instance.type
        self.superior_instance_name = self.superior_instance.name
        self.superior_instance_id = self.superior_instance.id
        self.id = "{}_{}_{}".format(self.super_instance_id, self.type, self.name)
        print("Initialized type: {} name: {} in superior instance type: {} name: {}"
              .format(self.type, self.name, 
                      self.superior_instance_type, self.superior_instance.name))
        
        # register data to probmodel
        # CalipyRegistry.register(self.id, self)
        
        # if self.name in CalipyInstrument._registry:
        #     print(f"Warning: An instrument with the name '{self.name}' already exists.")

    @abstractmethod
    def forward(self, input_data):
        pass
    


"""
    Instrument classes
"""


# i) EmptyInstrument class: Catchall class for effects unassociated to any specific 
# instrument
type_EmptyInstrument = 'empty_instrument'
name_EmptyInstrument = 'base'
info_dict_EmptyInstrument = {}

class EmptyInstrument(CalipyInstrument):
        
    def __init__(self, instrument_name):
        super().__init__(instrument_type = type_EmptyInstrument, 
                         instrument_name = instrument_name, 
                         info_dict = info_dict_EmptyInstrument)
        
    def forward(self):
        pass

empty_instrument = EmptyInstrument(name_EmptyInstrument)
