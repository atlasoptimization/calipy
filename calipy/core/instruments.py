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


class CalipyInstrument(ABC):
    """
    The CalipyProbModel class provides a comprehensive representation of the interactions
    between instruments and data. It contains several subobjects representing the
    physical instrument, random and systematic effects originating from instrument
    or environment, unknown parameters and variables, constraints, and the objective
    function. All of these subobjects form a probabilistic model that can be sampled
    and conditioned on measured data. For more information, see the separate
    documentation entries the CalipyProbModel class, for the subobjects, or the tutorial.   
    """
    
    _instrument_counters = {}
    
    def __init__(self, name, info_dict):
        self.name = name
        self.info_dict = info_dict
        
        # Upon instantiation either create or increment _instrument_counters dict
        if name not in CalipyInstrument._instrument_counters:
            CalipyInstrument._instrument_counters[name] = 0
        else:
            CalipyInstrument._instrument_counters[name] += 1

        # Create a unique identifier based on the name and the current count
        self.id = "{}_{}".format(name, CalipyInstrument._instrument_counters[name])        
        
    @abstractmethod
    def forward(self, input_data):
        # Subclasses must implement this method to create the specific instrument model.
        pass
        




