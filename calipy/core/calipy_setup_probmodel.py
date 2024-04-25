#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the CalipyProbModel base class that is useful for representing,
modifying, analyzing, and optimizing instrument models.

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

import numpy as np

from calipy_setup_components import  ComponentCollector
from calipy_setup_variables import VariableCollector
from calipy_setup_constraints import ConstraintCollector
from calipy_setup_optimizer import OptimizationCollector
from calipy_setup_illustrations import IllustrationCollector
from calipy_setup_utils import SupportCollector
import calipy_setup_utils as utils


# Probmodel class determines attributes and methods for summarizing and accessing
# instruments, components, effects, and data of the whole probabilistic model.

class CalipyProbModel(pyro.nn.PyroModule):

    # i) Initialization
    
    def __init__(self, design_params, data_info):
        super().__init__()

        
        # Create plant components
        self.support = SupportCollector(self)
        data_info = sf.CompletedDataInfo(self, data_info,design_params)
        self.design_params = design_params
        self.data_info = data_info
        self.components = PlantComponentCollector(self)
        self.variables = VariableCollector(self)
        self.constraints = ConstraintCollector(self)
        self.dynamics = DynamicsCollector(self)
        self.optimizer = OptimizationCollector(self)
        self.control = ControlCollector(self )
        self.illustrations = IllustrationCollector(self )
        self.iterator = IterationCollector(self)
        self.dict_all_tags = dict()
        info_string = setup_info_string(self,context = "calipy_probabilistic_model")
        self.info = 'Collector object containing all info pertaining to the calipy_probmodel' + info_string
        
    
