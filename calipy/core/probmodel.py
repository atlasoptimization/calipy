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
    CalipyDAG construction classes ----------------------------------------------
"""


import pyro
from calipy.core.utils import CalipyRegistry
from calipy.core.dag import CalipyNode, CalipyEdge, CalipyDAG
from abc import ABC, abstractmethod





"""
    CalipyProbModel class ----------------------------------------------------
"""


# Probmodel class determines attributes and methods for summarizing and accessing
# data, instruments, effects, and quantities of the whole probabilistic model.

class CalipyProbModel():

    # i) Initialization
    
    def __init__(self, model_type = None, model_name = None, info_dict = {}):
        self.dtype = self.__class__.__name__
        self.type = model_type
        self.name = model_name
        self.info_dict = info_dict
        self.model_dag = CalipyDAG('Model_DAG')
        self.guide_dag = CalipyDAG('Guide_DAG')
    


        self.id = "{}_{}".format(self.type, self.name)
    def __repr__(self):
        return "{}(type: {} name: {})".format(self.dtype, self.type,  self.name)

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






dag = CalipyDAG('daggie')
node1 = CalipyNode(node_type="Type1", node_name="Node1")
node2 = CalipyNode(node_type="Type2", node_name="Node2")
dag.add_node(node1)
dag.add_node(node2)
dag.add_edge("Node1", "Node2")
dag.display()


