#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the CalipyProbModel base class that is useful for representing,
modifying, analyzing, and optimizing instrument models based on observed data.
Furthermore, this module defines the CalipyDAG, CalipyNode, and CalipyEdge base
classes that enable construction of the durected acyclic graph used for model 
and guide representation.

The classes are
    CalipyProbModel: Short for Calipy Probabilistic Model. Base class providing
    functionality for integrating instruments, effects, and data into one 
    CalipyProbModel object. Allows simulation, inference, and illustration of 
    deep instrument models.
    
    CalipyDAG: Class representing the directed acyclic graph underlying the model
    or the guide. Contains nodes and edges together with methods of manipulating
    them and converting them to executable and inferrable models and guides.
   
    CalipyNode: Class representing the nodes in the DAG. This is the base class
    for data, instruments, effects, and quantities. Contains as attributes its 
    input/output signature and a simulate method. Further methods are related 
    automatically inferring ancestor and descendent nodes as well as incoming and
    outgoing edges.
    
    CalipyEdge: Class representing the edges in the DAG. This class contains as
    attributes source and target nodes and a dictionary edge_dict that summarizes
    the data flow along the edges. 
  
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
import copy
from calipy.core.utils import CalipyRegistry, format_mro
from abc import ABC, abstractmethod



# DAG class provides a data structure for a dag (directed acyclic graph) that 
# encodes all dependencies and neighborhood relations of the CalipyNode objects.

class CalipyDAG():
    """
    The CalipyDAG class provides a comprehensive representation of the relations
    between entities like data, instruments, effects, or quantities. This is done
    in terms of a directed acyclic graph. The class provides methods for constructing,
    illustrating, and manipulating the dag as well as for translating it to pyro.
    """
          
    def __init__(self, name):
        self.dtype = self.__class__.__name__
        self.name = name
        self.nodes = {}
        self.edges = []
        self.node_registry = CalipyRegistry()

    def add_node(self, node):
        if node.name in self.nodes:
            raise ValueError(f"Node {node.name} already exists.")
        self.nodes[node.name] = node

    def add_edge(self, from_node, to_node):
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError("Both nodes must exist in the DAG before adding an edge.")
        self.edges.append((from_node, to_node))
    
    def display(self):
        for node in self.nodes.values():
            print(node)
        for edge in self.edges:
            print(f"{edge[0]} -> {edge[1]}")
    
    def execute(self):
        # Implement execution logic based on node dependencies
        pass
    
    def __repr__(self):
        return "{}(name: {})".format(self.dtype, self.name)
    


# Node class is basis for data, instruments, effects, and quantities of calipy.
# Method and attributes are used mostly for abstract operations like DAG construction
# and execution

class CalipyNode(ABC):
    """
    The CalipyNode class provides a comprehensive representation of the data 
    flow and the dependencies between the nodes. 
    """
    
    _node_counters = 0   
    
    
    def __init__(self, node_type = None, node_name = None, info_dict = {}, **kwargs):
                
        # Basic infos
        # self.dtype_chain = [classname for classname in self.__class__.__mro__]
        self.dtype_chain = format_mro(self.__class__)
        self.dtype = self.__class__.__name__
        self.type = node_type
        self.name = node_name
        self.info = info_dict
        
        # Upon instantiation increment _node_counters dict
        CalipyNode._node_counters += 1
        
        self.node_nr = copy.copy(CalipyNode._node_counters)
        # self.id = "{}_{}_{}_{}".format(self.dtype_chain, self.type, self.name, self.node_number)
        
        self.probmodel = kwargs.get('probmodel', empty_probmodel)
        self.model_or_guide = kwargs.get('model_or_guide', 'model')
        
        if self.model_or_guide == 'model':
            self.probmodel.model_dag.node_registry.register(self.name, self)
        elif self.model_or_guide == 'guide':
            self.probmodel.guide_dag.node_registry.register(self.name, self)
        else :
            raise ValueError("KW Argument model_or_guide for class {} requires "
                             "values in ['model', 'guide'].".format(self.dtype))

        # Create a unique identifier based on type, name, and dag position
        # self.id = "{}_{}_{}".format(self.super_instrument_id, self.name, CalipyEffect._effect_counters[name])
        
    @abstractmethod
    def forward(self, input_vars, observations = None):
        pass
    
    def render(self, input_vars = None):
        graphical_model = pyro.render_model(model = self.forward, model_args= input_vars, render_distributions=True, render_params=True)
        return graphical_model
    
    def __repr__(self):
        return "{}(type: {} name: {})".format(self.dtype, self.type,  self.name)


# Edge class is basis for representation of dependencies between Node objects and
# is used mostly for abstract operations related to DAG construction.

class CalipyEdge(dict):
    """
    The CalipyEdge class provides a comprehensive representation of a specific 
    entity like data, instrument, effect, or quantity in terms of a node in a
    graph that describes dependence and relationships among the entities. It 
    provides attributes like depends_on and contributes_to that list ancestor and
    descendent nodes detailing the data flow between nodes. It contains setter
    and getter methods to investigate and manipulate the DAG that underlies the
    embedding procedure into pyro.
    """
    
        
    def __init__(self, node_type = None, node_name = None, info_dict = {}, **kwargs):
        
        # Basic infos
        self.dtype = self.__class__.__name__
        self.type = node_type
        self.name = node_name
        self.info = info_dict
        
        self.probmodel = kwargs.get('probmodel', empty_probmodel)
        self.model_or_guide = kwargs.get('model_or_guide', 'model')
        
        if self.model_or_guide == 'model':
            self.probmodel.model_dag.node_registry.register(self.name, self)
        elif self.model_or_guide == 'guide':
            self.probmodel.guide_dag.node_registry.register(self.name, self)
        else :
            raise ValueError("KW Argument model_or_guide for class {} requires "
                             "values in ['model', 'guide'].".format(self.dtype))

        # Create a unique identifier based on type, name, and dag position
        # self.id = "{}_{}_{}".format(self.super_instrument_id, self.name, CalipyEffect._effect_counters[name])
    
    def __repr__(self):
        return "{}(type: {} name: {})".format(self.dtype, self.type,  self.name)




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
    
        
    def train(self, model_fn, guide_fn, data, optim_opts,):
        self.optim_opts = optim_opts
        self.optimizer = optim_opts.get('optimizer', pyro.optim.NAdam({"lr": 0.01}))
        self.loss = optim_opts.get('loss', pyro.infer.Trace_ELBO())
        self.n_steps = optim_opts.get('n_steps', 1000)
        self.svi = pyro.infer.SVI(model_fn, guide_fn, self.optimizer, self.loss)
        
        self.loss_sequence = []
        for step in range(self.n_steps):
            loss = self.svi.step(observations = data)
            if step % 100 == 0:
                print('epoch: {} ; loss : {}'.format(step, loss))
            else:
                pass
            self.loss_sequence.append(loss)
            
        return self.loss_sequence
    
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



