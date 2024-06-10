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
import torchviz
from calipy.core.utils import format_mro
from abc import ABC, abstractmethod



# DAG class provides a data structure for a dag (directed acyclic graph) that 
# encodes all dependencies and neighborhood relations of the CalipyNode objects.

# class CalipyDAG():
#     """
#     The CalipyDAG class provides a comprehensive representation of the relations
#     between entities like data, instruments, effects, or quantities. This is done
#     in terms of a directed acyclic graph. The class provides methods for constructing,
#     illustrating, and manipulating the dag as well as for translating it to pyro.
#     """
          
#     def __init__(self, name):
#         self.dtype = self.__class__.__name__
#         self.name = name
#         self.nodes = {}
#         self.edges = []
#         self.node_registry = CalipyRegistry()

#     def add_node(self, node):
#         if node.name in self.nodes:
#             raise ValueError(f"Node {node.name} already exists.")
#         self.nodes[node.name] = node

#     def add_edge(self, from_node, to_node):
#         if from_node not in self.nodes or to_node not in self.nodes:
#             raise ValueError("Both nodes must exist in the DAG before adding an edge.")
#         self.edges.append((from_node, to_node))
    
#     def display(self):
#         for node in self.nodes.values():
#             print(node)
#         for edge in self.edges:
#             print(f"{edge[0]} -> {edge[1]}")
    
#     def execute(self):
#         # Implement execution logic based on node dependencies
#         pass
    
#     def __repr__(self):
#         return "{}(name: {})".format(self.dtype, self.name)
  
    
# NodeStructure class is basis for defining batch_shapes, event_shapes, and plate
# configurations for a CalipyNode object. Provides functionality for dictionary-
# like access and automated construction.

class NodeStructure():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.description = {}
        self.shapes = {}
        self.plates = {}
        self.plate_stacks = {}

    def set_shape(self, shape_name, shape_value, shape_description = None):
        self.shapes[shape_name] = shape_value
        if shape_description is not None or shape_name not in self.description.keys():
            self.description[shape_name] = shape_description
        # self.shape_example[shape_name] = shape_value

    def set_plate_stack(self, stack_name, plate_data_list, stack_description = None):
        """
        Set stack of plate configurations from a list of tuples and a name.
        Each tuple should contain (plate_name, plate_size, plate_dim, plate_description).
        
        :param stack_name: String, represents the name of the stack of plates.
        :param plate_data_list: List of tuples, each representing plate data.
        """
        # self.plate_stacks[stack_name] = []
        # for plate_name, plate_size, plate_dim, plate_description in plate_data_list:
        #     self.plate_stacks[stack_name].append({'name': plate_name, 'size': plate_size, 'dim': plate_dim})
        #     self.plates[plate_name] = {'name': plate_name, 'size': plate_size, 'dim': plate_dim}
        #     self.description[plate_name] = plate_description
        
        if stack_description is not None or stack_name not in self.description.keys():
            self.description[stack_name] = stack_description
            
        self.plate_stacks[stack_name] = []
        for plate_name, plate_size, plate_dim, plate_description in plate_data_list:
            self.plates[plate_name] = pyro.plate(plate_name, size = plate_size, dim = plate_dim)
            self.plate_stacks[stack_name].append(self.plates[plate_name])
            self.description[plate_name] = plate_description
            
    def update(self, shape_updates, plate_stack_updates):
        new_node_structure = copy.deepcopy(self)
        for shape_name, shape_value in  shape_updates.items():
            new_node_structure.set_shape(shape_name, shape_value)
        for stack_name, plate_data_list in plate_stack_updates.items():
            new_node_structure.set_plate_stack(stack_name, plate_data_list)
            
        return new_node_structure
        
            
            
    def print_shapes_and_plates(self):
        print('\nShapes :')
        for shape_name, shape in self.shapes.items():
            print(shape_name, '| ', shape, ' |', self.description[shape_name])
            
        print('\nPlates :')
        for plate_name, plate in self.plates.items():
            print(plate_name, '| size = {} , dim = {} |'.format(plate.size, plate.dim), self.description[plate_name])
        
        print('\nPlate_stacks :')
        for stack_name, stack in self.plate_stacks.items():
            print(stack_name, '| ', [plate.name for plate in stack], ' |', self.description[stack_name])
    
    def generate_template(self):
        lines = ["node_structure = NodeStructure()"]
        for shape_name, shape in self.shapes.items():
            line = "node_structure.set_shape('{}', {}, 'Shape description')".format(shape_name, shape)
            lines.append(line)
        for stack_name, stack in self.plate_stacks.items():
            line = "node_structure.set_plate_stack('{}', [".format(stack_name)
            for plate in stack:
                line += "({}, {}, {}, 'Plate description'),".format(plate.name, plate.size, plate.dim)
            line = line[:-1]
            line += ('], Plate stack description)')
            lines.append(line)
        return "\n".join(lines)
    
    
    def __str__(self):
        structure_description = super().__str__()
        meta_description = {k: f"{v} (Description: {self.description.get(k, 'No description')})" for k, v in self.items()}
        return f"Structure: {structure_description}\nMetadata: {meta_description}"




# Node class is basis for data, instruments, effects, and quantities of calipy.
# Method and attributes are used mostly for abstract operations like DAG construction
# and execution


class CalipyNode(ABC):
    """
    The CalipyNode class provides a comprehensive representation of the data 
    flow and the dependencies between the nodes. 
    """
    
    _instance_count = {}  # Class-level dictionary to keep count of instances per subclass
    
    
    def __init__(self, node_type = None, node_name = None, info_dict = {}, **kwargs):
                
        # Basic infos
        # self.dtype_chain = [classname for classname in self.__class__.__mro__]
        self.dtype_chain = format_mro(self.__class__)
        self.dtype = self.__class__.__name__
        self.type = node_type
        self.name = node_name
        self.info = info_dict
        
        
        # Build id
        
        # Using self.__class__ to get the class of the current instance
        for cls in reversed(self.__class__.__mro__[:-2]):
            if cls in CalipyNode._instance_count:
                CalipyNode._instance_count[cls] += 1
            else:
                CalipyNode._instance_count[cls] = 1
        self.id = self._generate_id()
        

        # self.model_or_guide = kwargs.get('model_or_guide', 'model')
        
        # if self.model_or_guide == 'model':
        #     self.probmodel.model_dag.node_registry.register(self.name, self)
        # elif self.model_or_guide == 'guide':
        #     self.probmodel.guide_dag.node_registry.register(self.name, self)
        # else :
        #     raise ValueError("KW Argument model_or_guide for class {} requires "
        #                      "values in ['model', 'guide'].".format(self.dtype))


        
    def _generate_id(self):
        # Generate the ID including all relevant class counts in the MRO
        id_parts = []
        for cls in reversed(self.__class__.__mro__[:-2]):
            count = CalipyNode._instance_count.get(cls, 0)
            id_parts.append(f"{cls.__name__}_{count}")
        return '__'.join(id_parts)
    
    @abstractmethod
    def forward(self, input_vars = None, observations = None):
        pass
    
    def render(self, input_vars = None):
        graphical_model = pyro.render_model(model = self.forward, model_args= (input_vars,), render_distributions=True, render_params=True)
        return graphical_model
    
    def render_comp_graph(self, input_vars = None):
        output = self.forward(input_vars)
        comp_graph = torchviz.make_dot(output)
        return comp_graph
    
    @classmethod
    def check_node_structure(cls, node_structure):
        """ Checks if the node_structure instance has all the keys and correct structure as the class template """
        if hasattr(cls, 'example_node_structure'):
            missing_shape_keys = [key for key in cls.example_node_structure.shapes.keys() if key not in node_structure.shapes]
            missing_plate_keys = [key for key in cls.example_node_structure.plates.keys() if key not in node_structure.plates]
            missing_stack_keys = [key for key in cls.example_node_structure.plate_stacks.keys() if key not in node_structure.plate_stacks]
            missing_keys = missing_shape_keys + missing_plate_keys + missing_stack_keys
            if missing_keys:
                return False, 'keys missing: {}'.format(missing_keys)
            return True, 'all keys from example_node_structure present in node_structure'
        else:
            raise NotImplementedError("This class does not define an example_node_structure.")

    @classmethod
    def build_node_structure(cls, basic_node_structure, shape_updates, plate_stack_updates):
        """ Create a new NodeStructure based on basic_node_structure but with updated values """
        new_node_structure = basic_node_structure.update(shape_updates, plate_stack_updates)
        return new_node_structure
    
    def __repr__(self):
        return "{}(type: {} name: {})".format(self.dtype, self.type,  self.name)


# Edge class is basis for representation of dependencies between Node objects and
# is used mostly for abstract operations related to DAG construction.

# class CalipyEdge(dict):
#     """
#     The CalipyEdge class provides a comprehensive representation of relationas
#     between entities like data, instrument, effect, or quantity. They help forming
#     a graph that describes dependence and relationships among the entities. In
#     particular it records the dimensionality of the flow of information between
#     the nodes that form the DAG that underlies the embedding procedure into pyro.
#     """
    
        
#     def __init__(self, node_type = None, node_name = None, info_dict = {}, **kwargs):
        
#         # Basic infos
#         self.dtype = self.__class__.__name__
#         self.type = node_type
#         self.name = node_name
#         self.info = info_dict
        

    
#     def __repr__(self):
#         return "{}(type: {} name: {})".format(self.dtype, self.type,  self.name)




"""
    CalipyProbModel class ----------------------------------------------------
"""


# Probmodel class determines attributes and methods for summarizing and accessing
# data, instruments, effects, and quantities of the whole probabilistic model.

class CalipyProbModel(CalipyNode):

    # i) Initialization
    def __init__(self, type = None, name = None, info = None):
        
        # Basic infos
        super().__init__(node_type = type, node_name = name, info_dict = info)
        self.input_data = None
        self.output_data = None
        # self.model_dag = CalipyDAG('Model_DAG')
        # self.guide_dag = CalipyDAG('Guide_DAG')
    
    def forward(self):
        pass

        # self.id = "{}_{}".format(self.type, self.name)
    
    @abstractmethod
    def model(self, input_data, output_data):
        pass
    
    @abstractmethod
    def guide(self, input_data, output_data):
        pass
    
        
    def train(self, input_data, output_data, optim_opts):
        self.optim_opts = optim_opts
        self.optimizer = optim_opts.get('optimizer', pyro.optim.NAdam({"lr": 0.01}))
        self.loss = optim_opts.get('loss', pyro.infer.Trace_ELBO())
        self.n_steps = optim_opts.get('n_steps', 1000)
        self.svi = pyro.infer.SVI(self.model, self.guide, self.optimizer, self.loss)
        
        self.loss_sequence = []
        for step in range(self.n_steps):
            loss = self.svi.step(input_vars = input_data, observations = output_data)
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
# type_EmptyProbModel = 'empty_probmodel'
# name_EmptyProbModel = 'base'
# info_dict_EmptyProbModel = {}

# class EmptyProbModel(CalipyProbModel):
        
#     def __init__(self, model_name):
#         super().__init__(model_type = type_EmptyProbModel, 
#                          model_name = model_name, 
#                          info_dict = info_dict_EmptyProbModel)
        

# empty_probmodel = EmptyProbModel(name_EmptyProbModel)

# ep_type = 'empty_probmodel'
# ep_name = 'base'
# ep_info = {'description': 'Demonstrator for CalipyProbModel class'}
# ep_dict = {'name': ep_name, 'type': ep_type, 'info' : ep_info}        

# empty_probmodel = CalipyProbModel(**ep_dict)


