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
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    CalipyDAG construction classes ----------------------------------------------
"""


import pyro
import copy
import inspect
import textwrap
from functools import wraps
import torchviz
from calipy.core.utils import format_mro, InputSchema
from calipy.core.data import DataTuple, CalipyDict, CalipyIO, preprocess_args
from abc import ABC, ABCMeta, abstractmethod

from types import MethodType
from typing import get_type_hints
from typing import Dict, Any, Optional, List, Type
from inspect import Parameter, Signature



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
#       

class NodeStructure():
    """ NodeStructure class is basis for defining batch_shapes, event_shapes, and plate
    configurations for a CalipyNode object. Provides functionality for attribute-
    like access and automated construction. Each object of NodeStructure class has
    attributes description, shapes, plates, plate_stacks.
    Methods include set_shape, set_plate_stack, update, print_shapes_and_plates,
    and generate_template which can be used to either set the properties of a
    newly instantiated object node_structure = NodeStructure() or to modify an
    existing object by updating it. NodeStructure objects are central for instantiating
    CalipyNode objects.
    
    :param args: optional arguments (can be None)
    :type args: list
    :param kwargs: dictionary containing keyword arguments (can be None)
    :type kwargs: dict
    :return: Empty Instance of the NodeStructure class to be populated by info
        via the set_shape and set_plate_stack methods.
    :rtype: NodeStructure
    
    Example usage: Run line by line to investigate Class
        
    .. code-block:: python
    
        # Set up NodeStructure -----------------------------------------------
        #
        # i) Imports and definitions
        import calipy
        from calipy.core.base import NodeStructure
        from calipy.core.effects import UnknownParameter
        #        
        # Specify some dimensions: param_dims, batch_dims feature in the template nodestructure
        # UnknownParameter.default_nodestructure while event_dims does not.
        param_dims = dim_assignment(['param_dim'], dim_sizes = [5])
        batch_dims = dim_assignment(['batch_dim'], dim_sizes = [20])
        event_dims = dim_assignment(['event_dim'], dim_sizes = [3])
        
        # ii) Set up generic node_structure
        # ... either directly via arguments:
        node_structure = NodeStructure()
        node_structure.set_dims(param_dims = param_dims, 
                                batch_dims = batch_dims, 
                                event_dims = event_dims)
        node_structure.set_dim_descriptions(param_dims = 'parameter dimensions',
                                            batch_dims = 'batch dimensions',
                                            event_dims = 'event_dimensions')
        # ... or by passing dictionaries
        node_structure.set_dims(**{'param_dims' : param_dims,
                                   'batch_dims' : batch_dims})
        node_structure.set_dim_descriptions(**{'param_dims' : 'parameter dimensions',
                                               'batch_dims' : 'batch dimensions'})
        
        # iii) Set up node structure tied to specific class
        param_ns_1 = NodeStructure(UnknownParameter)
        param_ns_2 = NodeStructure(UnknownParameter)
        param_ns_3 = NodeStructure(UnknownParameter)
        
        
        # Investigate NodeStructure -------------------------------------------
        #
        # The set_dims method inherits an updated docstring and autocompletion
        print(param_ns_1)   # Shows the default_nodestructure of UnknownParamter
        help(param_ns_1.set_dims)   # Shows that param_dims, batch_dims are arguments
        
        # The initialized node structure can be updated by inheritance or by directly setting
        # dimensions. It errors out, if a dimension is specified that is not specified
        # by the default_nodestructure
        #
        # Create nodestructure with custom param_dims and batch_dims
        param_ns_1.inherit_common_dims(node_structure)  
        print(param_ns_1)
        #
        # Create nodestructure with custom param_dims and default batch_dims
        param_ns_2.set_dims(param_dims = param_dims) 
        print(param_ns_2)   
        #
        # This errors out as it should: param_ns_3.set_dims(event_dims = event_dims)         
        #
        # iv) Investigate NodeStructure objects
        param_ns_1.dims
        param_ns_1.dim_names
        param_ns_1.dim_descriptions
        param_ns_1.node_cls
        #
        # It is possible to build the code that, if executed, generates the nodestructure
        param_ns_1.generate_template()
         
        # v) Build and check nodestructure via class methods
        empty_node_structure = NodeStructure()
        UnknownParameter.check_node_structure(empty_node_structure)
        UnknownParameter.check_node_structure(param_ns_1)
    
    """
    
    
    def __init__(self, node_cls=None):
        self.node_cls = node_cls   # Document if structure inherited from CalipyNode 
        self.node_cls_name = getattr(self.node_cls, 'name', None)
        self._strict_mode = False  # Whether to enforce template dims
        
        self.dims = {}  # {dim_name: DimTuple instance}
        self.dim_descriptions = {} # {dim_name : description string}
        self.dim_names = [] # list of keys to dims and dim_descriptions
        
        if node_cls is not None:
            # Clone the effect's default_nodestructure
            self._strict_mode = True
            default_ns = node_cls.default_nodestructure
            self.dims = copy.deepcopy(default_ns.dims)
            self.dim_descriptions = copy.deepcopy(default_ns.dim_descriptions)
            self.dim_names = list(self.dims.keys())
            
            # self.dim_sizes = {key: self.dims[key].sizes for key in self.dim_names}
            # self.dim_descriptions = {key: self.dims[key].descriptions for key in self.dim_names}
            
            self._generate_set_dims()

    def set_name(self, name):
        """ Sets name of the node_structure. Needed when defining node_structures
        from scratch, otherwise name is inherited."""
        self.node_cls_name = name

    def set_dims(self,  **kwargs):
        """ Base method; is dynamically overridden if initialized from an CalipyNode 
        subclass, (e.g. CalipyEffect, CalipyQuantity, etc).
        Sets dimensions of the node structure by defining them explicitly via
        kwargs {dim_name : dim_tuple}.
        
        :param kwargs: Optional keyword arguments declaring the dims manually via
            dim_name = dim_tuple, where dim_tuple is of DimTuple class
        :type kwargs: dict
        :return: None, changes dims directly in self
        :rtype: None
        :raises RuntimeError: If dims are set that do not belong to the default
            NodeStructure of some CalipyNode subclass.
        """
        
        # i) Iterate through kwargs
        for name, value in kwargs.items():
            self.dims[name] = value
        self.dim_names = list(self.dims.keys())
        
        
    def inherit_common_dims(self,  other_nodestructure):
        """ Inherits dimensions of other_nodestructure; takes only those DimTuple
        objects of other_nodestructure that are present also in self. Useful for
        injecting the dims of some prebuilt nodestructure into a default nodestructure.
        
        Note: Meant for modifying self with info from other_nodestructure. If you
        want to copy a default nodestructure, use ns = Nodestructure(NodeClass)
        instead.
        
        :param nodestructure: An optional nodestructure to inherit dimensions from
        :type nodestructure: NodeStructure
        :return: None, changes dims directly in self
        :rtype: None
        :raises RuntimeError: If dims are set that do not belong to the default
            NodeStructure of some CalipyNode subclass.
        """
        
        # i) Cycle throgh the dims of other and set self dims.
        for name, dims in other_nodestructure.dims.items():
            if name in self.dims.keys():
                self.set_dims(**{name : dims})
                
        
    def set_dim_descriptions(self, **kwargs):
        """ Base method; is dynamically overridden if initialized from an CalipyNode 
        subclass, (e.g. CalipyEffect, CalipyQuantity, etc).
        Sets dim_descriptions of the node structure by defining them explicitly 
        via kwargs {dim_name : dim_description}.
        
        :param kwargs: Optional keyword arguments declaring the descriptions manually
            via dim_name =   description, where description is a string
        :type kwargs: dict
        :return: None, changes dims directly in self
        :rtype: None
        """
        
        # i) Iterate through kwargs
        for name, desc in kwargs.items():
            self.dim_descriptions[name] = desc
    

    def _generate_set_dims(self):
        # Dynamically create a version of set_dims with parameters
        # matching the dims defined in the template
        params = []
        names = []
        predoc_args = ', '.join(self.dims.keys())
        
        # i) Initialize doc lines
        doc_lines = [
            textwrap.dedent(f"""\
            Sets dimensions for node structure of node class {self.node_cls_name} by either:
              - Defining them explicitly via kwargs {{dim_name: dim_tuple}}, or
              - Inheriting dims from another node structure.  
            
            Set dimensions:""")]
        
        # ii) Iterate through dims, add them to params
        for name, default_val in self.dims.items():
            param = inspect.Parameter(
                name,
                inspect.Parameter.KEYWORD_ONLY,
                default=default_val,
                annotation=type(default_val)
            )
            description = self.dim_descriptions[name]
            params.append(param)
            names.append(name)            
            doc_lines.append(f"\n {name}: {description} \n \t \t (node default: {default_val})")
        predoc_line = "nodestructure.set_dims({}) ".format(predoc_args)
        
        # iii) Define the strict set_dims method
        def strict_set_dims(self, **kwargs):
            for key, value in kwargs.items():
                if key not in self.dims.keys():
                    raise ValueError("Invalid dim '{}' for NodeStructure object "
                                     " with dims {} inherited from class {}."
                                     .format(key, self.dim_names, self.node_cls_name))
                self.dims[key] = value

        # iv) Attach signature and docstring for IDE support
        sig = inspect.Signature(params)
        strict_set_dims.__signature__ = sig
        strict_set_dims.__doc__ = predoc_line + "\n" + "\n".join(doc_lines)
        
        # Replace the base set_dims with the strict version
        self.set_dims = strict_set_dims.__get__(self)
        
        
    # def print_shapes_and_plates(self):
    #     print('\nShapes :')
    #     for shape_name, shape in self.shapes.items():
    #         print(shape_name, '| ', shape, ' |', self.description[shape_name])
            
    #     print('\nPlates :')
    #     for plate_name, plate in self.plates.items():
    #         print(plate_name, '| size = {} , dim = {} |'.format(plate.size, plate.dim), self.description[plate_name])
        
    #     print('\nPlate_stacks :')
    #     for stack_name, stack in self.plate_stacks.items():
    #         print(stack_name, '| ', [plate.name for plate in stack], ' |', self.description[stack_name])
    
    
    def generate_template(self):
        """ Generate code string that produces the current nodestructure and can
        be used to bake a specific nodestructure into a script.
        """
        
        # i) Initialize the dicts and lists
        dimset_dict = {}
        dims_names = []
        lines_dimgen = []
        lines_dimset = ["dimset_dict = {}"]
        lines = ["node_structure = NodeStructure({})\n".format(self.node_cls.name)]
        
        # ii) Cycle through the dims, extract generation code and setter code
        for dims_name, dims in self.dims.items():
            names = self.dims[dims_name].names
            sizes = self.dims[dims_name].sizes
            descriptions = self.dims[dims_name].descriptions
            
            dims_names.append(dims_name)
            lines_dimgen.append("{} = dim_assignment({}, dim_sizes = {}, dim_descriptions = {})"
                                .format(dims_name, names, sizes, descriptions))
            # dimset_dict[dims_name] = dims_name  
            lines_dimset.append("dimset_dict['{}'] = {}".format(dims_name, dims_name))
            
        # iii) Assemble strings and join commands
        lines_dimdesc = ["node_structure.set_dim_descriptions(**{})".format(self.dim_descriptions)]
        lines = lines + lines_dimgen + lines_dimset + lines_dimdesc
        lines.append("node_structure.set_dims(**dimset_dict)")
        return print("\n".join(lines))
    
    
    def __str__(self):
        return self.__repr__()


    def __repr__(self):
        dim_summary_list = []
        for name, dim in self.dims.items():
            dim_summary_list.append("name: {}, obj : {}, sizes : {} \n \t description : {}".format(
                name, self.dims[name], self.dims[name].sizes, self.dim_descriptions[name]))
        dim_summary = '\n '.join(dim_summary_list)
        repr_string = "NodeStructure instance for node class {} with dims: \n {}".format(
            self.node_cls_name, dim_summary)
        return repr_string




# # NodeMeta class is for having a MetaClass for CalipyNode that modifies class 
# # construction thereby allowing automatic inclusion of create_input_args, 
# # create_observation factory methods based in input_schema

# class NodeMeta(ABCMeta):
#     def __new__(cls, name, bases, attrs):
#         # Create the class first
#         new_cls = super().__new__(cls, name, bases, attrs)

#         # Generate create_input_args if input_schema exists
#         if hasattr(new_cls, 'input_schema'):
#             new_cls.create_input_args = cls._create_factory_method('input_schema')

#         # Generate create_observations if observation_schema exists
#         if hasattr(new_cls, 'observation_schema'):
#             new_cls.create_observations = cls._create_factory_method('observation_schema')

#         return new_cls

#     @classmethod
#     def _create_factory_method(cls, schema_attr):
#         def factory_method(self, **kwargs):
#             schema = getattr(self, schema_attr)
#             data = {}
            
#             # Validate required keys
#             for key in schema.required_keys:
#                 if key not in kwargs:
#                     raise ValueError(f"Missing required key: {key}")
#                 data[key] = kwargs[key]
            
#             # Add optional keys with defaults
#             for key in schema.optional_keys:
#                 data[key] = kwargs.get(key, schema.defaults.get(key))
            
#             return DataTuple(schema, data)

#         # Add parameter annotations for IDE autocompletion
#         schema = getattr(factory_method, '_schema', None)
#         if schema:
#             annotations = {key: schema.key_types.get(key, Any) for key in schema.required_keys + schema.optional_keys}
#             annotations['return'] = DataTuple
#             factory_method.__annotations__ = annotations

#         return factory_method

# class CalipyNode(ABC, metaclass = NodeMeta):
    


# class CustomClassMethod(classmethod):
#     """A classmethod that preserves __doc__, __annotations__, and __signature__."""
#     def __init__(self, func):
#         super().__init__(func)
#         self.__doc__ = func.__doc__
#         self.__annotations__ = func.__annotations__
#         self.__signature__ = inspect.signature(func)

#     def __get__(self, instance, owner):
#         # Bind the method and attach metadata
#         bound_method = super().__get__(instance, owner)
#         bound_method.__doc__ = self.__doc__
#         bound_method.__annotations__ = self.__annotations__
#         bound_method.__signature__ = self.__signature__
#         return bound_method


# Node class is basis for data, instruments, effects, and quantities of calipy.
# Method and attributes are used mostly for abstract operations like DAG construction
# and execution


class CalipyNode(ABC):
    """
    The CalipyNode class provides a comprehensive representation of the data 
    flow and the dependencies between the nodes. 
    """
    
    _instance_count = {}  # Class-level dictionary to keep count of instances per subclass
    
    # Optional: Define default schemes (subclasses can override)
    input_vars_schema: Optional[InputSchema] = None
    observation_schema: Optional[InputSchema] = None
    
    def __init__(self, node_type = None, node_name = None, info_dict = {}, **kwargs):
                
        # Basic infos
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
        self.id_short = self._generate_id_short()
        
    @classmethod
    def check_node_structure(cls, node_structure):
        """ Checks if the node_structure instance has all the keys and correct structure as the class template """
        if hasattr(cls, 'default_nodestructure'):
            default_ns = cls.default_nodestructure
            missing_dim_keys = [key for key in default_ns.dims.keys() if key not in node_structure.dims.keys()]            
            missing_keys = missing_dim_keys
            if missing_keys:
                return False, 'keys missing: {}'.format(missing_keys)
            return True, 'all keys from default_node_structure present in node_structure'
        else:
            raise NotImplementedError("This class does not define an example_node_structure.")
    
    # @classmethod
    # def build_node_structure(cls, basic_node_structure, shape_updates, plate_stack_updates):
    #     """ Create a new NodeStructure based on basic_node_structure but with updated values """
    #     new_node_structure = basic_node_structure.update(shape_updates, plate_stack_updates)
    #     return new_node_structure
        
    def _generate_id(self):
        # Generate the ID including all relevant class counts in the MRO
        id_parts = []
        for cls in reversed(self.__class__.__mro__[:-2]):
            count = CalipyNode._instance_count.get(cls, 0)
            id_parts.append(f"{cls.__name__}_{count}")
        return '__'.join(id_parts)
    
    
    def _generate_id_short(self):
        # Generate the short ID including only node counts 
        id_short_parts = []
        for cls in reversed(self.__class__.__mro__[:-2]):
            if cls.__name__ == 'CalipyNode':
                count = CalipyNode._instance_count.get(cls, 0)
                id_short_parts.append(f"Node_{count}")                           
        return '__'.join(id_short_parts)
    
    
    @abstractmethod
    def forward(self, input_vars = None, observations = None, subsample_indices = None, **kwargs):
        pass
    
    
    def render(self, input_vars = None):
        graphical_model = pyro.render_model(model = self.forward, model_args= (input_vars,), render_distributions=True, render_params=True)
        return graphical_model
    
    
    def render_comp_graph(self, input_vars = None):
        output = self.forward(input_vars)
        comp_graph = torchviz.make_dot(output)
        return comp_graph
        
    
    def __repr__(self):
        return "{}(type: {} name: {})".format(self.dtype, self.type,  self.name)
    
    
        

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # If the subclass hasn't overridden `name` at the class level, set it
        if 'name' not in cls.__dict__:
            cls.name = cls.__name__
        
        # Generate create_input_vars if input_schema exists
        if hasattr(cls, 'input_vars_schema') and cls.input_vars_schema is not None:
            cls.create_input_vars = cls._create_factory_method('input_vars_schema', 
                        'create_input_vars', 'Create a DataTuple for input_vars.')
            
        # Generate create_observations if observation_schema exists
        if hasattr(cls, 'observation_schema') and cls.observation_schema is not None:
            cls.create_observations = cls._create_factory_method('observation_schema', 
                            'create_observations', 'Create a DataTuple for observations.')
    
    @classmethod
    def _create_factory_method(cls, schema_attr: str, method_name: str, description: str):
        """Generates a factory method with documentation and type hints."""
        schema = getattr(cls, schema_attr)
        
        # Generate parameters for the function signature
        parameters = []
        for key in schema.required_keys:
            parameters.append(Parameter(
                name=key,
                kind=Parameter.KEYWORD_ONLY,
                default=Parameter.empty,
                annotation=schema.key_types.get(key, Any)
            ))
        for key in schema.optional_keys:
            parameters.append(Parameter(
                name=key,
                kind=Parameter.KEYWORD_ONLY,
                default=schema.defaults.get(key, Parameter.empty),
                annotation=schema.key_types.get(key, Any)
            ))
        signature = Signature(parameters)
        
        # Define the function
        def factory_func(_cls, **kwargs):
            key_list = []
            data_list = []
            for key in schema.required_keys:
                if key not in kwargs:
                    raise ValueError(f"Missing required key: {key}")
                key_list.append(key)
                data_list.append(kwargs[key])
            for key in schema.optional_keys:
                key_list.append(key)
                data_list.append(kwargs.get(key, schema.defaults.get(key)))
            return DataTuple(key_list, data_list)
        
        # Attach metadata
        factory_func.__name__ = method_name
        factory_func.__doc__ = (
            f"{method_name}{signature} -> DataTuple \n"
            f"Creates a DataTuple for {schema_attr}.\n\n"
            f"Parameters:\n"
            + "\n".join(
                f"{key} ({schema.key_types.get(key, 'Any')}): {'Required' if key in schema.required_keys else 'Optional'}"
                for key in (schema.required_keys + schema.optional_keys)
            )
            + "\n\nReturns:\n    DataTuple: The constructed DataTuple instance."
        )
        factory_func.__signature__ = signature
        factory_func.__annotations__ = {
            key: schema.key_types.get(key, Any)
            for key in (schema.required_keys + schema.optional_keys)
        }
        factory_func.__annotations__["return"] = DataTuple
        
        # Convert to classmethod and return
        return classmethod(factory_func)
        




"""
    CalipyProbModel class ----------------------------------------------------
"""


# Probmodel class determines attributes and methods for summarizing and accessing
# data, instruments, effects, and quantities of the whole probabilistic model.

class CalipyProbModel(CalipyNode):
    """ CalipyProbModel is an abstract base class that integrates the model, guide, and training components 
    for probabilistic models within the Calipy framework. It serves as the foundation for building and 
    training probabilistic models by providing methods to define the model, guide, and manage optimization
    and training procedures.

    This class is designed to be subclassed, where users define the specific `model` and `guide` methods 
    based on their probabilistic model requirements. The `train` method facilitates the training process 
    using stochastic variational inference (SVI) by interacting with Pyro's SVI module.

    :param type: An optional string representing the type of the model. This can be used to categorize 
        or identify the model within larger workflows.
    :type type: str, optional
    :param name: An optional string representing the name of the model. This name is useful for tracking 
        and referencing the model within a project or experiment.
    :type name: str, optional
    :param info: An optional dictionary containing additional information about the model, such as 
        metadata or configuration details.
    :type info: dict, optional
    
    :return: An instance of the CalipyProbModel class.
    :rtype: CalipyProbModel

    Example usage:

    .. code-block:: python
        
        # Architecture of CalipyProbModel objects is as below
        class MyProbModel(CalipyProbModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Integrate nodes or parameters specific to the model
                self.some_param = pyro.param("some_param", torch.tensor(1.0))

            def model(self, input_data, output_data):
                # Define the generative model
                pass

            def guide(self, input_data, output_data):
                # Define the guide (variational distribution)
                pass

        prob_model = MyProbModel(name="example_model")
        prob_model.train(input_data, output_data, optim_opts)
        
        
        # Here is a fuly worked example:
            
        # i) Imports and definitions
        import torch
        import pyro
        import calipy
        from calipy.core.utils import dim_assignment
        from calipy.core.data import DataTuple
        from calipy.core.tensor import CalipyTensor
        from calipy.core.effects import UnknownParameter, NoiseAddition
        from calipy.core.base import NodeStructure, CalipyProbModel
        
        # ii) Set up unknown mean parameter
        batch_dims_param = dim_assignment(['bd_p1'], dim_sizes = [10])
        param_dims_param = dim_assignment(['pd_p1'], dim_sizes = [2])
        param_ns = NodeStructure(UnknownParameter)
        param_ns.set_dims(param_dims = param_dims_param, batch_dims = batch_dims_param)
        mu_object = UnknownParameter(param_ns)
        
        # iii) Set up noise addition
        batch_dims_noise = dim_assignment(['bd_n1', 'bd_n2'], dim_sizes = [10,2])
        event_dims_noise = dim_assignment(['ed_n1'], dim_sizes = [0])
        noise_ns = NodeStructure(NoiseAddition)
        noise_ns.set_dims(batch_dims = batch_dims_noise, event_dims = event_dims_noise)
        noise_object = NoiseAddition(noise_ns)
        
        sigma = torch.ones(batch_dims_noise.sizes)
        
        # iv) Simulate some data
        mu_true = torch.tensor([0.0,5.0]).reshape([2])
        sigma_true = 1.0
        data_tensor = pyro.distributions.Normal(loc = mu_true, scale = sigma_true).sample([10]) 
        data_cp = CalipyTensor(data_tensor, batch_dims_noise)
        data = DataTuple(['sample'], [data_cp])
        
        # v) Define ProbModel
        class MyProbModel(CalipyProbModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
        
            def model(self, input_vars = None, observations = None):
                # Define the generative model
                mu = mu_object.forward()
                input_vars = DataTuple(['mean', 'standard_deviation'], [mu, sigma])
                sample = noise_object.forward(input_vars, observations = observations)
                return sample
        
            def guide(self, input_vars = None, observations = None):
                # Define the guide (variational distribution)
                pass
        
        # vi) Inference
        prob_model = MyProbModel(name="example_model")
        output = prob_model.model(observations = data)
        optim_opts = {'n_steps' : 2000, 'learning_rate' : 0.01}
        prob_model.train(input_data = None, output_data = data, optim_opts = optim_opts)
        
    """
    
    # i) Initialization
    def __init__(self, type = None, name = None, info = None):
        """ Initializes the CalipyProbModel with basic information, setting up the structure for input 
        and output data handling, and optionally categorizing the model.

        :param type: An optional string to specify the type of the model, aiding in categorization.
        :type type: str, optional
        :param name: An optional string to provide a name for the model, useful for identification and tracking.
        :type name: str, optional
        :param info: An optional dictionary to store additional metadata or configuration details about the model.
        :type info: dict, optional
        
        :return: None
        """
        
        # Basic infos
        super().__init__(node_type = type, node_name = name, info_dict = info)
        self.input_data = None
        self.output_data = None
        # self.model_dag = CalipyDAG('Model_DAG')
        # self.guide_dag = CalipyDAG('Guide_DAG')
    
    def forward(self):
        """  The forward method of CalipyProbModel is abstract and intended to be implemented by subclasses. 
        This method serves as a placeholder for the core logic that defines how data flows through the model.
        
        In the context of CalipyProbModel, this method might be left abstract if not required directly, 
        as the `model` and `guide` methods typically handle the main computational tasks.
        
        :return: None
        """
        pass

        # self.id = "{}_{}".format(self.type, self.name)
    
    @abstractmethod
    def model(self, input_data, output_data):
        """ Abstract method that must be implemented in subclasses. The `model` method defines the generative 
        process or the probabilistic model that describes how the observed data is generated from latent 
        variables. This method is expected to include parameters and sampling statements that define the model's 
        stochastic behavior.

        :param input_data: The input data required for the model, which might include explanatory variables
            influencing the probabilistic process.
        :type input_data: torch.tensor or tuple of torch.tensor
        :param output_data: The observed data that the model aims to describe or explain through the generative process.
        :type output_data: torch.tensor or tuple of torch.tensor
        
        :return: None
        """
        pass
    
    @abstractmethod
    def guide(self, input_data, output_data):
        """ Abstract method that must be implemented in subclasses. The `guide` method defines the variational 
        distribution used in the inference process. This distribution approximates the posterior distribution 
        of the latent variables given the observed data.

        :param input_data: The input data required for the guide, which might include explanatory variables
            influencing the variational distribution.
        :type input_data: torch.tensor or tuple of torch.tensor
        :param output_data: The observed data that guides the variational distribution in approximating 
            the posterior of the latent variables.
        :type output_data: torch.tensor or tuple of torch.tensor
        
        :return: None
        """
        pass
    
        
    # def train(self, input_data, output_data, optim_opts):
    #     """ Trains the probabilistic model using stochastic variational inference (SVI). The `train` method 
    #     iterates over a specified number of steps to optimize the model's parameters by minimizing the 
    #     specified loss function (default: ELBO).

    #     :param input_data: The input data to be used by the model during training, often comprising features 
    #         or covariates.
    #     :type input_data: Any
    #     :param output_data: The observed data that the model aims to fit, which could include measurements 
    #         or labels.
    #     :type output_data: Any
    #     :param optim_opts: A dictionary of options for the optimizer and loss function, including:
    #         - `optimizer`: The Pyro optimizer to be used (default: NAdam).
    #         - `loss`: The loss function used for optimization (default: Trace_ELBO).
    #         - `n_steps`: The number of optimization steps (default: 1000).
    #     :type optim_opts: dict
        
    #     :return: A list of loss values recorded during training.
    #     :rtype: list of float
    #     """
        
    #     self.optim_opts = optim_opts
    #     self.optimizer = optim_opts.get('optimizer', pyro.optim.NAdam({"lr": 0.01}))
    #     self.loss = optim_opts.get('loss', pyro.infer.Trace_ELBO())
    #     self.n_steps = optim_opts.get('n_steps', 1000)
    #     self.svi = pyro.infer.SVI(self.model, self.guide, self.optimizer, self.loss)
        
    #     self.loss_sequence = []
    #     for step in range(self.n_steps):
    #         loss = self.svi.step(input_vars = input_data, observations = output_data)
    #         if step % 100 == 0:
    #             print('epoch: {} ; loss : {}'.format(step, loss))
    #         else:
    #             pass
    #         self.loss_sequence.append(loss)
            
    #     return self.loss_sequence
    
    def train(self, input_data=None, output_data=None, dataloader=None, optim_opts=None):
        """ Trains the probabilistic model using stochastic variational inference (SVI). The `train` method 
        supports either direct input/output data or a single DataLoader object for batch processing.
    
        :param input_data: The input data to be used by the model during training. This should be provided 
            if not using a DataLoader.
        :type input_data: torch.tensor or tuple of torch.tensor, optional
        :param output_data: The observed data that the model aims to fit. This should be provided if not 
            using a DataLoader.
        :type output_data: torch.tensor or tuple of torch.tensor, optional
        :param dataloader: A DataLoader object that provides batches of synchronized input and output data.
            If this is provided, `input_data` and `output_data` should be None.
        :type dataloader: torch.utils.data.DataLoader, optional
        :param optim_opts: A dictionary of options for the optimizer and loss function, including:
            - `optimizer`: The Pyro optimizer to be used (default: NAdam).
            - `loss`: The loss function used for optimization (default: Trace_ELBO).
            - `n_steps`: The number of optimization steps (default: 1000).
            - `n_steps_report`: The number of optimization steps after which reporting is done (default: 100).
        :type optim_opts: dict, optional
        
        :return: A list of loss values recorded during training.
        :rtype: list of float
    
        :raises ValueError: If both `input_data`/`output_data` and `dataloader` are provided, or if neither 
            is provided.
        """
    
        # Validate input configuration
        if dataloader is not None:
            if input_data is not None or output_data is not None:
                raise ValueError("Either provide `input_data` and `output_data`, or `dataloader`, but not both.")
        elif output_data is None:
            raise ValueError("Either `input_data` and `output_data` must be provided, or `dataloader` must be set.")
    
        # Wrap input_data and output_data into CalipyDict
        input_data_io, output_data_io, subsample_index_io = preprocess_args(input_data,
                                        output_data, subsample_index = None)
        
        # Fetch optional arguments
        lr = optim_opts.get('learning_rate', 0.01)
        self.optim_opts = optim_opts or {}
        self.optimizer = self.optim_opts.get('optimizer', pyro.optim.NAdam({"lr": lr}))
        self.loss = self.optim_opts.get('loss', pyro.infer.Trace_ELBO())
        self.n_steps = self.optim_opts.get('n_steps', 1000)
        self.n_steps_report = self.optim_opts.get('n_steps_report', 100)
        
        # Set optimizer and initialize training
        self.svi = pyro.infer.SVI(self.model, self.guide, self.optimizer, self.loss)
        self.loss_sequence = []
    
        if dataloader is not None:
            # Handle DataLoader case
            for epoch in range(self.n_steps):
                epoch_loss = 0
                for batch_input, batch_output, batch_index in dataloader:
                    loss = self.svi.step(input_vars=batch_input,
                                         observations=batch_output,
                                         subsample_index = batch_index)
                    epoch_loss += loss
                
                epoch_loss /= len(dataloader)
                self.loss_sequence.append(epoch_loss)
    
                if epoch % self.n_steps_report == 0:
                    print(f'epoch: {epoch} ; loss : {epoch_loss}')
        else:
            # Handle direct data input case
            for step in range(self.n_steps):
                loss = self.svi.step(input_vars=input_data_io, observations=output_data_io)
                if step % self.n_steps_report == 0:
                    print(f'epoch: {step} ; loss : {loss}')
                self.loss_sequence.append(loss)
        
        return self.loss_sequence
    
            
    
    def render(self, input_vars = None):
        """ Renders a graphical representation of the probabilistic model and guide using Pyro's 
        `render_model` function. This visualization helps in understanding the structure of the model, 
        including the relationships between variables and distributions.

        :param input_vars: Optional input variables that might influence the model structure.
        :type input_vars: Any, optional
        
        :return: A tuple containing the graphical representations of the model and guide.
        :rtype: tuple of (graphical_model, graphical_guide)
        """
        
        graphical_model = pyro.render_model(model = self.model, model_args= (input_vars,), render_distributions=True, render_params=True)
        graphical_guide = pyro.render_model(model = self.guide, model_args= (input_vars,), render_distributions=True, render_params=True)
        return graphical_model, graphical_guide
    
    
    def render_comp_graph(self, input_vars = None):
        """ Renders the computational graph of the model and guide using `torchviz`. This method 
        visualizes the flow of computations within the model, which can be useful for debugging 
        or understanding the sequence of operations.

        :param input_vars: Optional input variables that influence the computational graph.
        :type input_vars: Any, optional
        
        :return: A tuple containing the computational graphs of the model and guide.
        :rtype: tuple of (comp_graph_model, comp_graph_guide)
        """
        
        # Model pass
        model_output = self.model(input_vars)
        comp_graph_model = torchviz.make_dot(model_output)
        # Guide pass
        guide_output = self.guide(input_vars)
        comp_graph_guide = torchviz.make_dot(guide_output)
        return comp_graph_model, comp_graph_guide
    
    def __repr__(self):
        """ Provides a string representation of the CalipyProbModel, including its type and name, 
        which is useful for logging or debugging.
        """
        return "{}(type: {} name: {})".format(self.dtype, self.type,  self.name)



