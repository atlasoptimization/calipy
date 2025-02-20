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
import inspect
from functools import wraps
import torchviz
from calipy.core.utils import format_mro, dim_assignment, DimTuple
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
  


# class NodeStructure:
#     def __init__(self, effect_cls=None):
#         self._strict_mode = False  # Whether to enforce template dims
#         self.dims = {}  # {dim_name: (value, description)}
        
#         if effect_cls is not None:
#             # Clone the effect's default_nodestructure
#             self._strict_mode = True
#             default_ns = effect_cls.default_nodestructure
#             self.dims = copy.deepcopy(default_ns.dims)
#             self._generate_set_dims()

#     def set_dims(self, **kwargs):
#         """Base method (dynamically overridden if initialized from an Effect)"""
#         if self._strict_mode:
#             raise RuntimeError("Cannot freely set dims in strict mode (use Effect-bound NodeStructure)")
#         for name, value in kwargs.items():
#             self.dims[name] = (value, "User-defined dimension")

#     def _generate_set_dims(self):
#         # Dynamically create a version of set_dims with parameters
#         # matching the dims defined in the template
#         params = []
#         doc_lines = ["Set dimensions:"]
        
#         for name, (default_val, desc) in self.dims.items():
#             param = inspect.Parameter(
#                 name,
#                 inspect.Parameter.KEYWORD_ONLY,
#                 default=default_val,
#                 annotation=type(default_val)
#             )
#             params.append(param)
#             doc_lines.append(f"{name}: {desc} (default: {default_val})")

#         # Define the strict set_dims method
#         def strict_set_dims(**kwargs):
#             for key, value in kwargs.items():
#                 if key not in self.dims:
#                     raise ValueError(f"Invalid dim '{key}' for this NodeStructure")
#                 self.dims[key] = (value, self.dims[key][1])  # Preserve description

#         # Attach signature and docstring for IDE support
#         sig = inspect.Signature(params)
#         strict_set_dims.__signature__ = sig
#         strict_set_dims.__doc__ = "\n".join(doc_lines)
        
#         # Replace the base set_dims with the strict version
#         self.set_dims = strict_set_dims.__get__(self)

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
    
        # Investigate NodeStructure -------------------------------------------
        #
        # i) Imports and definitions
        import calipy
        from calipy.core.base import NodeStructure
        from calipy.core.effects import NoiseAddition
        #
        # ii) Set up node_structure
        node_structure = NodeStructure()
        node_structure.set_shape('batch_shape', (10, ), 'Batch shape description')
        node_structure.set_shape('event_shape', (5, ), 'Event shape description')
        node_structure.set_plate_stack('noise_stack', [('batch_plate', 10, -1, 
                    'plate denoting independent data points')], 'Plate stack for noise ')
        #
        # Can also be set up by calling the example_node_structure of some node
        example_node_structure = NodeStructure.from_node_class(NoiseAddition)
        #
        # iii) Investigate NodeStructure objects
        node_structure.description
        node_structure.print_shapes_and_plates()
        node_structure.generate_template()
        #
        # iv) Inherit from prebuilt example_node_structure
        new_node_structure = NoiseAddition.example_node_structure
        new_node_structure.print_shapes_and_plates()
        shape_updates = {'new_shape' : (11,)}
        plate_stack_updates = {'noise_stack': [('batch_plate_1', 22, -2, 
                    'plate denoting independent realizations')]}
        new_node_structure = new_node_structure.update(shape_updates, plate_stack_updates)
        # 
        # v) Build and check via class methods
        empty_node_structure = NodeStructure()
        NoiseAddition.check_node_structure(empty_node_structure)
        NoiseAddition.check_node_structure(new_node_structure)
    
    """
    @classmethod
    def from_node_class(cls, node_class):
        if hasattr(node_class, 'example_node_structure'):
            return copy.deepcopy(node_class.example_node_structure)
        return cls()
    
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
                line += "('{}', {}, {}, 'Plate description'),".format(plate.name, plate.size, plate.dim)
            line = line[:-1]
            line += ("], 'Plate stack description')")
            lines.append(line)
        return print("\n".join(lines))
    
    
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
        self.id_short = self._generate_id_short()
        
        
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
            missing_stack_keys = [key for key in cls.example_node_structure.plate_stacks.keys() if key not in node_structure.plate_stacks]
            # missing_plate_keys = [key for key in cls.example_node_structure.plates.keys() if key not in node_structure.plates]
            missing_keys = missing_shape_keys + missing_stack_keys #+ missing_plate_keys
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
    
        # Fetch optional arguments
        self.optim_opts = optim_opts or {}
        self.optimizer = self.optim_opts.get('optimizer', pyro.optim.NAdam({"lr": 0.01}))
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
                for batch_input, batch_output, idx in dataloader:
                    loss = self.svi.step(input_vars=batch_input, observations=batch_output)
                    epoch_loss += loss
                
                epoch_loss /= len(dataloader)
                self.loss_sequence.append(epoch_loss)
    
                if epoch % self.n_steps_report == 0:
                    print(f'epoch: {epoch} ; loss : {epoch_loss}')
        else:
            # Handle direct data input case
            for step in range(self.n_steps):
                loss = self.svi.step(input_vars=input_data, observations=output_data)
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



