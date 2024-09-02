#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to model a simple measurement process 
with a single unknown mean and known variance. The measurement procedure has
been used to collect a single datasets, that features n_meas samples. Inference
is performed to estimate the value of the underlying expected value. We also
showcase how the torchdim formalism can be used to perform subbatching.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Load and customize effects
    4. Build the probmodel
    5. Perform inference
    6. Analyse results and illustrate

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    1. Imports and definitions
"""


# i) Imports

# base packages
import torch
import pyro
import numpy as np
import math
import matplotlib.pyplot as plt
from pyro.distributions import constraints
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functorch.dim import dims

# calipy
import calipy
from calipy.core.base import CalipyNode, NodeStructure, CalipyProbModel
from calipy.core.utils import dim_assignment, generate_trivial_dims, context_plate_stack



# ii) Definitions

n_meas = 30
n_subbatch = 10


"""
    2. Simulate some data
"""


# i) Set up sample distributions

mu_true = torch.tensor(0.0)
sigma_true = torch.tensor(0.1)


# ii) Sample from distributions

data_distribution = pyro.distributions.Normal(mu_true, sigma_true)
output_data = data_distribution.sample([n_meas])
input_data = None

# The data now is a tensor of shape [n_meas] and reflects measurements being
# being taken of a single object with a single measurement device.

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.


# Build CalipyDataset class that allows DataLoader to batch over multiple dimensions

class CalipyDataset(Dataset):
    def __init__(self, input_data, output_data, shape_dict):
        
        # dataset type can be 'tensor', 'tensortuple', or 'tensortuplelist'
        # - default is tensor which means input_data and output_data are tensors,
        # e.g. output_data = torch.randn([2,2])
        # dims: batch and event dims are recorded in shape_dict with keys 'batch_dim',
        # 'input_event_dim', 'output_event_dim'. This use allows subbatching over
        # a flattened batch_dim.
        # - tensortuple allows input_data, output_data to be a tuple of tensors,
        # e.g. output_data = (torch.randn([2,2]), torch.ones([3]))
        # dims: the output_data is considered to be a single event and will be
        # handed to inference algorithms. Subbatching can be done by conversion
        # to tensortuplelist, which assumes the list index to be the batching index
        # - tensortuplelist allows input_data, output_data to be a tuple of tensors 
        # (data_1, data_2, ...) and produces a list [(data_11, data_12 ,...) ,
        # ... (data_n1, data_n2, ...)] 
        # dims: each list entry is a tuple of tensors interpreted as a single
        # event and the list index is the batching index.
        
        self.batch_dim = dim_assignment(['batch_dim'], dim_shapes = shape_dict['batch_shape'])
        
        # input data
        self.input_data = input_data  # Assuming data is a list or array of samples
        # self.input_batch_dim = dim_assignment(['input_batch_dim'], dim_shapes = shape_dict['batch_shape'])
        self.input_event_dim = dim_assignment(['input_event_dim'], dim_shapes = shape_dict['input_event_shape'])
        self.input_dim = self.batch_dim + self.input_event_dim
        
        # output_data
        self.output_data = output_data  # Assuming data is a list or array of samples
        # self.output_batch_dim = dim_assignment(['output_batch_dim'], dim_shapes = shape_dict['batch_shape'])
        self.output_event_dim = dim_assignment(['output_event_dim'], dim_shapes = shape_dict['output_event_shape'])
        self.output_dim = self.batch_dim + self.output_event_dim
        
        # reshaping to single batch_dim
        self.batch_length_total = math.prod(self.batch_dim.sizes)
        self.batch_dim_flat = dim_assignment(['total_batch_dim'], dim_shapes= [self.batch_length_total]) \
            + generate_trivial_dims(len(self.batch_dim) -1)
        self.input_dim_single = generate_trivial_dims(len(self.batch_dim)) + self.input_event_dim
        self.output_dim_single = generate_trivial_dims(len(self.batch_dim)) + self.output_event_dim
            
        
        # reshape data into flattened form
        self.input_dim_flat = self.batch_dim_flat + self.input_event_dim
        self.output_dim_flat = self.batch_dim_flat + self.output_event_dim
        self.input_data_flattened = torch.reshape(self.input_data, self.input_dim_flat.sizes) if self.input_data is not None else None
        self.output_data_flattened = torch.reshape(self.output_data, self.output_dim_flat.sizes)
        # input_data_flattened = torch.reshape(self.input_data, (self.total_batch_length,) + (self.input_batch_dim.sizes))
        # output_data_flattened = torch.reshape(self.output_data, (self.total_batch_length,) + (self.output_batch_dim.sizes))
        
        self.data = (self.input_data_flattened, self.output_data_flattened)
        

    def __len__(self):
        # return the length of the dataset, i.e. the number of independent event samples
        return self.batch_length_total
        

    def __getitem__(self, idx):
        # Check if idx is a tensor or list of indices
        if isinstance(idx, (torch.Tensor, list)):
            # # Handle the case where idx is a tensor or list of indices
            # if isinstance(idx, torch.Tensor):
            #     idx = idx.tolist()  # Convert tensor to list
            
            # input_data_batch = [self.data[0][i, ...].reshape(self.input_dim_single.sizes) if self.data[0] is not None else None for i in idx]
            # output_data_batch = [self.data[1][i, ...].reshape(self.output_dim_single.sizes) for i in idx]
    
            # # Stack the batch of data together, handling None values if necessary
            # input_data_batch = torch.cat([item for item in input_data_batch if item is not None]) if input_data_batch[0] is not None else None
            # output_data_batch = torch.cat(output_data_batch)
    
            # return (input_data_batch, output_data_batch, idx)
            pass
        
        else:
            # Handle the case where idx is a single integer
            input_data_idx = self.data[0][idx, ...].reshape(self.input_dim_single.sizes) if self.data[0] is not None else None
            output_data_idx = self.data[1][idx, ...].reshape(self.output_dim_single.sizes)
    
            return (input_data_idx, output_data_idx, idx)

        
        

shape_dict = {'batch_shape' : (n_meas,) ,\
              'input_event_shape' : (0,) ,\
              'output_event_shape' : (0,)
              }
dataset = CalipyDataset(input_data = input_data, output_data = output_data, shape_dict = shape_dict)



# Custom collate function to handle None values
def custom_collate(batch):
    batch_input, batch_output, indices = zip(*batch)
    
    # Keep None as is, or handle as required
    batch_input = [element for element in batch_input]
    batch_output = torch.cat(batch_output, dim=0)
    indices = torch.tensor(indices)
    
    return batch_input, batch_output, indices

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=n_subbatch, shuffle=True, collate_fn=custom_collate)


# Iterate through the DataLoader
for batch_input, batch_output, index in dataloader:
    print(batch_input, batch_output, index)




"""
    LOCAL CLASS IMPORTS
"""

class CalipyEffect(CalipyNode):
    """
    The CalipyEffect class provides a comprehensive representation of a specific 
    effect. It is named, explained, and referenced in the effect description. The
    effect is incorporated as a differentiable function based on torch. This function
    can depend on known parameters, unknown parameters, and random variables. Known 
    parameters have to be provided during invocation of the effect. During training,
    unknown parameters and the posterior density of the random variables is inferred.
    This requires providing a unique name, a prior distribution, and a variational
    distribution for the random variables.
    """
    
    
    def __init__(self, type = None, name = None, info = None):
        
        # Basic infos
        super().__init__(node_type = type, node_name = name, info_dict = info)
        
        
        self._effect_model = None
        self._effect_guide = None


class NoiseAddition(CalipyEffect):
    """ NoiseAddition is a subclass of CalipyEffect that produces an object whose
    forward() method emulates uncorrelated noise being added to an input. 

    :param node_structure: Instance of NodeStructure that determines the internal
        structure (shapes, plate_stacks, plates, aux_data) completely.
    :type node_structure: NodeStructure
    :return: Instance of the NoiseAddition class built on the basis of node_structure
    :rtype: NoiseAddition (subclass of CalipyEffect subclass of CalipyNode)
    
    Example usage: Run line by line to investigate Class
        
    .. code-block:: python
    
        # Investigate 2D noise ------------------------------------------------
        #
        # i) Imports and definitions
        import calipy
        from calipy.core.effects import NoiseAddition
        node_structure = NoiseAddition.example_node_structure
        noisy_meas_object = NoiseAddition(node_structure, name = 'tutorial')
        #
        # ii) Sample noise
        mean = torch.zeros([10,5])
        std = torch.ones([10,5])
        noisy_meas = noisy_meas_object.forward(input_vars = (mean, std))
        #
        # iii) Investigate object
        noisy_meas_object.dtype_chain
        noisy_meas_object.id
        noisy_meas_object.noise_dist
        noisy_meas_object.node_structure.description
        noisy_meas_object.plate_stack
        render_1 = noisy_meas_object.render((mean, std))
        render_1
        render_2 = noisy_meas_object.render_comp_graph((mean, std))
        render_2
    """
    
    
    # Initialize the class-level NodeStructure
    example_node_structure = NodeStructure()
    example_node_structure.set_shape('batch_shape', (10, 5), 'Batch shape description')
    example_node_structure.set_shape('event_shape', (2, 3), 'Event shape description')
    # example_node_structure.set_shape('event_shape', (2, ), 'Event shape description')
    # example_node_structure.set_plate_stack('noise_stack', [('batch_plate_1', 5, -2, 'plate denoting independence in row dim'),
    #                                                           ('batch_plate_2', 10, -1, 'plate denoting independence in col dim')],
    #                                        'Plate stack for noise ')

    # Class initialization consists in passing args and building shapes
    def __init__(self, node_structure, **kwargs):
        super().__init__(**kwargs)
        self.node_structure = node_structure
        
        self.batch_dims = dim_assignment(dim_names = ['batch_dim'], dim_shapes = self.node_structure.shapes['batch_shape'])
        self.event_dims = dim_assignment(dim_names = ['event_dim'], dim_shapes = self.node_structure.shapes['event_shape'])
        self.full_dims = self.batch_dims + self.event_dims
    
    def plate_stack_from_shape(self, plate_stack_name, dim_tuple, stack_description = None):
        # This function could be part of the node_structure. It is supposed to
        # create a plate_stack with a certain name based on some dimensions.
        # It takes the batch dimensions, finds their location w.r.t. all dimensions
        # and sets the appropriate plate stack
        dim_name_list = dim_tuple.names
        dim_size_list = dim_tuple.sizes
        dim_loc_list = self.full_dims.find_indices(dim_name_list)
        dim_doc_list = dim_tuple.descriptions
        
        plate_data_list = [(name, size, loc, doc) for name, size, loc, doc in
                           zip(dim_name_list, dim_size_list, dim_loc_list, dim_doc_list)]
        self.node_structure.set_plate_stack(plate_stack_name, plate_data_list, stack_description)
        
    
    # Forward pass is passing input_vars and sampling from noise_dist
    def forward(self, input_vars, observations = None):
        """
        Create noisy samples using input_vars = (mean, standard_deviation) with
        shapes as indicated in the node_structures' plate_stack 'noise_stack' used
        for noisy_meas_object = NoiseAddition(node_structure).
        
        :param input vars: 2-tuple (mean, standard_deviation) of tensors with 
            equal (or at least broadcastable) shapes. 
        :type input_vars: 2-tuple of instances of torch.Tensor
        :return: Tensor representing simulation of a noisy measurement of the mean.
        :rtype: torch.Tensor
        """
        
        batch_dims = self.batch_dims.get_local_copy()
        event_dims = self.event_dims.get_local_copy()
        full_dims = batch_dims + event_dims
        
        mean_fd = input_vars[0][full_dims]
        # mean_ordered = mean_fd.order(*full_dims)
        
        self.plate_stack_from_shape('noise_stack', batch_dims, 'Plate stack for noise')
        self.noise_stack = self.node_structure.plate_stacks['noise_stack']
        
        self.noise_dist = pyro.distributions.Normal(loc = input_vars[0], scale = input_vars[1])
        
        # Sample within independence context
        with context_plate_stack(self.noise_stack):
            output = pyro.sample('{}__noise_{}'.format(self.id_short, self.name), self.noise_dist, obs = observations)
        return output


class CalipyQuantity(CalipyNode):
    """
    The CalipyQuantity class provides a comprehensive representation of a specific 
    quantity used in the construction of a CalipyEffect object. This could be a
    known parameter, an unknown parameter, or a random variable. This quantity
    is named, explained, and referenced in the quantity description. Quantities
    are incorporated into the differentiable function that define the CalipyEffect
    forward pass. Each quantity is subservient to an effect and gets a unique id
    that reflects this, quantities are local and cannot be shared between effects.
    """
    
    def __init__(self, type = None, name = None, info = None):
        
        # Basic infos
        super().__init__(node_type = type, node_name = name, info_dict = info)
        


class UnknownParameter(CalipyQuantity):
    """ UnknownParameter is a subclass of CalipyQuantity that produces an object whose
    forward() method produces a parameter that is subject to inference.

    :param node_structure: Instance of NodeStructure that determines the internal
        structure (shapes, plate_stacks, plates, aux_data) completely.
    :type node_structure: NodeStructure
    :param constraint: Pyro constraint that constrains the parameter of a distribution
        to lie in a pre-defined subspace of R^n like e.g. simplex, positive, ...
    :type constraint: pyro.distributions.constraints.Constraint
    :return: Instance of the UnknownParameter class built on the basis of node_structure
    :rtype: UnknownParameter (subclass of CalipyQuantity subclass of CalipyNode)
    
    Example usage: Run line by line to investigate Class
        
    .. code-block:: python
    
        # Investigate 2D bias tensor -------------------------------------------
        #
        # i) Imports and definitions
        import calipy
        from calipy.core.effects import UnknownParameter
        node_structure = UnknownParameter.example_node_structure
        bias_object = UnknownParameter(node_structure, name = 'tutorial')
        #
        # ii) Produce bias value
        bias = bias_object.forward()
        #
        # iii) Investigate object
        bias_object.dtype_chain
        bias_object.id
        bias_object.node_structure.description
        render_1 = bias_object.render()
        render_1
        render_2 = bias_object.render_comp_graph()
        render_2
    """
    
    
    # Initialize the class-level NodeStructure
    example_node_structure = NodeStructure()
    example_node_structure.set_shape('batch_shape', (10, ), 'Batch shape description')
    example_node_structure.set_shape('event_shape', (5, ), 'Event shape description')

    # Class initialization consists in passing args and building shapes
    def __init__(self, node_structure, constraint = constraints.real, **kwargs):  
        # The whole setup is entirely possible if we instead of shapes use dims
        # and e.g. set_dims(dims = 'batch_dims', dim_names = ('batch_dim_1', 'batch_dim_2'), dim_sizes = (10, 7))
        # maybe it is also possible, to set a 'required' flag for some of these
        # quantities and have this info pop up as class attribute.
        super().__init__(**kwargs)
        self.node_structure = node_structure
        self.batch_shape = self.node_structure.shapes['batch_shape']
        self.event_shape = self.node_structure.shapes['event_shape']
        self.param_shape = self.node_structure.shapes['param_shape']
        
        self.constraint = constraint
        
        self.batch_dims = dim_assignment(dim_names =  ['batch_dim'], dim_shapes = self.batch_shape)
        self.event_dims = dim_assignment(dim_names =  ['event_dim'], dim_shapes = self.event_shape)
        self.param_dims = dim_assignment(dim_names =  ['param_dim'], dim_shapes = self.param_shape)
        self.trivial_dims_param = generate_trivial_dims(len(self.param_dims)) 
        self.full_tensor_dims = self.batch_dims + self.param_dims + self.event_dims
        
    # Forward pass is initializing and passing parameter
    def forward(self, input_vars = None, observations = None):

        # Conversion to extension tensor
        extension_tensor_dims = self.batch_dims + self.trivial_dims_param + self.event_dims
        extension_tensor = torch.ones(extension_tensor_dims.sizes)[extension_tensor_dims]
        extension_tensor_ordered = extension_tensor.order(*self.trivial_dims_param)
        
        # initialize param
        self.init_tensor = torch.ones(self.param_dims.sizes)
        self.param = pyro.param('{}__param_{}'.format(self.id_short, self.name), init_tensor = self.init_tensor, constraint = self.constraint)
        
        # extend param tensor
        param_tensor_extended_ordered = self.param*extension_tensor_ordered
        param_tensor_extended_fd = param_tensor_extended_ordered[self.param_dims]
        self.extended_param = param_tensor_extended_fd.order(*self.full_tensor_dims)
        
        return self.extended_param


class ShapeExtension(CalipyEffect):
    """ 
    Shape extension class takes as input some tensor and repeats it multiple
    times such that in the end it has shape batch_shape + original_shape + event_shape
    """
    
    # Initialize the class-level NodeStructure
    # example_node_structure = NodeStructure()
    example_node_structure = None
    # example_node_structure.set_shape('batch_shape', (10, ), 'Batch shape description')
    # example_node_structure.set_shape('event_shape', (5, ), 'Event shape description')

    # Class initialization consists in passing args and building shapes
    def __init__(self, node_structure = None, **kwargs):
        super().__init__(**kwargs)
        self.node_structure = node_structure
        # self.batch_dims = dim_assignment(dim_names = ['batch_dim'], dim_shapes = self.node_structure.shapes['batch_shape'])
        # self.event_dims = dim_assignment(dim_names = ['event_dim'], dim_shapes = self.node_structure.shapes['event_shape'])
        
    
    # Forward pass is passing input_vars and extending them by broadcasting over
    # batch_dims (left) and event_dims (right)
    def forward(self, input_vars, observations = None):
        """
        input_vars = (tensor, batch_shape, event_shape)
        """
        
        # Fetch and distribute arguments
        tensor, batch_shape, event_shape = input_vars
        batch_dim = dim_assignment(dim_names =  ['batch_dim'], dim_shapes = batch_shape)
        event_dim = dim_assignment(dim_names =  ['event_dim'], dim_shapes = event_shape)
        tensor_dim = dim_assignment(dim_names =  ['tensor_dim'], dim_shapes = tensor.shape)

        # compute the extensions
        batch_extension_dims = batch_dim + generate_trivial_dims(len(tensor.shape) + len(event_shape))
        event_extension_dims = generate_trivial_dims(len(batch_shape) + len(tensor.shape)) +event_dim
        
        batch_extension_tensor = torch.ones(batch_extension_dims.sizes)
        event_extension_tensor = torch.ones(event_extension_dims.sizes)
        
        extended_tensor = batch_extension_tensor * tensor * event_extension_tensor
        output =  extended_tensor
        return output


"""
    3. Load and customize effects
"""


# i) Set up dimensions for mean parameter mu

# Setting up requires correctly specifying a NodeStructure object. You can get 
# a template for the node_structure by calling generate_template() on the example
# node_structure delivered with the class description.
# Here we modify the output of UnknownParam.example_node_structure.generate_template()

# mu setup
mu_ns = NodeStructure()
mu_ns.set_shape('batch_shape', (), 'Independent values')
mu_ns.set_shape('param_shape', (1,), 'Parameter values')
mu_ns.set_shape('event_shape', (), 'Repeated values')
mu_object = UnknownParameter(mu_ns, name = 'mu')


# iii) Set up the dimensions for noise addition
# This requires not batch_shapes and event shapes but plate stacks instead - these
# quantities determine conditional independence for stochastic objects. In our 
# case, everything is independent since we prescribe i.i.d. noise.
# Here we modify the output of NoiseAddition.example_node_structure.generate_template()
noise_ns = NodeStructure()
noise_ns.set_shape('batch_shape', (None,), 'Batch shape description')
noise_ns.set_shape('event_shape', (1,), 'Event shape description')
noise_object = NoiseAddition(noise_ns)

# iv) Set up the shape_extension
shape_extension_object = ShapeExtension()




"""
    4. Build the probmodel
"""


# i) Define the probmodel class 

class DemoProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # integrate nodes
        self.mu_object = mu_object
        # Here there should be an extension object
        self.shape_extension = shape_extension_object
        
        self.noise_object = noise_object 
        
    # Define model by forward passing
    def model(self, input_vars = None, observations = None):
        mu = self.mu_object.forward()       
        
        obs_batch_shape = (observations.shape[0],) if observations is not None else (n_meas,)
        
        # This here should be done by an extension object instead ideally
        mu_extension_batch_shape = obs_batch_shape
        mu_extension_event_shape = ()
        mu_extended = self.shape_extension.forward(input_vars = (mu, mu_extension_batch_shape, mu_extension_event_shape))
        
        
        output = self.noise_object.forward((mu_extended, sigma_true), observations = observations)     
        
        return output
    
    # Define guide (trivial since no posteriors)
    def guide(self, input_vars = None, observations = None):
        pass
    
demo_probmodel = DemoProbModel()
    



"""
    5. Perform inference
"""
    

# i) Set up optimization

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
n_steps = 1000

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps}


# ii) Train the model

# input_data = None
# output_data = data
dataloader_svi = dataloader
optim_results = demo_probmodel.train(dataloader = dataloader_svi, optim_opts = optim_opts)
    


"""
    6. Analyse results and illustrate
"""


# i)  Plot loss

plt.figure(1, dpi = 300)
plt.plot(optim_results)
plt.title('ELBO loss')
plt.xlabel('epoch')

# ii) Print  parameters

for param, value in pyro.get_param_store().items():
    print(param, '\n', value)
    
print('True values of mu = ', mu_true)
print('Results of taking empirical means for mu_1 = ', torch.mean(output_data))






















