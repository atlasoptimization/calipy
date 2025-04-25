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
import random
from pyro.distributions import constraints
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functorch.dim import dims

# calipy
import calipy
from calipy.core.base import CalipyNode, NodeStructure, CalipyProbModel
from calipy.core.utils import dim_assignment, generate_trivial_dims, context_plate_stack


# ii) Definitions

n_data_1 = 100
n_data_2 = 100
# n_data_total = n_data_1 + n_data_2
torch.manual_seed(1)
pyro.set_rng_seed(1)



"""
    2. Generate data
"""


# i) Set up data distribution (=standard normal))

mu_1_true = torch.zeros([1])
mu_2_true = torch.zeros([1])
sigma_true = torch.ones([1])

extension_tensor_1 = torch.ones([n_data_1])
extension_tensor_2 = torch.ones([n_data_2])
# extension_tensor_total = torch.ones([n_data_total])
data_dist_1 = pyro.distributions.Normal(loc = mu_1_true * extension_tensor_1, scale = sigma_true)
data_dist_2 = pyro.distributions.Normal(loc = mu_2_true * extension_tensor_2, scale = sigma_true)

# ii) Sample from dist to generate data

# Generate data
data_1 = data_dist_1.sample()
data_2 = data_dist_2.sample()
data_tensortuple = (data_1, data_2)


def flatten_tensortuple(tensortuple):
    # Takes as input a tuple of tensors (data_1, data_2, ...) and produces a list
    # [(data_11, data_12 ,...) , ... (data_n1, data_n2, ...)] where the length is
    # determined by the longest tensor and slicing is done on the first dim of each
    # tensor. If tensor lengths are not equal, None are used to fill up undersized
    # tensors. This function is useful to enable subbatching when multiple independent 
    # input and output tensors are given and need to be subbatched.
    
    # basic setup
    n_tuple = len(tensortuple)
    tensor_shapes = [tensor.shape for tensor in tensortuple]
    batch_lengths = [shape[0] for shape in tensor_shapes]
    
    # infer augmentation
    max_batch_length = np.max(batch_lengths)
    augmentation_nones = [[None]*(max_batch_length - bl) for bl in batch_lengths]
    
    # build list of lists
    list_of_lists = []
    for k_tuple in range(n_tuple):
        list_k = []
        for l_batch in range(max_batch_length):
            tuple_k_entry_l = tensortuple[k_tuple][l_batch, ...] if l_batch < batch_lengths[k_tuple] else None
            list_k.append(tuple_k_entry_l)
        # random.shuffle(list_k)
        list_of_lists.append(list_k)
    
    # compile to datalist
    datalist = []
    for l_batch in range(max_batch_length):
        entry_datalist = []
        for k_tuple in range(n_tuple):
            entry_datalist.append(list_of_lists[k_tuple][l_batch])
        datalist.append(tuple(entry_datalist))
    
    return datalist

# # Try it out
# tensortuple_ex = (torch.randn([2,2]), torch.ones([3]), torch.zeros([1]))
# datalist_ex = tensortuple_to_datalist(tensortuple_ex)

datalist = flatten_tensortuple(data_tensortuple)



# ii) Definitions

n_subbatch = 3

output_data = datalist
input_data = None

# The data now is a list with n_data_total entries and each entry is a tuple of
# the form (data_1, data_2) where data_1, data_2 are tensors representing a
# measurement or None representing the absence of measurement.

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.


# helper function to cat also none tensors
def safe_cat(tensor_list, dim=0):
    
    # Filter out None values
    filtered_tensors = [t for t in tensor_list if t is not None]
    
    # Concatenate the remaining tensors
    if filtered_tensors:
        return torch.cat(filtered_tensors, dim=dim)
    else:
        return None  # or raise an exception if needed

safe_cat.__doc__ = 'A torch.cat derivative that handles entries of None in the tensor list'\
    'by omitting them.' + torch.cat.__doc__


def reduce_nones(input_list):
    # reduces a list containing Nones by eliminating the Nones and replaces a
    # full None list [None, None, ...] with None
    reduced_list = [element for element in input_list if element is not None]
    reduced_list = reduced_list if not len(reduced_list) == 0 else None
    
    return reduced_list



# Build CalipyDataset class that allows DataLoader to batch over multiple dimensions

class CalipyDataset(Dataset):
    def __init__(self, input_data, output_data, dataset_type = 'tensor'):
        # dataset type can be 'tensor', 'tensortuple', or 'tensortuplelist'
        # - default is tensor which means input_data and output_data are tensors,
        # e.g. output_data = torch.randn([2,2])
        # dims: first dimenstion is assumed to be batch dimension and needs to
        # be consistent for input_data and output_data to allow subbatching.
        # - tensortuple allows input_data, output_data to be a tuple of tensors,
        # e.g. output_data = (torch.randn([2,2]), torch.ones([3]))
        # dims: the output_data is considered to be a single event and will be
        # handed to inference algorithms. Subbatching therefore cannot be done
        # without assumptions. It is possible though by considering each first
        # dimension of each tensor to be indexing an independent datapoint and converting
        # to tensortuplelist, which assumes the list index to be the batching index
        # - tensortuplelist allows input_data, output_data to be  a list
        # [(data_11, data_12 ,...) , ... (data_n1, data_n2, ...)] 
        # dims: each list entry is a tuple of tensors interpreted as a single
        # event and the list index is the batching index. Nr of tuples in list
        # input_data and list_output_data need to be identical.
        #
        # We have the following order of generality: tensor < tensortuple < tensortuplelist
        # tensor can be converted to  a tensortuple in an unambiguous way
        #   tensor -> (tensor,)
        # tensortuple can be converted to tensortuplelist in an unambiguous way
        #   (tensor_1, tensor_2) -> [(tensor_1, tensor_2)]
        # The converse operations do not exist in all generality. Tensortuple can
        # be inverted to tensor only in the trivial case of one tensor in the tuple.
        # Tensortuplelist can be converted to tensortuple by assuming independence
        # of the elements in the tuple and concatenating corresponding elements in
        # the tuple along the batch dimension - which only works if they all have 
        # the same shape (or are None).
        #
        # Furthermore, it is possible to perform subbatching via a DataLoader
        # by calling the flatten() method.
        # For tensors, it flattens along the batch_dimensions
        # For tensortuples, it produces a tensortuplelist assuming independence
        # between tuple elements, filling them up with Nones and shuffling them
        # For tensortuplelists, they are interpreted as independent data points
        # indexed by the list index.
        
        self.valid_dataset_types = ['tensor', 'tensortuple', 'tensortuplelist']
        self.dataset_type = dataset_type
        if dataset_type not in self.valid_dataset_types :
            raise(Exception('dataset type must be one of {}; is {}'.format(self.valid_dataset_types, dataset_type)))
        
        self.data = (input_data, output_data)
        self.input_data = input_data
        self.output_data = output_data
        self.len_input_data = self.infer_length(self.input_data, self.dataset_type)
        self.len_output_data = self.infer_length(self.output_data, self.dataset_type)

        self.flattened_data = self.flatten()
    
    def infer_length(self, query_data, dataset_type):
        # Infer the batch length of input_data or output_data
        if query_data is not None:
            if self.dataset_type == 'tensor':
                len_data = query_data.shape[0]
            elif self.dataset_type == 'tensortuple':
                len_data = query_data[0].shape[0]
            elif self.dataset_type == 'tensortuplelist':
                len_data = len(query_data)
        else:
            len_data = 0
        
        
        return len_data
        
    def flatten(self):
        # Independent of the dataset_type, this function returns a list of tuples
        # of tensors, a tensortuplelist where the list index indexes individual
        # datapoints of the batch.
        # In all cases, the first dimension of the tensor or a list is assumed
        # as the batch dimension
        
        input_data_flattened = []
        output_data_flattened = []
        
        if self.dataset_type == 'tensor' :
            # Convert tensor input_data, output_data to list of tensors with
            # each entry of the list representing an independent data point
            
            for k in range(self.len_input_data):
                input_data_flattened.append((self.input_data[k,...].unsqueeze(0),))
            input_data_flattened = None if len(input_data_flattened) == 0 else input_data_flattened
            for k in range(self.len_output_data):
                output_data_flattened.append((self.output_data[k,...].unsqueeze(0),))
            
            
        elif self.dataset_type == 'tensortuple':
            # Convert tensortuples input_data, output data to list of tensors with
            # each entry of the list representing an independent data point
            
            for k in range(self.len_input_data):
                input_data_flattened.append(tuple([ tensor[k,...].unsqueeze(0) for tensor in self.input_data ]))
            input_data_flattened = None if len(input_data_flattened) == 0 else input_data_flattened
            for k in range(self.len_output_data):
                output_data_flattened.append((tuple([ tensor[k,...].unsqueeze(0) for tensor in self.output_data ])))
            
            
        elif self.dataset_type == 'tensortuplelist':
            input_data_flattened = self.input_data
            output_data_flattened = self.output_data
        
        
        self.input_data_flattened = input_data_flattened
        self.output_data_flattened = output_data_flattened
        flattened_data = (self.input_data_flattened, self.output_data_flattened)
        return flattened_data
        
        
        
        # self.batch_dim = dim_assignment(['batch_dim'], dim_shapes = shape_dict['batch_shape'])
        
        # # input data
        # self.input_data = input_data  # Assuming data is a list or array of samples
        # self.input_event_dim = dim_assignment(['input_event_dim'], dim_shapes = shape_dict['input_event_shape'])
        # self.input_dim = self.batch_dim + self.input_event_dim
        
        # # output_data
        # self.output_data = output_data  # Assuming data is a list or array of samples
        # self.output_event_dim = dim_assignment(['output_event_dim'], dim_shapes = shape_dict['output_event_shape'])
        # self.output_dim = self.batch_dim + self.output_event_dim
        
        # # reshaping to single batch_dim
        # self.batch_length_total = math.prod(self.batch_dim.sizes)
        # self.batch_dim_flat = dim_assignment(['total_batch_dim'], dim_shapes= [self.batch_length_total]) \
        #     + generate_trivial_dims(len(self.batch_dim) -1)
        # self.input_dim_single = generate_trivial_dims(len(self.batch_dim)) + self.input_event_dim
        # self.output_dim_single = generate_trivial_dims(len(self.batch_dim)) + self.output_event_dim
            
        
        # # reshape data into flattened form
        # self.input_dim_flat = self.batch_dim_flat + self.input_event_dim
        # self.output_dim_flat = self.batch_dim_flat + self.output_event_dim
        # self.input_data_flattened = torch.reshape(self.input_data, self.input_dim_flat.sizes) if self.input_data is not None else None
        # self.output_data_flattened = torch.reshape(self.output_data, self.output_dim_flat.sizes)
        
        # self.data = (self.input_data_flattened, self.output_data_flattened)
        

    def __len__(self):
        # return the length of the dataset, i.e. the number of independent event samples
        # return self.batch_length_total
        return  self.len_output_data
        

    def __getitem__(self, idx):
        # print(idx)
        # Handle the case where idx is a single integer
        input_data_idx = self.input_data_flattened[idx] if self.input_data_flattened is not None else None
        output_data_idx = self.output_data_flattened[idx]

        return (input_data_idx, output_data_idx, idx)

        
        

# shape_dict = {'batch_shape' : (n_data_total,) ,\
#               'input_event_shape' : (0,) ,\
#               'output_event_shape' : (0,)
#               }
# dataset = CalipyDataset(input_data = input_data, output_data = output_data, shape_dict = shape_dict)


# Experimentation with flattening

# Standard tensor setting with first dim = batch_dim
none_input = None
tensor_input = torch.randn([2,1,4]) # Autointerpreted as single batch dim = 2, rest event dims
tensor_output = torch.randn([2,3,5]) # same interpretation

ds_1 = CalipyDataset(none_input, tensor_output, dataset_type = 'tensor')
# This produced ds_1.flattened_data as tuple (None, tensortuplelist)
# tensortuplelist is list with two elements and each element is tuple containing a
# 1,3,5 tensor. 
ds_2 = CalipyDataset(tensor_input, tensor_output, dataset_type = 'tensor')
# This produces flattened input data as a list with 2 elements of shape [1,1,4]
# flattened output data is a list with 2 elements of shape [1,3,5]

# Tensortuple setting with first tensor dims = batch_dim
tensortuple_input = (torch.randn([3,2]) , torch.ones([3,5,5]))
tensortuple_output = (torch.randn([3,2,2]), torch.ones([3,4,4]), torch.zeros([3,1]))

ds_3 = CalipyDataset(none_input, tensortuple_output, dataset_type = 'tensortuple')
# Produces a flattened output list with 3 elements, each entry is a tuple containing
# three tensors; they are of shapes  ([1,2,2], [1,4,4], [1,1])
ds_4 = CalipyDataset(tensortuple_input, tensortuple_output, dataset_type = 'tensortuple')
# Produces a flattened input list with 2 elements, each entry is a tuple containing
# two tensors; they are of shapes  ([1,2], [1,5,5])
# Produces the same flattened output as ds_3


# Tensortuplelist setting with list index denoting batch index
tensortuplelist_input = [(torch.tensor(1.0), torch.ones([2,2]), torch.randn([3,2])), 
                         (torch.zeros([1,2]), torch.tensor(2.3), torch.tensor(0.5))]
tensortuplelist_output = [(None, 2.2*torch.ones([2,2]), torch.randn([3,2])), 
                         (torch.zeros([1,2]), None, None)]

ds_5 = CalipyDataset(none_input, tensortuplelist_output, dataset_type = 'tensortuplelist')
ds_6 = CalipyDataset(tensortuplelist_input, tensortuplelist_output, dataset_type = 'tensortuplelist')
# Produces input_data_flattend as list with two elements, each element is a
# tuple with three elements. The same holds for output_data_flattened. The entries
# in those tuples do not necessarily match shapewise and some are simply None.



# Custom collate function that collates tensors by concatenating them among the 
# first dimension. Allows vectorization. Only possible when batch shapes are 
# consistent and event shapes for each element are independent of batch

# Input to the collate are assumed to be tensortuplelists
def tensor_collate(batch):
    
    batch_input, batch_output, indices = zip(*batch)
    reduced_batch_input = reduce_nones(batch_input)
    
    len_input = len(batch_input) if reduced_batch_input is not None else 0 
    len_input_tuple = len(batch_input[0]) if reduced_batch_input is not None else 0
    len_output = len(batch_output)
    len_output_tuple = len(batch_output[0])
    
    # Check if vectorizable
    batch_input_shapes = []
    batch_output_shapes = []
    
    # for k_in in range(len_input):
    #     input_shapes = [bi.shape for bi in batch_input[k_in]]
    #     batch_input_shapes.append(input_shapes)
    # for k_out in range(len_output):
    #     output_shapes = [bo.shape for bo in batch_output[k_out]]
    #     batch_output_shapes.append(output_shapes)
        
    batch_input_shapes = [[bi.shape for bi in batch_input[k]] for k in range(len_input)]
    batch_output_shapes = [[bo.shape for bo in batch_output[k]] for k in range(len_output)]
    input_consistency = [x == batch_input_shapes[0] for x in batch_input_shapes]
    output_consistency = [x == batch_output_shapes[0] for x in batch_output_shapes]
    batch_consistency = len_input == len_output if reduced_batch_input is not None else True
    
    if input_consistency is False:
        raise(Exception('The batch input has shapes that disallow vectorization. Need to be equal'\
                        ' for each tuple but are {}'.format(batch_input_shapes)))
    if output_consistency is False:
        raise(Exception('The batch output has shapes that disallow vectorization. Need to be equal'\
                        ' for each tuple but are {}'.format(batch_output_shapes)))
    if batch_consistency is False:
        raise(Exception('The number of list elements is inconsistent between input and' \
                        ' output batch. Should be equal but are len(batch_input) ={}' \
                        ' len(batch_output) = {}. Alternatively, input may be None'
                        .format(len_input, len_output)))
    
    
    
    # Keep None as is, or handle as required
    # collated_batch_input = tuple([torch.cat(bi, dim = 0) for bi in batch_input]) if reduced_batch_input is not None else None
    # collated_batch_output = tuple([torch.cat(bo, dim=0) for bo in batch_output])
    
    collated_batch_input = []
    collated_batch_output = []
    for k_inputs in range(len_input_tuple):
        input_batch_list = []
        for k_samples in range(len_input):
            input_batch_list.append(batch_input[k_samples][k_inputs])
        input_batch_tensor = torch.cat(input_batch_list, dim = 0)
        collated_batch_input.append(input_batch_tensor)
    collated_batch_input = collated_batch_input if not len(collated_batch_input) ==0 else None
    
    for k_outputs in range(len_output_tuple):
        output_batch_list = []
        for k_samples in range(len_output):
            output_batch_list.append(batch_output[k_samples][k_outputs])
        output_batch_tensor = torch.cat(output_batch_list, dim = 0)
        collated_batch_output.append(output_batch_tensor)
        
            
    #         collated_batch_input.append([torch.cat([bi[k_inputs] for bi in batch_input], dim = 0)])
    # for k_outputs in range(len_output_tuple):
    #     collated_batch_output.append([torch.cat([bo[k_outputs] for bo in batch_output], dim = 0)])
    
    # indices = torch.tensor(indices)
    
    return collated_batch_input, collated_batch_output, indices

# This collate handles tensortuplelists that are coming from sequentially calling
# the CalipyDataset.__getitem__() method. The lists are handed back verbatim.
def list_collate(batch):
    batch_input, batch_output, indices = zip(*batch)    
    return batch_input, batch_output, indices


# Create a DataLoader
dataset_via_tensortuple = CalipyDataset(None, data_tensortuple, dataset_type = 'tensortuple')
dataset_via_tensortuplelist = CalipyDataset(input_data, output_data, dataset_type = 'tensortuplelist')

dataloader = DataLoader(dataset_via_tensortuple, batch_size=n_subbatch, shuffle=True, collate_fn=tensor_collate)


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
        dim_name_list = dim_tuple.names #['{}_{}'.format(self.id_short,dim_name) for dim_name in dim_tuple.names]
        dim_size_list = dim_tuple.sizes
        dim_loc_list = self.full_dims.find_indices(dim_name_list)
        dim_doc_list = dim_tuple.descriptions
        
        plate_data_list = [(self.id_short + name, size, loc, doc) for name, size, loc, doc in
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
noise_object_1 = NoiseAddition(noise_ns)
noise_object_2 = NoiseAddition(noise_ns)

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
        
        self.noise_object_1 = noise_object_1 
        self.noise_object_2 = noise_object_2 
        
    # Define model by forward passing
    def model(self, input_vars = None, observations = None):
        mu = self.mu_object.forward()       
        
        obs_batch_shape = (observations[0].shape[0],) if observations is not None else (n_data_1,)
        
        # This here should be done by an extension object instead ideally
        mu_extension_batch_shape = obs_batch_shape
        mu_extension_event_shape = ()
        mu_extended = self.shape_extension.forward(input_vars = (mu, mu_extension_batch_shape, mu_extension_event_shape))
        
        output_1 = self.noise_object_1.forward((mu_extended, sigma_true), observations = observations[0] if observations is not None else None)     
        output_2 = self.noise_object_2.forward((mu_extended, sigma_true), observations = observations[1] if observations is not None else None)     
        
        output = (output_1, output_2) 
        
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
n_steps = 100

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps, 'n_steps_report' : 10}


# ii) Train the model

# input_data = None
output_data = data_tensortuple
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
    
print('True values of mu = ', mu_1_true)
print('Results of taking empirical means for mu_1 = ', torch.mean(output_data[0]),torch.mean(output_data[1]))
print('Everything averaged together = ' , torch.mean(torch.cat((output_data[0], output_data[1]), dim = 0)))






















