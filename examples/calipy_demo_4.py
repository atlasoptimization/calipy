#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to perform subbatched inference on
a model a simple measurement process with unknown mean and known variance. The
measurement procedure has been used to collect a single dataset, that features
n_meas samples. Inference is performed to estimate the value of the underlying
expected value.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data, enable subbatching
    3. Load and customize effects
    4. Build the probmodel
    5. Perform inference
    6. Analyse results and illustrate

The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    1. Imports and definitions
"""


# i) Imports

# base packages
import torch
import pyro
import math
import matplotlib.pyplot as plt

# calipy
import calipy
from calipy.core.base import NodeStructure, CalipyProbModel
from calipy.core.effects import UnknownParameter, NoiseAddition
from calipy.core.utils import dim_assignment
from calipy.core.data import DataTuple, CalipyDict, CalipyIO
from calipy.core.tensor import CalipyTensor, CalipyIndex
from calipy.core.funs import calipy_cat


# Todel
from torch.utils.data import Dataset, DataLoader
from calipy.core.utils import CalipyDim


# ii) Definitions

n_meas = 10
n_event = 1
n_subbatch = 7


# # # CLASSES FOR INJECTION INTO DATA 
# # PROVIDE FUNCTIONALITY FOR DATASETS AND SUBBATCHING; EXPORT



# Build CalipyDataset class that allows DataLoader to batch over multiple dimensions

class CalipyDataset(Dataset):
    """
    CalipyDataset is a class mimicking the functionality of the Dataset class in
    torch.utils.data but providing some streamlined prebuilt functions needed
    in the context of calipy. This includes support for subsampling based on 
    CalipyDict objects. Is meant to be subclassed for augmenting user specified
    datasets with additional, calipy-ready functionality.    
    
    :param input_data: The input_data of the dataset reflecting the inputs to 
        the model that evoke the corresponding outputs. Can be:
          - None => No input data (no input)
          - CalipyTensor => Single tensor (single input)
          - CalipyDict => Dictionary containing CalipyTensors (multiple inputs)
          - CalipyIO => List containing CalipyDict containing CalipyTensors 
              (multiple inputs, possibly of inhomogeneous shape and type)
    :type input_data: NoneType, CalipyTensor, CalipyDict, CalipyIO
    
    :param output_data: The output_data of the dataset reflecting the outputs of 
        the model evoked by the corresponding inputs. Can be:
          - None => No output data (no output)
          - CalipyTensor => Single tensor (single output)
          - CalipyDict => Dictionary containing CalipyTensors (multiple outputs)
          - CalipyIO => List containing CalipyDict containing CalipyTensors 
              (multiple inputs, possibly of inhomogeneous shape and type)
    :type output_data: NoneType, CalipyTensor, CalipyDict, CalipyIO
    :param batch_dims: A DimTuple object defining the batch dimensions among 
        which flattening and subsampling is performed.
    :type batch_dims: DimTuple
    
    :return: An instance of CalipyDataset, suitable for accessing datasets and
        passing them to DataLoader objects.
    :rtype: CalipyDataset

    Example usage:

    .. code-block:: python
        
        # Imports and definitions
        import torch
        from calipy.core.data import CalipyDataset
           

        # Create data for CalipyDict initialization

    
    """
    def __init__(self, input_data, output_data, batch_dims, homogeneous = False):
        
        # Preprocess I/O data
        self.batch_dims = batch_dims
        self.input_type = type(input_data)
        self.output_type = type(output_data)
        self.input_data = CalipyIO(input_data)
        self.output_data = CalipyIO(output_data)
        self.data = {'input_data' : input_data, 'output_data' :  output_data}
        
        # Error diagnostics
        self.valid_dataset_types = [CalipyTensor, CalipyDict, CalipyIO, type(None)]
        if self.input_type not in self.valid_dataset_types :
            raise(Exception('input_type must be one of {}; is {}'.format(self.valid_dataset_types, self.input_type)))
        if self.output_type not in self.valid_dataset_types :
            raise(Exception('output_type must be one of {}; is {}'.format(self.valid_dataset_types, self.output_type)))
        
        # Lengths and flattened data
        self.flattened_input = self.flatten(self.input_data)
        self.flattened_output = self.flatten(self.output_data)
        self.len_input_data = self.infer_length(self.flattened_input)
        self.len_output_data = self.infer_length(self.flattened_output)
    
    
    def infer_length(self, query_data):
        # Infer the batch length of input_data or output_data
        len_data = {key: query_data[key].shape[0] if query_data[key] is not None else None 
                    for key in query_data.keys()}
        return len_data
        
    def flatten(self, io_data):
        # This function returns a CalipyIO of tensors, where the first dimension
        # is the (only) batch dimension and for each key in calipy_dict.keys(),
        # CalipyDict[key][k, ...] is the kth datapoint in the dataset. The input
        # arg io_data is a CalipyDict of CalipyTensors.
        
        # i) Do the flattening
        data_flattened = {}
        for key, value in io_data.items():
            data_flattened[key] = value.flatten(self.batch_dims, 'batch_dim_flattened') if value is not None else None
        self.batch_dim_flattened = dim_assignment(['batch_dim_flattened'])
        
        # ii) Check if flattening consistent
        batch_dims_sizes = {key : value.dims[['batch_dim_flattened']].sizes if value is not None else [] 
                             for key, value in data_flattened.items()}
        first_dim_size = batch_dims_sizes[list(batch_dims_sizes.keys())[0]]
        if not all([dim_sizes == first_dim_size for key, dim_sizes in batch_dims_sizes.items()]):
            raise(Exception('For flattening, all DimTuples batch_dims must be of ' \
                            'same size for all keys but are {} for keys {}'.format(batch_dims_sizes,list(batch_dims_sizes.keys()))))
        
        self.batch_length = first_dim_size
        return CalipyDict(data_flattened)
    
        # data_flattened = {}
        # empty_dims = dim_assignment(['empty_dim'], dim_sizes = [])
        
        # # i) Extract the actual sizes of the batch dims and other dims
        # data_dims_bound = {key : value.indexer.local_index.bound_dims[0:-1] if value is not None
        #                 else empty_dims for key, value in io_data.items()}
        # batch_dims_dict = {key : value[self.batch_dims] if value is not None 
        #                    else empty_dims for key, value in data_dims_bound.items()}
        # other_dims_dict = {key : value.delete_dims(self.batch_dims) if value is not None 
        #                    else empty_dims for key, value in data_dims_bound.items()}
        
        # batch_dims_sizes = {key : value.sizes if value is not None else [] 
        #                     for key, value in batch_dims_dict.items()}
        # other_dims_sizes = {key : value.sizes if value is not None else [] 
        #                     for key, value in other_dims_dict.items()}
        
        # # ii) Check if flattening possible: compare eqality of batch dim sizes
        # first_key = list(batch_dims_sizes.keys())[0]
        # first_dim_sizes = batch_dims_sizes[first_key]
        # flattened_dim_size = [math.prod(first_dim_sizes)] if first_dim_sizes is not [] else []
        # if not all([dim_sizes == first_dim_sizes for key, dim_sizes in batch_dims_sizes.items()]):
        #     raise(Exception('For flattening, all DimTuples batch_dims must be of ' \
        #                     'same size for all keys but are {} for keys {}'.format(batch_dims_sizes,list(batch_dims_sizes.keys()))))

        # # iii) Do the flattening
        # data_flattened = {}
        # batch_dim_flattened = dim_assignment(['batch_dim_flattened'], dim_sizes = flattened_dim_size)
        # new_dims = {key: batch_dim_flattened + value for key, value in other_dims_dict.items()}
        
        # for key, tensor in io_data.items():
        #     if tensor is not None:
        #         reordered_tensor = tensor.reorder(batch_dims_dict[key] + other_dims_dict[key])
        #         reshaped_tensor = reordered_tensor.tensor.reshape(batch_dim_flattened.sizes + other_dims_sizes[key])
        #         data_flattened[key] = CalipyTensor(reshaped_tensor, new_dims[key])
        #     else:
        #         data_flattened = None
                        
        # return CalipyDict(data_flattened)
        
        
        
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
        return  self.batch_length[0]
        

    def __getitem__(self, idx):
        # Handle the case where idx is a single integer
        bd_flat = self.batch_dim_flattened
        input_data_idx = self.flattened_input.subsample_tensors(bd_flat, [idx]) if self.flattened_input is not None else None
        output_data_idx = self.flattened_output.subsample_tensors(bd_flat, [idx]) if self.flattened_output is not None else None
        data_dict = {'input' : CalipyIO(input_data_idx), 'output' : CalipyIO(output_data_idx),
                     'index' : (bd_flat, torch.tensor([idx]))}
        
        return data_dict



# Custom collate function that collates dicts together by stacking contained 
# tensors along their batch dimension (which is assumed to have the name of
# 'batch_dim_flattened').

def dict_collate(batch, reduce = False):
    """ Custom collate function that collates ios together by stacking contained 
    tensors along their batch dimension (which is assumed to have the name of
    'batch_dim_flattened'). Used primarily as collate function for the DataLoader
    to perform automatized subsampling.
    :param batch: A list of CalipyDict containing info on input_vars, observations, and
        corresponding index that was used to produce them via dataset.__getitem__[idx]
    :type batch: list of CalipyDict
        
    :return: An instance of CalipyDict, where multiple CalipyDict objects are 
        collated together into a calipy_dict containing stacked CalipyTensors.
    :rtype: CalipyDict
    """
    
    # Untangle batch list
    list_of_dicts = batch
    inputs_io = CalipyIO([batch_dict['input'] for batch_dict in list_of_dicts])
    outputs_io = CalipyIO([batch_dict['output'] for batch_dict in list_of_dicts] )
    indices_io = CalipyIO([batch_dict['index'] for batch_dict in list_of_dicts])
    
    # Check input signatures
    bool_reduction_inputs = inputs_io.is_reducible()
    bool_reduction_outputs = outputs_io.is_reducible()
    bool_reduction_indices = indices_io.is_reducible()
    bool_reduction_data = bool_reduction_inputs*bool_reduction_outputs*bool_reduction_indices
    
    # compatibility reduce argument
    if reduce == True and bool_reduction_data == False:
        raise Exception("Attempting to reduce dataset io's to single dicts but not " \
                        "all of the are reducible. Reducibility: inputs: {}, outputs : {}, indices : {}"
                        .format(bool_reduction_inputs, bool_reduction_outputs, bool_reduction_indices) )
    
    # If reduce = False, just concatenate the lists inside the IO's
    if reduce  == False:
        
        
    # Construct new dictionaries
    output_dict = {}
    output_keys = list_of_outputs[0].keys()
    for key in output_keys:
        tensors_to_concat = [d[key] for d in list_of_outputs]
        output_dict[key] = calipy_cat(tensors_to_concat, dim = 0)
        
    input_dict = {}
    input_keys = list_of_inputs[0].keys()
    for key in input_keys:
        tensors_to_concat = [d[key] for d in list_of_inputs]
        input_dict[key] = calipy_cat(tensors_to_concat, dim = 0)
        
    
    flattened_batch_dim = list_of_indices[0][0]
    index_dim = dim_assignment(['index_dim'])
    indices_to_concat = torch.cat([d[1] for d in list_of_indices], dim = 0).reshape([-1,1])
    ssi = CalipyIndex(indices_to_concat, flattened_batch_dim + index_dim, name = 'subsample_index')
    
    
    return  input_dict, output_dict, ssi





# # Custom collate function that collates tensors by concatenating them among the 
# # first dimension. Allows vectorization. Only possible when batch shapes are 
# # consistent and event shapes for each element are independent of batch
# # Input to the collate are assumed to be tensortuplelists
# def tensor_collate(batch):
    
#     batch_input, batch_output, indices = zip(*batch)
#     reduced_batch_input = reduce_nones(batch_input)
    
#     len_input = len(batch_input) if reduced_batch_input is not None else 0 
#     len_input_tuple = len(batch_input[0]) if reduced_batch_input is not None else 0
#     len_output = len(batch_output)
#     len_output_tuple = len(batch_output[0])
    
#     # Check if vectorizable
#     batch_input_shapes = []
#     batch_output_shapes = []
    
#     # for k_in in range(len_input):
#     #     input_shapes = [bi.shape for bi in batch_input[k_in]]
#     #     batch_input_shapes.append(input_shapes)
#     # for k_out in range(len_output):
#     #     output_shapes = [bo.shape for bo in batch_output[k_out]]
#     #     batch_output_shapes.append(output_shapes)
        
#     batch_input_shapes = [[bi.shape for bi in batch_input[k]] for k in range(len_input)]
#     batch_output_shapes = [[bo.shape for bo in batch_output[k]] for k in range(len_output)]
#     input_consistency = [x == batch_input_shapes[0] for x in batch_input_shapes]
#     output_consistency = [x == batch_output_shapes[0] for x in batch_output_shapes]
#     batch_consistency = len_input == len_output if reduced_batch_input is not None else True
    
#     if input_consistency is False:
#         raise(Exception('The batch input has shapes that disallow vectorization. Need to be equal'\
#                         ' for each tuple but are {}'.format(batch_input_shapes)))
#     if output_consistency is False:
#         raise(Exception('The batch output has shapes that disallow vectorization. Need to be equal'\
#                         ' for each tuple but are {}'.format(batch_output_shapes)))
#     if batch_consistency is False:
#         raise(Exception('The number of list elements is inconsistent between input and' \
#                         ' output batch. Should be equal but are len(batch_input) ={}' \
#                         ' len(batch_output) = {}. Alternatively, input may be None'
#                         .format(len_input, len_output)))
    
    
    
#     # Keep None as is, or handle as required
#     # collated_batch_input = tuple([torch.cat(bi, dim = 0) for bi in batch_input]) if reduced_batch_input is not None else None
#     # collated_batch_output = tuple([torch.cat(bo, dim=0) for bo in batch_output])
    
#     collated_batch_input = []
#     collated_batch_output = []
#     for k_inputs in range(len_input_tuple):
#         input_batch_list = []
#         for k_samples in range(len_input):
#             input_batch_list.append(batch_input[k_samples][k_inputs])
#         input_batch_tensor = torch.cat(input_batch_list, dim = 0)
#         collated_batch_input.append(input_batch_tensor)
#     collated_batch_input = collated_batch_input if not len(collated_batch_input) ==0 else None
    
#     for k_outputs in range(len_output_tuple):
#         output_batch_list = []
#         for k_samples in range(len_output):
#             output_batch_list.append(batch_output[k_samples][k_outputs])
#         output_batch_tensor = torch.cat(output_batch_list, dim = 0)
#         collated_batch_output.append(output_batch_tensor)
        
            
#     #         collated_batch_input.append([torch.cat([bi[k_inputs] for bi in batch_input], dim = 0)])
#     # for k_outputs in range(len_output_tuple):
#     #     collated_batch_output.append([torch.cat([bo[k_outputs] for bo in batch_output], dim = 0)])
    
#     # indices = torch.tensor(indices)
    
#     return collated_batch_input, collated_batch_output, indices


# # helper function to cat also none tensors
# def safe_cat(tensor_list, dim=0):
    
#     # Filter out None values
#     filtered_tensors = [t for t in tensor_list if t is not None]
    
#     # Concatenate the remaining tensors
#     if filtered_tensors:
#         return torch.cat(filtered_tensors, dim=dim)
#     else:
#         return None  # or raise an exception if needed

# safe_cat.__doc__ = 'A torch.cat derivative that handles entries of None in the tensor list'\
#     'by omitting them.' + torch.cat.__doc__


# def reduce_nones(input_list):
#     # reduces a list containing Nones by eliminating the Nones and replaces a
#     # full None list [None, None, ...] with None
#     reduced_list = [element for element in input_list if element is not None]
#     reduced_list = reduced_list if not len(reduced_list) == 0 else None
    
#     return reduced_list



# # This collate handles tensortuplelists that are coming from sequentially calling
# # the CalipyDataset.__getitem__() method. The lists are handed back verbatim.
# def list_collate(batch):
#     batch_input, batch_output, indices = zip(*batch)    
#     return batch_input, batch_output, indices


# # Create a DataLoader
# dataset_via_tensortuple = CalipyDataset(None, data_tensortuple, dataset_type = 'tensortuple')
# dataset_via_tensortuplelist = CalipyDataset(input_data, output_data, dataset_type = 'tensortuplelist')

# dataloader = DataLoader(dataset_via_tensortuple, batch_size=n_subbatch, shuffle=True, collate_fn=tensor_collate)


# # Iterate through the DataLoader
# for batch_input, batch_output, index in dataloader:
#     print(batch_input, batch_output, index)

# # STANDARD BATCH SITUATION

# mu_true = torch.tensor(0.0)
# sigma_true = torch.tensor(0.1)


# # ii) Sample from distributions

# data_distribution = pyro.distributions.Normal(mu_true, sigma_true)
# data = data_distribution.sample([n_meas, n_event])

# # The data now is a tensor of shape [n_meas,1] and reflects measurements being
# # taken of a single object with a single measurement device.

# # We now consider the data to be an outcome of measurement of some real world
# # object; consider the true underlying data generation process to be unknown
# # from now on.


# # iii) Enable subbatching
# data_dims = dim_assignment(['bd_data', 'ed_data'], dim_sizes = [n_meas, n_event])
# data_cp = CalipyTensor(data, data_dims, name = 'data')
# dataset = CalipyDataset(input_data = None, output_data = data_cp, batch_dims = data_dims[['bd_data']])

# batch_dim_flattened = dataset.batch_dim_flattened
# flattened_output = dataset.flattened_output.subsample_tensors(batch_dim_flattened, [0])

# dataloader = DataLoader(dataset, batch_size=n_subbatch, shuffle=True, collate_fn=dict_collate)

# # Iterate through the DataLoader
# for batch_input, batch_output, batch_index in dataloader:
#     print(batch_input, batch_output, batch_index)


# # MULTIBATCH Situation
# n_meas1 = 20
# n_meas2 = 5

# mu_true = torch.tensor(0.0)
# sigma_true = torch.tensor(0.1)


# # ii) Sample from distributions

# data_distribution = pyro.distributions.Normal(mu_true, sigma_true)
# data = data_distribution.sample([n_meas1, n_meas2, n_event])

# # The data now is a tensor of shape [n_meas] and reflects measurements being
# # taken of a single object with a single measurement device.

# # We now consider the data to be an outcome of measurement of some real world
# # object; consider the true underlying data generation process to be unknown
# # from now on.


# # iii) Enable subbatching
# data_dims = dim_assignment(['bd1_data', 'bd2_data', 'ed_data'], dim_sizes = [n_meas1, n_meas2, n_event])
# data_cp = CalipyTensor(data, data_dims, name = 'data')
# dataset = CalipyDataset(input_data = None, output_data = data_cp, batch_dims = data_dims[['bd1_data', 'bd2_data']])

# batch_dim_flattened = dataset.batch_dim_flattened
# flattened_output = dataset.flattened_output.subsample_tensors(batch_dim_flattened, [0])

# dataloader = DataLoader(dataset, batch_size=n_subbatch, shuffle=True, collate_fn=dict_collate)

# # Iterate through the DataLoader
# for batch_input, batch_output, batch_index in dataloader:
#     print(batch_input, batch_output, batch_index)
    
    
    
#########################################################


"""
    2. Simulate some data, enable subbatching
"""


# i) Set up sample distributions

mu_true = torch.tensor(0.0)
sigma_true = torch.tensor(0.1)


# ii) Sample from distributions

data_distribution = pyro.distributions.Normal(mu_true, sigma_true)
data = data_distribution.sample([n_meas, n_event])

# The data now is a tensor of shape [n_meas] and reflects measurements being
# taken of a single object with a single measurement device.

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.


# iii) Enable subbatching
data_dims = dim_assignment(['bd_data', 'ed_data'], dim_sizes = [n_meas, n_event])
data_cp = CalipyTensor(data, data_dims, name = 'data')
dataset = CalipyDataset(input_data = None, output_data = data_cp, batch_dims = data_dims[['bd_data']])


batch_dim_flattened = dataset.batch_dim_flattened
dataloader = DataLoader(dataset, batch_size=n_subbatch, shuffle=True, collate_fn=dict_collate)

# Iterate through the DataLoader
for batch_input, batch_output, batch_index in dataloader:
    print(batch_input, batch_output, batch_index)


"""
    3. Load and customize effects
"""


# i) Set up dimensions

batch_dims = dim_assignment(['bd_1'], dim_sizes = [n_meas])
event_dims = dim_assignment(['ed_1'], dim_sizes = [1])
param_dims = dim_assignment(['pd_1'], dim_sizes = [])


# ii) Set up dimensions for mean parameter mu

# Setting up requires correctly specifying a NodeStructure object. You can get 
# a template for the node_structure by calling generate_template() on the example
# node_structure delivered with the class description. Here, we call the example
# node structure, then set the dims; required dims that need to be provided can
# be found via help(mu_ns.set_dims).

# mu setup
mu_ns = NodeStructure(UnknownParameter)
mu_ns.set_dims(batch_dims = batch_dims, param_dims = param_dims,)
mu_object = UnknownParameter(mu_ns, name = 'mu')


# iii) Set up the dimensions for noise addition
# This requires again the batch shapes and event shapes. They are used to set
# up the dimensions in which the noise is i.i.d. and the dims in which it is
# copied. Again, required keys can be found via help(noise_ns.set_dims).
noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims = batch_dims, event_dims = event_dims)
noise_object = NoiseAddition(noise_ns, name = 'noise')
        


"""
    4. Build the probmodel
"""


# i) Define the probmodel class 

class DemoProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # integrate nodes
        self.mu_object = mu_object
        self.noise_object = noise_object 
        
    # Define model by forward passing
    def model(self, input_vars = None, observations = None, subsample_index = None):
        mu = self.mu_object.forward(subsample_index = subsample_index)    

        # Dictionary/DataTuple input is converted to CalipyDict internally. It
        # is also possible, to pass single element input_vars or observations;
        # these are also autowrapped.
        inputs = {'mean':mu, 'standard_deviation': sigma_true} 
        output = self.noise_object.forward(input_vars = inputs,
                                           observations = observations,
                                           subsample_index = subsample_index)
        
        return output
    
    # Define guide (trivial since no posteriors)
    def guide(self, input_vars = None, observations = None, subsample_index = None):
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
# When passed to any forward() method, input_vars and observations are wrapped
# in CalipyDict. Passing data_cp is equivalent to passing CalipyTensor(data_cp).
# Passing directly a CalipyDict is always ok, but in case of one-element
# input_vars or observations, only passing that element and letting the autowrap
# handle the rest can make code simpler to read.
input_data = None
data_cp = CalipyTensor(data, dims = batch_dims + event_dims)
optim_results = demo_probmodel.train(dataloader = dataloader, optim_opts = optim_opts)



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
print('Results of taking empirical means for mu_1 = ', torch.mean(data))






















