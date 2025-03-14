# This provides functionality for writing a model with a vectorizable flag where
# the model code actually does not branch depending on the flag. We investigate 
# here, how this can be used to perform subbatching without a model-rewrite
# We make our life a bit easier by only considering one batch dim for now.
# We feature nontrivial event_shape and functorch.dim based indexing.



import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import torch
import contextlib
import itertools
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functorch.dim import dims

import calipy
from calipy.core.utils import dim_assignment, DimTuple, TorchdimTuple, CalipyDim, ensure_tuple, multi_unsqueeze
from calipy.core.effects import CalipyQuantity, CalipyEffect, UnknownParameter, NoiseAddition
from calipy.core.data import DataTuple
from calipy.core.tensor import CalipyIndex, CalipyIndexer, TensorIndexer, CalipyTensor
from calipy.core.base import NodeStructure, CalipyProbModel
from calipy.core.primitives import param, sample

import numpy as np
import einops
import pandas as pd
import textwrap
import inspect
import random
import varname
import copy
import matplotlib.pyplot as plt
import warnings

torch.manual_seed(42)
pyro.set_rng_seed(42)


# i) Generate synthetic data A and B

# Generate data_A corresponding e.g. to two temperature vals on a grid on a plate
n_data_1_A = 6
n_data_2_A = 4
batch_shape_A = [n_data_1_A, n_data_2_A]
n_event_A = 2
n_event_dims_A = 1
n_data_total_A = n_data_1_A * n_data_2_A


# # Plate names and sizes
plate_names_A = ['batch_plate_1_A', 'batch_plate_2_A']
plate_sizes_A = [n_data_1_A, n_data_2_A]

mu_true_A = torch.zeros([1,1,n_event_A])
sigma_true_A = torch.tensor([[1,0],[0,0.01]])

extension_tensor_A = torch.ones([n_data_1_A,n_data_2_A,1])
data_dist_A = pyro.distributions.MultivariateNormal(loc = mu_true_A * extension_tensor_A, covariance_matrix = sigma_true_A)
data_A = data_dist_A.sample()

# Generate data_B corresponding e.g. to three temperature vals on each location on a rod
n_data_1_B = 5
batch_shape_B = [n_data_1_B]
n_event_B = 3
n_event_dims_B = 1
n_data_total_B = n_data_1_B


# # Plate names and sizes
plate_names_B = ['batch_plate_1_B']
plate_sizes_B = [n_data_1_B]

mu_true_B = torch.zeros([1,n_event_B])
sigma_true_B = torch.tensor([[1,0,0],[0,0.1,0],[0,0,0.01]])

extension_tensor_B = torch.ones([n_data_1_B,1])
data_dist_B = pyro.distributions.MultivariateNormal(loc = mu_true_B * extension_tensor_B, covariance_matrix = sigma_true_B)
data_B = data_dist_B.sample()


# functorch dims
batch_dims_A = dim_assignment(dim_names = ['bd_1_A', 'bd_2_A'])
event_dims_A = dim_assignment(dim_names = ['ed_1_A'])
data_dims_A = batch_dims_A + event_dims_A

batch_dims_B = dim_assignment(dim_names = ['bd_1_B'])
event_dims_B = dim_assignment(dim_names = ['ed_1_B'])
data_dims_B = batch_dims_B + event_dims_B



# # Test the DataTuple definitions
# # Define names and values for DataTuple
# names = ['tensor_a', 'tensor_b']
# values = [torch.ones(2, 3), torch.zeros(4, 5)]

# # Create DataTuple instance
# data_tuple = DataTuple(names, values)

# # 1. Apply a function dictionary to DataTuple
# fun_dict = {'tensor_a': lambda x: x + 1, 'tensor_b': lambda x: x - 1}
# result_tuple = data_tuple.apply_from_dict(fun_dict)
# print("Result of applying function dictionary:", result_tuple)

# # 2. Get the shapes of each tensor in DataTuple
# shapes_tuple = data_tuple.get_subattributes('shape')
# print("Shapes of each tensor in DataTuple:", shapes_tuple)

# # 3. Apply a custom class DifferentClass to each element in DataTuple
# class DifferentClass:
#     def __init__(self, tensor):
#         self.tensor = tensor

#     def __repr__(self):
#         return f"DifferentClass(tensor={self.tensor})"

# different_tuple = data_tuple.apply_class(DifferentClass)
# print("Result of applying DifferentClass to DataTuple:", different_tuple)

# # 4. Adding two DataTuples
# another_values = [torch.ones(2, 3) * 2, torch.ones(4, 5) * 3]
# another_data_tuple = DataTuple(names, another_values)
# added_tuple = data_tuple + another_data_tuple
# print("Result of adding two DataTuples:", added_tuple)   
  



# Define calipy wrappers for torch functions
# This should be converted to commands calipy.sum, calipy.mean etc
# These functions are useful to process CalipyTensors in such a way that the dim
# argument can be specified as a CalipyDim object; if this is not needed, just apply
# the torch versions torch.sum, torch.mean etc.

# Preprocessing input arguments
def preprocess_args(args, kwargs):
    if kwargs is None:
            kwargs = {}

    # Unwrap CalipyTensors to get underlying tensors
    def unwrap(x):
        return x.tensor if isinstance(x, CalipyTensor) else x

    unwrapped_args = tuple(unwrap(a) for a in args)
    unwrapped_kwargs = {k: unwrap(v) for k, v in kwargs.items()}
    
        
    return unwrapped_args, unwrapped_kwargs


# Alternative idea here: Build function calipy_op(func, *args, **kwargs) 
# then inspect and functools wraps
def calipy_sum(calipy_tensor, dim = None, keepdim = False, dtype = None):
    """ Wrapper function for torch.sum applying a dimension-aware sum to CalipyTensor
    objects. Input args are as for torch.sum but accept dim = dims for dims either
    a DimTuple of a CalipyDim.
    
    Notes:
    - This function acts on CalipyTensor objects
    - This function acts on dim args of class CalipyDim and DimTuple.
    - The behavior is equivalent to torch.sum on the CalipyTensor.tensor level
        but augments the result with dimensions.

    Original torch.sum docstring:
    """
    
    # Compile and unwrap arguments
    args = (calipy_tensor,)
    kwargs = {'dim' : dim,
              'keepdim' : keepdim,
              'dtype' : dtype}
        
    # Convert CalipyDims in 'dim' argument to int indices if present
    if dim is not None:
        kwargs['dim'] = tuple(calipy_tensor.dims.find_indices(kwargs['dim'].names))
    
    # Call torch function
    result = torch.sum(*args, **kwargs)
    return result
    
calipy_sum.__doc__ += "\n" + torch.sum.__doc__
   




    
# Check the functionality of CalipyTensor class
data_A_cp = CalipyTensor(data_A, data_dims_A, 'data_A')


# Check the functionality of this class with monkeypatch
data_A_cp = CalipyTensor(data_A, data_dims_A, 'data_A')
local_index = data_A_cp.indexer.local_index
local_index.tensor.shape
local_index.dims
assert (data_A_cp.tensor[local_index.tuple] == data_A_cp.tensor).all()
assert (((data_A_cp[local_index] - data_A_cp).tensor == 0).all())

# Indexing of CalipyTensors via int, tuple, slice, and CalipyIndex
data_A_cp[0,:]
data_A_cp[local_index]
subtensor_A = data_A_cp[0,...]  # identical to next line
subtensor_A = data_A_cp[0:1,...]

# During addressing, appropriate indexers are built
data_A_cp[0,:].indexer.global_index
data_A_cp[local_index].indexer.global_index

# Check reordering and torchdims for reordered
reordered_dims = DimTuple((data_dims_A[1], data_dims_A[2], data_dims_A[0]))
data_A_reordered = data_A_cp.indexer.reorder(reordered_dims)

data_tdims_A = data_dims_A.build_torchdims()
data_tdims_A_reordered = data_tdims_A[reordered_dims]
data_A_named = data_A[data_tdims_A]
data_A_named_reordered = data_A_reordered.tensor[data_tdims_A_reordered]
assert (data_A_named.order(*data_tdims_A) == data_A_named_reordered.order(*data_tdims_A)).all()

# Check simple subsampling
simple_subsamples, simple_subsample_indices = data_A_cp.indexer.simple_subsample(batch_dims_A[0], 5)
block_batch_dims_A = batch_dims_A
block_subsample_sizes_A = [5,3]
block_subsamples, block_subsample_indices = data_A_cp.indexer.block_subsample(block_batch_dims_A, block_subsample_sizes_A)

# Subsampling is also possible when dims are bound previously
data = data_A
data_dims = dim_assignment(['bd_1', 'bd_2', 'ed_1'], dim_sizes = [6,4,2])
data_cp = CalipyTensor(data, data_dims, 'data')
data_cp.indexer.simple_subsample(data_dims[0], 4)

# Check subsample indexing 
# Suppose we got data_D as a subset of data_C with derived ssi CalipyIndex and
# now want to index data_D with proper names and references
# First, generate data_C
batch_dims_C = dim_assignment(['bd_1_C', 'bd_2_C'])
event_dims_C = dim_assignment(['ed_1_C'])
data_dims_C = batch_dims_C + event_dims_C
data_C = torch.normal(0,1,[7,5,2])
data_C_cp =  CalipyTensor(data_C, data_dims_C, 'data_C')

# Then, subsample data_D from data_C
block_data_D, block_indices_D = data_C_cp.indexer.block_subsample(batch_dims_C, [5,3])
block_nr = 3
data_D_cp = block_data_D[block_nr]
block_index_D = block_indices_D[block_nr]

# Now look at global indices and names; the indexer has been inherited during subsampling
data_D_cp.indexer
data_D_cp.indexer.local_index
data_D_cp.indexer.global_index
data_D_cp.indexer.local_index.tensor
data_D_cp.indexer.global_index.tensor
data_D_cp.indexer.data_source_name

# If data comes out of some external subsampling and only the corresponding indextensors
# are known, the calipy_indexer can be evoked manually.
data_E = copy.copy(data_D_cp.tensor)
index_tensor_E = block_index_D.tensor

data_E_cp = CalipyTensor(data_E, data_dims_C, 'data_E')
data_E_cp.indexer.create_global_index(index_tensor_E, 'from_data_E')
data_E_index_tuple = data_E_cp.indexer.global_index.tuple

assert (data_E == data_C[data_E_index_tuple]).all()
assert(((data_E_cp - data_C_cp[data_E_cp.indexer.global_index]).tensor == 0).all())


# Check interaction with DataTuple class
data_names_list = ['data_A', 'data_B']
data_list = [data_A, data_B]
data_datatuple = DataTuple(data_names_list, data_list)

batch_dims_datatuple = DataTuple(data_names_list, [batch_dims_A, batch_dims_B])
event_dims_datatuple = DataTuple(data_names_list, [event_dims_A, event_dims_B])
data_dims_datatuple = batch_dims_datatuple + event_dims_datatuple

# Build the calipy_indexer; it does not exist before this line for e.g data_datatuple['data_B']
data_datatuple_cp = data_datatuple.calipytensor_construct(data_dims_datatuple)


# Show how to do consistent subbatching in multiple tensors by creating an index
# with reduced dims and expanding it as needed.

# First, create the dims (with unspecified size so no conflict later when subbatching)
batch_dims_FG = dim_assignment(['bd_1_FG', 'bd_2_FG'])
event_dims_F = dim_assignment(['ed_1_F', 'ed_2_F'])
event_dims_G = dim_assignment(['ed_1_G'])
data_dims_F = batch_dims_FG + event_dims_F
data_dims_G = batch_dims_FG + event_dims_G

# Sizes
batch_dims_FG_sizes = [10,7]
event_dims_F_sizes = [6,5]
event_dims_G_sizes = [4]
data_dims_F_sizes = batch_dims_FG_sizes + event_dims_F_sizes
data_dims_G_sizes = batch_dims_FG_sizes + event_dims_G_sizes

# Then create the data
data_F = torch.normal(0,1, data_dims_F_sizes)
data_F_cp = CalipyTensor(data_F, data_dims_F, 'data_F')
data_G = torch.normal(0,1, data_dims_G_sizes)
data_G_cp = CalipyTensor(data_G, data_dims_G, 'data_G')

# Create and expand the reduced_index
indices_reduced = TensorIndexer.create_block_subsample_indices(batch_dims_FG, batch_dims_FG_sizes, [9,5])
index_reduced = indices_reduced[0]
index_expanded_F = index_reduced.expand_to_dims(data_dims_F, [None]*len(batch_dims_FG) + event_dims_F_sizes)
index_expanded_G = index_reduced.expand_to_dims(data_dims_G, [None]*len(batch_dims_FG) + event_dims_G_sizes)
assert (data_F[index_expanded_F.tuple] == data_F[index_reduced.tensor[:,:,0], index_reduced.tensor[:,:,1], :,:]).all()

# Reordering is also possible
data_dims_F_reordered = dim_assignment(['ed_2_F', 'bd_2_FG', 'ed_1_F', 'bd_1_FG'])
data_dims_F_reordered_sizes = [5, None, 6, None]
index_expanded_F_reordered = index_reduced.expand_to_dims(data_dims_F_reordered, data_dims_F_reordered_sizes)
data_F_reordered = data_F_cp.indexer.reorder(data_dims_F_reordered)

data_F_subsample = data_F[index_expanded_F.tuple]
data_F_reordered_subsample = data_F_reordered[index_expanded_F_reordered.tuple]
assert (data_F_subsample == data_F_reordered_subsample.permute([3,1,2,0])).all()

# Alternatively, index expansion can also be performed by the indexer of a tensor
# this is usually more convenient
index_expanded_F_alt = data_F_cp.indexer.expand_index(index_reduced)
index_expanded_G_alt = data_G_cp.indexer.expand_index(index_reduced)

# Index can now be used for consistent subsampling
data_F_subsample_alt = data_F[index_expanded_F_alt.tuple]
data_G_subsample_alt = data_G[index_expanded_G_alt.tuple]
assert (data_F_subsample == data_F_subsample_alt).all()

# Inverse operation is index_reduction (only possible when index is cartensian product)
assert (index_expanded_F.is_reducible(batch_dims_FG))
assert (index_reduced.tensor == index_expanded_F.reduce_to_dims(batch_dims_FG).tensor).all()
assert (index_reduced.tensor == index_expanded_G.reduce_to_dims(batch_dims_FG).tensor).all()

# Illustrate nonseparable case
index_dim = dim_assignment(['index_dim'])
inseparable_index = CalipyIndex(torch.randint(10, [10,7,6,5,4]), data_dims_F + index_dim, name = 'inseparable_index')
inseparable_index.is_reducible(batch_dims_FG)
inseparable_index.reduce_to_dims(batch_dims_FG) # Produces a warning as it should


# Showcase interaction between DataTuple and CalipyTensor

batch_dims = dim_assignment(dim_names = ['bd_1'])
event_dims_A = dim_assignment(dim_names = ['ed_1_A', 'ed_2_A'])
data_dims_A = batch_dims + event_dims_A
event_dims_B = dim_assignment(dim_names = ['ed_1_B'])
data_dims_B = batch_dims + event_dims_B

data_A_torch = torch.normal(0,1,[6,4,2])
data_A_cp = CalipyTensor(data_A_torch, data_dims_A, 'data_A')
data_B_torch = torch.normal(0,1,[6,3])
data_B_cp = CalipyTensor(data_B_torch, data_dims_B, 'data_B')

data_AB_tuple = DataTuple(['data_A_cp', 'data_B_cp'], [data_A_cp, data_B_cp])

# subsample the data individually
data_AB_subindices = TensorIndexer.create_simple_subsample_indices(batch_dims[0], data_A_cp.shape[0], 5)
data_AB_subindex = data_AB_subindices[0]
data_A_subindex = data_AB_subindex.expand_to_dims(data_dims_A, data_A_cp.shape)
data_B_subindex = data_AB_subindex.expand_to_dims(data_dims_B, data_B_cp.shape)
data_AB_sub_1 = DataTuple(['data_A_cp_sub', 'data_B_cp_sub'], [data_A_cp[data_A_subindex], data_B_cp[data_B_subindex]])

# Use subsampling functionality for DataTuples, either by passing a DataTuple of
# CalipyIndex or a single CalipyIndex that is broadcasted
data_AB_subindex_tuple = DataTuple(['data_A_cp', 'data_B_cp'], [data_A_subindex, data_B_subindex])
data_AB_sub_2 = data_AB_tuple.subsample(data_AB_subindex_tuple)
data_AB_sub_3 = data_AB_tuple.subsample(data_AB_subindex)
assert ((data_AB_sub_1[0] - data_AB_sub_2[0]).tensor == 0).all()
assert ((data_AB_sub_2[0] - data_AB_sub_3[0]).tensor == 0).all()

# # THIS DOES NOT WORK YET, SINCE UNCLEAR HOW TENSORDIMS AND INDEXDIM MIMATCH TO BE HANDLED
# # If a dim does not feature in the CalipyTensor, that tensor is not subsampled
# data_C_torch = torch.normal(0,1,[6,4,2])
# data_dims_C = dim_assignment(dim_names = ['bd_1_C', 'ed_1_C', 'ed_2_C'])
# data_C_cp = CalipyTensor(data_C_torch, data_dims_C, 'data_C')
# data_ABC_tuple = DataTuple(['data_A_cp', 'data_B_cp', 'data_C_cp'], [data_A_cp, data_B_cp, data_C_cp])
# data_ABC_sub = data_ABC_tuple.subsample(data_AB_subindex)

# # Showcase torch functions acting on CalipyTensor
# specific_sum = calipy_sum(data_A_cp, dim = batch_dims_A)
# generic_sum = calipy_sum(data_A_cp)



# Check the calipy_cat function
from calipy.core.funs import calipy_cat

# Create data for CalipyDict initialization
tensor_dims = dim_assignment(['bd', 'ed'])
tensor_A_cp = CalipyTensor(torch.ones(2, 3), tensor_dims) 
tensor_B_cp = CalipyTensor(2*torch.ones(4, 3), tensor_dims) 
tensor_C_cp = CalipyTensor(2*torch.ones(2, 2), tensor_dims) 

# Create CalipyDict cat
tensor_cat_1 = calipy_cat([tensor_A_cp, tensor_B_cp], dim = 0)
tensor_cat_2 = calipy_cat([tensor_A_cp, tensor_C_cp], dim = 1)

tensor_cat_1_alt = calipy_cat([tensor_A_cp, tensor_B_cp], dim = tensor_dims[0:1])
tensor_cat_2_alt = calipy_cat([tensor_A_cp, tensor_C_cp], dim = tensor_dims[1:2])

assert(( tensor_cat_1.tensor - tensor_cat_1_alt.tensor == 0).all())
assert(( tensor_cat_2.tensor - tensor_cat_2_alt.tensor == 0).all())



# CalipyTensors work well even when some dims are empty
# Set up data and dimensions
data_0dim = torch.ones([])
data_1dim = torch.ones([5])
data_2dim = torch.ones([5,2])

batch_dim = dim_assignment(['bd'])
event_dim = dim_assignment(['ed'])
empty_dim = dim_assignment(['empty'], dim_sizes = [])

data_0dim_cp = CalipyTensor(data_0dim, empty_dim)
data_1dim_cp = CalipyTensor(data_1dim, batch_dim)
data_1dim_cp = CalipyTensor(data_1dim, batch_dim + empty_dim)
data_1dim_cp = CalipyTensor(data_1dim, empty_dim + batch_dim + empty_dim)

data_2dim_cp = CalipyTensor(data_2dim, batch_dim + event_dim)
data_2dim_cp = CalipyTensor(data_2dim, batch_dim + empty_dim + event_dim)

# Indexing a scalar with an empty index just returns the scalar
data_0dim_cp.indexer
zerodim_index = data_0dim_cp.indexer.local_index
zerodim_index.is_empty
data_0dim_cp[zerodim_index]

# # These produce errors or warnings as they should.
# data_0dim_cp = CalipyTensor(data_0dim, batch_dim) # Trying to assign nonempty dim to scalar
# data_1dim_cp = CalipyTensor(data_1dim, empty_dim) # Trying to assign empty dim to vector
# data_2dim_cp = CalipyTensor(data_2dim, batch_dim + empty_dim) # Trying to assign empty dim to vector



# Expand a tensor by copying it among some dimensions.
data_dims_A = data_dims_A.bind([6,4,2])
data_dims_B = data_dims_B.bind([6,3])
data_dims_expanded = data_dims_A + data_dims_B[1:]
data_A_expanded_cp = data_A_cp.expand_to_dims(data_dims_expanded)
assert((data_A_expanded_cp[:,:,:,0].tensor.squeeze() - data_A_cp.tensor == 0).all())
# Ordering of dims is also ordering of result
data_dims_expanded_reordered = data_dims_A[1:] + data_dims_A[0:1] + data_dims_B[1:]
data_A_expanded_reordered_cp = data_A_cp.expand_to_dims(data_dims_expanded_reordered)
assert((data_A_expanded_reordered_cp.tensor -
        data_A_expanded_cp.tensor.permute([1,2,0,3]) == 0).all())

# There also exists a CalipyTensor.reorder(dims) method
data_dims_A_reordered = event_dims_A + batch_dims
data_A_reordered_cp = data_A_cp.reorder(data_dims_A_reordered)
assert((data_A_reordered_cp.tensor - data_A_cp.tensor.permute([1,2,0]) == 0).all())
assert(data_A_reordered_cp.dims == data_dims_A_reordered)


# Indexing can also be done in typical torch fashion for CalipyTensors
# Create DimTuples and tensors
data_torch = torch.normal(0,1,[10,5,3])
batch_dims = dim_assignment(dim_names = ['bd_1', 'bd_2'], dim_sizes = [10,5])
event_dims = dim_assignment(dim_names = ['ed_1'], dim_sizes = [3])
data_dims = batch_dims + event_dims
data_cp = CalipyTensor(data_torch, data_dims, name = 'data')

# Access the single element where batch_dim 'bd_1' has the value 5
data_cp_element_1 = data_cp.get_element(batch_dims[0:1], [5])
assert((data_cp_element_1.tensor.squeeze() - data_cp.tensor[5,...] == 0).all())

# Access the single element where batch_dims has the value [5,2]
data_cp[5,2,0]
data_cp_element_2 = data_cp.get_element(batch_dims, [5,2])
assert((data_cp_element_2.tensor.squeeze() - data_cp.tensor[5,2,...] == 0).all())

# BASE CLASS EXPERIMENTATION
# Base classes




# TEST BASE CLASSES
generic_dims = dim_assignment(['bd_1', 'ed_1'], dim_sizes = [10,3], 
                              dim_descriptions = ['batch_dim_1', 'event_dim_1'])
ns_empty = NodeStructure()
ns_empty.set_dims

ns_prebuilt = NodeStructure(UnknownParameter)
ns_prebuilt.dims
ns_prebuilt.set_dims(batch_dims = (generic_dims[['bd_1']], 'batch dimensions'))


# i) Imports and definitions
# import calipy
# from calipy.core.base import NodeStructure
# from calipy.core.effects import UnknownParameter
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

# The set_dims method inherits an updated docstring and autocompletion
print(param_ns_1)   # Shows the default_nodestructure of UnknownParamter
help(param_ns_1.set_dims)   # Shows that param_dims, batch_dims are arguments

# The initialized node structure can be updated by inheritance or by directly setting
# dimensions. It errors out, if a dimension is specified that is not specified
# by the default_nodestructure

# Create nodestructure with custom param_dims and batch_dims
param_ns_1.inherit_common_dims(node_structure)  
print(param_ns_1)

# Create nodestructure with custom param_dims and default batch_dims
param_ns_2.set_dims(param_dims = param_dims) 
print(param_ns_2)   

# This errors out as it should: param_ns_3.set_dims(event_dims = event_dims) 

#
# iii) Investigate NodeStructure objects
param_ns_1.dims
param_ns_1.dim_names
param_ns_1.dim_descriptions
param_ns_1.node_cls

# It is possible to build the code that, if executed, generates the nodestructure
param_ns_1.generate_template()

 
# iv) Build and check nodestructure via class methods
empty_node_structure = NodeStructure()
UnknownParameter.check_node_structure(empty_node_structure)
UnknownParameter.check_node_structure(param_ns_1)


# TEST PARAM FUNCTION

# Create parameter ---------------------------------------------------
#
# i) Imports and definitions


batch_dims = dim_assignment(dim_names = ['bd_1_A'], dim_sizes = [4])
event_dims = dim_assignment(dim_names = ['ed_1_A'], dim_sizes = [2])
param_dims = batch_dims + event_dims
init_tensor = torch.ones(param_dims.sizes) + torch.normal(0,0.01, param_dims.sizes)

parameter = param('generic_param', init_tensor, param_dims)
print(parameter)

# Create constrained, subsampled parameter ---------------------------
#
param_constraint = pyro.distributions.constraints.positive
subsample_indices = TensorIndexer.create_simple_subsample_indices(batch_dims[0], batch_dims.sizes[0], 3)
ssi = subsample_indices[1]
ssi_expanded = ssi.expand_to_dims(param_dims, param_dims.sizes)
parameter_subsampled = param('positive_param_subsampled', init_tensor, param_dims,
                             constraint = param_constraint, subsample_index = ssi_expanded)
print(parameter_subsampled)
assert((parameter_subsampled.tensor - parameter.tensor[ssi_expanded.tuple] == 0).all())

# Investigate parameter ----------------------------------------------
#

# Parameters are CalipyTensors with names, dims, and populated indexers
parameter.name
parameter.dims
parameter.indexer
parameter.indexer.global_index
parameter.indexer.global_index.tensor

parameter_subsampled.name
parameter_subsampled.dims
parameter_subsampled.indexer
parameter_subsampled.indexer.global_index
parameter_subsampled.indexer.global_index.tensor

# The underlying tensors are also saved in pyro's param store
pyro_param = pyro.get_param_store()['generic_param']
assert (pyro_param - parameter.tensor == 0).all()




# TEST EFFECT CLASS UnknownParameter

# i) Invoke instance
node_structure = NodeStructure(UnknownParameter)
bias_object = UnknownParameter(node_structure, name = 'tutorial')
#
# ii) Produce bias value
bias = bias_object.forward()
#
# iii) Investigate object
bias_object.dtype_chain
bias_object.id
bias_object.node_structure
bias_object.node_structure.dims
render_1 = bias_object.render()
render_1
render_2 = bias_object.render_comp_graph()
render_2


# Introduce distributions as nodes with forward method and dims

CalipyNormal = calipy.core.dist.Normal
CalipyNormal.dists
CalipyNormal.input_vars
normal_ns = NodeStructure(CalipyNormal)
calipy_normal = CalipyNormal(node_structure = normal_ns, node_name = 'Normal')

calipy_normal.id
calipy_normal.node_structure
CalipyNormal.default_nodestructure

# Calling the forward method
normal_dims = normal_ns.dims['batch_dims'] + normal_ns.dims['event_dims']
normal_ns_sizes = normal_dims.sizes
mean = CalipyTensor(torch.zeros(normal_ns_sizes), normal_dims)
standard_deviation = CalipyTensor(torch.ones(normal_ns_sizes), normal_dims)
input_vars_normal = DataTuple(['loc', 'scale'], [mean, standard_deviation])
samples_normal = calipy_normal.forward(input_vars_normal)

# A more convenient way of creating the input_vars and observations data or
# at least getting the infor on the input signatures
create_input_vars = CalipyNormal.create_input_vars
help(create_input_vars)
input_vars_normal_2 = create_input_vars(loc = mean, scale = standard_deviation)
samples_normal_2 = calipy_normal.forward(input_vars_normal_2)
    


# sample_name = 'normal_sample'
# sample_dist = calipy_normal
# sample_ns = calipy_normal.node_structure
# sample_ns_sizes = sample_ns.dims['batch_dims'].sizes + sample_ns.dims['event_dims'].sizes
# sample_input_vars = DataTuple(['loc', 'scale'], [torch.zeros(sample_ns_sizes),
#                                                   torch.ones(sample_ns_sizes)])
# n_event = len(sample_ns.dims['event_dims'].sizes)
# sample_pyro_dist = calipy_normal.create_pyro_dist(sample_input_vars).to_event(n_event)
# sample_dist_dims = sample_ns.dims


# calipy_sample = sample(sample_name, sample_pyro_dist, sample_dist_dims, 
#                         observations = None, subsample_index = None, vectorizable = True)


# sample_name_bigger = 'normal_sample_bigger'
# sample_dist_bigger = calipy_normal
# sample_ns_bigger = calipy_normal.node_structure
# bigger_batch_dims = dim_assignment(['bd_1', 'bd_2'], dim_sizes = [10,6])
# bigger_event_dims = dim_assignment(['ed_1', 'ed_2'], dim_sizes = [4,2])
# sample_ns_bigger.set_dims(batch_dims = bigger_batch_dims, event_dims = bigger_event_dims)
# sample_ns_bigger_sizes = sample_ns_bigger.dims['batch_dims'].sizes + sample_ns_bigger.dims['event_dims'].sizes
# sample_input_vars_bigger = DataTuple(['loc', 'scale'], [torch.zeros(sample_ns_bigger_sizes),
#                                                  torch.ones(sample_ns_bigger_sizes)])
# n_event_bigger = len(sample_ns_bigger.dims['event_dims'].sizes)
# sample_pyro_dist_bigger = calipy_normal.create_pyro_dist(sample_input_vars_bigger).to_event(n_event_bigger)
# sample_dist_dims_bigger = sample_ns_bigger.dims


# calipy_sample_bigger = sample(sample_name_bigger, sample_pyro_dist_bigger, sample_dist_dims_bigger, 
#                        observations = None, subsample_index = None, vectorizable = True)




# TEST EFFECT CLASS NoiseAddition

# i) Invoke and investigate class
NoiseAddition = calipy.core.effects.NoiseAddition
help(NoiseAddition)
NoiseAddition.mro()
print(NoiseAddition.input_vars_schema)

# ii) Instantiate object
noise_ns = NodeStructure(NoiseAddition)
print(noise_ns)
print(noise_ns.dims)
noise_object = NoiseAddition(noise_ns)

# iii) Create arguments
noise_dims = noise_ns.dims['batch_dims'] + noise_ns.dims['event_dims']
mu = CalipyTensor(torch.zeros(noise_dims.sizes), noise_dims, 'mu')
sigma = CalipyTensor(torch.ones(noise_dims.sizes), noise_dims, 'sigma')
noise_input_vars = NoiseAddition.create_input_vars(mean = mu, standard_deviation = sigma)

# iv) Pass forward
noisy_output = noise_object.forward(input_vars = noise_input_vars, 
                                    observations = None, 
                                    subsample_index = None)
noisy_output.value.dims
help(noisy_output)



# TEST SIMPLE CalipyProbModel

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
data = {'sample' : data_cp}

# v) Define ProbModel
class MyProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def model(self, input_vars = None, observations = None):
        # Define the generative model
        mu = mu_object.forward()
        input_vars = {'mean' : mu, 'standard_deviation' : sigma}
        sample = noise_object.forward(input_vars, observations = observations)
        return sample

    def guide(self, input_vars = None, observations = None):
        # Define the guide (variational distribution)
        pass

# vi) Inference
prob_model = MyProbModel(name="example_model")
output = prob_model.model(observations = data)
optim_opts = {'n_steps' : 500, 'learning_rate' : 0.01}
prob_model.train(input_data = None, output_data = data, optim_opts = optim_opts)


















   