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
from calipy.core.effects import CalipyQuantity, CalipyEffect, UnknownParameter
from calipy.core.data import DataTuple
from calipy.core.tensor import CalipyIndex, CalipyIndexer, TensorIndexer, CalipyTensor
from calipy.core.base import NodeStructure
from calipy.core.primitives import param

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
calipy_normal = CalipyNormal(node_structure = normal_ns)

calipy_normal.id
calipy_normal.node_structure
CalipyNormal.default_nodestructure




# Introduce sample function

def sample(name, dist, dist_dims, batch_dims, vectorizable=True, observations=None, subsample_index=None):
    """
    Flexible sampling function handling multiple plates and four cases based on obs and subsample_indices.

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
        
        
    Parameters:
    -----------
    name : str
        Base name for the sample site.
    dist : pyro.distributions.Distribution
        The distribution to sample from.
    dist_dims : list of CalipyDim objects
        The dimensions of the sample from the distribution; need to contain batch_dims as subset.
    batch_dims : list of CalipyDim objects
        The dimensions that act as batch dimensions and over which independence is assumed.
    vectorizable : bool, optional
        If True, uses vectorized sampling. If False, uses sequential sampling. Default is True.
    obs : CalipyObservation or None, optional
        Observations wrapped in CalipyObservation. If provided, sampling is conditioned on these observations.
    subsample_index : list of torch.Tensor or None, optional
        Subsample indices for each plate dimension. If provided, sampling is performed over these indices.

    Returns:
    --------
    CalipyTensor
        The sampled data, preserving batch and event dimensions.
    """
    
    # Basic rename
    obs = observations
    ssi = subsample_indices
    vec = vectorizable
    
    # Set up dimensions
    event_dims = dist_dims.delete_dims(batch_dims.names)
    dist_dim_sizes = dist_dims.sizes
    batch_dim_sizes = batch_dims.sizes
    event_dim_sizes = event_dims.sizes
    batch_dim_positions = dist_dims.find_indices(batch_dims.names)
    
    # Plate setup
    plate_names = [name + '_plate' for name in batch_dims.names]
    plate_sizes = [size if size is not None else 1 for size in batch_dims.sizes]


    # cases [1,x,x] vectorizable
    if vectorizable == True:
        # Vectorized sampling using pyro.plate
        with contextlib.ExitStack() as stack:
            # Determine dimensions for plates

            
            # case [0,0] (obs, ssi)
            if obs == None and ssi == None:
                pass
            
            # case [0,1] (obs, ssi)
            if obs == None and ssi is not None:
                pass
            
            
            # case [1,0] (obs, ssi)
            if obs is not None and ssi == None:
                pass
            
            # case [1,1] (obs, ssi)
            if obs is not None and ssi is not None:
                pass
            
            # Handle multiple plates
            for i, (plate_name, plate_size, dim) in enumerate(zip(plate_names, plate_sizes, batch_dim_positions)):
                subsample = subsample_indices if subsample_indices is not None else None
                size = plate_size
                stack.enter_context(pyro.plate(plate_name, size=size, subsample=subsample, dim=dim))

            # Sample data
            data = pyro.sample(name, dist, obs=current_obs)
            batch_shape = data.shape[:n_plates]
            return CalipySample(data, batch_shape, event_shape, vectorizable=True)

        
    # cases [0,x,x] nonvectorizable
    elif vectorizable == False:
            
            # case [0,0] (obs, ssi)
            if obs == None and ssi == None:
                # Create new observations of shape batch_shape_default with ssi
                # a flattened list of product(range(n_default))
                sample_names = a
                pass
            
            # case [0,1] (obs, ssi)
            if obs == None and ssi is not None:
                # Create len(ssi) new observations with given ssi's
                pass
            
            # case [1,0] (obs, ssi)
            if obs is not None and ssi == None:
                #  Create obs with standard ssi derived from obs batch_shape
                pass
            
            # case [1,1] (obs, ssi)
            if obs is not None and ssi is not None:
                # Create obs associated to given ssi's
                pass

    
            # Construct tensor from samples
            # Need to sort samples based on indices to ensure correct tensor shape
            sample_value = pyro.sample(sample_name, dist, obs=obs_value)
            
    return CalipySample(data, batch_shape, event_shape, vectorizable=vectorizable)
    
    
# iv) Test the classes

obs_object = CalipyObservation(data, plate_names, batch_shape=[n_data_1], event_shape=[2])

# Subsample indices (if any)
subsample_indices = index_list[0]

sample_results = dict()
# Sample using calipy_sample and check results
for vectorizable in [True, False]:
    for obs in [None, obs_block_list[0]]:
        for ssi in [None, subsample_indices]:
            
            sample_dist = pyro.distributions.MultivariateNormal(loc = mu_true, covariance_matrix = sigma_true)
            sample_result = calipy_sample('my_sample', sample_dist, plate_names, plate_sizes, vectorizable=vectorizable, obs=obs, subsample_indices=ssi)
            
            vflag = 1 if vectorizable == True else 0
            oflag = 1 if obs is not None else 0
            sflag = 1 if ssi is not None else 0
            
            print('vect_{}_obs_{}_ssi_{}_batch_shape'.format(vflag, oflag, sflag), sample_result.batch_shape )
            print('vect_{}_obs_{}_ssi_{}_event_shape'.format(vflag, oflag, sflag), sample_result.event_shape )
            print('vect_{}_obs_{}_ssi_{}_data_shape'.format(vflag, oflag, sflag), sample_result.data.shape )
            # print('vect_{}_obs_{}_ssi_{}_data'.format(vflag, oflag, sflag), sample_result.data )
            sample_results[tuple(vflag,oflag,sflag)] = sample_result
            



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
    batch_dims = dim_assignment(dim_names = ['batch_dim'], dim_sizes = [10])
    event_dims = dim_assignment(dim_names = ['event_dim'], dim_sizes = [2])
    batch_dims_description = 'The dims in which the noise is independent'
    event_dims_description = 'The dims in which the noise is copied and repeated'
    
    default_nodestructure = NodeStructure()
    default_nodestructure.set_dims(batch_dims = batch_dims,
                                    event_dims = event_dims)
    default_nodestructure.set_dim_descriptions(batch_dims = batch_dims_description,
                                                event_dims = event_dims_description)
    
    # Class initialization consists in passing args and building dims
    def __init__(self, node_structure, constraint = constraints.real, **kwargs):  
        super().__init__(**kwargs)
        self.node_structure = node_structure
        self.batch_dims = self.node_structure.dims['batch_dims']
        self.event_dims = self.node_structure.dims['event_dims']
        self.dims = self.event_dims + self.param_dims

        
    # Forward pass is passing input_vars and sampling from noise_dist
    def forward(self, input_vars, observations = None, subsample_index = None):
        """
        Create noisy samples using input_vars = (mean, standard_deviation) with
        shapes as indicated in the node_structures' plate_stack 'noise_stack' used
        for noisy_meas_object = NoiseAddition(node_structure).
        
        :param input vars: DataTuple with names ['mean', 'standard_deviation'] of tensors
            with equal (or at least broadcastable) shapes. 
        :type input_vars: DataTuple of instances of CalipyTensor
        :param observations: DataTuple with names ['observation;]
        :type observations: DataTuple of instance of CalipyTensor
        :param subsample_index: CalipyIndex indexing a subsample of the noisy
            samples.
        :type subsample_index: CalipyIndex
        :return: CalipyTensor representing simulation of a noisy measurement of
            the mean.
        :rtype: torch.Tensor
        """
        
        self.noise_dist = pyro.distributions.Normal(loc = input_vars[0], scale = input_vars[1])
        
        # Sample within independence context
        with context_plate_stack(self.plate_stack):
            output = pyro.sample('{}__noise_{}'.format(self.id_short, self.name), self.noise_dist, obs = observations)
        return output


# # iv) Test the classes


# # Sample using calipy_sample and check results
# for vectorizable in [True, False]:
#     for obs in [None, obs_block_list[0]]:
#         for ssi in [None, subsample_indices]:
            
#             sample_dist = pyro.distributions.MultivariateNormal(loc = mu_true, covariance_matrix = sigma_true)
#             sample_result = calipy_sample('my_sample', sample_dist, plate_names, plate_sizes, vectorizable=vectorizable, obs=obs, subsample_indices=ssi)
            
#             vflag = 1 if vectorizable == True else 0
#             oflag = 1 if obs is not None else 0
#             sflag = 1 if ssi is not None else 0
            
#             print('vect_{}_obs_{}_ssi_{}_batch_shape'.format(vflag, oflag, sflag), sample_result.batch_shape )
#             print('vect_{}_obs_{}_ssi_{}_event_shape'.format(vflag, oflag, sflag), sample_result.event_shape )
#             print('vect_{}_obs_{}_ssi_{}_data_shape'.format(vflag, oflag, sflag), sample_result.data.shape )
#             # print('vect_{}_obs_{}_ssi_{}_data'.format(vflag, oflag, sflag), sample_result.data )
#             sample_results[tuple(vflag,oflag,sflag)] = sample_result
            
            


# TEST EFFECT CLASS NoiseAddition


# ii) Build dataloader
class CalipySample:
    """
    Stores samples along with batch and event dimensions in a new object that 
    provides indexing and naming functionality.
    
    :param samples_datatuple: A DataTuple containing tensors considered as samples
    :type obs: DataTuple
    :param batch_dims_datatuple: A DataTuple containing the batch_dims DimTuple for each
        tensor in the sample_datatuple
    :type batch_dims_datatuple: DataTuple
    :param event_dims_datatuple: A DataTuple containing the event_dims DimTuple for each
        tensor in the sample_datatuple
    :type event_dims_datatuple: DataTuple
    :param subsample_indices: ?
    :type subsample_indices: ?
    :param vectorizable: Flag indicating if observations are vectorizable
    :type vectorizable: bool
    :return: CalipySample object 
    :rtype: CalipySample
    
    
    """
    def __init__(self, samples_datatuple, batch_dims_datatuple, event_dims_datatuple,
                 subsample_indices=None, vectorizable=True):

        # Metadata: keep it directly tied with samples
        self.entry_names = [key for key in samples_datatuple.keys()]
        self.batch_dims = batch_dims_datatuple
        self.event_dims = event_dims_datatuple
        self.sample_dims = self.batch_dims + self.event_dims
        # self.index_dims = DataTuple(self.entry_names, [dim_assignment(['id_{}'.format(key)],
        #                             [len(self.sample_dims[key])]) for key in self.entry_names])
        # self.ssi_dims = self.sample_dims + self.index_dims
        self.vectorizable = vectorizable
        
        # Handle tensor tuples for samples and ssi
        self.samples = samples_datatuple
        # self.samples_bound = self.samples.bind_dims(self.obs_dims)
        self.subsample_indices = subsample_indices
        # self.subsample_indices_bound = subsample_indices.bind_dims(self.ssi_dims) if subsample_indices is not None else None

        # Initialize local and global indices for easy reference
        
        self.samples.indexer_construct(self.sample_dims)
        index_names, indexers = zip(*[(name, tensor.calipy.indexer) for name, tensor in self.samples.items()])
        self.indexers = DataTuple(list(index_names), list(indexers))


    def get_entry(self, **batch_dims_spec):
        # Retrieve an observation by specifying batch dimensions explicitly
        indices = [batch_dims_spec[dim] for dim in self.batch_dims]
        obs_values = {key: tensor[tuple(indices)] if len(tensor) > 0 else None for key, tensor in self.observations.items()}
        obs_name = {key: self.index_to_name_dict[(key, tuple(indices))] for key in obs_values.keys()}
        return obs_name, obs_values

    def get_local_index(self, key, idx):
        return self.obs_local_indices[key][idx]

    def get_global_index(self, key, idx):
        return self.obs_global_indices[key][idx]

    def __repr__(self):
        repr_str = 'CalipySample object with samples: {} \nand sample dims : {}'\
            .format(self.samples.__repr__(), self.sample_dims.__repr__())
        return repr_str

# Instantiate CalipyObservation
obs_name_list = ['T_grid', 'T_rod']
observations = DataTuple(obs_name_list, [data_A, data_B])
batch_dims = DataTuple(obs_name_list, [batch_dims_A, batch_dims_B])
event_dims = DataTuple(obs_name_list, [event_dims_A, event_dims_B])
obs_dims = batch_dims + event_dims

calipy_obs = CalipySample(observations, batch_dims, event_dims)
print(calipy_obs)



class SubbatchDataset(Dataset):
    def __init__(self, data, subsample_sizes, batch_shape=None, event_shape=None):
        self.data = data
        self.batch_shape = batch_shape
        self.subsample_sizes = subsample_sizes

        # Determine batch_shape and event_shape
        if batch_shape is None:
            batch_dims = len(subsample_sizes)
            self.batch_shape = data.shape[:batch_dims]
        else:
            self.batch_shape = batch_shape

        if event_shape is None:
            self.event_shape = data.shape[len(self.batch_shape):]
        else:
            self.event_shape = event_shape

        # Compute number of blocks using ceiling division to include all data
        self.num_blocks = [
            (self.batch_shape[i] + subsample_sizes[i] - 1) // subsample_sizes[i]
            for i in range(len(subsample_sizes))
        ]
        self.block_indices = list(itertools.product(*[range(n) for n in self.num_blocks]))
        random.shuffle(self.block_indices)

    def __len__(self):
        return len(self.block_indices)

    def __getitem__(self, idx):
        block_idx = self.block_indices[idx]
        slices = []
        indices_ranges = []
        for i, (b, s) in enumerate(zip(block_idx, self.subsample_sizes)):
            start = b * s
            end = min(start + s, self.batch_shape[i])
            slices.append(slice(start, end))
            indices_ranges.append(torch.arange(start, end))
        # Include event dimensions in the slices
        slices.extend([slice(None)] * len(self.event_shape))
        meshgrid = torch.meshgrid(*indices_ranges, indexing='ij')
        indices = torch.stack(meshgrid, dim=-1).reshape(-1, len(self.subsample_sizes))
        obs_block = CalipyObservation(self.data[tuple(slices)], self.batch_shape, self.event_shape, subsample_indices = indices)
        return obs_block, indices

# Usage example
subsample_sizes = [4, 2]
subbatch_dataset = SubbatchDataset(data, subsample_sizes, batch_shape=batch_shape, event_shape=[2])
subbatch_dataloader = DataLoader(subbatch_dataset, batch_size=None)


index_list = []
obs_block_list = []
mean_list = []
for obs_block, index_tensor in subbatch_dataloader:
    print("Obs block shape:", obs_block.observations.shape)
    print("Index tensor shape:", index_tensor.shape)
    print("obs block:", obs_block)
    print("Indices:", index_tensor)
    index_list.append(index_tensor)
    obs_block_list.append(obs_block)
    mean_list.append(torch.mean(obs_block.observations,0))
    print("----")
    
   
    
# ----------------------------------------------------------------------------

# Practical Tests

# ---------------------------------------------------------------------------



def model(observations = None, vectorizable = True, subsample_indices = None):
    # observations is CalipyObservation object
    mu = pyro.param(name = 'mu', init_tensor = torch.tensor([5.0,5.0])) 
    sigma = pyro.param( name = 'sigma', init_tensor = 5*torch.eye(2), constraint = dist.constraints.positive_definite)
    
    plate_names = ['batch_dim_1']
    plate_sizes = [n_data_1]
    obs_dist = dist.MultivariateNormal(loc = mu, covariance_matrix = sigma)
    obs = calipy_sample('obs', obs_dist, plate_names, plate_sizes, vectorizable=vectorizable, obs = observations, subsample_indices = subsample_indices)
    
    return obs


# Example usage:
# Run the model with vectorizable=True
output_vectorized = model(vectorizable=True)
print("Vectorized Output Shape:", output_vectorized.data.shape)
print("Vectorized Output:", output_vectorized)

# Run the model with vectorizable=False
output_non_vectorized = model(vectorizable=False)
print("Non-Vectorized Output Shape:", output_non_vectorized.data.shape)
print("Non-Vectorized Output:", output_non_vectorized)

# Plot the shapes
calipy_observation = CalipyObservation(data, plate_names, batch_shape = batch_shape, event_shape = [2])
model_trace = pyro.poutine.trace(model)
spec_trace = model_trace.get_trace(calipy_observation, vectorizable = True, subsample_indices = None)
spec_trace.nodes
print(spec_trace.format_shapes())
            
            
# v) Perform inference in different configurations

# Set up guide
def guide(observations = None, vectorizable= True, subsample_indices = None):
    pass


adam = pyro.optim.Adam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# Set up svi
signature = [1,1,1] # vect, obs, ssi = None or values
# signature = [1,1,0]
# signature = [1,0,1]
# signature = [1,0,0]
# signature = [0,1,1] 
# signature = [0,1,0]
# signature = [0,0,1]
# signature = [0,0,0]


# Handle DataLoader case
loss_sequence = []
for epoch in range(1000):
    epoch_loss = 0
    for batch_data, subsample_indices in subbatch_dataloader:
        
        # Set up kwargs for svi step
        obs_svi_dict = {}
        obs_svi_dict['vectorizable'] = True if signature[0] ==1 else False
        obs_svi_dict['observations'] = batch_data if signature[1] == 1 else None
        obs_svi_dict['subsample_indices'] = subsample_indices if signature[2] == 1 else None
        
        # loss = svi.step(batch_data, True, subsample_indices)
        loss = svi.step(**obs_svi_dict)
        # loss = svi.step(observations = obs_svi_dict['observations'],
        #                 vectorizable = obs_svi_dict['observations'],
        #                 subsample_indices = obs_svi_dict['subsample_indices'])
        # loss = svi.step(obs_svi_dict['observations'],
        #                 obs_svi_dict['observations'],
        #                 obs_svi_dict['subsample_indices'])
        epoch_loss += loss
    
    epoch_loss /= len(subbatch_dataloader)
    loss_sequence.append(epoch_loss)

    if epoch % 100 == 0:
        print(f'epoch: {epoch} ; loss : {epoch_loss}')


"""
    5. Plots and illustrations
"""


# i) Print results

print('True mu = {}, True sigma = {} \n Inferred mu = {}, Inferred sigma = {} \n mean = {} \n mean of batch means = {}'
      .format(mu_true, sigma_true, 
              pyro.get_param_store()['mu'],
              pyro.get_param_store()['sigma'],
              torch.mean(data,0),
              torch.mean(torch.vstack(mean_list),0)))
































   