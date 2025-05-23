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
from calipy.core.data import DataTuple, CalipyDict
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



import torch
from calipy.core.tensor import CalipyTensor
from calipy.core.utils import dim_assignment

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



# Imports and definitions
import torch
from calipy.core.tensor import CalipyTensor
from calipy.core.utils import dim_assignment
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


# BEHAVIOR OF ADDITION, MULTIPLICATION, DIVISION

from calipy.core.tensor import broadcast_dims, CalipyTensor
from calipy.core.utils import dim_assignment


# Build torch tensors
a_int = 1
a_flt = 1.0
a_np = np.ones([])

a = torch.tensor(1.0)
b = torch.ones([2])
c = torch.ones([2,1])
d = torch.ones([2,3])
e = torch.ones([2,3,4])

# Create dims
dims_full = dim_assignment(['dim_1' , 'dim_2', 'dim_3'], [2,3,4],
                           dim_descriptions = ['first_dim' , 'second_dim' , 'third_dim'])

dim_1 = dims_full[0:1]
dim_2 = dims_full[1:2]
dim_3 = dims_full[2:3]

# Invoke CalipyTensors
a_cp = CalipyTensor(a)
b_cp = CalipyTensor(b, dim_1)
c_cp = CalipyTensor(c, dim_1 + dim_2)
d_cp = CalipyTensor(d, dim_1 + dim_2)
e_cp = CalipyTensor(e, dim_1 + dim_2 + dim_3)


# build broadcasted dims of CalipyTensors
dims_aa = broadcast_dims(a_cp.bound_dims, a_cp.bound_dims)[2]
dims_ab = broadcast_dims(a_cp.bound_dims, b_cp.bound_dims)[2]
dims_ac = broadcast_dims(a_cp.bound_dims, c_cp.bound_dims)[2]
dims_ad = broadcast_dims(a_cp.bound_dims, d_cp.bound_dims)[2]
dims_ae = broadcast_dims(a_cp.bound_dims, e_cp.bound_dims)[2]

dims_ba = broadcast_dims(b_cp.bound_dims, a_cp.bound_dims)[2]
dims_bb = broadcast_dims(b_cp.bound_dims, b_cp.bound_dims)[2]
dims_bc = broadcast_dims(b_cp.bound_dims, c_cp.bound_dims)[2]
dims_bd = broadcast_dims(b_cp.bound_dims, d_cp.bound_dims)[2]
dims_be = broadcast_dims(b_cp.bound_dims, e_cp.bound_dims)[2]

dims_ca = broadcast_dims(c_cp.bound_dims, a_cp.bound_dims)[2]
dims_cb = broadcast_dims(c_cp.bound_dims, b_cp.bound_dims)[2]
dims_cc = broadcast_dims(c_cp.bound_dims, c_cp.bound_dims)[2]
dims_cd = broadcast_dims(c_cp.bound_dims, d_cp.bound_dims)[2]
dims_ce = broadcast_dims(c_cp.bound_dims, e_cp.bound_dims)[2]

dims_da = broadcast_dims(d_cp.bound_dims, a_cp.bound_dims)[2]
dims_db = broadcast_dims(d_cp.bound_dims, b_cp.bound_dims)[2]
dims_dc = broadcast_dims(d_cp.bound_dims, c_cp.bound_dims)[2]
dims_dd = broadcast_dims(d_cp.bound_dims, d_cp.bound_dims)[2]
dims_de = broadcast_dims(d_cp.bound_dims, e_cp.bound_dims)[2]

dims_ea = broadcast_dims(e_cp.bound_dims, a_cp.bound_dims)[2]
dims_eb = broadcast_dims(e_cp.bound_dims, b_cp.bound_dims)[2]
dims_ec = broadcast_dims(e_cp.bound_dims, c_cp.bound_dims)[2]
dims_ed = broadcast_dims(e_cp.bound_dims, d_cp.bound_dims)[2]
dims_ee = broadcast_dims(e_cp.bound_dims, e_cp.bound_dims)[2]

# Check broadcasted dims
assert(dims_aa.sizes == [])
assert(dims_ab.sizes == [2])
assert(dims_ac.sizes == [2,1])
assert(dims_ad.sizes == [2,3])
assert(dims_ae.sizes == [2,3,4])

assert(dims_ba.sizes == [2])
# Different from pytorch broadcasting since matching by dims means extension of
# [dim1] by [dim1, dim2] to [dim1, dim2] with sizes [2,1]:
#   [dim1, dim2] : [2,1] by [dim1] [2] -> [dim1, dim2] [2,1]
assert(dims_bb.sizes == [2])
assert(dims_bc.sizes == [2,1])
assert(dims_bd.sizes == [2,3])
assert(dims_be.sizes == [2,3,4])

assert(dims_ca.sizes == [2,1])
assert(dims_cb.sizes == [2,1]) # Different: [dim1, dim2] : [2,1] by [dim1] [2] -> [dim1, dim2] [2,1]
assert(dims_cc.sizes == [2,1])
assert(dims_cd.sizes == [2,3])
assert(dims_ce.sizes == [2,3,4])

assert(dims_da.sizes == [2,3])
assert(dims_db.sizes == [2,3])
assert(dims_dc.sizes == [2,3])
assert(dims_dd.sizes == [2,3])
assert(dims_de.sizes == [2,3,4])

assert(dims_ea.sizes == [2,3,4])
assert(dims_eb.sizes == [2,3,4])
assert(dims_ec.sizes == [2,3,4])
assert(dims_ed.sizes == [2,3,4])
assert(dims_ee.sizes == [2,3,4])


# Special cases

# Interleaved dims : Finding dim, supersequence then extending to it
dims_p = dim_assignment(['dim_2', 'dim_3'], [3,4])
dims_q = dim_assignment(['dim_1', 'dim_2', 'dim_4'], [2,3,5])
dims_pq = broadcast_dims(dims_p, dims_q)

assert(dims_pq[2].sizes == [2,3,4,5]) # Should deliver result dim of shape [2,3,4,5]
assert(dims_pq[0].sizes == [1,3,4,1]) # Should deliver expanded dim_1 of shape [1,3,4,1]
assert(dims_pq[1].sizes == [2,3,1,5]) # Should deliver expanded dim_2 of shape [2,3,1,5]

# Check correctness of elementwise operations on CalipyTensors

# Addition +

# Build sums of two CalipyTensors
add_aa = a_cp + a_cp
add_ab = a_cp + b_cp
add_ac = a_cp + c_cp
add_ad = a_cp + d_cp
add_ae = a_cp + e_cp

add_ba = b_cp + a_cp
add_bb = b_cp + b_cp
add_bc = b_cp + c_cp
add_bd = b_cp + d_cp
add_be = b_cp + e_cp

add_ca = c_cp + a_cp
add_cb = c_cp + b_cp
add_cc = c_cp + c_cp
add_cd = c_cp + d_cp
add_ce = c_cp + e_cp

add_da = d_cp + a_cp
add_db = d_cp + b_cp
add_dc = d_cp + c_cp
add_dd = d_cp + d_cp
add_de = d_cp + e_cp

add_ea = e_cp + a_cp
add_eb = e_cp + b_cp
add_ec = e_cp + c_cp
add_ed = e_cp + d_cp
add_ee = e_cp + e_cp

# Check dims of sums
assert(add_aa.bound_dims.sizes == dims_aa.sizes)
assert(add_ab.bound_dims.sizes == dims_ab.sizes)
assert(add_ac.bound_dims.sizes == dims_ac.sizes)
assert(add_ad.bound_dims.sizes == dims_ad.sizes)
assert(add_ae.bound_dims.sizes == dims_ae.sizes)

assert(add_ba.bound_dims.sizes == dims_ba.sizes)
assert(add_bb.bound_dims.sizes == dims_bb.sizes)
assert(add_bc.bound_dims.sizes == dims_bc.sizes)
assert(add_bd.bound_dims.sizes == dims_bd.sizes)
assert(add_be.bound_dims.sizes == dims_be.sizes)

assert(add_ca.bound_dims.sizes == dims_ca.sizes)
assert(add_cb.bound_dims.sizes == dims_cb.sizes)
assert(add_cc.bound_dims.sizes == dims_cc.sizes)
assert(add_cd.bound_dims.sizes == dims_cd.sizes)
assert(add_ce.bound_dims.sizes == dims_ce.sizes)

assert(add_da.bound_dims.sizes == dims_da.sizes)
assert(add_db.bound_dims.sizes == dims_db.sizes)
assert(add_dc.bound_dims.sizes == dims_dc.sizes)
assert(add_dd.bound_dims.sizes == dims_dd.sizes)
assert(add_de.bound_dims.sizes == dims_de.sizes)

assert(add_ea.bound_dims.sizes == dims_ea.sizes)
assert(add_eb.bound_dims.sizes == dims_eb.sizes)
assert(add_ec.bound_dims.sizes == dims_ec.sizes)
assert(add_ed.bound_dims.sizes == dims_ed.sizes)
assert(add_ee.bound_dims.sizes == dims_ee.sizes)



# Multiplication * of CalipyTensors

mult_aa = a_cp * a_cp
mult_ab = a_cp * b_cp
mult_ac = a_cp * c_cp
mult_ad = a_cp * d_cp
mult_ae = a_cp * e_cp

mult_ba = b_cp * a_cp
mult_bb = b_cp * b_cp
mult_bc = b_cp * c_cp
mult_bd = b_cp * d_cp
mult_be = b_cp * e_cp

mult_ca = c_cp * a_cp
mult_cb = c_cp * b_cp
mult_cc = c_cp * c_cp
mult_cd = c_cp * d_cp
mult_ce = c_cp * e_cp

mult_da = d_cp * a_cp
mult_db = d_cp * b_cp
mult_dc = d_cp * c_cp
mult_dd = d_cp * d_cp
mult_de = d_cp * e_cp

mult_ea = e_cp * a_cp
mult_eb = e_cp * b_cp
mult_ec = e_cp * c_cp
mult_ed = e_cp * d_cp
mult_ee = e_cp * e_cp

# Check dims of products
assert(mult_aa.bound_dims.sizes == dims_aa.sizes)
assert(mult_ab.bound_dims.sizes == dims_ab.sizes)
assert(mult_ac.bound_dims.sizes == dims_ac.sizes)
assert(mult_ad.bound_dims.sizes == dims_ad.sizes)
assert(mult_ae.bound_dims.sizes == dims_ae.sizes)

assert(mult_ba.bound_dims.sizes == dims_ba.sizes)
assert(mult_bb.bound_dims.sizes == dims_bb.sizes)
assert(mult_bc.bound_dims.sizes == dims_bc.sizes)
assert(mult_bd.bound_dims.sizes == dims_bd.sizes)
assert(mult_be.bound_dims.sizes == dims_be.sizes)

assert(mult_ca.bound_dims.sizes == dims_ca.sizes)
assert(mult_cb.bound_dims.sizes == dims_cb.sizes)
assert(mult_cc.bound_dims.sizes == dims_cc.sizes)
assert(mult_cd.bound_dims.sizes == dims_cd.sizes)
assert(mult_ce.bound_dims.sizes == dims_ce.sizes)

assert(mult_da.bound_dims.sizes == dims_da.sizes)
assert(mult_db.bound_dims.sizes == dims_db.sizes)
assert(mult_dc.bound_dims.sizes == dims_dc.sizes)
assert(mult_dd.bound_dims.sizes == dims_dd.sizes)
assert(mult_de.bound_dims.sizes == dims_de.sizes)

assert(mult_ea.bound_dims.sizes == dims_ea.sizes)
assert(mult_eb.bound_dims.sizes == dims_eb.sizes)
assert(mult_ec.bound_dims.sizes == dims_ec.sizes)
assert(mult_ed.bound_dims.sizes == dims_ed.sizes)
assert(mult_ee.bound_dims.sizes == dims_ee.sizes)



# Division / of CalipyTensors

div_aa = a_cp / a_cp
div_ab = a_cp / b_cp
div_ac = a_cp / c_cp
div_ad = a_cp / d_cp
div_ae = a_cp / e_cp

div_ba = b_cp / a_cp
div_bb = b_cp / b_cp
div_bc = b_cp / c_cp
div_bd = b_cp / d_cp
div_be = b_cp / e_cp

div_ca = c_cp / a_cp
div_cb = c_cp / b_cp
div_cc = c_cp / c_cp
div_cd = c_cp / d_cp
div_ce = c_cp / e_cp

div_da = d_cp / a_cp
div_db = d_cp / b_cp
div_dc = d_cp / c_cp
div_dd = d_cp / d_cp
div_de = d_cp / e_cp

div_ea = e_cp / a_cp
div_eb = e_cp / b_cp
div_ec = e_cp / c_cp
div_ed = e_cp / d_cp
div_ee = e_cp / e_cp

# Check dims of products
assert(div_aa.bound_dims.sizes == dims_aa.sizes)
assert(div_ab.bound_dims.sizes == dims_ab.sizes)
assert(div_ac.bound_dims.sizes == dims_ac.sizes)
assert(div_ad.bound_dims.sizes == dims_ad.sizes)
assert(div_ae.bound_dims.sizes == dims_ae.sizes)

assert(div_ba.bound_dims.sizes == dims_ba.sizes)
assert(div_bb.bound_dims.sizes == dims_bb.sizes)
assert(div_bc.bound_dims.sizes == dims_bc.sizes)
assert(div_bd.bound_dims.sizes == dims_bd.sizes)
assert(div_be.bound_dims.sizes == dims_be.sizes)

assert(div_ca.bound_dims.sizes == dims_ca.sizes)
assert(div_cb.bound_dims.sizes == dims_cb.sizes)
assert(div_cc.bound_dims.sizes == dims_cc.sizes)
assert(div_cd.bound_dims.sizes == dims_cd.sizes)
assert(div_ce.bound_dims.sizes == dims_ce.sizes)

assert(div_da.bound_dims.sizes == dims_da.sizes)
assert(div_db.bound_dims.sizes == dims_db.sizes)
assert(div_dc.bound_dims.sizes == dims_dc.sizes)
assert(div_dd.bound_dims.sizes == dims_dd.sizes)
assert(div_de.bound_dims.sizes == dims_de.sizes)

assert(div_ea.bound_dims.sizes == dims_ea.sizes)
assert(div_eb.bound_dims.sizes == dims_eb.sizes)
assert(div_ec.bound_dims.sizes == dims_ec.sizes)
assert(div_ed.bound_dims.sizes == dims_ed.sizes)
assert(div_ee.bound_dims.sizes == dims_ee.sizes)



# Now do the same for tensors with generic dimensions or standard torch tensors
# Under the hood, torch.tensors a are wrapped in a_cp = CalipyTensor(a) which 
# produces a CalipyTensor with generic dims, so the expressions a_cp + b_cp and
# CalipyTensor(a) + b_cp are equal.
# The results are equivalent to what standard pytorch produces during broadcasting

# Invoke CalipyTensors
a_gcp = CalipyTensor(a)
b_gcp = CalipyTensor(b)
c_gcp = CalipyTensor(c)
d_gcp = CalipyTensor(d)
e_gcp = CalipyTensor(e)


# build broadcasted dims of CalipyTensors
dims_aag = broadcast_dims(a_cp.bound_dims, a_gcp.bound_dims)[2]
dims_abg = broadcast_dims(a_cp.bound_dims, b_gcp.bound_dims)[2]
dims_acg = broadcast_dims(a_cp.bound_dims, c_gcp.bound_dims)[2]
dims_adg = broadcast_dims(a_cp.bound_dims, d_gcp.bound_dims)[2]
dims_aeg = broadcast_dims(a_cp.bound_dims, e_gcp.bound_dims)[2]

dims_bag = broadcast_dims(b_cp.bound_dims, a_gcp.bound_dims)[2]
dims_bbg = broadcast_dims(b_cp.bound_dims, b_gcp.bound_dims)[2]
dims_bcg = broadcast_dims(b_cp.bound_dims, c_gcp.bound_dims)[2]
# dims_bdg = broadcast_dims(b_cp.bound_dims, d_gcp.bound_dims)[2] # Not broadcastable: [2], [2,3]
# dims_beg = broadcast_dims(b_cp.bound_dims, e_gcp.bound_dims)[2] # Not broadcastable: [2], [2,3,4]

dims_cag = broadcast_dims(c_cp.bound_dims, a_gcp.bound_dims)[2]
dims_cbg = broadcast_dims(c_cp.bound_dims, b_gcp.bound_dims)[2]
dims_ccg = broadcast_dims(c_cp.bound_dims, c_gcp.bound_dims)[2]
dims_cdg = broadcast_dims(c_cp.bound_dims, d_gcp.bound_dims)[2]
# dims_ceg = broadcast_dims(c_cp.bound_dims, e_gcp.bound_dims)[2] # Not broadcastable: [2,1], [2,3,4]

dims_dag = broadcast_dims(d_cp.bound_dims, a_gcp.bound_dims)[2]
# dims_dbg = broadcast_dims(d_cp.bound_dims, b_gcp.bound_dims)[2] # Not broadcastable: [2,3], [2]
dims_dcg = broadcast_dims(d_cp.bound_dims, c_gcp.bound_dims)[2]
dims_ddg = broadcast_dims(d_cp.bound_dims, d_gcp.bound_dims)[2]
# dims_deg = broadcast_dims(d_cp.bound_dims, e_gcp.bound_dims)[2] # Not broadcastable: [2,3], [2,3,4]

dims_eag = broadcast_dims(e_cp.bound_dims, a_gcp.bound_dims)[2]
# dims_ebg = broadcast_dims(e_cp.bound_dims, b_gcp.bound_dims)[2] # Not broadcastable: [2,3,4], [2]
# dims_ecg = broadcast_dims(e_cp.bound_dims, c_gcp.bound_dims)[2] # Not broadcastable: [2,3,4], [2,1]
# dims_edg = broadcast_dims(e_cp.bound_dims, d_gcp.bound_dims)[2] # Not broadcastable: [2,3,4], [2,3]
dims_eeg = broadcast_dims(e_cp.bound_dims, e_gcp.bound_dims)[2]


# Check broadcasted dims
assert(dims_aag.sizes == [])
assert(dims_abg.sizes == [2])
assert(dims_acg.sizes == [2,1])
assert(dims_adg.sizes == [2,3])
assert(dims_aeg.sizes == [2,3,4])

assert(dims_bag.sizes == [2])
assert(dims_bbg.sizes == [2])

# Different from cp broadcasting since matching from right means extension of
# [2,1] by [2] to [2,2]
assert(dims_bcg.sizes == [2,2]) 

assert(dims_cag.sizes == [2,1])
assert(dims_cbg.sizes == [2,2]) # Different: [2,1] by [2] -> [2,2]
assert(dims_ccg.sizes == [2,1])
assert(dims_cdg.sizes == [2,3])

assert(dims_dag.sizes == [2,3])
assert(dims_dcg.sizes == [2,3])
assert(dims_ddg.sizes == [2,3])

assert(dims_eag.sizes == [2,3,4])
assert(dims_eeg.sizes == [2,3,4])


# Build sums of CalipyTensors with torch.tensors
add_aat = a_cp + a
add_abt = a_cp + b
add_act = a_cp + c
add_adt = a_cp + d
add_aet = a_cp + e

add_bat = b_cp + a
add_bbt = b_cp + b
add_bct = b_cp + c
# add_bdt = b_cp + d # Not broadcastable: [2], [2,3]
# add_bet = b_cp + e # Not broadcastable: [2], [2,3,4]

add_cat = c_cp + a
add_cbt = c_cp + b
add_cct = c_cp + c
add_cdt = c_cp + d
# add_cet = c_cp + e # Not broadcastable: [2,1], [2,3,4]

add_dat = d_cp + a
# add_dbt = d_cp + b # Not broadcastable: [2,3], [2]
add_dct = d_cp + c
add_ddt = d_cp + d
# add_det = d_cp + e # Not broadcastable: [2,3], [2,3,4]

add_eat = e_cp + a
# add_ebt = e_cp + b # Not broadcastable: [2,3,4], [2]
# add_ect = e_cp + c # Not broadcastable: [2,3,4], [2,1]
# add_edt = e_cp + d # Not broadcastable: [2,3,4], [2,3]
add_eet = e_cp + e

# Check dims of sums
assert(add_aat.bound_dims.sizes == dims_aag.sizes)
assert(add_abt.bound_dims.sizes == dims_abg.sizes)
assert(add_act.bound_dims.sizes == dims_acg.sizes)
assert(add_adt.bound_dims.sizes == dims_adg.sizes)
assert(add_aet.bound_dims.sizes == dims_aeg.sizes)

assert(add_bat.bound_dims.sizes == dims_bag.sizes)
assert(add_bbt.bound_dims.sizes == dims_bbg.sizes)
assert(add_bct.bound_dims.sizes == dims_bcg.sizes)

assert(add_cat.bound_dims.sizes == dims_cag.sizes)
assert(add_cbt.bound_dims.sizes == dims_cbg.sizes)
assert(add_cct.bound_dims.sizes == dims_ccg.sizes)
assert(add_cdt.bound_dims.sizes == dims_cdg.sizes)

assert(add_dat.bound_dims.sizes == dims_dag.sizes)
assert(add_dct.bound_dims.sizes == dims_dcg.sizes)
assert(add_ddt.bound_dims.sizes == dims_ddg.sizes)

assert(add_eat.bound_dims.sizes == dims_eag.sizes)
assert(add_eet.bound_dims.sizes == dims_eeg.sizes)

# Left addition and right addition are equal since by definition in CalipyTensor.__add__
# we have __add__(self, other) = __radd__(self, other)

assert(((a_cp + e_cp).tensor == (e_cp + a_cp).tensor).all())

# Addition, Multplication, Division also works naturally with Python integer and floats

# Addition
2 + b_cp
b_cp + 2
2.0 + b_cp
b_cp + 2.0
torch.tensor(np.ones([2])) + b_cp
b_cp + torch.tensor(np.ones([2]))

# Multiplication
2 * b_cp
b_cp * 2
2.0 * b_cp
b_cp * 2.0
torch.tensor(np.ones([2])) * b_cp
b_cp * torch.tensor(np.ones([2]))

# Division
2 / b_cp
b_cp / 2
2.0 / b_cp
b_cp / 2.0
torch.tensor(2*np.ones([2])) / b_cp
b_cp / torch.tensor(2*np.ones([2]))


# Numpy arrays behave different because they have their own __add__ methods.
b_cp + np.ones([2]) # This works partially
# np.ones([2]) + b_cp # This errors out since addition not defined in np
b_cp * np.ones([2]) # This works partially
# np.ones([2]) * b_cp # This errors out since addition not defined in np
b_cp / np.ones([2]) # This works partially
# np.ones([2]) / b_cp # This errors out since addition not defined in np

# If the tensors cannot be broadcasted together, an exception is raised:
# d_cp + CalipyTensor(torch.ones([20,20]) , dim_assignment(['dim_1', 'dim_2'])) # Shape mismatch
# d_cp + CalipyTensor(torch.ones([20,20]) , dim_assignment(['dim_2', 'dim_1'])) # Dim order mismatch

# Similar things hold also for multiplication and division.



# NULL OBJECTS

# CalipyTensors and CalipyIndex also work with None inputs to produce Null objects
# Create data for initialization
tensor_dims = dim_assignment(['bd', 'ed'])
tensor_cp = CalipyTensor(torch.ones(6, 3), tensor_dims) 
tensor_none = None

index_full = tensor_cp.indexer.local_index
index_none = None

# ii) Create and investigate null CalipyIndex
CI_none = CalipyIndex(None)
print(CI_none)
CI_expanded = CI_none.expand_to_dims(tensor_dims, [5,2])

# Passing a null index to CalipyTensor returns the orginal tensor.
tensor_cp[CI_none]
tensor_cp[CI_expanded]
# The following errors out, as intended: 
#   CalipyIndex(torch.ones([1]), index_tensor_dims = None)

# iii) Create and investigate null CalipyTensor
CT_none = CalipyTensor(None)
CT_none
CT_none[CI_none] 
CT_none[CI_expanded]

tensor_dims_bound = tensor_dims.bind(tensor_cp.shape)
CT_expanded = CT_none.expand_to_dims(tensor_dims_bound)
# The following errors out, as intended: 
#   CalipyIndex(torch.ones([1]), index_tensor_dims = None)


# SPECIAL CREATION RULES

# Special creation rules for calipy tensors:
#   i) If tensor is None, dims must be None. Produces null object
#   ii) If tensor exists and dims are None. Produces calipy tensor with generic dims
#   iii) If calipy tensor is passed as input. Produces the same calipy tensor
#   iv) If calipy tensor is passd as input and some dims. Produces new calipy tensor with new dims.

tensor_A = torch.ones([5,2])
dims_A = dim_assignment(['bd', 'ed'])
dims_A_alt = dim_assignment(['bd_alt', 'ed_alt'])
tensor_A_cp = CalipyTensor(tensor_A, dims_A)

tensor_cp_None = CalipyTensor(None)
tensor_cp_default = CalipyTensor(tensor_A)
tensor_cp_idempotent = CalipyTensor(tensor_A_cp)
tensor_cp_alt = CalipyTensor(tensor_A_cp, dims_A_alt)
print(tensor_cp_alt)




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



# TEST CALIPYDICT, CALIPYLIST, CALIPYIO

import torch
from calipy.core.data import DataTuple, CalipyDict, CalipyList, CalipyIO
   

# Create data for CalipyList
calipy_list_empty = CalipyList()
calipy_list = CalipyList(data = ['a','b'])
calipy_same_list = CalipyList(calipy_list)


# Create data for CalipyDict initialization
tensor_A = torch.ones(2, 3)
tensor_B = torch.ones(4, 5)
names = ['tensor_A', 'tensor_B']
values = [tensor_A, tensor_B]
data_tuple = DataTuple(names, values)
data_dict = {'tensor_A': tensor_A, 'tensor_B' : tensor_B}

# Create CalipyDict objects
dict_from_none = CalipyDict()
dict_from_dict = CalipyDict(data_dict)
dict_from_tuple = CalipyDict(data_tuple)
dict_from_calipy = CalipyDict(dict_from_dict)
dict_from_single = CalipyDict(tensor_A)

# Print contents and investigate 
for cp_dict in [dict_from_none, dict_from_dict, dict_from_tuple, 
                dict_from_calipy, dict_from_single]:
    print(cp_dict)
    
dict_from_single.has_single_item
dict_from_single.value
dict_from_dict.as_datatuple()


# Imports and definitions
import torch
from calipy.core.data import DataTuple, CalipyDict, CalipyList, CalipyIO
from calipy.core.tensor import CalipyTensor
from calipy.core.utils import dim_assignment
   

# Create data for CalipyList
calipy_list_empty = CalipyList()
calipy_list = CalipyList(data = ['a','b'])
calipy_same_list = CalipyList(calipy_list)


# Pass data into CalipyIO and investigate

# Legal input types are None, single object, dict, CalipyDict, DataTuple, list,
# CalipyList, CalipyIO. 

# Build inputs
none_input = None
single_input = torch.tensor([1.0])
dict_input = {'a': 1, 'b' : 2}
CalipyDict_input = CalipyDict(dict_input)
DataTuple_input = CalipyDict_input.as_datatuple()
list_input = [dict_input, {'c' : 3}, {'d' : 4}]
CalipyList_input = CalipyList(list_input)
CalipyIO_input = CalipyIO(dict_input)

# Build CalipyIO's
none_io = CalipyIO(none_input)
single_io = CalipyIO(single_input)
dict_io = CalipyIO(dict_input)
CalipyDict_io = CalipyIO(CalipyDict_input)
DataTuple_io = CalipyIO(DataTuple_input)
list_io = CalipyIO(list_input, name = 'io_from_list')
CalipyList_io = CalipyIO(CalipyList_input)
CalipyIO_io = CalipyIO(CalipyIO_input)


# Check properties
none_io.is_null
single_io.is_null
print(single_io)

    
# Functionality includes:
#   1. Iteration
#   2. Fetch by index
#   3. Associated CalipyIndex
#      -  Has global and local index
#   4. Comes with collate function

# 1. Iteration
# Proceed to investigate one of the built calipy_io objects, here list_io
for io in list_io:
    print(io)
    print(io.indexer.global_index)

# 2. Fetch by index
# Access values (special if list and dict only have 1 element)
single_io.dict
single_io.value
single_io.calipy_dict
single_io.calipy_list
single_io.data_tuple
single_io['__single__']
list_io[0]['a']

# 3. a) Associated Indexer
# Content of indexer
list_io.batch_dim_flattened
list_io.indexer
list_io.indexer.local_index
list_io_sub = list_io[1:2]
list_io_sub.indexer.data_source_name
list_io_sub.indexer.index_tensor_dims

# 3. b) Associated CalipyIndex
# Content of specific IOIndexer
list_io_sub.indexer.local_index.tuple
list_io_sub.indexer.local_index.tensor
list_io_sub.indexer.local_index.index_name_dict

list_io_sub.indexer.global_index.tuple
list_io_sub.indexer.global_index.tensor
list_io_sub.indexer.global_index.index_name_dict

# Iteration produces sub_io's
for io in list_io:
    print(io.indexer.global_index)
    print(io.indexer.global_index.tensor)

# 3. c) Index / IO interaction
# subsampling and indexing: via intes, tuples, slices, and CalipyIndex
sub_io_1 = list_io[0]
sub_io_2 = list_io[1]
sub_io_3 = list_io[1:3]

sub_io_1.indexer.local_index
sub_io_2.indexer.local_index
sub_io_3.indexer.local_index

sub_io_1.indexer.global_index
sub_io_2.indexer.global_index
sub_io_3.indexer.global_index

global_index_1 = sub_io_1.indexer.global_index
global_index_2 = sub_io_2.indexer.global_index
global_index_3 = sub_io_3.indexer.global_index

assert(list_io[global_index_1] == list_io[0])
assert(list_io[global_index_2] == list_io[1])
assert(list_io[global_index_3] == list_io[1:3])

# 4. Collate function
# Check collation functionality for autoreducing io s
mean_dims = dim_assignment(['bd_1', 'ed_1'])
var_dims = dim_assignment(['bd_1', 'ed_1'])

mean_1 = CalipyTensor(torch.randn(3, 2), mean_dims)
mean_2 = CalipyTensor(torch.randn(5, 2), mean_dims)
var_1 = CalipyTensor(torch.randn(3, 2), var_dims)
var_2 = CalipyTensor(torch.randn(5, 2), var_dims)

io_obj = CalipyIO([
    CalipyDict({'mean': mean_1, 'var': var_1}),
    CalipyDict({'mean': mean_2, 'var': var_2})
])

collated_io = io_obj.reduce_list()

# Rename all entries in the dicts in CalipyIO
rename_dict = {'a' : 'new_a', 'b' : 'new_b'}
renamed_io = list_io.rename_keys(rename_dict)


# TEST DATASET AND DATALOADER

# i) Imports and definitions
import torch
import pyro        
from calipy.core.utils import dim_assignment
from calipy.core.data import  CalipyDataset, io_collate
from calipy.core.tensor import CalipyTensor
from torch.utils.data import DataLoader
        
# Definitions        
n_meas = 2
n_event = 1
n_subbatch = 7


# ii) Create data for dataset

# Set up sample distributions
mu_true = torch.tensor(0.0)
sigma_true = torch.tensor(0.1)

# Sample from distributions & wrap result
data_distribution = pyro.distributions.Normal(mu_true, sigma_true)
data = data_distribution.sample([n_meas, n_event])
data_dims = dim_assignment(['bd_data', 'ed_data'], dim_sizes = [n_meas, n_event])
data_cp = CalipyTensor(data, data_dims, name = 'data')

# dataset_inputs
data_none = None
data_ct = data_cp
data_cd = {'a': data_cp, 'b' : data_cp}
data_io = [data_cd, data_cd]
data_io_mixed = [data_cd, {'a' : None, 'b' : data_cp} , {'a': data_cp, 'b':None}, data_cd]


# iii) Build datasets

# Build datasets and check
dataset_none_none = CalipyDataset(input_data = data_none, output_data = data_none)
dataset_none_ct = CalipyDataset(input_data = data_none, output_data = data_ct)
dataset_none_cd = CalipyDataset(input_data = data_none, output_data = data_cd)
dataset_none_io = CalipyDataset(input_data = data_none, output_data = data_io)
dataset_none_iomixed = CalipyDataset(input_data = data_none, output_data = data_io_mixed)

dataset_ct_ct = CalipyDataset(input_data = data_ct, output_data = data_ct)
dataset_ct_cd = CalipyDataset(input_data = data_ct, output_data = data_cd)
dataset_ct_io = CalipyDataset(input_data = data_ct, output_data = data_io)
dataset_ct_iomixed = CalipyDataset(input_data = data_ct, output_data = data_io_mixed)

dataset_cd_ct = CalipyDataset(input_data = data_cd, output_data = data_ct)
dataset_cd_cd = CalipyDataset(input_data = data_cd, output_data = data_cd)
dataset_cd_io = CalipyDataset(input_data = data_cd, output_data = data_io)
dataset_cd_iomixed = CalipyDataset(input_data = data_cd, output_data = data_io_mixed)

dataset_io_ct = CalipyDataset(input_data = data_io, output_data = data_ct)
dataset_io_cd = CalipyDataset(input_data = data_io, output_data = data_cd)
dataset_io_io = CalipyDataset(input_data = data_io, output_data = data_io)
dataset_io_iomixed = CalipyDataset(input_data = data_io, output_data = data_io_mixed)

dataset_iomixed_ct = CalipyDataset(input_data = data_io_mixed, output_data = data_ct)
dataset_iomixed_cd = CalipyDataset(input_data = data_io_mixed, output_data = data_cd)
dataset_iomixed_io = CalipyDataset(input_data = data_io_mixed, output_data = data_io)
dataset_iomixed_iomixed = CalipyDataset(input_data = data_io_mixed, output_data = data_io_mixed)


# iv) Build dataloader and subsample

dataset = CalipyDataset(input_data = [None, data_ct, data_cd],
                        output_data = [None, data_ct, data_cd] )
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=io_collate)

# Iterate through the DataLoader
for batch_input, batch_output, batch_index in dataloader:
    print(batch_input, batch_output, batch_index)



# TEST SAMPLE FUNCTION

# General distribution setup
CalipyNormal = calipy.core.dist.Normal
normal_ns = NodeStructure(CalipyNormal)
calipy_normal = CalipyNormal(node_structure = normal_ns, node_name = 'Normal')

dims_normal = normal_ns.dims['batch_dims'] + normal_ns.dims['event_dims']
mean_cp = CalipyTensor(torch.zeros(dims_normal.sizes), dims_normal)
sigma_cp = CalipyTensor(torch.ones(dims_normal.sizes), dims_normal)
input_vars_normal = CalipyDict({'loc' : mean_cp, 'scale' : sigma_cp})
pyro_normal = calipy_normal.create_pyro_dist(input_vars_normal).to_event(1)

# Data generation
data = torch.normal(10,1, [10,2])
data_cp = CalipyTensor(data, dims_normal, 'data')
data_ss, data_ssi = data_cp.indexer.simple_subsample(dims_normal[0],3)
data_subsample_cp = data_ss[0]
data_ssi = data_ssi[0]

# Sample vectorizable, no observations, no subsample_indices
vec = True
obs = None
ssi = None

sample_result_100 = sample('Sample_100', pyro_normal, normal_ns.dims, observations = obs,
                       subsample_index = ssi, vectorizable = vec)

# Sample vectorizable, no observations, subsample_indices
vec = True
obs = None
ssi = data_ssi

sample_result_101 = sample('Sample_101', pyro_normal, normal_ns.dims, observations = obs,
                       subsample_index = ssi, vectorizable = vec)


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
data = {'sample'}

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
































   