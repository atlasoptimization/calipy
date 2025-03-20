#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to check the behavior of the null objects that are 
formed when CalipyTensor and CalipyIndex are called on None. These should behave
in a nice way that reduces boilerplate code by avoiding recurring statements of
type x = y[z] if not y or z is None else W where x,y are CalipyTensors and z
is a CalipyIndex. The overall rule is that for null objects ct_n, ci_n formed 
from None, we want:
    1. ct[ci] works like usual
    2. ct_n[ci] = ct_n
    3. ct[ci_n] = ct
    4. ct_n[ci_n] = ct_n
    
This ensures that a null index has no subsampling effect and just returns the
original tensor and that a null tensor can be subsampled in whatever way and still
produces a null tensor. Practically this avoids the following constructs:
    1. subsample = tensor[ssi] if tensor is not None else None
    2. subsample = tensor[ssi] if ssi is not None else tensor
"""


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
single_io.value
single_io.calipy_dict
single_io.calipy_list
single_io.data_tuple


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

collated_io = io_obj.collate()


# Check indexing of tensors and compare to io_indexing
tensor_A = torch.ones([5])
tensor_A_dim = dim_assignment(['dim_1'])
tensor_A_cp = CalipyTensor(tensor_A, tensor_A_dim)
tensor_A_cp[0]

tensor_B = torch.ones([5,2])
tensor_B_dim = dim_assignment(['dim_1','dim_2'])
tensor_B_cp = CalipyTensor(tensor_B, tensor_B_dim)
tensor_B_cp[0,0]




# # Create data for CalipyDict initialization
# tensor_A = torch.ones(2, 3)
# tensor_B = torch.ones(4, 5)
# names = ['tensor_A', 'tensor_B']
# values = [tensor_A, tensor_B]
# data_tuple = DataTuple(names, values)
# data_dict = {'tensor_A': tensor_A, 'tensor_B' : tensor_B}

# # Create CalipyDict objects
# dict_from_none = CalipyDict()
# dict_from_dict = CalipyDict(data_dict)
# dict_from_tuple = CalipyDict(data_tuple)
# dict_from_calipy = CalipyDict(dict_from_dict)
# dict_from_single = CalipyDict(tensor_A)

# # Print contents and investigate 
# for cp_dict in [dict_from_none, dict_from_dict, dict_from_tuple, 
#                 dict_from_calipy, dict_from_single]:
#     print(cp_dict)
    
# dict_from_single.has_single_item()
# dict_from_single.value
# dict_from_dict.as_datatuple()








