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


# Check indexing of tensors again
aa = torch.ones([5])
aa_dim = dim_assignment(['aa'])
aa_cp = CalipyTensor(aa, aa_dim)
aa_cp[0]

bb = torch.ones([5,2])
bb_dim = dim_assignment(['aa','bb'])
bb_cp = CalipyTensor(bb, bb_dim)
bb_cp[0,0]


# Pass data into CalipyIO and investigate

# Legal input types are None, single object, dict, CalipyDict, DataTuple, list,
# CalipyList, CalipyIO. 

# Build inputs
none_input = None
single_input = torch.tensor([1.0])
dict_input = {'a': 1, 'b' : 2}
CalipyDict_input = CalipyDict(dict_input)
DataTuple_input = CalipyDict_input.as_datatuple()
list_input = [dict_input, {'c' : 3}]
CalipyList_input = CalipyList(list_input)
CalipyIO_input = CalipyIO(dict_input)

# Build CalipyIO's
none_io = CalipyIO(none_input)
single_io = CalipyIO(single_input)
dict_io = CalipyIO(dict_input)
CalipyDict_io = CalipyIO(CalipyDict_input)
DataTuple_io = CalipyIO(DataTuple_input)
list_io = CalipyIO(list_input)
CalipyList_io = CalipyIO(CalipyList_input)
CalipyIO_io = CalipyIO(CalipyIO_input)


# Check properties
print(single_io)
none_io.is_null
single_io.is_null

# subsampling
list_io[0]
list_io[0:2]
list_io[0].value

# Functionality includes:
#   1. Iteration
#   2. Fetch by index
#   3. Associated CalipyIndex
#      -  Has global and local index
#   4. Automatic reduction
#   5. Comes with collate function





