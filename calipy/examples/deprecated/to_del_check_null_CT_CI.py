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
from calipy.core.tensor import CalipyTensor, CalipyIndex
from calipy.core.utils import dim_assignment
from calipy.core.funs import calipy_cat

# Create data for initialization
tensor_dims = dim_assignment(['bd', 'ed'])
tensor_cp = CalipyTensor(torch.ones(6, 3), tensor_dims) 
tensor_none = None

index_full = tensor_cp.indexer.local_index
index_none = None

# Create and investigate null CalipyIndex

CI_none = CalipyIndex(None)
print(CI_none)
# Empty Index can be upgraded by extension
CI_expanded = CI_none.expand_to_dims(tensor_dims, [5,2])
# The following errors out, as intended: 
#   CalipyIndex(torch.ones([1]), index_tensor_dims = None)

# Passing a null index to CalipyTensor returns the orginal tensor.
tensor_cp[CI_none]
tensor_cp[CI_expanded]
# CT_none = CalipyTensor(None)

