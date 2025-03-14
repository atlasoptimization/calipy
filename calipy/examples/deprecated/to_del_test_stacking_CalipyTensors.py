#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Short script to test stacking CalipyTensors
"""

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