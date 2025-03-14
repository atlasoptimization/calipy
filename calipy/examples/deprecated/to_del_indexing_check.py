#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the indexing behavior for calipy tensors, especially annoying stuff like
if it drops dims of size 1.
"""


import torch
from calipy.core.tensor import CalipyTensor
from calipy.core.utils import dim_assignment

# Create DimTuples and tensors
data_torch = torch.normal(0,1,[10,5,3])
batch_dims = dim_assignment(dim_names = ['bd_1', 'bd_2'], dim_sizes = [10,5])
event_dims = dim_assignment(dim_names = ['ed_1'], dim_sizes = [3])
data_dims = batch_dims + event_dims
data_cp = CalipyTensor(data_torch, data_dims, name = 'data')

# # Access the single element where batch_dim 'bd_1' has the value 5
# data_cp_element_1 = data_cp.get_element(batch_dims[0:1], [5])
# assert((data_cp_element_1.tensor.squeeze() - data_cp.tensor[5,...] == 0).all())

# Access the single element where batch_dims has the value [5,2]
data_cp[5,2,0]
data_cp_element_2 = data_cp.get_element(batch_dims, [5,2])
assert((data_cp_element_2.tensor.squeeze() - data_cp.tensor[5,2,...] == 0).all())