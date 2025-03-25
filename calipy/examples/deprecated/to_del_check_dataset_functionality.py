#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to check the dataset functionality and investigate
typical types of homogeneous and inhomogeneous data as well as their representations
in terms of CalipyDatasets. We also check subbatching procedures and their
outputs. For this, do the following:
    1. Definitions and imports
    2. Create homogeneous data
    3. Build and investigate homogeneous dataset
    4. Create inhomogeneous data
    5. Build and investigate inhomogeneous dataset
    
The following scenarios need to be evaluated:
    i) (Input, Ouptut) =  (None, CalipyTensor)
    ii) (Input, Ouptut) =  (Calipytensor, CalipyTensor)
    iii) (Input, Ouptut) =  (None, dict(CalipyTensor))
    iv) (Input, Output) = (dict(CalipyTensor), dict(CalipyTensor))
    v) (Input, Ouptut) =  (None, list(dict(CalipyTensor)))
    vi) (Input, Output) = (list(dict(CalipyTensor)), list(dict(CalipyTensor)))
    vii) (Input, Output) = (None, list_mixed)
    viii) (Input, Output) = (list_mixed, list_mixed)
    
where list_mixed means a list of dicts with entries to keys sometimes being
None or of nonmatching shapes.
    
"""


"""
    1. Definitions and imports
"""


# i) Imports

import torch
import pyro

import calipy
from calipy.core.base import NodeStructure, CalipyProbModel
from calipy.core.effects import UnknownParameter, NoiseAddition
from calipy.core.utils import dim_assignment
from calipy.core.data import DataTuple, CalipyDict, CalipyIO, CalipyDataset, io_collate
from calipy.core.tensor import CalipyTensor, CalipyIndex
from calipy.core.funs import calipy_cat

from torch.utils.data import Dataset, DataLoader


# ii) Definitions

n_meas = 2
n_event = 1
n_subbatch = 7



"""
    2. Create data for dataset
"""

# i) Set up sample distributions

mu_true = torch.tensor(0.0)
sigma_true = torch.tensor(0.1)


# ii) Sample from distributions & wrap result

data_distribution = pyro.distributions.Normal(mu_true, sigma_true)
data = data_distribution.sample([n_meas, n_event])

data_dims = dim_assignment(['bd_data', 'ed_data'], dim_sizes = [n_meas, n_event])
data_cp = CalipyTensor(data, data_dims, name = 'data')

data_none = None
data_ct = data_cp
data_cd = {'a': data_cp, 'b' : data_cp}
data_io = [data_cd, data_cd]
data_io_mixed = [data_cd, {'a' : None, 'b' : data_cp} , {'a': data_cp, 'b':None}, data_cd]





"""
    3. Build datasets
"""


# i) Build dataset and check

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


# ii) Build dataloader and subsample

dataset = CalipyDataset(input_data = [None, data_ct, data_cd],
                        output_data = [None, data_ct, data_cd] )
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=io_collate)

# Iterate through the DataLoader
for batch_input, batch_output, batch_index in dataloader:
    print(batch_input, batch_output, batch_index)





"""
    4. Investigate datasets
"""


"""
    5. Build and investigate dataloaders
"""



