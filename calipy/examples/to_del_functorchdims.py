#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Describe the functorch.dims issue
"""

import torch
from functorch.dim import dims

# current behavior leads to confusing output.
batch_dims = dims(sizes = [10,5])
event_dims = dims(sizes = [3,2])

print(batch_dims)                           # produces (d0, d1)
print(event_dims)                           # produces (d0, d1)

print([dim.size for dim in batch_dims])     # produces [10,5]
print([dim.size for dim in event_dims])     # produces [3,2]

# This is confusing as there exist now two dimensions d0 with different properties

# The issue becomes even clearer, when using the dimensions to define tensors
# with first class dims:

some_dims = dims(2)
unrelated_dims = dims(2)

A = torch.randn([2,2])
A_fc_1 = A[some_dims]   
A_fc_2 = A[unrelated_dims]   

print(A_fc_1)       # produces tensor([[ ...]]) with dims=(d0,d1), sizes=(2,2)
print(A_fc_2)       # produces tensor([[ ...]]) with dims=(d0,d1), sizes=(2,2)

# this makes it look like there the same dimensions are indexing A_fc_1 and A_fc_2
# even though this is not the case and i am unsure about the behavior of e.g. the
# einsum-style ops.


# The following is envisioned after resolving the issue
batch_dims = dims(sizes = [10,5], names = ['bd_1', 'bd_2'])
event_dims = dims(sizes = [3,2], names = ['ed_1', 'ed_2'])

print(batch_dims)    # produces (bd_1, bd_2)
print(event_dims)    # produces (ed_1, ed_2)

# This would allow for a construction that produces a variable number of batch_dims
# depending on user input.
bd_sizes = [2,3,4]
bd_names = ['bd_1', 'bd_2', 'bd_3']
batch_dims = dims(sizes = bd_sizes, names = bd_names)

# Where in the above, it is to be understood that bd_sizes and bd_names are reflecting
# the input of some user that specifies the dimensionality of a problem (e.g. if
# some offset should be applied to some 1d, 2d, 3d, tensor).