#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides basic functionality to produce and adapt functions that
interact well with calipy's CalipyTensor and CalipyDistribution classes and
maintain dimension-awareness.

The classes and functions are
    calipy_sum: a dimension-aware sum, that acts directly on CalipyTensors and
        keeps track of dimensions, produces a CalipyTensor.
    
    sample:
  
The param function is the basic function called to declare unknown parameters;
it is often found as an ingredient when defining effects.
        

The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



import torch

# Define calipy wrappers for torch functions
# This should be converted to commands calipy.sum, calipy.mean etc
# These functions are useful to process CalipyTensors in such a way that the dim
# argument can be specified as a CalipyDim object; if this is not needed, just apply
# the torch versions torch.sum, torch.mean etc.



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