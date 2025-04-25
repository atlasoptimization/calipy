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
import textwrap
from calipy.tensor import preprocess_args, CalipyTensor

# Define calipy wrappers for torch functions
# This should be converted to commands calipy.sum, calipy.mean etc
# These functions are useful to process CalipyTensors in such a way that the dim
# argument can be specified as a CalipyDim object; if this is not needed, just apply
# the torch versions torch.sum, torch.mean etc.



# Alternative idea here: Build function calipy_op(func, *args, **kwargs) 
# then inspect and functools wraps
def calipy_sum(calipy_tensor, dim = None, keepdim = False, dtype = None):
    """
    Wrapper function for torch.sum applying a dimension-aware sum to CalipyTensor
    objects. Input args are as for torch.sum but accept dim = dims for dims that are
    either a DimTuple or a CalipyDim.

    Notes:
    - This function acts on CalipyTensor objects
    - This function acts on dim args of class CalipyDim and DimTuple
    - The behavior is equivalent to torch.sum on the CalipyTensor.tensor level,
      but augments the result with dimensions.

    Original torch.sum docstring:

    .. code-block:: none

    {}
    """.format(textwrap.indent(torch.sum.__doc__, "    "))
    
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
    
# calipy_sum.__doc__ += "\n" + torch.sum.__doc__


def calipy_cat(calipy_tensors, dim = 0):
    """ 
    Wrapper function for torch.cat applying a dimension-aware sum to CalipyTensor
    objects. Input args are as for torch.cat but accept dim = dims for dims either
    a DimTuple or an integer.
    
    Notes:
    - This function acts on CalipyTensor objects
    - This function acts on dim args of class DimTuple and int.
    - The behavior is equivalent to torch.cat on the CalipyTensor.tensor level
      but augments the result with dimensions.
        
    Example usage:

    .. code-block:: python
        
        # Imports and definitions
        import torch
        from calipy.tensor import CalipyTensor
        from calipy.utils import dim_assignment
        from calipy.funs import calipy_cat
        
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


    Original torch.cat docstring:

    .. code-block:: none

    {}
    """.format(textwrap.indent(torch.cat.__doc__, "    "))
    
    # Compile and unwrap arguments
    args = calipy_tensors
    kwargs = {'dim' : dim}
    
    # Convert CalipyDims in 'dim' argument to int indices if present
    if not isinstance(dim, int):
        kwargs['dim'] = (calipy_tensors[0].dims.find_indices(kwargs['dim'].names))[0]
    tensor_list, unwrapped_kwargs = preprocess_args(args, kwargs)
    
    
    # Call torch function
    result = torch.cat(tensor_list, **unwrapped_kwargs)
    result_dims = calipy_tensors[0].dims
    result_name = calipy_tensors[0].name
    result_cp = CalipyTensor(result, result_dims, name = result_name)
    
    return result_cp
    
# calipy_cat.__doc__ += "\n" + torch.cat.__doc__






















