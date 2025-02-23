#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides primitives needed for performing basic probabilistic actions
like declaring parameters and smapling distributions

The classes and functions are
    param: Declares a tensor as a parameter subject to optimization with SVI.
        Produces a CalipyTensor with dims and subsampling capability
    
    sample:
  
The param function is the basic function called to declare unknown parameters;
it is often found as an ingredient when defining effects.
        

The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""

import pyro
from pyro.distributions import constraints
from calipy.core.tensor import CalipyTensor


def param(name, init_tensor, dims, constraint = constraints.real, subsample_index = None):
    """ Wrapper function for pyro.param producing a CalipyTensor valued parameter.
    The tensor is inialized once, then placed in the param store and can be used
    like a regular CalipyTensor.
    
    :param name: Unique name of the parameter
    :type name: string
    :param init_tensor: The initial value of the parameter tensor, adjusted later
        on by optimization
    :type init_tensor: torch.tensor
    :param dims: A tuple of dimensions indicating the dims of the CalipyTensor
        created by param()
    :type dims: DimTuple
    :param constraint: Pyro constraint that constrains the parameter of a distribution
        to lie in a pre-defined subspace of R^n like e.g. simplex, positive, ...
    :type constraint: pyro.distributions.constraints.Constraint
    :param subsample_index: The subsampling index indicating how subsampling
        of the parameter is to be performed
    :type subsample_index: CalipyIndex
    :return: A CalipyTensor parameter being tracked by gradient tape and marked for 
        optimization. Starts as init_tensor, has dims and constraints as specified
        and is automatically subsampled by subsampling_index.
    :rtype: CalipyTensor
    
    Example usage: Run line by line to investigate Class
        
    .. code-block:: python
    
        # Create parameter ---------------------------------------------------
        #
        # i) Imports and definitions
        import calipy
        from calipy.core.base import param
        
        batch_dims = dim_assignment(dim_names = ['bd_1_A'], dim_sizes = [4])
        event_dims = dim_assignment(dim_names = ['ed_1_A'], dim_sizes = [2])
        param_dims = batch_dims + event_dims
        init_tensor = torch.ones(param_dims.sizes) + torch.normal(0,0.01, param_dims.sizes)
        parameter = param('generic_param', init_tensor, param_dims)
        
        # Create constrained, subsampled parameter ---------------------------
        #
        param_constraint = pyro.distributions.constraints.positive
        subsample_indices = TensorIndexer.create_simple_subsample_indices(batch_dims[0],
                                                                batch_dims.sizes[0], 3)
        ssi = subsample_indices[1]
        ssi_expanded = ssi.expand_to_dims(param_dims, param_dims.sizes)
        parameter_subsampled = param('param_subsampled', init_tensor, param_dims,
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
        

    pyro.param doc: 
        
    """
    
    # TODO Check event_dim argument
    param_tensor = pyro.param(name, init_tensor = init_tensor, constraint = constraint)
    param_cp = CalipyTensor(param_tensor, dims =  dims, name = name)
    
    # Subsample the CalipyTensor
    if subsample_index == None:
        subsample_index = param_cp.indexer.local_index
        
    param_subsampled_cp = param_cp[subsample_index]
    

    return param_subsampled_cp

param.__doc__ = param.__doc__ + pyro.param.__doc__