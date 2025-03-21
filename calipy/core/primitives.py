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
from calipy.core.data import CalipyIO, preprocess_args


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
    
    ssi_io = CalipyIO(subsample_index)
    
    # TODO Check event_dim argument
    param_tensor = pyro.param(name, init_tensor = init_tensor, constraint = constraint)
    param_cp = CalipyTensor(param_tensor, dims =  dims, name = name)
    
    # Subsample the CalipyTensor
    if ssi_io.is_null:
        ssi = param_cp.indexer.local_index
    else:
        ssi = ssi_io.value
        
    param_subsampled_cp = param_cp[ssi]
    
    return param_subsampled_cp

param.__doc__ = param.__doc__ + pyro.param.__doc__




import contextlib

# Introduce sample function

def sample(name, dist, dist_dims, observations=None, subsample_index=None, vectorizable=True):
    """
    Flexible sampling function handling multiple plates and four cases based on obs and subsample_indices.

    :param node_structure: Instance of NodeStructure that determines the internal
        structure (shapes, plate_stacks, plates, aux_data) completely.
    :type node_structure: NodeStructure
    :return: Instance of the NoiseAddition class built on the basis of node_structure
    :rtype: NoiseAddition (subclass of CalipyEffect subclass of CalipyNode)
    
    Example usage: Run line by line to investigate Class
        
    .. code-block:: python
    
        # Investigate 2D noise ------------------------------------------------
        #
        # i) Imports and definitions
        
        
    Parameters:
    -----------
    name : str
        Base name for the sample site.
    dist : pyro.distributions.Distribution
        The distribution to sample from.
    dist_dims : list of CalipyDim objects
        The dimensions of the sample from the distribution; need to contain batch_dims as subset.
    batch_dims : list of CalipyDim objects
        The dimensions that act as batch dimensions and over which independence is assumed.
    vectorizable : bool, optional
        If True, uses vectorized sampling. If False, uses sequential sampling. Default is True.
    obs : CalipyObservation or None, optional
        Observations wrapped in CalipyObservation. If provided, sampling is conditioned on these observations.
    subsample_index : list of torch.Tensor or None, optional
        Subsample indices for each plate dimension. If provided, sampling is performed over these indices.

    Returns:
    --------
    CalipyTensor
        The sampled data, preserving batch and event dimensions.
    """
    
    # Basic preprocess and rename
    input_vars_io, observations_io, subsample_index_io = preprocess_args(None,
                                            observations, subsample_index)
    obs = observations_io
    ssi = subsample_index_io
    vec = vectorizable
    
    # Set up dimensions
    batch_dims = dist_dims['batch_dims']
    event_dims = dist_dims['event_dims']
    batch_dim_sizes = batch_dims.sizes
    event_dim_sizes = event_dims.sizes
    dist_dims = batch_dims + event_dims
    dist_dims_sizes = dist_dims.sizes
    
    # Compute plate dim arguments as counted from right from leftmost event dim
    batch_dim_positions_from_right = dist_dims.find_indices(batch_dims.names, from_right = True) 
    batch_dim_positions = [dim_pos + len(event_dim_sizes) for dim_pos in batch_dim_positions_from_right]
    
    
    # Plate setup
    plate_names = [name + '_plate' for name in batch_dims.names]
    plate_sizes = [size if size is not None else 1 for size in batch_dims.sizes]
    
    # Plate lengths of the subsamples or None if no subsampling.
    ssi_bound_dims = ssi.bound_dims[batch_dims] if not ssi.is_null else []
    plate_ssi_lengths = [ssi_dim.size for ssi_dim in ssi_bound_dims]
    if len(plate_ssi_lengths) == 0:
        plate_ssi_lengths = [None] * len(batch_dims)
    
    # plate_ssi_lengths = []
    # for ssi_dim in ssi_bound_dims:
    #     plate_ssi_lengths.append(ssi_dim.size)
        
    # for k, (name, dim) in enumerate(zip(plate_names, ssi.bound_dims)):
    #     plate_ssi_lengths.append(dim.size)
        
    # bound_dims = [ssi.bound_dims if ssi is not None else None]
    # plate_ssi_lengths = [None for name in batch_dims.names]


    # cases [1,x,x] vectorizable
    if vectorizable == True:
        # Vectorized sampling using pyro.plate
        with contextlib.ExitStack() as stack:
            # Determine dimensions for plates

            
            # case [0,0] (obs, ssi)
            if obs.is_null and ssi.is_null :
                current_obs = None
                
            
            # case [0,1] (obs, ssi)
            if obs.is_null and not ssi.is_null:
                current_obs = None
        
            
            # case [1,0] (obs, ssi)
            if not obs.is_null and ssi.is_null:
                current_obs = observations.tensor
            
            # case [1,1] (obs, ssi)
            if not obs.is_null and not ssi.is_null:
                pass
            
            # Handle multiple plates
            for i, (plate_name, plate_size, dim, subsample_size) in enumerate(zip(plate_names, plate_sizes, batch_dim_positions, plate_ssi_lengths)):
                
                stack.enter_context(pyro.plate(plate_name, size=plate_size, subsample_size=subsample_size, dim=dim))

            # Sample data
            sample_vals = pyro.sample(name, dist, obs=current_obs)
            sample_vals_cp = CalipyTensor(sample_vals, dist_dims, name = name)
            

        
    # cases [0,x,x] nonvectorizable
    elif vectorizable == False:
            
            # case [0,0] (obs, ssi)
            if obs == None and ssi == None:
                # Create new observations of shape batch_shape_default with ssi
                # a flattened list of product(range(n_default))
                
                pass
            
            # case [0,1] (obs, ssi)
            if obs == None and ssi is not None:
                # Create len(ssi) new observations with given ssi's
                pass
            
            # case [1,0] (obs, ssi)
            if obs is not None and ssi == None:
                #  Create obs with standard ssi derived from obs batch_shape
                pass
            
            # case [1,1] (obs, ssi)
            if obs is not None and ssi is not None:
                # Create obs associated to given ssi's
                pass

    
            # Construct tensor from samples
            # Need to sort samples based on indices to ensure correct tensor shape
            # sample_value = pyro.sample(sample_name, dist, obs=obs_value)
            
    return sample_vals_cp