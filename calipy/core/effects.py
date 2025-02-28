#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module provides the CalipyEffect base class that is used for specifying
random and deterministic phenomena affecting measurements and provides a list
of basic effects as well as the functionality needed to integrate them into
the CalipyProbModel class for simulation and inference.

The classes are
    CalipyEffect: Base class from which all concrete effects inherit. Blueprint
        for effects that involve known parameters, unknown parameters, and random
        variables. Provides effect in form of differentiable forward map.
    
    CalipyQuantity: Base class from which all concrete quantities inherit. Blueprint
        for quantities, i.e. known parameters, unknown parameters, and random
        variables that are the building blocks for CalipyEffect objects.
  

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    CalipyEffect class ----------------------------------------------------
"""


# i) Imports

import pyro
import torch
import math
from calipy.core.primitives import param
from calipy.core.base import CalipyNode, NodeStructure
from calipy.core.tensor import CalipyTensor
from calipy.core.utils import multi_unsqueeze, context_plate_stack, dim_assignment, InputSchema
from calipy.core.base import NodeStructure
import calipy.core.dist as dist
from pyro.distributions import constraints
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type


# ii) Definitions


class CalipyEffect(CalipyNode):
    """
    The CalipyEffect class provides a comprehensive representation of a specific 
    effect. It is named, explained, and referenced in the effect description. The
    effect is incorporated as a differentiable function based on torch. This function
    can depend on known parameters, unknown parameters, and random variables. Known 
    parameters have to be provided during invocation of the effect. During training,
    unknown parameters and the posterior density of the random variables is inferred.
    This requires providing a unique name, a prior distribution, and a variational
    distribution for the random variables.
    """
    
    
    def __init__(self, type = None, name = None, info = None):
        
        # Basic infos
        super().__init__(node_type = type, node_name = name, info_dict = info)
        
        
        self._effect_model = None
        self._effect_guide = None
        

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # If the subclass hasn't overridden `name` at the class level, set it
        if 'name' not in cls.__dict__:
            cls.name = cls.__name__
            
            

"""
    CalipyQuantity class ----------------------------------------------------
"""



class CalipyQuantity(CalipyNode):
    """
    The CalipyQuantity class provides a comprehensive representation of a specific 
    quantity used in the construction of a CalipyEffect object. This could be a
    known parameter, an unknown parameter, or a random variable. This quantity
    is named, explained, and referenced in the quantity description. Quantities
    are incorporated into the differentiable function that define the CalipyEffect
    forward pass. Each quantity is subservient to an effect and gets a unique id
    that reflects this, quantities are local and cannot be shared between effects.
    """
    
    def __init__(self, type = None, name = None, info = None):
        
        # Basic infos
        super().__init__(node_type = type, node_name = name, info_dict = info)
        

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # If the subclass hasn't overridden `name` at the class level, set it
        if 'name' not in cls.__dict__:
            cls.name = cls.__name__
    



"""
    Classes of quantities
"""  


# i) Deterministic known parameter

class KnownParam(CalipyQuantity):
    pass


# ii) Deterministic unknown parameter

# Define a class of quantities that consists in unknown deterministic parameters
# that serve as model parameters to be inferred. This means that invocation of 
# the class produces objects that are interpretable as scalars, vectors, or matrices
# of constant values. These objects dont necessarily need to contain the same 
# constant in each of the dimensions. 
# The node_structure passed upon instantiation determines batch_shape (indicating 
# independent parameter values) and event_shape (indicating identical parameter
# values). E.g batch_shape = (5,), event_shape = (20,) produces a (5,20) tensor in which
# 5 independent constant values are repeated 20 times. This tensor can be used to
# e.g. construct a tensor mu to be passed into a distribution that represents 5
# different mean values that have been sampled 20 times each. 


class UnknownParameter(CalipyQuantity):
    """ UnknownParameter is a subclass of CalipyQuantity that produces an object whose
    forward() method produces a parameter that is subject to inference.

    :param node_structure: Instance of NodeStructure that determines the internal
        structure (shapes, plate_stacks, plates, aux_data) completely.
    :type node_structure: NodeStructure
    :param constraint: Pyro constraint that constrains the parameter of a distribution
        to lie in a pre-defined subspace of R^n like e.g. simplex, positive, ...
    :type constraint: pyro.distributions.constraints.Constraint
    :return: Instance of the UnknownParameter class built on the basis of node_structure
    :rtype: UnknownParameter (subclass of CalipyQuantity subclass of CalipyNode)
    
    Example usage: Run line by line to investigate Class
        
    .. code-block:: python
    
        # Investigate 2D bias tensor -------------------------------------------
        #
        # i) Imports and definitions
        import calipy
        from calipy.core.base import NodeStructure
        from calipy.core.effects import UnknownParameter
        node_structure = NodeStructure(UnknownParameter)
        bias_object = UnknownParameter(node_structure, name = 'tutorial')
        #
        # ii) Produce bias value
        bias = bias_object.forward()
        #
        # iii) Investigate object
        bias_object.dtype_chain
        bias_object.id
        bias_object.node_structure
        bias_object.node_structure.dims
        render_1 = bias_object.render()
        render_1
        render_2 = bias_object.render_comp_graph()
        render_2
    """
    
    
    # Initialize the class-level NodeStructure
    batch_dims = dim_assignment(dim_names = ['batch_dim'], dim_sizes = [10])
    param_dims = dim_assignment(dim_names = ['param_dim'], dim_sizes = [2])
    batch_dims_description = 'The dims in which the parameter is copied and repeated'
    param_dims_description = 'The dims of the parameter, in which it can vary'
    
    default_nodestructure = NodeStructure()
    default_nodestructure.set_dims(param_dims = param_dims,
                                    batch_dims = batch_dims)
    default_nodestructure.set_dim_descriptions(param_dims = param_dims_description,
                                                batch_dims = batch_dims_description)
    default_nodestructure.set_name("UnknownParameter")
    
    # Define the input schema for the forward method
    input_vars_schema = InputSchema(required_keys=[])
    observation_schema = InputSchema(required_keys=["sample"],
                                     key_types={"sample": CalipyTensor})

    
    # Class initialization consists in passing args and building dims
    def __init__(self, node_structure, constraint = constraints.real, **kwargs):  
        super().__init__(**kwargs)
        self.node_structure = node_structure
        self.batch_dims = self.node_structure.dims['batch_dims']
        self.param_dims = self.node_structure.dims['param_dims']
        self.dims = self.batch_dims + self.param_dims
        
        self.constraint = constraint
        self.init_tensor = torch.ones(self.param_dims.sizes)
        
    # Forward pass is initializing and passing parameter
    def forward(self, input_vars = None, observations = None, subsample_index = None, **kwargs):
        """
        Create a parameter of dimension param_dims.shape that is copied n times
        where n = batch_dims.size to yield an extendet tensor with the shape 
        [batch_dims.sizes, param_dims.sizes]. It can be passed to subsequent 
        effects and will be tracked to adjust it when training the model.
        
        :param input vars: None
        :type input_vars: None
        param observations: DataTuple with key ['sample'] containing CalipyTensor
            object that is considered to be observed and used for inference.
        type observations: DataTuple
        param subsample_index:
        type subsample_index:
        :return: CalipyTensor containing parameter tensor and dimension info.
        :rtype: CalipyTensor
        """
        self.param = param(name = '{}__param_{}'.format(self.id_short, self.name), 
                           init_tensor = self.init_tensor,
                           dims = self.param_dims,
                           constraint = self.constraint,
                           subsample_index = subsample_index)
        self.extended_param = self.param.expand_to_dims(self.dims)
        return self.extended_param



# ii) - a) Subclass of UnknownParameter for variances (featuring positivity constraint)

class UnknownVariance(UnknownParameter):
    docstring = "UnknownVariance is a subclass of UnknownParameter that includes a positivity constraint."
    __doc__ = docstring + UnknownParameter.__doc__ # Inherit docstrings from superclass
    def __init__(self, node_structure, **kwargs):  
        super().__init__(node_structure, constraint = constraints.positive, **kwargs)



# iii) Random variable

# class RandomVar(CalipyQuantity):
    
#     # Initialize the class-level NodeStructure
#     example_node_structure = NodeStructure()
#     example_node_structure.set_shape('batch_shape', (10, ), 'Batch shape description')
#     example_node_structure.set_shape('event_shape', (5, ), 'Event shape description')

    
#     def __init__(self, node_structure, distribution, distribution_args, **kwargs):  
#         super().__init__(**kwargs)
#         self.node_structure = node_structure
#         self.distribution = distribution
        
#         self.batch_shape = self.node_structure.shapes['batch_shape']
#         self.event_shape = self.node_structure.shapes['event_shape']
        
#     def forward(self, input_vars = None, observations = None):
#         self.distribution_args = distribution_args
        
#         return output


# iv) Gaussian process



# v) Neural net




"""
    Classes of simple effects
"""



# ii) Addition of noise

# Define a class of effects that consist in a noise distribution that can be 
# sampled to serve as observations used for inference. This means that invocation of 
# the class produces objects that are interpretable as noisy observations of scalars,
# vectors, or matrices. Neither the mean nor the standard deviation for the noise
# distribution need to be constant.
# The node_structure passed upon instantiation determines the shape of the noise
# being added to the mean. E.g batch_plate_1 with size 5, batch_plate_2 with 
# size 20 produces (5,20) tensors of independent noise when using the function
# call forward(input_vars = (mean, standard_deviation)).


class NoiseAddition(CalipyEffect):
    """ NoiseAddition is a subclass of CalipyEffect that produces an object whose
    forward() method emulates uncorrelated noise being added to an input. 

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
        import calipy
        from calipy.core.effects import NoiseAddition
        node_structure = NoiseAddition.example_node_structure
        noisy_meas_object = NoiseAddition(node_structure, name = 'tutorial')
        #
        # ii) Sample noise
        mean = torch.zeros([10,5])
        std = torch.ones([10,5])
        noisy_meas = noisy_meas_object.forward(input_vars = (mean, std))
        #
        # iii) Investigate object
        noisy_meas_object.dtype_chain
        noisy_meas_object.id
        noisy_meas_object.noise_dist
        noisy_meas_object.node_structure.description
        noisy_meas_object.plate_stack
        render_1 = noisy_meas_object.render((mean, std))
        render_1
        render_2 = noisy_meas_object.render_comp_graph((mean, std))
        render_2
    """
    
    
    # Initialize the class-level NodeStructure
    batch_dims = dim_assignment(dim_names = ['batch_dim'], dim_sizes = [10])
    event_dims = dim_assignment(dim_names = ['event_dim'], dim_sizes = [2])
    batch_dims_description = 'The dims in which the noise is independent'
    event_dims_description = 'The dims in which the noise is copied and repeated'
    
    default_nodestructure = NodeStructure()
    default_nodestructure.set_dims(batch_dims = batch_dims,
                                   event_dims = event_dims)
    default_nodestructure.set_dim_descriptions(batch_dims = batch_dims_description,
                                               event_dims = event_dims_description)
    default_nodestructure.set_name("NoiseAddition")
    
    
    # Define the input schema for the forward method
    input_vars_schema = InputSchema(required_keys=["mean", "standard_deviation"],
                                    optional_keys=["validate_args"],
                                    defaults={"validate_args": None},
                                    key_types={"mean": CalipyTensor, 
                                               "standard_deviation": CalipyTensor, 
                                               "validate_args": Optional[bool]})

    observation_schema = InputSchema(required_keys=["sample"],
                                     key_types={"sample": CalipyTensor})

    # Class initialization consists in passing args and building shapes
    def __init__(self, node_structure, **kwargs):
        super().__init__(**kwargs)
        self.node_structure = node_structure
        self.batch_dims = self.node_structure.dims['batch_dims']
        self.event_dims = self.node_structure.dims['event_dims']
        self.dims = self.batch_dims + self.event_dims
        
    # Forward pass is passing input_vars and sampling from noise_dist
    def forward(self, input_vars, observations = None, subsample_index = None, **kwargs):
        """
        Create noisy samples using input_vars = (mean, standard_deviation) with
        shapes as indicated in the node_structures' 'batch_dims' and 'event_dims'.
        
        :param input vars: DataTuple with keys ['mean', 'standard_deviation']
            containing CalipyTensor objects defining the underlying mean onto
            which noise with distribution N(0, standard_deviation) is added.
        :type input_vars: DataTuple
        param observations: DataTuple with key ['sample'] containing CalipyTensor
            object that is considered to be observed and used for inference.
        type observations: DataTuple
        param subsample_index:
        type subsample_index:
        :return: CalipyTensor representing simulation of a noisy measurement of
            the mean.
        :rtype: CalipyTensor
        """
        
        # Set up NodeStructure object normal_ns for the Normal distribution
        CalipyNormal = dist.Normal
        normal_ns = NodeStructure(CalipyNormal)
        normal_ns.set_dims(batch_dims = self.batch_dims, event_dims = self.event_dims)
        
        # Instantiate the distribution and initiate forward() pass
        calipy_normal = CalipyNormal(normal_ns)     
        output = calipy_normal.forward(input_vars, observations, subsample_index)
        
        return output
    
    
# # iii) Polynomial trend

# # Define a class of effects that consist in a (potentially multidimensional) 
# # polynomial trend that can be evaluated on some input_vars to produce trend values.
# # Invocation of the class produces objects that contain the monomial trend 
# # functions encoded in a design matrix and corresponding coefficients as parameters.
# # The node_structure passed upon instantiation determines the shape of the trend.\
# # E.g a batch_shape of (3,) and event_shape of (100,) produces 3 different trends
# # each with a length of 100 interpretable as 3 trends generating 3 timeseries.
# # Producing the actual trend values requires input_vars to be passed to the forward
# # call via e.g. forward(input_vars = (t,)) (where (t,) here is a tuple containing
# # copies of a 1D tensor time but might also be (x,y) or any other tuple of identically 
# # shaped tensors).

# class PolynomialTrend(CalipyEffect):
#     """ PolynomialTrend is a subclass of CalipyEffect that produces an object whose
#     forward() method computes polynomial trends based on input_vars.

#     :param node_structure: Instance of NodeStructure that determines the internal
#         structure (shapes, plate_stacks, plates, aux_data) completely.
#     :type node_structure: NodeStructure
#     :param degrees: Instance of Tuple that contains the degree of the polynomial
#         trend in different dimensions.
#     :type degrees: Tuple of Int
#     :return: Instance of the PolynomialTrend class built on the basis of node_structure
#     :rtype: PolynomialTrend (subclass of CalipyEffect subclass of CalipyNode)
    
#     Example usage: Run line by line to investigate Class
        
#     .. code-block:: python
    
#         # Investigate 1D trend ------------------------------------------------
#         #
#         # i) Imports and definitions
#         import calipy
#         from calipy.core.effects import PolynomialTrend
#         node_structure = PolynomialTrend.example_node_structure
#         trend_object = PolynomialTrend(node_structure, name = 'tutorial')
#         #
#         # ii) Compute trend
#         time = torch.linspace(0,1,100)
#         trend = trend_object.forward(input_vars = (time,))
#         #
#         # iii) Investigate object
#         trend_object.dtype_chain
#         trend_object.id
#         trend_object.noise_dist
#         trend_object.node_structure.description
#         trend_object.plate_stack
#         render_1 = trend_object.render((time,))
#         render_1
#         render_2 = trend_object.render_comp_graph((time,))
#         render_2
#     """
    
#     # Initialize the class-level NodeStructure
#     example_node_structure = NodeStructure()
#     example_node_structure.set_shape('batch_shape', (3, ), 'Batch shape description')
#     example_node_structure.set_shape('event_shape', (100,), 'Event shape description')

#     # Class initialization consists in passing args and building shapes
#     def __init__(self, node_structure, degrees = (2,), **kwargs):  
#         super().__init__(**kwargs)
#         self.node_structure = node_structure
#         self.batch_shape = self.node_structure.shapes['batch_shape']
#         self.event_shape = self.node_structure.shapes['event_shape']
        
#         self.n_vars =len(self.batch_shape)
#         self.n_coeffs = tuple([degree + 1 for degree in degrees])
#         # self.n_coeffs_total = math.comb(self.n_vars + self.)
#         self.init_tensor = torch.ones(self.batch_shape + (self.n_coeffs,))
        
#     # Forward pass produces trend values
#     def forward(self, input_vars, observations = None):
#         """
#         Create samples of the polynomial trend function using as input vars the
#         tensors var_1, var_2, ... that encode the value of some explanatory variable
#         for each point of interest; input_vars = (var_1, var_2, ..). The shape
#         of the resultant samples is as indicated in the node_structures' batch_shape,
#         event_shape.
        
#         :param input vars: Tuple (var_1, var_2, ...) of identically shaped tensors with 
#             equal (or at least broadcastable) shapes. 
#         :type input_vars: Tuple of instances of torch.Tensor
#         :return: Tensor representing polynomial trend evaluated at the values of input_var.
#         :rtype: torch.Tensor
#         """
        
#         self.coeffs = pyro.param('{}__coeffs_{}'.format(self.id_short, self.name), init_tensor = self.init_tensor)
#         self.A_mat = torch.cat([input_vars.unsqueeze(-1)**k for k in range(self.n_coeffs)], dim = -1)

#         output = torch.einsum('bjk, bk -> bj' , self.A_mat, self.coeffs )
#         return output
    
    

 
# # iii) Cyclical trend

# # Define a class of effects that consist in a (potentially multidimensional) 
# # cyclical trend that can be evaluated on some input_vars to produce trend values.
# # Invocation of the class produces objects that contain the monomial trend 
# # functions encoded in a design matrix and corresponding coefficients as parameters.
# # The node_structure passed upon instantiation determines the shape of the trend.
# # E.g a batch_shape of (3,) and event_shape of (100,) produces 3 different trends
# # each with a length of 100 interpretable as 3 trends generating 3 timeseries.
# # Producing the actual trend values requires input_vars to be passed to the forward
# # call via e.g. forward(input_vars = (t,)) (where (t,) here is a tuple containing
# # copies of a 1D tensor time but might also be (x,y) or any other tuple of identically 
# # shaped tensors).

# class CyclicalTrend(CalipyEffect):
#     """ CyclicalTrend is a subclass of CalipyEffect that produces an object whose
#     forward() method computes cyclical trends based on input_vars.

#     :param node_structure: Instance of NodeStructure that determines the internal
#         structure (shapes, plate_stacks, plates, aux_data) completely.
#     :type node_structure: NodeStructure
#     :param freq_shape: Instance of Tuple that contains the number of the frequencies
#         for different dimensions n_dim.
#     :type degrees: Tuple of Int
#     :return: Instance of the CyclicalTrend class built on the basis of node_structure
#     :rtype: CyclicalTrend (subclass of CalipyEffect subclass of CalipyNode)
    
#     Example usage: Run line by line to investigate Class
        
#     .. code-block:: python
    
#         # Investigate 1D trend ------------------------------------------------
#         #
#         # i) Imports and definitions
#         import calipy
#         from calipy.core.effects import PolynomialTrend
#         node_structure = PolynomialTrend.example_node_structure
#         trend_object = PolynomialTrend(node_structure, name = 'tutorial')
#         #
#         # ii) Compute trend
#         time = torch.linspace(0,1,100)
#         trend = trend_object.forward(input_vars = (time,))
#         #
#         # iii) Investigate object
#         trend_object.dtype_chain
#         trend_object.id
#         trend_object.noise_dist
#         trend_object.node_structure.description
#         trend_object.plate_stack
#         render_1 = trend_object.render((time,))
#         render_1
#         render_2 = trend_object.render_comp_graph((time,))
#         render_2
#     """
    
#     # Initialize the class-level NodeStructure
#     example_node_structure = NodeStructure()
#     example_node_structure.set_shape('batch_shape', (3, ), 'Batch shape description')
#     example_node_structure.set_shape('event_shape', (100,), 'Event shape description')

#     # Class initialization consists in passing args and building shapes
#     def __init__(self, node_structure, degrees = (2,), **kwargs):  
#         super().__init__(**kwargs)
#         self.node_structure = node_structure
#         self.batch_shape = self.node_structure.shapes['batch_shape']
#         self.event_shape = self.node_structure.shapes['event_shape']
        
#         self.n_vars =len(self.batch_shape)
#         self.n_coeffs = tuple([degree + 1 for degree in degrees])
#         # self.n_coeffs_total = math.comb(self.n_vars + self.)
#         self.init_tensor = torch.ones(self.batch_shape + (self.n_coeffs,))
        
#     # Forward pass produces trend values
#     def forward(self, input_vars, observations = None):
#         """
#         Create samples of the polynomial trend function using as input vars the
#         tensors var_1, var_2, ... that encode the value of some explanatory variable
#         for each point of interest; input_vars = (var_1, var_2, ..). The shape
#         of the resultant samples is as indicated in the node_structures' batch_shape,
#         event_shape.
        
#         :param input vars: Tuple (var_1, var_2, ...) of identically shaped tensors with 
#             equal (or at least broadcastable) shapes. 
#         :type input_vars: Tuple of instances of torch.Tensor
#         :return: Tensor representing polynomial trend evaluated at the values of input_var.
#         :rtype: torch.Tensor
#         """
        
#         self.coeffs = pyro.param('{}__coeffs_{}'.format(self.id_short, self.name), init_tensor = self.init_tensor)
#         self.A_mat = torch.cat([input_vars.unsqueeze(-1)**k for k in range(self.n_coeffs)], dim = -1)

#         output = torch.einsum('bjk, bk -> bj' , self.A_mat, self.coeffs )
#         return output
    



# i) OffsetDeterministic class 
# Define a class of errors that transform the input by adding deterministic 
# offsets. This means that invocation of the class produces objects that are 
# interpretable as scalars, vectors, or matrices of constant values. These offsets
# dont necessarily need to be constant in each dimension. E.g. 5 with different
# series of 20 measurements for each of which the offset remains constant, the
# corresponding OffsetDeterministic object would be a 5 x 20 tensor containing 5
# different constants repeated 20 times per row.



# List of Effects
#
# primarily deterministic:
# DeterministicOffset
# DeterministicScale
# LinearTransform
# PolynomialTrend
# FunctionalTrend
# Convolution
# SpectralFiltering
# DynamicSystem
# ANN
# AxisDeviation
# Misalignment
# Hysteresis
#
# primarily physical:
#
# lens abberation
# reflection
# 
# 
#
# primarily stochastic:
# NoiseAddition
# GrossError
# 
# RandomConvolution
#

# List of Quantities
#
# KnownParameter
# UnknownParameter
# UnknownVariance
# UnknownCovarianceMatrix
# UnknownProbability
# UnknownProbabilityVector
# RandomVariable
# CorrelatedNoise
# GaussianProcess
# WienerProcess
# CovarianceFunction
# TrainableBounds
# TrainableInequality
# TrainableEquality
# ProbabilityDistribution




# Some more ideas:
#
# primarily deterministic:
# AffineTransformation
# NonlinearTransformation
# WaveletTransform
# KalmanFilter
# Interpolation
# SplineFitting
# GeometricDistortion
# AdaptiveSmoothing
# GainControl
# TimeSeriesDecomposition
# FourierTransform
# LaplacianTransformation
# Decimation
# Resampling
#
# primarily physical:
# TemperatureDrift
# MagneticInterference
# VibrationResponse
# HumidityInfluence
# CorrosionAndWear
# RadiationEffects
# AtmosphericRefraction
# ThermalExpansion
# WindLoading
# HydrodynamicEffects
#
# primarily stochastic:
# ColoredNoise
# FlickerNoise
# MultiplicativeNoise
# StateDependentNoise
# ShotNoise
# CompoundPoissonNoise
# AdditiveOutliers
# RandomWalk
# JumpDiffusion
# HawkesProcess

# List of Quantities
#
# UnknownDrift
# UncertainInitialCondition
# AnisotropicNoise
# PeriodicFunction
# TimeVaryingParameter
# SpatiallyVaryingParameter
# RandomField
# StochasticDifferentialEquation
# MarkovChain
# LogNormalDistribution






