#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase how CaliPy's RandomVariable class is used.
Since the class has multiple different input/output signatures and degrees of 
automation built in, we survey the classes functionalities in various scenarios
and illustrate the use cases for the different RandomVariable setupts
For this, do the following:
    1. Imports and definitions
    2. Autobuilt RandomVariable
    3. 
    4. 
    5. Manually built RandomVariable
    6. Analyse results and illustrate

The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


import torch
import calipy
import pyro
from calipy.utils import dim_assignment, site_name
from calipy.base import NodeStructure
from calipy.tensor import CalipyTensor
from calipy.data import DataTuple
from calipy.primitives import param
from calipy.effects import Distribution, RandomVariable, NoiseAddition, UnknownParameter

#




# Check model-guide pairs under different degrees of automatization
# More automatization -> more convenience -> less flexibility
# Auto level 0 =  Allows for arbitrary constructions by using different nodes
#                   in model and guide
# Auto level 1 =  Allows providing callables for the guide
# Auto level 2 =  Allows specifying distributions and providing input params
# Auto level 3 =  Autobuilds model-guide as Normal - Normal pairs in which the
#                   guide is N(mu_post, sigma_post) with two new params
#
# Example model, guide pairs
# (2,2)




"""
    1. Investigate class
"""

# Initialize the class-level NodeStructure
batch_dims = dim_assignment(dim_names = ['batch_dim'], dim_sizes = [10])
event_dims = dim_assignment(dim_names = ['event_dim'], dim_sizes = [2])
param_dims = dim_assignment(dim_names = ['param_dim'], dim_sizes = [2])
batch_dims_description = 'The dims in which the realizations are independent'
event_dims_description = 'The dims in which the realizations are dependent'
param_dims_description = 'The dims in which the params vary independently'

default_nodestructure = NodeStructure()
default_nodestructure.set_dims(batch_dims = batch_dims,
                               event_dims = event_dims,
                               param_dims = param_dims)
default_nodestructure.set_dim_descriptions(batch_dims = batch_dims_description,
                                           event_dims = event_dims_description,
                                           param_dims = param_dims_description)
default_nodestructure.set_name("SimpleNS")

# Parameters
param_ns = NodeStructure(UnknownParameter)
param_ns.set_dims(batch_dims = batch_dims,
                  param_dims = param_dims)
default_nodestructure.set_name("ParamNS")


"""
    1. Most general setup: Distribution class
"""

# prior_dist_setup
ns_prior_dist = default_nodestructure
name_prior_dist = 'NormalDistribution'
prior_dist_obj = Distribution(ns_prior_dist, name_prior_dist, calipy.dist.Normal)

# prior forward setup
prior_dist_sizes = ns_prior_dist.dims['batch_dims'].sizes + ns_prior_dist.dims['event_dims'].sizes
input_vars_prior = {'loc' :  torch.zeros(prior_dist_sizes), 'scale': torch.ones(prior_dist_sizes) }


def model_dist(obs = None):
    sample = prior_dist_obj.forward(input_vars = input_vars_prior)
    
    
# guide_dist_setup
ns_guide_dist = default_nodestructure
name_guide_dist = 'NormalDistribution'
guide_dist_obj = Distribution(ns_guide_dist, name_guide_dist, calipy.dist.Normal)

# guide forward setup
guide_dist_sizes = ns_guide_dist.dims['batch_dims'].sizes + ns_guide_dist.dims['event_dims'].sizes
loc_guide = UnknownParameter(ns_guide_dist, name = 'loc_posterior')
scale_guide = UnknownParameter(ns_guide_dist, name = 'scale_posterior')
    
def guide_dist(obs = None):
    input_vars_guide = {'loc' :  loc_guide.forward(), 'scale': scale_guide.forward() }
    sample = guide_dist_obj.forward(input_vars = input_vars_guide)


graphical_model_dist = pyro.render_model(model = model_dist, model_args= (None,),
                                    render_distributions=True,
                                    render_params=True)

graphical_guide_dist = pyro.render_model(model = guide_dist, model_args= (None,),
                                    render_distributions=True,
                                    render_params=True)

model_dist_trace = pyro.poutine.trace(model_dist).get_trace(None)
print(model_dist_trace.format_shapes())
model_dist_trace.nodes

guide_dist_trace = pyro.poutine.trace(guide_dist).get_trace(None)
print(guide_dist_trace.format_shapes())
guide_dist_trace.nodes



"""
    1. Autobuilt RandomVariables
"""

# i) 


ns_auto = default_nodestructure
name_auto = 'AutoBuiltRV'
rv_obj_auto = RandomVariable(ns_auto, name_auto)


def model_auto(obs = None):
    sample = rv_obj_auto.forward()
    
def guide_auto(obs = None):
    sample = rv_obj_auto.forward(in_guide = True)
    
    
# This is equivalent to 
rv_obj_auto = RandomVariable(ns_auto, name_auto, 
                              prior_dist_cls = calipy.dist.Normal, prior_dist_params = 'auto',
                              posterior_dist_cls = calipy.dist.Normal, posterior_dist_params = 'auto')

def model(obs = None):
    sample = rv_obj_auto.forward(input_vars = rv_obj_auto.prior_dist_vars, obs = obs)

def guide(obs = None):
    sample = rv_obj_auto.forward(input_vars = rv_obj_auto.posterior_dist_vars, obs = obs, in_guide = True)
    
    




"""
    5. Manually built RandomVariable
"""

ns_manual = default_nodestructure
name_manual = 'ManBuiltRV'
rv_obj_manual = RandomVariable(ns_manual, name_manual, 
                              prior_dist_cls = calipy.dist.Normal, prior_dist_params = 'auto',
                              posterior_dist_cls = calipy.dist.Normal, posterior_dist_params = 'auto')

ns_manual_params = param_ns



def model_manual(obs = None):
    # dim_sizes = ns_manual.dims['batch_dims'].sizes + ns_manual.dims['event_dims'].sizes
    mu_sample = UnknownParameter(ns_manual_params, 'mu_prior_' + site_name(rv_obj_manual, 'rv'),
                                 init_tensor = torch.ones(ns_manual_params.dims['param_dims'].sizes))
    sigma_sample = UnknownParameter(ns_manual_params, 'sigma_prior_' + site_name(rv_obj_manual, 'rv'),
                                 init_tensor = torch.ones(ns_manual_params.dims['param_dims'].sizes))
    sample = rv_obj_manual.forward(input_vars = {'mean' : mu_sample, 'standard_deviation' : sigma_sample},
                                   obs = obs)
    return sample
    
    
def guide_manual(obs = None):
    mu_sample_post = UnknownParameter(ns_manual_params, 'mu_posterior_' + site_name(rv_obj_manual, 'rv'),
                                 init_tensor = torch.ones(ns_manual_params.dims['param_dims'].sizes))
    sigma_sample_post = UnknownParameter(ns_manual_params, 'sigma_posterior_' + site_name(rv_obj_manual, 'rv'),
                                 init_tensor = torch.ones(ns_manual_params.dims['param_dims'].sizes))
    sample_post = rv_obj_manual.forward(input_vars = {'mean' : mu_sample_post, 'standard_deviation' : sigma_sample_post},
                                   obs = obs, in_guide = True)
    return sample_post

    
# This is equivalent to 
rv_obj_auto = RandomVariable(ns_auto, name_auto, 
                              prior_dist_cls = calipy.dist.Normal, prior_dist_params = 'auto',
                              posterior_dist_cls = calipy.dist.Normal, posterior_dist_params = 'auto')

def model(obs = None):
    sample = rv_obj_auto.forward(input_vars = rv_obj_auto.prior_dist_vars, obs = obs)

def guide(obs = None):
    sample = rv_obj_auto.forward(input_vars = rv_obj_auto.posterior_dist_vars, obs = obs, in_guide = True)
    
    





# ii) Invoke and investigate CalipyDistribution
CalipyNormal = calipy.dist.Normal
CalipyNormal.dists
CalipyNormal.input_vars
CalipyNormal.input_vars_schema

# iii) Build a concrete Node
normal_ns = NodeStructure(CalipyNormal)
print(normal_ns)
calipy_normal = CalipyNormal(node_structure = normal_ns, name = 'tutorial', add_uid = True)

calipy_normal.id
calipy_normal.node_structure
CalipyNormal.default_nodestructure

# Calling the forward method
normal_dims = normal_ns.dims['batch_dims'] + normal_ns.dims['event_dims']
normal_ns_sizes = normal_dims.sizes
mean = CalipyTensor(torch.zeros(normal_ns_sizes), normal_dims)
standard_deviation = CalipyTensor(torch.ones(normal_ns_sizes), normal_dims)
input_vars_normal = DataTuple(['loc', 'scale'], [mean, standard_deviation])
samples_normal = calipy_normal.forward(input_vars_normal)
samples_normal
samples_normal.value.dims


def model(obs = None):
    calipy_normal.forward(input_vars_normal)
    
model_trace = pyro.poutine.trace(model).get_trace()
print('These are the shapes of the involved objects : \n{} \nFormat: batch_shape,'\
      ' event_shape'.format(model_trace.format_shapes()))
model_trace.nodes