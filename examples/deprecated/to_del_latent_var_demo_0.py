#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to showcase how CaliPy's Distribution class is used.
The class is very flexible and it is the basis of e.g. the RandomVariable class
that comes with more predefinitions and convenient defaults for model-guide pairing.
What we illustrate here is a very simple latent variable posterior estimation
problem, in which some product is produced and measured with randomness impacting
both actions. We focus on interpreting and explaining the Distribution class.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Set up prior and guide dists
    4. Set up model and guide
    5. Perform inference
    6. Plots and illustrations

The script is meant solely for educational and illustrative purposes. Written by
Dr. Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

# base packages
import torch
import calipy
import pyro

# calipy
from calipy.utils import dim_assignment, site_name
from calipy.base import NodeStructure, CalipyProbModel
from calipy.tensor import CalipyTensor
from calipy.data import DataTuple
from calipy.effects import Distribution, RandomVariable, NoiseAddition, UnknownParameter


# ii) Definitions

n_prod = 5
n_meas = 200



"""
    2. Simulate some data
"""


# i) Set up sample distributions

mu_prod_true = torch.tensor(1.0)
sigma_prod_true = torch.tensor(0.1)
sigma_meas_true = torch.tensor(0.01)


# ii) Sample from distributions

prod_distribution = pyro.distributions.Normal(mu_prod_true, sigma_prod_true)
prod_lengths = prod_distribution.sample([n_prod])

data_distribution = pyro.distributions.Normal(prod_lengths, sigma_meas_true)
data = data_distribution.sample([n_meas]).T

# The data now is a tensor of shape [n_prod, n_meas] and reflects 5 products being
# produced with different length characteristics that are then subsequently measured
# 200 times

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.






"""
    3. Set up prior and guide dists
"""



# i) Set up dims

# Initialize the dims
prod_dims = dim_assignment(dim_names = ['prod_dim'], dim_sizes = [n_prod])
meas_dims = dim_assignment(dim_names = ['meas_dim'], dim_sizes = [n_meas])
unitary_dims = dim_assignment(dim_names = ['unitary_dim'], dim_sizes = [1])
empty_dims = dim_assignment(dim_names = ['empty_dim'], dim_sizes = [])
param_dims = dim_assignment(dim_names = ['param_dim'], dim_sizes = [])

# # Master ns containing all the dims
# master_ns = NodeStructure()
# master_ns.set_dims(batch_dims = prod_dims,
#                    event_dims = meas_dims,
#                    param_dims = param_dims)

# NodeStructure for production randomness
prod_ns = NodeStructure()
prod_ns.set_dims(batch_dims = prod_dims,
                   event_dims = unitary_dims)
prod_ns.set_name("Production_nodestructure")

# NodeStructure for measurement randomness
meas_ns = NodeStructure()
meas_ns.set_dims(batch_dims = prod_dims + meas_dims,
                 event_dims = empty_dims)
meas_ns.set_name("Measurement_nodestructure")


# ii) Set up for production prior_dist

# prior_dist_setup
ns_prior_dist = prod_ns
name_prior_dist = 'Production'
prod_prior_dist = Distribution(ns_prior_dist, name_prior_dist, calipy.dist.Normal)

# prior forward setup
prior_dist_sizes = ns_prior_dist.dims['batch_dims'].sizes + ns_prior_dist.dims['event_dims'].sizes
input_vars_prior = {'loc' :  mu_prod_true * torch.ones(prior_dist_sizes), 
                    'scale': sigma_prod_true* torch.ones(prior_dist_sizes) }


# iii) Set up for meas_dist

# meas_dist_setup
ns_meas_dist = meas_ns
name_meas_dist = 'Measurement'
meas_dist = Distribution(ns_meas_dist, name_meas_dist, calipy.dist.Normal)


# iv) Set up for production guide_dist
    
# guide_dist_setup
ns_guide_dist = NodeStructure()
ns_guide_dist.set_dims(batch_dims = prod_dims,
                       event_dims = unitary_dims,
                       param_dims = param_dims)
name_guide_dist = 'Production'
prod_guide_dist = Distribution(ns_guide_dist, name_guide_dist, calipy.dist.Normal)

# guide forward setup
guide_dist_sizes = ns_guide_dist.dims['batch_dims'].sizes + ns_guide_dist.dims['event_dims'].sizes
loc_guide = UnknownParameter(ns_guide_dist, name = 'loc_posterior')
scale_guide = UnknownParameter(ns_guide_dist, name = 'scale_posterior')
    



"""
    4. Build the probmodel
"""

# This model describes a production and measurement process. The product has a
# certain (lets say length) and the production of the product is impacted by
# uncertainty leading to different lengths for each product. These lengths are
# then measured which adds another source of randomness. From the measurements,
# the posterior distribution of the product length should be inferred.
# The model looks like this:
#   prod_len ~ N(1, 0.1)
#   meas_len ~ N(prod_len, 0.01)


# i) Define the probmodel class 

class DemoProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # integrate nodes
        self.prod_prior_dist = prod_prior_dist
        self.meas_dist = meas_dist
        
        self.prod_guide_dist = prod_guide_dist
        self.loc_guide = loc_guide
        self.scale_guide = scale_guide
        
    # Define model by forward passing
    def model(self, input_vars = None, observations = None):
        # Sample product lengths from the prior, then measure them using measure_dist
        prod_lengths = self.prod_prior_dist.forward(input_vars = input_vars_prior)
        input_vars_meas = {'loc' :  prod_lengths.value, 'scale': sigma_meas_true }
        meas_lengths = self.meas_dist.forward(input_vars = input_vars_meas)
        
        return meas_lengths
        
    # Define guide 
    def guide(self, input_vars = None, observations = None):
        # Posterior distribution to be approximated is production distribution
        # prod_guide_dist has unknown loc and scale params to be chosen to approx
        # the true posterior after updating prod_prior_dist with data.
        input_vars_guide = {'loc' :  self.loc_guide.forward(), 
                            'scale': self.scale_guide.forward() }
        prod_lengths = prod_guide_dist.forward(input_vars = input_vars_guide)
        
        return prod_lengths
    
        
demo_probmodel = DemoProbModel()
    



graphical_model_dist = pyro.render_model(model = demo_probmodel.model, model_args= (None, None),
                                    render_distributions=True,
                                    render_params=True)

graphical_guide_dist = pyro.render_model(model = demo_probmodel.guide, model_args= (None, None),
                                    render_distributions=True,
                                    render_params=True)

model_dist_trace = pyro.poutine.trace(demo_probmodel.model).get_trace(None)
print(model_dist_trace.format_shapes())
model_dist_trace.nodes

guide_dist_trace = pyro.poutine.trace(demo_probmodel.guide).get_trace(None)
print(guide_dist_trace.format_shapes())
guide_dist_trace.nodes



"""
    5. Perform inference
"""


# i) Set up optimization

adam = pyro.optim.NAdam({"lr": 0.03})
elbo = pyro.infer.Trace_ELBO()
n_steps = 1000

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps}


# ii) 

input_data = None
output_data = CalipyTensor(data, dims = batch_dims_meas)
optim_results = demo_probmodel.train(input_data, output_data, optim_opts = optim_opts)
    


"""
    6. Plots and illustrations
"""




