#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides tasks related to the estimation of mean and variance of
a simple univariate gaussian distribution based on some observed data. An exemplary
solution is also provided; however the idea of the script is to use the commands
and building blocks found in the pyro_tutorial series to assemble a solution. 
The goal is for you to learn to set up a stochastic model, define an inference 
scheme, and fit the model to some data.

For this, you will do the following:
    1. Imports and definitions
    2. Generate data
    3. Setup stochastic model       <-- your task
    4. Fit model to data
    5. Plots and illustrations
    
For the purpose of this task, think of a measurement device measuring the length
of some object several times; each measurement results in one number and they 
are all performed in the same manner. From the observations, infer the expected
value and the spread of individual measurements.
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import torch
import pyro

import inspect
import matplotlib.pyplot as plt


# ii) Definitions

torch.manual_seed(0)
n_data = 100



"""
    2. Generate data
"""


# The data consists of a bunch of numbers that are all drawn i.i.d. from a normal
# distribution with mean mu_true and standard_deviation sigma_true. Both of these
# parameters are used only for the generation of the dataset and are not assumed
# known starting from section 3. We generate the random numbers by sampling from 
# a pytorch distribution torch.distributions.Normal().


# i) Set up ground truth parameters

mu_true = 1
sigma_true = 2


# ii) Define distribution and sample

data_distribution = torch.distributions.Normal(loc = mu_true, scale = sigma_true)
dataset = data_distribution.sample([n_data])


# Consider everything that has happened up till this line to be result of some
# black box process found in nature. Only the data is available to you.
# ----------------------------------------------------------------------------



"""
    3. Set up stochastic model      <--- your task
"""


# You now want to build your stochastic model in pyro. This requires setting up
# a function model() in which parameters, sampling, and independence statements
# are combined to reflect the model you want to fit.
# 
# The model you want to fit is a simple univariate normal distribution with un-
# known mean and variance. From this distribution, n_data samples are taken and
# for these samples we want to plug in the numbers in "dataset".
#
# The model() function should take as input potentially a bunch of numbers and 
# condition on them, alternatively just simulate some new numbers by running
# all the commands and returning the samples drawn from a probability distribution.
# This means, your model should look something like this:
#
# def model(observations = None):
#     # Declare parameters
#     mu = pyro.param( arguments_for_pyro_param)
#     sigma = pyro.param( arguments_for_pyro_param )
#
#     # Declare distribution and sample n_data independent samples, condition on observations
#     model_dist = pyro.distributions.your_distribution( arguments_for_distribution )
#     with pyro.plate( arguments_for_pyro_plate ):    
#         model_sample = pyro.sample( arguments_for_pyro_sample, obs = observations)
    
#     return model_sample
#
# In the model, you make use of the following commands:
# pyro.param() : Define a parameter that will be optimized during training
# pyro.distributions.Normal() : Define a normal distribution
# pyro.plate() : Declare that independend sampling happens
# pyro.sample() : Declare some numbers as coming from a distribution
#
# Write your own stuff starting from here : ----------------------------------


# YOUR OWN CODE


# Then you need to instantiate one neural network with appropriate dimensions
# by calling the class with some input arguments.


# YOUR OWN CODE


# You can now stop writing your own stuff for section 3. ---------------------



"""
    4. Fit model to data
"""


# i) Defining the guide

# Variational distribution
def guide(observations = None):
    pass


# ii) Set up Optimization options

n_iter = 100
adam = pyro.optim.Adam({"lr": 0.1})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)


# iii) Inference and print results

mu_est_sequence = []
sigma_est_sequence = []
for step in range(n_iter):
    loss = svi.step(dataset)
    mu_est_sequence.append(pyro.get_param_store()['mu'].item())
    sigma_est_sequence.append(pyro.get_param_store()['sigma'].item())

print('True mu = {}, True sigma = {} \n Inferred mu = {:.3f}, Inferred sigma = {:.3f}'
      .format(mu_true, sigma_true, 
              pyro.get_param_store()['mu'].item(),
              pyro.get_param_store()['sigma'].item()))



"""
    5. Plots and illustrations
"""


# i) Plot training progress

iter_steps = torch.linspace(0, n_iter-1, n_iter)
fig, axs = plt.subplots(1,2, figsize = (10,5), dpi = 300)
axs[0].plot(iter_steps, mu_est_sequence)
axs[0].plot(iter_steps, mu_true*torch.ones([n_iter]))
axs[0].set_title('Estimated mu')
axs[0].set_xlabel('Iterations')
    
axs[1].plot(iter_steps, sigma_est_sequence)
axs[1].plot(iter_steps, sigma_true*torch.ones([n_iter]))
axs[1].set_title('Estimated sigma')
axs[1].set_xlabel('Iterations')



"""
    Additional tasks and questions
"""


# Interpret the output of the model inspection with pyro.poutine.trace.
# You can use: print(pyro.poutine.trace(model).get_trace(None).format_shapes())
#         and: print(pyro.poutine.trace(model).get_trace(None).nodes)

# Which other probability distributions does pyro support? Try to get a list!
# You can use: for name, attr in inspect.getmembers(pyro.distributions): print(name)

# The task of estimating mean and standard deviation is fairly standard. How 
# does the solution found by pyro compare to the typical empirical estimators 
# for mean and standard deviation?

# What is the effect of the learning rate and which potential problems do you
# see with setting it too high?

# The guide is also called the variational distribution. It is a model that is
# fitted to the posterior density of unobserved variables. Since we dont have 
# any unobserved variables in this example, the guide is empty. What could you
# change in the model or the problem setup to end up with a nontrivial guide?

# The pyro.plate context declares that sampling along a specific direction is
# independent, i.e. when sampling i.i.d. samples are drawn from a distribution
# and when performing inference, the data is considered independent in this
# direction. What would happen (in general and in our example), if we dont 
# specify independence?




""" 
    6. Exemplary solution 
"""


# ---------SPOILERS BELOW, TRY WITHOUT EXEMPLARY SOLUTION FIRST----------------



# For section 3: Define the model function


# # i) Defining the model 

# def model(observations = None):
#     # Declare parameters
#     mu = pyro.param(name = 'mu', init_tensor = torch.tensor([0.0]))
#     sigma = pyro.param(name = 'sigma', init_tensor = torch.tensor([1.0]))
    
#     # Declare distribution and sample n_data independent samples, condition on observations
#     model_dist = pyro.distributions.Normal(loc = mu, scale = sigma)
#     with pyro.plate('batch_plate', size = n_data):    
#         model_sample = pyro.sample('model_sample', model_dist, obs = observations)
    
#     return model_sample



