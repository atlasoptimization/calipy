#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The goal of this script is to employ calipy to model a tape measure that is affected
by an additive bias. We will create the instrument model and the probmodel, then
train it on data to showcase the grammar and typical use cases of calipy.
For this, do the following:
    1. Imports and definitions
    2. Simulate some data
    3. Build the instrument model
    4. Build the probmodel
    5. Perform inference
    6. Analyse results and illustrate
In this example we will have n_tape different tape measures and n_meas different
measurements per tape measure. The bias mu of the measurements is supposed to
be an unknown constant that is different for each tape measure.
These tape measures are used to mease an object with a known length of 1 m, the
bias of the tape measures is to be inferred.

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""


"""
    1. Imports and definitions
"""


# i) Imports

# base packages
import torch
import pyro
import matplotlib.pyplot as plt

# calipy
# import calipy
from calipy_setup_instruments import CalipyInstrument
from calipy_setup_effects import CalipyEffect, CalipyQuantity, OffsetDeterministic
from calipy_setup_probmodel import CalipyProbModel


# ii) Definitions

n_tape = 5
n_meas = 20



"""
    2. Simulate some data
"""


# i) Set up sample distributions

length_true = torch.ones([n_tape])

coin_flip = pyro.distributions.Bernoulli(probs = torch.tensor(0.5)).sample([n_tape])
bias_true = coin_flip * torch.tensor(0.01)      # measurement bias is either 0 cm or 1 cm
sigma_true = torch.tensor(0.01)                 # measurement standard deviation is 1 cm


# ii) Sample from distributions

data_distribution = pyro.distributions.Normal(length_true + bias_true, sigma_true)
data = data_distribution.sample([n_meas]).T

# The data now has shape [n_tape, n_meas] and reflects n_tape tape measures each
# with a bias of either 0 cm or 1 cm being used to measure an object of length 1 m.

# We now consider the data to be an outcome of measurement of some real world
# object; consider the true underlying data generation process to be unknown
# from now on.



"""
    3. Build the instrument model
"""




"""
    4. Build the probmodel
"""


# i) Probabilistic model for tape measurements





"""
    5. Perform inference
"""


"""
    6. Analyse results and illustrate
"""






























