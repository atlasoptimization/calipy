#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is part of a series of calipy tutorials. During these tutorials, we
will tackle a sequence of problems of increasing complexity that will take us
from understanding simple calipy concepts to fitting complex measurement models
that feature latent random variables and neural networks.
The tutorials consist in the following:
    - first contact with calipy             (tutorial_1)
    - forward model and diagnosis           (tutorial_2)
    - fitting simple model                  (tutorial_3)
    - fitting multivariate model            (tutorial_4)
    - fitting model with hidden variables   (tutorial_5)
    - fitting model with neural networks    (tutorial_6)

This script will initiate first contact with the calibration and inference framework
calipy. We will showcase the differences between numpy, pytorch, and calipy, get an
overview of different modules of calipy and sample some noise.
For this, do the following:
    1. Imports and definitions
    2. calipy estimation example
    3. Comparison, numpy, pytorch, calipy
    4. A look at calipy's objects
    5. Overview calipy functionality
    
The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.
"""



"""
    1. Imports and definitions
"""


# i) Imports

import numpy as np
import torch
import pyro
import calipy

import inspect

import matplotlib.pyplot as plt


# ii) Definitions

np.random.seed(0)
torch.manual_seed(0)



"""
    2. calipy model fitting example
"""


# We start the tutorial directly by teasering the typical form of a program
# written in calipy and what it can do. The example will be to estimate the
# mean parameter of a normal distribution with known variance based on some data.
# This example will feature many commands that are to be explained only later.
# The aim is therefore to familiarize the reader with the look and feel and power
# of calipy syntax and to motivate a deeper delve into the tutorial.


# i) Generate data

mu_true = 0
sigma_true = 1

n_data = 100
dataset = torch.distributions.Normal(loc = mu_true, scale = sigma_true).sample([n_data])


# ii) Set up shapes for mean parameter mu and noise addition

# mu setup
mu_ns = calipy.base.NodeStructure()
mu_ns.set_shape('batch_shape', (), 'Independent values')
mu_ns.set_shape('event_shape', (n_data,), 'Repeated values')
mu_object = calipy.effects.UnknownParameter(mu_ns, name = 'mu')

# noise setup
noise_ns = calipy.base.NodeStructure()
noise_ns.set_plate_stack('noise_stack', [('batch_plate', n_data, -1, 'independent noise 1')], 'Stack containing noise')
noise_object = calipy.effects.NoiseAddition(noise_ns)


# iii) Define the probmodel class 

class DemoProbModel(calipy.base.CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # integrate nodes
        self.mu_object = mu_object
        self.noise_object = noise_object 
        
    # Define model by forward passing
    def model(self, input_vars = None, observations = None):
        mu = self.mu_object.forward()       
        output = self.noise_object.forward((mu, sigma_true), observations = observations)     
        
        return output
    
    # Define guide (trivial since no posteriors)
    def guide(self, input_vars = None, observations = None):
        pass


# iv) Initialize and train model  
  
demo_probmodel = DemoProbModel()
optim_results = demo_probmodel.train(input_data = None, output_data = dataset, optim_opts = {})

mu_estimated = calipy.utils.get_params(name = 'mu')
print('True mu = {}'.format(mu_true))  
print('Inferred mu = ', mu_estimated)

# In this example, we have generated a new dataset (no calipy involved) and then
# proceeded to declare a model. This involved declaring a parameter mu as an
# UnknownParameter object with a specific batch_shape and event shape. In addition,
# the effect of adding noise was created by instantiating an object of NoiseAddition
# type with some declaration of the shape of the noise and its independence structure.
#
# The actual model construction happens by constructing a new class based on the
# CalipyProbModel class that offers functionality for simulation and inference.
# This new class DemoProbModel integrates the unknown parameter mu and the noise
# and defines a forward model consisting of calling a parameter, then calling the
# noise. Note here, that both mu and noise get called using the forward() function.
# In calipy, the object.forward() function of objects like CalipyQuantities,
# CalipyEffects, and CalipyInstruments takes all of the inert info of the objects
# and compiles them into a numerical computation that outputs a tensor (or a tuple
# of tensors). This computation involves pytorch and pyro statements and can be
# traced by the computational graph engine.
#
# Maybe not a lot of this code is immediately clear but it looks understandable
# and the syntax is clean and meaningful. A lot of the actual meaning and content 
# is visible only after looking in detail at the different components of calipy, 
# which is what will happen in the next tutorials.



"""
    3. Comparison, numpy, pytorch, pyro, calipy
"""


# Lets start off by comparing numpy, pytorch, pyro, and calipy. While calipy is
# the main focus of this tutorial, it is good to understand its relations to the other
# three, more fundamental libraries. numpy is a library for basic numeric computation
# providing the array classes and associated methods. numpy arrays are collection
# of numbers without much context. pytorch, in comparison, is a library for deep
# learning and is built around the tensor class. pytorch tensors are a collection
# of numbers together with a history of where they came from thereby enabling gradient
# computation. Finally, pyro is a library for enabling deep learning in a probabilistic
# context by augmenting pytorch with probabilistic concepts like probability 
# distributions and sampling. 
# In comparison to that, calipy is much more modest. It aims to take pyro's powerful
# framework for universal deep probabilistic programming and make it more easily
# accessible for tasks involving calibration of measurement instruments. Instead
# of providing fundamentally new functionality, it provides a lego-like framework
# to implement and chain instrument effects (like noise, drifts, and gross errors).
# This allows rapidly building instrument models and training them based on real
# data; these instrument models can be uploaded and used by anybody.
# 
# In short:
#   numpy = numbers
#   pytorch = numbers + history
#   pyro = pytorch + probability
#   calipy = pyro for instrument models


# i) numpy arrays and methods

# Apart from the standard analytical functions, there exists a submodule called
# numpy.random that allows to sample from probability distributions. However, 
# the normal distribution itself is not a function in numpy. You can write it
# and use it to compute the probability density values but overall this makes
# it clear that the ecosystem in numpy is not geared towards complicated 
# statistical problems and their analysis.

# Use numpy to compute and sample from standard normal.
sample_z = np.random.normal(loc = 0, scale = 1, size = [1000,1])
def standard_normal_pdf(index_variable):
    return (1/np.sqrt(2*np.pi))*np.exp(-index_variable**2/2)
index = np.linspace(-3,3,100)
distribution_z = standard_normal_pdf(index)

# Plot the distribution and the random data
fig, axs = plt.subplots(2, 1, figsize=(8, 10))
axs[0].plot(np.linspace(-3,3,100), distribution_z)
axs[0].set_title('Probability density')
axs[1].hist(sample_z, bins=20)
axs[1].set_title('Histogram of sample_z')
plt.tight_layout()
plt.show()


# ii) pytorch tensors and methods

# Compared to numpy, pytorch's support for probability distributions is more 
# extensive. In pytorch, probability distributions exist as differentiable 
# functions and it is also possible to sample from them. However, there is no
# native support for tying samples to probability distributions and then 
# differentiate the distribution w.r.t. the sample. Although this can be done
# manually, making this type of task more convenient is what pyro was made for.

# Use torch to compute and sample from the standard normal.
t_standard_normal = torch.distributions.Normal(loc = 0, scale = 1)
t_index = torch.linspace(-3,3,100)
t_distribution_z = torch.exp(t_standard_normal.log_prob(t_index))
t_sample_z = t_standard_normal.sample(sample_shape = [10,1])


# We could plot the the samples and the distribution here but there is nothing
# new to see here compared to the numpy samples and distribution. However, since
# pytorch allows for the computation of gradients, we can compute the
# gradients of the probability w.r.t. the mean mu. numpy does not allow this.

# Compute log probability of sample vector given mean parameter t_mu
t_mu = torch.tensor(0.0, requires_grad =True)
t_mu_normal = torch.distributions.Normal(loc = t_mu, scale = 1)
t_mu_log_prob = t_mu_normal.log_prob(t_sample_z)
t_mu_log_prob_sum = t_mu_log_prob.sum()

# Backward pass to compute gradients
t_mu_log_prob_sum.backward()
t_mu_grad = t_mu.grad
print('The logs of the probabilities for the independent samples t_sample_z are', t_mu_log_prob)
print('The sum of the log probs is {} and the gradient is {}'.format(t_mu_log_prob_sum, t_mu_grad))
print('Note that sum of log probs = constant - sum_i ( (1/2)*(x_i - mu)**2) where x_i is sample i. \n'
      'This implies the gradient to be sum_i (x_i - mu). We see that analytic derivative and torch-'
      'based derivative coincide (sum of the observations is {}; mu has been initialized to 0)'.format(torch.sum(t_sample_z)))

# Using the above relationships we could now call an optimizer to adapt the mean
# to maximize the probability of t_sample_z. This corresponds to maximum
# likelihood estimation. The computation is not very convenient in pytorch, 
# pyro was designed to handle this in an easier fashion.


# iii) Distributions in pyro

# Since pyro = pytorch + probability, pyro modelbuilding is centered around 
# tensors. What we did above in pytorch with declaring some data as samples
# of a probabilistic model and then performing gradient descent to estimate
# some parameters - under the hood pyro does basically the same thing. It 
# provides convenient representations of sampling, observation, inference.
# The above problem could be solved in pyro using
#   1. pyro.distributions(...) for declaring a specific distribution
#   2. pyro.sample(... obs = True) for declaring a sample observed and 
#       preparing for inference 
#   3. pyro.plate(...) context manager for declaring independence of samples
#   4. pyro.infer.SVI(...) for setting up the optimization problem
# Finally, a training loop would be executed by calling the SVI's step() 
# functionality.

# Lets instantiate a probabilistic model in pyro and perform inference on an
# unknown mean parameter. For this, we define a function 'model', that combines
# declarations of probability distributions and sample statements within some
# independence contexts.

def model(observations = None):
    # Define parameter, then pass it to distribution
    p_mu = pyro.param('mu', torch.tensor(0.0))
    obs_dist = pyro.distributions.Normal(p_mu, torch.tensor(1.0))
    
    # Sample from distribution within independence context
    with pyro.plate('batch_plate', size = 10, dim = -1):
        obs = pyro.sample('obs', obs_dist, obs = observations)
    return obs

# We also define a guide function which acts as an approximation to the posterior
# density (empty here since no latent variables).
def guide(observations = None):
    pass

# Call training loop by first defining some optimization options, then performing
# gradient based parameter update steps.
adam = pyro.optim.Adam({"lr": 0.1})
elbo = pyro.infer.Trace_ELBO()
svi = pyro.infer.SVI(model, guide, adam, elbo)

n_steps = 100
for step in range(n_steps):
    elbo_loss = svi.step(t_sample_z.squeeze())
    if step % 10 == 0:
        print('epoch: {} ; loss : {}'.format(step, elbo_loss))
print("Best guess for the mean parameter using pyro's svi = {}. Coincides with analytical solution ={}"
      .format(pyro.get_param_store()['mu'], torch.mean(t_sample_z)))


# iv) Special calipy constructs

# Since calipy  = pyro for instrument models, calipy modelbuilding is centered 
# around stochastic effects that can be chained together to produce interpretible
# instrument models that optimally explain observed data. What we did above in 
# pyro by defining a model and a guide function and then calling an optimization 
# loop, we can also do in calipy. In calipy we would import a class representing
# unknown parameters and a class representing noise and then chain them together
# using their respective .forward() methods. An example of this was shown in 
# paragraph 2, where we showcased a model-fitting example.
# Under the hood calipy does basically the same thing as pyro but the syntax is
# easier and there is a prebuilt library of instruments, effects, and quantities.
# It provides convenient methods for declaring unknown parameters, random variables,
# adding noise, nonlinear transformations, sampling, observation, inference.
#
# The code in calipy for this simple problem is more complicated than in pyro
# due to the use of special constructs like the NodeStructure object that determines
# batch_shape, event_shape, and independence structures for all important objects
# in calipy. In the end, all the objects that should contribute to the stochastic 
# model to be optimized are called via their .forward() function and this chain
# of effects is integrated into a CalipyProbModel object; that is an object that
# comes with extra functionality for performing statistical inference. The reader
# may at this point ask why to use calipy at all but they may rest assured that
# all these extra provisions of clearly defining
#   - input and output shapes
#   - conditional independence structures
#   - data flow via .forward()
#   - prebuilt quantities, effects, instruments
# become essential and convenient in case the instrument models get more complicated.
#
# When multiple instruments, stochastic effects, unknown parametrs, latent random
# variables, nonlinear transformations, and neural networks are coupled with the
# complicated control flow of a nontrivial python program, writing a handtailored 
# pyro program is feasible but complicated. Furthermore, such a pyro script is less
# modular and lends itself less towards being shared; it is our explicit goal
# to have effects and instruments added in the form of classes to a shared modular
# library that anyone can use with minimum overhead. In short, calipy has additional
# declarative overhead but the coding effort  scales very well to complicated 
# instrument models.



"""
    4. A look at calipy's objects
"""


# i) The UnknwonParameter class
#
# The UnknwnParameter class is a simple class for representing parameters that
# are to be estimated during inference. The parameters can be tensors of arbitrary
# shapes representing scalars, vectors, or matrices for example. In paragraph two,
# we constructed mu_object, an instance of the UnknownParameter class that represents
# a single scalar mean parameter (but copied n_data times).

# The bject itself has a name, inherits from some base classes and gets a unique
# id upon instantiation.
mu_object
mu_object.name
mu_object.dtype_chain
mu_object.id

# The shapes are encoded in the node_structure of the object. We have a batch_shape
# of () encoding the existence of only one independent scalar and an event_shape of
# (100,) encoding the extension of this scalar to a 100 dim vector.

mu_object.batch_shape
mu_object.event_shape

# The forward pass produces a vector of the shape described above. This pass
# instantiates all the active ingredients in the stochastic model and allows for
# tracking via gradient tape. We can also show a graphical representation of the
# forward pass in terms of a graphical model.

mu_object.forward()
mu_object.render()
mu_object.render_comp_graph()


# ii)  The NoiseAddition class
#
# The NoiseAddition class is also a very simple class. However, it represents not
# a quantity but rather the effect of adding noise to a mean. This requires to
# additionally specify the idependence structure of the noise by passing the
# plate_stack argument during construction.

noise_object
noise_object.name
noise_object.dtype_chain
noise_object.id

# The plate stack is arder to interpret, it is not simply a tuple of numbers but
# adeclaration of specific dimensions of the noise distribution as independent.

noise_object.noise_dist
noise_object.plate_stack

# Compared to the forward() pass of the mu_object, the forward() pass of the 
# noise_object takes as input the mean and the noise variance. Both are allowed
# to be UnknownParameters and subject of later inference.

input_vars = (mu_object.forward(), torch.tensor(1.0))
noise_object.forward(input_vars)
noise_object.render(input_vars)
noise_object.render_comp_graph(input_vars)


# iii) The CalipyNode class
#
# Both of the previously seen classes are examples of the foundational CalipyNode
# class. This class is itself an ABC (abstract base class) with abstract method 
# forward() that needs to be specified for each subclass at instantiation time.
# It provides the basic functionality that we have used in the previous two classes
# without ever explicitly defining it, e.g. the render() or render_comp_graph()
# method and universal properties like the id and the instance counters.
#
# Most users will not interact with this class directly and therefore we will not
# pursue investigation of this class further.


# iv) The NodeStructure class

mu_ns = mu_object.node_structure
noise_ns = noise_object.node_structure

# We can print the details of these node_structures
mu_ns.print_shapes_and_plates()
mu_ns.description

noise_ns.print_shapes_and_plates()
noise_ns.description

# We can check if these node_structures are actually valid for the classes
calipy.effects.UnknownParameter.check_node_structure(mu_ns)
calipy.effects.NoiseAddition.check_node_structure(noise_ns)

# ... and in case they aren't, we can either look at some valid example node_structure
# or print out the template for generating one

example_ns_param = calipy.effects.UnknownParameter.example_node_structure
example_ns_param
example_ns_param.generate_template()




"""
    5. Overview calipy functionality
"""


# i) Summary

# We have looked into numpy arrays, pytorch tensors, and pyro distributions and
# figured out that calipy benefits from being built on top of pyro due to being
# able to perform stochastic variational inference to fir complicated stochastic
# models. Everything in calipy is built around prebuilt classes that represent
# different quantities, effects, instruments that might feature in instrument
# models we might want to build to explain some data. Everything revolves around
# using tensors an pyro's / pytorch's autograd framework to optimize probability
# distributions. calipy's syntax is clean and interpretable and the modular nature
# of the stochastic effects being containered in classes lends itself well to the
# construction of complex models.

# calipy offers everything necessary to perform simulation and inference for a
# broad range of instrument models. This includes functionality for handling
# distributions, sampling, diagnostics, and inference. The functionalities are
# bundled into different submodules as outlines below.
#
# calipy.base: Submodule is foundational for providing the base classes used
#   in calipy. It provides (among others) the CalipyNode and NodeStructure classes.
#   Most other classes are subclassing the base classes; they allow stochastic
#   effects to be written in such a way that they are tracked by gradient tape.
#
# calipy.effects: Submodule contains the CalipyEffects and CalipyQuantity
#   classes. These are essential for defining models. CalipyQuantity objects
#   are representative of atomic objects like parameters and probability distributions
#   while CalipyEffect objects use pytorch functions and pyro sample statements
#   to mimick some stochastic effect.
#
# calipy.instruments: Submodule contains the CalipyInstrument class and a
#   collection of basic concrete instrument classes like the generic  TotalStation
#   class or the LevellingInstrument class.
#
# calipy.utils: Submodule contains utility functions that allow formatted
#   printing of infos, manipulation of NodeStructure's, and a set of conversion
#   functions translating between calipy code and pyro's optimization formulations.
#
# calipy.library: Submodule that contains a set of concrete classes of quantities,
#   effects, instruments that go beyond the basic functionality and which have
#   been contributed by the community for public use.
#
# calipy.examples: Submodule containing examples and tutorials. These give some 
#   insight not only into how the prebuilt classes can be used but also into the
#   design of new classes representing new quantites, effects, or instruments.
#
# There are also other submodules like calipy.tests and calipy.docs but interacting
# with them will not be important during this tutorial.
    

# ii) Outlook

# Apart from the motivating initial example, we have only looked at fundamental
# concepts like CalipyNode's and NodeStructure's in calipy. We still have not fully 
# investigated the impact of the calipy syntax on model building and inference
# is a black box to us for now. In the next two tutorials we will build more
# complex models and learn how to diagnose models to ensure their meaning aligns
# with our intent. Afterwards, we will delve deeper into inference and inject
# complexity into our models. In the end we will have introduced posterior 
# densities over hidden variables and have trained deep instrument models whose
# parameters are outputs of a neural network.

# The next tutorial, however will stay modest. We will build instrument models
# very similar to the ones in this tutorial but we will understand them better.
# This will mean looking at the internal representation that calipy builds of a
# model - the node_structure representing shapes and independence contexts as 
# well as the forward() pass of any CalipyNode object. This will represent most
# of what calipy knows about the relationships between random variables and parameters
# and can be a bit hard to parse. Nonetheless it is important to understand. 
# Otherwise it is hard to ensure that the models do what we want them to do and 
# that the training data is used responsibly without unintended and undocumented
# assumptions of e.g. independence or normality.
    
