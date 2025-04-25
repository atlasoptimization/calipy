==============================
Core Concepts & Architecture
==============================

.. contents::
   :local:
   :depth: 1


Why a torch-based calibration library?
=======================================

Goal and concept
----------------

CaliPy’s goal is to make **building and solving** complicated
**probabilistic instrument models** easy and intuitive. By 
focusing entirely on Bayesian Inference and outsourcing the 
math to **PyTorch** and **Pyro** for automatic gradient computation
and stochastic variational inference, *declaring* a model is
basically the same as *solving* it.

Due to its simple effect-based architecture, model declaration
is done by invoking **effects**, endowing them with structure and
then chaining them together. Think of it as building modular
models by sticking together lego blocks. CaliPy automatically 
turns the resulting directed‑acyclic graph (DAG) into a runnable
model, that can be used forward (for simulation) and backward (for inference)

This avoids the classical pitfalls of limiting yourself to oversimplified
models that can be solved by hand‑coded least‑squares algorithms.

High‑level architecture
-----------------------

.. mermaid::

   flowchart TD
       subgraph user_code
           A[Define NodeStructure] -->|instantiates| B[CalipyNode subclasses]
           B --> C[CalipyProbModel(model, guide)]
           C -->|.train()| D[Inference Engine / Pyro SVI / MCMC]
       end
       D -->|posteriors| E[(Results)]

       %% cosmetic styling
       classDef node fill:#90caf9,stroke:#333,stroke-width:1px;
       class A,B,C,D,E node;



Key abstractions
================


CalipyTensor
------------

A thin wrapper around a ``torch.Tensor`` that **remembers its
DimTuple** and carries an attached **TensorIndexer**, supporting
subsampling and traceable indices.

Represent a torch tensor **plus**:

* *dims* (``DimTuple``) – names, sizes & semantics  
* automatic *indexer* (local / global)  
* rich helpers: ``expand_to_dims``, ``flatten``, broadcasting aware arithmetic

Example::

   batch = dim_assignment(['bd'], [100])
   event = dim_assignment(['ed'], [2])
   x = CalipyTensor(torch.randn([100,2]), batch+event, 'x')

CalipyDim & DimTuple
~~~~~~~~~~~~~~~~~~~~

*One* dimension object per logical axis (``bd_1``, ``ed_1`` …)  
keeps names, sizes, descriptions.  A **DimTuple** behaves like a
tuple but is *dimension‑aware*: filtering, broadcasting checks,
`torchdim` binding, etc.


CalipyIndex
~~~~~~~~~~~

An index tensor + dimension metadata, created automatically (or by
``TensorIndexer*``) and consumable by any CalipyTensor for indexing::

   x_sub = x[10,:]
   x_sub_local = x_sub.indexer.local_index
   x_sub_global = x_sub.indexer.global_index


Nodes & NodeStructure
---------------------

``CalipyNode`` is the abstract base for everything executable:
effects, parameters, distributions, even whole probability
models.  
Each node is parameterised by a **NodeStructure** that declares
its *batch/event/parameter* dims (with plate semantics).

For building a NodeStructure:
* keys are *dim roles* – typically ``batch_dims``, ``event_dims``, …  
* values are ``DimTuple`` objects  
* every concrete *CalipyNode* receives one NodeStructure object on construction

Effects & Quantities
--------------------

* **Effects** (e.g. ``NoiseAddition``, ``PolynomialTrend``)
  compute new tensors—think “stochastic layers”.
* **Quantities** (e.g. ``UnknownParameter``, ``UnknownVariance``)
  hold deterministic or random parameters used by effects.

Both are subclasses of ``CalipyNode``; calling ``forward`` on
them *samples* or *computes* their output.

CalipyProbModel
---------------

A container that wires nodes together and exposes

* ``model`` – generative process  
* ``guide`` – variational family (often empty if no latent RVs)  
* ``train`` – one‑line SVI loop with sensible defaults

Directed Acyclic Graph (DAG)
----------------------------

Internally CaliPy constructs a DAG of nodes/edges.  
Edges carry **CalipyIO** objects, ensuring every tensor has a
name and dimension context.

.. hint::

   You rarely touch the DAG directly—forward methods plus
   ``CalipyIO`` objects are enough for 99 % of use‑cases.


Overview cheat‑sheet
====================

Roles and classes
-----------------

+-----------------------+------------------------------------------+
| **Role**              | **Object type**                          |
+=======================+==========================================+
| RAW data (torch)      | ``torch.Tensor``                         |
+-----------------------+------------------------------------------+
| Dimension‑aware       | ``CalipyTensor``                         |
+-----------------------+------------------------------------------+
| Named collection      | ``CalipyIO``                             |
+-----------------------+------------------------------------------+
| Processing step       | ``CalipyNode`` subclasses                |
+-----------------------+------------------------------------------+
| Whole instrument model| ``CalipyProbModel``                      |
+-----------------------+------------------------------------------+

Actions and commands
--------------------

+--------------------------+----------------------------------------------+
| **Action**               | **One‑liner**                                |
+==========================+==============================================+
| Build batch/event dims   | ``dim_assignment(['bd','ed'], [N, None])``   |
+--------------------------+----------------------------------------------+
| Turn torch‑tensor into   | ``CalipyTensor(t, dims, 'name')``            |
| dimension‑aware tensor   |                                              |
+--------------------------+----------------------------------------------+
| Create learnable scalar  | ``theta = UnknownParameter(ns).forward()``   |
+--------------------------+----------------------------------------------+
| Add Gaussian noise       | ``y = NoiseAddition(ns).forward({'mean':mu,``|
|                          | ``'standard_deviation':sigma}, obs)``        |
+--------------------------+----------------------------------------------+
| Train whole model        | ``loss_trace = probmodel.train(x, y)``       |
+--------------------------+----------------------------------------------+


Expected output
===============

After ``probmodel.train(...)`` you get

* list / ndarray of ELBO values (for plotting)  
* parameters stored in ``pyro.get_param_store()``  
* convenience access through each node instance, e.g.::

      print(theta.tensor)      # MAP/ML estimate


Further reading
===============

* :doc:`quickstart` – hands‑on 20‑line bias‑and‑noise example
* :doc:`usage` – practical guides (models, effects, data, inference)
* API reference generated via autodoc (see side‑bar)


