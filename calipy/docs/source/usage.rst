Usage
=====


.. _label_installation:

Installation
------------

To use calipy, first install it using pip:

.. code-block:: console

   (.venv) $ pip install calipy


Basic classes
----------------

.. autoclass :: calipy.core.base.CalipyNode

.. autoclass :: calipy.core.base.NodeStructure

.. autoclass :: calipy.core.base.CalipyProbModel



Effect classes
----------------

.. autoclass :: calipy.core.effects.CalipyEffect

.. autoclass :: calipy.core.effects.CalipyQuantity

.. autoclass :: calipy.core.effects.UnknownParameter

.. autoclass :: calipy.core.effects.UnknownVariance

.. autoclass :: calipy.core.effects.NoiseAddition





Utility classes and functions
--------------------------------

.. autofunction :: calipy.core.utils.multi_unsqueeze

.. autofunction :: calipy.core.utils.robust_meshgrid

.. autofunction :: calipy.core.utils.ensure_tuple

.. autofunction :: calipy.core.utils.get_params

.. autofunction :: calipy.core.utils.format_mro

.. autofunction :: calipy.core.utils.check_schema

.. autofunction :: calipy.core.utils.dim_assignment

.. autofunction :: calipy.core.utils.generate_trivial_dims

.. autoclass :: calipy.core.utils.CalipyDim

.. autoclass :: calipy.core.utils.TorchdimTuple

.. autoclass :: calipy.core.utils.DimTuple


Tensor classes and functions
------------------------------

.. autoclass :: calipy.core.tensor.CalipyIndex

.. autoclass :: calipy.core.tensor.CalipyIndexer

.. autoclass :: calipy.core.tensor.IOIndexer

.. autoclass :: calipy.core.tensor.TensorIndexer

.. autoclass :: calipy.core.tensor.CalipyTensor

.. autofunction :: calipy.core.tensor.preprocess_args

.. autofunction :: calipy.core.tensor.build_dim_supersequence

.. autofunction :: calipy.core.tensor._is_subsequence

.. autofunction :: calipy.core.tensor.broadcast_dims


Primitive classes and functions
---------------------------------

.. autofunction :: calipy.core.primitives.param

.. autofunction :: calipy.core.primitives.sample


Function classes and functions
---------------------------------

.. autofunction :: calipy.core.funs.calipy_sum

.. autofunction :: calipy.core.funs.calipy_cat


Data classes and functions
-------------------------------

.. autoclass :: calipy.core.data.DataTuple

.. autoclass :: calipy.core.data.CalipyDict

.. autoclass :: calipy.core.data.CalipyList

.. autoclass :: calipy.core.data.CalipyIO

.. autoclass :: calipy.core.data.CalipyDataset

.. autofunction :: calipy.core.data.io_collate

.. autofunction :: calipy.core.data.preprocess_args


Distribution classes and functions
------------------------------------

.. autoclass :: calipy.core.dist.distributions.CalipyDistribution

.. autofunction :: calipy.core.dist.distributions.build_default_nodestructure


