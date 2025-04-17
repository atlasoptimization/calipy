=====================
Quick‑Start Tutorial
=====================

.. admonition:: TL;DR
   :class: tip

   *You have CaliPy installed and want to fit a first simple model to showcase grammar and signs of life.*  
   Copy‑paste the short script below; it builds a tiny probabilistic
   model, runs stochastic variational inference, and prints the inferred
   bias **mu**.

Prerequisites
=============

* CaliPy, Pyro and PyTorch ≥ 2.2 installed (see :doc:`installation`).
* Python ≥ 3.8; GPU optional.

A tiny probabilistic model
============================

.. literalinclude:: ../../examples/calipy_demo_0.py
   :language: python
   :caption: *bias_plus_noise.py* – minimal runnable demo
   :emphasize-lines: 33, 36, 38-45, 50

Expected output
---------------

.. code-block:: console

   Posterior mu: around true mu = 0.0

(The exact numbers vary but should be identical to **mean(data)**.)

Where to go next
================

* :doc:`concepts` – glossary & architecture overview
* :doc:`usage` – in‑depth guides (models, effects, data, inference)
* Example notebooks in ``examples/engineering_geodesy/`` for **mean_estimation**, **level_calibration**, **totalstation_calibration**, …

Troubleshooting
===============

.. rubric:: Common hiccups

* **ImportErrors** – verify your environment matches the *Installation* page
  (“Bleeding‑edge install” section).
* **Diverging ELBO** – set ``torch.manual_seed(0)`` and/or lower the learning
  rate (``optim.Adam({"lr": 0.005})``).


