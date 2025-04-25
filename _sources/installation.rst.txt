========================
Installing **CaliPy**
========================

.. contents::
   :local:
   :depth: 2

Overview
========
CaliPy is *pre‑alpha* software.  You can either

1. install the latest **stable wheel** from PyPI (recommended for users), or  
2. clone the **development branch** from GitHub (recommended for
   contributors and bleeding‑edge features).

.. _install-prereqs:

Requirements
============

* Python ≥ 3.8 (3.11 tested)  ⎯ **64‑bit only**
* PyTorch ≥ 2.2 with CPU or CUDA ⩾ 11.8
* Pyro ≥ 1.9.0
* Optional: JupyterLab or VS Code for the example notebooks.
Quick install (PyPI)
====================

.. code-block:: console

   # create & activate a fresh environment (recommended)
    python3 -m venv calipy_env
    source calipy_env/bin/activate

   # ... then install calipy (including docs, if preferred)
   pip install calipy[docs]

That’s it!  Verify the installation:

.. code-block:: pycon

   >>> import calipy, torch, pyro
   >>> print(calipy.__version__)
   0.5.0

Bleeding‑edge install (GitHub)
==============================

.. code-block:: console

   (calipy) $ git clone https://github.com/atlasoptimization/calipy.git
   (calipy) $ cd calipy
   (calipy) $ pip install -e '.[dev,docs]'

The ``-e`` flag installs CaliPy in *editable* mode so local changes are
picked up immediately—handy when you extend the library.

Optional CUDA wheels
--------------------

If you have an NVIDIA GPU:

#. Check your CUDA version with ``nvidia-smi``.  
#. Install the matching PyTorch wheel, e.g.

   .. code-block:: console

      (calipy) $ pip install 'torch>=2.2+cu121' --index-url https://download.pytorch.org/whl/cu121

#. Re‑install CaliPy (same ``pip install calipy[docs]``).

Building the documentation locally
==================================

.. code-block:: console

   (calipy) $ cd calipy/docs
   (calipy) $ make clean html      # outputs to build/html/index.html
   (calipy) $ open build/html/index.html  # macOS; use xdg-open on Linux

Troubleshooting
===============

* **Mismatched PyTorch/Pyro versions**  
  Make sure the Pyro release you pick supports your installed PyTorch.
* **CUDA libraries not found**  
  Install the *matching* CUDA toolkit for your PyTorch wheel, or fall back to the CPU wheel.


Next steps
==========

* :doc:`quickstart` – run your first bias‑plus‑noise example
* :doc:`concepts` – core abstractions & design philosophy
* :doc:`usage` – how to build models, effects, data wrappers, inference



