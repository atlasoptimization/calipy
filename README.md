# CaliPy: Calibration Library in Python
> **Pre-Alpha**: In active development — not yet production-ready

---

## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Bayesian vs. Classical (LS) Estimation](#bayesian-vs-classical-ls-estimation)
- [Installation](#installation)
- [Quick Example](#quick-example)
- [Use Cases](#use-cases)
- [References](#references)
- [License](#license)
- [Contributing](#contributing)

---

## Introduction

CaliPy (Calibration Library for Python) is designed to **simplify and standardize** how we **build and solve probabilistic instrument models** in geodesy (and related fields). While many real-world analyses rely on classical **least squares** (LS) estimation or maximum-likelihood solutions, these approaches can fail for:

- Nonlinear systems
- Non-Gaussian noise
- Latent or unobserved variables
- Large-scale or sub-batched data

CaliPy addresses these limitations by:

- **Leveraging** [Pyro](https://pyro.ai/) for flexible **Bayesian inference** (e.g., Stochastic Variational Inference [(Blei et al., 2017)](#references)).
- **Allowing chainable “stochastic effects”** to reflect real-world complexities in measurement instruments.
- **Encouraging** a “declare-then-solve” approach: once the user sets up a model, CaliPy automatically handles inference—whether you want a posterior distribution or a maximum-likelihood point estimate.

Originally developed for **geodetic measurement instruments**, CaliPy can also apply to other domains requiring advanced calibration or hierarchical Bayesian workflows under a friendly, composable interface.

---

## Key Features

1. **Chainable Instrument Models**  
   Construct models from small, **pre-built node classes** (like `UnknownParameter`, `NoiseAddition`) that capture aspects such as bias, drift, or axis misalignments in measurement instruments.

2. **Seamless Integration with PyTorch & Pyro**  
   CaliPy builds on [PyTorch](https://pytorch.org/) [(Paszke et al., 2019)](#references) for automatic differentiation and [Pyro](https://pyro.ai/) [(Bingham et al., 2018)](#references) for advanced Bayesian inference algorithms.

3. **Bayesian or Classical**  
   - For linear Gaussian models, **maximum-likelihood** solutions coincide with least squares (LS).  
   - For more general models, **Stochastic Variational Inference** or other Pyro-based inference can handle non-Gaussian, nonlinear, or latent-variable scenarios automatically.

4. **Dimension-Aware Data Structures**  
   CaliPy offers dimension-handling (inspired by `functorch.torchdims`), letting you specify shapes, subbatching strategies, and independence assumptions with minimal overhead.

5. **Rapid Prototyping & Extendability**  
   - The library’s architecture is modular; you can add new “effects” for specialized phenomena (e.g., scanning-laser sensor errors, environment-driven drifts).  
   - A “declare-then-solve” approach means minimal user code from model definition to inference result.

---

## Architecture Overview

1. **Node Classes**  
   Each effect (e.g., `UnknownParameter`, `NoiseAddition`) inherits from a base node class, providing a `forward()` method for random draws or deterministic transformations.

2. **Indexing & Subbatching**  
   - **CalipyIndexer**: centralizes creation of local/global index tensors, block indices for subbatching, and index naming.  
   - **CalipyObservation**: aggregates data plus metadata about batch/event dims, bridging user data with inference routines.

3. **CalipyProbModel**  
   - Users typically write a `model()` (and optionally a `guide()`) capturing the generative structure of their problem.  
   - This class compiles your code with Pyro’s back-end to perform SVI or other inference algorithms seamlessly.

---

## Bayesian vs. Classical (LS) Estimation

1. **Classical Approach**  
   - **Least Squares (LS)** solves a maximum-likelihood problem in Gaussian, linear scenarios by minimizing 
     \[
       \sum \bigl(\text{predictions} - \text{observations}\bigr)^2.
     \]
   - It has closed-form solutions for linear, Gaussian assumptions [(Ghilani and Wolf, 2006)](#references).

2. **Bayesian Extension**  
   - Instead of a single estimate \(\hat{\theta}\), we specify a full model:
     \[
       p(\theta, \mathrm{data}) = p(\mathrm{data}\mid\theta)\,p(\theta).
     \]
     Then infer the posterior
     \[
       p(\theta\mid \mathrm{data}) = \frac{p(\mathrm{data}\mid\theta)\,p(\theta)}{p(\mathrm{data})}.
     \]
   - **Stochastic Variational Inference (SVI)** approximates the posterior with an ELBO objective, bridging complex or large-scale problems.  
   - This approach gracefully handles **non-Gaussian noise**, **nonlinear** relationships, or **latent variables**—cases where LS is no longer straightforward.

3. **Connection**  
   - In purely Gaussian, linear scenarios with non-informative priors, Bayesian approaches reduce to classical LS solutions.  
   - For more complex or large-scale problems, the same code in CaliPy can use SVI or MCMC to produce approximate posteriors or ML solutions.

---

## Installation

**Project Status**: Pre-alpha

Once published on PyPI, CaliPy can be installed via:

```bash
pip install calipy
```

Until then, clone this repository and install dependencies manually:

```bash
git clone https://github.com/atlasoptimization/calipy.git
cd calipy
pip install -r requirements.txt
```

---

## Quick Example

Below is a toy snippet demonstrating how you might declare a simple **bias-plus-noise model**:

```python

"""
    1. Imports and definitions
"""

# i) Imports
# base packages
import torch
import pyro
import matplotlib.pyplot as plt

# calipy
from calipy.core.base import NodeStructure, CalipyProbModel
from calipy.core.effects import UnknownParameter, NoiseAddition
from calipy.core.utils import dim_assignment
from calipy.core.tensor import CalipyTensor


# ii) Definitions

n_meas = 20


"""
    2. Simulate some data
"""

# i) Set up sample distributions
mu_true = torch.tensor(0.0)
sigma_true = torch.tensor(0.1)


# ii) Sample from distributions
data_distribution = pyro.distributions.Normal(mu_true, sigma_true)
data = data_distribution.sample([n_meas])


"""
    3. Load and customize effects
"""

# i) Set up dimensions
batch_dims = dim_assignment(['bd_1'], dim_sizes = [n_meas])
event_dims = dim_assignment(['ed_1'], dim_sizes = [])
param_dims = dim_assignment(['pd_1'], dim_sizes = [])

# ii) Set up dimensions for mean parameter mu
# mu setup
mu_ns = NodeStructure(UnknownParameter)
mu_ns.set_dims(batch_dims = batch_dims, param_dims = param_dims,)
mu_object = UnknownParameter(mu_ns, name = 'mu')

# iii) Set up the dimensions for noise addition
noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims = batch_dims, event_dims = event_dims)
noise_object = NoiseAddition(noise_ns, name = 'noise')
        

"""
    4. Build the probmodel
"""

# i) Define the probmodel class 
class DemoProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # integrate nodes
        self.mu_object = mu_object
        self.noise_object = noise_object 
        
    # Define model by forward passing
    def model(self, input_vars = None, observations = None):
        mu = self.mu_object.forward()       
        inputs = {'mean':mu, 'standard_deviation': sigma_true} 
        output = self.noise_object.forward(input_vars = inputs,
                                           observations = observations)
        return output
    
    # Define guide (trivial since no posteriors)
    def guide(self, input_vars = None, observations = None):
        pass
    
demo_probmodel = DemoProbModel()
    

"""
    5. Perform inference
"""

# i) Set up optimization
adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()
n_steps = 1000

optim_opts = {'optimizer': adam, 'loss' : elbo, 'n_steps': n_steps}

# ii) Train the model
input_data = None
data_cp = CalipyTensor(data, dims = batch_dims + event_dims)
optim_results = demo_probmodel.train(input_data, data_cp, optim_opts = optim_opts)


"""
    6. Analyse results and illustrate
"""

# i)  Plot loss

plt.figure(1, dpi = 300)
plt.plot(optim_results)
plt.title('ELBO loss')
plt.xlabel('epoch')

# ii) Print  parameters
for param, value in pyro.get_param_store().items():
    print(param, '\n', value)
    
print('True values of mu = ', mu_true)
print('Results of taking empirical means for mu_1 = ', torch.mean(data))


```

This snippet shows how you might define a node-based approach to an unknown parameter \(\mu\) and noise, letting CaliPy handle the inference behind the scenes.

---

## Use Cases

1. **Tape Bias Estimation**  
   - Classic scenario: measure a known rod length with a tape that has an unknown offset \(\theta\).  
   - Equivalent to linear maximum-likelihood in simplest form, but easily extended in CaliPy for more complex error structures or prior knowledge.

2. **Two-Peg Test (Level Collimation)**  
   - Solve for the collimation angle \(\alpha\) in a leveling instrument.  
   - Chain multiple observations across distinct geometric setups. For more complicated geometry or weighting, SVI seamlessly generalizes the solution.

3. **Axis Errors in Total Stations**  
   - Model collimation or trunnion axis misalignments, even under strongly nonlinear geometry or face configurations.  
   - Simple to incorporate discrete “face” variables, e.g. Face I/Face II, in a single forward pass.

These three example were presented at JISDM 2025 in Karlsruhe; documented code can be found in the [examples folder](https://github.com/atlasoptimization/calipy/tree/master/calipy/examples/engineering_geodesy) 
---

## References

- **Probabilistic Programming**  
  - Bingham, E., et al. (2018). *Pyro: Deep Universal Probabilistic Programming*. arXiv e-prints, 1810.09538.  
  - Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). *Variational Inference: A Review for Statisticians*. JASA, 112(518), 859–877.  
  - Gelman, A., et al. (2020). *Bayesian Workflow*. arXiv:2011.01808.  
  - Paszke, A., et al. (2019). *PyTorch: an imperative style, high-performance deep learning library*. NeurIPS.  

- **Calibration, Geodesy & LS**  
  - Ghilani, C. D., & Wolf, P. R. (2006). *Adjustment Computations - Spatial Data Analysis*. Wiley.  
  - Uren, J. & Price, W. F. (2006). *Surveying for Engineers*. Palgrave Macmillan.  
  - Phillips, S. D. et al. (2001). *A Careful Consideration of the Calibration Concept*. J. Res. NIST.  
  - Deumlich, F. (1980). *Instrumentenkunde der Vermessungstechnik*. VEB Verlag fuer Bauwesen.  
  - Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.  
  - Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.

---

## License

CaliPy is released under a **Prosperity Public License** for non-commercial use. For commercial usage, a separate license is available. See [LICENSE](./LICENSE.md) for details.

---

## Contributing

The project is currently **Pre-Alpha**, so expect changes. We welcome bug reports, new effect classes, and general improvements. To contribute:

1. **Fork** the [GitHub repo](https://github.com/atlasoptimization/calipy).
2. Make your changes or propose new classes/effects.
3. **Open a pull request** to discuss your contribution.

Thank you for your interest in CaliPy! We hope it accelerates your research or engineering in advanced instrument calibration and beyond.


