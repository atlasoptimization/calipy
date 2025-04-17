# Bias Estimation (Mean Estimation)
This example demonstrates how to estimate the mean value $\mu$ of a normally distributed quantity using CaliPy.

A single effect chain is defined:
- An unknown parameter $\mu$ (`UnknownParameter`)
- Combined with known Gaussian noise (`NoiseAddition`)

We simulate noisy data $y_i \sim \mathcal{N}(\mu, \sigma)$ and estimate $\mu$ via stochastic variational inference (SVI).

## Setup

```python
import torch
import pyro
import matplotlib.pyplot as plt

from calipy.core.base import NodeStructure, CalipyProbModel
from calipy.core.effects import UnknownParameter, NoiseAddition
from calipy.core.utils import dim_assignment
from calipy.core.tensor import CalipyTensor

n_meas = 20
mu_true = torch.tensor(0.0)
sigma_true = torch.tensor(0.1)

data = pyro.distributions.Normal(mu_true, sigma_true).sample([n_meas])
```

We simulate $n = 20$ noisy measurements from a true process with mean $\mu = 0.0$ and standard deviation $\sigma = 0.1$. The data are assumed i.i.d. and normally distributed:

$$
y_i \sim \mathcal{N}(\mu, \sigma), \quad i = 1, \dots, n
$$

This forms the input to our calibration model.

In CaliPy, we also define *dimension objects* (`DimTuple`) to represent the batch and event axes of each tensor. These give semantics to the data and enable CalipyTensor operations to be dimension-aware.


## Model Definition

```python
batch_dims = dim_assignment(['bd_1'], [n_meas])
event_dims = dim_assignment(['ed_1'], [])
param_dims = dim_assignment(['pd_1'], [])

mu_ns = NodeStructure(UnknownParameter)
mu_ns.set_dims(batch_dims=batch_dims, param_dims=param_dims)
mu_object = UnknownParameter(mu_ns, name='mu')

noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims=batch_dims, event_dims=event_dims)
noise_object = NoiseAddition(noise_ns, name='noise')

class DemoProbModel(CalipyProbModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mu_object = mu_object
        self.noise_object = noise_object

    def model(self, input_vars=None, observations=None):
        mu = self.mu_object.forward()
        inputs = {'mean': mu, 'standard_deviation': sigma_true}
        return self.noise_object.forward(inputs, observations)

    def guide(self, input_vars=None, observations=None):
        pass

demo_probmodel = DemoProbModel()
```

In CaliPy, models are composed from **effect nodes** that wrap known or unknown components of the process.

In this example:

- `UnknownParameter` defines $\mu$ as a latent scalar parameter. It is a learnable random variable with a simple default prior.
- `NoiseAddition` defines the observed variable $y_i$, modeled as $\mu + \varepsilon_i$, where $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$.

We declare a `NodeStructure` for each node, specifying its role (batch/event/parameter). These dimensions ensure tensors are aligned correctly, and nodes can broadcast over batch or event axes without ambiguity.

The entire model is wrapped in a `CalipyProbModel`, which provides the interface to Pyro's inference engine.


## Training

```python
data_cp = CalipyTensor(data, dims=batch_dims + event_dims)

adam = pyro.optim.NAdam({"lr": 0.01})
elbo = pyro.infer.Trace_ELBO()

optim_opts = {'optimizer': adam, 'loss': elbo, 'n_steps': 1000}
loss = demo_probmodel.train(None, data_cp, optim_opts=optim_opts)

plt.figure(dpi=300)
plt.plot(loss)
plt.title("ELBO Loss")
plt.xlabel("Epoch")
plt.tight_layout()
plt.show()
```

To perform inference, we use Pyro's stochastic variational inference (SVI).

- The model is defined by calling `forward()` on the effect chain
- The guide is empty here (since the only latent is the `UnknownParameter`, which already registers a variational posterior via Pyro’s `param`)
- We use `Trace_ELBO` as the loss
- The `.train()` method handles SVI optimization automatically

During training, CaliPy collects the ELBO loss at each epoch, and stores parameters in Pyro’s `param_store`.


## Results

```python
for name, val in pyro.get_param_store().items():
    print(name, val)

print("True μ:", mu_true.item())
print("Empirical mean:", torch.mean(data).item())
```

After training, we inspect the parameter values and ELBO:

- The inferred value $\hat{\mu}$ converges to the empirical average of the data
- The ELBO decreases smoothly over training epochs, confirming successful optimization
- The model learns a variational posterior over $\mu$ centered near the true value

This example demonstrates how CaliPy mirrors classical least-squares estimation in structure, while embracing a fully probabilistic treatment. Model structure, dimensional semantics, and probabilistic inference are all encoded cleanly and declaratively using effects.

