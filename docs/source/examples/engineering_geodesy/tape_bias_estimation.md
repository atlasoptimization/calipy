# Tape-Measure Bias Estimation — Finding a Hidden Offset  
> *“Can we recover a constant tape-bias automatically instead of
>  hand-deriving the arithmetic mean?”*

Example adapted from *Building and Solving Probabilistic Instrument Models with CaliPy*  
(Jemil Avers Butt, JISDM 2025, Karlsruhe) DOI:&nbsp;10.5445/IR/1000179733

```{admonition} What you’ll learn
:class: tip
* How to encode the Gaussian model \(y\sim\mathcal N(\mu-\theta,\sigma)\) in **CaliPy**
* How SVI reproduces the closed-form Least-Squares estimator for \(\theta\)
* How dimension-aware nodes shrink boiler-plate in even the simplest model
```

---

## 1  Background — Why Bias Matters

A steel tape often reads **too short or too long** by a fixed amount
\(\theta\).  When you measure a rod of known length \(\mu\) (Fig.&nbsp;1),
each observation deviates by that unknown bias plus random noise \(\varepsilon\):

\[
\boxed{y = \mu - \theta + \varepsilon},\qquad
\varepsilon\sim\mathcal N(0,\sigma).
\]

Classically you solve
\(\hat\theta = \tfrac1n\sum_{k=1}^n(\mu - y_k)\) by hand
(eq.&nbsp;(1) in the paper).  
*CaliPy* lets us phrase exactly that likelihood and let **SVI** return the
same answer automatically.

```{figure} ../../_static/figures/examples/engineering_geodesy/tapemeasure_sketch_1.png
:alt: Tape reads a rod that is longer than the tape shows by the bias θ
:width: 100%

**Figure 1 —** Schematic tape measurement with unknown bias \(\theta\). Figure taken from the paper by Butt(2025).
```

---

## 2  Implementation — 12 Lines to a Probabilistic Model

### 2.1  Dimensions & Nodes

| Component | CaliPy class | Shape *(batch × param)* |
|-----------|--------------|-------------------------|
| Bias \(\theta\) | `UnknownParameter` | \(n_\text{obs} \times 1\) |
| Gaussian noise | `NoiseAddition` | \(n_\text{obs} \times 1\) |

```python
n_obs   = 20
batch   = dim_assignment(['obs'], [n_obs])   # i.i.d. samples
scalar  = dim_assignment(['_'],    [])       # empty (=param axis)

# unknown bias θ
θ_ns = NodeStructure(UnknownParameter)
θ_ns.set_dims(batch_dims=batch, param_dims=scalar)
θ     = UnknownParameter(θ_ns, name='theta')

# noise wrapper  𝒩(mean, σ)
noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims=batch, event_dims=scalar)
noise = NoiseAddition(noise_ns)
```

### 2.2  Forward Model

```python
class TapeBiasProbModel(CalipyProbModel):
    def model(self, _, observations=None):
        θ_val = θ.forward()                       # shape (n_obs,)
        mean  = mu_true - θ_val
        out   = noise.forward({'mean': mean,
                               'standard_deviation': sigma_true},
                              observations)
        return out
```

*No manual plates, broadcasting, or log-prob math required.*

---

## 3  Running Inference

```python
probmodel = TapeBiasProbModel()

y_obs = CalipyTensor(data, dims=batch)          # simulated data
elbo = probmodel.train(None, y_obs,
                       optim_opts=dict(optimizer = pyro.optim.NAdam({"lr":0.01}),
                                       loss      = pyro.infer.Trace_ELBO(),
                                       n_steps   = 1_000))
```

---

## 4  Results

```text
Node_1__param_theta
 tensor(0.0103, requires_grad=True)
True θ     : 0.0100
Empirical  : 0.0101   # arithmetic mean (LS)
```

```{figure} ../_static/figures/examples/engineering_geodesy/tapemeasure_elbo.png
:alt: ELBO learning curve for tape-bias example
:width: 100%

**Figure 2 —** ELBO converges smoothly to the global optimum.
```

> **Take-aways**
>
> * **SVI = LS** for a linear-Gaussian one-parameter model — check.
> * Declaring `UnknownParameter` removed *all* algebra from eq.&nbsp;(1).
> * The same scaffold scales to heteroscedastic σ or priors in one line.

---

## 5  Key Insights

* **Declarative ≠ verbose** – this entire example uses < 40 lines of
  actual model code.
* **Dimension-aware tensors** keep sample shapes and math symbols in sync.
* **Swap-in inference** – change `Trace_ELBO` to `NUTS` and you
  immediately get MCMC draws.

---

## 6  Next Steps

1. Make \(\sigma\) an `UnknownVariance` and learn it too.  
2. Give \(\theta\) a Gaussian prior centred at the manufacturer’s spec.  
3. Simulate 10 000 observations and try sub-batch SVI.

---

## 7  Full Code

```{literalinclude} ../../../../examples/engineering_geodesy/tapemeasure_calibration.py
:language: python
:caption: tapemeasure_calibration.py
```
