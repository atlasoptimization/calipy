# Two‑Peg Level Test — Estimating a Collimation Angle

> *“How do we turn a textbook levelling test into a fully‑Bayesian
>  inference problem that runs in a few lines of code?”*

Example adopted from *Building and Solving Probabilistic Instrument Models with CaliPy*
(Jemil Avers Butt, JISDM 2025, Karlsruhe). Available [here](https://publikationen.bibliothek.kit.edu/1000179733) DOI: 10.5445/IR/1000179733

```{admonition} What you’ll learn
:class: tip
* How to encode a classical two‑peg test in **CaliPy**
* How Bayesian inference (SVI) replaces hand‑derived Least Squares (LS) formulae
* How bayesian estimates and LS estimates coincide for simple Maximum likelihood estimation
```

---

## 1  Background — What **is** the Two‑Peg Test?

When you calibrate a digital or optical level you must know how far the
instrument’s line‑of‑sight deviates from a true horizontal.  That
mis‑alignment is called the **collimation angle** $\alpha$ (Fig.&nbsp;1).

<div align="center">

```{figure} ../../_static/figures/examples/engineering_geodesy/levelling_sketch_1.png
:alt: Geometry of the two-peg test with rods A and B and instrument setup
:width: 100%

**Figure 1 —** Two configurations of the two‑peg test. Figure taken from the paper by Butt(2025).  


For the classical two-peg test, rod A is always placed 60 m away from rod B and we level the instrument twice:

* **Config 1**: 30 m from each rod.  
* **Config 2**: directly in front of rod A and 60 m from rod B.

</div>

The classical (“deterministic”) solution uses two height readings
$y_A^k,\;y_B^k$ per configuration and some algebra to isolate
$\tan\alpha$.  That works *only* because the geometry is simple and the
error model is Gaussian and homoscedastic.

*CaliPy* lets us write a **probabilistic** version instead:

$$
\begin{aligned}
y_A^k &\sim \mathcal N \bigl(h_I^{(k)}
              + l_A^{(k)}\tan\alpha,\; \sigma \bigr) \\
y_B^k &\sim \mathcal N \bigl(h_I^{(k)}-\Delta H
              + l_B^{(k)}\tan\alpha,\; \sigma \bigr) ,
\end{aligned}
$$

where

| symbol | meaning | dim |
| ------ | ------- | --- |
| $k$          | configuration index             | $(n_\text{conf})$ |
| $l_A, l_B$   | distances instrument→rod        | $(n_\text{conf}\times 2)$ |
| $h_I^{(k)}$  | unknown sight‑line height       | $(n_\text{conf})$ |
| $\Delta H$   | unknown rod‑to‑rod true height difference | scalar |
| $\alpha$     | collimation angle (wanted)      | scalar |
| $\sigma$     | known standard deviation| scalar of measurements |

```{admonition} Modelling insight
:class: note
Treating the instrument height $h_I^{(k)}$ as an *UnknownParameter* lets CaliPy
estimate it automatically. You get the same estimator for $\alpha$
while having to perform zero extra algebra manually.
```

---

## 2  Implementation — *CaliPy* Building Blocks

### 2.1  Dim‑aware anatomy

| Building block | Code class | Why we need it |
| -------------- | ---------- | -------------- |
| **Parameters** $\alpha,\;h_I,\;\Delta H$ | `UnknownParameter` | Provide init value & learn later |
| **Noise injection** for eq.&nbsp;\eqref{eq:model} | `NoiseAddition` | Wraps Pyro’s `Normal` |
| **Node structure** (dims, plates) | `NodeStructure` | Tells each node where batch & event axes live |
| **Probabilistic model** | subclass of `CalipyProbModel` | Chains everything & calls `forward()` |
| **Inference engine** | Pyro’s SVI (Trace_ELBO) | Runs automatically inside `probmodel.train()` |

```python
# ── Dimensions ─────────────────────────────────────────
n_conf = 2
batch_k   = dim_assignment(['conf'],  [n_conf])   # configurations k
batch_AB  = dim_assignment(['AB'],    [2])        # A or B per conf
scalar    = dim_assignment(['_'],     [])         # empty (=scalar)

# ── Unknowns ───────────────────────────────────────────
alpha_ns = NodeStructure(UnknownParameter)
alpha_ns.set_dims(batch_dims=batch_k+batch_AB, param_dims=scalar)
alpha    = UnknownParameter(alpha_ns, name='alpha',
                            init_tensor=torch.tensor(1e-2))

hI_ns = NodeStructure(UnknownParameter)
hI_ns.set_dims(batch_dims=batch_AB, param_dims=batch_k)
hI    = UnknownParameter(hI_ns, name='h_I')

dH_ns = NodeStructure(UnknownParameter)
dH_ns.set_dims(batch_dims=batch_k+batch_AB, param_dims=scalar)
dH    = UnknownParameter(dH_ns, name='dH')

# ── Measurement noise ─────────────────────────────────
noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims=batch_k+batch_AB, event_dims=scalar)
noise    = NoiseAddition(noise_ns)
```

### 2.2  Forward model in code

```python
class TwoPegProbModel(CalipyProbModel):
    def model(self, input_vars, observations=None):
        l_AB = input_vars.value                 # shape  (n_conf, 2)

        # draw unknowns
        a   = alpha.forward()                  # shape  (n_conf,2) broadcast
        h_I = hI.forward().T                   # shape  (n_conf,1)
        dH  = dH.forward()                     # scalar broadcast

        # deterministic signal  y_true + Δ
        scaler = torch.hstack([torch.zeros([n_conf,1]),
                               torch.ones ([n_conf,1])])
        y_true   = h_I - scaler * dH
        y_mean   = y_true + torch.tan(a) * l_AB

        # observation node
        out = noise.forward({'mean':y_mean, 'standard_deviation':sigma_true},
                            observations)
        return out
```

Internally `.forward()` places every `sample()` (Pyro) site under
vectorised `plate`s **generated automatically** from your
`NodeStructure`.  No manual plates & broadcasting gymnastics needed.

---

## 3  Running Inference

```python
probmodel = TwoPegProbModel()
input_data = l_mat                              # distances (torch tensor)
y_obs      = CalipyTensor(data, dims=batch_k+batch_AB)

elbo_curve = probmodel.train(input_data, y_obs,
                             optim_opts=dict(optimizer=pyro.optim.NAdam({"lr":1}),
                                             loss      =pyro.infer.Trace_ELBO(),
                                             n_steps   =1000))
```

*Behind the scenes*

1. **SVI loop** builds the computation graph once, then
   back‑propagates the ELBO gradient each iteration.
2. Parameters `alpha`, `h_I`, `dH` live in Pyro’s *param store*.
3. Sampling statements are conditioned on `y_obs` automatically.

```python
epoch: 0 ; loss : 63942.6328125
epoch: 100 ; loss : 35856.48046875
epoch: 200 ; loss : -21.9062442779541
epoch: 300 ; loss : -23.955263137817383
epoch: 400 ; loss : -23.955265045166016
epoch: 500 ; loss : 6660.3935546875
epoch: 600 ; loss : 12589833.0
epoch: 700 ; loss : 56793.0546875
epoch: 800 ; loss : 101.84868621826172
epoch: 900 ; loss : -23.81795310974121
Node_1__param_alpha 
 tensor(-12.5654, requires_grad=True)
Node_2__param_hI 
 tensor([0.9879, 1.0109], requires_grad=True)
Node_3__param_dh 
 tensor(0.4997, requires_grad=True)
True values 
 alpha : 0.0010000000474974513 
 dh : 0.5 
 hI : tensor([0.9889, 1.0120])
Values estimated by least squares 
 alpha : 0.0010212835622951388 
 dh : 0.49987077713012695 
 hI : tensor([[0.9878],
        [1.0108]])
```

---

## 4  Results

The following graph illustrates the value of the ELBO loss. Note that the behavior is non-monotonic 
and at around epoch 500 the optimization converges to an equivalent value for alpha by adding $4 \pi$
to the previous estimate. in the end the LS and calipy estimates coincide - apart from the irrelevant
constant $ 4 \pi$ in the collimation angle.

```{figure} ../../_static/figures/examples/engineering_geodesy/levelling_sketch_2.png
:alt: Sketch of the ELBO learning curve
:width: 100%

**Figure 2 —** The ELBO loss during learning of the parameters.  


```python
for name,val in pyro.get_param_store().items():
    print(f"{name:18s}  {val.detach().cpu().numpy()}")
```

| quantity | true | inferred (SVI) | classical LS |
| -------- | ---- | -------------- | ------------ |
| $\alpha$     | 1.0 mrad | 0.97 mrad - 4 pi | 1.0 mrad |
| $\Delta H$   | 0.5 m   | 0.4997 m   | 0.4998 m |
| $h_I^{(1)}$  | … | … | … |
| $h_I^{(2)}$  | … | … | … |

> **Take‑aways**
>
> * The Bayesian estimates matches the closed‑form LS estimate for this
>   simple geometry — exactly what theory predicts.
> * UnknownParameter trick: leaving $h_I^{(k)}$ and $\Delta H$ undetermined
>   saves you some manual math.
> * The optimizer can behave unexpectedly. Since most gradient based optimizers
>   explicitly try to escape local minima, the estimators can jump around quite a bit.
> * If you add more configurations, heteroscedastic noise, or
>   hierarchical priors, `CaliPy` scales effortlessly while the
>   hand‑derived LS formula breaks down.

---

## 5  Key Insights

* **Declarative modelling** – once nodes are chained, *CaliPy* converts
  the graph into Pyro sample sites and plates.
* **Dimension‑aware tensors** – `CalipyTensor` stores both data **and**
  semantic dimensions.  Broadcasting & sub‑batching are handled for you.
* **Swap‑in inference** – ELBO/SVI here, but you could plug in HMC or an
  AutoGuide without touching the model.

---

## 6  Next Steps

1. Play with `n_conf > 2`, unequal $\sigma$, or informative priors.
2. Replace the `UnknownParameter` nodes by `CalipyDistribution.Normal`
   to let $\alpha$ have a *Gaussian* prior.
3. Try sub‑batch training on large simulated levelling campaigns.

---


## 7  Full code

```{literalinclude} ../../../../examples/engineering_geodesy/level_calibration.py
:language: python
:caption: level_calibration.py
```

