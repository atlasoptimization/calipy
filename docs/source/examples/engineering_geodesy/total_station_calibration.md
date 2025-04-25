# Total-Station Axis-Error Estimation — Collimation & Trunnion in One Shot  
> *“Why juggle two clever face-swaps when you can let a probabilistic model do the algebra?”*

Example adapted from *Building and Solving Probabilistic Instrument Models with CaliPy*  
(Jemil Avers Butt, JISDM 2025)

```{admonition} What you’ll learn
:class: tip
* How to express the **collimation** (\(c\)) and **trunnion-axis** (\(i\)) errors in CaliPy
* How to encode face-I / face-II switching & elevation dependence without branching logic
* How SVI reproduces (and generalises) the classical two-step least-squares procedure
```

---

---

## 2 Instrument model

### 2.1 Geometry recap

| symbol | meaning | zero‑order impact on \(\phi_{obs}\) |
|--------|---------|---------------------------------------|
| \(c\) | angle between line‑of‑sight & collimation axis | \(\gamma_c = \dfrac{c}{\cos\beta}\) |
| \(i\) | non‑orthogonality of trunnion & vertical axis | \(\gamma_i = i\tan\beta\) |
| *face* | 0 (face I) / 1 (face II)                       | adds \(\pi\) for face II |

Linearised observation equation \[1]:

\[\phi_{obs}\;\sim\; \mathcal N\bigl(\underbrace{\tilde\phi\; + \; \text{face}\,\pi}_{\text{ideal reading}} - \gamma_c + 2\,\text{face}\,\gamma_c  - \gamma_i + 2\,\text{face}\,\gamma_i,\; \sigma\bigr).\]

*Literature derivation*: Deumlich (1980), pp. 132–134.

```{figure} ../../_static/figures/examples/engineering_geodesy/totalstation_sketch_1.png
:alt: Axis misalignments of a total station
:width: 80%

**Figure 1 —** Collimation error \(c\) (line-of-sight vs. collimation axis) and
trunnion error \(i\) (vertical vs. trunnion axis) distort horizontal angles. Figure taken from the paper by Butt(2025).
```

### 2.2 Why classical LS is cumbersome

1. **Two‑step trick required**: shoot a horizontal target (\(\beta\!=\!0\)) to isolate \(c\); then a steep target to solve for \(i\).
2. **Face pairing** must be perfect.  Any additional observations are discarded or averaged.
3. No uncertainty propagation for \(c,i\) beyond a posterior variance from the LS normal matrix.

*CaliPy* lets us encode Equation [1] directly; Stochastic Variational Inference (SVI) optimises the ELBO and returns a full posterior.

### 2.3 Calipy nodes

| Symbol | CaliPy class | Shape *(config × face)* |
|--------|--------------|-------------------------|
| \(c\)  | `UnknownParameter` | \(2\times2\) |
| \(i\)  | `UnknownParameter` | \(2\times2\) |
| Noise  | `NoiseAddition`    | \(2\times2\) |

```python
cfg   = dim_assignment(['cfg'],  [2])   # two target configurations
face  = dim_assignment(['face'], [2])   # Face-I / Face-II
scal  = dim_assignment(['_'],    [])    # parameter axis (empty)

c_ns = NodeStructure(UnknownParameter)
c_ns.set_dims(batch_dims = cfg+face, param_dims = scal)
c = UnknownParameter(c_ns, name='c')

i_ns = NodeStructure(UnknownParameter)
i_ns.set_dims(batch_dims = cfg+face, param_dims = scal)
i = UnknownParameter(i_ns, name='i')

noise_ns = NodeStructure(NoiseAddition)
noise_ns.set_dims(batch_dims = cfg+face, event_dims = scal)
noise = NoiseAddition(noise_ns)
```

### 2.4 Forward model

```python
class AxisErrorModel(CalipyProbModel):
    def model(self, input_vars, observations=None):
        face = input_vars['faces']            # 0 or 1
        β    = input_vars['beta']             # elevation angle

        c_val = c.forward()
        i_val = i.forward()

        γc = c_val / torch.cos(β)
        γi = i_val * torch.tan(β)

        μφ = (φ_true + torch.pi*face
              - γc + 2*face*γc
              - γi + 2*face*γi)

        return noise.forward({'mean': μφ,
                              'standard_deviation': σ_true},
                             observations)
    def guide(self,*_): pass
```

All non-linear trigonometry lives in **three** tensor lines.

---

## 3 Running Inference

```python
prob = AxisErrorModel()

inputs = {'faces': face_tensor,        # 2×2
          'beta' : β_tensor}           # 2×1  broadcast to 2×2

ϕ_obs = CalipyTensor(phi_data, dims=cfg+face)
loss  = prob.train(inputs, ϕ_obs,
                   optim_opts=dict(optimizer = pyro.optim.NAdam({"lr":0.01}),
                                   loss      = pyro.infer.Trace_ELBO(),
                                   n_steps   = 1_000))
```

---

## 4 Results

```
Node_1__param_c tensor(0.0100, grad=True)
Node_2__param_i tensor(0.0100, grad=True)

True values            : c=0.0100, i=0.0100
Hand-derived LS values : c=0.0099, i=0.0101
```

```{figure} ../_static/figures/examples/engineering_geodesy/axis_errors_elbo.png
:alt: ELBO learning curve for axis-error example
:width: 100%

**Figure 2 —** ELBO reaches the optimum in < 1 s (CPU).
```

> **Take-aways**
>
> * CaliPy matched *and* slightly refined the two-stage LS estimate
>   while using every observation jointly.
> * Face toggling & angle-dependent weights were expressed as ordinary tensors.
> * The same graph works for **many configs**, unknown \(\sigma\), or a prior on \(c,i\).  


## 6 Take‑away cheat‑sheet

| Concept | Classical workflow | CaliPy workflow |
|---------|--------------------|-----------------|
| Axis‑error compensation | two carefully chosen shots + manual algebra | *any* number of shots, any \(\beta\), automatic SVI |
| Extra points | discarded or averaged | increase posterior precision |
| Uncertainty | from normal‑matrix var‑cov | full approximate posterior |
| Code length | 100 + LOC incl. symbol mgmt | **< 60 LOC including simulation** |

> “CaliPy lets you **declare** the physics – the math and optimisation fade into the background.”

---

## 5 Next Steps

1. Feed **10 000** mixed-face shots with random \(\beta\); observe SVI’s advantage.
2. Treat \(\sigma\) as `UnknownVariance` for variance-component estimation.
3. Extend to **horizontal circle division errors** by adding a cyclic `NoiseAddition`.

---

## 6 Full Reproducible Code
```{literalinclude} ../../../../examples/engineering_geodesy/totalstation_calibration.py
:language: python
:caption: totalstation_calibration.py
```
