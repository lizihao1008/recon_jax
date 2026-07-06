# recon_jax

Differentiable density-field reconstruction from galaxy surveys, in JAX.

`recon_jax` is a re-implementation of the **galaxy (spectroscopic + photometric)**
reconstruction idea from [TARDIS](https://arxiv.org/abs/1903.09049)
(*Tomographic Absorption Reconstruction and Density Inference Scheme*, the
`tardis-tf` code in the parent folder) on top of the modern JAX cosmology stack:

| role | TARDIS (TensorFlow) | recon_jax (JAX) |
|------|---------------------|-----------------|
| differentiable N-body forward model | flowPM | **JaxPM** (`lpt`, `make_ode_fn`, `cic_paint`) + `diffrax` |
| cosmology / growth / P(k) | astropy + tabulated file | **jax_cosmo** |
| optimiser | SciPy `L-BFGS-B` via TF | **optax `lbfgs`** (zoom line search) |

The Lyman-α forest part of TARDIS is intentionally **not** ported — this package
reconstructs the density field from galaxy positions and redshift errors only.

## The idea

We look for the *initial* (linear) density field whose gravitationally evolved,
biased galaxy distribution best matches the observed catalogue. Because the whole
forward model is differentiable, we can optimise the ~`nc³` field values directly
with gradient-based L-BFGS.

The objective is a negative log-posterior. The **default** likelihood follows
[Horowitz & Melchior (2023)](https://arxiv.org/abs/2306.15733), *Joint Cosmic
Density Reconstruction from Photometric and Spectroscopic Samples*, which
improves on the earlier L2 galaxy likelihood:

```
L(δ_lin) =  galaxy_fac · Σ_i [ φ_i − n'_i · log φ_i ]     (Poisson point-process likelihood, Eq. 4)
         +  prior_fac  · Σ_k |δ_lin(k)|² / P(k)           (Gaussian field prior, Eq. 10)
```

* **Poisson likelihood** (`mode="poisson"`, default). The galaxies are modelled
  as a draw from an inhomogeneous Poisson process whose rate `φ_i = N̄·(1+δ_g)`
  is the forward-modelled expected galaxy count in cell `i`. This is more
  accurate for small-scale power and — unlike the Gaussian/L2 approximation —
  stays valid at **low occupancy (≲5 galaxies per cell)**.
* **Continuous Poisson process for photo-z** (Eqs. 7–8). Instead of giving each
  galaxy a free line-of-sight coordinate, every galaxy's count is spread over
  cells along the line of sight by its redshift PDF `p(z_k)` (a Gaussian of width
  `σ_z`). The resulting fractional counts `n'_i` are handled by the continuous
  (incomplete-Gamma) generalisation of the Poisson distribution; its `Γ(n'_i)`
  normalisation is independent of the field and drops out of the optimisation.
  Photo-z uncertainty is thus baked into the fixed data field — **no per-galaxy
  parameters**.

The earlier TARDIS-2021 objective is kept as an option (`mode="l2"`), where the
redshift errors instead enter through a per-galaxy prior on free line-of-sight
coordinates:

```
L(δ_lin, z_gal) =  galaxy_fac · ‖smooth(δ_g^model) − smooth(δ_g^data)‖²
                +  redshift_fac · Σ_g ((z_g − z_g^obs)/σ_z,g)²
                +  prior_fac  · Σ_k |δ_lin(k)|²/P(k)
```

* **Forward model** (`forward.py`): `δ_lin → 2LPT (→ optional PM integration) → RSD → CIC → δ_m`.
* **Redshift-space distortions** (`forward.py`): TARDIS's RSD operator, in the
  particle representation. Each particle is shifted along the line of sight by its
  peculiar velocity. In JaxPM units the comoving shift is `Δs = a·(dx/da) =
  p_los / (a²·E(a))`, which for the Zel'dovich flow is exactly the Kaiser form
  `s = x + f·Ψ`. The galaxy field is therefore built in **redshift space**, to be
  compared with a redshift-space catalogue. Toggle with `config.rsd`; choose the
  line of sight with `config.los_axis`.
* **Galaxy bias** (`painting.py`): `δ_g = b₁·δ_m + b₂²·δ_m²` (Eulerian).
* **Redshift errors** — the input `los_sigma`. In the default Poisson mode they
  set the width of each galaxy's redshift kernel `p(z_k)` in the
  continuous-Poisson data field (`loss.build_data_counts`); small `σ_z` →
  spectroscopic (near-delta kernel), large `σ_z` → photometric (broad kernel).
  In the legacy L2 mode they instead weight a per-galaxy prior on free
  line-of-sight coordinates. Either way the same catalogue handles spec-z and
  photo-z in one model.
* **Annealing** (`reconstruct.py`): L-BFGS is run at a sequence of decreasing
  Gaussian smoothing scales (coarse → fine). The Poisson likelihood prefers a
  light schedule (e.g. `(2, 1, 0)` cells); heavy smoothing flattens the Poisson
  term and starves the gradient.

## Inputs

The only observational input is a `GalaxyCatalog`:

* `positions` — `(N_gal, 3)` array in **grid units** `[0, nc)`; column 2 is the
  line-of-sight (redshift) coordinate.
* `los_sigma` — `(N_gal,)` per-galaxy 1σ redshift uncertainty, in grid cells.

(Convert physical Mpc/h to grid units with `x_cells = x_mpc / config.cell_size`,
and a redshift error to a comoving line-of-sight error the same way.)

## Setup (no `pip install` for this repo)

`recon_jax` is a **path-based** package: clone the repo and put the project
root on `PYTHONPATH`. Only the third-party JAX stack needs to be installed.

```bash
# once per shell / Jupyter kernel
export PYTHONPATH="/path/to/field_recon/recon_jax:$PYTHONPATH"

# or
source /path/to/field_recon/recon_jax/env.sh
```

Install dependencies (jax, jaxpm, jax-cosmo, diffrax, optax, numpy):

```bash
pip install -r requirements.txt
```

In a script under ``examples/``, add the project root explicitly (``demo.py`` does
this), or set ``PYTHONPATH`` / ``source env.sh`` first.

In a notebook at the project root (usual Jupyter cwd):

```python
import bootstrap
from recon_jax import ReconConfig, Reconstructor
```

Directory layout (the folder added to `PYTHONPATH` is the outer `recon_jax/`):

```
field_recon/recon_jax/          ← add this to PYTHONPATH
  bootstrap.py
  recon_jax/                    ← importable package
    __init__.py
    config.py
    ...
  examples/demo.py
```

Optional: `pip install -e .` still works via `pyproject.toml`, but is not required.

## Usage

```python
import numpy as np
from recon_jax import ReconConfig, Reconstructor, GalaxyCatalog

cfg = ReconConfig(
    nc=32, box_size=128.0,      # 128 Mpc/h box on a 32³ mesh
    z_init=20.0, z_final=0.0,   # evolve from z=20 to the observed redshift
    n_body=False,               # 2LPT only (fast); True = full PM via diffrax
    rsd=True, los_axis=2,       # galaxy field placed in redshift space
    b1=1.0, b2=0.0,             # galaxy bias
    anneal_scales=(2., 1., 0.), # light schedule suits the Poisson likelihood
    maxiter=(25, 25, 50),
)

cat = GalaxyCatalog.from_arrays(xyz_grid_units, los_sigma_cells)

rec = Reconstructor(cfg, likelihood="poisson")   # default; "l2" for the legacy loss
out = rec.run(cat)

out["delta_m"]  # reconstructed evolved density field  (nc, nc, nc)
out["linear"]   # reconstructed linear/initial field
```

The observation epoch is set by **redshift**: `z_final` is the redshift of your
galaxies (e.g. `z_final=7.0` for a z≈7 sample), and `z_init` is the (higher)
starting redshift of the forward evolution. The scale factors `a = 1/(1+z)` are
derived automatically (`cfg.a_init`, `cfg.a_final`), and the RSD operator and
growth factors follow `z_final` self-consistently. Leave the power spectrum at
z=0 — JaxPM's `lpt` handles the growth from the initial field internally.

For spectroscopic data, pass a small `los_sigma` (a near-delta redshift kernel).
The legacy `likelihood="l2"` mode additionally returns `out["los"]` (the
optimised per-galaxy line-of-sight coordinates) and honours `fit_los`.

## Demo

```bash
python examples/demo.py
```

Draws a known field, evolves it into **redshift space**, samples a mock
redshift-space catalogue with photo-z scatter, reconstructs from the scattered
catalogue, and saves `examples/demo_result.png` (redshift-space truth vs.
redshift-space reconstruction vs. real-space reconstruction with RSD removed vs.
loss curve).

**Validation — Poisson vs L2.** The paper's key claim is that the Poisson
likelihood beats the Gaussian/L2 one at low occupancy, where the Gaussian
approximation is no longer valid. On identical redshift-space mocks (32³ mesh,
2LPT), comparing real-space recovery correlation:

| occupancy (gal/cell) | L2 (legacy) | Poisson (default) |
|---------------------:|:-----------:|:-----------------:|
| 0.61 (dense photo-z) | 0.58 | 0.56 |
| **0.09 (sparse spec-z)** | **0.44** | **0.55** |

At high density the two are comparable (L2 can exploit its extra per-galaxy
free coordinates); in the sparse regime the Poisson likelihood is markedly
better (+26 %), reproducing the paper's finding.

**RSD** is correct: enabling it decorrelates the real- and redshift-space fields
by ~9 % (`corr ≈ 0.91`) with line-of-sight shifts of rms ≈ 0.56 cells, i.e. the
Kaiser `f·Ψ` displacement (`f ≈ Ω_m^0.55`). With RSD on, the forward model
recovers the *real-space* field from a redshift-space catalogue (demo panel 3).

## Package layout

```
recon_jax/
  config.py       ReconConfig — all physical / numerical settings
  cosmology.py    jax_cosmo wrapper, P(k), JaxPM growth-cache priming
  forward.py      ForwardModel — linear field → 2LPT/PM → RSD → δ_m
  painting.py     galaxy bias, CIC deposit, Fourier-space Gaussian smoothing
  galaxy.py       GalaxyCatalog + mock-catalogue generation
  loss.py         Poisson likelihood + continuous-Poisson photo-z + field prior
                  (build_data_counts, poisson_nll; legacy L2/xcorr retained)
  reconstruct.py  Reconstructor — optax L-BFGS with annealing
examples/demo.py  end-to-end mock reconstruction
```

## Why is the reconstructed field diffuse?

A Poisson (or any MAP) reconstruction of **photometric** data is expected to look
smooth — the paper's Fig. 1 calls the photometric map "accurate but diffuse".
Two effects, both correct, are at play:

1. **Photo-z blur.** Each galaxy's count is spread along the line of sight by its
   redshift kernel `p(z_k)`, so small-scale line-of-sight structure is genuinely
   not measured. In the directional power spectrum this shows up as the LOS
   high-k power collapsing (ratio ~0.01) while the transverse high-k power
   survives far better (~0.12) — the blur is anisotropic, as it should be.
2. **MAP shrinkage.** The Gaussian field prior pulls modes the data cannot
   constrain toward zero (a Wiener-filter effect), lowering the amplitude of the
   reconstructed field where the signal-to-noise is low.

This is not under-convergence (the loss plateaus) and not a bug: on a
**spectroscopic** catalogue the same code is sharp (amplitude ratio ~0.8, vs
~0.5 for photo-z).

Levers if you want a sharper map:

* **Lower `prior_fac`** (e.g. `1.0 → 0.2`). This reduces small-scale shrinkage
  and roughly doubles the recovered high-k power, at the cost of a noisier field.
* **Denser sampling / smaller `los_sigma`** — more information, sharper result.
* **Combine spec-z with photo-z** in one catalogue: the spectroscopic subset
  pins down the sharp structure while the photometric sample fills in the
  low-density web. This joint reconstruction is the paper's main result.
* For a visually sharper (but less statistically principled) map, the legacy
  `likelihood="l2"` mode with free per-galaxy line-of-sight coordinates places
  galaxies at sharp positions — at the risk of over-fitting under-constrained
  redshifts.

## Notes / gotchas

* **Growth-cache priming.** JaxPM's `lpt` needs the `h`/`h2` keys in the
  `jax_cosmo` growth cache; if a `jax_cosmo` background call populates that cache
  first, `lpt` raises `KeyError: 'h'`. `cosmology.prime_growth_cache` (called
  automatically by `Reconstructor`) fixes this by filling the cache with JaxPM's
  own `growth_factor` before anything else.
* **CIC of galaxies.** The public `jaxpm.painting.cic_paint` assumes exactly
  `nc³` particles; scattered galaxy catalogues are painted with the lower-level
  `_cic_paint_impl`, which accepts an arbitrary count.
* **Poisson smoothing.** Heavy Gaussian annealing flattens the Poisson term and
  starves the gradient (the paper smooths by only 0.05 Mpc/h). Use a light
  `anneal_scales` schedule; the last stage should reach 0.
* The Poisson `φ_i = N̄·(1+δ_g)` stays non-negative for `b₁≈1`; it is clipped at
  a small `eps` before the `log` for safety. A survey selection `W(x)` can be
  folded into `φ_i` when needed.

## References

* Horowitz & Melchior, *Joint Cosmic Density Reconstruction from Photometric and
  Spectroscopic Samples* (2023), https://arxiv.org/abs/2306.15733 — the Poisson
  likelihood and continuous-Poisson photo-z model implemented here.
* Horowitz et al., *TARDIS I* (2019), https://arxiv.org/abs/1903.09049
* Nguyen et al. (2021) — Poisson vs Gaussian likelihoods for density inference.
* JaxPM — https://github.com/DifferentiableUniverseInitiative/JaxPM
* jax_cosmo — https://github.com/DifferentiableUniverseInitiative/jax_cosmo
