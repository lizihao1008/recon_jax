"""End-to-end mock demo for recon_jax.

1. Draw a known Gaussian linear field and evolve it with the JaxPM forward model.
2. Sample a mock galaxy catalogue from the evolved field and add photo-z scatter
   to the line-of-sight coordinate.
3. Reconstruct the density field from the *scattered* catalogue alone.
4. Report the correlation between the reconstruction and the truth, and save a
   diagnostic figure.

Run:  python examples/demo.py
"""
import os
import sys
from pathlib import Path

# Project root on sys.path (same effect as PYTHONPATH or `import bootstrap` in notebooks)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from recon_jax import ReconConfig, Reconstructor, make_mock_catalog
from recon_jax.painting import galaxy_bias


def corr(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def main():
    cfg = ReconConfig(
        nc=32,
        box_size=160.0,
        n_body=True,            # 2LPT forward model (fast); set True for full PM
        rsd=True,                # galaxy field lives in redshift space
        los_axis=2,
        b1=1.0,
        b2=0.0,
        galaxy_fac=1.0,
        prior_fac=0.3,           # lighter prior -> less small-scale shrinkage
        anneal_scales=(2.0, 1.0, 0.0),
        maxiter=(25, 25, 50),
        seed=0,
        z_init=19.0,             # initial/LPT epoch  (a = 0.05)
        z_final=7.0,             # observed snapshot redshift
    )

    # Poisson point-process likelihood with the continuous-Poisson photo-z model
    # (Horowitz & Melchior 2023) is the default.
    rec = Reconstructor(cfg, likelihood="poisson")
    rng = np.random.default_rng(42)

    # --- 1. ground-truth fields (real space AND redshift space) ---------
    truth_linear = rec.forward.sample_linear_field(cfg.seed)
    truth_real = np.asarray(
        rec.forward.matter_overdensity(truth_linear, redshift_space=False))
    truth_rsd = np.asarray(
        rec.forward.matter_overdensity(truth_linear, redshift_space=True))
    # the observed galaxy field is built in REDSHIFT space
    truth_g_rsd = np.asarray(galaxy_bias(truth_rsd, cfg.b1, cfg.b2))

    # --- 2. mock redshift-space catalogue with photo-z scatter ----------
    n_gal = 8000
    los_sigma = 0.1  # grid cells of redshift error (photo-z-like)
    catalog, true_pos = make_mock_catalog(truth_g_rsd, n_gal, los_sigma, rng, cfg.mesh_shape)
    nx, ny, nz = cfg.mesh_shape
    occ = n_gal / (nx * ny * nz)
    print(f"mock: {catalog.num_gal} redshift-space galaxies, "
          f"los_sigma = {los_sigma} cells, occupancy = {occ:.2f} gal/cell")

    # --- 3. reconstruction ----------------------------------------------
    out = rec.run(catalog, seed=1)

    # --- 4. diagnostics -------------------------------------------------
    # redshift-space match (directly constrained by the data)
    c_rsd = corr(out["delta_m"], truth_rsd)
    # real-space match: RSD removed by the forward model (the science goal)
    recon_real = np.asarray(
        rec.forward.matter_overdensity(out["linear"], redshift_space=False))
    c_real = corr(recon_real, truth_real)
    print(f"\nredshift-space field vs truth:  correlation = {c_rsd:.3f}")
    print(f"real-space field vs truth (RSD removed): correlation = {c_real:.3f}")
    c = c_rsd
    truth_delta = truth_rsd

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sl = cfg.mesh_shape[0] // 2
        fig, ax = plt.subplots(1, 4, figsize=(17, 4))
        vmin, vmax = np.percentile(truth_rsd[sl], [2, 98])
        ax[0].imshow(truth_rsd[sl], vmin=vmin, vmax=vmax, cmap="magma")
        ax[0].set_title("truth (redshift space)")
        ax[1].imshow(out["delta_m"][sl], vmin=vmin, vmax=vmax, cmap="magma")
        ax[1].set_title(f"recon, z-space (r={c_rsd:.2f})")
        ax[2].imshow(recon_real[sl], vmin=vmin, vmax=vmax, cmap="magma")
        ax[2].set_title(f"recon, real space (r={c_real:.2f})")
        ax[3].plot(out["loss_history"])
        ax[3].set_yscale("log")
        ax[3].set_title("loss")
        ax[3].set_xlabel("L-BFGS iteration")
        for a in ax[:3]:
            a.set_xticks([]); a.set_yticks([])
        fig.tight_layout()
        outpng = os.path.join(os.path.dirname(__file__), "demo_result.png")
        fig.savefig(outpng, dpi=110)
        print(f"figure saved to {outpng}")
    except Exception as e:  # pragma: no cover
        print(f"(plotting skipped: {e})")


if __name__ == "__main__":
    main()
