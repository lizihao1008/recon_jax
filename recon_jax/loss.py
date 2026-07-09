"""The reconstruction objective (negative log-posterior).

The default likelihood follows **Horowitz & Melchior (2023)**, *Joint Cosmic
Density Reconstruction from Photometric and Spectroscopic Samples*, which
improves on the earlier L2 galaxy likelihood of TARDIS (Horowitz et al. 2021):

* **Poisson point-process likelihood** (their Eq. 4).  The observed galaxies are
  modelled as a draw from an inhomogeneous Poisson process whose rate density
  ``phi_i`` is the forward-modelled expected galaxy count in cell ``i``:

      log P(d | s) = sum_i [ n_i log phi_i - phi_i ]

  This is more accurate for small-scale power and, unlike the Gaussian/L2
  approximation, remains valid at low occupancy (<~5 galaxies per cell).

* **Differentiable continuous Poisson process for photo-z** (their Eqs. 7-8).
  Rather than giving every galaxy a *free* line-of-sight coordinate (the 2021
  approach), each galaxy's count is redistributed over cells along the line of
  sight according to its redshift PDF ``p(z_k)`` -- a Gaussian of width
  ``sigma_z``.  The resulting counts ``n'_i`` are fractional, so the Poisson
  distribution is generalised to continuous counts via the incomplete Gamma
  function; the ``Gamma(n'_i)`` normalisation is independent of ``s`` and drops
  out of the optimisation, leaving ``sum_i [n'_i log phi_i - phi_i]``.  The
  photo-z uncertainty is thus baked into the (fixed) data field and no
  per-galaxy parameters are needed.

* **Field prior** (their Eq. 10) -- the Gaussian prior on the initial field,
  ``1/2 s^T S^-1 s = sum_k |delta_lin(k)|**2 / P(k)``.

The earlier L2 / cross-correlation galaxy likelihood is retained as an option
(``mode='l2'`` / ``'xcorr'``) for comparison.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxpm.kernels import fftk

from .painting import galaxy_bias, gaussian_smooth, paint_galaxies


# ----------------------------------------------------------------------------
# Field prior  (Horowitz & Melchior 2023, Eq. 10)
# ----------------------------------------------------------------------------
def _pk_mesh(mesh_shape, box, pk_fn):
    """Fiducial P(k) on the 3-D grid, JaxPM ``linear_field`` convention.

    Supports rectangular meshes: per-axis wavenumbers ``kk_i / L_i * n_i`` and
    the amplitude normalisation ``(nx*ny*nz)/(Lx*Ly*Lz)``.
    """
    dummy = jnp.zeros(tuple(mesh_shape), dtype=jnp.complex64)
    kvec = fftk(dummy)
    kmesh = sum((kk / box[i] * mesh_shape[i]) ** 2 for i, kk in enumerate(kvec)) ** 0.5
    n_cells = mesh_shape[0] * mesh_shape[1] * mesh_shape[2]
    volume = box[0] * box[1] * box[2]
    pkmesh = pk_fn(kmesh) * n_cells / volume
    # avoid division by zero at the k=0 (DC) mode
    return pkmesh.at[0, 0, 0].set(1.0)


def field_prior(linear, pkmesh, n_cells):
    """Gaussian prior  sum_k |delta_lin(k)|**2 / P(k)  (normalised per mode)."""
    dk = jnp.fft.fftn(linear)
    power = jnp.abs(dk) ** 2
    chi2 = jnp.sum(power / pkmesh)
    return chi2 / n_cells


# ----------------------------------------------------------------------------
# Continuous Poisson process for photo-z  (Horowitz & Melchior 2023, Eqs. 7-8)
# ----------------------------------------------------------------------------
def build_data_counts(catalog, mesh_shape, los_axis=2):
    """Build the fractional data counts ``n'_i`` for the continuous Poisson process.

    Each galaxy is deposited with bilinear (CIC) weights on the two transverse
    axes and spread along the line of sight by a Gaussian of width its own
    ``los_sigma`` (grid cells), i.e. its redshift PDF ``p(z_k)``.  The result is
    a fixed, differentiable-free data field with ``sum(n') = N_gal``.

    This is the numpy pre-computation of Eq. 7; for uniform ``sigma_z`` it is the
    1-D radial convolution mentioned in the paper, but the per-galaxy form here
    also supports a different uncertainty for every galaxy.
    """
    mesh_shape = tuple(int(x) for x in mesh_shape)
    pos = np.asarray(catalog.positions, dtype=np.float64)
    sig = np.asarray(catalog.los_sigma, dtype=np.float64)
    sig = np.clip(sig, 1e-3, None)  # a spec-z galaxy is a near-delta kernel

    axes = [0, 1, 2]
    axes.remove(los_axis)
    a0, a1 = axes
    n0, n1, n_los = mesh_shape[a0], mesh_shape[a1], mesh_shape[los_axis]

    # transverse CIC (bilinear) weights on the two non-LOS axes
    t0, t1 = pos[:, a0], pos[:, a1]
    f0 = np.floor(t0).astype(int)
    f1 = np.floor(t1).astype(int)
    d0, d1 = t0 - f0, t1 - f1

    # line-of-sight Gaussian kernel over the LOS-axis cells, periodic
    grid = np.arange(n_los)
    diff = grid[None, :] - pos[:, los_axis][:, None]
    diff = (diff + n_los / 2.0) % n_los - n_los / 2.0
    los_w = np.exp(-0.5 * (diff / sig[:, None]) ** 2)
    los_w /= los_w.sum(axis=1, keepdims=True)  # normalise each galaxy to unit count

    # accumulate into a [a0, a1, los] work array, then move axes back
    work = np.zeros((n0, n1, n_los), dtype=np.float64)
    work2 = work.reshape(n0 * n1, n_los)
    for c0, w0 in ((f0, 1.0 - d0), (f0 + 1, d0)):
        for c1, w1 in ((f1, 1.0 - d1), (f1 + 1, d1)):
            idx = (c0 % n0) * n1 + (c1 % n1)          # flattened transverse cell
            contrib = (w0 * w1)[:, None] * los_w      # (K, n_los)
            np.add.at(work2, idx, contrib)

    counts = np.moveaxis(work, [0, 1, 2], [a0, a1, los_axis])
    return counts.astype(np.float32)


def poisson_nll(rate, counts, radius, eps=1e-6):
    """Negative Poisson log-likelihood  sum_i [phi_i - n'_i log phi_i].

    Both the model rate and the data counts are Gaussian-smoothed at the current
    annealing ``radius`` (coarse -> fine); smoothing preserves non-negativity and
    total counts, so the Poisson form stays well defined and reduces to the exact
    likelihood as ``radius -> 0``.
    """
    r = gaussian_smooth(rate, radius)
    n = gaussian_smooth(counts, radius)
    r = jnp.clip(r, eps, None)
    return jnp.sum(r - n * jnp.log(r))


# ----------------------------------------------------------------------------
# L2 / cross-correlation galaxy likelihood (legacy TARDIS-2021 option)
# ----------------------------------------------------------------------------
def redshift_prior(los, los_obs, los_sigma):
    """sum ((los - los_obs) / sigma)**2 -- per-galaxy redshift-error constraint."""
    return jnp.sum(((los - los_obs) / los_sigma) ** 2)


def galaxy_likelihood(model_g, data_g, radius, weight=None, mode="l2"):
    """Compare model and data galaxy overdensity at smoothing scale ``radius``."""
    m = gaussian_smooth(model_g, radius)
    d = gaussian_smooth(data_g, radius)
    if mode == "xcorr":
        return -jnp.sum(m * d)
    resid = m - d
    if weight is None:
        weight = 1.0
    return jnp.sum(weight * resid ** 2)


# ----------------------------------------------------------------------------
# Objective assembly
# ----------------------------------------------------------------------------
def build_loss(config, forward_model, catalog, pk_fn, mode="poisson"):
    """Return a jittable ``loss(params, radius)`` closure.

    ``mode='poisson'`` (default, Horowitz & Melchior 2023): ``params`` is
    ``{"linear": (nc,nc,nc)}``; photo-z is handled by the fixed continuous-Poisson
    data field, so there are no per-galaxy parameters.

    ``mode='l2'`` / ``'xcorr'`` (legacy): ``params`` is
    ``{"linear": ..., "los": (N_gal,)}`` and the redshift errors enter through a
    per-galaxy prior on the free line-of-sight coordinates.
    """
    mesh_shape = config.mesh_shape
    n_cells = mesh_shape[0] * mesh_shape[1] * mesh_shape[2]
    pkmesh = _pk_mesh(mesh_shape, config.box, pk_fn)

    if mode == "poisson":
        counts = jnp.asarray(build_data_counts(catalog, mesh_shape, config.los_axis))
        mean_count = counts.sum() / n_cells  # expected counts per cell

        def loss(params, radius):
            linear = params["linear"]
            delta_m = forward_model.matter_overdensity(linear)
            delta_g = galaxy_bias(delta_m, config.b1, config.b2)
            rate = mean_count * (1.0 + delta_g)  # phi_i, expected counts per cell
            loss_val = config.galaxy_fac * poisson_nll(rate, counts, radius)
            if config.prior_fac != 0.0:
                loss_val += config.prior_fac * field_prior(linear, pkmesh, n_cells)
            return loss_val

        return loss, {"counts": counts, "mean_count": mean_count}

    # ---- legacy L2 / xcorr path ----
    xy = jnp.asarray(catalog.positions[:, :2])
    los_obs = jnp.asarray(catalog.positions[:, 2])
    los_sigma = jnp.asarray(catalog.los_sigma)

    def loss(params, radius):
        linear = params["linear"]
        los = params["los"]
        delta_m = forward_model.matter_overdensity(linear)
        model_g = galaxy_bias(delta_m, config.b1, config.b2)
        positions = jnp.concatenate([xy, los[:, None]], axis=1)
        data_g = paint_galaxies(positions, mesh_shape)
        like = galaxy_likelihood(model_g, data_g, radius, mode=mode)
        loss_val = config.galaxy_fac * like
        loss_val += config.redshift_fac * redshift_prior(los, los_obs, los_sigma)
        if config.prior_fac != 0.0:
            loss_val += config.prior_fac * field_prior(linear, pkmesh, n_cells)
        return loss_val

    return loss, {"los_obs": los_obs, "los_sigma": los_sigma, "xy": xy}
