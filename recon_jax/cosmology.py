"""Cosmology helpers.

Wraps ``jax_cosmo`` and provides:
  * a fiducial cosmology,
  * a JAX-differentiable linear matter power-spectrum callable ``pk(k)`` used
    both to draw initial conditions and to build the Gaussian field prior,
  * a one-time *priming* of the JaxPM growth cache.

The priming step is important.  JaxPM's ``dGfa`` reads the keys ``h``/``h2``
from ``cosmo._workspace['background.growth_factor']``.  If any ``jax_cosmo``
background routine (e.g. ``Esqr`` inside ``lpt``) populates that cache first it
does so *without* those keys, and ``lpt`` then raises ``KeyError: 'h'``.  Calling
JaxPM's own ``growth_factor`` once, before anything else, fills the cache with
the full key set and the ``if not in keys`` guards keep it that way.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo.scipy.interpolate import interp
from jaxpm.growth import growth_factor


def get_cosmo():
    """Return the fiducial cosmology (Planck 2015)."""
    return jc.Planck15()


def prime_growth_cache(cosmo, a_init: float = 0.1) -> None:
    """Populate the JaxPM growth cache with the full key set (see module docstring)."""
    if "background.growth_factor" not in cosmo._workspace:
        growth_factor(cosmo, jnp.atleast_1d(a_init))


def power_spectrum_fn(cosmo, n_k: int = 256, k_min: float = 1e-4, k_max: float = 1e1):
    """Return a differentiable ``pk(k)`` for the *linear* matter power spectrum.

    The spectrum is tabulated once on a log-k grid and linearly interpolated so
    that it can be evaluated on the 3-D k-mesh inside jitted code.
    """
    k_tab = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), n_k)
    pk_tab = jc.power.linear_matter_power(cosmo, k_tab)

    def pk(k):
        shape = k.shape
        return interp(k.reshape(-1), k_tab, pk_tab).reshape(shape)

    return pk
