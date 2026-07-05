"""Painting galaxies onto the grid, biasing, and Fourier-space smoothing.

Ports TARDIS's ``galaxy`` (Eulerian bias), ``smoothing`` (Gaussian filter in
k-space) and the CIC deposit of galaxy positions.
"""
from __future__ import annotations

import jax.numpy as jnp
from jaxpm.kernels import fftk
from jaxpm.painting import _cic_paint_impl


def galaxy_bias(delta_m, b1, b2):
    """Eulerian galaxy overdensity:  delta_g = b1*delta_m + b2**2 * delta_m**2."""
    return b1 * delta_m + (b2 ** 2) * delta_m ** 2


def paint_galaxies(positions, mesh_shape):
    """CIC-deposit galaxies (positions in grid units) and return their overdensity.

    ``positions`` has shape ``(N_gal, 3)`` and is differentiable, so the
    line-of-sight coordinate can be optimised.  Uses the low-level CIC deposit
    which (unlike the public ``cic_paint``) accepts an arbitrary galaxy count.
    """
    counts = _cic_paint_impl(jnp.zeros(mesh_shape), positions)
    mean = jnp.mean(counts)
    return counts / (mean + 1e-8) - 1.0


def kmesh_squared(shape):
    """|k|**2 on the grid in (rad/cell)**2.  ``fftk`` needs a complex array."""
    kvec = fftk(jnp.zeros(shape, dtype=jnp.complex64))
    return sum(kk ** 2 for kk in kvec)


def gaussian_smooth(field, radius_cells):
    """Isotropic Gaussian smoothing in Fourier space.

    ``radius_cells`` is the smoothing scale in grid cells (``fftk`` returns
    wavenumbers in rad/cell).  radius 0 is a no-op.
    """
    ksq = kmesh_squared(field.shape)
    smooth = jnp.exp(-0.5 * ksq * radius_cells ** 2)
    fk = jnp.fft.fftn(field)
    return jnp.real(jnp.fft.ifftn(fk * smooth))
