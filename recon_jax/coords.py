"""Sky coordinates -> transverse comoving -> rotated grid units."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from astropy.constants import c as c_light
from astropy.cosmology import Planck18


@dataclass
class SkyToGridResult:
    """Sky positions after comoving projection, rotation, and grid scaling."""

    x_com: np.ndarray
    y_com: np.ndarray
    z_com: np.ndarray
    x_rot: np.ndarray
    y_rot: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    z_grid: np.ndarray
    ra_ref: float
    dec_ref: float
    z_ref: float
    xy_buffer: float
    box_size: tuple[float, float, float]
    z_err_com: np.ndarray | None = None
    z_err_grid: np.ndarray | None = None


def _box_tuple(box_size, z_span: float | None = None) -> tuple[float, float, float]:
    """Normalize ``box_size`` to ``(Lx, Ly, Lz)`` in comoving Mpc."""
    if isinstance(box_size, (int, float)):
        bx = by = bz = float(box_size)
    else:
        parts = tuple(box_size)
        if len(parts) != 3:
            raise ValueError("box_size must be a float or a 3-tuple (Lx, Ly, Lz)")
        bx, by, bz = float(parts[0]), float(parts[1]), parts[2]
        if bz is None:
            if z_span is None:
                raise ValueError("box_size[2] is None but the z_com span is unknown")
            bz = float(z_span)
        else:
            bz = float(bz)
    if bx <= 0.0 or by <= 0.0 or bz <= 0.0:
        raise ValueError("box_size components must be positive")
    return bx, by, bz


def z_err_to_comoving_mpc(z, z_err, cosmo=Planck18):
    """Convert redshift uncertainty to comoving LOS distance uncertainty.

    Uses ``dchi/dz = c / H(z)`` so ``sigma_chi = sigma_z * c / H(z)``.
    """
    z = np.asarray(z, dtype=np.float64)
    z_err = np.asarray(z_err, dtype=np.float64)
    dchi_dz = (c_light / cosmo.H(z)).to_value("Mpc") * cosmo.h
    return z_err * dchi_dz


def sky_to_transverse_grid(
    ra,
    dec,
    z,
    *,
    z_err=None,
    cosmo=Planck18,
    ra_ref=None,
    dec_ref=None,
    z_ref=None,
    rotate_deg=0.0,
    xy_buffer=0.0,
    box_size: float | Sequence[float | None] = (100.0, 100.0, None),
    nc=64,
):
    """Map (RA, DEC, z) to comoving and grid coordinates for reconstruction.

    Pipeline
    --------
    1. Flat-sky projection around ``(ra_ref, dec_ref)`` using each source's
       comoving transverse distance ``D_M(z)``.
    2. Line-of-sight coordinate from ``comoving_distance(z)``.
    3. Rotate the transverse plane by ``rotate_deg`` (counter-clockwise).
    4. Shift each axis so its minimum is 0, then add a comoving buffer on x/y.
    5. Scale comoving Mpc to grid units ``[0, nc)`` with ``box_size``.

    The transverse buffer insets the catalogue from the box edges on the low-x
    and low-y sides.  Choose ``box_size`` large enough to hold the field span
    plus ``xy_buffer`` on the high side as well (i.e. at least
    ``field_span + 2 * xy_buffer`` for symmetric padding).

    Parameters
    ----------
    ra, dec, z : array-like
        Right ascension and declination in degrees, redshift (e.g. spec-z).
    z_err : array-like, optional
        Redshift uncertainty (same units as ``z``).  Converted with
        ``sigma_chi = sigma_z * c / H(z)`` and also scaled to ``z_err_grid``.
    cosmo : astropy.cosmology object, optional
        Cosmology for distance conversions.
    ra_ref, dec_ref : float, optional
        Projection centre in degrees.  Default: sample medians.
    z_ref : float, optional
        Reference redshift recorded in the result (default: median of ``z``).
        The LOS coordinate uses ``comoving_distance(z)`` for each source, then
        is shifted by its minimum over the sample.
    rotate_deg : float
        Counter-clockwise rotation angle in degrees.
    xy_buffer : float
        Comoving Mpc padding added to x and y after the min-shift.
    box_size : float or (Lx, Ly, Lz)
        Physical box side lengths in comoving Mpc.  A scalar gives a cubic box.
        If ``Lz`` is ``None``, it defaults to the shifted ``z_com`` span.
    nc : int or (nx, ny, nz)
        Number of grid cells; a single int (cubic) or a per-axis 3-tuple.

    Returns
    -------
    SkyToGridResult
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if ra_ref is None:
        ra_ref = float(np.median(ra))
    if dec_ref is None:
        dec_ref = float(np.median(dec))
    if z_ref is None:
        z_ref = float(np.median(z))

    dra = np.deg2rad(ra - ra_ref) * np.cos(np.deg2rad(dec_ref))
    ddec = np.deg2rad(dec - dec_ref)

    dm = cosmo.comoving_transverse_distance(z).to_value("Mpc") * cosmo.h
    x_com = dm * dra
    y_com = dm * ddec

    z_com = cosmo.comoving_distance(z).to_value("Mpc") * cosmo.h

    theta = np.deg2rad(rotate_deg)
    ct, st = np.cos(theta), np.sin(theta)
    x_rot = ct * x_com - st * y_com
    y_rot = st * x_com + ct * y_com

    if xy_buffer < 0.0:
        raise ValueError("xy_buffer must be non-negative")

    x_rot = x_rot - x_rot.min() + xy_buffer
    y_rot = y_rot - y_rot.min() + xy_buffer
    z_com = z_com - z_com.min()

    box_x, box_y, box_z = _box_tuple(box_size, z_span=float(z_com.max()))

    nx, ny, nz = (int(nc),) * 3 if isinstance(nc, int) else tuple(int(v) for v in nc)
    x_grid = x_rot / box_x * nx
    y_grid = y_rot / box_y * ny
    z_grid = z_com / box_z * nz

    z_err_com = None
    z_err_grid = None
    if z_err is not None:
        z_err_com = z_err_to_comoving_mpc(z, z_err, cosmo=cosmo)
        z_err_grid = z_err_com / box_z * nz

    return SkyToGridResult(
        x_com=x_com,
        y_com=y_com,
        z_com=z_com,
        x_rot=x_rot,
        y_rot=y_rot,
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        ra_ref=ra_ref,
        dec_ref=dec_ref,
        z_ref=z_ref,
        xy_buffer=float(xy_buffer),
        box_size=(box_x, box_y, box_z),
        z_err_com=z_err_com,
        z_err_grid=z_err_grid,
    )
