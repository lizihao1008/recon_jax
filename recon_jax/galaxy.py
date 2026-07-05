"""Galaxy catalogue container and mock-catalogue generation.

The *only* observational input required by the reconstruction is a
:class:`GalaxyCatalog`: 3-D positions (in grid units) plus a per-galaxy
line-of-sight (redshift) uncertainty.  The third coordinate (axis 2) is treated
as the line of sight, consistent with TARDIS's photo-z model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GalaxyCatalog:
    """Observed galaxies.

    positions : (N_gal, 3) float array, grid units in [0, nc).  Column 2 is the
                (uncertain) line-of-sight / redshift coordinate.
    los_sigma : (N_gal,) float array, 1-sigma uncertainty on the line-of-sight
                coordinate, in grid units.  Small -> spectroscopic, large ->
                photometric.
    """

    positions: np.ndarray
    los_sigma: np.ndarray

    @property
    def num_gal(self) -> int:
        return int(self.positions.shape[0])

    @classmethod
    def from_arrays(cls, xyz, los_sigma):
        xyz = np.asarray(xyz, dtype=np.float32)
        los_sigma = np.broadcast_to(np.asarray(los_sigma, dtype=np.float32), (len(xyz),)).copy()
        return cls(positions=xyz, los_sigma=los_sigma)


def sample_galaxies_from_field(delta_g, n_gal, rng, nc):
    """Poisson/rejection sample ``n_gal`` galaxy positions from an overdensity field.

    Probability of drawing a galaxy in a cell is proportional to ``1 + delta_g``
    (clipped at 0).  Returns positions in grid units with sub-cell jitter.
    """
    prob = np.clip(1.0 + np.asarray(delta_g), 0.0, None).ravel()
    prob = prob / prob.sum()
    idx = rng.choice(prob.size, size=n_gal, p=prob)
    ix, iy, iz = np.unravel_index(idx, (nc, nc, nc))
    jitter = rng.random((n_gal, 3))
    return (np.stack([ix, iy, iz], axis=-1) + jitter).astype(np.float32)


def make_mock_catalog(delta_g, n_gal, los_sigma, rng, nc):
    """Build a mock :class:`GalaxyCatalog` with photo-z-style scatter.

    True positions are sampled from ``delta_g``; the line-of-sight coordinate is
    then perturbed by Gaussian noise of width ``los_sigma`` (grid units) to
    emulate redshift errors.  Both the scattered catalogue and the true
    positions are returned so a demo can check the recovery.
    """
    true_pos = sample_galaxies_from_field(delta_g, n_gal, rng, nc)
    los_sigma = np.broadcast_to(np.asarray(los_sigma, np.float32), (n_gal,)).copy()
    obs_pos = true_pos.copy()
    obs_pos[:, 2] = obs_pos[:, 2] + rng.normal(0.0, los_sigma)
    obs_pos[:, 2] = np.mod(obs_pos[:, 2], nc)  # keep inside the periodic box
    cat = GalaxyCatalog(positions=obs_pos, los_sigma=los_sigma)
    return cat, true_pos
