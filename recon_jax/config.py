"""Configuration objects for recon_jax.

A single :class:`ReconConfig` holds every physical / numerical choice used by the
forward model, the loss and the optimizer.  It is a plain frozen dataclass so it
is trivially hashable and can be closed over inside ``jax.jit``-ed functions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class ReconConfig:
    """Everything that defines a reconstruction problem.

    Grid / geometry
    ---------------
    nc         : particle-mesh side length (the cube has ``nc**3`` cells).
    box_size   : physical box size in Mpc/h (cubic box assumed).

    Time stepping  (specified as REDSHIFTS)
    -------------
    z_init     : redshift at which the LPT displacement is evaluated
                 (start of the N-body integration; should be well before z_final).
    z_final    : redshift of the observed snapshot (the epoch of the galaxies).
    n_body     : if True integrate the full PM ODE from z_init to z_final,
                 otherwise evolve straight to z_final with 2LPT (much faster).
    pm_steps   : max adaptive steps for the diffrax N-body solver.
    lpt_order  : 1 or 2 (2 = 2LPT, recommended).

    The scale factors used internally are exposed as the derived properties
    ``a_init`` and ``a_final`` (``a = 1 / (1 + z)``).

    Galaxy bias  (Eulerian):  delta_g = b1 * delta_m + b2**2 * delta_m**2
    -----------
    b1, b2     : linear and quadratic bias coefficients.

    Loss weights
    ------------
    galaxy_fac : weight of the galaxy-clustering likelihood term.
    prior_fac  : weight of the Gaussian (power-spectrum) prior on the
                 linear field.  Set to 0 to disable.
    redshift_fac : weight of the per-galaxy redshift-error prior.

    Optimiser (optax L-BFGS with annealing)
    ---------
    anneal_scales : Gaussian smoothing scales (in *grid cells*) applied to the
                    galaxy fields, largest first.  Reconstruction proceeds from
                    coarse to fine.
    maxiter       : L-BFGS iterations per annealing stage (one int per scale).
    fit_los       : if True the line-of-sight coordinate of every galaxy is a
                    free parameter (TARDIS photo-z behaviour).  For pure spec-z
                    data set it False (or leave True with tiny z_err).
    """

    # --- grid / geometry -------------------------------------------------
    nc: int = 32
    box_size: float = 128.0

    # --- time stepping (specified as REDSHIFTS) --------------------------
    z_init: float = 20     # z of the LPT/initial epoch  (a_init = 1/(1+z_init))
    z_final: float = 0.0    # z of the observed snapshot  (a_final = 1/(1+z_final))
    n_body: bool = False
    pm_steps: int = 500
    lpt_order: int = 2

    # --- redshift-space distortions (RSD) --------------------------------
    rsd: bool = True          # place the galaxy field in redshift space
    los_axis: int = 2         # line-of-sight grid axis (0/1/2); 2 = z
    rsd_factor: float = 1.0   # overall scaling of the RSD shift (1 = physical)

    # --- galaxy bias -----------------------------------------------------
    b1: float = 1.0
    b2: float = 0.0

    # --- loss weights ----------------------------------------------------
    galaxy_fac: float = 1.0
    prior_fac: float = 1.0
    redshift_fac: float = 1.0

    # --- optimiser -------------------------------------------------------
    anneal_scales: Tuple[float, ...] = (4.0, 2.0, 1.0, 0.0)
    maxiter: Tuple[int, ...] = (40, 40, 40, 60)
    fit_los: bool = True

    # --- misc ------------------------------------------------------------
    seed: int = 0

    @property
    def mesh_shape(self) -> Tuple[int, int, int]:
        return (self.nc, self.nc, self.nc)

    @property
    def box(self) -> Tuple[float, float, float]:
        return (self.box_size, self.box_size, self.box_size)

    @property
    def cell_size(self) -> float:
        """Physical size of one grid cell in Mpc/h."""
        return self.box_size / self.nc

    @property
    def a_init(self) -> float:
        """Scale factor of the initial/LPT epoch, ``1 / (1 + z_init)``."""
        return 1.0 / (1.0 + self.z_init)

    @property
    def a_final(self) -> float:
        """Scale factor of the observed snapshot, ``1 / (1 + z_final)``."""
        return 1.0 / (1.0 + self.z_final)

    def __post_init__(self):
        if len(self.anneal_scales) != len(self.maxiter):
            raise ValueError("anneal_scales and maxiter must have equal length")
        if self.z_init <= self.z_final:
            raise ValueError("z_init must be greater than z_final (evolve forward in time)")
