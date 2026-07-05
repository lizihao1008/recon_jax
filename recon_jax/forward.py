"""JaxPM forward model:  linear density field  ->  evolved matter overdensity.

This is the JAX/JaxPM replacement for TARDIS's ``universe.pm`` (which used
flowPM).  Given a linear (initial) density field it

  1. computes the 2LPT displacement with :func:`jaxpm.pm.lpt`,
  2. optionally integrates the full particle-mesh ODE to ``a_final`` with
     ``diffrax`` (JaxPM's ``make_ode_fn``),
  3. paints the particles back onto the grid with CIC to obtain the evolved
     matter *overdensity* ``delta_m``.

Everything is differentiable w.r.t. the input linear field, which is what makes
the L-BFGS reconstruction possible.
"""
from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jaxpm.painting import cic_paint
from jaxpm.pm import linear_field, lpt, make_ode_fn

from .config import ReconConfig


class ForwardModel:
    """Callable wrapping the JaxPM particle-mesh forward evolution."""

    def __init__(self, config: ReconConfig, cosmo, pk_fn):
        self.cfg = config
        self.cosmo = cosmo
        self.pk_fn = pk_fn
        nc = config.nc
        # Lagrangian (grid) positions of the particles, shape (nc**3, 3).
        self.q = jnp.stack(
            jnp.meshgrid(jnp.arange(nc), jnp.arange(nc), jnp.arange(nc), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3).astype(jnp.float32)
        self._ode = make_ode_fn(config.mesh_shape)

    # -- initial conditions ------------------------------------------------
    def sample_linear_field(self, seed):
        """Draw a Gaussian linear field with the fiducial power spectrum."""
        if isinstance(seed, int):
            seed = jax.random.PRNGKey(seed)
        return linear_field(self.cfg.mesh_shape, self.cfg.box, self.pk_fn, seed=seed)

    # -- evolution ---------------------------------------------------------
    def evolve(self, linear):
        """Evolve to ``a_final`` and return ``(positions, velocities)``.

        Both are ``(nc**3, 3)`` arrays in grid units; ``velocities`` are the
        JaxPM canonical momenta ``p`` (needed for the redshift-space mapping).
        """
        cfg = self.cfg
        dx, p, _ = lpt(self.cosmo, linear, a=cfg.a_init, order=cfg.lpt_order)
        if not cfg.n_body:
            # Evolve straight to a_final with LPT (fast, no ODE).
            dx_f, p_f, _ = lpt(self.cosmo, linear, a=cfg.a_final, order=cfg.lpt_order)
            return self.q + dx_f.reshape(-1, 3), p_f.reshape(-1, 3)

        # Full PM integration a_init -> a_final.
        cosmo = self.cosmo
        ode = self._ode

        def term(a, state, args):
            return ode(state, a, cosmo)

        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(term),
            diffrax.Dopri5(),
            t0=cfg.a_init,
            t1=cfg.a_final,
            dt0=0.05,
            y0=(self.q + dx.reshape(-1, 3), p.reshape(-1, 3)),
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-3),
            max_steps=cfg.pm_steps,
        )
        return sol.ys[0][-1], sol.ys[1][-1]

    def displaced_positions(self, linear):
        """Real-space final particle positions (grid units)."""
        return self.evolve(linear)[0]

    # -- redshift-space distortions ---------------------------------------
    def to_redshift_space(self, positions, velocities):
        """Shift particles into redshift space along the line of sight.

        This is TARDIS's RSD operator expressed in the particle representation.
        In comoving coordinates the redshift-space shift is  Δs = a · dx/da  along
        the line of sight; with JaxPM's units ``dx/da = p / (a**3 E)`` this is

            Δs = p_los / (a**2 E(a))            (evaluated at a = a_final)

        which for the Zel'dovich flow reduces to the familiar Kaiser form
        ``s = x + f·Ψ``.
        """
        cfg = self.cfg
        a = cfg.a_final
        E = jnp.sqrt(jc.background.Esqr(self.cosmo, jnp.atleast_1d(a))[0])
        shift = cfg.rsd_factor * velocities[:, cfg.los_axis] / (a ** 2 * E)
        return positions.at[:, cfg.los_axis].add(shift)

    # -- density field -----------------------------------------------------
    def matter_overdensity(self, linear, redshift_space=None):
        """Evolved matter overdensity ``delta_m`` on the grid (mean 0).

        If ``redshift_space`` is True (default: follow ``config.rsd``) the
        particles are mapped into redshift space before being painted, so the
        resulting field — and hence the biased galaxy field built from it — lives
        in redshift space, matching a redshift-space galaxy catalogue.
        """
        if redshift_space is None:
            redshift_space = self.cfg.rsd
        pos, vel = self.evolve(linear)
        if redshift_space:
            pos = self.to_redshift_space(pos, vel)
        painted = cic_paint(jnp.zeros(self.cfg.mesh_shape), pos)  # mean ~ 1
        return painted - jnp.mean(painted)

    __call__ = matter_overdensity


def make_forward_model(config: ReconConfig, cosmo=None) -> "ForwardModel":
    """Construct a ready-to-use :class:`ForwardModel` from a config.

    Handles the fiducial cosmology, the power-spectrum callable and the one-time
    JaxPM growth-cache priming, so external code (e.g. the LAE_pinn torch bridge)
    can obtain a differentiable ``linear -> density`` operator in one call::

        fm = make_forward_model(cfg)
        delta = fm.matter_overdensity(linear)     # differentiable in `linear`
    """
    from .cosmology import get_cosmo, power_spectrum_fn, prime_growth_cache

    cosmo = get_cosmo() if cosmo is None else cosmo
    prime_growth_cache(cosmo, config.a_init)
    pk_fn = power_spectrum_fn(cosmo)
    return ForwardModel(config, cosmo, pk_fn)
