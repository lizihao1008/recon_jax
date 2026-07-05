"""Top-level reconstruction driver: optax L-BFGS with annealed smoothing.

This is the JAX/optax replacement for TARDIS's ``reconstruct_photoz.run_model``,
which used SciPy's L-BFGS-B through TensorFlow.  The annealing loop (coarse ->
fine smoothing) is preserved: at each stage the smoothing scale is fixed and the
linear field (and, optionally, the galaxy line-of-sight coordinates) are
optimised with L-BFGS.
"""
from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .config import ReconConfig
from .cosmology import get_cosmo, power_spectrum_fn, prime_growth_cache
from .forward import ForwardModel
from .loss import build_loss


class Reconstructor:
    """Reconstruct the density field from a galaxy catalogue."""

    def __init__(self, config: ReconConfig, cosmo=None, likelihood="poisson"):
        self.cfg = config
        self.cosmo = cosmo if cosmo is not None else get_cosmo()
        prime_growth_cache(self.cosmo, config.a_init)
        self.pk_fn = power_spectrum_fn(self.cosmo)
        self.forward = ForwardModel(config, self.cosmo, self.pk_fn)
        self.likelihood = likelihood
        self.history = []

    # -- initial guess -----------------------------------------------------
    def _init_params(self, catalog, seed):
        rng = np.random.default_rng(seed)
        linear = jnp.asarray(
            0.1 * rng.standard_normal(self.cfg.mesh_shape), dtype=jnp.float32
        )
        params = {"linear": linear}
        # legacy L2/xcorr mode also optimises per-galaxy line-of-sight coords;
        # the Poisson mode bakes photo-z into the data field and needs none.
        if self.likelihood != "poisson":
            params["los"] = jnp.asarray(catalog.positions[:, 2], dtype=jnp.float32)
        return params

    # -- one annealing stage ----------------------------------------------
    @staticmethod
    def _run_stage(loss_fn, params, radius, maxiter, fit_los):
        """Run L-BFGS at a fixed smoothing ``radius`` for ``maxiter`` iterations."""

        def objective(p):
            # optionally freeze the line-of-sight coordinates (legacy spec-z mode)
            if not fit_los and "los" in p:
                p = {"linear": p["linear"], "los": jax.lax.stop_gradient(p["los"])}
            return loss_fn(p, radius)

        objective = jax.jit(objective)
        opt = optax.lbfgs()
        value_and_grad = optax.value_and_grad_from_state(objective)

        @jax.jit
        def step(carry):
            p, state = carry
            value, grad = value_and_grad(p, state=state)
            updates, state = opt.update(
                grad, state, p, value=value, grad=grad, value_fn=objective
            )
            p = optax.apply_updates(p, updates)
            return (p, state), value

        state = opt.init(params)
        carry = (params, state)
        values = []
        for _ in range(maxiter):
            carry, value = step(carry)
            values.append(float(value))
        return carry[0], values

    # -- public API --------------------------------------------------------
    def run(self, catalog, seed=None, verbose=True):
        """Reconstruct from ``catalog``.  Returns a result dict."""
        cfg = self.cfg
        seed = cfg.seed if seed is None else seed
        params = self._init_params(catalog, seed)
        loss_fn, _ = build_loss(cfg, self.forward, catalog, self.pk_fn, self.likelihood)

        t0 = time.time()
        for stage, (radius, maxiter) in enumerate(zip(cfg.anneal_scales, cfg.maxiter)):
            params, values = self._run_stage(
                loss_fn, params, float(radius), int(maxiter), cfg.fit_los
            )
            self.history.extend(values)
            if verbose:
                print(
                    f"[stage {stage}] R={radius:>4} cells  "
                    f"loss {values[0]:.4e} -> {values[-1]:.4e}  "
                    f"({time.time() - t0:.1f}s)"
                )

        linear = params["linear"]
        delta_m = self.forward.matter_overdensity(linear)
        result = {
            "linear": np.asarray(linear),
            "delta_m": np.asarray(delta_m),
            "loss_history": self.history,
        }
        if "los" in params:
            result["los"] = np.asarray(params["los"])
        return result
