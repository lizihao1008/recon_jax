"""recon_jax -- JAX/JaxPM density-field reconstruction from galaxy surveys.

A port of the galaxy (spec-z / photo-z) reconstruction idea from TARDIS
(Tomographic Absorption Reconstruction and Density Inference Scheme) to the
JAX ecosystem: JaxPM for the differentiable particle-mesh forward model and
optax's L-BFGS for the optimisation.  The Lyman-alpha forest machinery of TARDIS
is intentionally omitted.

Typical usage
-------------
    from recon_jax import ReconConfig, Reconstructor, GalaxyCatalog

    cfg = ReconConfig(nc=32, box_size=128.0)
    cat = GalaxyCatalog.from_arrays(xyz_grid_units, los_sigma_cells)
    rec = Reconstructor(cfg)
    out = rec.run(cat)
    delta = out["delta_m"]      # reconstructed evolved density field
"""
from .config import ReconConfig
from .galaxy import GalaxyCatalog, make_mock_catalog, sample_galaxies_from_field
from .forward import ForwardModel
from .reconstruct import Reconstructor

__all__ = [
    "ReconConfig",
    "GalaxyCatalog",
    "ForwardModel",
    "Reconstructor",
    "make_mock_catalog",
    "sample_galaxies_from_field",
]

__version__ = "0.1.0"
