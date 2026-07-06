"""Make ``recon_jax`` importable without ``pip install``.

Add the project root (this file's directory) to ``sys.path`` so that
``import recon_jax`` resolves to the local ``recon_jax/`` package.

Typical use in a script or notebook (run once before other imports)::

    import bootstrap
    from recon_jax import ReconConfig, Reconstructor

Alternatively, set the environment variable once per shell / Jupyter kernel::

    export PYTHONPATH="/path/to/field_recon/recon_jax:$PYTHONPATH"

Only third-party dependencies (jax, jaxpm, …) need to be installed; this
repository itself is not installed as a package.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def ensure_path() -> Path:
    """Insert project root at the front of ``sys.path`` if missing."""
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _ROOT


# Convenience: ``import bootstrap`` is enough.
ensure_path()
