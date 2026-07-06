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

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _configure_jax_rocm() -> None:
    """Apply ROCm/HIP workarounds before jax is imported.

    Set ``RECON_JAX_ROCM=1`` or ``source env_rocm.sh`` on AMD GPU clusters.
    Disables XLA GPU command buffers, which avoids errors such as
    ``HIP_ERROR_InvalidValue: Failed to set memcpy d2d node params``.
    """
    if os.environ.get("RECON_JAX_ROCM", "").lower() not in ("1", "true", "yes"):
        return
    flag = "--xla_gpu_enable_command_buffer="
    xla = os.environ.get("XLA_FLAGS", "")
    if flag not in xla:
        os.environ["XLA_FLAGS"] = f"{xla} {flag}".strip() if xla else flag


def ensure_path() -> Path:
    """Insert project root at the front of ``sys.path`` if missing."""
    root = str(_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _ROOT


# Convenience: ``import bootstrap`` is enough.
_configure_jax_rocm()
ensure_path()
