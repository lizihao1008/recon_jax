#!/usr/bin/env bash
# AMD GPU / ROCm clusters: work around JAX HIP graph bugs (memcpy d2d / hipGraphLaunch).
# Usage: source /path/to/recon_jax/env_rocm.sh
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="$_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RECON_JAX_ROCM=1
# Must be set before Python imports jax (see bootstrap.py).
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer="
