#!/usr/bin/env bash
# Source this file to use recon_jax without pip install:
#   source /path/to/field_recon/recon_jax/env.sh
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export PYTHONPATH="$_ROOT${PYTHONPATH:+:$PYTHONPATH}"
