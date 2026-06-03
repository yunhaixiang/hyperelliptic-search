#!/bin/bash
set -euo pipefail

: "${SLURM_ACCOUNT:?Set SLURM_ACCOUNT before submitting, e.g. export SLURM_ACCOUNT=def-example}"

sbatch --account="$SLURM_ACCOUNT" submit_h2_g24_trillium.slurm
