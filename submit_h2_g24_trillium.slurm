#!/bin/bash
#SBATCH --job-name=h2-g24
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/scratch/yunhai/slurm-%x-%j.out
#SBATCH --error=/scratch/yunhai/slurm-%x-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load python/3.11.5
module load apptainer/1.4.5

source ~/venvs/hyperelliptic-search/bin/activate

mkdir -p /scratch/yunhai
mkdir -p /scratch/yunhai/hyperelliptic-checkpoints

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export DOT_SAGE="$SLURM_TMPDIR/sage-dot-cache"
export SAGE_DOT_DIR="$DOT_SAGE"
mkdir -p "$SAGE_DOT_DIR"

export SAGE_PYTHON_CMD="apptainer exec $HOME/containers/sagemath_latest.sif sage -python"

echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
echo "Submit dir: $SLURM_SUBMIT_DIR"
nvidia-smi

python - <<'PY'
import os
import subprocess
import torch

print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
subprocess.run(os.environ["SAGE_PYTHON_CMD"].split() + ["-c", "import sage.all; print('sage ok')"], check=True)
PY

python train.py \
  --env_name hyperelliptic2 \
  --exp_name h2_p3_pgl2_highgenus_g24_frontier_slurm \
  --dump_path /scratch/yunhai/hyperelliptic-checkpoints \
  --N 24 \
  --p 3 \
  --max_len 2504 \
  --gensize 0 \
  --pop_size 6000 \
  --ntest 500 \
  --initial_data_sqlite training_data/hyperelliptic2_pgl2_highgenus_g100/train_data.pkl \
  --initial_data_max_rows 0 \
  --keep_only_unique true \
  --always_search true \
  --redeem_only false \
  --local_search_steps 2 \
  --local_search_batch_size 32 \
  --local_search_high_genus_bias true \
  --local_search_grow_prob 0.8 \
  --local_search_remove_prob 0.0 \
  --genus_sampling frontier \
  --frontier_top_width 2 \
  --frontier_high_width 6 \
  --frontier_low_cutoff 4 \
  --frontier_low_prob 0.10 \
  --frontier_mid_prob 0.45 \
  --frontier_high_prob 0.35 \
  --frontier_top_prob 0.10 \
  --frontier_max_repeat 20 \
  --score_batch_size 16 \
  --process_pool false \
  --batch_size 8 \
  --eval_batch_size 4 \
  --gen_batch_size 64 \
  --max_epochs 80 \
  --max_steps 1200 \
  --num_eval_steps 200 \
  --num_samples_from_model 2000 \
  --n_layer 3 \
  --n_head 4 \
  --n_embd 192 \
  --temperature 1.0 \
  --temp_span 4 \
  --inc_temp 0.03 \
  --top_k -1
