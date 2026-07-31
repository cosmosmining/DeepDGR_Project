#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gpus=v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH -t 04:00:00
#SBATCH -A cis260079p
#SBATCH -o logs/final_%j.out
#SBATCH -e logs/final_%j.err
#SBATCH -J final
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs RESULTS
module load cuda/12.1.1
export PATH=/ocean/projects/cis260079p/ctsai4/miniconda3/envs/deepdgr/bin:$PATH
export PYTHONUNBUFFERED=1
/ocean/projects/cis260079p/ctsai4/miniconda3/envs/deepdgr/bin/python3 \
  optimize_final.py --device 0 > validation_out/final_opt.log 2>&1
echo "rc=$?"; tail -20 validation_out/final_opt.log
