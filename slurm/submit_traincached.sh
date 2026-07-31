#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH -t 03:00:00
#SBATCH -A cis260079p
#SBATCH -q low
#SBATCH -o logs/tcached_%j.out
#SBATCH -e logs/tcached_%j.err
#SBATCH -J tcached
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs RESULTS
PY=/ocean/projects/cis260079p/ctsai4/miniconda3/envs/deepdgr/bin/python3
export PYTHONUNBUFFERED=1 TMPDIR=${LOCAL:-/tmp} OMP_NUM_THREADS=12
$PY train_cached.py --packs '/ocean/projects/cis260079p/ctsai4/synthdata/cong10k/*.npz' \
   --sizes 50,200,800,3200 --steps 600 --cache 800 --device -1 \
   > validation_out/train_cached.log 2>&1
echo "rc=$?"; tail -12 validation_out/train_cached.log
