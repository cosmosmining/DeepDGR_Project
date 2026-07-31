#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -t 08:00:00
#SBATCH -A cis260079p
#SBATCH -q low
#SBATCH -o logs/gencong_%j.out
#SBATCH -e logs/gencong_%j.err
#SBATCH -J gencong
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs validation_out
PY=/ocean/projects/cis260079p/ctsai4/miniconda3/envs/deepdgr/bin/python3
export PYTHONUNBUFFERED=1 TMPDIR=${LOCAL:-/tmp}
SD=/ocean/projects/cis260079p/ctsai4/synthdata
BASE=$1; N=$2
# CONGESTED (no --certify => keeps overflow>0 instances) + wide difficulty
$PY gen_synth_batch.py --run cong10k --n $N --base_seed $BASE --workers 32 \
   --omp_threads 1 --grid 48x44,64x56,96x84,128x112 \
   --n_nets 1200,2400,4800,9000 --util 0.85,0.95,1.10 \
   --keep_uncertified --out_root "$SD" --resume > validation_out/gencong_${BASE}.log 2>&1
echo "chunk $BASE rc=$? packs=$(ls $SD/cong10k/*.npz 2>/dev/null | wc -l)"
