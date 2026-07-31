#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH -t 03:00:00
#SBATCH -A cis260079p
#SBATCH -q low
#SBATCH -o logs/genbig_%j.out
#SBATCH -e logs/genbig_%j.err
#SBATCH -J genbig
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs validation_out
PY=/ocean/projects/cis260079p/ctsai4/miniconda3/envs/deepdgr/bin/python3
export PYTHONUNBUFFERED=1 TMPDIR=${LOCAL:-/tmp}
SD=/ocean/projects/cis260079p/ctsai4/synthdata
# thousands of CERTIFIED-routable bounded-box packs across a WIDE difficulty range
$PY gen_synth_batch.py --run big --n 3000 --workers 32 --omp_threads 1 \
   --certify --certify_iters 250 \
   --grid 48x44,64x56,96x84,128x112 \
   --n_nets 800,1500,3000,6000 \
   --util 0.50,0.60,0.70 \
   --out_root "$SD" --resume > validation_out/gen_big.log 2>&1
rc=$?
echo "gen_big rc=$rc  certified packs=$(ls $SD/big/*.npz 2>/dev/null | wc -l)" | tee validation_out/gen_big_done.txt
