#!/bin/bash
#SBATCH -p RM-shared
#SBATCH -N 1
#SBATCH --ntasks-per-node=24
#SBATCH -t 04:00:00
#SBATCH -A cis260079p
#SBATCH -q low
#SBATCH -o logs/genispd_%j.out
#SBATCH -e logs/genispd_%j.err
#SBATCH -J genispd
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs validation_out
PY=/ocean/projects/cis260079p/ctsai4/miniconda3/envs/deepdgr/bin/python3
export PYTHONUNBUFFERED=1 TMPDIR=${LOCAL:-/tmp}
SD=/ocean/projects/cis260079p/ctsai4/synthdata
# ISPD-ACROSS, teacher-free, NO overflow-0 certify (ISPD is inherently
# congested) -> variants are routable like the real chips. 100/family x6 = 600.
$PY gen_synth_ispd.py --per 100 --workers 12 --mode translate_jitter \
   --window_frac 0.04 --run ispd --out_root "$SD" \
   > validation_out/gen_ispd.log 2>&1
rc=$?
echo "genispd rc=$rc packs=$(ls $SD/ispd/*.npz 2>/dev/null | wc -l) du=$(du -sh $SD/ispd 2>/dev/null | cut -f1)" | tee validation_out/gen_ispd_done.txt
tail -8 validation_out/gen_ispd.log
