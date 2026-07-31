#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gpus=h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH -t 04:00:00
#SBATCH -A cis260079p
#SBATCH -o logs/trainfull_%j.out
#SBATCH -e logs/trainfull_%j.err
#SBATCH -J trainfull
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs validation_out
module load cuda/12.1.1
export PATH=/ocean/projects/cis260079p/ctsai4/miniconda3/envs/deepdgr/bin:$PATH
export PYTHONUNBUFFERED=1
PY=/ocean/projects/cis260079p/ctsai4/miniconda3/envs/deepdgr/bin/python3
SD=/ocean/projects/cis260079p/ctsai4/synthdata

echo "===== (1) GENERALIZED e2e on the 6 REAL full-size ISPD benchmarks (held-out=19_t9, only eval'd) ====="
$PY e2e_stream.py --instances 'cu-gr-2/run/ispd1*_test*_metal5.pt' \
   --steps 500 --lr 1e-2 --device 0 --hidden 64 --layers 3 --cache 2 \
   --holdout 1 --eval_every 100 --log_every 25 --save_gnn stream_gnn_ispd_full.pth \
   > validation_out/train_e2e_fullsize.log 2>&1 ; echo "  e2e rc=$?"

echo "===== (2) BATCH training (teacher-free physics) on full-size ISPD-derived packs ====="
$PY train_scalable.py --packs "$SD/ispd1k/*.npz" --holdout 8 --batch 1 \
   --milestones 50,200,400 --max_steps 500 --device 0 \
   --out_csv validation_out/train_batch_fullsize_curve.csv \
   > validation_out/train_batch_fullsize.log 2>&1 ; echo "  batch rc=$?"
echo "DONE trainfull"
