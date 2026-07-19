#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 48:00:00
#SBATCH --gpus=h100-80:4
#SBATCH --cpus-per-task=20
#SBATCH -o logs/compact_gen_%j.out
#SBATCH -e logs/compact_gen_%j.err
#SBATCH -J compact_gen
# ============================================================================
# submit_compact_gen.sh
# ----------------------------------------------------------------------------
# Generates BOUNDED compact synthetic data for ONE benchmark, create-one /
# delete-one (generate_compact_graphs.py keeps a single temp file at a time).
#
# GPU rationale: the DGR teacher (main_stochastic.py, 2000 iters) is GPU
# gradient descent and is hopeless on CPU. The graph build is CPU+RAM. On
# GPU-shared you get 5 CPUs x 2000MB = 10GB per GPU, which OOM'd the builder.
# Requesting 4 h100-80 buys 20 CPUs x 2000MB = 40GB RAM (GPU 0 runs the
# teacher; the rest just supply the headroom). One allocation covers both.
#
# Usage:
#   sbatch submit_compact_gen.sh <benchmark> [n_variants] [rand_mode]
# Examples:
#   sbatch submit_compact_gen.sh ispd18_test5_metal5 1          # verify one
#   sbatch submit_compact_gen.sh ispd18_test5_metal5 50         # full run
#   sbatch submit_compact_gen.sh ispd18_test5_metal5 50 translate_jitter
# ============================================================================

module load cuda/12.1.1
export PATH=/ocean/projects/cis260079p/ctsai4/miniconda3/envs/deepdgr/bin:$PATH
export PYTHONUNBUFFERED=1

# Per-variant temp (.pt + intermediate graph) goes to NODE-LOCAL disk so it
# never touches the shared /ocean quota. With bounded pins the temp graph is
# small and is deleted after each variant anyway.
if [ -n "$LOCAL" ] && [ -d "$LOCAL" ]; then
    export TMPDIR="$LOCAL"
else
    export TMPDIR="/ocean/projects/cis260079p/ctsai4/tmp/$SLURM_JOB_ID"
fi
mkdir -p "$TMPDIR"
echo "TMPDIR=$TMPDIR"

PYTHON=/ocean/projects/cis260079p/ctsai4/miniconda3/envs/deepdgr/bin/python3
DGR_DIR=/ocean/projects/cis260079p/ctsai4/Differentiable-Global-Router
COMPACT_DIR=/ocean/projects/cis260079p/ctsai4/compact

BENCH="${1:?Usage: sbatch submit_compact_gen.sh <benchmark> [n_variants] [rand_mode]}"
NVARIANTS="${2:-50}"
RAND_MODE="${3:-translate}"

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Benchmark=$BENCH  variants=$NVARIANTS  rand_mode=$RAND_MODE"
echo "RAM: $(free -h | awk '/Mem:/{print $2}')"
$PYTHON -c "import numpy,torch;print(f'numpy={numpy.__version__} torch={torch.__version__} cuda={torch.cuda.is_available()}')" || exit 1

cd "$DGR_DIR"
cp cu-gr-2/run/POWV9.dat . 2>/dev/null
cp cu-gr-2/run/POST9.dat . 2>/dev/null
mkdir -p "$COMPACT_DIR/$BENCH"

stdbuf -oL -eL $PYTHON -u generate_compact_graphs.py generate \
    --template "${BENCH}.pt" \
    --output_dir "$COMPACT_DIR/$BENCH" \
    --n_variants "$NVARIANTS" \
    --seed 0 \
    --device 0 \
    --dgr_iter 2000 \
    --rand_mode "$RAND_MODE"

echo "EXIT CODE: $?"
echo "FINISHED AT: $(date)"
