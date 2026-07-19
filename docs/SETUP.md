# Setup

## 1. Python environment

Reproduces the environment used for all reported numbers (PSC Bridges-2,
H100-80GB, CUDA 12.1).

```bash
conda create -n deepdgr python=3.10 -y
conda activate deepdgr
pip install -r requirements.txt
```

`requirements.txt` pins the important ones. `torch-scatter` /
`torch-geometric` must match your CUDA/torch build — install from the PyG
wheel index if pip cannot resolve them:

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-geometric \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

GPU is required only for GNN **training** and DGR **optimization**
(`main_stochastic.py`, `dgr_fast.py`, `train_*.py`). Data generation, routing,
and every plotting script run on CPU.

## 2. Build the CUGR2 router

The router is a separate C++ project. Clone and build it into `cu-gr-2/`:

```bash
git clone https://github.com/wadmes/cu-gr-2 cu-gr-2
cd cu-gr-2
# follow that repo's build instructions (cmake + make); it produces:
#   cu-gr-2/run/route     <- the global router binary this pipeline calls
#   cu-gr-2/run/drcu      <- (optional) detailed router
cd ..
```

All scripts locate the binary at **`cu-gr-2/run/route`** (see
`tune_cugr2.py: ROUTE`). The common invocation base point used everywhere is
`-wsa 500 -via_cost 20 -sort 1` (`tune_cugr2.py: BASE_POINT`); per-benchmark
knobs (`-cls`, `-vm`, `-wsa`) are listed in the report §2 and applied
symmetrically to all methods.

### FLUTE tables

CUGR2/FLUTE needs `POWV9.dat` and `POST9.dat` in the working directory. Keep
the copies that ship with cu-gr-2 at **`cu-gr-2/run/POWV9.dat`** and
**`cu-gr-2/run/POST9.dat`** — isolated routing symlinks them into each private
run directory automatically (`iso_compare.py`, `tune_cugr2.py`,
`compare_heatmaps.py`).

## 3. Benchmarks

See **[`BENCHMARKS.md`](BENCHMARKS.md)** for where to obtain and place the
ISPD'18/'19 LEF/DEF files.

## 4. Sanity check

```bash
# native route of one benchmark, isolated, parsed to WL/via/overflow
python3 iso_compare.py --bench ispd18_test5_metal5 --method native
```

If this prints wirelength / via / overflow matching the "native" column of
`RESULTS/final_best.csv` (26.57M / 856k / 5), the toolchain is correctly set
up.

## Notes / gotchas

- **AMP off for the physics loss** — cuSPARSE has no fp16 SpMV; training uses
  fp32 for the sparse routing objective (`train_all6_robust.py`).
- **Candidate-pool cache** — `main_stochastic.py` / `dgr_fast.py` cache the
  per-benchmark candidate pool under `tmp/<name>_candidate_pool.pt`. If you
  change benchmark geometry or candidate params, delete `tmp/` (or pass
  `--read_new_tree True`) to force regeneration.
- **`torch_geometric` cold import** on network filesystems can take minutes;
  run training/eval on a compute node, not a login node.
