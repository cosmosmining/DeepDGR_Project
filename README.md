# Differentiable Global Routing with a Cross-Benchmark Warm-Start GNN

Reproducibility release for the report **`report/final_report.pdf`**.

We extend **DGR** (Differentiable Global Router, DAC'24) with a **warm-start
GNN** trained end-to-end (teacher-free) on all six ISPD'18/'19 benchmarks at
full size. With the warm start, DGR needs only **~15–50 iterations** (instead
of 2000), and an ensemble finishing stage converts the saved time into extra
quality. Routed through **CUGR2** (EDGE, DAC'23), the combined system
**strictly beats both native CUGR2 and DGR on all six benchmarks**
(wirelength −0.3% to −3.5%, vias −1.4% to −8.1%, overflow ≤ both;
e.g. 18→10 overflow on ispd19_test8).

Everything in the report is measured by CUGR2 itself, under **isolated
routing** (each route in its own temp dir → no cross-run corruption).

---

## What this repo reproduces

| Report item | What it shows | Script(s) |
|---|---|---|
| **Table 1** — beats CUGR2 & DGR 6/6 | main quality result | `test_gnn_all6.py`, `optimize_final.py` |
| **Fig 1** — 12-setting parameter sweep | via vs WL, bubble = overflow | `route_scatter12.py` → `plot_scatter12_pro.py` |
| **Table 2 / Fig 2** — runtime & early-stop | 71–177× optimize; stop ~15–50 it | `overflow_curve.py`, `make_figset.py` |
| **Table 3** — entire-flow wall time | all overhead + route, 2.8–4.8× | `make_figset.py` |
| **Fig 3** — overflow heatmaps (18 maps) | native/DGR/ours, all 6 chips | `compare_heatmaps.py` → `render_overflow_maps.py` |
| **§4.5** — flat data-scaling (honest) | model-capacity bound | `scaling_diverse.py`, `train_cached.py` |
| **§2.2–2.3** — GNN + training | grid-coarsening OOM fix, teacher-free e2e | `train_all6_robust.py`, `deepdgr_e2e.py`, `e2e_stream.py`, `compare_e2e_vs_supervised.py` |
| **§2.5 / §3.2** — synthetic data | CPU generators, ~18k instances | `gen_synth_batch.py`, `gen_synth_ispd.py` |

Full command list: **[`docs/REPRODUCE.md`](docs/REPRODUCE.md)**.

---

## Layout (run everything from the repo root)

Scripts are **flat at the top level** on purpose — they import each other as
plain modules and resolve paths relative to the repo root (`cu-gr-2/run`,
`RESULTS/`, `tmp/`, `CUGR2_guide/`). Do not move them into subfolders.

```
.
├── data.py model.py util.py main_stochastic.py data_process_CUGR2.py   # core DGR (NVlabs, Apache-2.0)
├── dgr_fast.py fastdgr_core.py discrete_refine.py                      # fast optimizer + discrete finishing
├── train_all6_robust.py deepdgr_e2e.py e2e_stream.py train_distributed.py
│   train_cached.py train_scalable.py compare_e2e_vs_supervised.py      # warm-start GNN training
├── gen_synth_batch.py gen_synth_ispd.py generate_scratch_benchmarks.py
│   generate_compact_graphs.py generate_e2e_graphs.py
│   imf_*.py step1_utils.py                                             # synthetic data generation
│   compact_to_hetero.sh                                               # (SLURM submit wrapper)
├── tune_cugr2.py test_gnn_all6.py iso_compare.py optimize_final.py
│   warmstart_converge.py guide_writer.py                              # routing / evaluation (isolated)
├── overflow_curve.py scaling_diverse.py                               # curves + scaling study
├── congestion_viz.py compare_heatmaps.py render_overflow_maps.py      # congestion / overflow maps
├── route_scatter12.py plot_scatter12_pro.py                           # 12-setting parameter sweep
├── make_figset.py make_color_plots.py make_pptx.py                    # report/slide figures
├── ispd24_adapter.py test_ispd24_adapter.py                          # ISPD'24 industrial adapter
├── gnn_all6_robust.pth        # the trained warm-start GNN (deployed model)
├── report/                    # final_report.pdf (+ .tex, slides .tex)
├── RESULTS/                   # measured CSV tables + headline figures
├── docs/                      # SETUP, BENCHMARKS, REPRODUCE
├── cu-gr-2/                   # <- you build the CUGR2 router here (see docs/SETUP.md)
├── tmp/ CUGR2_guide/          # working dirs (candidate-pool cache, generated guides)
└── requirements.txt
```

`RESULTS/*.csv` are the exact numbers behind every table/figure in the report
(`final_best.csv` = Table 1, `flow_runtime.csv` = Table 3, `scatter12.csv` =
Fig 1, `overflow_curve_*.csv` = Fig 2, `scaling_*.csv` = §4.5).

---

## Quick start

```bash
# 1. environment  (see docs/SETUP.md for the exact conda spec)
conda create -n deepdgr python=3.10 -y && conda activate deepdgr
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install -r requirements.txt

# 2. verify env + regenerate the report figures from shipped RESULTS/*.csv
#    (needs NO benchmarks, NO GPU — proves the toolchain works)
./verify.sh

# 3. build the CUGR2 router (clones + builds the exact pinned commit)
./setup_cugr2.sh          # -> cu-gr-2/run/route (+ FLUTE tables)

# 4. place the ISPD benchmarks   (see docs/BENCHMARKS.md — not redistributable)
#    -> cu-gr-2/benchmark/test/<bench>/<bench>.input.lef / .input.def

# 5. preprocess a benchmark into a .pt   (one-time, per benchmark)
python3 data_process_CUGR2.py cu-gr-2 cu-gr-2/benchmark/test/ispd18_test5_metal5

# 6. reproduce the headline table (GNN warm start -> few-iter DGR -> isolated CUGR2)
python3 test_gnn_all6.py --load_gnn gnn_all6_robust.pth --iter 50 --device 0
#    -> RESULTS/all6_test.csv   (compare to RESULTS/final_best.csv)
```

**`./verify.sh`** (step 2) imports the stack, byte-compiles every script, and
regenerates Fig 1 / Fig 2 / Table 3 figures from the shipped CSVs — confirm the
environment is correct before touching CUGR2 or benchmarks. Full per-figure
reproduction: **[`docs/REPRODUCE.md`](docs/REPRODUCE.md)**.

### Verified to run out-of-the-box
- **Environment + figure regeneration** — `./verify.sh` (no benchmarks/GPU).
- **Synthetic benchmark generation from scratch** — `gen_synth_batch.py` (CPU).
- Raw-benchmark → routing steps (`data_process_CUGR2.py`, `test_gnn_all6.py`,
  `route_scatter12.py`, `compare_heatmaps.py`, `overflow_curve.py`) need CUGR2
  built (step 3) + ISPD benchmarks in place (step 4); see `docs/REPRODUCE.md`.

---

## Requirements at a glance

- Linux + NVIDIA GPU (training / DGR optimization). CPU is fine for data
  generation, routing, and all plotting.
- Python 3.10, PyTorch 2.5.1 (cu121), PyTorch-Geometric, torch-scatter, numpy,
  matplotlib, seaborn, python-pptx.
- CUGR2 built from source (`github.com/wadmes/cu-gr-2`) → `cu-gr-2/run/route`.
- ISPD'18 & ISPD'19 metal5 benchmarks (LEF/DEF) — see `docs/BENCHMARKS.md`.

## Cost (measured)

One-time GNN training **~15 GPU-minutes** (all six full-size designs; ~25 incl.
warm-start emission). Per design at inference: GNN ~1 s + optimize 3–11 s +
rounding/refine 35–93 s + route 9–100 s → **entire flow 84–420 s** (2.8–4.8×
faster than the DGR flow). Data generation is CPU-only.

## Honest scope (from the report)

Per-benchmark CUGR2 knob tuning helps **all** methods equally and is applied
symmetrically — not claimed as a learned gain. Data scaling of the GNN is
**flat** beyond ~50 distinct designs (model/feature capacity bound). The
robust, defensible contributions are: the **speed** (few-iteration DGR), the
**one-model cross-benchmark** warm start (grid-coarsening OOM fix), **teacher-
free** training, and the **congested-chip overflow** win (18→10 on 19_t8).

## License

Core DGR files (`data.py`, `data_process_CUGR2.py`, `util.py`, `model.py`,
`main_stochastic.py`, `test.py`, and the `*.sh` drivers) originate from
NVlabs/DGR under **Apache-2.0**. The warm-start GNN extension and all other
scripts are released under the same terms. CUGR2 is a separate project with
its own license. See `docs/SETUP.md`.
