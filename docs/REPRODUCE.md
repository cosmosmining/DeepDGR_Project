# Reproduce every table and figure in `report/final_report.pdf`

All commands run **from the repo root**, with the conda env active, CUGR2 built
at `cu-gr-2/run/route`, and the six benchmarks in place (see `SETUP.md`,
`BENCHMARKS.md`). Every script has a docstring header with its own usage; the
options below match the report's numbers.

The exact numbers behind each item are already in `RESULTS/*.csv` so you can
diff your run against ours.

---

## 0. One-time preprocessing (per benchmark)

```bash
for b in ispd18_test5_metal5 ispd18_test8_metal5 ispd18_test10_metal5 \
         ispd19_test7_metal5 ispd19_test8_metal5 ispd19_test9_metal5; do
  python3 data_process_CUGR2.py cu-gr-2 cu-gr-2/benchmark/test/$b
done
```

Produces `<bench>.pt` for the optimizer. (Delete `tmp/` if you regenerate
benchmarks — it caches candidate pools.)

---

## Table 1 — beats native CUGR2 and DGR on 6/6  (§4.1)

The deployed warm-start GNN is shipped as `gnn_all6_robust.pth`.

```bash
# GNN warm start -> few-iteration DGR -> isolated CUGR2 route, per chip
python3 test_gnn_all6.py --load_gnn gnn_all6_robust.pth --iter 50 --device 0
#   -> RESULTS/all6_test.csv

# ensemble finishing (§2.6): 4 seeds x 3 objective weightings, best kept
python3 optimize_final.py --seeds 4 --iter 160 --rounding_k 32 \
        --refine_passes 6 --refine_moves 12000 --device 0
#   -> RESULTS/final_optimized.csv
```

Reference: `RESULTS/final_best.csv` (per-chip best of the two), and
`RESULTS/all6_beats_native_and_dgr.csv`.

To (re)train the GNN from scratch instead of using the shipped weights:

```bash
python3 train_all6_robust.py --steps 900 --max_grid 120000 --lr 1e-2 \
        --amp False --save_gnn gnn_all6_robust.pth --device 0
# then emit per-chip warm-start npz (used by optimize_final.py):
python3 train_all6_robust.py --test_only --load_gnn gnn_all6_robust.pth --device 0
```

---

## Figure 1 — 12-setting parameter sweep  (§4.2)

```bash
# route all 3 methods under the same 12 CUGR2 parameter settings (isolated)
python3 route_scatter12.py --workers 4          # -> RESULTS/scatter12.csv
# improved bubble/trade-off plot (real data): whitegrid, alpha, smart sqrt sizing
python3 plot_scatter12_pro.py                   # -> RESULTS/figs/scatter12pro_all-1.png
python3 plot_scatter12_pro.py --pareto          # -> ...scatter12pro_pareto_all-1.png
```

---

## Table 2 + Figure 2 — runtime and early stopping  (§4.3)

```bash
# overflow-vs-iteration, warm vs cold, per chip (Fig 2)
python3 overflow_curve.py --bench ispd18_test5_metal5 \
        --warmstart ispd18_test5_metal5_ROBUST_ws.npz \
        --budgets 5,15,30,60,120,250 --tol 0.02 --device 0
python3 overflow_curve.py --bench ispd19_test8_metal5 \
        --warmstart ispd19_test8_metal5_ROBUST_ws.npz \
        --budgets 5,15,30,60,120,250 --tol 0.02 --device 0
#   -> RESULTS/overflow_curve_18_t5.csv , overflow_curve_19_t8.csv
```

Runtime tables (Table 2 cold-vs-warm, and the figures) are assembled by:

```bash
python3 make_figset.py         # -> RESULTS/flow_runtime.csv + RESULTS/figs/fig_*.png
python3 make_color_plots.py    # -> recolored overflow/scaling/runtime figures
```

---

## Table 3 — entire-flow wall time  (§4.3)

`make_figset.py` writes `RESULTS/flow_runtime.csv` (native route-only, DGR
flow, ours flow, and the 2.8–4.8× ratio) from the measured component times.

---

## Figure 3 — routed-overflow heatmaps, 18 maps  (§4.4)

```bash
# route native/DGR/ours for all 6 chips (isolated), cache the router heatmaps
python3 compare_heatmaps.py --workers 3
# render paper-standard overflow maps (white bg, only overflow colored)
python3 render_overflow_maps.py
#   -> RESULTS/figs/heatmap_<chip>_<method>.png (18)
#   -> RESULTS/figs/congestion_compare_<chip>.png (6) + congestion_compare_all.png
```

---

## §4.5 — data scaling is flat (honest negative)

```bash
# distinct-set-size sweep, fixed held-out, equal step budget
python3 scaling_diverse.py --packs '<synth_pack_glob>' \
        --sizes 25,50,100,250,1000 --steps 600 --max_grid 120000 --device 0
#   -> RESULTS/scaling_diverse.csv   (certified packs)
# congested-data variant + cached trainer (§4.6, ~42 s / 600 steps on CPU):
python3 train_cached.py --packs '<congested_glob>' --sizes 50,200,800,3200 \
        --steps 600 --holdout 24 --cache 64 --device -1
#   -> RESULTS/scaling_congested.csv
```

---

## §2.3 / §4.6 — teacher-free e2e vs supervised

```bash
python3 compare_e2e_vs_supervised.py --instances '<pack_glob>' --steps 800 \
        --lr 1e-2 --holdout 24 --teacher_iters 2000 --device 0 \
        --out_csv RESULTS/e2e_vs_sup.csv
# streaming (switch-benchmark) e2e training of one shared GNN:
python3 e2e_stream.py --device 0
```

---

## §2.5 / §3.2 — synthetic data generation (CPU)

```bash
# from-scratch bounded-box, routable, certified overflow=0
python3 gen_synth_batch.py --run gen --out_root <dir> --n 1000 --workers 32 \
        --util 0.7 --certify
# congested variants (genuine overflow signal)
python3 gen_synth_batch.py --run gen --out_root <dir> --n 1000 --workers 32 \
        --util 0.95 --keep_uncertified
# ISPD-template bounded pin randomization (locality-preserving)
python3 gen_synth_ispd.py --out_root <dir> --n 200
```

---

## Rebuild the report and slides

```bash
cd report && pdflatex final_report.tex && pdflatex final_report.tex   # figures from ../RESULTS/figs
python3 make_pptx.py       # -> RESULTS/DeepDGR_final.pptx (if you want the deck)
```

---

### Notes

- Every routing step is **isolated** (unique temp cwd + FLUTE symlinks); this
  is what makes concurrent `--workers > 1` safe and the numbers reproducible.
- GPU indices are `--device 0` (or `-1`/omit for CPU where supported).
- `<..._glob>` placeholders point at wherever you generated synthetic packs in
  the data-generation step.
