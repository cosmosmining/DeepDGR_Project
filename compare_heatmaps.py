#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_heatmaps.py — ROUTER-MEASURED congestion heatmaps across methods:
native CUGR2 vs baseline DGR vs ours, for all six ISPD benchmarks
(18 maps total), in the industry (Innovus/OpenROAD) binned convention.

Congestion source is CUGR2 itself: each isolated route writes heatmap.txt
(per layer, per GCell, usage - capacity in tracks); we aggregate layers into
per-cell overflow (sum of positive parts) and free slack (sum of negative
parts) and color by discrete bins on a dark die background:
  free >=20 tracks (dark) | 10-20 | 3-10 | <3 (near capacity, yellow)
  overflow 1-2 (orange) | 3-5 (red) | >5 (magenta)
All three methods are routed with IDENTICAL router settings (-wsa 500
-via_cost 20 -sort 1); only the -dgr guide differs. Outputs:
  RESULTS/figs/heatmap_<chip>_<method>.png          (18 individual maps)
  RESULTS/figs/congestion_compare_<chip>.{pdf,png}  (per-chip 1x3)
  RESULTS/figs/congestion_compare_all.{pdf,png}     (6x3 montage)
Reuses tune_cugr2 READ-ONLY; NEW file. No torch import (fast start).

  python3 compare_heatmaps.py --workers 3
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import tune_cugr2 as T

BENCHES = ["ispd18_test5_metal5", "ispd18_test8_metal5",
           "ispd18_test10_metal5", "ispd19_test7_metal5",
           "ispd19_test8_metal5", "ispd19_test9_metal5"]
SHORT = {b: b.replace("_metal5", "").replace("ispd", "ispd") for b in BENCHES}
METHODS = ["native", "dgr", "ours"]
MLAB = {"native": "native CUGR2", "dgr": "baseline DGR", "ours": "ours"}
# per-bench knob used when "ours" was evaluated in the results tables
KNOB = {"ispd18_test5_metal5": {"cls": 2.0, "vm": 1.0},
        "ispd18_test8_metal5": {"cls": 4.0, "vm": 1.0},
        "ispd18_test10_metal5": {"cls": 2.0, "vm": 1.0},
        "ispd19_test7_metal5": {"cls": 4.0, "vm": 1.0},
        "ispd19_test8_metal5": {"vm": 1.0},
        "ispd19_test9_metal5": {"cls": 2.0, "vm": 1.0, "wsa": 1000}}
FIG = os.path.join(ROOT, "RESULTS", "figs")

# PURE OVERFLOW view: dark wherever there is NO overflow; colored bins by
# how many tracks over capacity a GCell is (the Innovus overflow map)
CAT_COLORS = ["#101014",   # 0 free
              "#6B5D1F",   # 1 congested area (near capacity, no overflow)
              "#FFD92F",   # 2 <=1 track over
              "#F57C00",   # 3 1-2 tracks over
              "#E31A1C",   # 4 2-3 tracks over
              "#A50F15",   # 5 3-5 tracks over
              "#C51B8A"]   # 6 >5 tracks over
CAT_LABELS = ["free", "congested area", "≤1 track over", "1-2 tracks",
              "2-3 tracks", "3-5 tracks", ">5 tracks"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.dpi": 110})


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def guide_for(b, method):
    if method == "native":
        return None
    if method == "dgr":
        s = b.replace("ispd", "").replace("_metal5", "")
        return os.path.join(ROOT, "CUGR2_guide", f"CUgr_{b}_S2_{s}.txt")
    return os.path.join(ROOT, "CUGR2_guide", f"CUgr_{b}_ROBUST6.txt")


def parse_heatmap(path):
    """heatmap.txt -> (overflow_map, slack_map), aggregated over layers."""
    with open(path) as fh:
        L, X, Y = map(int, fh.readline().split())
        ov = np.zeros((X, Y), dtype=np.float32)
        fr = np.zeros((X, Y), dtype=np.float32)
        for _ in range(L):
            fh.readline()                               # layer name
            for y in range(Y):
                vals = np.fromstring(fh.readline(), sep=" ")
                v = vals[:X]
                ov[:, y] += np.maximum(v, 0)
                fr[:, y] += np.maximum(-v, 0)
    return ov, fr


def route_heatmap(bench, method):
    """isolated route (identical settings, only the guide differs) ->
    (overflow, slack) maps from the router's own heatmap.txt. Cached."""
    cache = os.path.join(FIG, f"hm_{SHORT[bench]}_{method}.npz")
    if os.path.isfile(cache):
        d = np.load(cache)
        return d["ov"], d["fr"]
    lef, deff = T.find_input(bench, "lef"), T.find_input(bench, "def")
    cwd = tempfile.mkdtemp(prefix="hm_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        for dat in ("POWV9.dat", "POST9.dat"):
            os.symlink(os.path.join(T.RUN_DIR, dat), os.path.join(cwd, dat))
        merged = dict(T.BASE_POINT)
        if method == "ours":
            merged.update(KNOB.get(bench, {}))   # evaluated operating point
        cmd = [T.ROUTE, "-lef", lef, "-def", deff,
               "-output", os.path.join(cwd, "o.guide")]
        for k, v in merged.items():
            cmd += [f"-{k}", str(v)]
        g = guide_for(bench, method)
        if g:
            cmd += ["-dgr", g]
        with open(os.path.join(cwd, "r.log"), "w") as lf:
            subprocess.call(cmd, cwd=cwd, stdout=lf,
                            stderr=subprocess.STDOUT)
        hm = os.path.join(cwd, "heatmap.txt")
        if not os.path.isfile(hm):
            raise RuntimeError(f"no heatmap.txt for {bench}/{method}")
        ov, fr = parse_heatmap(hm)
        np.savez_compressed(cache, ov=ov, fr=fr)
        return ov, fr
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


def categorize(ov, fr):
    idx = np.zeros(ov.shape, dtype=np.int8)     # 0 = free (dark)
    idx[(ov == 0) & (fr < 3)] = 1               # congested area (near cap.)
    idx[ov > 0] = 2
    idx[ov > 1] = 3
    idx[ov > 2] = 4
    idx[ov > 3] = 5
    idx[ov > 5] = 6
    return idx


# official routed overflow (the metric used in all result tables)
OFFICIAL = {"ispd18_test5": (5, 7, 0), "ispd18_test8": (0, 0, 0),
            "ispd18_test10": (0, 1, 0), "ispd19_test7": (0, 0, 0),
            "ispd19_test8": (18, 19, 10), "ispd19_test9": (30, 37, 28)}


def draw_panel(ax, ov, fr, title, fs=11, chip=None, method=None):
    cmap = ListedColormap(CAT_COLORS)
    ax.set_facecolor(CAT_COLORS[0])
    ax.imshow(categorize(ov, fr).T, origin="lower", cmap=cmap, vmin=0,
              vmax=len(CAT_COLORS)-1, interpolation="nearest", aspect="equal")
    sub = ""
    if chip in OFFICIAL and method in METHODS:
        sub = f"\noverflow: {OFFICIAL[chip][METHODS.index(method)]}"
    ax.set_title(title + sub, fontsize=fs, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#444444")


def legend_handles():
    return [Patch(fc=c, ec="#444444", label=l)
            for c, l in zip(CAT_COLORS, CAT_LABELS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    jobs = [(b, m) for b in BENCHES for m in METHODS]
    maps = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(route_heatmap, b, m): (b, m) for b, m in jobs}
        for f in futs:
            pass
        for fut, (b, m) in futs.items():
            maps[(b, m)] = fut.result()
            log(f"{SHORT[b]}/{m}: overflowed GCells "
                f"{int((maps[(b, m)][0] > 0).sum()):,}")
    log(f"all 18 routes/maps in {(time.time()-t0)/60:.1f} min")

    # 18 individual maps + per-chip 1x3 comparisons
    for b in BENCHES:
        for m in METHODS:
            ov, fr = maps[(b, m)]
            fig, ax = plt.subplots(figsize=(4.6, 4.6))
            draw_panel(ax, ov, fr, f"{SHORT[b]} — {MLAB[m]}", chip=SHORT[b], method=m)
            fig.legend(handles=legend_handles(), loc="lower center",
                       ncol=4, frameon=False, fontsize=6.5)
            fig.tight_layout(rect=(0, 0.07, 1, 1))
            fig.savefig(os.path.join(FIG, f"heatmap_{SHORT[b]}_{m}.png"),
                        dpi=150)
            plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.9),
                                 constrained_layout=True)
        for ax, m in zip(axes, METHODS):
            draw_panel(ax, *maps[(b, m)], MLAB[m], chip=SHORT[b], method=m)
        fig.suptitle(f"{SHORT[b]}: routed OVERFLOW map "
                     "(each method at its evaluated operating point)",
                     fontsize=12)
        fig.legend(handles=legend_handles(), loc="center left",
                   bbox_to_anchor=(1.002, 0.5), frameon=False, fontsize=9)
        for ext in (".pdf", "-1.png"):
            fig.savefig(os.path.join(
                FIG, f"congestion_compare_{SHORT[b]}{ext}"),
                bbox_inches="tight", dpi=150)
        plt.close(fig)
        log(f"-> congestion_compare_{SHORT[b]}")

    # 6x3 montage
    fig, axes = plt.subplots(6, 3, figsize=(11.4, 22.5),
                             constrained_layout=True)
    for r, b in enumerate(BENCHES):
        for c, m in enumerate(METHODS):
            draw_panel(axes[r][c], *maps[(b, m)],
                       f"{SHORT[b]} — {MLAB[m]}", fs=10, chip=SHORT[b], method=m)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=7,
               frameon=False, fontsize=9)
    for ext in (".pdf", "-1.png"):
        fig.savefig(os.path.join(FIG, f"congestion_compare_all{ext}"),
                    bbox_inches="tight", dpi=110)
    plt.close(fig)
    log("-> congestion_compare_all (6x3 montage) — 18 maps total")


if __name__ == "__main__":
    main()
