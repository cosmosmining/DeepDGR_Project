#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_overflow_maps.py — PAPER-STANDARD overflow heatmaps from the cached
router data (hm_<chip>_<method>.npz), following the routability-paper
convention (RouteNet / CircuitNet style): WHITE background, only overflow is
colored, jet scale by magnitude (blue = mild -> red = severe), shared
colorbar in tracks. Sparse single-GCell overflow is dilated (3x3/5x5 max
filter) purely for print visibility, as congestion figures in papers do.

Outputs (same filenames the docs reference):
  RESULTS/figs/heatmap_<chip>_<method>.png            (18 maps)
  RESULTS/figs/congestion_compare_<chip>.{pdf,-1.png} (per-chip 1x3)
  RESULTS/figs/congestion_compare_all.{pdf,-1.png}    (6x3 montage)
NEW file; rendering only, no routing.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "RESULTS", "figs")
BENCHES = ["ispd18_test5", "ispd18_test8", "ispd18_test10",
           "ispd19_test7", "ispd19_test8", "ispd19_test9"]
METHODS = ["native", "dgr", "ours"]
MLAB = {"native": "native CUGR2", "dgr": "baseline DGR", "ours": "ours"}
OFFICIAL = {"ispd18_test5": (5, 7, 0), "ispd18_test8": (0, 0, 0),
            "ispd18_test10": (0, 1, 0), "ispd19_test7": (0, 0, 0),
            "ispd19_test8": (18, 19, 10), "ispd19_test9": (30, 37, 28)}
VMAX = 5.0                    # tracks; >=5 saturates red
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.dpi": 110, "figure.facecolor": "white"})


def dilate(a, k):
    """k x k maximum filter (visibility of isolated overflowed GCells)."""
    out = a.copy()
    r = k // 2
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            out = np.maximum(out, np.roll(np.roll(a, dx, 0), dy, 1))
    return out


def load(chip, method):
    d = np.load(os.path.join(FIG, f"hm_{chip}_{method}.npz"))
    return d["ov"]


def panel(ax, ov, title, fs=11):
    G = max(ov.shape)
    k = 3 if G < 800 else 5                     # bigger grids need more dilation
    img = dilate(ov, k)
    cmap = cm.get_cmap("jet").copy()
    cmap.set_under("white")                     # zero overflow -> white die
    m = ax.imshow(img.T, origin="lower", cmap=cmap,
                  norm=Normalize(vmin=1e-6, vmax=VMAX), aspect="equal",
                  interpolation="nearest")
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=fs, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#333333")
        sp.set_linewidth(0.8)
    return m


def cbar(fig, m, axes):
    cb = fig.colorbar(m, ax=axes, shrink=0.85, pad=0.015, aspect=26,
                      extend="max")
    cb.set_label("overflow (tracks per GCell)", fontsize=10)
    cb.outline.set_linewidth(0.6)
    return cb


def main():
    # 18 individual maps
    for c in BENCHES:
        for i, mth in enumerate(METHODS):
            ov = load(c, mth)
            fig, ax = plt.subplots(figsize=(4.8, 4.5))
            m = panel(ax, ov, f"{c} — {MLAB[mth]}\n"
                      f"overflow: {OFFICIAL[c][i]}")
            cbar(fig, m, [ax])
            fig.savefig(os.path.join(FIG, f"heatmap_{c}_{mth}.png"),
                        dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        # per-chip 1x3
        fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.7),
                                 constrained_layout=True)
        for ax, mth, i in zip(axes, METHODS, range(3)):
            m = panel(ax, load(c, mth),
                      f"{MLAB[mth]}\noverflow: {OFFICIAL[c][i]}")
        fig.suptitle(f"{c}: routed overflow (white = no overflow; "
                     "color = tracks over capacity)", fontsize=12)
        cbar(fig, m, axes)
        for ext in (".pdf", "-1.png"):
            fig.savefig(os.path.join(FIG, f"congestion_compare_{c}{ext}"),
                        dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"-> congestion_compare_{c}")
    # montage 6x3
    fig, axes = plt.subplots(6, 3, figsize=(11.5, 22.0),
                             constrained_layout=True)
    for r, c in enumerate(BENCHES):
        for col, mth in enumerate(METHODS):
            m = panel(axes[r][col], load(c, mth),
                      f"{c} — {MLAB[mth]}   overflow: {OFFICIAL[c][col]}",
                      fs=10)
    cbar(fig, m, axes)
    for ext in (".pdf", "-1.png"):
        fig.savefig(os.path.join(FIG, f"congestion_compare_all{ext}"),
                    dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("-> congestion_compare_all (18 maps)")


if __name__ == "__main__":
    main()
