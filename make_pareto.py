#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pareto.py — definitive Pareto-frontier comparison of native CUGR2 vs
baseline DGR vs ours, from the REAL 12-setting sweep
(RESULTS/scatter12.csv + experiments/cugr2_tune/iso_clean.csv).

For each chip and method we compute the lower-left non-dominated frontier in
(via count, wirelength) — the standard routing quality trade-off — and draw it
as a staircase. Non-frontier operating points are faded; frontier points are
solid, sized by overflow. A method whose frontier lies to the lower-left is
strictly better. Outputs:
  RESULTS/figs/pareto_all-1.png / .pdf        (2x3, one panel per chip)
  RESULTS/figs/pareto_<chip>-1.png            (per chip)
NEW file; reads only, no routing.

  python3 make_pareto.py
"""
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "RESULTS", "figs")
CSV = os.path.join(ROOT, "RESULTS", "scatter12.csv")
ISO = os.path.join(ROOT, "experiments", "cugr2_tune", "iso_clean.csv")
CHIPS = ["ispd18_test5", "ispd18_test8", "ispd18_test10",
         "ispd19_test7", "ispd19_test8", "ispd19_test9"]
METHODS = ["native", "dgr", "ours"]
MCOLOR = {"native": "#8C8C8C", "dgr": "#2C6FB0", "ours": "#C0392B"}
MLAB = {"native": "native CUGR2", "dgr": "baseline DGR", "ours": "ours"}


def darker(hexc, f=0.62):
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return (r*f/255, g*f/255, b*f/255)


def bubble(overflow):
    return 26.0 + 30.0 * math.sqrt(max(overflow, 0))


def load():
    data = {c: {m: set() for m in METHODS} for c in CHIPS}
    for r in csv.DictReader(open(CSV)):
        c = r["benchmark"].replace("_metal5", ""); m = r["method"]
        if c in data and m in data[c]:
            data[c][m].add((int(r["via_count"]) / 1e6,
                            int(r["wirelength"]) / 1e6, int(r["overflow"])))
    if os.path.isfile(ISO):
        for r in csv.DictReader(open(ISO)):
            if r["method"] in ("native", "dgr"):
                c = r["chip"].replace("_metal5", "")
                if c in data:
                    data[c][r["method"]].add(
                        (int(r["via_count"]) / 1e6,
                         int(r["wirelength"]) / 1e6, int(r["overflow"])))
    return {c: {m: sorted(v) for m, v in d.items()} for c, d in data.items()}


def frontier(pts):
    """lower-left non-dominated set in (via=x, wl=y); returns sorted by x."""
    s = sorted(pts, key=lambda p: (p[0], p[1]))
    out, best = [], math.inf
    for p in s:
        if p[1] <= best + 1e-9:
            out.append(p); best = p[1]
    return out


def staircase(front):
    """x,y arrays that draw a left-continuous staircase through the frontier."""
    xs, ys = [], []
    for i, p in enumerate(front):
        if i:
            xs.append(p[0]); ys.append(front[i-1][1])   # horizontal step
        xs.append(p[0]); ys.append(p[1])
    return xs, ys


def draw(ax, dc, fs=11):
    for m in METHODS:
        pts = dc[m]
        if not pts:
            continue
        fr = frontier(pts)
        frset = set(fr)
        # faded non-frontier points
        nx = [p[0] for p in pts if p not in frset]
        ny = [p[1] for p in pts if p not in frset]
        ax.scatter(nx, ny, s=22, color=MCOLOR[m], alpha=0.20,
                   edgecolors="none", zorder=2)
        # frontier staircase + solid frontier points sized by overflow
        sx, sy = staircase(fr)
        ax.plot(sx, sy, "-", color=MCOLOR[m], lw=1.8, alpha=0.9, zorder=3)
        ax.scatter([p[0] for p in fr], [p[1] for p in fr],
                   s=[bubble(p[2]) for p in fr], color=MCOLOR[m], alpha=0.85,
                   edgecolors=darker(MCOLOR[m]), linewidths=0.9, zorder=4)
    ax.set_xlabel("via count (M)", fontsize=fs)
    ax.set_ylabel("wirelength (M)", fontsize=fs)
    ax.tick_params(labelsize=fs-1)


def mhandles():
    return [Line2D([], [], marker="o", color=MCOLOR[m], lw=2, markersize=9,
                   markeredgecolor=darker(MCOLOR[m]), label=MLAB[m])
            for m in METHODS]


def shandles():
    return [Line2D([], [], marker="o", ls="", color="#B0B0B0",
                   markeredgecolor="#606060", markersize=math.sqrt(bubble(o)),
                   label=f"overflow = {o}") for o in (0, 10, 30, 70)]


def main():
    sns.set_style("whitegrid"); sns.set_context("notebook")
    data = load()
    # 2x3
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.8))
    for ax, c in zip(axes.ravel(), CHIPS):
        draw(ax, data[c]); ax.set_title(c, fontsize=13, weight="bold")
    fig.suptitle("Pareto frontiers across methods — via vs. wirelength "
                 "(lower-left = better; bubble ∝ overflow)",
                 fontsize=15, weight="bold")
    fig.legend(handles=mhandles(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.955), frameon=False, fontsize=12)
    fig.legend(handles=shandles(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.004), frameon=True, fontsize=11,
               title="bubble size reference", title_fontsize=11)
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    for ext in (".pdf", "-1.png"):
        fig.savefig(os.path.join(FIG, "pareto_all"+ext), dpi=150,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("-> pareto_all")
    # per chip
    for c in CHIPS:
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        draw(ax, data[c], fs=12)
        ax.set_title(f"{c} — Pareto frontier (lower-left is better)",
                     fontsize=13, weight="bold")
        ax.legend(handles=mhandles()+shandles(), fontsize=9, ncol=2,
                  loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"pareto_{c}-1.png"), dpi=150,
                    bbox_inches="tight", facecolor="white")
        plt.close(fig)
    print("-> pareto_<chip> x6")


if __name__ == "__main__":
    main()
