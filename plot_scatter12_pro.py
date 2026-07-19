#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_scatter12_pro.py — improved 2x3 bubble/trade-off plot from the REAL
measured sweep (RESULTS/scatter12.csv), not dummy data. Applies the
readability upgrades: seaborn whitegrid, per-method trade-off line (points
SORTED BY VIA), translucent bubbles with darker edges, and SMART sqrt bubble
scaling so overflow outliers (e.g. ispd19_test9) don't swamp a subplot.
Unified method legend + a separate overflow size-reference legend, both
outside the axes. Per-subplot autoscale (chips differ by 10x in magnitude).

Reads RESULTS/scatter12.csv (columns benchmark,method,cfg,wirelength,
via_count,overflow); dedupes to one row per (chip,method,cfg). NEW file;
does not modify route_scatter12.py.

  python3 plot_scatter12_pro.py                 # all + per-chip
  python3 plot_scatter12_pro.py --pareto        # line = lower-left frontier
"""
import argparse
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
CHIPS = ["ispd18_test5", "ispd18_test8", "ispd18_test10",
         "ispd19_test7", "ispd19_test8", "ispd19_test9"]
METHODS = ["native", "dgr", "ours"]
MCOLOR = {"native": "#8C8C8C", "dgr": "#2C6FB0", "ours": "#C0392B"}
MLAB = {"native": "native CUGR2", "dgr": "baseline DGR", "ours": "ours"}


def darker(hexc, f=0.6):
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return (r*f/255, g*f/255, b*f/255)


def bubble_size(overflow):
    """Smart scaling: area ~ sqrt(overflow) so big outliers stay legible."""
    return 28.0 + 30.0 * math.sqrt(max(overflow, 0))


ISO = os.path.join(ROOT, "experiments", "cugr2_tune", "iso_clean.csv")


def _add(data, seen, c, m, via, wl, of):
    """dedupe by rounded (via,wl) so the two sources don't double-count."""
    if c not in data or m not in METHODS:
        return
    k = (c, m, round(via, 4), round(wl, 4))
    if k in seen:
        return
    seen.add(k)
    data[c][m].append((via, wl, of))


def load():
    seen, data = set(), {c: {m: [] for m in METHODS} for c in CHIPS}
    # primary: the 12-config sweep (all of "ours", part of native/dgr)
    for r in csv.DictReader(open(CSV)):
        _add(data, seen, r["benchmark"].replace("_metal5", ""), r["method"],
             int(r["via_count"]) / 1e6, int(r["wirelength"]) / 1e6,
             int(r["overflow"]))
    # merge the remaining native/dgr operating points from iso_clean
    # (same isolated protocol); together they complete 12 points each
    if os.path.isfile(ISO):
        for r in csv.DictReader(open(ISO)):
            if r["method"] in ("native", "dgr"):
                _add(data, seen, r["chip"].replace("_metal5", ""),
                     r["method"], int(r["via_count"]) / 1e6,
                     int(r["wirelength"]) / 1e6, int(r["overflow"]))
    return data


def pareto_front(pts):
    """lower-left non-dominated set in (via, WL), sorted by via."""
    s = sorted(pts, key=lambda p: (p[0], p[1]))
    front, best_y = [], math.inf
    for p in s:
        if p[1] <= best_y + 1e-9:
            front.append(p)
            best_y = p[1]
    return front


def draw(ax, data_c, use_pareto, fs=11):
    for m in METHODS:
        pts = data_c[m]
        if not pts:
            continue
        line_pts = (pareto_front(pts) if use_pareto
                    else sorted(pts, key=lambda p: p[0]))
        lx = [p[0] for p in line_pts]
        ly = [p[1] for p in line_pts]
        ax.plot(lx, ly, "-", color=MCOLOR[m], lw=1.6, alpha=0.75, zorder=2)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   s=[bubble_size(p[2]) for p in pts], color=MCOLOR[m],
                   alpha=0.55, edgecolors=darker(MCOLOR[m]), linewidths=0.8,
                   zorder=3)
    ax.set_xlabel("via count (M)", fontsize=fs)
    ax.set_ylabel("wirelength (M)", fontsize=fs)
    ax.tick_params(labelsize=fs-1)


def method_handles():
    return [Line2D([], [], marker="o", color=MCOLOR[m], lw=2,
                   markersize=9, markeredgecolor=darker(MCOLOR[m]),
                   label=MLAB[m]) for m in METHODS]


def size_handles():
    hs = []
    for o in (0, 10, 30, 70):
        hs.append(Line2D([], [], marker="o", ls="", color="#B0B0B0",
                         markeredgecolor="#606060",
                         markersize=math.sqrt(bubble_size(o)),
                         label=f"overflow = {o}"))
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pareto", action="store_true",
                    help="line = lower-left Pareto frontier (else sorted-by-via)")
    a = ap.parse_args()
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    data = load()
    tag = "pareto" if a.pareto else ""
    lineword = ("lower-left trade-off frontier" if a.pareto
                else "trade-off trajectory (points sorted by via)")

    # ── 2x3 grid ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6))
    for ax, c in zip(axes.ravel(), CHIPS):
        draw(ax, data[c], a.pareto, fs=11)
        ax.set_title(c, fontsize=13, weight="bold")
    fig.suptitle("Router parameter sweep — via vs. wirelength, bubble area "
                 f"∝ overflow;  line = {lineword}", fontsize=15,
                 weight="bold")
    fig.legend(handles=method_handles(), loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.955), frameon=False, fontsize=12)
    fig.legend(handles=size_handles(), loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.005), frameon=True, fontsize=11,
               title="bubble size reference", title_fontsize=11)
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    stem = os.path.join(FIG, f"scatter12pro{('_'+tag) if tag else ''}_all")
    for ext in (".pdf", "-1.png"):
        fig.savefig(stem + ext, dpi=150, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print("->", os.path.basename(stem))

    # ── per chip ───────────────────────────────────────────────────────────
    for c in CHIPS:
        fig, ax = plt.subplots(figsize=(6.6, 5.2))
        draw(ax, data[c], a.pareto, fs=12)
        ax.set_title(f"{c}  —  bubble ∝ overflow", fontsize=13,
                     weight="bold")
        ax.legend(handles=method_handles() + size_handles(), frameon=True,
                  fontsize=9, loc="best", ncol=2)
        fig.tight_layout()
        st = os.path.join(FIG, f"scatter12pro_{c}")
        fig.savefig(st + "-1.png", dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
    print("-> scatter12pro_<chip> x6")


if __name__ == "__main__":
    main()
