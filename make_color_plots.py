#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_color_plots.py — re-render the CSV-backed figures with a restrained,
professional color palette (muted steel blue / firebrick, light grid, clean
spines). Writes PDF+PNG with the SAME filenames used by the report, beamer
deck, and PPTX, into RESULTS/figs/, so every document picks up color on
rebuild. Pure numpy/matplotlib (no torch import). NEW file.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "RESULTS", "figs")
os.makedirs(FIG, exist_ok=True)

WARM = "#C0392B"      # firebrick
COLD = "#2C5F8A"      # steel blue
ACC = "#7B9E4E"       # muted olive (third series if needed)

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5})


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, stem + ".pdf"))
    fig.savefig(os.path.join(FIG, stem + "-1.png"), dpi=150)
    plt.close(fig)
    print("  ->", stem, "(pdf + png)")


# ── overflow-vs-iteration curves ────────────────────────────────────────────
for chip in ("18_t5", "19_t8"):
    p = os.path.join(ROOT, "RESULTS", f"overflow_curve_{chip}.csv")
    if not os.path.isfile(p):
        continue
    rows = list(csv.DictReader(open(p)))
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for label, color, sty, mk in (("warm", WARM, "-", "o"),
                                  ("cold", COLD, "--", "s")):
        xs = [int(r["iter"]) for r in rows if r["init"] == label]
        ys = [float(r["overflow_units"]) for r in rows if r["init"] == label]
        if xs:
            ax.plot(xs, ys, sty, color=color, marker=mk, mfc="white",
                    ms=4.5, lw=1.6, label=f"{label}-start")
    ax.set_xlabel("FastDGR iteration")
    ax.set_ylabel("overflow (discrete units)")
    ax.set_title(f"{chip}: overflow vs iteration — warm vs cold", fontsize=11)
    ax.legend(frameon=False)
    save(fig, f"overflow_curve_{chip}")

# ── scaling curves ──────────────────────────────────────────────────────────
def scaling_plot(csv_path, xcol, ycol, title, stem):
    if not os.path.isfile(csv_path):
        return
    rows = list(csv.DictReader(open(csv_path)))
    xs = [int(r[xcol]) for r in rows]
    ys = [float(r[ycol]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ax.plot(xs, ys, "-", color=COLD, marker="o", mfc="white", ms=5, lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("# distinct training designs (log)")
    ax.set_ylabel("held-out objective")
    ax.set_title(title, fontsize=11)
    save(fig, stem)


scaling_plot(os.path.join(ROOT, "RESULTS", "scaling_congested.csv"),
             "n_distinct_train", "heldout_objective",
             "Scaling on congested data (fixed held-out)",
             "scaling_congested")
scaling_plot(os.path.join(ROOT, "RESULTS", "scaling_diverse.csv"),
             "n_distinct_train", "heldout_objective",
             "Scaling on certified data (fixed held-out)",
             "scaling_diverse")

# ── runtime speedup chart (new) ─────────────────────────────────────────────
chips = ["18_t5", "18_t8", "18_t10", "19_t7", "19_t8", "19_t9"]
dgr_s = [343, 485, 377, 834, 1195, 1933]
fast_s = [4.8, 5.2, 3.5, 5.7, 8.9, 10.9]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = range(len(chips))
ax.bar([i - 0.2 for i in x], dgr_s, width=0.4, color=COLD, label="DGR 2000-it")
ax.bar([i + 0.2 for i in x], fast_s, width=0.4, color=WARM, label="FastDGR")
for i, (d, f) in enumerate(zip(dgr_s, fast_s)):
    ax.text(i, d * 1.15, f"{d/f:.0f}×", ha="center", fontsize=9,
            color="#333333")
ax.set_yscale("log")
ax.set_xticks(list(x))
ax.set_xticklabels(chips)
ax.set_ylabel("guide optimize time (s, log)")
ax.set_title("Guide-generation time: DGR vs FastDGR", fontsize=11)
ax.legend(frameon=False)
save(fig, "runtime_speedup")
print("done")
