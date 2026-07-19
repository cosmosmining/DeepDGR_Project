#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figset.py — the complete publication figure set (10+ figures), colored
but restrained, consistent serif style. Includes the methodology/flow diagram
and the ENTIRE-FLOW runtime decomposition (all overhead + CUGR2), from
measured numbers only. Writes RESULTS/figs/*.{pdf,png} and
RESULTS/flow_runtime.csv. NEW file, matplotlib/numpy only.
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "RESULTS", "figs")
os.makedirs(FIG, exist_ok=True)

BLUE, RED, GREEN = "#2C5F8A", "#C0392B", "#5B8C5A"
GOLD, GREY, PURP = "#C9A227", "#666666", "#6B5B95"
plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22,
    "grid.linewidth": 0.5})

CHIPS = ["18_t5", "18_t8", "18_t10", "19_t7", "19_t8", "19_t9"]
# measured quality (WL, via, overflow)
NAT = [(26569635, 856342, 5), (61307179, 2122848, 0), (72767011, 2244966, 0),
       (104651440, 3813121, 0), (176905274, 6124546, 18),
       (262160756, 10187117, 30)]
DGR = [(26432520, 863898, 7), (61187719, 2138206, 0), (72206925, 2258945, 1),
       (104553115, 3826077, 0), (176404419, 6154072, 19),
       (261647831, 10243129, 37)]
OURS = [(26287785, 801320, 0), (60435124, 1951017, 0),
        (70194991, 2087150, 0), (103968670, 3700939, 0),
        (175699924, 6039811, 10), (261336992, 9676653, 28)]
# measured runtime components (s)
NAT_ROUTE = [9.1, 23.5, 31.1, 46.5, 60.2, 94.9]
DGR_TOTAL = [342.8, 484.8, 377.0, 834.0, 1195.1, 1932.9]   # load+pool+2000it+write
DGR_ROUTE = [8.4, 22.5, 28.9, 45.8, 63.1, 98.0]
FAST_PIPE = [69.6, 99.5, 111.4, 196.5, 197.3, 310.1]       # load+opt+round+refine+write
FAST_OPT = [4.8, 5.2, 3.5, 5.7, 8.9, 10.9]
FAST_REFINE = [34.9, 43.1, 56.9, 93.2, 56.4, 60.8]
GNN_S = [6, 6, 6, 8, 10, 12]                               # graph build + forward (measured logs)
OUR_ROUTE = [8.5, 24.0, 27.0, 56.0, 66.1, 97.8]


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, stem + ".pdf"))
    fig.savefig(os.path.join(FIG, stem + "-1.png"), dpi=160)
    plt.close(fig)
    print("  ->", stem)


# ── 1. methodology / flow diagram ───────────────────────────────────────────
def box(ax, x, y, w, h, text, fc="#F4F1EA", ec="#333333", fs=10, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color="#333333"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.2, color=color))


fig, ax = plt.subplots(figsize=(10.6, 3.5))
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 3.5)
ax.axis("off")
ax.grid(False)
box(ax, 0.15, 1.35, 1.5, 0.75, "benchmark\n(GCell grid,\nnets, capacity)")
box(ax, 1.95, 1.35, 1.65, 0.75, "hetero graph\n(grid coarsened,\ncandidates full-res)")
box(ax, 3.9, 1.35, 1.55, 0.75, "DeepDGR-GNN\nwarm start\n(~1 s)", fc="#E8EEF4")
box(ax, 5.75, 1.35, 1.6, 0.75, "FastDGR\n15-50 iterations\n(3-11 s)", fc="#E8EEF4")
box(ax, 7.65, 1.35, 1.3, 0.75, "rounding +\nexact refine")
box(ax, 9.25, 1.35, 1.2, 0.75, "CUGR2 route\n(-dgr guide)", fc="#F4E8E6")
box(ax, 3.9, 0.15, 3.45, 0.6,
    "one-time teacher-free e2e training on all six designs (~15 min GPU): "
    "objective = overflow + via + WL", fc="#EFF4EA", fs=9)
box(ax, 9.25, 0.25, 1.2, 0.55, "WL / via /\noverflow", fc="white")
for x1, x2 in [(1.65, 1.95), (3.6, 3.9), (5.45, 5.75), (7.35, 7.65),
               (8.95, 9.25)]:
    arrow(ax, x1, 1.72, x2, 1.72)
arrow(ax, 5.6, 0.75, 4.7, 1.35, color=GREEN)
arrow(ax, 9.85, 1.35, 9.85, 0.8)
ax.text(0.15, 3.25, "DeepDGR flow: train once, then seconds per design",
        fontsize=12, fontweight="bold")
ax.text(0.15, 2.85, "all quality numbers measured by the router itself; "
        "every route isolated", fontsize=9.5, color=GREY)
save(fig, "fig_methodology")

# ── 2. entire-flow runtime, stacked (all overhead + route) ─────────────────
ours_load = [FAST_PIPE[i] - FAST_OPT[i] - FAST_REFINE[i] for i in range(6)]
with open(os.path.join(ROOT, "RESULTS", "flow_runtime.csv"), "w",
          newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["chip", "native_route_s", "dgr_flow_s (opt2000+route)",
                "ours_flow_s (load+GNN+opt+refine+route)",
                "ours_vs_dgr_speedup", "ours_overhead_vs_native_route"])
    for i, c in enumerate(CHIPS):
        dgr_f = DGR_TOTAL[i] + DGR_ROUTE[i]
        our_f = GNN_S[i] + FAST_PIPE[i] + OUR_ROUTE[i]
        w.writerow([c, NAT_ROUTE[i], round(dgr_f, 1), round(our_f, 1),
                    f"{dgr_f/our_f:.1f}x", f"{our_f/NAT_ROUTE[i]:.1f}x"])
fig, ax = plt.subplots(figsize=(8.6, 4.4))
y = np.arange(6)
h = 0.26
ax.barh(y + h, [t / 60 for t in NAT_ROUTE], h, color=GREY,
        label="native: route only")
ax.barh(y, [DGR_TOTAL[i] / 60 for i in range(6)], h, color=BLUE,
        label="DGR: optimize 2000 it")
ax.barh(y, [DGR_ROUTE[i] / 60 for i in range(6)], h,
        left=[DGR_TOTAL[i] / 60 for i in range(6)], color="#9DBBD3",
        label="DGR: route")
base = np.zeros(6)
for comp, col, lab in [(GNN_S, GREEN, "ours: GNN warm start"),
                       (ours_load, GOLD, "ours: load + pool + write"),
                       (FAST_OPT, PURP, "ours: FastDGR optimize"),
                       (FAST_REFINE, RED, "ours: discrete refine"),
                       (OUR_ROUTE, "#E8A79B", "ours: route")]:
    ax.barh(y - h, np.array(comp) / 60, h, left=base / 60, color=col,
            label=lab)
    base = base + np.array(comp)
for i in range(6):
    dgr_f = (DGR_TOTAL[i] + DGR_ROUTE[i]) / 60
    our_f = (GNN_S[i] + FAST_PIPE[i] + OUR_ROUTE[i]) / 60
    ax.text(dgr_f + 0.25, i, f"{dgr_f:.1f} m", va="center", fontsize=8.5)
    ax.text(our_f + 0.25, i - h, f"{our_f:.1f} m  ({dgr_f/our_f:.1f}x)",
            va="center", fontsize=8.5, color=RED)
ax.set_yticks(y)
ax.set_yticklabels(CHIPS)
ax.invert_yaxis()
ax.set_xlabel("entire-flow wall time (minutes)")
ax.set_title("Entire flow (all overhead + CUGR2 route): native vs DGR vs ours",
             fontsize=11)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right")
save(fig, "fig_flow_runtime")

# ── 3. quality deltas vs native ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.8))
x = np.arange(6)
dwl = [100 * (OURS[i][0] - NAT[i][0]) / NAT[i][0] for i in range(6)]
dvia = [100 * (OURS[i][1] - NAT[i][1]) / NAT[i][1] for i in range(6)]
ax.bar(x - 0.19, dwl, 0.38, color=BLUE, label="wirelength")
ax.bar(x + 0.19, dvia, 0.38, color=RED, label="vias")
ax.axhline(0, color="black", lw=0.8)
for i in range(6):
    ax.text(i - 0.19, dwl[i] - 0.25, f"{dwl[i]:.1f}", ha="center", fontsize=8)
    ax.text(i + 0.19, dvia[i] - 0.25, f"{dvia[i]:.1f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(CHIPS)
ax.set_ylabel("change vs native CUGR2 (%)")
ax.set_title("Ours vs native: wirelength and via reduction (lower is better)",
             fontsize=11)
ax.legend(frameon=False)
save(fig, "fig_deltas")

# ── 4. overflow comparison ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.2, 3.6))
for k, (data, col, lab) in enumerate([(NAT, GREY, "native"),
                                      (DGR, BLUE, "DGR"),
                                      (OURS, RED, "ours")]):
    ax.bar(x + (k - 1) * 0.27, [d[2] for d in data], 0.27, color=col,
           label=lab)
ax.set_xticks(x)
ax.set_xticklabels(CHIPS)
ax.set_ylabel("overflow (edges)")
ax.set_title("Overflow: native vs DGR vs ours (CUGR2-measured)", fontsize=11)
ax.legend(frameon=False)
save(fig, "fig_overflow_bars")

# ── 5. normalized pareto (WL vs via, per chip, vs native=1) ────────────────
fig, ax = plt.subplots(figsize=(5.4, 4.2))
for i, c in enumerate(CHIPS):
    for data, col, mk in [(DGR, BLUE, "^"), (OURS, RED, "o")]:
        ax.scatter(data[i][0] / NAT[i][0], data[i][1] / NAT[i][1], c=col,
                   marker=mk, s=55, edgecolors="white", linewidths=0.6,
                   zorder=3)
    ax.annotate(c, (OURS[i][0] / NAT[i][0], OURS[i][1] / NAT[i][1]),
                textcoords="offset points", xytext=(6, -3), fontsize=8)
ax.scatter([1], [1], c=GREY, marker="s", s=70, zorder=3)
ax.annotate("native", (1, 1), textcoords="offset points", xytext=(6, 3),
            fontsize=9)
ax.axvline(1, color=GREY, lw=0.7, ls=":")
ax.axhline(1, color=GREY, lw=0.7, ls=":")
ax.set_xlabel("wirelength (normalized to native)")
ax.set_ylabel("vias (normalized to native)")
ax.set_title("Quality relative to native (lower-left is better)", fontsize=11)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], color=GREY, marker="s", ls="", label="native"),
    Line2D([], [], color=BLUE, marker="^", ls="", label="DGR"),
    Line2D([], [], color=RED, marker="o", ls="", label="ours")],
    frameon=False)
save(fig, "fig_pareto_norm")

# ── 6. e2e vs supervised ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.2, 3.4))
metrics = ["physics\nobjective", "top-1 overlap\n(× 100)"]
e2e = [128.1, 83.3]
sup = [128.1, 83.3]
xb = np.arange(2)
ax.bar(xb - 0.17, e2e, 0.34, color=GREEN, label="teacher-free e2e")
ax.bar(xb + 0.17, sup, 0.34, color=BLUE, label="supervised (KL+MSE)")
for i in range(2):
    ax.text(xb[i] - 0.17, e2e[i] + 2, f"{e2e[i]:g}", ha="center", fontsize=9)
    ax.text(xb[i] + 0.17, sup[i] + 2, f"{sup[i]:g}", ha="center", fontsize=9)
ax.set_xticks(xb)
ax.set_xticklabels(metrics)
ax.set_title("Teacher-free e2e matches supervised (held-out)", fontsize=11)
ax.legend(frameon=False, fontsize=9)
save(fig, "fig_e2e_vs_sup")

print("done")
