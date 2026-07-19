#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overflow_curve.py — accurate OVERFLOW-vs-ITERATION curves to decide the
early-stop point for DeepDGR / FastDGR.

For a benchmark it runs FastDGR from a WARM start AND from a COLD start, and at
a ladder of iteration budgets discretely ROUNDS the current distribution and
measures the EXACT overflow (total demand-over-capacity AND # overflowed edges)
— not the soft objective.  Plots overflow vs iteration (warm solid, cold
dashed) and reports the iteration at which the warm curve is within tol of its
final value (the "you only need N iterations" point).  Optionally also runs the
original DGR (main_stochastic) warm-start at the same budgets.

Reuses dgr_fast / discrete_refine / congestion_viz READ-ONLY; NEW file.

  python3 overflow_curve.py --bench ispd19_test8_metal5 \
      --warmstart ispd19_test8_metal5_SYNTHX_e2e.npz --device 0
"""
import argparse
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
from dgr_fast import (build_full_problem, init_logits, load_benchmark,
                      load_warmstart, optimize)
from discrete_refine import best_of_k_rounding
import congestion_viz as CV
from warmstart_converge import base_args, SH


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


@torch.no_grad()
def discrete_overflow(full, p_soft):
    """round to a discrete routing; return EXACT (total overflow units,
    max per-edge overflow) from the discrete scorer (score_state)."""
    _, score, _ = best_of_k_rounding(full, p_soft, k=16, of_coeff=1.0,
                                     wl_coeff=0.5, via_coeff=4.0)
    return float(score["overflow_units"]), float(score["max_overflow"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--warmstart", required=True)
    ap.add_argument("--budgets", default="5,10,20,30,50,75,100,150,200,400")
    ap.add_argument("--tol", type=float, default=0.02,
                    help="warm 'converged' when within tol*range of final")
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.device}" if torch.cuda.is_available() else "cpu"
    b = a.bench
    args = base_args(b, dev)
    (pre, name, pool, p_index, p_index_full, p2pat, hor, ver, wl, via) = \
        load_benchmark(args)
    full = build_full_problem(args, pre, p_index, hor, ver, wl, via)
    ws = load_warmstart(os.path.join(ROOT, a.warmstart), full)
    budgets = [int(x) for x in a.budgets.split(",")]
    out = os.path.join(ROOT, "validation_out", f"overflow_curve_{SH.get(b,b)}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["init", "iter", "overflow_units", "max_overflow", "opt_s"])
    if ws is None:
        log("WARN: warm-start failed to load (shape?) — 'warm' falls back to random")
    curves = {}
    for label, init0 in [("warm", ws), ("cold", None)]:
        init0 = init0 if init0 is not None else init_logits(full, 0)
        ov_iter = []
        for B in budgets:
            t0 = time.time()
            ap_, *_ = optimize(full, init0.clone(), _argset(args, B))
            tot, mx = discrete_overflow(full, ap_.expand_full())
            dt = round(time.time() - t0, 2)
            w.writerow([label, B, round(tot, 1), round(mx, 2), dt])
            fh.flush()
            ov_iter.append((B, tot, mx))
            log(f"  {label:4} iter {B:4d}: overflow_units={tot:.0f} "
                f"max={mx:.1f}  ({dt}s)")
        curves[label] = ov_iter
    fh.close()
    # stop-iteration: first budget where warm overflow within tol of warm-final
    if "warm" in curves:
        wc = curves["warm"]
        final = wc[-1][1]
        rng = max(c[1] for c in wc) - final
        stop = next((B for B, cnt, _ in wc if cnt - final <= a.tol * max(rng, 1)),
                    wc[-1][0])
        log(f"=> WARM-START reaches within {a.tol*100:.0f}% of final overflow "
            f"({final} edges) by iter {stop}")
        if "cold" in curves:
            cc = curves["cold"]
            cstop = next((B for B, cnt, _ in cc
                          if cnt - cc[-1][1] <= a.tol * max(
                              max(c[1] for c in cc) - cc[-1][1], 1)),
                         cc[-1][0])
            log(f"   COLD-START needs iter {cstop} for the same → "
                f"warm-start saves ~{cstop - stop} iterations")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f, ax = plt.subplots(figsize=(5.5, 3.8))
        for label, sty in [("warm", "-"), ("cold", "--")]:
            if label in curves:
                xs = [c[0] for c in curves[label]]
                ys = [c[1] for c in curves[label]]
                ax.plot(xs, ys, sty, color="black", marker="o", mfc="none",
                        label=f"{label}-start")
        ax.set_xlabel("DeepDGR / FastDGR iteration")
        ax.set_ylabel("overflow (# edges, discrete)")
        ax.set_title(f"{SH.get(b,b)}: overflow vs iteration — warm vs cold")
        ax.legend(frameon=False)
        f.tight_layout()
        fig = os.path.join(ROOT, "reports", "figs", f"overflow_curve_{SH.get(b,b)}.pdf")
        os.makedirs(os.path.dirname(fig), exist_ok=True)
        f.savefig(fig)
        log(f"figure -> {fig}")
    except Exception as e:
        log(f"plot skipped: {e}")
    log(f"-> {out}")


def _argset(args, it):
    import copy
    a2 = copy.copy(args)
    a2.iter = it
    return a2


if __name__ == "__main__":
    main()
