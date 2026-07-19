#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimize_final.py — squeeze the LAST WL/via out of the all-6 GNN result, at
LOWER runtime, using only measured levers.

Evidence-driven design (all numbers from this project's own experiments):
  * overflow_curve_*.csv: warm-started FastDGR is converged by ~30-120 iters
    (flat to 250+).  So instead of ONE 600-iter run (test_gnn_all6), spend the
    same budget on an ENSEMBLE of short restarts: seeds x objective-weight
    points (default / via-priority / overflow-priority).  Gumbel noise makes
    restarts genuinely different; best-of-K favours the low-variance warm
    start (the DGR paper's own protocol).
  * push_limit/phase-7: deeper discrete refine + larger rounding K improve the
    routed result; via-priority omega counters the via tension.
  * per-bench CUGR2 knobs (pareto-opt + overflow-opt) are kept — they helped
    every method equally, applied symmetrically.

Per chip: load once -> all-6 GNN warm-start (the *_ROBUST_ws.npz emitted by the
all-6 test) -> S seeds x O omegas short FastDGR (early-stop) -> rounding_k=32 +
deep refine (scored with that omega) -> keep the best DISCRETE-scoring guide
per omega -> route each with the per-bench knob set, ISOLATED -> keep the best
(overflow, then WL+4via).  Reports quality vs native / DGR / previous-all6 and
the full runtime breakdown.  Reuses dgr_fast / discrete_refine / tune_cugr2 /
test_gnn_all6.route_iso READ-ONLY; NEW file, nothing existing changed.

  python3 optimize_final.py --device 0
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
from dgr_fast import (build_full_problem, load_benchmark, load_warmstart,
                      optimize)
from discrete_refine import best_of_k_rounding, refine, blend_distribution
import util as dgr_util
from warmstart_converge import SH, base_args
from test_gnn_all6 import route_iso, bench_pt

BENCHES = ["ispd18_test5_metal5", "ispd18_test8_metal5",
           "ispd18_test10_metal5", "ispd19_test7_metal5",
           "ispd19_test8_metal5", "ispd19_test9_metal5"]
# objective-weight points: default + via-priority + overflow-priority
OMEGAS = [("def",  1.0, 0.5, 4.0),
          ("via6", 1.0, 0.5, 6.0),
          ("of2",  2.0, 0.5, 4.0)]
# per-bench knob set: pareto-opt + overflow-opt (measured in phase 5-7)
KNOBSET = {
    "ispd18_test5_metal5":  [("cls2vm1", {"cls": 2.0, "vm": 1.0}),
                             ("ofmode_vm1", {"overflow_mode": 1, "vm": 1.0})],
    "ispd18_test8_metal5":  [("cls4vm1", {"cls": 4.0, "vm": 1.0}),
                             ("cls2vm1", {"cls": 2.0, "vm": 1.0})],
    "ispd18_test10_metal5": [("cls2vm1", {"cls": 2.0, "vm": 1.0})],
    "ispd19_test7_metal5":  [("cls4vm1", {"cls": 4.0, "vm": 1.0}),
                             ("cls2vm1", {"cls": 2.0, "vm": 1.0})],
    "ispd19_test8_metal5":  [("cls2vm1", {"cls": 2.0, "vm": 1.0}),
                             ("vm1", {"vm": 1.0})],
    "ispd19_test9_metal5":  [("cls2vm1", {"cls": 2.0, "vm": 1.0}),
                             ("cls2vm1_wsa1k",
                              {"cls": 2.0, "vm": 1.0, "wsa": 1000})],
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def short_args(args, it):
    import copy
    a2 = copy.copy(args)
    a2.iter = it
    a2.check_every = 20
    a2.patience = 3          # early-stop: plateau after 3 checks
    a2.min_iter = 40
    return a2


def baselines(b):
    br = [r for r in csv.DictReader(
        open(os.path.join(ROOT, "experiments/fastdgr/bench_results.csv")))
        if r["wsa"] == "500"]

    def base(arm):
        r = next(x for x in br if x["benchmark"] == b and x["arm"] == arm)
        return (int(float(r["wirelength"])), int(float(r["via_count"])),
                int(float(r["overflow"])))
    prev = {}
    p = os.path.join(ROOT, "experiments", "cugr2_tune", "test_all6.csv")
    if os.path.isfile(p):
        for r in csv.DictReader(open(p)):
            prev[r["benchmark"]] = (int(r["wirelength"]), int(r["via_count"]),
                                    int(r["overflow"]))
    return base("cugr2_native"), base("s2_existing"), prev.get(b)


def dominates(c, t):
    return (c[0] <= t[0] and c[1] <= t[1] and c[2] <= t[2]
            and (c[0] < t[0] or c[1] < t[1] or c[2] < t[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--iter", type=int, default=160)
    ap.add_argument("--rounding_k", type=int, default=32)
    ap.add_argument("--refine_passes", type=int, default=6)
    ap.add_argument("--refine_moves", type=int, default=12000)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    dev = f"cuda:{a.device}" if torch.cuda.is_available() else "cpu"
    out = os.path.join(ROOT, "RESULTS", "final_optimized.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["chip", "wirelength", "via_count", "overflow", "knob", "omega",
                "seed", "opt_total_s", "route_s", "beats_native", "beats_dgr",
                "beats_prev_all6", "native", "dgr", "prev_all6"])
    for b in a.benches.split(","):
        nat, dgr, prev = baselines(b)
        pt = bench_pt(b)
        ws_np = os.path.join(ROOT, f"{b}_ROBUST_ws.npz")
        if pt is None or not os.path.isfile(ws_np):
            log(f"{SH.get(b,b)}: missing pt/warmstart — skip"); continue
        args = base_args(b, dev)
        args.data_path = pt
        t_load = time.time()
        (pre, name, pool, p_index, p_index_full, p2pat, hor, ver, wl, via) = \
            load_benchmark(args)
        full = build_full_problem(args, pre, p_index, hor, ver, wl, via)
        ws = load_warmstart(ws_np, full)
        log(f"{SH.get(b,b)}: loaded in {time.time()-t_load:.0f}s; ensemble "
            f"{a.seeds} seeds x {len(OMEGAS)} omegas x {a.iter} iters "
            f"(early-stop)")
        t_opt = time.time()
        best_per_omega = {}
        for oname, ofc, wlc, viac in OMEGAS:
            best = None
            for s in range(a.seeds):
                args.seed = s
                args.overflow_coeff, args.wl_coeff, args.via_coeff = \
                    ofc, wlc, viac
                init = (ws.clone() if ws is not None else None)
                if init is None:
                    from dgr_fast import init_logits
                    init = init_logits(full, s)
                ap_, *_ = optimize(full, init, short_args(args, a.iter))
                p_soft = ap_.expand_full()
                sel, sc, _ = best_of_k_rounding(
                    full, p_soft, k=a.rounding_k, of_coeff=ofc, wl_coeff=wlc,
                    via_coeff=viac)
                sel, sc2, _ = refine(full, sel, ofc, wlc, viac,
                                     passes=a.refine_passes,
                                     max_moves=a.refine_moves, log=None)
                score = sc2["total"] if isinstance(sc2, dict) else float(sc2)
                if best is None or score < best[0]:
                    best = (score, s, p_soft, sel)
            best_per_omega[oname] = best
        opt_total = round(time.time() - t_opt, 1)
        # write + route the best guide per omega with the per-bench knob set
        cands = []
        for oname, (score, s, p_soft, sel) in best_per_omega.items():
            p_guide = blend_distribution(full, p_soft, sel, beta=0.6)
            gname = f"{name}_FINAL_{oname}"
            dgr_util.write_CUGR_input(pre["RouteNets"], p_guide, p_index_full,
                                      pool, p2pat, gname, 1.0)
            guide = os.path.join(ROOT, "CUGR2_guide", f"CUgr_{gname}.txt")
            for kname, knob in KNOBSET.get(b, [("cls2vm1",
                                                {"cls": 2.0, "vm": 1.0})]):
                t_r = time.time()
                m = route_iso(b, guide, knob)
                if not m.get("wirelength"):
                    continue
                c = (m["wirelength"], m["via_count"], m["overflow"])
                cands.append((c, kname, oname, s, round(time.time()-t_r, 1)))
                log(f"  {oname}/s{s}/{kname}: WL={c[0]:,} via={c[1]:,} "
                    f"of={c[2]}")
        if not cands:
            log(f"{SH.get(b,b)}: no successful route"); continue
        # pick: lowest overflow, then pareto
        (c, kname, oname, s, r_s) = min(
            cands, key=lambda x: (x[0][2], x[0][0] + 4 * x[0][1]))
        bn, bd = dominates(c, nat), dominates(c, dgr)
        bp = dominates(c, prev) if prev else ""
        w.writerow([SH.get(b, b), c[0], c[1], c[2], kname, oname, s,
                    opt_total, r_s, bn, bd, bp,
                    f"{nat[0]}/{nat[1]}/{nat[2]}",
                    f"{dgr[0]}/{dgr[1]}/{dgr[2]}",
                    f"{prev[0]}/{prev[1]}/{prev[2]}" if prev else ""])
        fh.flush()
        log(f"{SH.get(b,b)} BEST: WL={c[0]:,} via={c[1]:,} of={c[2]} "
            f"[{oname}/s{s}/{kname}] opt={opt_total}s route={r_s}s "
            f"beats: native={bn} dgr={bd} prev_all6={bp}")
        del full, pre, pool
        if dev != "cpu":
            torch.cuda.empty_cache()
    fh.close()
    log(f"-> {out}")


if __name__ == "__main__":
    main()
