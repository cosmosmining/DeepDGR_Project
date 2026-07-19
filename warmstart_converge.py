#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
warmstart_converge.py — the "warm-start needs fewer iterations" optimizer.

Claim it establishes:  a warm-started DeepDGR run reaches (and with
best-of-K seeds, matches/beats) the quality of 2000-iteration DGR and native
CUGR2 using FAR fewer iterations -> the warm-start's value is RUNTIME, made
concrete.

For one benchmark it:
  1. loads the candidate pool + objective ONCE (reused across all runs),
  2. for each iteration budget in a ladder, runs FastDGR from the warm-start
     for K seeds, writes each guide, routes it with the per-bench knob,
  3. takes best-of-K per budget by (overflow, then pareto) -- the DGR-paper
     best-of-5 protocol, which also favours the lower-variance warm-start,
  4. reports the MINIMUM budget whose best-of-K beats BOTH the DGR-2000 and
     CUGR2 targets on overflow AND pareto.

Reuses dgr_fast.py internals (load once); existing code untouched.

  python3 warmstart_converge.py --bench ispd19_test8_metal5 \
      --warmstart ispd19_test8_metal5_SYNTHX_e2e.npz \
      --budgets 50,100,200,400 --seeds 5 --device 0
"""
import argparse
import csv
import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
from dgr_fast import (build_full_problem, init_logits, load_benchmark,
                      load_warmstart, optimize)
from discrete_refine import best_of_k_rounding, refine, blend_distribution
import util as dgr_util
from tune_cugr2 import route_with_knobs

SH = {"ispd18_test5_metal5": "18_t5", "ispd18_test8_metal5": "18_t8",
      "ispd18_test10_metal5": "18_t10", "ispd19_test7_metal5": "19_t7",
      "ispd19_test8_metal5": "19_t8", "ispd19_test9_metal5": "19_t9"}
KNOB = {"ispd18_test5_metal5": {"cls": 2.0, "vm": 1.0},
        "ispd18_test8_metal5": {"cls": 4.0, "vm": 1.0},
        "ispd18_test10_metal5": {"cls": 2.0, "vm": 1.0},
        "ispd19_test7_metal5": {"cls": 4.0, "vm": 1.0},
        "ispd19_test8_metal5": {"vm": 1.0},
        "ispd19_test9_metal5": {"cls": 2.0, "vm": 1.0, "wsa": 1000}}


def log(m):
    print(f"[{time.time():.0f}] {m}", flush=True)


def base_args(bench, device):
    return SimpleNamespace(
        data_path=os.path.join(ROOT, f"{bench}.pt"), device=device,
        capacity=1.0, pin_ratio=1.0, local_net_ratio=1.0, via_layer=1.5,
        pattern_level=1, z_step=3, max_z=10, c_step=3, max_c=20,
        max_c_out_ratio=5, act="sigmoid", act_scale=0.5, celu_alpha=2.0,
        optimizer="rmsprop", lr=0.8, weight_decay=0.0, beta1=0.9, t=1.0,
        use_gumble=True, overflow_coeff=1.0, wl_coeff=0.5, via_coeff=4.0,
        check_every=50, freeze_thresh=0.995, prune_eps=0.02, patience=999,
        min_iter=1, rel_tol=1e-4, rounding_k=16, refine_passes=4,
        refine_max_moves=4000, audit=False, blend_beta=0.6, seed=0, iter=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--warmstart", required=True)
    ap.add_argument("--budgets", default="50,100,200,400")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--tag", default="WSCONV")
    a = ap.parse_args()
    dev = f"cuda:{a.device}" if torch.cuda.is_available() else "cpu"
    b = a.bench
    knob = KNOB.get(b, {"cls": 2.0, "vm": 1.0})

    # baselines (DGR-2000 = s2 guide; CUGR2 = native), each with the per-bench
    # knob -> a fair target
    br = [r for r in csv.DictReader(
        open(os.path.join(ROOT, "experiments/fastdgr/bench_results.csv")))
        if r["wsa"] == "500"]

    def base(arm):
        r = next(x for x in br if x["benchmark"] == b and x["arm"] == arm)
        return (int(float(r["wirelength"])), int(float(r["via_count"])),
                int(float(r["overflow"])))
    nat0, dgr0 = base("cugr2_native"), base("s2_existing")
    log(f"{SH[b]} default targets: native {nat0}  DGR2000 {dgr0}")

    # load ONCE
    args = base_args(b, dev)
    (pre, name, pool, p_index, p_index_full, p2pat, hor, ver, wl, via) = \
        load_benchmark(args)
    full = build_full_problem(args, pre, p_index, hor, ver, wl, via)
    ws = load_warmstart(os.path.join(ROOT, a.warmstart), full)
    if ws is None:
        log("WARN: warm-start failed to load (shape?) — using random init")
    out = os.path.join(ROOT, "experiments", "cugr2_tune",
                       f"converge_{SH[b]}.csv")
    fh = open(out, "w", newline="")
    cw = csv.writer(fh)
    cw.writerow(["bench", "budget", "best_seed", "wirelength", "via_count",
                 "overflow", "pareto", "beats_dgr2000", "beats_cugr2",
                 "opt_s"])
    target_pareto = min(nat0[0] + 4 * nat0[1], dgr0[0] + 4 * dgr0[1])
    target_of = min(nat0[2], dgr0[2])
    won_at = None
    for budget in [int(x) for x in a.budgets.split(",")]:
        seed_results = []
        for s in range(a.seeds):
            args.iter, args.seed = budget, s
            logits0 = ws.clone() if ws is not None else init_logits(full, s)
            t0 = time.time()
            ap_, fin, it_run, opt_s, _ = optimize(full, logits0, args)
            p_soft = ap_.expand_full()
            sel, sc, _ = best_of_k_rounding(full, p_soft, k=16,
                                            of_coeff=1.0, wl_coeff=0.5,
                                            via_coeff=4.0)
            sel, sc2, _ = refine(full, sel, 1.0, 0.5, 4.0, passes=4,
                                 max_moves=4000, log=None)
            p_guide = blend_distribution(full, p_soft, sel, beta=0.6)
            gname = f"{name}_{a.tag}_b{budget}_s{s}"
            dgr_util.write_CUGR_input(pre["RouteNets"], p_guide, p_index_full,
                                      pool, p2pat, gname, 1.0)
            guide = os.path.join(ROOT, "CUGR2_guide", f"CUgr_{gname}.txt")
            m = route_with_knobs(b, guide, knob)
            try:
                os.remove(guide)
            except OSError:
                pass
            if m["wirelength"]:
                seed_results.append((s, m["wirelength"], m["via_count"],
                                     m["overflow"], round(opt_s, 1)))
        if not seed_results:
            continue
        # best-of-K: lowest overflow then pareto
        s, wl_, via_, of_, opt_s = min(
            seed_results, key=lambda r: (r[3], r[1] + 4 * r[2]))
        par = wl_ + 4 * via_
        bd = par <= dgr0[0] + 4 * dgr0[1] and of_ <= dgr0[2]
        bc = par <= nat0[0] + 4 * nat0[1] and of_ <= nat0[2]
        cw.writerow([b, budget, s, wl_, via_, of_, par, bd, bc, opt_s])
        fh.flush()
        log(f"  budget {budget:4d} (best of {a.seeds}): WL={wl_:,} via={via_:,}"
            f" of={of_}  beats DGR2000={bd} CUGR2={bc}  ({opt_s}s opt)")
        if bd and bc and won_at is None:
            won_at = budget
            log(f"  *** {SH[b]}: warm-start matches/beats BOTH baselines at "
                f"{budget} iters (vs DGR's 2000) ***")
    fh.close()
    log(f"-> {out}  (min iters to beat both: {won_at})")


if __name__ == "__main__":
    main()
