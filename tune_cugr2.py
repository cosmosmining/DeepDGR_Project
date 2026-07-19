#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_cugr2.py — fine-tune the CUGR2 router's UNSWEPT cost knobs to optimize
routed wirelength and overflow (front-part lever; see PHASE5_PLAN.md §1).

Every prior sweep varied only (-wsa, -via_cost).  This fork's CLI
(cu-gr-2/src/global.h) also exposes, all confirmed accepted by the binary:

    -wl   weight_wire_length    (0.5)   wirelength vs everything else
    -cls  cost_logistic_slope   (1.0)   pattern-routing congestion sharpness
    -mls  maze_logistic_slope   (0.5)   stage-3 maze congestion sharpness
    -mdr  max_detour_ratio      (0.25)  stage-2 detour budget
    -tdc  target_detour_count   (20)    stage-2 detour candidates
    -vm   via_multiplier        (2.0)   via min-area demand weight
    -sort new_sort              (0/1)   net ordering (num_paths first)
    -phase2                     (1)     enable/disable detour stage
    -overflow_mode 1                    preset: wsa=1500, mdr=0.4, tdc=30

Modes:
    plan     print the job table + cost estimate, run nothing
    run      execute (resumable; skips rows already in the CSV)
    analyze  per-benchmark table vs the baseline config; writes
             experiments/cugr2_tune/TUNING_ANALYSIS.md
    smoke    one quick default route on the smallest bench (sanity, ~15 s)

Routing is CPU-only (the `route` binary) — this tool never touches a GPU.
NEW file; imports step1_utils.parse_cugr2_log read-only; does not modify
scripts/sweep_methods_params.py (its route_once hardcodes flags).

    python3 tune_cugr2.py plan
    python3 tune_cugr2.py run --benches ispd18_test5_metal5 --arms native
    python3 tune_cugr2.py analyze
"""

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from step1_utils import parse_cugr2_log                  # noqa: E402

RUN_DIR = os.path.join(ROOT, "cu-gr-2", "run")
ROUTE = os.path.join(RUN_DIR, "route")
OUT_DIR = os.path.join(ROOT, "experiments", "cugr2_tune")
OUT_CSV = os.path.join(OUT_DIR, "results.csv")

BENCHES = ["ispd18_test5_metal5", "ispd18_test8_metal5",
           "ispd18_test10_metal5", "ispd19_test7_metal5",
           "ispd19_test8_metal5", "ispd19_test9_metal5"]

# (cfg_name, knob dict) — knobs omitted = binary defaults.  One-knob-at-a-time
# around the production point (wsa=500, via=20, sort=1) so effects attribute
# cleanly; combos come later from analyze's winners.
PRESETS = [
    ("baseline",   {}),
    ("cls0.5",     {"cls": 0.5}),
    ("cls2",       {"cls": 2.0}),
    ("cls4",       {"cls": 4.0}),
    ("mls1",       {"mls": 1.0}),
    ("wl0.25",     {"wl": 0.25}),
    ("wl1",        {"wl": 1.0}),
    ("vm1",        {"vm": 1.0}),
    ("vm3",        {"vm": 3.0}),
    ("detour+",    {"mdr": 0.4, "tdc": 30}),
    ("ofmode",     {"overflow_mode": 1}),
    ("sort0",      {"sort": 0}),
    ("nophase2",   {"phase2": 0}),
    # round 2: compositions of the round-1 winners (cls2 / wl1 / vm1)
    ("cls2vm1",    {"cls": 2.0, "vm": 1.0}),
    ("cls2wl1",    {"cls": 2.0, "wl": 1.0}),
    ("wl1vm1",     {"wl": 1.0, "vm": 1.0}),
    ("cls2wl1vm1", {"cls": 2.0, "wl": 1.0, "vm": 1.0}),
]
BASE_POINT = {"wsa": 500, "via_cost": 20, "sort": 1}
EST_ROUTE_S = {"ispd18_test5_metal5": 14, "ispd18_test8_metal5": 25,
               "ispd18_test10_metal5": 30, "ispd19_test7_metal5": 47,
               "ispd19_test8_metal5": 63, "ispd19_test9_metal5": 100}

FIELDS = ["benchmark", "arm", "cfg", "knobs", "wirelength", "via_count",
          "overflow", "pareto_cost", "route_s"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def find_input(bench, ext):
    for pat in ("cu-gr-2/benchmark/test/{b}/{b}.input.{e}",
                "cu-gr-2/benchmark/{b}/{b}.input.{e}",
                "cu-gr-2/benchmark/{b}.input.{e}"):
        p = os.path.join(ROOT, pat.format(b=bench, e=ext))
        if os.path.isfile(p):
            return p
    return None


def guide_for(bench, arm):
    if arm == "native":
        return None
    if arm == "s2":
        hits = sorted(glob.glob(os.path.join(
            ROOT, "CUGR2_guide", f"CUgr_{bench}_S2_*.txt")))
        return hits[0] if hits else "MISSING"
    if arm == "fast":
        p = os.path.join(ROOT, "CUGR2_guide", f"CUgr_{bench}_FASTB_NR.txt")
        return p if os.path.isfile(p) else "MISSING"
    return "MISSING"


def route_with_knobs(bench, guide, knobs):
    """Run the route binary with arbitrary knob dict; return parsed metrics."""
    lef, deff = find_input(bench, "lef"), find_input(bench, "def")
    if not (lef and deff):
        return dict(rc=2, wirelength=None, via_count=None, overflow=None,
                    runtime_s=0.0)
    merged = dict(BASE_POINT)
    merged.update(knobs)
    scratch = tempfile.mkdtemp(prefix="ctune_",
                               dir=os.environ.get("TMPDIR", "/tmp"))
    out_guide = os.path.join(scratch, "out.guide")
    log_path = os.path.join(scratch, "route.log")
    cmd = [ROUTE, "-lef", lef, "-def", deff, "-output", out_guide]
    for k, v in merged.items():
        cmd += [f"-{k}", str(v)]
    if guide:
        cmd += ["-dgr", guide]
    t0 = time.time()
    with open(log_path, "w") as lf:
        rc = subprocess.call(cmd, cwd=RUN_DIR, stdout=lf,
                             stderr=subprocess.STDOUT)
    dur = time.time() - t0
    with open(log_path) as fh:
        bad = "Unrecognized arg" in fh.read()
    m = parse_cugr2_log(log_path)
    for f in (out_guide, log_path):
        try:
            os.remove(f)
        except OSError:
            pass
    try:
        os.rmdir(scratch)
    except OSError:
        pass
    if bad:
        return dict(rc=3, wirelength=None, via_count=None, overflow=None,
                    runtime_s=round(dur, 1))
    return dict(rc=rc, wirelength=m.get("wirelength"),
                via_count=m.get("via_count"), overflow=m.get("overflow"),
                runtime_s=round(dur, 1))


def jobs_for(benches, arms, cfgs):
    out = []
    for b in benches:
        for a in arms:
            g = guide_for(b, a)
            if g == "MISSING":
                continue
            for name, knobs in PRESETS:
                if cfgs and name not in cfgs:
                    continue
                out.append((b, a, name, knobs, g))
    return out


def done_keys():
    if not os.path.isfile(OUT_CSV):
        return set()
    with open(OUT_CSV) as fh:
        return {(r["benchmark"], r["arm"], r["cfg"])
                for r in csv.DictReader(fh)}


def cmd_plan(args):
    jobs = jobs_for(args.benches.split(","), args.arms.split(","),
                    set(a for a in args.cfgs.split(",") if a))
    done = done_keys()
    pend = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    est = sum(EST_ROUTE_S.get(j[0], 60) for j in pend)
    log(f"{len(jobs)} jobs total, {len(pend)} pending, "
        f"~{est/60:.0f} min single-thread CPU (no GPU needed)")
    for b, a, n, k, _ in pend:
        print(f"  {b:24s} {a:7s} {n:10s} {k}")


def cmd_run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = jobs_for(args.benches.split(","), args.arms.split(","),
                    set(a for a in args.cfgs.split(",") if a))
    done = done_keys()
    new = not os.path.isfile(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for b, a, name, knobs, g in jobs:
            if (b, a, name) in done:
                continue
            log(f"{b} [{a}] {name} {knobs}")
            m = route_with_knobs(b, g, knobs)
            if m["rc"] == 3:
                log("  UNRECOGNIZED FLAG — config skipped")
                continue
            if not m["wirelength"]:
                log(f"  parse failed (rc={m['rc']}) — skipped")
                continue
            w.writerow(dict(
                benchmark=b, arm=a, cfg=name,
                knobs=";".join(f"{k}={v}" for k, v in knobs.items()),
                wirelength=m["wirelength"], via_count=m["via_count"],
                overflow=m["overflow"],
                pareto_cost=m["wirelength"] + 4 * m["via_count"],
                route_s=m["runtime_s"]))
            fh.flush()
            log(f"  WL={m['wirelength']:,} via={m['via_count']:,} "
                f"of={m['overflow']} ({m['runtime_s']}s)")
    log(f"-> {OUT_CSV}")


def cmd_analyze(args):
    assert os.path.isfile(OUT_CSV), "no results yet — run first"
    rows = list(csv.DictReader(open(OUT_CSV)))
    lines = ["# CUGR2 knob-tuning analysis",
             f"_generated {time.strftime('%Y-%m-%d %H:%M')} from "
             f"{os.path.relpath(OUT_CSV, ROOT)}; pareto = WL + 4*via; "
             f"deltas vs the `baseline` cfg of the same (bench, arm)._", ""]
    for b in sorted({r["benchmark"] for r in rows}):
        for a in sorted({r["arm"] for r in rows if r["benchmark"] == b}):
            sub = [r for r in rows
                   if r["benchmark"] == b and r["arm"] == a]
            base = next((r for r in sub if r["cfg"] == "baseline"), None)
            if not base:
                continue
            bp, bo = float(base["pareto_cost"]), float(base["overflow"])
            lines.append(f"## {b} — arm `{a}`")
            lines.append("| cfg | knobs | overflow | Δof | pareto | Δpareto% |")
            lines.append("|---|---|---|---|---|---|")
            for r in sorted(sub, key=lambda x: float(x["pareto_cost"])):
                p, o = float(r["pareto_cost"]), float(r["overflow"])
                lines.append(
                    f"| {r['cfg']} | {r['knobs'] or '-'} | {o:.0f} | "
                    f"{o-bo:+.0f} | {p:,.0f} | {100*(p-bp)/bp:+.3f} |")
            best_p = min(sub, key=lambda x: float(x["pareto_cost"]))
            best_o = min(sub, key=lambda x: (float(x["overflow"]),
                                             float(x["pareto_cost"])))
            lines.append(f"\n- best pareto: `{best_p['cfg']}` "
                         f"({100*(float(best_p['pareto_cost'])-bp)/bp:+.3f}%)"
                         f"; best overflow: `{best_o['cfg']}` "
                         f"({float(best_o['overflow']):.0f} vs {bo:.0f})\n")
    out = os.path.join(OUT_DIR, "TUNING_ANALYSIS.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    log(f"-> {out}")


def cmd_smoke(args):
    m = route_with_knobs("ispd18_test5_metal5", None, {"cls": 2.0})
    log(f"smoke: rc={m['rc']} WL={m['wirelength']} via={m['via_count']} "
        f"of={m['overflow']} ({m['runtime_s']}s)")
    sys.exit(0 if m["wirelength"] else 1)


def cmd_adopt(args):
    """Write experiments/cugr2_tune/BEST_KNOBS.json: per (bench, arm) the
    pareto-best config whose overflow does not regress by more than
    --of_slack edges vs baseline.  Consumed by route_tuned.py."""
    import json
    assert os.path.isfile(OUT_CSV), "no results yet — run first"
    rows = list(csv.DictReader(open(OUT_CSV)))
    knob_map = dict(PRESETS)
    best = {}
    for b in sorted({r["benchmark"] for r in rows}):
        for a in sorted({r["arm"] for r in rows if r["benchmark"] == b}):
            sub = [r for r in rows if r["benchmark"] == b and r["arm"] == a]
            base = next((r for r in sub if r["cfg"] == "baseline"), None)
            if not base:
                continue
            bo, bp = float(base["overflow"]), float(base["pareto_cost"])
            ok = [r for r in sub
                  if float(r["overflow"]) <= bo + args.of_slack]
            pick = min(ok, key=lambda r: float(r["pareto_cost"]))
            best[f"{b}|{a}"] = {
                "cfg": pick["cfg"], "knobs": knob_map.get(pick["cfg"], {}),
                "pareto_delta_pct": round(
                    100 * (float(pick["pareto_cost"]) - bp) / bp, 3),
                "overflow": float(pick["overflow"]),
                "overflow_baseline": bo}
    out = os.path.join(OUT_DIR, "BEST_KNOBS.json")
    with open(out, "w") as fh:
        json.dump(best, fh, indent=2)
    for k, v in best.items():
        log(f"{k:34s} -> {v['cfg']:9s} ({v['pareto_delta_pct']:+.3f}% "
            f"pareto, of {v['overflow']:.0f} vs {v['overflow_baseline']:.0f})")
    log(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--benches", default=",".join(BENCHES))
        p.add_argument("--arms", default="native,s2")
        p.add_argument("--cfgs", default="", help="comma filter of cfg names")
    sub.add_parser("analyze")
    sub.add_parser("smoke")
    p = sub.add_parser("adopt")
    p.add_argument("--of_slack", type=float, default=2.0,
                   help="max overflow regression (edges) allowed vs baseline")
    args = ap.parse_args()
    {"plan": cmd_plan, "run": cmd_run, "analyze": cmd_analyze,
     "smoke": cmd_smoke, "adopt": cmd_adopt}[args.cmd](args)


if __name__ == "__main__":
    main()
