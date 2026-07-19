#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iso_compare.py — CLEAN, collision-proof iso-overflow comparison.

Why: tune_cugr2.route_with_knobs runs the router in the SHARED cu-gr-2/run/
dir, so concurrent routes overwrite each other's fixed-name intermediates
(capacity3D.txt, dgr_maze.txt, ...) and corrupt the metrics.  Here every
route gets its OWN temp cwd with the FLUTE tables (POWV9/POST9.dat) symlinked
in -> fully isolated, parallel-safe, reproduces native exactly.

It routes each method (native / DGR-S2 / synthetic) across a knob grid, then
the analysis picks, per chip, a knob per method that MATCHES overflow, and
compares the other two metrics (WL, via).  Existing code untouched.

  python3 iso_compare.py --workers 6
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import tune_cugr2 as T

CHIPS = ["ispd18_test5_metal5", "ispd18_test8_metal5", "ispd18_test10_metal5",
         "ispd19_test7_metal5", "ispd19_test8_metal5", "ispd19_test9_metal5"]
KNOBS = [("default", {}), ("cls2vm1", {"cls": 2.0, "vm": 1.0}),
         ("vm1", {"vm": 1.0}), ("cls4vm1", {"cls": 4.0, "vm": 1.0}),
         ("cls2vm1_wsa1k", {"cls": 2.0, "vm": 1.0, "wsa": 1000}),
         ("cls2vm2_wsa1k", {"cls": 2.0, "vm": 2.0, "wsa": 1000}),
         ("ofmode_vm1", {"overflow_mode": 1, "vm": 1.0})]
OUT = os.path.join(ROOT, "experiments", "cugr2_tune", "iso_clean.csv")


def short(c):
    return c.replace("ispd", "").replace("_metal5", "")


def guide_for(chip, method):
    if method == "native":
        return None
    g = {"dgr": f"CUgr_{chip}_S2_{short(chip)}.txt",
         "synth": f"CUgr_{chip}_SYNTHX.txt",
         "synth_orig": f"CUgr_{chip}_SYNTHXORIG.txt",
         "cross": f"CUgr_{chip}_CROSS64.txt"}[method]
    p = os.path.join(ROOT, "CUGR2_guide", g)
    return p if os.path.isfile(p) else "MISSING"


def route_iso(chip, guide, knobs):
    lef, deff = T.find_input(chip, "lef"), T.find_input(chip, "def")
    if not (lef and deff):
        return None
    merged = dict(T.BASE_POINT)
    merged.update(knobs)
    cwd = tempfile.mkdtemp(prefix="iso_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        for dat in ("POWV9.dat", "POST9.dat"):
            os.symlink(os.path.join(T.RUN_DIR, dat), os.path.join(cwd, dat))
        og = os.path.join(cwd, "out.guide")
        lp = os.path.join(cwd, "route.log")
        cmd = [T.ROUTE, "-lef", lef, "-def", deff, "-output", og]
        for k, v in merged.items():
            cmd += [f"-{k}", str(v)]
        if guide:
            cmd += ["-dgr", guide]
        with open(lp, "w") as lf:
            subprocess.call(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
        m = T.parse_cugr2_log(lp)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--methods", default="native,dgr,synth,synth_orig,cross")
    ap.add_argument("--chips", default=",".join(CHIPS))
    a = ap.parse_args()
    methods = a.methods.split(",")
    chips = a.chips.split(",")
    done = set()
    if os.path.isfile(OUT):
        done = {(r["chip"], r["method"], r["knob"])
                for r in csv.DictReader(open(OUT))}
    jobs = []
    for c in chips:
        for me in methods:
            g = guide_for(c, me)
            if g == "MISSING":
                continue
            for kn, kd in KNOBS:
                if (c, me, kn) not in done:
                    jobs.append((c, me, g, kn, kd))
    print(f"[iso] {len(jobs)} clean isolated routes on {a.workers} workers",
          flush=True)
    new = not os.path.isfile(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["chip", "method", "knob", "wirelength", "via_count",
                    "overflow", "pareto"])

    def run(job):
        c, me, g, kn, kd = job
        m = route_iso(c, g, kd)
        return (c, me, kn, m)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(run, j)
                                              for j in jobs]), 1):
            c, me, kn, m = fut.result()
            if m and m.get("wirelength"):
                wl, via, of = m["wirelength"], m["via_count"], m["overflow"]
                w.writerow([c, me, kn, wl, via, of, wl + 4 * via])
                fh.flush()
                print(f"  [{i}/{len(jobs)}] {short(c):8} {me:10} {kn:14} "
                      f"WL={wl:,} via={via:,} of={of}", flush=True)
            else:
                print(f"  [{i}/{len(jobs)}] {short(c)} {me} {kn} FAILED",
                      flush=True)
    fh.close()
    print(f"[iso] done in {(time.time()-t0)/60:.1f} min -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
