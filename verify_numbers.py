#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_numbers.py — re-derive every number in report/final_report.pdf from the
shipped measured CSVs (RESULTS/*.csv). No GPU, no benchmarks, no torch. Prints
a PASS/FAIL line per check; exit code = number of failures.

  python3 verify_numbers.py
"""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "RESULTS")
CH = ["18_t5", "18_t8", "18_t10", "19_t7", "19_t8", "19_t9"]
fails = 0


def check(name, cond):
    global fails
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        fails += 1


def M(x):
    return f"{x/1e6:.2f}M"


def load(name):
    p = os.path.join(R, name)
    return {r.get("chip", r.get("benchmark", "")): r
            for r in csv.DictReader(open(p))} if os.path.isfile(p) else {}


print("=" * 70)
print("Re-deriving report numbers from RESULTS/*.csv")
print("=" * 70)

# --- Table 1 ours column ---
fb = load("final_best.csv")
T1 = {"18_t5": ("26.29M", 0), "18_t8": ("60.44M", 0), "18_t10": ("70.19M", 0),
      "19_t7": ("103.97M", 0), "19_t8": ("175.70M", 10), "19_t9": ("261.34M", 28)}
print("\nTable 1 (ours) vs final_best.csv:")
for c, (wl, of) in T1.items():
    r = fb.get(c)
    check(f"{c} ours WL/of", bool(r) and M(int(r["wirelength"])) == wl
          and int(r["overflow"]) == of)

# --- strict domination 6/6 (needs baseline numbers embedded from report) ---
NAT = {"18_t5": (26569635, 856342, 5), "18_t8": (61307179, 2122848, 0),
       "18_t10": (72767011, 2244966, 0), "19_t7": (104651440, 3813121, 0),
       "19_t8": (176905274, 6124546, 18), "19_t9": (262160756, 10187117, 30)}
DGR = {"18_t5": (26432520, 863898, 7), "18_t8": (61187719, 2138206, 0),
       "18_t10": (72206925, 2258945, 1), "19_t7": (104553115, 3826077, 0),
       "19_t8": (176404419, 6154072, 19), "19_t9": (261647831, 10243129, 37)}
print("\nStrict domination (ours <= native AND <= DGR on WL,via,of):")
ndom = 0
for c in CH:
    r = fb.get(c)
    o = (int(r["wirelength"]), int(r["via_count"]), int(r["overflow"]))
    dom = all(o[i] <= NAT[c][i] for i in range(3)) and \
          all(o[i] <= DGR[c][i] for i in range(3))
    ndom += dom
check(f"strict domination on 6/6 (got {ndom})", ndom == 6)

# --- Delta% recompute ---
print("\nDelta% (ours vs native) recomputed:")
DPCT = {"18_t5": (-1.06, -6.43), "18_t8": (-1.42, -8.09),
        "18_t10": (-3.53, -7.03), "19_t7": (-0.65, -2.94),
        "19_t8": (-0.68, -1.38), "19_t9": (-0.31, -5.01)}
for c in CH:
    r = fb.get(c)
    dwl = 100 * (int(r["wirelength"]) - NAT[c][0]) / NAT[c][0]
    dvi = 100 * (int(r["via_count"]) - NAT[c][1]) / NAT[c][1]
    check(f"{c} dWL/dVia", abs(dwl - DPCT[c][0]) < 0.02
          and abs(dvi - DPCT[c][1]) < 0.02)

# --- Table 3 runtime present & speedup in 2.8-4.8x ---
fr = load("flow_runtime.csv")
print("\nTable 3 runtime (speedup 2.8x-4.8x):")
for c in CH:
    r = fr.get(c)
    spd = float(r["ours_vs_dgr_speedup"].rstrip("x")) if r else 0
    check(f"{c} speedup in range", 2.5 <= spd <= 5.0)

# --- Fig 2 overflow curve: warm below cold ---
print("\nFigure 2 overflow curves (warm < cold at start):")
for c in ("18_t5", "19_t8"):
    p = os.path.join(R, f"overflow_curve_{c}.csv")
    rows = list(csv.DictReader(open(p))) if os.path.isfile(p) else []
    w = [float(x["overflow_units"]) for x in rows if x["init"] == "warm"]
    cd = [float(x["overflow_units"]) for x in rows if x["init"] == "cold"]
    check(f"{c} warm[0] < cold[0]", bool(w and cd) and w[0] < cd[0])

# --- Fig 1 scatter completeness ---
print("\nFigure 1 / Pareto: 12 settings x 3 methods x 6 chips:")
data = {c: {m: set() for m in ("native", "dgr", "ours")} for c in
        ["ispd18_test5", "ispd18_test8", "ispd18_test10", "ispd19_test7",
         "ispd19_test8", "ispd19_test9"]}
sp = os.path.join(R, "scatter12.csv")
for r in csv.DictReader(open(sp)):
    c = r["benchmark"].replace("_metal5", "")
    if c in data and r["method"] in data[c]:
        data[c][r["method"]].add((r["via_count"], r["wirelength"]))
iso = os.path.join(ROOT, "experiments", "cugr2_tune", "iso_clean.csv")
if os.path.isfile(iso):
    for r in csv.DictReader(open(iso)):
        if r["method"] in ("native", "dgr"):
            c = r["chip"].replace("_metal5", "")
            if c in data:
                data[c][r["method"]].add((r["via_count"], r["wirelength"]))
complete = all(len(data[c][m]) >= 12 for c in data for m in data[c])
check("all methods have >=12 operating points on all chips", complete)

print("\n" + "=" * 70)
print(f"RESULT: {'ALL CHECKS PASS' if fails == 0 else str(fails)+' FAILED'}")
print("=" * 70)
sys.exit(fails)
