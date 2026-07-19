#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
route_scatter12.py — the advisor-requested 12-point scatter: for each of the
six benchmarks route ALL THREE methods (native CUGR2 / baseline DGR / ours)
under the SAME 12 router-parameter combinations, then plot per chip:
x = via count, y = wirelength, bubble size = overflow, color = method.

Reuses already-measured rows from iso_clean.csv where available (native/dgr,
7 configs); routes only the missing combos, isolated, cached to
RESULTS/scatter12.csv. Figures: RESULTS/figs/scatter12_<chip>.{pdf,png} and
a 2x3 grid scatter12_all. No torch import. NEW file.

  python3 route_scatter12.py --workers 4          # route missing + plot
  python3 route_scatter12.py --plot_only          # plot from cache
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tune_cugr2 as T

BENCHES = ["ispd18_test5_metal5", "ispd18_test8_metal5",
           "ispd18_test10_metal5", "ispd19_test7_metal5",
           "ispd19_test8_metal5", "ispd19_test9_metal5"]
SHORT = {b: b.replace("_metal5", "") for b in BENCHES}
METHODS = ["native", "dgr", "ours"]
MCOLOR = {"native": "#7F7F7F", "dgr": "#2C5F8A", "ours": "#C0392B"}
MLAB = {"native": "native CUGR2", "dgr": "baseline DGR", "ours": "ours"}
CONFIGS = [
    ("default", {}),
    ("cls2", {"cls": 2.0}),
    ("cls4", {"cls": 4.0}),
    ("vm1", {"vm": 1.0}),
    ("wl1", {"wl": 1.0}),
    ("cls2vm1", {"cls": 2.0, "vm": 1.0}),
    ("cls4vm1", {"cls": 4.0, "vm": 1.0}),
    ("cls2wl1vm1", {"cls": 2.0, "wl": 1.0, "vm": 1.0}),
    ("cls2vm1_wsa1k", {"cls": 2.0, "vm": 1.0, "wsa": 1000}),
    ("cls2vm2_wsa1k", {"cls": 2.0, "vm": 2.0, "wsa": 1000}),
    ("vm1_wsa1k", {"vm": 1.0, "wsa": 1000}),
    ("ofmode_vm1", {"overflow_mode": 1, "vm": 1.0}),
]
OUT = os.path.join(ROOT, "RESULTS", "scatter12.csv")
FIG = os.path.join(ROOT, "RESULTS", "figs")
ISO = os.path.join(ROOT, "experiments", "cugr2_tune", "iso_clean.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.25})


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def guide_for(b, method):
    if method == "native":
        return None
    if method == "dgr":
        s = b.replace("ispd", "").replace("_metal5", "")
        return os.path.join(ROOT, "CUGR2_guide", f"CUgr_{b}_S2_{s}.txt")
    return os.path.join(ROOT, "CUGR2_guide", f"CUgr_{b}_ROBUST6.txt")


def route_iso(bench, guide, knobs):
    lef, deff = T.find_input(bench, "lef"), T.find_input(bench, "def")
    merged = dict(T.BASE_POINT)
    merged.update(knobs)
    cwd = tempfile.mkdtemp(prefix="s12_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        for dat in ("POWV9.dat", "POST9.dat"):
            os.symlink(os.path.join(T.RUN_DIR, dat), os.path.join(cwd, dat))
        cmd = [T.ROUTE, "-lef", lef, "-def", deff,
               "-output", os.path.join(cwd, "o.guide")]
        for k, v in merged.items():
            cmd += [f"-{k}", str(v)]
        if guide:
            cmd += ["-dgr", guide]
        lp = os.path.join(cwd, "r.log")
        with open(lp, "w") as lf:
            subprocess.call(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
        return T.parse_cugr2_log(lp)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


def load_done():
    done = {}
    if os.path.isfile(OUT):
        for r in csv.DictReader(open(OUT)):
            done[(r["benchmark"], r["method"], r["cfg"])] = r
    # seed from iso_clean (same isolated protocol; synth there is NOT ours)
    if os.path.isfile(ISO):
        for r in csv.DictReader(open(ISO)):
            if r["method"] in ("native", "dgr"):
                k = (r["chip"], r["method"], r["knob"])
                if k not in done:
                    done[k] = {"benchmark": r["chip"], "method": r["method"],
                               "cfg": r["knob"], "wirelength": r["wirelength"],
                               "via_count": r["via_count"],
                               "overflow": r["overflow"]}
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plot_only", action="store_true")
    a = ap.parse_args()
    done = load_done()
    rows = dict(done)
    if not a.plot_only:
        jobs = [(b, m, cn, cd) for b in BENCHES for m in METHODS
                for cn, cd in CONFIGS if (b, m, cn) not in done]
        log(f"{len(jobs)} missing routes (of {6*3*12})")
        new = not os.path.isfile(OUT)
        fh = open(OUT, "a", newline="")
        w = csv.writer(fh)
        if new:
            w.writerow(["benchmark", "method", "cfg", "wirelength",
                        "via_count", "overflow"])

        def run(j):
            b, m, cn, cd = j
            r = route_iso(b, guide_for(b, m), cd)
            return (b, m, cn, r)
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for b, m, cn, r in ex.map(run, jobs):
                if r.get("wirelength"):
                    w.writerow([b, m, cn, r["wirelength"], r["via_count"],
                                r["overflow"]])
                    fh.flush()
                    rows[(b, m, cn)] = {
                        "benchmark": b, "method": m, "cfg": cn,
                        "wirelength": r["wirelength"],
                        "via_count": r["via_count"],
                        "overflow": r["overflow"]}
                    log(f"{SHORT[b]}/{m}/{cn}: WL={r['wirelength']:,} "
                        f"via={r['via_count']:,} of={r['overflow']}")
                else:
                    log(f"{SHORT[b]}/{m}/{cn}: FAILED")
        fh.close()

    # ── plots: per chip + 2x3 grid ─────────────────────────────────────────
    def draw(ax, b, fs=10, legend=False):
        for m in METHODS:
            pts = [rows[k] for k in rows if k[0] in (b, SHORT[b])
                   and k[1] == m]
            if not pts:
                continue
            x = [int(p["via_count"]) / 1e6 for p in pts]
            y = [int(p["wirelength"]) / 1e6 for p in pts]
            of = [int(p["overflow"]) for p in pts]
            size = [30 + 22 * o for o in of]
            ax.scatter(x, y, s=size, c=MCOLOR[m], alpha=0.75,
                       edgecolors="white", linewidths=0.6,
                       label=MLAB[m] if legend else None)
        ax.set_xlabel("via count (M)", fontsize=fs)
        ax.set_ylabel("wirelength (M)", fontsize=fs)
        ax.set_title(f"{SHORT[b]}  (bubble size = overflow)", fontsize=fs+1)

    os.makedirs(FIG, exist_ok=True)
    for b in BENCHES:
        fig, ax = plt.subplots(figsize=(5.6, 4.4))
        draw(ax, b, legend=True)
        ax.legend(frameon=False)
        fig.tight_layout()
        for ext in (".pdf", "-1.png"):
            fig.savefig(os.path.join(FIG, f"scatter12_{SHORT[b]}{ext}"),
                        dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for ax, b in zip(axes.ravel(), BENCHES):
        draw(ax, b, fs=9, legend=(b == BENCHES[0]))
        if b == BENCHES[0]:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("12 router-parameter combinations per method: via vs "
                 "wirelength (bubble = overflow)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in (".pdf", "-1.png"):
        fig.savefig(os.path.join(FIG, f"scatter12_all{ext}"), dpi=130)
    plt.close(fig)
    log("-> scatter12_<chip> x6 + scatter12_all")


if __name__ == "__main__":
    main()
