#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_gnn_all6.py — test a trained warm-start GNN on ALL 6 ISPD benchmarks.

For each of the six metal5 chips: build the real full-size instance, run the
GNN forward to get a per-candidate warm-start, hand it to FastDGR (dgr_fast
--warmstart_file), route the guide ISOLATED, and report WL/via/overflow vs the
native + DGR baselines. One row per chip -> experiments/cugr2_tune/test_all6.csv.

Pairs with training on ALL 6 (e2e_stream --holdout 0): train-on-all-6 then
test-on-all-6. Reuses e2e_stream/deepdgr_e2e/dgr_fast/iso_compare READ-ONLY;
NEW file, nothing existing changed.

  python3 test_gnn_all6.py --load_gnn stream_gnn_all6.pth --device 0
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
import e2e_stream as ES                                       # read-only
from deepdgr_e2e import DeepDGR_GNN                           # read-only
import tune_cugr2 as T                                        # read-only
from warmstart_converge import KNOB, SH

BENCHES = ["ispd18_test5_metal5", "ispd18_test8_metal5",
           "ispd18_test10_metal5", "ispd19_test7_metal5",
           "ispd19_test8_metal5", "ispd19_test9_metal5"]
PY = sys.executable


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bench_pt(b):
    for c in (os.path.join(ROOT, f"{b}.pt"),
              os.path.join(ROOT, "cu-gr-2", "run", f"{b}.pt")):
        if os.path.isfile(c):
            return c
    return None


def route_iso(bench, guide, knobs):
    """isolated CUGR2 route (iso_compare pattern): own cwd + FLUTE symlinks."""
    lef, deff = T.find_input(bench, "lef"), T.find_input(bench, "def")
    merged = dict(T.BASE_POINT)
    merged.update(knobs)
    cwd = tempfile.mkdtemp(prefix="t6_", dir=os.environ.get("TMPDIR", "/tmp"))
    try:
        for dat in ("POWV9.dat", "POST9.dat"):
            os.symlink(os.path.join(T.RUN_DIR, dat), os.path.join(cwd, dat))
        og, lp = os.path.join(cwd, "o.guide"), os.path.join(cwd, "r.log")
        cmd = [T.ROUTE, "-lef", lef, "-def", deff, "-output", og]
        for k, v in merged.items():
            cmd += [f"-{k}", str(v)]
        cmd += ["-dgr", guide]
        with open(lp, "w") as lf:
            subprocess.call(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
        return T.parse_cugr2_log(lp)
    finally:
        shutil.rmtree(cwd, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_gnn", required=True)
    ap.add_argument("--benches", default=",".join(BENCHES))
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--iter", type=int, default=600)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    a = ap.parse_args()
    dev = (f"cuda:{a.device}" if a.device >= 0 and torch.cuda.is_available()
           else "cpu")
    benches = a.benches.split(",")

    # native + DGR baselines for the deltas
    br = [r for r in csv.DictReader(
        open(os.path.join(ROOT, "experiments/fastdgr/bench_results.csv")))
        if r["wsa"] == "500"]

    def base(b, arm):
        try:
            r = next(x for x in br if x["benchmark"] == b and x["arm"] == arm)
            return (int(float(r["wirelength"])), int(float(r["via_count"])),
                    int(float(r["overflow"])))
        except StopIteration:
            return (0, 0, 0)

    # build GNN, init lazy convs from the first instance, load weights
    gnn = DeepDGR_GNN(grid_in=4, cand_in=4, hidden=a.hidden,
                      num_layers=a.layers).to(dev)
    out = os.path.join(ROOT, "experiments", "cugr2_tune", "test_all6.csv")
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["benchmark", "wirelength", "via_count", "overflow",
                "native_WL", "native_via", "native_of", "dgr_WL", "dgr_via",
                "dgr_of", "dWL_vs_native_%", "dvia_vs_native_%", "opt_s",
                "route_s"])
    inited = False
    for b in benches:
        pt = bench_pt(b)
        if pt is None:
            log(f"{b}: .pt missing — skip"); continue
        log(f"{SH.get(b,b)}: build instance + GNN warm-start")
        inst = ES.build_instance_real(pt, dev, wl_coeff=0.5, via_coeff=4.0,
                                      pattern_level=1, max_c=20)
        if not inited:
            with torch.no_grad():
                gnn(inst.x_dict, inst.edge_index_dict)        # lazy build
            ck = torch.load(a.load_gnn, map_location=dev, weights_only=False)
            sd = (ck.get("gnn_state_dict") or ck.get("model_state_dict") or ck
                  if isinstance(ck, dict) else ck)
            miss = gnn.load_state_dict(sd, strict=False)
            log(f"  loaded {os.path.basename(a.load_gnn)} "
                f"(missing={len(miss.missing_keys)})")
            inited = True
        with torch.no_grad():
            logits = gnn(inst.x_dict, inst.edge_index_dict).detach()
        ws = os.path.join(ROOT, f"{b}_GNNALL6_ws.npz")
        np.savez(ws, logits=logits.cpu().numpy().astype(np.float32))
        # FastDGR from the GNN warm-start -> guide
        t0 = time.time()
        rc = subprocess.call(
            [PY, os.path.join(ROOT, "dgr_fast.py"), "--data_path", pt,
             "--warmstart_file", ws, "--output_name", "GNNALL6",
             "--iter", str(a.iter), "--device", str(a.device),
             "--out_dir", os.path.join(ROOT, "experiments", "fastdgr")],
            cwd=ROOT)
        opt_s = round(time.time() - t0, 1)
        guide = os.path.join(ROOT, "CUGR2_guide", f"CUgr_{b}_GNNALL6.txt")
        if rc != 0 or not os.path.isfile(guide):
            log(f"  {b}: FastDGR failed rc={rc}"); continue
        m = route_iso(b, guide, KNOB.get(b, {"cls": 2.0, "vm": 1.0}))
        if not m.get("wirelength"):
            log(f"  {b}: route failed"); continue
        nv, dg = base(b, "cugr2_native"), base(b, "s2_existing")
        dwl = 100 * (m["wirelength"] - nv[0]) / nv[0] if nv[0] else 0
        dvia = 100 * (m["via_count"] - nv[1]) / nv[1] if nv[1] else 0
        w.writerow([b, m["wirelength"], m["via_count"], m["overflow"],
                    nv[0], nv[1], nv[2], dg[0], dg[1], dg[2],
                    round(dwl, 3), round(dvia, 3), opt_s, m["runtime_s"]])
        fh.flush()
        log(f"  {SH.get(b,b)}: WL={m['wirelength']:,} via={m['via_count']:,} "
            f"of={m['overflow']}  vs native WL{dwl:+.2f}% via{dvia:+.2f}%")
    fh.close()
    log(f"-> {out}")


if __name__ == "__main__":
    main()
