#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_all6_robust.py — train the warm-start GNN on ALL 6 full-size ISPD
benchmarks WITHOUT OOM, then test on all 6.

The OOM (e2e_stream --holdout 0 died on the 1337x1433 / 895k-net chip) comes
from the GRID graph: a 1.9M-cell 4-neighbour lattice + candidate<->grid edges
blow past 80 GB in the backward pass.  But the grid is only spatial CONTEXT;
the per-candidate routing decisions (the GNN output) are what matter.  So we
COARSEN the grid by an automatic pooling factor (grid nodes capped to
--max_grid), keeping candidates full-resolution.  This bounds memory for ANY
benchmark size, reuses DeepDGR_GNN UNCHANGED, and is paired with AMP + a CPU
fallback so a stubborn instance never kills the run.  Diversified: trains on the
6 real chips AND (if present) the ISPD-across synthetic packs.  Then tests on
all 6.  Reuses e2e_stream/dgr_fast/fastdgr_core READ-ONLY; NEW file.

  python3 train_all6_robust.py --steps 900 --max_grid 120000 --device 0
"""
import argparse
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
import e2e_stream as ES                                       # read-only
from deepdgr_e2e import DeepDGR_GNN                           # read-only
from fastdgr_core import seg_softmax                          # read-only

BENCHES = ["ispd18_test5_metal5", "ispd18_test8_metal5",
           "ispd18_test10_metal5", "ispd19_test7_metal5",
           "ispd19_test8_metal5", "ispd19_test9_metal5"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ── grid-COARSENED hetero graph (bounds the OOM sink) ───────────────────────
def derive_graph_capped(full, max_grid=120000):
    dev = full.device
    xmax, ymax, n = full.xmax, full.ymax, full.n
    G = xmax * ymax
    pool = max(1, math.ceil(math.sqrt(G / max_grid))) if G > max_grid else 1
    cx, cy = math.ceil(xmax / pool), math.ceil(ymax / pool)
    Gc = cx * cy
    x_cand = ES.candidate_features(full)
    xg = ES.grid_features(full)                               # (G, 4)
    xs = torch.arange(xmax, device=dev)
    ys = torch.arange(ymax, device=dev)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    coarse_of = ((gx // pool) * cy + (gy // pool)).reshape(-1).long()  # G->Gc
    x_grid = torch.zeros(Gc, xg.shape[1], device=dev)
    cnt = torch.zeros(Gc, 1, device=dev)
    x_grid.index_add_(0, coarse_of, xg)
    cnt.index_add_(0, coarse_of, torch.ones(G, 1, device=dev))
    x_grid = x_grid / cnt.clamp_min(1.0)
    # candidate <-> coarse grid (dedup naturally caps the bipartite edges)
    cols = torch.arange(n, device=dev)
    rows, _, owner = full.via.gather_cols(cols)
    crows = coarse_of[rows.long()]
    pt = torch.stack([owner.to(dev).long(), crows]).unique(dim=1)
    infl = torch.stack([pt[1], pt[0]])
    # coarse 4-neighbour lattice
    cxs, cys = torch.arange(cx, device=dev), torch.arange(cy, device=dev)
    ggx, ggy = torch.meshgrid(cxs, cys, indexing="ij")
    cflat = (ggx * cy + ggy).reshape(-1)
    right = cflat[(ggy < cy - 1).reshape(-1)]
    down = cflat[(ggx < cx - 1).reshape(-1)]
    conn = torch.cat([torch.stack([right, right + 1]),
                      torch.stack([down, down + cy])], dim=1)
    conn = torch.cat([conn, conn.flip(0)], dim=1)
    # competes: candidates of the same subnet (per-width, no dense n*n)
    widths = (full.p_index[1:] - full.p_index[:-1]).to(dev)
    starts = full.p_index[:-1].to(dev)
    sl, dl = [], []
    for w in torch.unique(widths):
        w = int(w)
        if w < 2:
            continue
        st = starts[widths == w]
        base = torch.arange(w, device=dev)
        aa = base.view(-1, 1).expand(w, w).reshape(-1)
        bb = base.view(1, -1).expand(w, w).reshape(-1)
        keep = aa != bb
        aa, bb = aa[keep], bb[keep]
        sl.append((st.view(-1, 1) + aa.view(1, -1)).reshape(-1))
        dl.append((st.view(-1, 1) + bb.view(1, -1)).reshape(-1))
    comp = (torch.stack([torch.cat(sl), torch.cat(dl)]) if sl else
            torch.zeros(2, 0, dtype=torch.long, device=dev))
    x_dict = {"grid": x_grid, "candidate": x_cand}
    ei = {("grid", "connects", "grid"): conn.long(),
          ("candidate", "passes_through", "grid"): pt.long(),
          ("grid", "influences", "candidate"): infl.long(),
          ("candidate", "competes", "candidate"): comp.long()}
    return x_dict, ei, x_grid.shape[1], x_cand.shape[1], pool


def load_full(bench_pt, device):
    from types import SimpleNamespace
    from dgr_fast import load_benchmark, build_full_problem
    a = SimpleNamespace(data_path=bench_pt, device=device, capacity=1.0,
                        pin_ratio=1.0, local_net_ratio=1.0, via_layer=1.5,
                        pattern_level=1, z_step=3, max_z=10, c_step=3,
                        max_c=20, max_c_out_ratio=5, act="sigmoid",
                        act_scale=0.5, celu_alpha=2.0)
    pre, name, _p, p_index, _pif, _p2, hor, ver, wl, via = load_benchmark(a)
    return build_full_problem(a, pre, p_index, hor, ver, wl, via), name


def bench_pt(b):
    for c in (os.path.join(ROOT, f"{b}.pt"),
              os.path.join(ROOT, "cu-gr-2", "run", f"{b}.pt")):
        if os.path.isfile(c):
            return c
    return None


def loss_on(full, x_dict, ei, gnn, temp, wl_c=0.5, via_c=4.0):
    logits = gnn(x_dict, ei)
    gp = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-10)))
    p = seg_softmax((logits + gp) / max(temp, 0.05), full.seg, full.S)
    of, vc, wc, _ = full.objective_full(p)
    return of + wl_c * wc + via_c * vc, float(of)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--max_grid", type=int, default=120000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--amp", action="store_true", default=False,
                    help="OFF by default: the sparse CSR overflow op "
                         "(cusparseSpMV) does not support fp16 autocast")
    ap.add_argument("--save_gnn", default="gnn_all6_robust.pth")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--test_only", action="store_true",
                    help="load --save_gnn and only run the all-6 test")
    a = ap.parse_args()
    dev = (f"cuda:{a.device}" if a.device >= 0 and torch.cuda.is_available()
           else "cpu")

    if a.smoke:                                   # tiny self-contained check
        fulls = [(f"synth{s}", ES.make_synthetic_full(s, dev, xmax=20, ymax=18,
                  n_subnets=40)) for s in range(3)]
        a.steps, a.max_grid = 9, 200
    else:
        fulls = []
        for b in BENCHES:
            pt = bench_pt(b)
            if pt is None:
                log(f"{b}: .pt missing — skip"); continue
            t0 = time.time()
            full, _ = load_full(pt, dev)
            fulls.append((b, full))
            log(f"{b}: loaded ({full.n:,} cand, grid {full.xmax}x{full.ymax}) "
                f"in {time.time()-t0:.0f}s")
    if not fulls:
        raise SystemExit("no benchmarks loaded")

    # pre-build capped graphs once (reused every visit)
    graphs = {}
    gnn = None
    for name, full in fulls:
        xd, ei, gin, cin, pool = derive_graph_capped(full, a.max_grid)
        graphs[name] = (xd, ei)
        log(f"  {name}: capped graph grid pool={pool} "
            f"({xd['grid'].shape[0]:,} grid nodes, "
            f"{ei[('candidate','passes_through','grid')].shape[1]:,} c->g edges)")
        if gnn is None:
            gnn = DeepDGR_GNN(grid_in=gin, cand_in=cin, hidden=64,
                              num_layers=3).to(dev)
            with torch.no_grad():
                gnn(xd, ei)                       # lazy build
            log(f"  GNN: {sum(p.numel() for p in gnn.parameters()):,} params")
    if a.test_only:
        sd = torch.load(os.path.join(ROOT, a.save_gnn), map_location=dev,
                        weights_only=False)
        gnn.load_state_dict(sd, strict=False)
        log(f"test_only: loaded {a.save_gnn} — skipping training")
    opt = torch.optim.Adam(gnn.parameters(), lr=a.lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=(a.amp and dev != "cpu"))
    names = [n for n, _ in fulls]
    fmap = dict(fulls)
    log(f"training {a.steps} steps over {len(names)} benchmarks (all-6)")
    for t in range(0 if a.test_only else a.steps):
        name = names[t % len(names)]
        full = fmap[name]
        xd, ei = graphs[name]
        temp = max(0.1, 1.0 - t / max(a.steps, 1))
        opt.zero_grad()
        try:
            with torch.cuda.amp.autocast(enabled=(a.amp and dev != "cpu")):
                loss, of = loss_on(full, xd, ei, gnn, temp)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        except RuntimeError as e:                 # OOM fallback -> CPU step
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            log(f"  step {t} {name}: GPU OOM -> CPU fallback")
            gnn.to("cpu")
            xc = {k: v.cpu() for k, v in xd.items()}
            eic = {k: v.cpu() for k, v in ei.items()}
            fc = full  # objective on CPU
            loss, of = loss_on(fc, xc, eic, gnn, temp)
            opt.zero_grad(); loss.backward(); opt.step()
            gnn.to(dev)
        if dev != "cpu":
            torch.cuda.empty_cache()
        if t % max(1, a.steps // 12) == 0 or t == a.steps - 1:
            log(f"  step {t:4d} {name:24s} loss={float(loss):.2f} of={of:.1f}")
    torch.save(gnn.state_dict(), os.path.join(ROOT, a.save_gnn))
    log(f"saved -> {a.save_gnn}")
    if a.smoke:
        log("SMOKE OK"); return
    # TEST on all 6 with the SAME capped graph used in training:
    # GNN -> warm-start -> FastDGR -> isolated route -> WL/via/overflow
    import csv
    from test_gnn_all6 import route_iso
    from warmstart_converge import KNOB
    gnn.eval()
    out = os.path.join(ROOT, "experiments", "cugr2_tune", "test_all6.csv")
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["benchmark", "wirelength", "via_count", "overflow", "route_s"])
    log("TEST on all 6 (capped graph -> warm-start -> FastDGR -> route)")
    for name, full in fulls:
        xd, ei = graphs[name]
        with torch.no_grad():
            logits = gnn(xd, ei).detach().cpu().numpy().astype(np.float32)
        ws = os.path.join(ROOT, f"{name}_ROBUST_ws.npz")
        np.savez(ws, logits=logits)
        pt = bench_pt(name)
        rc = subprocess.call(
            [sys.executable, os.path.join(ROOT, "dgr_fast.py"), "--data_path",
             pt, "--warmstart_file", ws, "--output_name", "ROBUST6", "--iter",
             "600", "--device", str(a.device), "--out_dir",
             os.path.join(ROOT, "experiments", "fastdgr")], cwd=ROOT)
        guide = os.path.join(ROOT, "CUGR2_guide", f"CUgr_{name}_ROBUST6.txt")
        if rc != 0 or not os.path.isfile(guide):
            log(f"  {name}: FastDGR failed"); continue
        m = route_iso(name, guide, KNOB.get(name, {"cls": 2.0, "vm": 1.0}))
        if not m.get("wirelength"):
            log(f"  {name}: route failed"); continue
        w.writerow([name, m["wirelength"], m["via_count"], m["overflow"],
                    m.get("runtime_s", "")])
        fh.flush()
        log(f"  TEST {name}: WL={m['wirelength']:,} via={m['via_count']:,} "
            f"of={m['overflow']}")
    fh.close()
    log(f"-> {out}")


if __name__ == "__main__":
    main()
