#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scaling_diverse.py — the REAL data-scaling law on the diverse, full-size,
ISPD-across packs (all 6 families).

Earlier scaling proxies were flat because the data was homogeneous and the
milestone counted instances-SEEN (epochs), not DISTINCT designs.  This does the
rigorous version: a FIXED held-out split, and for each training-set size N it
trains a FRESH GNN on N DISTINCT packs and measures held-out generalization, so
the curve answers "does MORE DISTINCT data help?".  Full-size packs are handled
with the grid-COARSENED graph (train_all6_robust.derive_graph_capped) so they
do not OOM.  Reuses train_all6_robust / train_scalable READ-ONLY; NEW file.

  python3 scaling_diverse.py --packs '/ocean/.../synthdata/ispd1k/*.npz' \
      --sizes 25,50,100,200 --epochs 3 --device 0
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
from train_all6_robust import derive_graph_capped, loss_on
from train_scalable import pack_to_full
from deepdgr_e2e import DeepDGR_GNN


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def family(p):
    m = re.search(r"ispd_(ispd1[89]_test[0-9]+_metal5)_", os.path.basename(p))
    return m.group(1) if m else "?"


def build(p, dev, max_grid):
    full = pack_to_full(p, dev).full
    xd, ei, *_ = derive_graph_capped(full, max_grid)
    return full, xd, ei


@torch.no_grad()
def heldout(gnn, held, dev, max_grid):
    """held-out FULL objective (of + wl + via) — informative even when the
    instances are certified overflow-0 (then it tracks wl+via quality)."""
    gnn.eval()
    tot = 0.0
    for p in held:
        full, xd, ei = build(p, dev, max_grid)
        l, _ = loss_on(full, xd, ei, gnn, temp=1.0)
        tot += float(l)
        del full, xd, ei
        if dev != "cpu":
            torch.cuda.empty_cache()
    gnn.train()
    return tot / max(len(held), 1)


def train_on(paths, held, dev, max_grid, steps, lr, hidden, layers):
    """fresh GNN trained for a FIXED `steps` budget, CYCLING the N distinct
    packs (so every set-size gets the same total training — isolates 'more
    DISTINCT data' from 'more steps').  Returns held-out objective."""
    full0, xd0, ei0 = build(paths[0], dev, max_grid)
    gnn = DeepDGR_GNN(grid_in=xd0["grid"].shape[1],
                      cand_in=xd0["candidate"].shape[1], hidden=hidden,
                      num_layers=layers).to(dev)
    with torch.no_grad():
        gnn(xd0, ei0)
    del full0, xd0, ei0
    opt = torch.optim.Adam(gnn.parameters(), lr=lr, weight_decay=1e-5)
    for t in range(steps):
        p = paths[t % len(paths)]                # cycle the distinct pool
        full, xd, ei = build(p, dev, max_grid)
        temp = max(0.1, 1.0 - t / max(steps, 1))
        opt.zero_grad()
        loss, _ = loss_on(full, xd, ei, gnn, temp)
        loss.backward()
        opt.step()
        del full, xd, ei
        if dev != "cpu":
            torch.cuda.empty_cache()
    return heldout(gnn, held, dev, max_grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--sizes", default="25,50,100,200")
    ap.add_argument("--steps", type=int, default=400,
                    help="fixed total training steps per set-size (cycled)")
    ap.add_argument("--per_family_holdout", type=int, default=4)
    ap.add_argument("--max_grid", type=int, default=120000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    dev = (f"cuda:{a.device}" if a.device >= 0 and torch.cuda.is_available()
           else "cpu")
    paths = sorted(glob.glob(a.packs))
    byfam = {}
    for p in paths:
        byfam.setdefault(family(p), []).append(p)
    log(f"{len(paths)} packs across {len(byfam)} families: "
        f"{ {k: len(v) for k, v in byfam.items()} }")
    # FIXED held-out: first K of each family; train pool = the rest, family-
    # interleaved so a size-N subset is balanced across families.
    held, pool = [], []
    for fam, ps in byfam.items():
        held += ps[:a.per_family_holdout]
        pool.append(ps[a.per_family_holdout:])
    inter = []
    for i in range(max(len(x) for x in pool)):
        for x in pool:
            if i < len(x):
                inter.append(x[i])
    log(f"held-out {len(held)} (fixed), train pool {len(inter)}")
    sizes = [n for n in (int(s) for s in a.sizes.split(",")) if n <= len(inter)]
    out = os.path.join(ROOT, "validation_out", "scaling_diverse.csv")
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["n_distinct_train", "heldout_objective", "n_families",
                "total_steps", "s"])
    rows = []
    for N in sizes:
        t0 = time.time()
        ho = train_on(inter[:N], held, dev, a.max_grid, a.steps, a.lr,
                      a.hidden, a.layers)
        dt = round(time.time() - t0, 1)
        w.writerow([N, round(ho, 1), len(byfam), a.steps, dt])
        fh.flush()
        rows.append((N, ho))
        log(f"  N={N:4d} distinct ({a.steps} steps) -> held-out objective "
            f"{ho:.1f}  ({dt}s)")
    fh.close()
    if len(rows) >= 2:
        d = rows[0][1] - rows[-1][1]
        rel = 100 * d / rows[0][1] if rows[0][1] else 0
        log(f"SCALING: held-out objective {rows[0][1]:.0f} (N={rows[0][0]}) -> "
            f"{rows[-1][1]:.0f} (N={rows[-1][0]}) = {rel:+.1f}% "
            f"({'scaling HELPS' if d > 0 else 'flat/no help'})")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-",
                    color="black", mfc="none")
            ax.set_xscale("log")
            ax.set_xlabel("# distinct training designs (log)")
            ax.set_ylabel("held-out objective (of+wl+via)")
            ax.set_title("Data-scaling law (diverse ISPD-across, fixed held-out)")
            f.tight_layout()
            fig = os.path.join(ROOT, "reports", "figs", "scaling_diverse.pdf")
            os.makedirs(os.path.dirname(fig), exist_ok=True)
            f.savefig(fig)
            log(f"figure -> {fig}")
        except Exception as e:
            log(f"plot skipped: {e}")
    log(f"-> {out}")


if __name__ == "__main__":
    main()
