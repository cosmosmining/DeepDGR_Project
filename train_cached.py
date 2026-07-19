#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cached.py — FAST batch trainer: same teacher-free physics training as
scaling_diverse.py but with a GRAPH CACHE, removing its dominant cost.

Measured problem: scaling_diverse rebuilds pack_to_full + derive_graph_capped
EVERY optimizer step (~60s per size sweep is mostly rebuild).  Packs are small
(0.02-0.2 MB), so here each pack's (FullProblem, x_dict, edge_index_dict) is
built ONCE and LRU-cached in RAM (--cache instances resident).  Steps become
pure forward/backward -> 10-50x faster epochs, same math, same GNN
(DeepDGR_GNN via train_all6_robust.derive_graph_capped, UNCHANGED).

Also runs the distinct-set-size scaling sweep (fixed held-out) on the CONGESTED
packs — the data with actual overflow signal.  Reuses train_all6_robust /
train_scalable / deepdgr_e2e READ-ONLY; NEW file.

  python3 train_cached.py --packs '/ocean/.../synthdata/cong10k/*.npz' \
      --sizes 50,200,800,3200 --steps 600 --device -1
"""
import argparse
import csv
import glob
import os
import sys
import time
from collections import OrderedDict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
from train_all6_robust import derive_graph_capped, loss_on
from train_scalable import pack_to_full
from deepdgr_e2e import DeepDGR_GNN


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class GraphCache:
    """path -> (full, x_dict, ei), built once, LRU-bounded."""

    def __init__(self, dev, max_grid, cap=512):
        self.dev, self.max_grid, self.cap = dev, max_grid, cap
        self._c = OrderedDict()
        self.builds = 0
        self.build_s = 0.0

    def get(self, p):
        if p in self._c:
            self._c.move_to_end(p)
            return self._c[p]
        t0 = time.time()
        full = pack_to_full(p, self.dev).full
        xd, ei, *_ = derive_graph_capped(full, self.max_grid)
        self.builds += 1
        self.build_s += time.time() - t0
        self._c[p] = (full, xd, ei)
        while len(self._c) > self.cap:
            self._c.popitem(last=False)
        return self._c[p]


@torch.no_grad()
def heldout(gnn, cache, held):
    gnn.eval()
    tot = 0.0
    for p in held:
        full, xd, ei = cache.get(p)
        l, _ = loss_on(full, xd, ei, gnn, temp=1.0)
        tot += float(l)
    gnn.train()
    return tot / max(len(held), 1)


def train_size(paths, held, cache, steps, lr, hidden, layers, dev):
    full0, xd0, ei0 = cache.get(paths[0])
    gnn = DeepDGR_GNN(grid_in=xd0["grid"].shape[1],
                      cand_in=xd0["candidate"].shape[1], hidden=hidden,
                      num_layers=layers).to(dev)
    with torch.no_grad():
        gnn(xd0, ei0)
    opt = torch.optim.Adam(gnn.parameters(), lr=lr, weight_decay=1e-5)
    t0 = time.time()
    for t in range(steps):
        full, xd, ei = cache.get(paths[t % len(paths)])
        temp = max(0.1, 1.0 - t / max(steps, 1))
        opt.zero_grad()
        loss, _ = loss_on(full, xd, ei, gnn, temp)
        loss.backward()
        opt.step()
    step_s = time.time() - t0
    return heldout(gnn, cache, held), step_s, gnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", required=True)
    ap.add_argument("--sizes", default="50,200,800,3200")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--holdout", type=int, default=24)
    ap.add_argument("--cache", type=int, default=512)
    ap.add_argument("--max_grid", type=int, default=120000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--device", type=int, default=-1)
    ap.add_argument("--save_best", default="gnn_cong_scaled.pth")
    a = ap.parse_args()
    dev = (f"cuda:{a.device}" if a.device >= 0 and torch.cuda.is_available()
           else "cpu")
    paths = sorted(glob.glob(a.packs))
    if len(paths) < a.holdout + 10:
        raise SystemExit(f"only {len(paths)} packs")
    rng = np.random.RandomState(0)
    order = rng.permutation(len(paths))
    held = [paths[i] for i in order[:a.holdout]]          # FIXED held-out
    pool = [paths[i] for i in order[a.holdout:]]
    cache = GraphCache(dev, a.max_grid, cap=a.cache)
    log(f"{len(pool)} train / {len(held)} held-out packs; cache cap {a.cache}")
    sizes = [n for n in (int(s) for s in a.sizes.split(",")) if n <= len(pool)]
    out = os.path.join(ROOT, "RESULTS", "scaling_congested.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fh = open(out, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["n_distinct_train", "heldout_objective", "steps", "train_s",
                "graph_builds", "graph_build_s"])
    rows = []
    best = None
    for N in sizes:
        ho, step_s, gnn = train_size(pool[:N], held, cache, a.steps, a.lr,
                                     a.hidden, a.layers, dev)
        w.writerow([N, round(ho, 1), a.steps, round(step_s, 1), cache.builds,
                    round(cache.build_s, 1)])
        fh.flush()
        rows.append((N, ho))
        log(f"  N={N:5d} -> held-out {ho:.1f}  (train {step_s:.1f}s; cache: "
            f"{cache.builds} builds {cache.build_s:.0f}s total)")
        if best is None or ho < best[0]:
            best = (ho, N)
            torch.save(gnn.state_dict(), os.path.join(ROOT, a.save_best))
    fh.close()
    if len(rows) >= 2:
        d = rows[0][1] - rows[-1][1]
        rel = 100 * d / abs(rows[0][1]) if rows[0][1] else 0
        log(f"SCALING (congested): {rows[0][1]:.0f} (N={rows[0][0]}) -> "
            f"{rows[-1][1]:.0f} (N={rows[-1][0]}) = {rel:+.1f}% "
            f"({'HELPS' if d > 0 else 'flat'})")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-",
                    color="black", mfc="none")
            ax.set_xscale("log")
            ax.set_xlabel("# distinct congested training designs (log)")
            ax.set_ylabel("held-out objective")
            ax.set_title("Scaling on congested data (fixed held-out, cached)")
            f.tight_layout()
            fig = os.path.join(ROOT, "RESULTS", "figs",
                               "scaling_congested.pdf")
            os.makedirs(os.path.dirname(fig), exist_ok=True)
            f.savefig(fig)
            log(f"figure -> {fig}")
        except Exception as e:
            log(f"plot skipped: {e}")
    log(f"best held-out {best[0]:.1f} at N={best[1]} -> {a.save_best}")
    log(f"-> {out}")


if __name__ == "__main__":
    main()
