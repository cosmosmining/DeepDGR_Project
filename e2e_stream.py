#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e2e_stream.py — a GENERALIZED *streaming* end-to-end DGR trainer.
===============================================================================

What the original (e2e_backprop.py / deepdgr_e2e.py) does
--------------------------------------------------------
It runs the e2e gradient feedback (GNN output replaces net.p; the DGR physical
objective backprops into the GNN weights) on **one** benchmark for `--iter`
steps.  Because the same instance is used at every optimizer step, the GNN
*overfits that one benchmark* — it learns an init that is good only there.

What THIS file does differently  (the round-robin)
--------------------------------------------------
One SHARED GNN, a POOL of N instances, and **each optimizer step uses a
DIFFERENT instance**:

    step t  ->  inst = pool[t % N]                         # <- round-robin
                logits = GNN(inst.graph)                    # GNN forward
                p      = gumbel_softmax(logits)             # per-subnet dist
                of,via,wl = inst.objective(p)               # DGR physics
                loss.backward();  opt.step()                # update SHARED GNN

So the gradient the GNN receives is averaged (across steps) over the whole
pool, not a single instance.  The GNN is therefore pushed toward a
*generalized* initialization that lowers the DGR objective on EVERY pool
instance, instead of memorizing one.  Instances are loaded **on demand** with a
small **LRU cache** so memory stays bounded no matter how big the pool is, and a
held-out split is scored periodically to measure generalization.

The e2e mechanism (lazy-conv init fix, gumbel-softmax over the per-subnet
segment index, FullProblem.objective_full) is reused verbatim from
e2e_backprop.py / fastdgr_core.py.

NEW file.  Imports are READ-ONLY:
    deepdgr_e2e.DeepDGR_GNN
    dgr_fast.load_benchmark, dgr_fast.build_full_problem
    fastdgr_core.FullProblem, ColMatrix, seg_softmax, seg_sum

CLI
---
  # real benchmarks (a glob of *.pt), GPU:
  python3 e2e_stream.py \
      --instances 'cu-gr-2/run/ispd1*_test*_metal5.pt' \
      --steps 4000 --lr 1e-2 --device 0 \
      --save_gnn stream_gnn.pth

  # tiny CPU smoke test (no .pt files needed):
  python3 e2e_stream.py --synthetic 4 --steps 40 --lr 5e-2 --device -1
"""

import argparse
import glob
import os
import sys
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from deepdgr_e2e import DeepDGR_GNN                       # read-only (class)
from dgr_fast import load_benchmark, build_full_problem   # read-only
from fastdgr_core import (ColMatrix, FullProblem,         # read-only
                          seg_softmax, seg_sum)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  An instance = (DGR FullProblem objective) + (hetero graph for the GNN)
# ════════════════════════════════════════════════════════════════════════

class Instance:
    """One routable problem: a fastdgr_core.FullProblem (the DGR physics) plus
    the GNN's input graph (x_dict + edge_index_dict).  Carries everything the
    streaming loop needs and nothing else, so it can be dropped from the LRU
    cache to free memory."""

    def __init__(self, name, full, x_dict, edge_index_dict, grid_in, cand_in,
                 wl_coeff, via_coeff):
        self.name = name
        self.full = full
        self.x_dict = x_dict
        self.edge_index_dict = edge_index_dict
        self.grid_in = grid_in
        self.cand_in = cand_in
        self.wl_coeff = wl_coeff
        self.via_coeff = via_coeff
        self.seg = full.seg                  # per-candidate subnet id
        self.n = full.n
        self.S = full.S

    def loss(self, gnn, gen, temp):
        """One forward: GNN -> gumbel-softmax per subnet -> DGR objective.
        Identical math to e2e_backprop.py:120-126."""
        logits = gnn(self.x_dict, self.edge_index_dict)          # GNN forward
        gp = -torch.log(-torch.log(
            torch.rand_like(logits).clamp_min(1e-10)))           # gumbel noise
        p = seg_softmax((logits + gp) / max(temp, 0.05),
                        self.seg, self.S)
        of, via_c, wl_c, _ = self.full.objective_full(p)         # DGR physics
        loss = of + self.wl_coeff * wl_c + self.via_coeff * via_c
        return loss, of, via_c, wl_c

    @torch.no_grad()
    def eval_loss(self, gnn):
        """Noiseless objective (no gumbel) — generalization metric."""
        logits = gnn(self.x_dict, self.edge_index_dict)
        p = seg_softmax(logits, self.seg, self.S)
        of, via_c, wl_c, _ = self.full.objective_full(p)
        return float(of + self.wl_coeff * wl_c + self.via_coeff * via_c), \
            float(of)


# ════════════════════════════════════════════════════════════════════════
#  Loaders: real .pt benchmarks AND tiny in-file synthetic problems
# ════════════════════════════════════════════════════════════════════════

def graph_to_dicts(gpath, device):
    """*_graph.pt (dict with 'graph'=HeteroData) -> dicts; matches
    e2e_backprop.graph_to_dicts / deepdgr_e2e.py:313-327."""
    gd = torch.load(gpath, map_location="cpu", weights_only=False)
    graph = gd["graph"]
    x_dict = {k: graph[k].x.float().to(device) for k in ("grid", "candidate")}
    ei = {et: graph[et].edge_index.long().to(device)
          for et in graph.edge_types}
    return x_dict, ei, x_dict["grid"].shape[1], x_dict["candidate"].shape[1]


def candidate_features(full):
    """Per-candidate features straight off the FullProblem (so a graph is not
    required to exist on disk).  [wire_length, via_count, subnet_width,
    in-subnet-rank] — all cheap, all per candidate, no benchmark re-parse."""
    dev = full.device
    wl = full.wire_length.float().view(-1, 1)
    vc = full.via_count.float().view(-1, 1)
    widths = (full.p_index[1:] - full.p_index[:-1]).float()
    width_per_cand = widths[full.seg].view(-1, 1)
    rank = (torch.arange(full.n, device=dev).float()
            - full.p_index[:-1][full.seg].float()).view(-1, 1)
    x = torch.cat([wl, vc, width_per_cand, rank], dim=1)
    # normalize columns for stable GNN inputs
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
    return x.to(dev)


def grid_features(full):
    """Per-grid-cell features: [hor_cap, ver_cap, hor_pin_demand,
    ver_pin_demand] flattened over the xmax*ymax cells."""
    dev = full.device
    G = full.xmax * full.ymax
    hc = torch.zeros(G, device=dev)
    vc = torch.zeros(G, device=dev)
    hc[:full.hor_cap.numel()] = full.hor_cap[:G]
    vc[:full.ver_cap.numel()] = full.ver_cap[:G]
    hp = full.hor_pin_demand.reshape(-1)[:G]
    vp = full.ver_pin_demand.reshape(-1)[:G]
    x = torch.stack([hc, vc, hp, vp], dim=1)
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
    return x.to(dev)


def derive_graph_from_full(full):
    """Build the 4-edge-type hetero graph the DeepDGR_GNN expects directly from
    a FullProblem — used for synthetic instances and as a fallback when no
    *_graph.pt is on disk.  Edge types exactly match DeepDGR_GNN.convs:
      ('grid','connects','grid'), ('candidate','passes_through','grid'),
      ('grid','influences','candidate'), ('candidate','competes','candidate').
    """
    dev = full.device
    G = full.xmax * full.ymax
    x_grid = grid_features(full)
    x_cand = candidate_features(full)

    # candidate <-> grid: a candidate "passes through" the grid cells whose
    # via/hor/ver columns it occupies (use the via map: cell == flat grid id).
    via = full.via                                   # ColMatrix (G, n)
    cols = torch.arange(full.n, device=dev)
    rows, _, owner = via.gather_cols(cols)           # rows=grid cell, owner=cand
    pt = torch.stack([owner.to(dev), rows.to(dev)])  # candidate -> grid
    infl = torch.stack([rows.to(dev), owner.to(dev)])  # grid -> candidate

    # grid 'connects' grid: 4-neighbour lattice (row-major flat ids).
    xs = torch.arange(full.xmax, device=dev)
    ys = torch.arange(full.ymax, device=dev)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    flat = (gx * full.ymax + gy).reshape(-1)
    right = flat[(gy < full.ymax - 1).reshape(-1)]
    down = flat[(gx < full.xmax - 1).reshape(-1)]
    conn = torch.cat([
        torch.stack([right, right + 1]),
        torch.stack([down, down + full.ymax])], dim=1)
    conn = torch.cat([conn, conn.flip(0)], dim=1)    # undirected

    # candidate 'competes' candidate: candidates of the same subnet.  Built
    # per-subnet from p_index offsets (no dense n*n matrix, so it scales to
    # real benchmarks): for each subnet of width w, all w*(w-1) ordered pairs.
    widths = (full.p_index[1:] - full.p_index[:-1]).to(dev)
    starts = full.p_index[:-1].to(dev)
    src_list, dst_list = [], []
    for w in torch.unique(widths):
        w = int(w)
        if w < 2:
            continue
        st = starts[widths == w]                     # subnet start offsets
        base = torch.arange(w, device=dev)
        a = base.view(-1, 1).expand(w, w).reshape(-1)
        b = base.view(1, -1).expand(w, w).reshape(-1)
        keep = a != b                                # drop self-loops
        a, b = a[keep], b[keep]
        src_list.append((st.view(-1, 1) + a.view(1, -1)).reshape(-1))
        dst_list.append((st.view(-1, 1) + b.view(1, -1)).reshape(-1))
    comp = (torch.stack([torch.cat(src_list), torch.cat(dst_list)])
            if src_list else
            torch.zeros(2, 0, dtype=torch.long, device=dev))

    x_dict = {"grid": x_grid, "candidate": x_cand}
    ei = {
        ("grid", "connects", "grid"): conn.long(),
        ("candidate", "passes_through", "grid"): pt.long(),
        ("grid", "influences", "candidate"): infl.long(),
        ("candidate", "competes", "candidate"): comp.long(),
    }
    return x_dict, ei, x_grid.shape[1], x_cand.shape[1]


def make_synthetic_full(seed, device, xmax=6, ymax=6, n_subnets=12,
                        cand_per=3):
    """A tiny, self-contained FullProblem for CPU smoke tests.  Random sparse
    hor/ver/via columns + capacities -> a real (if small) DGR objective with
    the SAME math as a parsed benchmark.  No candidate-pool generation, no big
    .pt load -> fast and CPU-friendly."""
    g = torch.Generator().manual_seed(seed)
    Eh = xmax * (ymax - 1)
    Ev = (xmax - 1) * ymax
    G = xmax * ymax
    n = n_subnets * cand_per
    p_index = torch.arange(0, n + 1, cand_per, dtype=torch.long)

    def rand_coo(rows_dim, nnz_per_col):
        ri, ci, vv = [], [], []
        for c in range(n):
            r = torch.randint(0, rows_dim, (nnz_per_col,), generator=g)
            ri.append(r)
            ci.append(torch.full((nnz_per_col,), c))
            vv.append(torch.rand(nnz_per_col, generator=g) * 0.8 + 0.2)
        idx = torch.stack([torch.cat(ri), torch.cat(ci)])
        val = torch.cat(vv)
        return torch.sparse_coo_tensor(idx, val, (rows_dim, n)).coalesce()

    hor = rand_coo(Eh, 2)
    ver = rand_coo(Ev, 2)
    via = rand_coo(G, 2)
    wire_length = (torch.rand(n, generator=g) * 5 + 1)
    via_count = torch.randint(1, 4, (n,), generator=g).float()
    # capacities tight enough that overflow is non-trivial (so loss can move)
    hor_cap = torch.full((Eh,), 1.0)
    ver_cap = torch.full((Ev,), 1.0)
    w_h = torch.ones(Eh)
    w_v = torch.ones(Ev)
    hor_pin = torch.rand(xmax, ymax, generator=g) * 0.1
    ver_pin = torch.rand(xmax, ymax, generator=g) * 0.1

    full = FullProblem(
        xmax=xmax, ymax=ymax,
        hor=ColMatrix.from_coo(hor, device=device),
        ver=ColMatrix.from_coo(ver, device=device),
        via=ColMatrix.from_coo(via, device=device),
        wire_length=wire_length.to(device), via_count=via_count.to(device),
        p_index=p_index,
        hor_cap=hor_cap.to(device), ver_cap=ver_cap.to(device),
        w_h=w_h.to(device), w_v=w_v.to(device),
        hor_pin_demand=hor_pin.to(device), ver_pin_demand=ver_pin.to(device),
        via_layer=1.5, m2_pitch=1.0, act="sigmoid", act_scale=0.5,
        celu_alpha=2.0, add_via=True, device=device)
    return full


def build_instance_real(pt_path, device, wl_coeff, via_coeff,
                        pattern_level, max_c):
    """Load a real benchmark .pt -> Instance, exactly as e2e_backprop.py does
    (load_benchmark + build_full_problem), and attach its hetero graph.  Prefer
    an on-disk <stem>_graph.pt; otherwise derive the graph from the problem."""
    args = SimpleNamespace(
        data_path=pt_path, device=device, capacity=1.0, pin_ratio=1.0,
        local_net_ratio=1.0, via_layer=1.5, pattern_level=pattern_level,
        z_step=3, max_z=10, c_step=3, max_c=max_c, max_c_out_ratio=5,
        act="sigmoid", act_scale=0.5, celu_alpha=2.0)
    pre, name, _pool, p_index, _pif, _p2pat, hor, ver, wl, via = \
        load_benchmark(args)
    full = build_full_problem(args, pre, p_index, hor, ver, wl, via)

    stem = pt_path.rsplit(".", 1)[0]
    gpath = stem + "_graph.pt"
    gp2 = os.path.join(ROOT, os.path.basename(stem) + "_graph.pt")
    chosen = gpath if os.path.isfile(gpath) else (
        gp2 if os.path.isfile(gp2) else None)
    if chosen is not None:
        try:
            x_dict, ei, gin, cin = graph_to_dicts(chosen, device)
            if x_dict["candidate"].shape[0] == full.n:
                log(f"  {name}: graph {os.path.basename(chosen)} "
                    f"({full.n:,} cand / {full.S:,} subnets)")
                return Instance(name, full, x_dict, ei, gin, cin,
                                wl_coeff, via_coeff)
            log(f"  {name}: on-disk graph candidate count "
                f"{x_dict['candidate'].shape[0]} != {full.n} -> derive")
        except Exception as e:                          # pragma: no cover
            log(f"  {name}: graph load failed ({e}) -> derive")
    x_dict, ei, gin, cin = derive_graph_from_full(full)
    log(f"  {name}: derived graph ({full.n:,} cand / {full.S:,} subnets)")
    return Instance(name, full, x_dict, ei, gin, cin, wl_coeff, via_coeff)


# ════════════════════════════════════════════════════════════════════════
#  LRU pool: load-on-demand, bounded memory
# ════════════════════════════════════════════════════════════════════════

class InstancePool:
    """Round-robin pool with an LRU cache.  `specs[i]` is a (kind, payload)
    descriptor; the heavy Instance is materialized only when requested and at
    most `cache` of them are kept resident."""

    def __init__(self, specs, device, wl_coeff, via_coeff, pattern_level,
                 max_c, cache=3):
        self.specs = specs
        self.device = device
        self.wl_coeff = wl_coeff
        self.via_coeff = via_coeff
        self.pattern_level = pattern_level
        self.max_c = max_c
        self.cache = max(1, cache)
        self._lru = OrderedDict()                       # idx -> Instance
        self.n_load = 0

    def __len__(self):
        return len(self.specs)

    def _materialize(self, i):
        kind, payload = self.specs[i]
        if kind == "pt":
            return build_instance_real(payload, self.device, self.wl_coeff,
                                       self.via_coeff, self.pattern_level,
                                       self.max_c)
        # synthetic
        full = make_synthetic_full(seed=payload, device=self.device)
        x_dict, ei, gin, cin = derive_graph_from_full(full)
        return Instance(f"synth{payload}", full, x_dict, ei, gin, cin,
                        self.wl_coeff, self.via_coeff)

    def get(self, i):
        if i in self._lru:
            self._lru.move_to_end(i)
            return self._lru[i]
        inst = self._materialize(i)
        self.n_load += 1
        self._lru[i] = inst
        self._lru.move_to_end(i)
        while len(self._lru) > self.cache:              # evict LRU
            old_i, old_inst = self._lru.popitem(last=False)
            del old_inst
            if str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
        return inst

    def feature_dims(self):
        """Peek the first instance to size the GNN (grid_in / cand_in)."""
        inst = self.get(0)
        return inst.grid_in, inst.cand_in


# ════════════════════════════════════════════════════════════════════════
#  Streaming trainer
# ════════════════════════════════════════════════════════════════════════

def init_gnn(pool, hidden, layers, device, load_gnn=None):
    gin, cin = pool.feature_dims()
    gnn = DeepDGR_GNN(grid_in=gin, cand_in=cin, hidden=hidden,
                      num_layers=layers).to(device)
    with torch.no_grad():                               # init lazy SAGEConv
        inst = pool.get(0)
        _ = gnn(inst.x_dict, inst.edge_index_dict)
    n_params = sum(p.numel() for p in gnn.parameters())
    log(f"GNN initialised (lazy convs built): {n_params:,} params "
        f"(grid_in={gin}, cand_in={cin})")
    if load_gnn and os.path.isfile(load_gnn):
        ck = torch.load(load_gnn, map_location=device, weights_only=False)
        sd = ck.get("gnn_state_dict") or ck.get("model_state_dict") or ck
        miss = gnn.load_state_dict(sd, strict=False)
        log(f"loaded checkpoint {os.path.basename(load_gnn)} "
            f"(missing {len(miss.missing_keys)})")
    return gnn


def evaluate(gnn, pool, holdout_idx):
    gnn.eval()
    tot, ofs = [], []
    for i in holdout_idx:
        ev, of = pool.get(i).eval_loss(gnn)
        tot.append(ev)
        ofs.append(of)
    gnn.train()
    return float(np.mean(tot)), float(np.mean(ofs))


def run(args):
    dev = (f"cuda:{args.device}" if torch.cuda.is_available() and
           args.device >= 0 else "cpu")
    torch.manual_seed(args.seed)

    # --- build the spec list (round-robin order) ---
    specs = []
    if args.synthetic > 0:
        for k in range(args.synthetic):
            specs.append(("synth", 100 + k))
        log(f"synthetic pool: {args.synthetic} tiny in-file problems")
    if args.instances:
        files = sorted(glob.glob(args.instances))
        if not files:
            log(f"WARNING: glob '{args.instances}' matched no files")
        for f in files:
            specs.append(("pt", f))
        log(f"benchmark pool: {len(files)} file(s) from '{args.instances}'")
    if not specs:
        raise SystemExit("no instances: pass --instances <glob> or --synthetic N")

    # held-out split (last `holdout` specs are never trained on)
    n = len(specs)
    holdout = min(args.holdout, max(0, n - 1)) if n > 1 else 0
    train_idx = list(range(n - holdout)) if holdout else list(range(n))
    holdout_idx = list(range(n - holdout, n)) if holdout else []
    log(f"pool size {n}: {len(train_idx)} train, {len(holdout_idx)} held-out")

    pool = InstancePool(specs, dev, args.wl_coeff, args.via_coeff,
                        args.pattern_level, args.max_c, cache=args.cache)

    gnn = init_gnn(pool, args.hidden, args.layers, dev, args.load_gnn)
    opt = torch.optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=1e-5)
    gen = (torch.Generator(device=dev).manual_seed(args.seed)
           if dev.startswith("cuda") else
           torch.Generator().manual_seed(args.seed))

    log("=" * 72)
    log("STREAMING e2e: each STEP uses a DIFFERENT instance (round-robin); "
        "ONE shared GNN learns a generalized init")
    log("=" * 72)

    temp = 1.0
    per_inst_first = {}
    per_inst_last = {}
    history = []
    t0 = time.time()
    for t in range(args.steps):
        if t and t % max(1, args.steps // 10) == 0:
            temp = max(temp * 0.9, 0.1)
        # ---- round-robin instance selection (THE difference vs the original)
        ti = train_idx[t % len(train_idx)]
        inst = pool.get(ti)

        loss, of, via_c, wl_c = inst.loss(gnn, gen, temp)
        opt.zero_grad()
        loss.backward()
        gnorm = torch.sqrt(sum((pp.grad.detach() ** 2).sum()
                               for pp in gnn.parameters()
                               if pp.grad is not None))
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 5.0)
        opt.step()

        lv = float(loss)
        per_inst_first.setdefault(inst.name, lv)
        per_inst_last[inst.name] = lv
        history.append((t, inst.name, lv))
        if t % args.log_every == 0 or t == args.steps - 1:
            log(f"  step {t:5d}  inst={inst.name:<24s} loss={lv:.5g}  "
                f"of={float(of):.4g}  gnorm={float(gnorm):.4g}  T={temp:.3f}")
        if holdout_idx and args.eval_every > 0 and \
                (t % args.eval_every == 0 or t == args.steps - 1):
            ev, evof = evaluate(gnn, pool, holdout_idx)
            log(f"    [generalization @ step {t}] held-out mean loss={ev:.5g} "
                f"(overflow={evof:.4g}) over {len(holdout_idx)} instance(s)")

    dt = time.time() - t0
    log("=" * 72)
    log(f"done: {args.steps} steps in {dt:.1f}s ({args.steps/max(dt,1e-9):.1f} "
        f"steps/s); pool materializations={pool.n_load} (cache={args.cache})")
    log("per-instance loss (first time trained -> last time trained):")
    for nm in sorted(per_inst_first):
        f, l = per_inst_first[nm], per_inst_last[nm]
        log(f"   {nm:<26s} {f:.5g} -> {l:.5g} "
            f"({100*(l-f)/abs(f) if f else 0:+.1f}%)")
    if holdout_idx:
        ev, evof = evaluate(gnn, pool, holdout_idx)
        log(f"FINAL held-out generalization: mean loss={ev:.5g} "
            f"(overflow={evof:.4g})")

    if args.save_gnn:
        torch.save({"gnn_state_dict": gnn.state_dict(),
                    "args": vars(args),
                    "pool": [s[1] for s in specs]}, args.save_gnn)
        log(f"shared GNN saved: {args.save_gnn}")
    return gnn, history


def build_parser():
    p = argparse.ArgumentParser(
        description="Generalized streaming e2e DGR trainer (round-robin pool)")
    p.add_argument("--instances", default=None,
                   help="glob of benchmark .pt files (the round-robin pool)")
    p.add_argument("--synthetic", type=int, default=0,
                   help="N tiny in-file synthetic problems (CPU smoke test)")
    p.add_argument("--steps", type=int, default=2000,
                   help="optimizer steps; each step uses one pool instance")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--via_coeff", type=float, default=4.0)
    p.add_argument("--wl_coeff", type=float, default=0.5)
    p.add_argument("--pattern_level", type=int, default=1)
    p.add_argument("--max_c", type=int, default=20)
    p.add_argument("--cache", type=int, default=3,
                   help="LRU cache size (resident instances)")
    p.add_argument("--holdout", type=int, default=1,
                   help="last K pool instances reserved for generalization eval")
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--load_gnn", default=None)
    p.add_argument("--save_gnn", default=None)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
