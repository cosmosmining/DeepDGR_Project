#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_scratch_benchmarks.py — synthesize global-routing benchmarks FROM
SCRATCH (no ISPD template, no LEF/DEF, no CUGR2 run), routable by
construction, in the exact `results`-dict format the whole existing pipeline
consumes (`main_stochastic.py` / `dgr_fast.py` --data_path,
`generate_e2e_graphs.build_pack_from_results` for training packs).

Why: every existing generator (compact, e2e_v1, e2e_v2) perturbs one of six
ISPD templates, so the dataset inherits their netlist statistics and needs
the benchmarks + CUGR2 preprocessing to exist.  This generator removes that
dependency entirely — infinite, license-free training data whose difficulty
is a knob.  (Precedent: the DGR paper itself, Table 1, validates on synthetic
grids of 20x20..1000x1000 with 20..100k nets.)

How routability is achieved (two layers of guarantee):
  1. CONSTRUCTION: nets are sampled with bounded windows (locality, like the
     bounded randomization of the template generators); a probabilistic
     L-route demand field D is accumulated (each 2-pin subnet spreads 1/2 on
     each of its two L paths); per-direction capacity is set to
        cap = blur(D)/utilization + floor,
     then pin/local-net demand (which main_stochastic SUBTRACTS from
     capacity) is measured via the real util.get_pin_demand/get_local_net
     and ADDED back on top, plus an epsilon margin.  So the effective
     capacity seen by the optimizer dominates the expected demand by
     1/utilization at every edge.
  2. CERTIFICATE (--certify): the bundled FastDGR solver (CPU) optimizes the
     instance and the exact discrete evaluator must reach overflow == 0,
     else the instance is rejected.  This is an existence proof of a
     zero-overflow routing within the L/Z/C candidate space.

Outputs per instance: <out>/<name>.pt (results dict) [+ --emit_pack: an
e2e_v1-format training pack] + a manifest row.  NEW file; data.py / util.py /
generate_e2e_graphs.py are imported READ-ONLY.

  # one small certified instance + training pack (CPU, ~1 min)
  python3 generate_scratch_benchmarks.py generate --out /ocean/.../scratch \
      --n 1 --xmax 80 --ymax 72 --n_nets 2000 --utilization 0.55 --certify
"""

import argparse
import copy
import csv
import json
import os
import sys
import time

import numpy as np
import torch

# tiny-op workloads thrash on many-core login nodes (measured: 4.4 s/iter on
# a 2.6k-candidate certify at default threads) — cap unless caller overrides
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from data import Net, Pin, routing_region, routing_region_3D   # read-only


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  Netlist sampling (clustered, chain-tree topology per the pool contract:
#  pin.parent_pin is an INT index into net.pins[tree]; root has None)
# ════════════════════════════════════════════════════════════════════════

def sample_netlist(rng, xmax, ymax, n_nets, max_pins=6, window_frac=0.06,
                   local_frac=0.03, pin_span=(1, 3)):
    """Returns list[Net].  Net sizes ~ geometric (2-pin dominant); pins
    clustered in a window around a net center; tree = nearest-neighbor
    chain; a small fraction are 1-pin local nets (demand realism)."""
    w = max(2, int(window_frac * min(xmax, ymax)))
    nets = []
    for nid in range(n_nets):
        name = f"snet_{nid}"
        if rng.rand() < local_frac:
            x, y = rng.randint(0, xmax), rng.randint(0, ymax)
            span = rng.randint(pin_span[0], pin_span[1] + 1)
            p = Pin(int(x), int(y), parent_pin=None,
                    physical_pin_layers=[1, span])
            nets.append(Net(name, nid, pins=[[p]], num_pins=1))
            continue
        k = min(2 + rng.geometric(0.55) - 1, max_pins)
        cx = rng.randint(0, xmax)
        cy = rng.randint(0, ymax)
        seen, pts = set(), []
        while len(pts) < k:
            x = int(np.clip(cx + rng.randint(-w, w + 1), 0, xmax - 1))
            y = int(np.clip(cy + rng.randint(-w, w + 1), 0, ymax - 1))
            if (x, y) not in seen:
                seen.add((x, y))
                pts.append((x, y))
        # nearest-neighbor chain ordering -> connected tree
        order = [0]
        rest = set(range(1, len(pts)))
        while rest:
            last = pts[order[-1]]
            nxt = min(rest, key=lambda j: abs(pts[j][0] - last[0])
                      + abs(pts[j][1] - last[1]))
            order.append(nxt)
            rest.remove(nxt)
        pins = []
        for rank, j in enumerate(order):
            span = rng.randint(pin_span[0], pin_span[1] + 1)
            pins.append(Pin(pts[j][0], pts[j][1],
                            parent_pin=None if rank == 0 else rank - 1,
                            physical_pin_layers=[1, span]))
        nets.append(Net(name, nid, pins=[pins], num_pins=len(pins)))
    return nets


# ════════════════════════════════════════════════════════════════════════
#  Probabilistic L-route demand field
#  (hor edges: (xmax, ymax-1) run along y; ver edges: (xmax-1, ymax))
# ════════════════════════════════════════════════════════════════════════

def demand_field(nets, xmax, ymax):
    D_h = np.zeros((xmax, ymax - 1), dtype=np.float64)
    D_v = np.zeros((xmax - 1, ymax), dtype=np.float64)
    for net in nets:
        tree = net.pins[0]
        for pin in tree:
            if pin.parent_pin is None:
                continue
            par = tree[pin.parent_pin]
            x1, x2 = sorted((pin.x, par.x))
            y1, y2 = sorted((pin.y, par.y))
            if y1 != y2:                       # horizontal travel (along y)
                if x1 == x2:
                    D_h[x1, y1:y2] += 1.0
                else:
                    D_h[pin.x, y1:y2] += 0.5
                    D_h[par.x, y1:y2] += 0.5
            if x1 != x2:                       # vertical travel (along x)
                if y1 == y2:
                    D_v[x1:x2, y1] += 1.0
                else:
                    D_v[x1:x2, pin.y] += 0.5
                    D_v[x1:x2, par.y] += 0.5
    return D_h, D_v


def box_blur(a, k=2):
    """Cheap separable mean blur with edge clamping (spread slack so the
    optimizer has room to detour around the expected paths)."""
    if k <= 0:
        return a
    out = a.copy()
    for axis in (0, 1):
        pad = np.concatenate([np.repeat(out.take([0], axis=axis), k, axis),
                              out,
                              np.repeat(out.take([-1], axis=axis), k, axis)],
                             axis=axis)
        c = np.cumsum(pad, axis=axis)
        n = 2 * k + 1
        out = (np.take(c, range(n - 1, pad.shape[axis]), axis=axis)
               - np.concatenate([np.zeros_like(out.take([0], axis=axis)),
                                 np.take(c, range(0, pad.shape[axis] - n),
                                         axis=axis)], axis=axis)) / n
    return out


# ════════════════════════════════════════════════════════════════════════
#  Benchmark assembly (results dict, exactly what the pipeline loads)
# ════════════════════════════════════════════════════════════════════════

LAYER_TEMPLATE = dict(layerMinLength=60.0, unit_length_short_cost=1.0)


def build_results(rng, xmax, ymax, n_nets, utilization=0.6, num_layer=4,
                  edge_len=3000.0, m2_pitch=300.0, margin=0.10,
                  blur_k=2, cap_floor=2.0, **net_kw):
    assert 0 < utilization < 1, "utilization must be in (0,1) for routability"
    nets = sample_netlist(rng, xmax, ymax, n_nets, **net_kw)
    D_h, D_v = demand_field(nets, xmax, ymax)

    cap_h = box_blur(D_h, blur_k) / utilization + cap_floor
    cap_v = box_blur(D_v, blur_k) / utilization + cap_floor

    n_hor_layers = (num_layer + 1) // 2          # hor_first=True: layers 0,2,..
    n_ver_layers = num_layer - n_hor_layers
    layers = [dict(LAYER_TEMPLATE) for _ in range(num_layer)]

    # measure the demand main_stochastic will SUBTRACT from capacity and add
    # it back on top (uses the real util functions, read-only)
    import util as dgr_util
    class _A:                                    # the arg fields they read
        pass
    a = _A()
    a.xmax, a.ymax, a.num_layer = xmax, ymax, num_layer
    a.hor_first, a.read_new_tree = True, False
    # post-swap edge_length convention: [0]=hor lengths (ymax-1), [1]=ver
    el_post = [np.full(ymax - 1, edge_len), np.full(xmax - 1, edge_len)]
    hor_ed, ver_ed, _, _ = dgr_util.get_pin_demand(nets, a, layers, el_post)
    hor_ld, ver_ld = dgr_util.get_local_net(nets, a)
    cap_h = (cap_h + hor_ed + hor_ld) * (1.0 + margin)
    cap_v = (cap_v + ver_ed + ver_ld) * (1.0 + margin)

    cap3d_h = [torch.tensor(cap_h / n_hor_layers, dtype=torch.float32)
               for _ in range(n_hor_layers)]
    cap3d_v = [torch.tensor(cap_v / n_ver_layers, dtype=torch.float32)
               for _ in range(n_ver_layers)]
    region3d = routing_region_3D(xmax, ymax, (cap3d_h, cap3d_v),
                                 hor_first=True)
    results = {
        "net": nets,
        "region": routing_region(xmax, ymax),
        "region3D": region3d,
        # ON-DISK (pre-swap) order: [ver(xmax-1), hor(ymax-1)] — the loader
        # swaps to [hor, ver] (main_stochastic.py:127 / preprocess_results)
        "edge_length": [torch.full((xmax - 1,), edge_len),
                        torch.full((ymax - 1,), edge_len)],
        "layers": layers,
        "m2_pitch": m2_pitch,
        "scratch_meta": {"generator": "scratch_v1", "utilization": utilization,
                         "n_nets": n_nets, "grid": [xmax, ymax],
                         "num_layer": num_layer, "margin": margin},
    }
    stats = {"n_nets": len(nets), "demand_h_max": float(D_h.max()),
             "demand_v_max": float(D_v.max()),
             "cap_h_mean": float(cap_h.mean()),
             "cap_h_std": float(cap_h.std())}
    return results, stats


# ════════════════════════════════════════════════════════════════════════
#  Routability certificate: FastDGR optimize + exact discrete refine -> OF 0
# ════════════════════════════════════════════════════════════════════════

def certify(results, iters=300, seed=0, verbose=False):
    """Returns (ok, info).  CPU-only; sized for small/medium instances."""
    from types import SimpleNamespace
    import data as dgr_data
    import util as dgr_util
    from generate_e2e_graphs import preprocess_results
    from fastdgr_core import ColMatrix, FullProblem
    from dgr_fast import build_full_problem, init_logits, optimize
    from discrete_refine import best_of_k_rounding, refine

    pre = preprocess_results(copy.deepcopy(results))
    a = pre["args"]
    pool = dgr_util.get_initial_candidate_pool(
        pre["RouteNets"], a.xmax, a.ymax, device="cpu",
        edge_length=pre["edge_length"], pattern_level=1,
        max_z=10, z_step=3, c_step=3, max_c=20, max_c_out_ratio=5)
    (p_index, _pif, _p2p, hor, ver, wl, via_info,
     *_r) = dgr_data.process_pool(pool, a.xmax, a.ymax, device="cpu")
    args = SimpleNamespace(device="cpu", act="sigmoid", act_scale=0.5,
                           celu_alpha=2.0, iter=iters, t=1.0, use_gumble=True,
                           overflow_coeff=1.0, wl_coeff=0.5, via_coeff=4.0,
                           optimizer="rmsprop", lr=0.8, weight_decay=0.0,
                           beta1=0.9, check_every=50, freeze_thresh=0.995,
                           prune_eps=0.02, patience=4, min_iter=100,
                           rel_tol=1e-4, seed=seed)
    full = build_full_problem(args, pre, p_index, hor, ver, wl, via_info)
    ap, fin, it_run, opt_s, _ = optimize(full, init_logits(full, seed), args)
    p_soft = ap.expand_full()
    sel, sc, _ = best_of_k_rounding(full, p_soft, k=8, of_coeff=1.0,
                                    wl_coeff=0.5, via_coeff=4.0)
    sel, sc, _ = refine(full, sel, 1.0, 0.5, 4.0, passes=4, max_moves=5000,
                        log=log if verbose else None)
    info = {"iters": it_run, "opt_s": round(opt_s, 1),
            "overflow_units": sc["overflow_units"],
            "max_overflow": sc["max_overflow"],
            "wl_cost": round(sc["wl_cost"], 1),
            "n_candidates": int(full.n), "n_subnets": int(full.S)}
    return sc["overflow_units"] == 0.0, info


# ════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════

def cmd_generate(args):
    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, "scratch_manifest.csv")
    for k in range(args.n):
        seed = args.base_seed + k
        rng = np.random.RandomState(seed * 7919 + 11)
        name = f"scratch_x{args.xmax}y{args.ymax}n{args.n_nets}_s{seed:04d}"
        fpath = os.path.join(args.out, name + ".pt")
        if os.path.exists(fpath) and not args.overwrite:
            log(f"[{k+1}/{args.n}] {name} exists — skip")
            continue
        t0 = time.time()
        results, stats = build_results(
            rng, args.xmax, args.ymax, args.n_nets,
            utilization=args.utilization, num_layer=args.num_layer,
            margin=args.margin, max_pins=args.max_pins,
            window_frac=args.window_frac)
        row = {"file": name + ".pt", "seed": seed, "xmax": args.xmax,
               "ymax": args.ymax, "n_nets": args.n_nets,
               "utilization": args.utilization, **stats}
        if args.certify:
            ok, info = certify(results, iters=args.certify_iters, seed=seed)
            row.update({f"cert_{k2}": v for k2, v in info.items()})
            row["certified_of0"] = int(ok)
            log(f"  certificate: overflow={info['overflow_units']} "
                f"({'PASS' if ok else 'REJECT'}; {info['opt_s']}s, "
                f"{info['n_candidates']:,} cands)")
            if not ok and not args.keep_uncertified:
                log(f"[{k+1}/{args.n}] {name} REJECTED — not saved")
                continue
        torch.save(results, fpath)
        if args.emit_pack:
            from generate_e2e_graphs import build_pack_from_results
            pack, pstats = build_pack_from_results(
                copy.deepcopy(results), benchmark=name, pattern_level=1,
                extra_meta={"generator": "scratch_v1", "seed": seed,
                            "utilization": args.utilization})
            torch.save(pack, os.path.join(args.out, name + "_e2e.pt"))
            row["pack_mb"] = round(os.path.getsize(
                os.path.join(args.out, name + "_e2e.pt")) / 2**20, 1)
        row["elapsed_s"] = round(time.time() - t0, 1)
        new = not os.path.exists(manifest)
        with open(manifest, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        log(f"[{k+1}/{args.n}] saved {name}.pt ({row['elapsed_s']}s)")
    log(f"manifest: {manifest}")


def cmd_inspect(args):
    d = torch.load(args.path, map_location="cpu", weights_only=False)
    print(json.dumps(d.get("scratch_meta", {}), indent=2))
    print("nets:", len(d["net"]), " grid:",
          d["region3D"].xmax, "x", d["region3D"].ymax)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--xmax", type=int, default=80)
    p.add_argument("--ymax", type=int, default=72)
    p.add_argument("--n_nets", type=int, default=2000)
    p.add_argument("--max_pins", type=int, default=6)
    p.add_argument("--window_frac", type=float, default=0.06)
    p.add_argument("--utilization", type=float, default=0.6,
                   help="difficulty knob; <1 keeps the construction routable")
    p.add_argument("--num_layer", type=int, default=4)
    p.add_argument("--margin", type=float, default=0.10)
    p.add_argument("--certify", action="store_true",
                   help="require a zero-overflow FastDGR solution (CPU)")
    p.add_argument("--certify_iters", type=int, default=300)
    p.add_argument("--keep_uncertified", action="store_true")
    p.add_argument("--emit_pack", action="store_true",
                   help="also write an e2e_v1-format training pack")
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("inspect")
    p.add_argument("path")
    args = ap.parse_args()
    {"generate": cmd_generate, "inspect": cmd_inspect}[args.cmd](args)


if __name__ == "__main__":
    main()
