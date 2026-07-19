#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ispd24_adapter.py — convert ISPD'24 contest "Simple_inputs" (.cap/.net) into
the `results`-dict this repo's whole pipeline consumes (FastDGR /
main_stochastic --data_path, generate_e2e_graphs.build_pack_from_results).
No LEF/DEF, no CUGR2 preprocessing needed — the contest format IS the
GCell-grid abstraction we already operate on.

Format implemented from the authoritative parser in InstantGR
(github.com/cuhk-eda/InstantGR, src/database.hpp db::read):

  .cap:  L X Y
         unit_length_wire_cost unit_via_cost
         unit_length_short_costs[0..L-1]
         x_edge_len[0..X-2]
         y_edge_len[0..Y-2]
         L blocks:  name dir min_len
                    capacity rows: for y in 0..Y-1: for x in 0..X-1: cap[l][x][y]
  .net:  NetName
         (
           [(l,x,y), (l,x,y), ...]     # one bracket group per pin
           ...
         )

Conventions mapped to this repo (NOTE the repo's unusual naming: its "hor"
edge tensor (xmax, ymax-1) runs ALONG Y; "ver" (xmax-1, ymax) runs ALONG X):
  * a layer routing along X contributes to the repo's "ver" stack, along Y
    to "hor".  `--dir_along_x 0` (default) means .cap dir==0 routes along X;
    pass 1 to flip — the validator prints per-direction capacity totals so a
    wrong guess is immediately visible.
  * ISPD layer 0 (unroutable; capacity normally 0) is dropped; remaining
    L-1 layers become the routing stack; pin layers are clamped to >= 1.
  * per-cell capacity[l][x][y] is treated as the capacity of the edge from
    (x,y) to the next cell in the layer's direction (contest convention);
    the trailing row/column is dropped accordingly.
  * on-disk edge_length order is the repo's pre-swap [ver(X-1), hor(Y-1)]
    = [x_edge_len, y_edge_len].
  * m2_pitch is not in .cap: default = min positive edge length (WL-cost
    scale only; override with --m2_pitch).

NEW file; data.py imported read-only.  Validate-first workflow:
  python3 ispd24_adapter.py convert --cap X.cap --net X.net --out X.pt
  python3 ispd24_adapter.py convert ... --validate   # stats, no write
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from data import Net, Pin, routing_region, routing_region_3D   # read-only


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  .cap parsing
# ════════════════════════════════════════════════════════════════════════

def parse_cap(path):
    toks = iter(open(path).read().split())

    def ni():
        return int(next(toks))

    def nf():
        return float(next(toks))

    L, X, Y = ni(), ni(), ni()
    unit_wire, unit_via = nf(), nf()
    short_costs = [nf() for _ in range(L)]
    x_edge_len = np.array([nf() for _ in range(X - 1)])
    y_edge_len = np.array([nf() for _ in range(Y - 1)])
    layers = []
    for _l in range(L):
        name = next(toks)
        dir_ = ni()
        min_len = nf()
        cap = np.empty((X, Y), dtype=np.float64)
        for y in range(Y):
            for x in range(X):
                cap[x, y] = nf()
        layers.append({"name": name, "dir": dir_, "min_len": min_len,
                       "cap": cap})
    leftover = sum(1 for _ in toks)
    assert leftover == 0, f".cap parse incomplete: {leftover} tokens left"
    return {"L": L, "X": X, "Y": Y, "unit_wire": unit_wire,
            "unit_via": unit_via, "short_costs": short_costs,
            "x_edge_len": x_edge_len, "y_edge_len": y_edge_len,
            "layers": layers}


# ════════════════════════════════════════════════════════════════════════
#  .net parsing
# ════════════════════════════════════════════════════════════════════════

_AP = re.compile(r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")


def parse_net(path):
    """Returns list of (net_name, [pin][ (l,x,y), ... ])."""
    nets, name, pins, depth = [], None, None, 0
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line == "(":
                depth += 1
                continue
            if line == ")":
                depth -= 1
                if depth == 0 and name is not None:
                    nets.append((name, pins))
                    name, pins = None, None
                continue
            if depth == 0:
                name, pins = line.split()[0], []
            else:
                aps = [(int(a), int(b), int(c))
                       for a, b, c in _AP.findall(line)]
                if aps:
                    pins.append(aps)
    return nets


# ════════════════════════════════════════════════════════════════════════
#  Conversion
# ════════════════════════════════════════════════════════════════════════

def convert(cap, nets_raw, dir_along_x=0, m2_pitch=None):
    X, Y = cap["X"], cap["Y"]
    routing = cap["layers"][1:]                 # drop unroutable layer 0
    num_layer = len(routing)
    hor_stack, ver_stack, layer_meta = [], [], []
    is_hor_flags = []
    for li, lay in enumerate(routing):
        along_x = (lay["dir"] == dir_along_x)
        if along_x:                              # repo "ver": (X-1, Y)
            ver_stack.append(torch.tensor(lay["cap"][:-1, :],
                                          dtype=torch.float32))
            is_hor_flags.append(False)
        else:                                    # repo "hor": (X, Y-1)
            hor_stack.append(torch.tensor(lay["cap"][:, :-1],
                                          dtype=torch.float32))
            is_hor_flags.append(True)
        layer_meta.append({"layerMinLength": lay["min_len"],
                           "unit_length_short_cost": cap["short_costs"][li + 1],
                           "name": lay["name"]})
    assert hor_stack and ver_stack, \
        "one routing direction is empty — wrong --dir_along_x?"
    hor_first = is_hor_flags[0]

    nets = []
    nid = 0
    n_multi_ap = 0
    for name, pin_groups in nets_raw:
        gpins, seen = [], set()
        for aps in pin_groups:
            n_multi_ap += len(aps) > 1
            l0, x0, y0 = min(aps)                # lowest-layer access point
            lmin = max(1, min(a[0] for a in aps))
            lmax = max(1, max(a[0] for a in aps))
            if (x0, y0) in seen:
                continue
            seen.add((x0, y0))
            gpins.append(Pin(int(x0), int(y0), parent_pin=None,
                             physical_pin_layers=[lmin, lmax]))
        if not gpins:
            continue
        if len(gpins) > 1:                       # nearest-neighbor chain tree
            order = [0]
            rest = set(range(1, len(gpins)))
            while rest:
                lp = gpins[order[-1]]
                nxt = min(rest, key=lambda j: abs(gpins[j].x - lp.x)
                          + abs(gpins[j].y - lp.y))
                order.append(nxt)
                rest.remove(nxt)
            gpins = [gpins[j] for j in order]
            for rank, p in enumerate(gpins):
                p.parent_pin = None if rank == 0 else rank - 1
        nets.append(Net(name, nid, pins=[gpins], num_pins=len(gpins)))
        nid += 1

    m2 = float(m2_pitch if m2_pitch else
               min(cap["x_edge_len"].min(), cap["y_edge_len"].min()))
    results = {
        "net": nets,
        "region": routing_region(X, Y),
        "region3D": routing_region_3D(X, Y, (hor_stack, ver_stack),
                                      hor_first=hor_first),
        # repo pre-swap order: [ver lengths (X-1) = x_edge_len,
        #                       hor lengths (Y-1) = y_edge_len]
        "edge_length": [torch.tensor(cap["x_edge_len"], dtype=torch.float32),
                        torch.tensor(cap["y_edge_len"], dtype=torch.float32)],
        "layers": layer_meta,
        "m2_pitch": m2,
        "ispd24_meta": {"source_format": "ispd24_simple_inputs",
                        "L_raw": cap["L"], "grid": [X, Y],
                        "num_routing_layer": num_layer,
                        "dir_along_x": dir_along_x,
                        "unit_wire": cap["unit_wire"],
                        "unit_via": cap["unit_via"],
                        "layer0_cap_sum": float(cap["layers"][0]["cap"].sum()),
                        "n_nets": len(nets),
                        "pins_with_multi_ap": int(n_multi_ap)},
    }
    return results


def validate_report(cap, results):
    r3d = results["region3D"]
    h = torch.stack(r3d.cap_mat_3D[0]).sum(0)
    v = torch.stack(r3d.cap_mat_3D[1]).sum(0)
    meta = results["ispd24_meta"]
    sizes = [len(n.pins[0]) for n in results["net"]]
    rep = {
        "grid": meta["grid"], "routing_layers": meta["num_routing_layer"],
        "hor_first": bool(r3d.hor_first),
        "hor_cap": {"shape": list(h.shape), "mean": round(float(h.mean()), 3),
                    "std": round(float(h.std()), 3)},
        "ver_cap": {"shape": list(v.shape), "mean": round(float(v.mean()), 3),
                    "std": round(float(v.std()), 3)},
        "layer0_cap_sum_dropped": meta["layer0_cap_sum"],
        "nets": len(sizes), "pins_mean": round(float(np.mean(sizes)), 2),
        "pins_max": int(max(sizes)), "m2_pitch": results["m2_pitch"],
    }
    print(json.dumps(rep, indent=2))
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("convert")
    p.add_argument("--cap", required=True)
    p.add_argument("--net", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--dir_along_x", type=int, default=0,
                   help=".cap dir value that means routing along X (def 0)")
    p.add_argument("--m2_pitch", type=float, default=None)
    p.add_argument("--max_nets", type=int, default=0,
                   help="truncate netlist (0 = all) — for first-contact runs")
    p.add_argument("--validate", action="store_true",
                   help="parse + report only, write nothing")
    args = ap.parse_args()

    t0 = time.time()
    cap = parse_cap(args.cap)
    log(f".cap: L={cap['L']} grid {cap['X']}x{cap['Y']} "
        f"({time.time()-t0:.1f}s)")
    t0 = time.time()
    nets_raw = parse_net(args.net)
    log(f".net: {len(nets_raw):,} nets ({time.time()-t0:.1f}s)")
    if args.max_nets:
        nets_raw = nets_raw[:args.max_nets]
    results = convert(cap, nets_raw, dir_along_x=args.dir_along_x,
                      m2_pitch=args.m2_pitch)
    validate_report(cap, results)
    if not args.validate:
        out = args.out or os.path.splitext(args.cap)[0] + ".pt"
        torch.save(results, out)
        log(f"saved {out} ({os.path.getsize(out)/2**20:.1f} MB)")


if __name__ == "__main__":
    main()
