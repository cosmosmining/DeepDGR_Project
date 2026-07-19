#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ispd24_adapter.py — CPU self-test for ispd24_adapter.py against a
hand-crafted mini benchmark written in the EXACT InstantGR read order
(github.com/cuhk-eda/InstantGR src/database.hpp).  Then the converted dict is
pushed through the real pipeline preamble (preprocess_results) and the
FastDGR certificate.  Run: python3 test_ispd24_adapter.py
"""
import os
import sys
import tempfile

import numpy as np
import torch

torch.set_num_threads(2)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ispd24_adapter import parse_cap, parse_net, convert

PASS = FAIL = 0


def check(name, cond):
    global PASS, FAIL
    PASS, FAIL = PASS + cond, FAIL + (not cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")


X, Y, L = 6, 5, 3                       # layer0 unroutable + 1 along-x + 1 along-y
cap_lines = [f"{L} {X} {Y}", "0.5 4.0", "100 200 300",
             " ".join(["10"] * (X - 1)), " ".join(["20"] * (Y - 1))]
rng = np.random.RandomState(0)
grids = []
for l, (nm, d) in enumerate([("M0", 0), ("M1", 0), ("M2", 1)]):
    cap_lines.append(f"{nm} {d} 3.0")
    g = np.zeros((X, Y)) if l == 0 else rng.uniform(4, 8, (X, Y)).round(1)
    grids.append(g)
    for y in range(Y):
        cap_lines.append(" ".join(str(g[x, y]) for x in range(X)))
net_lines = ["netA", "(", "[(1, 1, 1), (2, 1, 1)]", "[(1, 4, 3)]", ")",
             "netB", "(", "[(1, 0, 4)]", "[(1, 5, 0)]", "[(2, 3, 2)]", ")",
             "netLocal", "(", "[(1, 2, 2)]", ")"]

with tempfile.TemporaryDirectory() as td:
    cp, npth = os.path.join(td, "m.cap"), os.path.join(td, "m.net")
    open(cp, "w").write("\n".join(cap_lines) + "\n")
    open(npth, "w").write("\n".join(net_lines) + "\n")

    print("== T1: parse ==")
    cap = parse_cap(cp)
    check("header", (cap["L"], cap["X"], cap["Y"]) == (L, X, Y))
    check("costs", cap["unit_wire"] == 0.5 and cap["unit_via"] == 4.0
          and cap["short_costs"] == [100.0, 200.0, 300.0])
    check("edge lens", len(cap["x_edge_len"]) == X - 1
          and len(cap["y_edge_len"]) == Y - 1)
    check("grid values round-trip",
          all(np.allclose(cap["layers"][l]["cap"], grids[l])
              for l in range(L)))
    nets_raw = parse_net(npth)
    check("3 nets, pin counts", [len(p) for _, p in nets_raw] == [2, 3, 1])
    check("multi-AP pin kept once", nets_raw[0][1][0] == [(1, 1, 1), (2, 1, 1)])

    print("== T2: convert ==")
    res = convert(cap, nets_raw, dir_along_x=0)
    r3d = res["region3D"]
    check("layer0 dropped, 2 routing layers",
          res["ispd24_meta"]["num_routing_layer"] == 2)
    check("along-x layer -> repo ver stack (X-1,Y)",
          len(r3d.cap_mat_3D[1]) == 1
          and tuple(r3d.cap_mat_3D[1][0].shape) == (X - 1, Y))
    check("along-y layer -> repo hor stack (X,Y-1)",
          len(r3d.cap_mat_3D[0]) == 1
          and tuple(r3d.cap_mat_3D[0][0].shape) == (X, Y - 1))
    check("ver cap values = M1 grid minus last col",
          torch.allclose(r3d.cap_mat_3D[1][0],
                         torch.tensor(grids[1][:-1, :], dtype=torch.float32)))
    check("hor_first reflects first routing layer (along-x -> ver -> False)",
          r3d.hor_first is False)
    check("edge_length pre-swap order [x(X-1), y(Y-1)]",
          res["edge_length"][0].shape[0] == X - 1
          and res["edge_length"][1].shape[0] == Y - 1)
    nA = res["net"][0]
    check("netA chain tree (root parent None, child idx)",
          nA.pins[0][0].parent_pin is None and nA.pins[0][1].parent_pin == 0)
    check("pin layer span from APs", nA.pins[0][0].physical_pin_layers == [1, 2])
    check("local net single pin", len(res["net"][2].pins[0]) == 1)

    print("== T3: pipeline round-trip + certificate ==")
    from generate_e2e_graphs import preprocess_results
    import copy
    pre = preprocess_results(copy.deepcopy(res))
    check("preprocess grid", (pre["args"].xmax, pre["args"].ymax) == (X, Y))
    check("effective caps shaped",
          tuple(pre["RoutingRegion"].cap_mat[0].shape) == (X, Y - 1)
          and tuple(pre["RoutingRegion"].cap_mat[1].shape) == (X - 1, Y))
    from generate_scratch_benchmarks import certify
    ok, info = certify(res, iters=120, seed=0)
    check(f"FastDGR certificate runs (of={info['overflow_units']})",
          info["n_subnets"] >= 2)
    check("mini instance routes overflow-free", ok)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
