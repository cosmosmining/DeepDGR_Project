#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_e2e_graphs.py — TEACHER-FREE synthetic benchmark generation for
end-to-end (e2e / iMF) training.

Why this exists
---------------
generate_compact_graphs.py runs the DGR *teacher* (main_stochastic.py,
2000 GPU iterations) on every variant to produce soft labels — that is what
made scaling 50 -> 1000 graphs cost ~63 SU / 92 GB and got rejected.
End-to-end training (deepdgr_e2e.py-style, and the iMF pipeline in
imf_train.py) needs NO teacher: it only needs the *differentiable DGR
objective itself*, i.e. the capacity / path / via tensors.  So this
generator:

    template .pt  --(bounded randomize, SAME transform as compact gen)-->
    candidate pool + process_pool  (in-process, no subprocesses) -->
    hetero graph tensors + OBJECTIVE ("field") tensors  -->  e2e pack .pt

  * NO main_stochastic.py run. NO teacher target. ~10-100x cheaper/variant.
  * Saves everything imf_field.DGRField needs to evaluate/differentiate the
    DGR cost (format "e2e_v1", documented in imf_field.py).
  * Per-variant DGR MEMORY PROBE (--probe): one objective forward+backward
    on the GPU, recording peak memory and catching CUDA OOM — this is the
    "observe DGR errors (memory)" step that decides which template sizes are
    feasible before committing to the 1000-graph run.
  * Identical pin randomization to the 235 existing compact graphs
    (imports _randomize_pins from generate_compact_graphs.py read-only), and
    the same seeds — so e2e packs PAIR 1:1 with existing teacher-labeled
    compact files for hybrid supervised+e2e experiments.

Phases (matching the research plan)
-----------------------------------
  pilot : 50 graphs across all 6 templates (38 MB ispd18_test5 ... 385 MB
          ispd19_test9), probe ON  ->  e2e_manifest.csv with peak-mem/OOM.
  scale : 1000 graphs (teacher-free, so now affordable); the plan is derived
          from the pilot manifest (templates whose probes OOM'd or exceed
          --mem_budget_gb are dropped). Supports --shard for SLURM arrays
          and runs probe-OFF on CPU nodes to save GPU SUs.
  pack  : one pack for a REAL benchmark (no randomization) — needed by
          imf_warmstart.py to emit warm-starts for real routing runs.
  inspect: print stats of an e2e pack.

USAGE
-----
  # smoke test (tiny subsampled variant, CPU-safe)
  python3 generate_e2e_graphs.py pilot --smoke --output_root ./e2e_smoke

  # the 50-graph pilot with memory probe (GPU node)
  python3 generate_e2e_graphs.py pilot \
      --template_dir cu-gr-2/run --output_root /ocean/projects/cis260079p/ctsai4/e2e

  # the 1000-graph scale-up (CPU nodes, probe off), 8-way sharded
  python3 generate_e2e_graphs.py scale --total 1000 --shard $i 8 \
      --template_dir cu-gr-2/run --output_root /ocean/projects/cis260079p/ctsai4/e2e

  # real-benchmark pack for warm-start export
  python3 generate_e2e_graphs.py pack --template cu-gr-2/run/ispd18_test5_metal5.pt \
      --output_root /ocean/projects/cis260079p/ctsai4/e2e_real

This file is NEW code; it does not modify any existing file.
"""

import argparse
import copy
import csv
import gc
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

# repo modules — used READ-ONLY
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import util as dgr_util                      # noqa: E402
import data as dgr_data                      # noqa: E402
from generate_compact_graphs import _randomize_pins   # noqa: E402  (identical
#                              bounded randomization as the compact pipeline)

ALL_BENCHES = [
    "ispd18_test5_metal5",     # ~38 MB template
    "ispd18_test10_metal5",
    "ispd18_test8_metal5",
    "ispd19_test7_metal5",
    "ispd19_test8_metal5",
    "ispd19_test9_metal5",     # ~385 MB template
]

# 50-graph pilot spread over the 38 MB -> 400 MB size ladder
DEFAULT_PILOT_PLAN = [
    ("ispd18_test5_metal5", 9),
    ("ispd18_test8_metal5", 9),
    ("ispd18_test10_metal5", 9),
    ("ispd19_test7_metal5", 8),
    ("ispd19_test8_metal5", 8),
    ("ispd19_test9_metal5", 7),
]

MANIFEST_FIELDS = [
    "file", "benchmark", "seed", "rand_mode", "net_frac", "n_nets",
    "n_grid", "n_cand", "n_subnets", "nnz_hor", "nnz_ver", "nnz_via",
    "edges_total", "file_mb", "gen_time_s",
    "probe_device", "probe_peak_mb", "probe_oom", "probe_err",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  Preprocessing — mirrors main_stochastic.py lines 110-148 exactly
# ════════════════════════════════════════════════════════════════════════

class _Args:
    pass


def preprocess_results(results, capacity=1.0, pin_ratio=1.0,
                       local_net_ratio=1.0, via_layer=1.5):
    """Capacity rebuild, edge-length swap, pin/local-net demand subtraction —
    the same statements as main_stochastic.py:110-148 (and the e2e/graph
    scripts), so packs are consistent with what --warmstart_file runs see."""
    RouteNets = results["net"]
    RoutingRegion = results["region"]
    RoutingRegion3D = results["region3D"]

    a = _Args()
    a.xmax, a.ymax = RoutingRegion.xmax, RoutingRegion.ymax
    a.net_num = len(RouteNets)
    a.num_layer = (len(RoutingRegion3D.cap_mat_3D[0])
                   + len(RoutingRegion3D.cap_mat_3D[1]))
    a.hor_first = RoutingRegion3D.hor_first
    a.num_hor_layer = len(RoutingRegion3D.cap_mat_3D[0])
    a.num_ver_layer = len(RoutingRegion3D.cap_mat_3D[1])
    a.via_layer = float(np.sqrt(a.num_layer)) * via_layer
    a.read_new_tree = False

    RoutingRegion.cap_mat = [
        torch.stack(RoutingRegion3D.cap_mat_3D[0]).sum(0) * capacity,
        torch.stack(RoutingRegion3D.cap_mat_3D[1]).sum(0) * capacity,
    ]
    results["edge_length"] = [results["edge_length"][1],
                              results["edge_length"][0]]   # the swap (:127)

    hor_ed, ver_ed, hor_pd, ver_pd = dgr_util.get_pin_demand(
        RouteNets, a, results["layers"], results["edge_length"])
    hor_ld, ver_ld = dgr_util.get_local_net(RouteNets, a)
    RoutingRegion.cap_mat[0] = (RoutingRegion.cap_mat[0]
                                - hor_ld * local_net_ratio
                                - torch.tensor(hor_ed) * pin_ratio)
    RoutingRegion.cap_mat[1] = (RoutingRegion.cap_mat[1]
                                - ver_ld * local_net_ratio
                                - torch.tensor(ver_ed) * pin_ratio)

    return {
        "args": a, "RouteNets": RouteNets, "RoutingRegion": RoutingRegion,
        "edge_length": results["edge_length"],
        "hor_pin_demand": hor_pd, "ver_pin_demand": ver_pd,
        "m2_pitch": results["m2_pitch"],
        "min_ulsc": min(l["unit_length_short_cost"]
                        for l in results["layers"]),
    }


# ════════════════════════════════════════════════════════════════════════
#  Vectorized hetero-graph tensors (same semantics as
#  deepdgr_graph_from_dgr.build_hetero_graph, minus its python loops)
# ════════════════════════════════════════════════════════════════════════

def build_grid_features(xmax, ymax, hor_cap, ver_cap):
    xs = (torch.arange(xmax, dtype=torch.float32).unsqueeze(1)
          .expand(xmax, ymax).reshape(-1) / max(xmax - 1, 1))
    ys = (torch.arange(ymax, dtype=torch.float32).unsqueeze(0)
          .expand(xmax, ymax).reshape(-1) / max(ymax - 1, 1))
    avg_h = torch.zeros(xmax, ymax)
    if ymax > 1:
        avg_h[:, 1:-1] = (hor_cap[:, :-1] + hor_cap[:, 1:]) / 2
        avg_h[:, 0], avg_h[:, -1] = hor_cap[:, 0], hor_cap[:, -1]
    avg_v = torch.zeros(xmax, ymax)
    if xmax > 1:
        avg_v[1:-1, :] = (ver_cap[:-1, :] + ver_cap[1:, :]) / 2
        avg_v[0, :], avg_v[-1, :] = ver_cap[0, :], ver_cap[-1, :]
    cmax = max(avg_h.abs().max().item(), avg_v.abs().max().item(), 1.0)
    return torch.stack([xs, ys, avg_h.reshape(-1) / cmax,
                        avg_v.reshape(-1) / cmax], dim=1)


def build_candidate_features(wire_length, via_count, sizes):
    """[norm_wl, norm_via, subnet_size/10, is_unique] — vectorized version of
    deepdgr_graph_from_dgr.py:205-215."""
    wl = wire_length.float().cpu()
    vc = via_count.float().cpu()
    size_per_cand = torch.repeat_interleave(sizes.float(), sizes)
    x = torch.zeros(wl.shape[0], 4)
    x[:, 0] = wl / max(wl.max().item(), 1.0)
    x[:, 1] = vc / max(vc.max().item(), 1.0)
    x[:, 2] = size_per_cand / 10.0
    x[:, 3] = (size_per_cand == 1).float()
    return x


def build_connects_edges(xmax, ymax):
    hi = torch.arange(xmax).unsqueeze(1).expand(xmax, ymax - 1).reshape(-1)
    hj = torch.arange(ymax - 1).unsqueeze(0).expand(xmax, ymax - 1).reshape(-1)
    h1, h2 = (hi * ymax + hj).long(), (hi * ymax + hj + 1).long()
    vi = torch.arange(xmax - 1).unsqueeze(1).expand(xmax - 1, ymax).reshape(-1)
    vj = torch.arange(ymax).unsqueeze(0).expand(xmax - 1, ymax).reshape(-1)
    v1, v2 = (vi * ymax + vj).long(), ((vi + 1) * ymax + vj).long()
    src = torch.cat([h1, h2, v1, v2])
    dst = torch.cat([h2, h1, v2, v1])
    return torch.stack([src, dst]).to(torch.int32)


def build_path_edges(hor_idx, ver_idx, xmax, ymax):
    """candidate->grid ('passes_through') from the path sparse indices, with
    the same endpoint expansion + dedup as deepdgr_graph_from_dgr.py:255-274."""
    num_grid = xmax * ymax
    ef_h, c_h = hor_idx[0].long(), hor_idx[1].long()
    i_h, j_h = ef_h // (ymax - 1), ef_h % (ymax - 1)
    g1_h, g2_h = i_h * ymax + j_h, i_h * ymax + j_h + 1
    ef_v, c_v = ver_idx[0].long(), ver_idx[1].long()
    i_v, j_v = ef_v // ymax, ef_v % ymax
    g1_v, g2_v = i_v * ymax + j_v, (i_v + 1) * ymax + j_v
    src = torch.cat([c_h, c_h, c_v, c_v])
    dst = torch.cat([g1_h, g2_h, g1_v, g2_v])
    key = (src * num_grid + dst).unique()
    return torch.stack([(key // num_grid), (key % num_grid)]).to(torch.int32)


def build_competes_edges(p_index_t):
    """Same-subnet cliques (deepdgr_graph_from_dgr.py:295-307), vectorized by
    grouping subnets of equal size."""
    sizes = p_index_t[1:] - p_index_t[:-1]
    chunks = []
    for s in torch.unique(sizes):
        s = int(s.item())
        if s <= 1:
            continue
        starts = p_index_t[:-1][sizes == s]                       # [G]
        a_loc = torch.arange(s).repeat_interleave(s)              # [s*s]
        b_loc = torch.arange(s).repeat(s)
        keep = a_loc != b_loc
        a_loc, b_loc = a_loc[keep], b_loc[keep]
        a = (starts.unsqueeze(1) + a_loc.unsqueeze(0)).reshape(-1)
        b = (starts.unsqueeze(1) + b_loc.unsqueeze(0)).reshape(-1)
        chunks.append(torch.stack([a, b]))
    if not chunks:
        return torch.zeros(2, 0, dtype=torch.int32)
    return torch.cat(chunks, dim=1).to(torch.int32)


# ════════════════════════════════════════════════════════════════════════
#  Pack builder
# ════════════════════════════════════════════════════════════════════════

def build_pack_from_results(results, benchmark, pattern_level=1,
                            pool_params=None, extra_meta=None,
                            share_static_dir=None):
    """results (a loaded/randomized benchmark dict) -> (pack dict, stats)."""
    pp = dict(max_z=10, z_step=3, c_step=3, max_c=20, max_c_out_ratio=5)
    if pool_params:
        pp.update(pool_params)
    t0 = time.time()
    pre = preprocess_results(results)
    a = pre["args"]
    xmax, ymax = a.xmax, a.ymax
    log(f"    preprocess: {time.time()-t0:.1f}s  grid {xmax}x{ymax}, "
        f"{a.net_num:,} nets, {a.num_layer} layers")

    t0 = time.time()
    candidate_pool = dgr_util.get_initial_candidate_pool(
        pre["RouteNets"], xmax, ymax, device="cpu",
        edge_length=pre["edge_length"], pattern_level=pattern_level, **pp)
    log(f"    candidate pool: {time.time()-t0:.1f}s")

    t0 = time.time()
    (p_index, p_index_full, _p2pat, hor_path, ver_path, wire_length_count,
     via_info, *_rest) = dgr_data.process_pool(
        candidate_pool, xmax, ymax, device="cpu")
    del candidate_pool
    gc.collect()
    via_map, via_count = via_info
    n_cand = int(p_index[-1])
    p_index_t = torch.tensor(p_index, dtype=torch.long)
    sizes = p_index_t[1:] - p_index_t[:-1]
    log(f"    process_pool: {time.time()-t0:.1f}s  "
        f"{n_cand:,} candidates / {len(p_index)-1:,} subnets")

    t0 = time.time()
    hor = hor_path.coalesce()
    ver = ver_path.coalesce()
    via = via_map.coalesce()
    del hor_path, ver_path, via_map
    gc.collect()

    edges = {
        ("candidate", "passes_through", "grid"):
            build_path_edges(hor.indices(), ver.indices(), xmax, ymax),
        ("candidate", "competes", "candidate"): build_competes_edges(p_index_t),
    }
    edges[("grid", "influences", "candidate")] = \
        edges[("candidate", "passes_through", "grid")].flip(0)
    connects = build_connects_edges(xmax, ymax)

    static_rel = None
    if share_static_dir is not None:
        static_rel = f"static_{benchmark}.pt"
        static_path = os.path.join(share_static_dir, static_rel)
        if not os.path.isfile(static_path):
            os.makedirs(share_static_dir, exist_ok=True)
            torch.save({"connects": connects,
                        "xmax": xmax, "ymax": ymax}, static_path)
    else:
        edges[("grid", "connects", "grid")] = connects

    hor_cap = pre["RoutingRegion"].cap_mat[0].float()
    ver_cap = pre["RoutingRegion"].cap_mat[1].float()
    grid_x = build_grid_features(xmax, ymax, hor_cap, ver_cap)
    cand_x = build_candidate_features(wire_length_count, via_count, sizes)
    log(f"    hetero tensors: {time.time()-t0:.1f}s")

    def el_tensor(x):
        return (x.detach().clone().float() if isinstance(x, torch.Tensor)
                else torch.tensor(np.asarray(x), dtype=torch.float32))

    pack = {
        "grid_x": grid_x.half(),
        "candidate_x": cand_x.half(),
        "edges": edges,
        "p_index": p_index_t.to(torch.int32),
        "field": {
            "hor_idx": hor.indices().to(torch.int32),
            "hor_val": hor.values().half(),
            "ver_idx": ver.indices().to(torch.int32),
            "ver_val": ver.values().half(),
            "via_idx": via.indices().to(torch.int32),
            "via_val": via.values().half(),
            "via_count": via_count.to(torch.int32),
            "wire_length": wire_length_count.float(),
            "hor_cap": hor_cap, "ver_cap": ver_cap,
            "hor_pin_demand": torch.tensor(pre["hor_pin_demand"],
                                           dtype=torch.float32),
            "ver_pin_demand": torch.tensor(pre["ver_pin_demand"],
                                           dtype=torch.float32),
            "hor_edge_length": el_tensor(pre["edge_length"][0]).flatten(),
            "ver_edge_length": el_tensor(pre["edge_length"][1]).flatten(),
            "m2_pitch": float(pre["m2_pitch"]),
            "min_unit_length_short_cost": float(pre["min_ulsc"]),
            "via_layer": float(a.via_layer),
            "xmax": xmax, "ymax": ymax,
            "num_layer": a.num_layer, "hor_first": bool(a.hor_first),
        },
        "metadata": {
            "format": "e2e_v1", "benchmark": benchmark,
            "n_grid": grid_x.shape[0], "n_candidates": n_cand,
            "n_subnets": len(p_index) - 1, "n_nets": a.net_num,
            "pattern_level": pattern_level, "static_pack": static_rel,
            **(extra_meta or {}),
        },
    }
    stats = {
        "n_nets": a.net_num, "n_grid": grid_x.shape[0], "n_cand": n_cand,
        "n_subnets": len(p_index) - 1,
        "nnz_hor": hor.values().shape[0], "nnz_ver": ver.values().shape[0],
        "nnz_via": via.values().shape[0],
        "edges_total": int(sum(e.shape[1] for e in edges.values())
                           + connects.shape[1]),
    }
    return pack, stats


# ════════════════════════════════════════════════════════════════════════
#  Variant synthesis (teacher-free) — subsample + bounded randomize
# ════════════════════════════════════════════════════════════════════════

def make_variant(template_results, seed, rand_mode="translate",
                 window=-1, window_frac=0.02, net_frac=1.0):
    """Deep-copy the template, optionally subsample routed nets (size knob /
    smoke tests), then apply the SAME bounded pin randomization as
    generate_compact_graphs.py."""
    variant = copy.deepcopy(template_results)
    if net_frac < 1.0:
        rng = random.Random(seed * 7919 + 13)
        routed = [n for n in variant["net"] if getattr(n, "need_route", True)]
        keep_n = max(2, int(len(routed) * net_frac))
        kept = set(id(n) for n in rng.sample(routed, keep_n))
        variant["net"] = [n for n in variant["net"]
                          if not getattr(n, "need_route", True)
                          or id(n) in kept]
        for i, n in enumerate(variant["net"]):
            n.net_ID = i
    _randomize_pins(variant, seed, mode=rand_mode, window=window,
                    window_frac=window_frac)
    return variant


# ════════════════════════════════════════════════════════════════════════
#  DGR memory probe — "observe DGR errors (memory)"
# ════════════════════════════════════════════════════════════════════════

def probe_dgr_memory(pack, device):
    """One DGR objective forward + backward on `device` — the per-iteration
    memory footprint of both the (now-removed) teacher and of iMF field
    evaluations.  Catches CUDA OOM instead of dying."""
    from imf_field import DGRField
    out = {"probe_device": device, "probe_peak_mb": "",
           "probe_oom": 0, "probe_err": ""}
    if device.startswith("cuda") and not torch.cuda.is_available():
        out["probe_err"] = "no-cuda"
        return out
    try:
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        field = DGRField(pack, device=device)
        z = field.init().requires_grad_(True)
        total, parts = field.objective(z)
        total.backward()
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            out["probe_peak_mb"] = round(
                torch.cuda.max_memory_allocated(device) / 2**20, 1)
        out["probe_total_cost"] = float(total.item())
        del field, z, total, parts
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e)
        out["probe_oom"] = int("out of memory" in msg.lower())
        out["probe_err"] = msg.splitlines()[0][:160]
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


# ════════════════════════════════════════════════════════════════════════
#  Manifest
# ════════════════════════════════════════════════════════════════════════

def append_manifest(manifest_path, row):
    new = not os.path.isfile(manifest_path)
    with open(manifest_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def read_manifest(manifest_path):
    if not os.path.isfile(manifest_path):
        return []
    with open(manifest_path) as fh:
        return list(csv.DictReader(fh))


# ════════════════════════════════════════════════════════════════════════
#  Job runner shared by pilot / scale
# ════════════════════════════════════════════════════════════════════════

def run_jobs(jobs, args):
    """jobs: list of (benchmark, seed). One template is loaded at a time."""
    os.makedirs(args.output_root, exist_ok=True)
    manifest_path = os.path.join(args.output_root, "e2e_manifest.csv")
    probe_dev = (f"cuda:{args.device}"
                 if args.probe and torch.cuda.is_available() else "")
    if args.probe and not probe_dev:
        log("  WARN: --probe requested but CUDA unavailable; probe skipped")

    by_bench = {}
    for bench, seed in jobs:
        by_bench.setdefault(bench, []).append(seed)

    done = failed = 0
    total_bytes = 0
    for bench, seeds in by_bench.items():
        tpl_path = os.path.join(args.template_dir, f"{bench}.pt")
        if not os.path.isfile(tpl_path):
            log(f"ERROR: template missing: {tpl_path}")
            failed += len(seeds)
            continue
        out_dir = os.path.join(args.output_root, bench)
        os.makedirs(out_dir, exist_ok=True)
        log(f"Loading template {tpl_path} "
            f"({os.path.getsize(tpl_path)/2**20:.1f} MB) for seeds {seeds}")
        template = torch.load(tpl_path, map_location="cpu", weights_only=False)

        for seed in seeds:
            out_path = os.path.join(out_dir, f"e2e_s{seed:04d}.pt")
            if os.path.isfile(out_path):
                log(f"  CACHED {out_path}")
                done += 1
                continue
            if total_bytes / 2**30 > args.max_total_gb:
                log(f"  STOP: --max_total_gb={args.max_total_gb} reached")
                return done, failed
            log(f"  [{bench} seed={seed}] synthesizing variant "
                f"(rand={args.rand_mode}, net_frac={args.net_frac})")
            t0 = time.time()
            try:
                variant = make_variant(
                    template, seed, rand_mode=args.rand_mode,
                    window=args.window, window_frac=args.window_frac,
                    net_frac=args.net_frac)
                pack, stats = build_pack_from_results(
                    variant, bench, pattern_level=args.pattern_level,
                    extra_meta={"seed": seed, "rand_mode": args.rand_mode,
                                "net_frac": args.net_frac, "teacher": "none"},
                    share_static_dir=out_dir if args.share_static else None)
                del variant
                gc.collect()

                probe = {"probe_device": "", "probe_peak_mb": "",
                         "probe_oom": "", "probe_err": ""}
                if probe_dev:
                    log(f"    probing DGR memory on {probe_dev} ...")
                    probe = probe_dgr_memory(pack, probe_dev)
                    log(f"    probe: peak={probe.get('probe_peak_mb','')} MB"
                        f"  oom={probe.get('probe_oom')}"
                        f"  err={probe.get('probe_err','')[:60]}")

                torch.save(pack, out_path)
                fmb = os.path.getsize(out_path) / 2**20
                total_bytes += os.path.getsize(out_path)
                gen_s = time.time() - t0
                append_manifest(manifest_path, {
                    "file": os.path.relpath(out_path, args.output_root),
                    "benchmark": bench, "seed": seed,
                    "rand_mode": args.rand_mode, "net_frac": args.net_frac,
                    **stats, "file_mb": round(fmb, 1),
                    "gen_time_s": round(gen_s, 1), **probe,
                })
                done += 1
                log(f"    saved {out_path}  ({fmb:.1f} MB, {gen_s:.0f}s)  "
                    f"[{done} done / {failed} failed]")
                del pack
            except Exception as e:                      # noqa: BLE001
                failed += 1
                log(f"    FAILED seed {seed}: {type(e).__name__}: {e}")
            finally:
                gc.collect()
        del template
        gc.collect()
    return done, failed


def parse_plan(plan_str):
    plan = []
    for part in plan_str.split(","):
        bench, cnt = part.split(":")
        plan.append((bench.strip(), int(cnt)))
    return plan


def next_free_seed(output_root, bench):
    existing = glob.glob(os.path.join(output_root, bench, "e2e_s*.pt"))
    seeds = []
    for f in existing:
        try:
            seeds.append(int(os.path.basename(f)[5:9]))
        except ValueError:
            pass
    return max(seeds) + 1 if seeds else 0


# ════════════════════════════════════════════════════════════════════════
#  Commands
# ════════════════════════════════════════════════════════════════════════

def cmd_pilot(args):
    plan = parse_plan(args.plan) if args.plan else DEFAULT_PILOT_PLAN
    if args.smoke:
        plan = [("ispd18_test5_metal5", args.smoke_n)]
        args.net_frac = min(args.net_frac, 0.01)
        log(f"SMOKE MODE: {plan}, net_frac={args.net_frac}")
    total = sum(c for _, c in plan)
    log(f"PILOT: {total} teacher-free graphs over {len(plan)} templates "
        f"(probe={'on' if args.probe else 'off'})")
    jobs = [(b, args.base_seed + i) for b, c in plan for i in range(c)]
    if args.shard:
        i, n = args.shard
        jobs = jobs[i::n]
        log(f"  shard {i}/{n}: {len(jobs)} jobs")
    done, failed = run_jobs(jobs, args)
    log(f"PILOT complete: {done} done, {failed} failed.  Manifest: "
        f"{os.path.join(args.output_root, 'e2e_manifest.csv')}")
    log("Next: inspect e2e_manifest.csv probe_peak_mb/probe_oom columns, "
        "then run the `scale` phase.")


def cmd_scale(args):
    manifest = read_manifest(os.path.join(args.output_root,
                                          "e2e_manifest.csv"))
    feasible = []
    for bench in ALL_BENCHES:
        rows = [r for r in manifest if r["benchmark"] == bench]
        ooms = [r for r in rows if str(r.get("probe_oom", "")) == "1"]
        peaks = [float(r["probe_peak_mb"]) for r in rows
                 if r.get("probe_peak_mb") not in ("", None)]
        if rows and ooms:
            log(f"  EXCLUDE {bench}: {len(ooms)} OOM in pilot probe")
            continue
        if peaks and max(peaks) > args.mem_budget_gb * 1024 * 0.9:
            log(f"  EXCLUDE {bench}: peak {max(peaks):.0f} MB > 90% of "
                f"{args.mem_budget_gb} GB budget")
            continue
        if not rows:
            log(f"  WARN {bench}: no pilot rows (run pilot first); "
                f"{'including' if args.include_unprobed else 'excluding'}")
            if not args.include_unprobed:
                continue
        feasible.append(bench)
    if args.benches:
        feasible = [b for b in args.benches.split(",") if b]
    if not feasible:
        log("ERROR: no feasible templates. Run `pilot` with --probe first.")
        sys.exit(1)

    per = args.total // len(feasible)
    rem = args.total - per * len(feasible)
    plan = [(b, per + (1 if i < rem else 0)) for i, b in enumerate(feasible)]
    log(f"SCALE: {args.total} graphs over feasible templates: {plan}")

    jobs = []
    for bench, cnt in plan:
        s0 = args.base_seed if args.base_seed >= 0 \
            else next_free_seed(args.output_root, bench)
        jobs += [(bench, s0 + i) for i in range(cnt)]
    if args.shard:
        i, n = args.shard
        jobs = jobs[i::n]
        log(f"  shard {i}/{n}: {len(jobs)} jobs")
    done, failed = run_jobs(jobs, args)
    log(f"SCALE complete: {done} done, {failed} failed")


def cmd_pack(args):
    bench = os.path.splitext(os.path.basename(args.template))[0]
    os.makedirs(args.output_root, exist_ok=True)
    out_path = os.path.join(args.output_root, f"{bench}_e2e.pt")
    if os.path.isfile(out_path) and not args.overwrite:
        log(f"exists: {out_path} (use --overwrite)")
        return
    log(f"PACK (real benchmark, no randomization): {args.template}")
    results = torch.load(args.template, map_location="cpu",
                         weights_only=False)
    pack, stats = build_pack_from_results(
        results, bench, pattern_level=args.pattern_level,
        extra_meta={"seed": -1, "rand_mode": "none", "net_frac": 1.0,
                    "real_benchmark": True, "teacher": "none"},
        share_static_dir=None)
    if args.probe and torch.cuda.is_available():
        probe = probe_dgr_memory(pack, f"cuda:{args.device}")
        log(f"  probe: {probe}")
    torch.save(pack, out_path)
    log(f"saved {out_path} ({os.path.getsize(out_path)/2**20:.1f} MB)  "
        f"stats={stats}")


def cmd_probe(args):
    """Probe EXISTING packs on a GPU and update the manifest in place.

    SU saver: generation is CPU-bound, so it runs probe-off on the cheapest
    GPU share (or anywhere); this short job then measures DGR memory for all
    packs on the SAME device type used for training (H100-80), which is the
    budget that actually matters."""
    from imf_field import load_e2e_pack
    if not torch.cuda.is_available():
        log("ERROR: probe needs a GPU")
        sys.exit(1)
    device = f"cuda:{args.device}"
    manifest_path = os.path.join(args.output_root, "e2e_manifest.csv")
    rows = read_manifest(manifest_path)
    by_file = {r["file"]: r for r in rows}
    files = sorted(glob.glob(os.path.join(args.output_root, "*", "e2e_s*.pt")))
    if args.benches:
        keep = set(args.benches.split(","))
        files = [f for f in files
                 if os.path.basename(os.path.dirname(f)) in keep]
    log(f"PROBE: {len(files)} packs on {device} "
        f"({torch.cuda.get_device_name(device)})")
    for i, f in enumerate(files):
        rel = os.path.relpath(f, args.output_root)
        row = by_file.get(rel)
        if row and row.get("probe_peak_mb") not in ("", None) \
                and not args.reprobe:
            continue
        t0 = time.time()
        try:
            pack = load_e2e_pack(f)
        except Exception as e:                          # noqa: BLE001
            log(f"  [{i+1}/{len(files)}] {rel}: LOAD FAILED {e}")
            continue
        probe = probe_dgr_memory(pack, device)
        meta = pack.get("metadata", {})
        if row is None:
            row = {k: "" for k in MANIFEST_FIELDS}
            row.update({"file": rel, "benchmark": meta.get("benchmark", ""),
                        "seed": meta.get("seed", ""),
                        "n_cand": meta.get("n_candidates", ""),
                        "n_subnets": meta.get("n_subnets", ""),
                        "file_mb": round(os.path.getsize(f) / 2**20, 1)})
            by_file[rel] = row
        for k in ("probe_device", "probe_peak_mb", "probe_oom", "probe_err"):
            row[k] = probe.get(k, "")
        log(f"  [{i+1}/{len(files)}] {rel}: "
            f"peak={probe.get('probe_peak_mb','')} MB oom={probe['probe_oom']}"
            f" ({time.time()-t0:.0f}s)")
        del pack
        gc.collect()
        torch.cuda.empty_cache()
    with open(manifest_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS,
                           extrasaction="ignore")
        w.writeheader()
        for rel in sorted(by_file):
            w.writerow(by_file[rel])
    log(f"manifest updated: {manifest_path}")


def cmd_inspect(args):
    from imf_field import load_e2e_pack
    pack = load_e2e_pack(args.path)
    meta = pack.get("metadata", {})
    f = pack["field"]
    print(f"\nE2E pack: {args.path}")
    print(f"  size: {os.path.getsize(args.path)/2**20:.1f} MB")
    print(f"  metadata: {json.dumps(meta, indent=4, default=str)}")
    print(f"  grid_x {tuple(pack['grid_x'].shape)}  "
          f"candidate_x {tuple(pack['candidate_x'].shape)}  "
          f"p_index {tuple(pack['p_index'].shape)}")
    for et, ei in pack["edges"].items():
        print(f"  edge [{et[0]}->{et[1]}->{et[2]}]: {ei.shape[1]:,}")
    print(f"  field: nnz hor={f['hor_idx'].shape[1]:,} "
          f"ver={f['ver_idx'].shape[1]:,} via={f['via_idx'].shape[1]:,}  "
          f"grid {f['xmax']}x{f['ymax']}  via_layer={f['via_layer']:.2f}  "
          f"m2_pitch={f['m2_pitch']}")
    print("  teacher: NONE (e2e/iMF pack — objective tensors instead)\n")


# ════════════════════════════════════════════════════════════════════════

def add_common(p):
    p.add_argument("--template_dir", default="cu-gr-2/run")
    p.add_argument("--output_root", required=True)
    p.add_argument("--rand_mode", default="translate",
                   choices=["translate", "jitter", "translate_jitter"])
    p.add_argument("--window", type=int, default=-1)
    p.add_argument("--window_frac", type=float, default=0.02)
    p.add_argument("--net_frac", type=float, default=1.0,
                   help="subsample fraction of routed nets (size knob)")
    p.add_argument("--pattern_level", type=int, default=1)
    p.add_argument("--probe", action=argparse.BooleanOptionalAction,
                   default=True, help="DGR memory probe per graph (needs GPU)")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--share_static", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="share per-template 'connects' edges in one file")
    p.add_argument("--max_total_gb", type=float, default=200.0)
    p.add_argument("--shard", type=int, nargs=2, metavar=("I", "N"),
                   default=None, help="process job i of N (SLURM arrays)")
    p.add_argument("--base_seed", type=int, default=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pilot", help="50-graph teacher-free pilot + mem probe")
    add_common(p)
    p.add_argument("--plan", default=None,
                   help='override, e.g. "ispd18_test5_metal5:9,..."')
    p.add_argument("--smoke", action="store_true",
                   help="tiny subsampled single-variant validation run")
    p.add_argument("--smoke_n", type=int, default=1)

    p = sub.add_parser("scale", help="1000-graph scale-up (teacher-free)")
    add_common(p)
    p.set_defaults(probe=False, base_seed=-1)
    p.add_argument("--total", type=int, default=1000)
    p.add_argument("--mem_budget_gb", type=float, default=80.0,
                   help="GPU memory budget used to filter templates")
    p.add_argument("--include_unprobed", action="store_true")
    p.add_argument("--benches", default=None,
                   help="comma list overriding feasibility filtering")

    p = sub.add_parser("pack", help="pack one REAL benchmark (no randomize)")
    p.add_argument("--template", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--pattern_level", type=int, default=1)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("probe", help="probe existing packs on a GPU, "
                                     "update e2e_manifest.csv")
    p.add_argument("--output_root", required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--benches", default=None, help="comma filter")
    p.add_argument("--reprobe", action="store_true",
                   help="re-probe rows that already have results")

    p = sub.add_parser("inspect", help="print stats of an e2e pack")
    p.add_argument("path")

    args = ap.parse_args()
    {"pilot": cmd_pilot, "scale": cmd_scale, "pack": cmd_pack,
     "probe": cmd_probe, "inspect": cmd_inspect}[args.cmd](args)


if __name__ == "__main__":
    main()
