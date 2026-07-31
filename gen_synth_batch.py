#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_synth_batch.py — CPU multiprocessing ENGINE for routable, BOUNDED-BOX
(locality-preserving) synthetic global-routing instances + TEACHER-FREE
training packs, scalable to 100k+ instances.

WHY
---
generate_scratch_benchmarks.py builds a single from-scratch, routable-by-
construction instance and (with --certify) proves overflow == 0 with the
bundled CPU FastDGR solver.  generate_e2e_graphs.build_pack_from_results
turns a results-dict into a TEACHER-FREE "e2e_v1" pack whose `field`
sub-dict is everything the differentiable DGR objective (imf_field.DGRField)
needs — and NOTHING ELSE (no teacher / no target distribution).

This driver wraps both, READ-ONLY, into a multiprocessing batch generator:
it sweeps a (grid size x n_nets x utilization) grid, certifies each instance
routable on a worker process, and writes one compact `.npz` per instance to
  /ocean/projects/cis260079p/ctsai4/synthdata/<run>/inst_*.npz
plus a manifest.csv (one row / instance).  It uses created-one/deleted-one
temp discipline (per-instance unique TMPDIR + unique candidate-pool ./tmp
cache names) so the tight /ocean quota is never loaded with intermediates.

NOTE: instances are sampled with BOUNDED windows around per-net centers
(generate_scratch_benchmarks.sample_netlist, window_frac knob) — locality-
preserving, NOT random pin locations.  This is the same locality discipline
the compact / e2e template generators use.

TEACHER-FREE PACK — keys saved to each inst_*.npz
-------------------------------------------------
The npz holds ONLY the physics-loss inputs (the `field` dict of e2e_v1,
flattened to numpy) + the CSR subnet structure + small metadata.  Mapping to
imf_field.DGRField (the consumer of the DGR physics loss):

  PHYSICS-LOSS tensors (consumed by DGRField.__init__/objective):
    hor_cap   [xmax, ymax-1] f32   per-edge HORIZONTAL capacity   (pin/local
    ver_cap   [xmax-1, ymax] f32   per-edge VERTICAL   capacity    demand already
                                                                  subtracted)
    hor_idx   [2, nnz_h] i32 \  candidate->HOR-edge incidence (sparse COO:
    hor_val   [nnz_h]    f32 /  row=edge index, col=candidate, val=usage)
    ver_idx   [2, nnz_v] i32 \  candidate->VER-edge incidence
    ver_val   [nnz_v]    f32 /
    via_idx   [2, nnz_z] i32 \  candidate->grid-cell VIA incidence
    via_val   [nnz_z]    f32 /
    via_count    [Nc] i32          per-candidate via cost
    wire_length  [Nc] f32          per-candidate wirelength cost
    hor_pin_demand [xmax, ymax] f32 \  pin-via congestion fields (used in the
    ver_pin_demand [xmax, ymax] f32 /  add_via term of the overflow demand)
    hor_edge_length [ymax-1] f32   \  physical edge lengths (post edge-swap)
    ver_edge_length [xmax-1] f32   /
    m2_pitch, min_unit_length_short_cost, via_layer  (scalars; via_layer is
                                                     already sqrt(L)-scaled)
    xmax, ymax, num_layer, hor_first                 (geometry scalars)

  STRUCTURE (defines per-2-pin-subnet softmax groups for the loss):
    p_index   [S+1] i32            CSR offsets; p_index[i]:p_index[i+1] are the
                                   candidate logits of the i-th 2-pin subnet.

  METADATA (bookkeeping only, NOT used by the loss):
    grid, n_nets, n_cand, n_subnets, utilization, seed, certified, format,
    n_hor_edges, n_ver_edges, n_grid.

  >>> NO teacher / target / soft-label distribution is stored.  Confirmed: the
      DGR objective is recomputed at train time from the tensors above; there
      is no `target`, `teacher`, `probabilities`, `logits`, or `best_*` key.
      (Contrast generate_compact_graphs.py packs, which carry a `target`.)

A pack saved here is consumed by rebuilding the `field` dict (helper
`load_synth_field_pack` below restores it) and constructing
imf_field.DGRField(field_pack), exactly as the e2e/iMF training loop does.

USAGE
-----
  # CPU validation: 2 tiny certified instances
  python3 gen_synth_batch.py --run smoke --n 2 --workers 2 \
      --grid 40x36 --n_nets 300 --util 0.55 --certify

  # mass-generate 1000 certified instances across a difficulty grid (CPU)
  python3 gen_synth_batch.py --run v1 --n 1000 --workers 32 \
      --grid 64x56,96x84,128x112 --n_nets 1500,3000,6000 \
      --util 0.50,0.60,0.70 --certify

NEW file.  generate_scratch_benchmarks.py / generate_e2e_graphs.py / data.py /
util.py / imf_field.py and the rest are imported READ-ONLY.
"""

import argparse
import copy
import csv
import itertools
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DEFAULT_OUT_ROOT = os.environ.get("DEEPDGR_DATA", "./synthdata")

# Manifest schema (one row / instance).
MANIFEST_FIELDS = [
    "path", "grid", "xmax", "ymax", "n_nets", "n_cand", "n_subnets",
    "util", "seed", "certified", "max_overflow", "wl_cost",
    "nnz_hor", "nnz_ver", "nnz_via", "size_mb", "gen_s",
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  Size / difficulty grid
# ════════════════════════════════════════════════════════════════════════

def parse_grid_arg(s):
    """'64x56,96x84' -> [(64,56),(96,84)]."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        xs, ys = part.lower().split("x")
        out.append((int(xs), int(ys)))
    return out


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def parse_float_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def build_cells(grids, n_nets_list, utils):
    """Cartesian product of the difficulty grid -> ordered list of
    (xmax, ymax, n_nets, util) cells."""
    cells = []
    for (xm, ym), nn, ut in itertools.product(grids, n_nets_list, utils):
        cells.append((xm, ym, nn, ut))
    return cells


def assign_jobs(cells, n, base_seed):
    """Round-robin N instances across the difficulty cells; each gets a unique
    global index (-> seed) so reruns are reproducible and resumable."""
    jobs = []
    for i in range(n):
        xm, ym, nn, ut = cells[i % len(cells)]
        jobs.append(dict(idx=i, seed=base_seed + i,
                         xmax=xm, ymax=ym, n_nets=nn, util=ut))
    return jobs


# ════════════════════════════════════════════════════════════════════════
#  Teacher-free pack extraction (field-only npz) — created/deleted-one
# ════════════════════════════════════════════════════════════════════════

def _to_np(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def pack_to_field_npz(pack):
    """Take an e2e_v1 pack (generate_e2e_graphs.build_pack_from_results) and
    flatten ONLY its teacher-free physics-loss inputs (`field`) + `p_index`
    + bookkeeping metadata into a flat dict of numpy arrays/scalars suitable
    for np.savez_compressed.  No teacher / target is ever read."""
    f = pack["field"]
    meta = pack.get("metadata", {})
    out = {
        # --- candidate->edge incidence (sparse COO) ---
        "hor_idx": _to_np(f["hor_idx"]).astype(np.int32),
        "hor_val": _to_np(f["hor_val"]).astype(np.float32),
        "ver_idx": _to_np(f["ver_idx"]).astype(np.int32),
        "ver_val": _to_np(f["ver_val"]).astype(np.float32),
        "via_idx": _to_np(f["via_idx"]).astype(np.int32),
        "via_val": _to_np(f["via_val"]).astype(np.float32),
        # --- per-candidate costs ---
        "via_count": _to_np(f["via_count"]).astype(np.int32),
        "wire_length": _to_np(f["wire_length"]).astype(np.float32),
        # --- per-edge capacities (pin/local demand already subtracted) ---
        "hor_cap": _to_np(f["hor_cap"]).astype(np.float32),
        "ver_cap": _to_np(f["ver_cap"]).astype(np.float32),
        # --- via congestion fields ---
        "hor_pin_demand": _to_np(f["hor_pin_demand"]).astype(np.float32),
        "ver_pin_demand": _to_np(f["ver_pin_demand"]).astype(np.float32),
        # --- physical edge lengths (post-swap) ---
        "hor_edge_length": _to_np(f["hor_edge_length"]).astype(np.float32),
        "ver_edge_length": _to_np(f["ver_edge_length"]).astype(np.float32),
        # --- CSR subnet structure ---
        "p_index": _to_np(pack["p_index"]).astype(np.int32),
        # --- scalars consumed by the loss ---
        "m2_pitch": np.float32(f["m2_pitch"]),
        "min_unit_length_short_cost": np.float32(
            f["min_unit_length_short_cost"]),
        "via_layer": np.float32(f["via_layer"]),
        "xmax": np.int32(f["xmax"]),
        "ymax": np.int32(f["ymax"]),
        "num_layer": np.int32(f["num_layer"]),
        "hor_first": np.int32(1 if bool(f["hor_first"]) else 0),
        # --- bookkeeping metadata (NOT used by the loss) ---
        "n_candidates": np.int32(_to_np(f["via_count"]).shape[0]),
        "n_subnets": np.int32(meta.get("n_subnets",
                                       _to_np(pack["p_index"]).shape[0] - 1)),
        "n_grid": np.int32(meta.get("n_grid", int(f["xmax"]) * int(f["ymax"]))),
        "n_nets": np.int32(meta.get("n_nets", 0)),
        "utilization": np.float32(meta.get("utilization", 0.0)),
        "seed": np.int32(meta.get("seed", -1)),
        "format": "synth_field_v1",
        "teacher": "none",
    }
    return out


def load_synth_field_pack(npz_path):
    """Restore the `field` dict + `p_index` from a synth_field_v1 npz so it can
    be fed to imf_field.DGRField (which expects pack['field'] + pack['p_index']
    + pack['candidate_x']).  Provided for the training loop; not used here."""
    import torch
    d = np.load(npz_path, allow_pickle=False)
    field = {
        "hor_idx": torch.from_numpy(d["hor_idx"]),
        "hor_val": torch.from_numpy(d["hor_val"]),
        "ver_idx": torch.from_numpy(d["ver_idx"]),
        "ver_val": torch.from_numpy(d["ver_val"]),
        "via_idx": torch.from_numpy(d["via_idx"]),
        "via_val": torch.from_numpy(d["via_val"]),
        "via_count": torch.from_numpy(d["via_count"]),
        "wire_length": torch.from_numpy(d["wire_length"]),
        "hor_cap": torch.from_numpy(d["hor_cap"]),
        "ver_cap": torch.from_numpy(d["ver_cap"]),
        "hor_pin_demand": torch.from_numpy(d["hor_pin_demand"]),
        "ver_pin_demand": torch.from_numpy(d["ver_pin_demand"]),
        "hor_edge_length": torch.from_numpy(d["hor_edge_length"]),
        "ver_edge_length": torch.from_numpy(d["ver_edge_length"]),
        "m2_pitch": float(d["m2_pitch"]),
        "min_unit_length_short_cost": float(d["min_unit_length_short_cost"]),
        "via_layer": float(d["via_layer"]),
        "xmax": int(d["xmax"]), "ymax": int(d["ymax"]),
        "num_layer": int(d["num_layer"]),
        "hor_first": bool(int(d["hor_first"])),
    }
    n_cand = int(d["n_candidates"])
    return {
        "field": field,
        "p_index": torch.from_numpy(d["p_index"]),
        # DGRField only reads candidate_x.shape[0]; a placeholder suffices
        "candidate_x": torch.zeros(n_cand, 1),
        "metadata": {"format": "synth_field_v1", "n_candidates": n_cand},
    }


# ════════════════════════════════════════════════════════════════════════
#  One instance — runs on a worker process, fully isolated
# ════════════════════════════════════════════════════════════════════════

def _worker_init(omp_threads):
    # tiny-op CPU workloads thrash with default thread pools on a 256-core node
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    import torch
    torch.set_num_threads(omp_threads)


def gen_one(job, cfg):
    """Generate + certify + write ONE teacher-free instance.

    Isolation discipline (HARD RULE 3): each call gets its own unique TMPDIR
    and a unique candidate-pool ./tmp cache prefix, so nothing collides and
    no shared cu-gr-2/run cwd is ever touched.  The .pt results-dict that the
    scratch generator / pack builder need is created in TMPDIR and deleted
    immediately after the npz is written (created-one/deleted-one)."""
    import torch
    from generate_scratch_benchmarks import build_results, certify
    from generate_e2e_graphs import build_pack_from_results

    idx, seed = job["idx"], job["seed"]
    xmax, ymax, n_nets, util = (job["xmax"], job["ymax"],
                                job["n_nets"], job["util"])
    t0 = time.time()

    # Per-instance unique temp dir (HARD RULE 3).  TMPDIR also steers the
    # util candidate-pool ./tmp cache via a unique name below.
    tmp = tempfile.mkdtemp(prefix=f"agent_synth_{os.getpid()}_{idx}_",
                           dir=cfg["scratch_tmp"])
    old_tmpdir = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = tmp
    row = {k: "" for k in MANIFEST_FIELDS}
    row.update(dict(grid=f"{xmax}x{ymax}", xmax=xmax, ymax=ymax,
                    n_nets=n_nets, util=util, seed=seed, certified=0))
    try:
        rng = np.random.RandomState(seed * 7919 + 11)
        results, _stats = build_results(
            rng, xmax, ymax, n_nets, utilization=util,
            num_layer=cfg["num_layer"], margin=cfg["margin"],
            max_pins=cfg["max_pins"], window_frac=cfg["window_frac"])

        certified, info, max_of, wl_cost = 1, {}, "", ""
        if cfg["certify"]:
            ok, info = certify(results, iters=cfg["certify_iters"], seed=seed)
            certified = int(ok)
            max_of = info.get("max_overflow", "")
            wl_cost = info.get("wl_cost", "")
            if not ok and not cfg["keep_uncertified"]:
                row.update(dict(certified=0, max_overflow=max_of,
                                wl_cost=wl_cost, gen_s=round(time.time()-t0, 1)))
                return ("reject", row, idx)

        name = f"inst_{idx:06d}_x{xmax}y{ymax}n{n_nets}_s{seed:06d}"
        pack, _pstats = build_pack_from_results(
            copy.deepcopy(results), benchmark=name, pattern_level=1,
            extra_meta={"generator": "synth_batch_v1", "seed": seed,
                        "utilization": util, "teacher": "none"})
        del results
        flat = pack_to_field_npz(pack)
        del pack

        out_path = os.path.join(cfg["out_dir"], name + ".npz")
        # write to TMPDIR then move into place (atomic-ish, quota-safe)
        tmp_npz = os.path.join(tmp, name + ".npz")
        np.savez_compressed(tmp_npz, **flat)
        shutil.move(tmp_npz, out_path)

        size_mb = round(os.path.getsize(out_path) / 2**20, 3)
        row.update(dict(
            path=os.path.relpath(out_path, cfg["out_root"]),
            n_cand=int(flat["n_candidates"]),
            n_subnets=int(flat["n_subnets"]),
            certified=certified, max_overflow=max_of, wl_cost=wl_cost,
            nnz_hor=int(flat["hor_idx"].shape[1]),
            nnz_ver=int(flat["ver_idx"].shape[1]),
            nnz_via=int(flat["via_idx"].shape[1]),
            size_mb=size_mb, gen_s=round(time.time() - t0, 1)))
        return ("ok", row, idx)
    except Exception as e:                                   # noqa: BLE001
        row["gen_s"] = round(time.time() - t0, 1)
        row["certified"] = -1
        return ("error", {"row": row, "idx": idx,
                          "err": f"{type(e).__name__}: {e}",
                          "tb": traceback.format_exc()}, idx)
    finally:
        # created-one/deleted-one: nuke the whole per-instance temp tree
        shutil.rmtree(tmp, ignore_errors=True)
        if old_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_tmpdir


# ════════════════════════════════════════════════════════════════════════
#  Driver
# ════════════════════════════════════════════════════════════════════════

def append_manifest(path, row):
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS,
                           extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def existing_done_indices(out_dir):
    """Indices already written (resume support)."""
    done = set()
    if not os.path.isdir(out_dir):
        return done
    for fn in os.listdir(out_dir):
        if fn.startswith("inst_") and fn.endswith(".npz"):
            try:
                done.add(int(fn.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return done


def run(args):
    out_root = args.out_root
    out_dir = os.path.join(out_root, args.run)
    scratch_tmp = (args.scratch_tmp
                   or os.environ.get("TMPDIR", tempfile.gettempdir()))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(scratch_tmp, exist_ok=True)
    manifest = os.path.join(out_dir, "manifest.csv")

    grids = parse_grid_arg(args.grid)
    n_nets_list = parse_int_list(args.n_nets)
    utils = parse_float_list(args.util)
    cells = build_cells(grids, n_nets_list, utils)
    jobs = assign_jobs(cells, args.n, args.base_seed)

    if args.resume:
        done = existing_done_indices(out_dir)
        before = len(jobs)
        jobs = [j for j in jobs if j["idx"] not in done]
        log(f"resume: {len(done)} already present, {before-len(jobs)} skipped")

    cfg = dict(
        out_root=out_root, out_dir=out_dir, scratch_tmp=scratch_tmp,
        num_layer=args.num_layer, margin=args.margin, max_pins=args.max_pins,
        window_frac=args.window_frac, certify=args.certify,
        certify_iters=args.certify_iters,
        keep_uncertified=args.keep_uncertified)

    log(f"RUN '{args.run}': N={args.n} over {len(cells)} difficulty cells "
        f"(grids={grids}, n_nets={n_nets_list}, util={utils})")
    log(f"  out_dir={out_dir}")
    log(f"  workers={args.workers}  omp_threads={args.omp_threads}  "
        f"certify={'on' if args.certify else 'OFF'}")
    log(f"  jobs to run: {len(jobs)}")

    n_ok = n_reject = n_err = 0
    wall0 = time.time()
    gen_times = []

    if args.workers <= 1:
        _worker_init(args.omp_threads)
        for job in jobs:
            status, payload, idx = gen_one(job, cfg)
            n_ok, n_reject, n_err, gen_times = _handle(
                status, payload, idx, manifest, n_ok, n_reject, n_err,
                gen_times, len(jobs))
    else:
        with ProcessPoolExecutor(
                max_workers=args.workers, initializer=_worker_init,
                initargs=(args.omp_threads,)) as ex:
            futs = {ex.submit(gen_one, job, cfg): job["idx"] for job in jobs}
            for fut in as_completed(futs):
                status, payload, idx = fut.result()
                n_ok, n_reject, n_err, gen_times = _handle(
                    status, payload, idx, manifest, n_ok, n_reject, n_err,
                    gen_times, len(jobs))

    wall = time.time() - wall0
    per = (sum(gen_times) / len(gen_times)) if gen_times else 0.0
    log(f"DONE run '{args.run}': ok={n_ok} reject={n_reject} err={n_err} "
        f"in {wall:.1f}s wall")
    if gen_times:
        log(f"  mean per-instance compute: {per:.1f}s  "
            f"(throughput ~{n_ok/max(wall,1e-9):.2f} inst/s with "
            f"{args.workers} workers)")
        log(f"  EST for 100k @ this throughput: "
            f"{100000/max(n_ok/max(wall,1e-9),1e-9)/3600:.1f} core-wall-hours "
            f"(scale by worker count)")
    log(f"  manifest: {manifest}")
    return n_ok, n_reject, n_err


def _handle(status, payload, idx, manifest, n_ok, n_reject, n_err,
            gen_times, total):
    if status == "ok":
        n_ok += 1
        append_manifest(manifest, payload)
        gen_times.append(float(payload["gen_s"]))
        log(f"  [{n_ok+n_reject+n_err}/{total}] OK idx={idx} "
            f"{payload['grid']} n_nets={payload['n_nets']} "
            f"n_cand={payload['n_cand']} util={payload['util']} "
            f"({payload['size_mb']} MB, {payload['gen_s']}s)")
    elif status == "reject":
        n_reject += 1
        append_manifest(manifest, payload)
        log(f"  [{n_ok+n_reject+n_err}/{total}] REJECT idx={idx} "
            f"(overflow!=0; not saved)")
    else:
        n_err += 1
        append_manifest(manifest, payload["row"])
        log(f"  [{n_ok+n_reject+n_err}/{total}] ERROR idx={idx}: "
            f"{payload['err']}")
        log(payload["tb"])
    return n_ok, n_reject, n_err, gen_times


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1] if __doc__ else "")
    ap.add_argument("--run", required=True,
                    help="subdir name under --out_root for this batch")
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--n", type=int, required=True,
                    help="number of instances to generate")
    ap.add_argument("--workers", type=int, default=8,
                    help="multiprocessing workers (CPU)")
    ap.add_argument("--omp_threads", type=int, default=2,
                    help="OMP/torch threads PER worker")
    ap.add_argument("--base_seed", type=int, default=0)
    # difficulty grid
    ap.add_argument("--grid", default="64x56,96x84,128x112",
                    help="comma list of WxH grid sizes")
    ap.add_argument("--n_nets", default="1500,3000,6000",
                    help="comma list of net counts")
    ap.add_argument("--util", default="0.50,0.60,0.70",
                    help="comma list of utilizations (difficulty knob)")
    # netlist sampling (bounded-box locality)
    ap.add_argument("--window_frac", type=float, default=0.06,
                    help="bounded window size as fraction of min(grid)")
    ap.add_argument("--max_pins", type=int, default=6)
    ap.add_argument("--num_layer", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.10)
    # certificate
    ap.add_argument("--certify", action="store_true",
                    help="require FastDGR overflow==0 (CPU) per instance")
    ap.add_argument("--certify_iters", type=int, default=300)
    ap.add_argument("--keep_uncertified", action="store_true")
    # temp / resume
    ap.add_argument("--scratch_tmp", default=None,
                    help="dir for per-instance temp trees (default $TMPDIR)")
    ap.add_argument("--resume", action="store_true",
                    help="skip indices whose npz already exists")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
