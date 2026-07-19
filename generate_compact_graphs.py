#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_compact_graphs.py
==========================
Generates compact GNN-ready synthetic data DIRECTLY from the original
benchmark .pt — no intermediate bloated files ever touch disk.

** BOUNDED PIN RANDOMIZATION (the important fix) **
Pins are perturbed in a LOCALITY-PRESERVING way, not scattered uniformly
across the die.  Uniform `randint(0, xmax-1)` placement makes every net
chip-spanning, which (1) destroys the congestion structure so the GNN does
not transfer to real benchmarks, and (2) explodes the candidate graph from a
few million edges to ~240M edges (~12 GB graph, multi-GB "compact" files).
A bounded transform keeps each net's wirelength close to the original and
fixes both problems at once.  See `_randomize_pins` for the modes.

Pipeline per variant:
  original .pt  ──►  BOUNDED randomize pins in-memory  ──►  temp .pt (one at a time)
       ──►  build graph  ──►  run DGR teacher  ──►  extract compact tensors
       ──►  save compact .pt  ──►  DELETE temp files immediately

Disk footprint during generation: ~1–1.5 GB temp (temp .pt + small temp graph),
deleted after every variant.  Final output: ~50–300 MB per variant (the edge
index lists dominate; this is the GNN's actual input, far smaller than the
12 GB raw graph but NOT 8 MB — 8 MB is impossible while keeping edges).

What gets saved per compact file:
  - grid_x:        [N_grid, 4]   float16
  - candidate_x:   [N_cand, 4]   float16
  - edge_indices:   dict of 4 edge types → [2, E] int32
  - target:         [N_cand]      float32   (teacher soft-labels)
  - p_index:        [N_subnets+1] int32     (subnet→candidate mapping)
  - metadata:       benchmark, grid dims, candidate count, seed, rand_mode

COMMANDS:
  generate  — Create N synthetic variants directly from an original benchmark .pt
              (one temp file at a time, cleaned up immediately)
  from_raw  — Convert already-generated raw synthetic .pt files that are sitting
              on disk into compact format (builds graph + teacher, then deletes raw)
  convert   — Pack existing _graph.pt + _teacher.npz pairs into compact format
  inspect   — Print stats for a compact .pt file (and flag edge-count bloat)

USAGE:
  cd /path/to/Differentiable-Global-Router

  # ── PRIMARY: generate BOUNDED synthetic compact data from scratch ──
  python3 generate_compact_graphs.py generate \
      --template ispd18_test5_metal5.pt \
      --output_dir /path/compact/ispd18_test5_metal5 \
      --n_variants 50 --seed 0 --device 0 --rand_mode translate

  # ── INSPECT a compact file (prints OK / BLOATED verdict) ──────────
  python3 generate_compact_graphs.py inspect synth_0000_compact.pt
"""

import argparse
import copy
import gc
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

print("[generate_compact_graphs] Loading torch (this can take 30-60s on NFS)...",
      flush=True)
import numpy as np
import torch
print(f"[generate_compact_graphs] torch {torch.__version__} loaded, "
      f"CUDA={'yes' if torch.cuda.is_available() else 'NO'}",
      flush=True)


# ════════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════════
ALL_BENCHES = [
    "ispd18_test5_metal5",
    "ispd18_test8_metal5",
    "ispd18_test10_metal5",
    "ispd19_test7_metal5",
    "ispd19_test8_metal5",
    "ispd19_test9_metal5",
]

EDGE_TYPES = [
    ("grid", "connects", "grid"),
    ("candidate", "passes_through", "grid"),
    ("grid", "influences", "candidate"),
    ("candidate", "competes", "candidate"),
]

# Above this many edges in any single edge type, the candidate graph is
# almost certainly built from chip-spanning (unbounded) nets.
BLOAT_EDGE_THRESHOLD = 50_000_000


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _detect_dgr_dir(hint=None):
    """
    Auto-detect DGR directory.  Fully portable — no hardcoded paths.
    Priority: explicit hint > CWD > DGR_DIR env > ~/Differentiable-Global-Router
    > script dir.  Only checks LOCAL paths; never probes remote NFS mounts
    that could hang.
    """
    marker = "main_stochastic.py"   # file that must exist in DGR root

    if hint:
        resolved = os.path.abspath(hint)
        if os.path.isfile(os.path.join(resolved, marker)):
            return resolved
        if os.path.isdir(resolved):
            log(f"  WARN: {resolved} has no {marker}, using anyway")
            return resolved
        log(f"  WARN: --dgr_dir {hint} does not exist, ignoring")

    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, marker)):
        return cwd

    env = os.environ.get("DGR_DIR")
    if env:
        env = os.path.abspath(env)
        if os.path.isfile(os.path.join(env, marker)):
            return env

    home_dgr = os.path.expanduser("~/Differentiable-Global-Router")
    if os.path.isfile(os.path.join(home_dgr, marker)):
        return home_dgr

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(script_dir, marker)):
        return script_dir

    log(f"  WARN: could not find {marker} anywhere, using CWD: {cwd}")
    return cwd


def _find_python(dgr_dir):
    """Use the currently-running interpreter (respects conda/venv activation)."""
    return sys.executable


def _run_cmd(cmd, tag, cwd=None):
    """Run a shell command, return (ok, duration). Output goes to stdout/stderr."""
    log(f"  RUN: {tag}")
    log(f"  CMD: {cmd}")
    t0 = time.time()
    rc = subprocess.call(cmd, shell=True, cwd=cwd)
    dur = time.time() - t0
    ok = rc == 0
    log(f"  {'OK' if ok else 'FAIL'}: {tag} ({dur:.0f}s)")
    return ok, dur


# ════════════════════════════════════════════════════════════════════════
#  Core: extract compact tensors from a graph .pt + teacher .npz
# ════════════════════════════════════════════════════════════════════════
def _extract_compact(graph_pt_path, teacher_npz_path, output_path,
                     benchmark_name="unknown", extra_metadata=None):
    """
    Given a built graph .pt and teacher .npz, extract ONLY the tensors the
    GNN needs and save in compact format.  Every tensor is `.clone()`d so
    torch.save does not drag in the parent graph's underlying storage.

    Returns dict with stats or None on failure.
    """
    try:
        gd = torch.load(graph_pt_path, map_location="cpu", weights_only=False)
        graph = gd["graph"]
        total_cand = gd["total_candidates"]

        # .clone() detaches each tensor from the 12 GB graph storage so the
        # saved file only contains the bytes we actually keep.
        grid_x = graph["grid"].x.half().clone()       # [N_grid, 4] float16
        cand_x = graph["candidate"].x.half().clone()  # [N_cand, 4] float16

        edges = {}
        for et in EDGE_TYPES:
            if et in graph.edge_types:
                edges[et] = graph[et].edge_index.to(torch.int32).clone()

        teacher = np.load(teacher_npz_path)
        if "probabilities" in teacher:
            target = torch.from_numpy(teacher["probabilities"]).float().clone()
        elif "logits" in teacher:
            target = torch.from_numpy(teacher["logits"]).float().clone()
        else:
            log(f"  WARN: no probabilities or logits in {teacher_npz_path}")
            return None

        p_index = None
        if "p_index" in teacher:
            p_index = torch.from_numpy(teacher["p_index"]).to(torch.int32).clone()

        # Handle shape mismatch
        if cand_x.shape[0] != target.shape[0]:
            n = min(cand_x.shape[0], target.shape[0])
            cand_x = cand_x[:n].clone()
            target = target[:n].clone()

        metadata = {
            "benchmark": benchmark_name,
            "n_grid": grid_x.shape[0],
            "n_candidates": cand_x.shape[0],
            "total_candidates": total_cand,
            "grid_feat_dim": grid_x.shape[1],
            "cand_feat_dim": cand_x.shape[1],
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        compact = {
            "grid_x": grid_x,
            "candidate_x": cand_x,
            "edges": edges,
            "target": target,
            "metadata": metadata,
        }
        if p_index is not None:
            compact["p_index"] = p_index

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(compact, output_path)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        max_edges = max((e.shape[1] for e in edges.values()), default=0)
        if max_edges > BLOAT_EDGE_THRESHOLD:
            log(f"  ⚠ WARNING: {max_edges:,} edges in one edge type — pins look "
                f"UNBOUNDED. Compact = {size_mb:.0f} MB. Check --rand_mode.")
        return {
            "output": output_path,
            "size_mb": size_mb,
            "n_grid": grid_x.shape[0],
            "n_cand": cand_x.shape[0],
            "n_edges": sum(e.shape[1] for e in edges.values()),
            "max_edges": max_edges,
        }

    except Exception as e:
        log(f"  ERROR extracting compact: {e}")
        return None
    finally:
        gc.collect()


# ════════════════════════════════════════════════════════════════════════
#  Core: build graph + teacher from a raw .pt, return compact
# ════════════════════════════════════════════════════════════════════════
def _build_and_extract(raw_pt_path, output_compact_path, dgr_dir, device="0",
                       benchmark_name="unknown", extra_metadata=None,
                       pattern_level=1, dgr_iter=2000):
    """
    Given a raw benchmark .pt:
      1. Build heterogeneous graph (subprocess → temp file)
      2. Run DGR teacher (subprocess → temp file)
      3. Extract compact tensors
      4. Delete temp graph + teacher files

    The raw .pt is NOT deleted here (caller decides).
    Returns stats dict or None.
    """
    python = _find_python(dgr_dir)
    base = os.path.splitext(os.path.basename(raw_pt_path))[0]

    tmp_dir = tempfile.mkdtemp(prefix="deepdgr_compact_")
    tmp_graph = os.path.join(tmp_dir, f"{base}_graph.pt")
    tmp_teacher = os.path.join(tmp_dir, f"{base}_teacher.npz")

    try:
        # Step 1: Build graph
        graph_script = os.path.join(dgr_dir, "deepdgr_graph_from_dgr.py")
        if not os.path.isfile(graph_script):
            log(f"  ERROR: {graph_script} not found")
            return None

        ok, _ = _run_cmd(
            f"{python} {graph_script}"
            f" --data_path {raw_pt_path}"
            f" --dgr_dir {dgr_dir}"
            f" --output {tmp_graph}"
            f" --pattern_level {pattern_level}",
            f"Build graph [{base}]",
            cwd=dgr_dir,
        )
        if not ok or not os.path.isfile(tmp_graph):
            log(f"  ERROR: graph build failed for {raw_pt_path}")
            return None

        # Step 2: Run DGR teacher
        teacher_script = os.path.join(dgr_dir, "main_stochastic.py")
        if not os.path.isfile(teacher_script):
            log(f"  ERROR: {teacher_script} not found")
            return None

        ok, _ = _run_cmd(
            f"{python} {teacher_script}"
            f" --data_path {raw_pt_path}"
            f" --warmstart_file __nonexistent__"
            f" --save_target {tmp_teacher}"
            f" --output_name {base}_teacher"
            f" --iter {dgr_iter} --pattern_level {pattern_level}"
            f" --device {device}",
            f"DGR teacher [{base}]",
            cwd=dgr_dir,
        )
        if not ok or not os.path.isfile(tmp_teacher):
            log(f"  ERROR: DGR teacher failed for {raw_pt_path}")
            return None

        # Step 3: Extract compact
        result = _extract_compact(
            tmp_graph, tmp_teacher, output_compact_path,
            benchmark_name=benchmark_name,
            extra_metadata=extra_metadata,
        )
        return result

    finally:
        # Step 4: Always clean up temp files
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()


# ════════════════════════════════════════════════════════════════════════
#  Pin randomization — BOUNDED (locality-preserving)
# ════════════════════════════════════════════════════════════════════════
def _iter_net_pins(net):
    """Yield every pin object in a net, regardless of nesting depth."""
    for pin_list in net.pins:
        if isinstance(pin_list, (list, tuple)):
            for pin in pin_list:
                if hasattr(pin, "x") and hasattr(pin, "y"):
                    yield pin
        elif hasattr(pin_list, "x") and hasattr(pin_list, "y"):
            yield pin_list


def _randomize_pins(results, seed, mode="translate", window=-1,
                    window_frac=0.02):
    """
    Create a synthetic variant by perturbing net pin positions WHILE
    PRESERVING LOCALITY (a bounded transform), then relabel the nets.
    Modifies `results` IN-PLACE.

    Why bounded (and NOT uniform-random):
      Uniform `randint(0, xmax-1)` placement scatters every pin across the
      whole die.  Pins that sit a few cells apart in a real design become
      chip-spanning.  Consequences:
        1. Routing stops being realistic — congestion no longer resembles
           any real benchmark, so a GNN trained on it does not transfer.
        2. Candidate-route length explodes, so the candidate graph balloons
           from a few million edges to ~240M edges (~12 GB graph, multi-GB
           "compact" files).  This is the disk blow-up you saw.
      A bounded transform keeps each net's wirelength distribution close to
      the original, fixing BOTH problems at once.

    Modes:
      "translate"        (default) — rigid-translate each net by a random
                         (dx, dy); the SAME offset is applied to every pin in
                         the net, so the net's shape and half-perimeter
                         wirelength are preserved EXACTLY.  Different variants
                         relocate nets to different places → different
                         congestion maps → genuinely different routing
                         problems, with ZERO risk of edge blow-up.
      "jitter"           — perturb each pin within ±window of its ORIGINAL
                         cell (no whole-net translation).
      "translate_jitter" — translate the net, then add small per-pin jitter
                         for extra diversity (still bounded by `window`).

    window: jitter half-width in grid cells.  If < 0, computed as
            max(1, int(window_frac * min(xmax, ymax))).  Ignored by "translate".
    """
    xmax = results["region"].xmax
    ymax = results["region"].ymax
    nets = results["net"]

    rng = random.Random(seed)

    if window < 0:
        window = max(1, int(window_frac * min(xmax, ymax)))

    def clamp(v, lo, hi):
        return lo if v < lo else (hi if v > hi else v)

    use_translate = mode in ("translate", "translate_jitter")
    use_jitter = mode in ("jitter", "translate_jitter")

    for i, net in enumerate(nets):
        pins = list(_iter_net_pins(net))

        if pins:
            xs = [p.x for p in pins]
            ys = [p.y for p in pins]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # Rigid translation that keeps the WHOLE net inside the grid.
            if use_translate:
                dx_lo, dx_hi = -min_x, (xmax - 1) - max_x
                dy_lo, dy_hi = -min_y, (ymax - 1) - max_y
                dx = rng.randint(dx_lo, dx_hi) if dx_hi >= dx_lo else 0
                dy = rng.randint(dy_lo, dy_hi) if dy_hi >= dy_lo else 0
            else:
                dx = dy = 0

            for p in pins:
                nx = p.x + dx
                ny = p.y + dy
                if use_jitter:
                    nx += rng.randint(-window, window)
                    ny += rng.randint(-window, window)
                p.x = clamp(nx, 0, xmax - 1)
                p.y = clamp(ny, 0, ymax - 1)

        net.net_name = f"synth_net_{i}"
        net.net_index = i

    return results


# ════════════════════════════════════════════════════════════════════════
#  COMMAND: generate
# ════════════════════════════════════════════════════════════════════════
def cmd_generate(args):
    """
    Generate N synthetic compact variants directly from the original
    benchmark .pt.  Only ONE temp file exists on disk at any time.
    """
    log(f"generate_compact_graphs.py — starting generate command")
    log(f"  args: template={args.template} dgr_dir={args.dgr_dir} "
        f"n_variants={args.n_variants} seed={args.seed} device={args.device} "
        f"rand_mode={args.rand_mode} window={args.window} "
        f"window_frac={args.window_frac}")

    log(f"Detecting DGR directory...")
    dgr_dir = _detect_dgr_dir(args.dgr_dir)
    log(f"  DGR dir: {dgr_dir}")

    template_path = args.template
    if not os.path.isabs(template_path):
        template_path = os.path.join(dgr_dir, template_path)

    if not os.path.isfile(template_path):
        log(f"ERROR: template not found: {template_path}")
        sys.exit(1)

    benchmark = os.path.splitext(os.path.basename(template_path))[0]
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    log(f"{'='*60}")
    log(f"  Generating {args.n_variants} compact synthetic variants")
    log(f"  Template: {template_path}")
    log(f"  Output:   {output_dir}")
    log(f"  Seeds:    {args.seed} → {args.seed + args.n_variants - 1}")
    log(f"  Device:   {args.device}")
    log(f"  Rand:     mode={args.rand_mode} window={args.window} "
        f"frac={args.window_frac}  (BOUNDED — locality preserved)")
    log(f"  DGR dir:  {dgr_dir}")
    log(f"{'='*60}")

    # Load the template ONCE — we deep-copy per variant
    log(f"Loading template: {template_path}")
    template_results = torch.load(template_path, map_location='cpu',
                                   weights_only=False)
    xmax = template_results['region'].xmax
    ymax = template_results['region'].ymax
    n_nets = len(template_results['net'])
    auto_window = max(1, int(args.window_frac * min(xmax, ymax)))
    log(f"Template grid: {xmax}×{ymax}, nets: {n_nets:,}  "
        f"(auto jitter window = {auto_window} cells)")

    total_size = 0.0
    done = 0
    failed = 0
    t_start = time.time()

    for i in range(args.n_variants):
        seed = args.seed + i
        out_name = f"synth_{i:04d}_compact.pt"
        out_path = os.path.join(output_dir, out_name)

        if os.path.isfile(out_path):
            sz = os.path.getsize(out_path) / (1024 * 1024)
            total_size += sz
            done += 1
            if done % 50 == 0:
                log(f"  CACHED {done}/{args.n_variants} ({total_size:.1f} MB)")
            continue

        log(f"\n  Variant {i}/{args.n_variants} (seed={seed})")

        # ── 1. Deep-copy template and BOUNDED-randomize pins ────────────
        log(f"    Randomizing pins (mode={args.rand_mode})...")
        variant = copy.deepcopy(template_results)
        _randomize_pins(variant, seed, mode=args.rand_mode,
                        window=args.window, window_frac=args.window_frac)

        # ── 2. Write temp .pt (needed by graph builder + DGR subprocesses)
        tmp_dir = tempfile.mkdtemp(prefix=f"synth_{i:04d}_")
        tmp_pt = os.path.join(tmp_dir, f"synth_{i:04d}.pt")

        log(f"    Writing temp .pt → {tmp_pt}")
        torch.save(variant, tmp_pt)
        tmp_size_mb = os.path.getsize(tmp_pt) / (1024 * 1024)
        log(f"    Temp size: {tmp_size_mb:.0f} MB")

        del variant
        gc.collect()

        # ── 3-6. Build graph + teacher → extract compact → cleanup ──────
        try:
            result = _build_and_extract(
                raw_pt_path=tmp_pt,
                output_compact_path=out_path,
                dgr_dir=dgr_dir,
                device=args.device,
                benchmark_name=benchmark,
                extra_metadata={"seed": seed, "variant": i,
                                "rand_mode": args.rand_mode},
                pattern_level=args.pattern_level,
                dgr_iter=args.dgr_iter,
            )

            if result:
                total_size += result["size_mb"]
                done += 1
                log(f"    Compact: {result['size_mb']:.1f} MB "
                    f"({result['n_grid']:,} grid, {result['n_cand']:,} cand, "
                    f"max_edges={result['max_edges']:,})")
            else:
                failed += 1
                log(f"    FAILED variant {i}")

        except Exception as e:
            failed += 1
            log(f"    EXCEPTION variant {i}: {e}")

        finally:
            # ── 7. Delete temp .pt (the big one) ───────────────────────
            shutil.rmtree(tmp_dir, ignore_errors=True)
            guide_dir = os.path.join(dgr_dir, "CUGR2_guide")
            if os.path.isdir(guide_dir):
                for gf in glob.glob(os.path.join(guide_dir, f"*synth_{i:04d}*")):
                    try:
                        os.remove(gf)
                    except OSError:
                        pass
            gc.collect()

        elapsed = time.time() - t_start
        rate = done / max(elapsed, 1) * 3600
        log(f"    Progress: {done}/{args.n_variants} done, "
            f"{failed} failed, {total_size:.1f} MB total, ~{rate:.1f}/hr")

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    manifest = {
        "benchmark": benchmark,
        "template": os.path.basename(template_path),
        "n_variants": args.n_variants,
        "n_completed": done,
        "n_failed": failed,
        "total_size_mb": total_size,
        "base_seed": args.seed,
        "dgr_iter": args.dgr_iter,
        "pattern_level": args.pattern_level,
        "rand_mode": args.rand_mode,
        "window": args.window,
        "window_frac": args.window_frac,
        "generation_time_s": elapsed,
        "format": "compact_v1",
        "dtype_features": "float16",
        "dtype_edges": "int32",
        "dtype_target": "float32",
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    log(f"\n{'='*60}")
    log(f"  Generation complete!")
    log(f"  Completed: {done}/{args.n_variants}  Failed: {failed}")
    log(f"  Total output: {total_size:.1f} MB")
    log(f"  Time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    log(f"  Manifest: {manifest_path}")
    log(f"{'='*60}")


# ════════════════════════════════════════════════════════════════════════
#  COMMAND: from_raw
# ════════════════════════════════════════════════════════════════════════
def cmd_from_raw(args):
    """
    For users who already have raw synthetic .pt files on disk.  Processes
    each: build graph + teacher → extract compact → optionally delete raw.
    NOTE: this does NOT randomize — it uses the .pt files as-is.
    """
    dgr_dir = _detect_dgr_dir(args.dgr_dir)
    raw_dir = args.raw_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    all_pts = sorted(glob.glob(os.path.join(raw_dir, "*.pt")))
    raw_pts = [
        f for f in all_pts
        if not f.endswith("_graph.pt")
        and not f.endswith("_compact.pt")
        and "_teacher" not in os.path.basename(f)
    ]

    if not raw_pts:
        log(f"No raw .pt files found in {raw_dir}")
        return

    log(f"{'='*60}")
    log(f"  Converting {len(raw_pts)} raw .pt files → compact")
    log(f"  Source:     {raw_dir}")
    log(f"  Output:     {output_dir}")
    log(f"  Delete raw: {args.delete_raw}")
    log(f"{'='*60}")

    total_size = 0.0
    done = 0
    freed_mb = 0.0
    t_start = time.time()

    for idx, raw_path in enumerate(raw_pts):
        base = os.path.splitext(os.path.basename(raw_path))[0]
        out_path = os.path.join(output_dir, f"{base}_compact.pt")

        if os.path.isfile(out_path):
            sz = os.path.getsize(out_path) / (1024 * 1024)
            total_size += sz
            done += 1
            continue

        log(f"\n  [{idx+1}/{len(raw_pts)}] {base}")
        raw_size = os.path.getsize(raw_path) / (1024 * 1024)

        existing_graph = raw_path.replace(".pt", "_graph.pt")
        existing_teacher = raw_path.replace(".pt", "_teacher.npz")

        if os.path.isfile(existing_graph) and os.path.isfile(existing_teacher):
            log(f"    Found existing graph + teacher, extracting...")
            result = _extract_compact(
                existing_graph, existing_teacher, out_path,
                benchmark_name=base,
            )
        else:
            result = _build_and_extract(
                raw_pt_path=raw_path,
                output_compact_path=out_path,
                dgr_dir=dgr_dir,
                device=args.device,
                benchmark_name=base,
                pattern_level=args.pattern_level,
                dgr_iter=args.dgr_iter,
            )

        if result:
            total_size += result["size_mb"]
            done += 1
            log(f"    {raw_size:.0f} MB raw → {result['size_mb']:.1f} MB compact")

            if args.delete_raw:
                os.remove(raw_path)
                freed_mb += raw_size
                log(f"    Deleted raw .pt (+{raw_size:.0f} MB freed)")
                for f in [existing_graph, existing_teacher]:
                    if os.path.isfile(f):
                        freed_mb += os.path.getsize(f) / (1024 * 1024)
                        os.remove(f)
        else:
            log(f"    FAILED: {base}")

        gc.collect()

    elapsed = time.time() - t_start
    log(f"\n{'='*60}")
    log(f"  Conversion complete: {done}/{len(raw_pts)} files")
    log(f"  Output size: {total_size:.1f} MB")
    if args.delete_raw:
        log(f"  Disk freed: {freed_mb:.0f} MB ({freed_mb/1024:.1f} GB)")
    log(f"  Time: {elapsed:.0f}s")
    log(f"{'='*60}")


# ════════════════════════════════════════════════════════════════════════
#  COMMAND: convert
# ════════════════════════════════════════════════════════════════════════
def cmd_convert(args):
    """Convert pre-existing graph + teacher pairs to compact format."""
    source_dir = args.source_dir
    output_dir = args.output_dir
    benchmark = args.benchmark
    os.makedirs(output_dir, exist_ok=True)

    log(f"Scanning {source_dir} for graph/teacher pairs...")

    graph_files = sorted(glob.glob(os.path.join(source_dir, "*_graph.pt")))
    if not graph_files:
        graph_files = sorted(glob.glob(
            os.path.join(source_dir, f"*{benchmark}*_graph.pt")))

    if not graph_files:
        log(f"  No graph .pt files found in {source_dir}")
        return

    tasks = []
    for gf in graph_files:
        base = gf.replace("_graph.pt", "")
        teacher = base + "_teacher.npz"
        if not os.path.isfile(teacher):
            continue
        out_name = os.path.basename(base) + "_compact.pt"
        out_path = os.path.join(output_dir, out_name)
        if os.path.isfile(out_path):
            continue
        tasks.append((gf, teacher, out_path, benchmark))

    log(f"  Found {len(graph_files)} graph files, {len(tasks)} to convert")

    if not tasks:
        log(f"  Nothing to convert")
        return

    total_size = 0.0
    done = 0

    if args.workers <= 1:
        for gf, tf, op, bn in tasks:
            result = _extract_compact(gf, tf, op, bn)
            if result:
                total_size += result["size_mb"]
                done += 1
                if done % 50 == 0:
                    log(f"  {done}/{len(tasks)} done, {total_size:.1f} MB")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_extract_compact, gf, tf, op, bn): gf
                for gf, tf, op, bn in tasks
            }
            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    total_size += result["size_mb"]
                    done += 1
                    if done % 50 == 0:
                        log(f"  {done}/{len(tasks)} done, {total_size:.1f} MB")

    log(f"  Complete: {done}/{len(tasks)} files, {total_size:.1f} MB")

    manifest = {
        "benchmark": benchmark,
        "n_files": done,
        "total_size_mb": total_size,
        "format": "compact_v1",
        "dtype_features": "float16",
        "dtype_edges": "int32",
        "dtype_target": "float32",
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


# ════════════════════════════════════════════════════════════════════════
#  COMMAND: inspect
# ════════════════════════════════════════════════════════════════════════
def cmd_inspect(args):
    """Print stats for a compact .pt file and flag edge-count bloat."""
    path = args.path
    data = torch.load(path, map_location="cpu", weights_only=False)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\nCompact graph: {path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  grid_x:      {data['grid_x'].shape}  dtype={data['grid_x'].dtype}")
    print(f"  candidate_x: {data['candidate_x'].shape}  dtype={data['candidate_x'].dtype}")
    print(f"  target:       {data['target'].shape}  dtype={data['target'].dtype}")

    max_edges = 0
    for et, ei in data["edges"].items():
        etype_str = f"{et[0]}→{et[1]}→{et[2]}"
        print(f"  edge [{etype_str}]: {ei.shape[1]:,} edges  dtype={ei.dtype}")
        max_edges = max(max_edges, ei.shape[1])

    if "p_index" in data:
        pi = data["p_index"]
        print(f"  p_index:     {pi.shape}  ({len(pi) - 1:,} subnets)")
    print(f"  metadata: {json.dumps(data.get('metadata', {}), indent=4)}")
    print(f"  max edge count: {max_edges:,}")

    if max_edges > BLOAT_EDGE_THRESHOLD:
        print("  VERDICT: ⚠ BLOATED — pins look UNBOUNDED (chip-spanning nets).")
        print("           Do NOT mass-generate. Use --rand_mode translate.")
    else:
        print("  VERDICT: ✓ OK — bounded edge count, safe to mass-generate.")
    print()


# ════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate compact GNN-ready synthetic data directly "
                    "from original benchmark .pt files (BOUNDED pins)")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── generate ────────────────────────────────────────────────────────
    p_gen = sub.add_parser("generate",
        help="Create N synthetic compact variants from an original benchmark .pt "
             "using BOUNDED pin randomization (no edge blow-up).")
    p_gen.add_argument("--template", required=True,
                       help="Original benchmark .pt (e.g. ispd18_test5_metal5.pt)")
    p_gen.add_argument("--dgr_dir", default=None,
                       help="DGR source root (auto-detected if omitted)")
    p_gen.add_argument("--output_dir", required=True,
                       help="Where to write compact .pt files")
    p_gen.add_argument("--n_variants", type=int, default=50,
                       help="Number of synthetic variants to generate")
    p_gen.add_argument("--seed", type=int, default=0,
                       help="Base random seed (variant i uses seed+i)")
    p_gen.add_argument("--device", type=str, default="0",
                       help="GPU device id for DGR teacher")
    p_gen.add_argument("--pattern_level", type=int, default=1)
    p_gen.add_argument("--dgr_iter", type=int, default=2000,
                       help="DGR iterations for teacher generation")
    p_gen.add_argument("--rand_mode",
                       choices=["translate", "jitter", "translate_jitter"],
                       default="translate",
                       help="BOUNDED pin randomization. 'translate' (default) "
                            "relocates each net rigidly, preserving wirelength "
                            "exactly — safest, no edge blow-up.")
    p_gen.add_argument("--window", type=int, default=-1,
                       help="Jitter half-width in grid cells (-1 = auto from "
                            "--window_frac). Used by jitter/translate_jitter.")
    p_gen.add_argument("--window_frac", type=float, default=0.02,
                       help="If --window<0, jitter window = "
                            "window_frac * min(xmax, ymax).")

    # ── from_raw ────────────────────────────────────────────────────────
    p_raw = sub.add_parser("from_raw",
        help="Convert already-generated raw synthetic .pt files to compact.")
    p_raw.add_argument("--raw_dir", required=True)
    p_raw.add_argument("--dgr_dir", default=None)
    p_raw.add_argument("--output_dir", required=True)
    p_raw.add_argument("--device", type=str, default="0")
    p_raw.add_argument("--pattern_level", type=int, default=1)
    p_raw.add_argument("--dgr_iter", type=int, default=2000)
    p_raw.add_argument("--delete_raw", action="store_true")

    # ── convert ─────────────────────────────────────────────────────────
    p_conv = sub.add_parser("convert",
        help="Pack existing _graph.pt + _teacher.npz pairs into compact format")
    p_conv.add_argument("--source_dir", required=True)
    p_conv.add_argument("--output_dir", required=True)
    p_conv.add_argument("--benchmark", required=True)
    p_conv.add_argument("--workers", type=int, default=4)

    # ── inspect ─────────────────────────────────────────────────────────
    p_insp = sub.add_parser("inspect",
        help="Print stats for a compact .pt file (flags edge bloat)")
    p_insp.add_argument("path", help="Path to compact .pt file")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "from_raw":
        cmd_from_raw(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "inspect":
        cmd_inspect(args)


if __name__ == "__main__":
    main()
