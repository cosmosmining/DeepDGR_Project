#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_utils.py
==============
Shared utilities for Step 1 preliminary global routing experiments.
Import this module from each run_step1_*.py script.

Provides:
  - CUGR2 runner (with and without DGR guide)
  - CUGR2 log parser (wirelength, via, overflow)
  - Heatmap generation wrapper
  - Summary TSV writer
  - Common constants and helpers
"""
import os
import re
import sys
import time
import subprocess

# ════════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════════
def _detect_dgr_dir():
    """Auto-detect DGR directory: env var > CWD > hardcoded fallback."""
    # 1. Environment variable
    env = os.environ.get("DGR_DIR")
    if env and os.path.isdir(env):
        return env
    # 2. Current working directory (if it has main_stochastic.py)
    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, "main_stochastic.py")):
        return cwd
    # 3. Common paths
    for path in [
        "/home/chiungct/Differentiable-Global-Router",       # moura-rtx
        os.path.expanduser("~/Differentiable-Global-Router"), # any host
    ]:
        if os.path.isdir(path):
            return path
    # 4. Give up — use CWD and hope for the best
    return cwd

DGR_DIR = _detect_dgr_dir()
CUGR_RUN = os.path.join(DGR_DIR, "cu-gr-2", "run")
BENCH_DIR = os.path.join(DGR_DIR, "cu-gr-2", "benchmark", "test")
CUGR2_BIN = os.path.join(CUGR_RUN, "route")
DIAG_ROOT = os.path.join(DGR_DIR, "experiments", "diagnostics")
LOG_ROOT = os.path.join(DGR_DIR, "experiments", "gr_logs")
MODEL_DIR = os.path.join(DGR_DIR, "experiments", "models")
VIA_COST = 20

ALL_BENCHES = [
    "ispd18_test5_metal5",
    "ispd18_test8_metal5",
    "ispd18_test10_metal5",
    "ispd19_test7_metal5",
    "ispd19_test8_metal5",
    "ispd19_test9_metal5",
]

GRID_DIMS = {
    "ispd18_test5_metal5":  (619, 613),
    "ispd18_test8_metal5":  (905, 883),
    "ispd18_test10_metal5": (606, 522),
    "ispd19_test7_metal5":  (1053, 1011),
    "ispd19_test8_metal5":  (1202, 1138),
    "ispd19_test9_metal5":  (1337, 1433),
}

# Short label for each benchmark
def short_name(bench):
    """ispd18_test5_metal5 → 18_test5"""
    return bench.replace("ispd", "").replace("_metal5", "")


# ════════════════════════════════════════════════════════════════════════
#  Logging
# ════════════════════════════════════════════════════════════════════════
def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  Command runner
# ════════════════════════════════════════════════════════════════════════
# Cache the current Python interpreter path so subprocess calls use
# the same conda/venv Python that's running this script.
_PYTHON = '/ocean/projects/cis260079p/ctsai4/miniconda3/envs/deepdgr/bin/python3'

def run_cmd(cmd, tag, cwd=None):
    """Run a shell command, log result, return (ok, duration_seconds).
    Replaces bare 'python3' with the current interpreter ('/ocean/projects/cis260079p/ctsai4/miniconda3/envs/deepdgr/bin/python3')
    so conda/venv environments propagate to subprocesses."""
    # Replace leading 'python3 ' or 'python ' with the real interpreter
    if cmd.startswith("python3 "):
        cmd = _PYTHON + cmd[len("python3"):]
    elif cmd.startswith("python "):
        cmd = _PYTHON + cmd[len("python"):]
    log(f"  RUN: {tag}")
    log(f"  CMD: {cmd}")
    t0 = time.time()
    rc = subprocess.call(cmd, shell=True, cwd=cwd)
    dur = time.time() - t0
    ok = rc == 0
    log(f"  {'OK' if ok else 'FAIL'}: {tag} ({dur:.0f}s)")
    return ok, dur


# ════════════════════════════════════════════════════════════════════════
#  CUGR2 log parser
# ════════════════════════════════════════════════════════════════════════
def parse_cugr2_log(log_path):
    """
    Parse a CUGR2 log file and extract routing metrics.

    Returns dict with keys:
      wirelength: int or None
      via_count:  int or None
      overflow:   int or None
      max_overflow: int or None

    Handles multiple common CUGR2 output formats:
      - "total wirelength = 12345"  /  "total wirelength: 12345"
      - "#wire = 12345  #via = 678"
      - "total wire overflow: 99"  /  "total wire overflow = 99"
      - "total via count: 678"  /  "total via count = 678"
      - "max edge overflow: 5"
    """
    result = {
        "wirelength": None,
        "via_count": None,
        "overflow": None,
        "max_overflow": None,
    }

    if not os.path.isfile(log_path):
        return result

    with open(log_path, "r", errors="replace") as f:
        text = f.read()

    # Wirelength — CUGR2 format: "wire length (metric):  26569635"
    for pat in [
        r"wire\s+length\s*(?:\([^)]*\))?\s*[=:]?\s*(\d+)",
        r"total\s+wirelength\s*[=:]?\s*(\d+)",
        r"wirelength\s*[=:]?\s+(\d+)",
        r"#wire\s*[=:]?\s*(\d+)",
        r"wl\s*[=:]?\s+(\d+)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            result["wirelength"] = int(m.group(1))
            break

    # Via count — handles all separators
    for pat in [
        r"total\s+via\s+count\s*[=:]?\s*(\d+)",
        r"#via\s*[=:]?\s*(\d+)",
        r"via\s*count\s*[=:]?\s*(\d+)",
        r"(?<![a-z])via\s*[=:]?\s+(\d+)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            result["via_count"] = int(m.group(1))
            break

    # Total overflow (take LAST occurrence — CUGR2 may print intermediate values)
    for pat in [
        r"total\s+wire\s+overflow\s*[=:]?\s*(\d+)",
        r"total\s+overflow\s*[=:]?\s*(\d+)",
    ]:
        matches = re.findall(pat, text, re.IGNORECASE)
        if matches:
            result["overflow"] = int(matches[-1])
            break

    # Max edge overflow
    for pat in [
        r"max\s+(?:edge\s+)?overflow\s*[=:]?\s*(\d+)",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            result["max_overflow"] = int(m.group(1))

    return result


# ════════════════════════════════════════════════════════════════════════
#  CUGR2 runner — with DGR guide
# ════════════════════════════════════════════════════════════════════════
def run_cugr2_with_guide(bench, label, guide_path, log_path=None):
    """
    Run CUGR2 on a benchmark using a DGR guide file.

    Args:
        bench:      e.g. "ispd18_test5_metal5"
        label:      e.g. "BASELINE_18_test5"
        guide_path: absolute path to DGR guide .txt
        log_path:   where to write CUGR2 stdout (default: LOG_ROOT/<label>_cugr2.log)

    Returns:
        (metrics_dict, duration_seconds) or (None, duration) on failure
    """
    if log_path is None:
        os.makedirs(LOG_ROOT, exist_ok=True)
        log_path = os.path.join(LOG_ROOT, f"{label}_cugr2.log")

    lef = os.path.join(BENCH_DIR, bench, f"{bench}.input.lef")
    def_file = os.path.join(BENCH_DIR, bench, f"{bench}.input.def")
    output_guide = os.path.join(BENCH_DIR, bench, f"{bench}.{label}.guide")

    if not os.path.isfile(lef):
        log(f"  ERROR: LEF not found: {lef}")
        return None, 0
    if not os.path.isfile(def_file):
        log(f"  ERROR: DEF not found: {def_file}")
        return None, 0
    if not os.path.isfile(guide_path):
        log(f"  ERROR: guide not found: {guide_path}")
        return None, 0

    cmd = (
        f"{CUGR2_BIN} -via_cost {VIA_COST}"
        f" -lef {lef} -def {def_file} -sort 1"
        f" -output {output_guide}"
        f" -dgr {guide_path}"
        f" > {log_path} 2>&1"
    )

    ok, dur = run_cmd(cmd, f"CUGR2 [{label}]", cwd=CUGR_RUN)
    if not ok:
        return None, dur

    metrics = parse_cugr2_log(log_path)
    return metrics, dur


# ════════════════════════════════════════════════════════════════════════
#  CUGR2 runner — pure baseline (NO DGR guide)
# ════════════════════════════════════════════════════════════════════════
def run_cugr2_pure(bench, label, log_path=None):
    """
    Run CUGR2 on a benchmark WITHOUT any DGR guide (pure CUGR2 baseline).

    Returns:
        (metrics_dict, duration_seconds) or (None, duration) on failure
    """
    if log_path is None:
        os.makedirs(LOG_ROOT, exist_ok=True)
        log_path = os.path.join(LOG_ROOT, f"{label}_cugr2.log")

    lef = os.path.join(BENCH_DIR, bench, f"{bench}.input.lef")
    def_file = os.path.join(BENCH_DIR, bench, f"{bench}.input.def")
    output_guide = os.path.join(BENCH_DIR, bench, f"{bench}.{label}.guide")

    if not os.path.isfile(lef):
        log(f"  ERROR: LEF not found: {lef}")
        return None, 0
    if not os.path.isfile(def_file):
        log(f"  ERROR: DEF not found: {def_file}")
        return None, 0

    cmd = (
        f"{CUGR2_BIN} -via_cost {VIA_COST}"
        f" -lef {lef} -def {def_file} -sort 1"
        f" -output {output_guide}"
        f" > {log_path} 2>&1"
    )

    ok, dur = run_cmd(cmd, f"CUGR2 pure [{label}]", cwd=CUGR_RUN)
    if not ok:
        return None, dur

    metrics = parse_cugr2_log(log_path)
    return metrics, dur


# ════════════════════════════════════════════════════════════════════════
#  Snapshot diagnostics (heatmap.txt, overflow_edges.txt, capacity3D.txt)
# ════════════════════════════════════════════════════════════════════════
def snapshot_diagnostics(label):
    """
    Copy heatmap.txt, overflow_edges.txt, capacity3D.txt from cu-gr-2/run/
    into experiments/diagnostics/<label>/.
    Returns the diagnostic directory path.
    """
    diag_dir = os.path.join(DIAG_ROOT, label)
    os.makedirs(diag_dir, exist_ok=True)

    for fn in ("heatmap.txt", "overflow_edges.txt", "capacity3D.txt"):
        src = os.path.join(CUGR_RUN, fn)
        if os.path.isfile(src):
            os.system(f"cp -f {src} {diag_dir}/")

    return diag_dir


# ════════════════════════════════════════════════════════════════════════
#  Heatmap generation
# ════════════════════════════════════════════════════════════════════════
def generate_heatmap(label, diag_dir=None):
    """
    Run visualize_heatmap_v3.py on the diagnostics directory.
    Returns (ok, duration).
    """
    if diag_dir is None:
        diag_dir = os.path.join(DIAG_ROOT, label)

    vis_script = os.path.join(DGR_DIR, "visualize_heatmap_v3.py")
    if not os.path.isfile(vis_script):
        log(f"  WARN: visualize_heatmap_v3.py not found")
        return False, 0

    heatmap_file = os.path.join(diag_dir, "heatmap.txt")
    if not os.path.isfile(heatmap_file):
        log(f"  WARN: heatmap.txt not found in {diag_dir}")
        return False, 0

    out_dir = os.path.join(diag_dir, "figs_v3")
    return run_cmd(
        f"python3 {vis_script} --dir {diag_dir} --out {out_dir}"
        f" --format png --dilate 5 --min_overflow 0.3 --skip_empty",
        f"Heatmap [{label}]",
        cwd=DGR_DIR,
    )


# ════════════════════════════════════════════════════════════════════════
#  DGR runner (with or without warmstart)
# ════════════════════════════════════════════════════════════════════════
def run_dgr(bench, label, warmstart=None, device="0", dgr_iter=2000,
            save_target=None):
    """
    Run main_stochastic.py (DGR) on a benchmark.

    Args:
        bench:       e.g. "ispd18_test5_metal5"
        label:       used for --output_name and naming save_target
        warmstart:   path to warmstart .npz (None → no warmstart)
        device:      GPU device id string
        dgr_iter:    number of DGR iterations
        save_target: path for output .npz (default: <label>_dgr.npz)

    Returns:
        (ok, duration_seconds)
    """
    data_path = os.path.join(DGR_DIR, f"{bench}.pt")
    if not os.path.isfile(data_path):
        log(f"  ERROR: {data_path} not found")
        return False, 0

    if save_target is None:
        save_target = os.path.join(DGR_DIR, f"{label}_dgr.npz")

    ws_arg = warmstart if warmstart else "__nonexistent__"

    cmd = (
        f"python3 main_stochastic.py"
        f" --data_path {data_path}"
        f" --warmstart_file {ws_arg}"
        f" --save_target {save_target}"
        f" --output_name {label}"
        f" --iter {dgr_iter} --pattern_level 1 --device {device}"
    )

    return run_cmd(cmd, f"DGR [{label}]", cwd=DGR_DIR)


# ════════════════════════════════════════════════════════════════════════
#  Find DGR guide file in CUGR2_guide/
# ════════════════════════════════════════════════════════════════════════
def find_guide(label):
    """
    Search CUGR2_guide/ for a guide file matching the label.
    Returns absolute path or None.
    """
    guide_dir = os.path.join(DGR_DIR, "CUGR2_guide")
    if not os.path.isdir(guide_dir):
        return None
    for f in sorted(os.listdir(guide_dir)):
        if label in f and f.endswith(".txt"):
            return os.path.join(guide_dir, f)
    return None


# ════════════════════════════════════════════════════════════════════════
#  Full DGR → CUGR2 → heatmap pipeline
# ════════════════════════════════════════════════════════════════════════
def dgr_cugr2_heatmap(bench, label, warmstart=None, device="0",
                       dgr_iter=2000):
    """
    End-to-end: DGR → find guide → CUGR2 → snapshot → heatmap.

    Returns dict:
      dgr_ok, dgr_time, cugr2_ok, cugr2_time, heatmap_ok, heatmap_time,
      wirelength, via_count, overflow, max_overflow
    """
    result = {
        "dgr_ok": False, "dgr_time": 0,
        "cugr2_ok": False, "cugr2_time": 0,
        "heatmap_ok": False, "heatmap_time": 0,
        "wirelength": None, "via_count": None,
        "overflow": None, "max_overflow": None,
    }

    save_target = os.path.join(DGR_DIR, f"{label}_dgr.npz")

    # 1. DGR
    if not os.path.isfile(save_target):
        ok, dur = run_dgr(bench, label, warmstart=warmstart,
                          device=device, dgr_iter=dgr_iter,
                          save_target=save_target)
        result["dgr_ok"] = ok
        result["dgr_time"] = dur
        if not ok:
            return result
    else:
        log(f"  DGR cached: {save_target}")
        result["dgr_ok"] = True

    # 2. Find guide
    guide = find_guide(label)
    if guide is None:
        log(f"  ERROR: no CUGR2 guide for {label}")
        return result

    # 3. CUGR2
    log_path = os.path.join(LOG_ROOT, f"{label}_cugr2.log")
    os.makedirs(LOG_ROOT, exist_ok=True)

    if not os.path.isfile(log_path) or os.path.getsize(log_path) == 0:
        metrics, dur = run_cugr2_with_guide(bench, label, guide, log_path)
        result["cugr2_time"] = dur
        if metrics is None:
            return result
        result["cugr2_ok"] = True
        result.update(metrics)
    else:
        log(f"  CUGR2 log cached: {log_path}")
        result["cugr2_ok"] = True
        metrics = parse_cugr2_log(log_path)
        result.update(metrics)

    # 4. Snapshot diagnostics
    diag_dir = snapshot_diagnostics(label)

    # 5. Heatmap
    ok, dur = generate_heatmap(label, diag_dir)
    result["heatmap_ok"] = ok
    result["heatmap_time"] = dur

    return result


# ════════════════════════════════════════════════════════════════════════
#  Summary TSV writer
# ════════════════════════════════════════════════════════════════════════
def write_summary_tsv(rows, columns, output_path):
    """
    Write a list of row dicts to a TSV file.

    Args:
        rows:        list of dicts, each mapping column names to values
        columns:     list of column name strings (determines order)
        output_path: path to write the .tsv file
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\t".join(columns) + "\n")
        for row in rows:
            vals = []
            for c in columns:
                v = row.get(c, "")
                if v is None:
                    vals.append("?")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")
    log(f"Summary written: {output_path}")


def print_summary_tsv(path):
    """Pretty-print a TSV file."""
    if not os.path.isfile(path):
        return
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return

    # Compute column widths
    rows = [line.rstrip("\n").split("\t") for line in lines]
    ncols = max(len(r) for r in rows)
    widths = [0] * ncols
    for r in rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(v))

    print()
    for r in rows:
        parts = []
        for i, v in enumerate(r):
            w = widths[i] if i < len(widths) else len(v)
            parts.append(v.rjust(w) if i > 0 else v.ljust(w))
        print("  " + "  ".join(parts))
    print()


# ════════════════════════════════════════════════════════════════════════
#  Build graph
# ════════════════════════════════════════════════════════════════════════
def ensure_graph(bench):
    """Build the heterogeneous graph .pt if it doesn't exist. Returns path."""
    graph_pt = os.path.join(DGR_DIR, f"{bench}_graph.pt")
    if os.path.isfile(graph_pt):
        log(f"  Graph cached: {graph_pt}")
        return graph_pt

    data_path = os.path.join(DGR_DIR, f"{bench}.pt")
    if not os.path.isfile(data_path):
        # Check alternate location
        alt = os.path.join(CUGR_RUN, f"{bench}.pt")
        if os.path.isfile(alt):
            data_path = alt
        else:
            log(f"  ERROR: {bench}.pt not found")
            return None

    ok, _ = run_cmd(
        f"python3 deepdgr_graph_from_dgr.py"
        f" --data_path {data_path} --dgr_dir {DGR_DIR}"
        f" --output {graph_pt} --pattern_level 1",
        f"Build graph [{bench}]",
        cwd=DGR_DIR,
    )
    return graph_pt if ok else None


# ════════════════════════════════════════════════════════════════════════
#  Build teacher (DGR baseline → teacher .npz)
# ════════════════════════════════════════════════════════════════════════
def ensure_teacher(bench, device="0"):
    """Run DGR baseline to produce teacher .npz. Returns path."""
    teacher_npz = os.path.join(DGR_DIR, f"{bench}_teacher.npz")
    if os.path.isfile(teacher_npz):
        log(f"  Teacher cached: {teacher_npz}")
        return teacher_npz

    data_path = os.path.join(DGR_DIR, f"{bench}.pt")
    if not os.path.isfile(data_path):
        alt = os.path.join(CUGR_RUN, f"{bench}.pt")
        if os.path.isfile(alt):
            data_path = alt
        else:
            log(f"  ERROR: {bench}.pt not found")
            return None

    ok, _ = run_cmd(
        f"python3 main_stochastic.py"
        f" --data_path {data_path}"
        f" --warmstart_file __nonexistent__"
        f" --save_target {teacher_npz}"
        f" --output_name {bench}_teacher"
        f" --iter 2000 --pattern_level 1 --device {device}",
        f"Teacher [{bench}]",
        cwd=DGR_DIR,
    )
    return teacher_npz if ok else None
