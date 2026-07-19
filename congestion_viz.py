#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
congestion_viz.py — does CHEAP pattern routing already encode the final
routed congestion?

This script answers a single research question: if you take the *cheapest
possible* congestion estimate — each 2-pin subnet's L/Z/C candidates spread
UNIFORMLY (no optimization at all) — does the resulting per-edge demand field
predict where the *optimized* (post-DGR) overflow ends up?  If yes, the cheap
field is a usable feature/target for a congestion predictor and we do not need
to run the full differentiable optimizer to know where congestion will be.

It produces, for a benchmark:
  (i)  CHEAP map:  per-edge demand at the uniform (1/width) candidate
       distribution  ==  full.hor/ver @ p_uniform  (+ the via additive term,
       exactly as the DGR objective accounts for it).
  (ii) DGR  map :  per-edge demand at the post-DGR optimized distribution
       (FastDGR optimize -> noiseless probabilities).
  optional (iii): CUGR2's own per-edge overflow, if a route is run isolated.

Maps are rendered as side-by-side grayscale heatmaps (overflow = demand-cap,
relu'd) to reports/figs/congestion_<name>.pdf, and the predictive power of the
CHEAP field for the DGR overflow is quantified with Pearson + Spearman
correlation (all edges and top-congested edges) + top-k recall, appended to
congestion_corr.csv.

NEW file.  Imports util.py / data.py / dgr_fast.py / fastdgr_core.py /
tune_cugr2.py READ-ONLY.  CPU-friendly.

  # tiny synthetic instance (CPU, < 2 min) — also generates the .pt
  python3 congestion_viz.py --synth --iter 200

  # a real benchmark
  python3 congestion_viz.py \
      --data_path cu-gr-2/run/ispd18_test5_metal5.pt --iter 600

  # add CUGR2's own overflow as a third reference panel (isolated route)
  python3 congestion_viz.py --data_path cu-gr-2/run/ispd18_test5_metal5.pt \
      --iter 600 --cugr2 --guide CUGR2_guide/CUgr_ispd18_test5_metal5_S2_18_test5.txt
"""

import argparse
import csv
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

# small CPU op-heavy workloads thrash on many-core login nodes — cap threads
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import dgr_fast                                   # read-only (load/build/opt)
from fastdgr_core import ActiveProblem            # read-only

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.2f}] {m}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  args plumbing: dgr_fast.load_benchmark/build_full_problem expect a full
#  knob namespace.  Mirror dgr_fast.build_parser()'s defaults, CPU device.
# ════════════════════════════════════════════════════════════════════════

def make_args(data_path, iters):
    a = SimpleNamespace(
        data_path=data_path, output_name="VIZ", device="cpu", seed=0,
        lr=0.8, optimizer="rmsprop", iter=iters, t=1.0,
        act="sigmoid", act_scale=0.5, celu_alpha=2.0,
        weight_decay=0.0, beta1=0.9,
        via_coeff=4.0, wl_coeff=0.5, overflow_coeff=1.0,
        via_layer=1.5, capacity=1.0, pin_ratio=1.0, local_net_ratio=1.0,
        pattern_level=1, z_step=3, max_z=10, c_step=3, max_c=20,
        max_c_out_ratio=5, select_threshold=1.0, use_gumble=True,
        warmstart_file="", save_target=None,
        # FastDGR optimization knobs
        freeze_thresh=0.995, prune_eps=0.02, check_every=50, patience=6,
        min_iter=min(100, iters), rel_tol=1e-4)
    return a


# ════════════════════════════════════════════════════════════════════════
#  Per-edge demand / overflow maps from a candidate distribution
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def edge_demand(full, p_full):
    """Per-edge horizontal/vertical demand for a FULL-size candidate
    distribution p_full, INCLUDING the via additive term — i.e. exactly the
    `d_h`, `d_v` the DGR objective compares against capacity
    (fastdgr_core.FullProblem.objective_full)."""
    hor_spmv = full.hor.to_spmv()
    ver_spmv = full.ver.to_spmv()
    d_h = hor_spmv(p_full)
    d_v = ver_spmv(p_full)
    if full.add_via:
        via_spmv = full.via.to_spmv()
        V = via_spmv(p_full).view(full.xmax, full.ymax)
        add_h, add_v = full.via_add(V)
        d_h = d_h + add_h
        d_v = d_v + add_v
    return d_h, d_v


@torch.no_grad()
def uniform_distribution(full):
    """p[i] = 1/width(subnet(i)): the cheapest estimate — every L/Z/C
    candidate of a subnet equally likely, NO optimization."""
    widths = (full.p_index[1:] - full.p_index[:-1]).float()
    return (1.0 / widths)[full.seg]


@torch.no_grad()
def maps_from_demand(full, d_h, d_v):
    """Reshape flat per-edge demand to grids and compute overflow = relu(
    demand - capacity).  Returns dict of 2-D numpy arrays."""
    Eh = (full.xmax, full.ymax - 1)
    Ev = (full.xmax - 1, full.ymax)
    dem_h = d_h.view(*Eh).cpu().numpy()
    dem_v = d_v.view(*Ev).cpu().numpy()
    cap_h = full.hor_cap.view(*Eh).cpu().numpy()
    cap_v = full.ver_cap.view(*Ev).cpu().numpy()
    of_h = np.maximum(dem_h - cap_h, 0.0)
    of_v = np.maximum(dem_v - cap_v, 0.0)
    return {"dem_h": dem_h, "dem_v": dem_v, "cap_h": cap_h, "cap_v": cap_v,
            "of_h": of_h, "of_v": of_v}


def edges_to_gcell(of_h, of_v, xmax, ymax):
    """Collapse the two edge grids (hor: (xmax,ymax-1); ver: (xmax-1,ymax))
    onto a common (xmax,ymax) gcell grid by averaging incident edges — for a
    single combined heatmap panel."""
    g = np.zeros((xmax, ymax), dtype=np.float64)
    cnt = np.zeros((xmax, ymax), dtype=np.float64)
    # horizontal edge (x, j) sits between gcells (x,j) and (x,j+1)
    g[:, :-1] += of_h
    cnt[:, :-1] += 1
    g[:, 1:] += of_h
    cnt[:, 1:] += 1
    # vertical edge (i, y) sits between gcells (i,y) and (i+1,y)
    g[:-1, :] += of_v
    cnt[:-1, :] += 1
    g[1:, :] += of_v
    cnt[1:, :] += 1
    return g / np.maximum(cnt, 1)


# ════════════════════════════════════════════════════════════════════════
#  Correlation / prediction probe
# ════════════════════════════════════════════════════════════════════════

def _rankdata(x):
    """Average-rank transform (Spearman support without scipy)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0   # 1-based average rank
        i = j + 1
    return ranks


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    da = np.sqrt((a * a).sum())
    db = np.sqrt((b * b).sum())
    if da < 1e-12 or db < 1e-12:
        return float("nan")
    return float((a * b).sum() / (da * db))


def _spearman(a, b):
    return _pearson(_rankdata(a), _rankdata(b))


def prediction_probe(cheap, final, top_frac=0.05):
    """How well does the CHEAP per-edge field predict the FINAL (post-DGR)
    per-edge field?  cheap/final are 1-D arrays over the SAME edges (here we
    use demand, the smooth congestion signal; overflow is mostly-zero and
    sparse, so correlation on demand is the meaningful predictive question —
    'will this edge be hot?').  Returns metrics dict."""
    cheap = np.asarray(cheap, dtype=np.float64).ravel()
    final = np.asarray(final, dtype=np.float64).ravel()
    n = len(cheap)
    res = {"n_edges": n,
           "pearson": _pearson(cheap, final),
           "spearman": _spearman(cheap, final)}
    # restrict to the top-congested edges of the FINAL map (the ones we care
    # about predicting) and re-correlate there
    k = max(1, int(round(top_frac * n)))
    top_final = np.argpartition(final, -k)[-k:]
    res["topk"] = k
    res["top_frac"] = top_frac
    res["pearson_top"] = _pearson(cheap[top_final], final[top_final])
    res["spearman_top"] = _spearman(cheap[top_final], final[top_final])
    # top-k recall: of the k hottest FINAL edges, how many are in the k
    # hottest CHEAP edges?  (does cheap routing FIND the hot spots?)
    top_cheap = set(np.argpartition(cheap, -k)[-k:].tolist())
    hit = sum(1 for e in top_final.tolist() if e in top_cheap)
    res["topk_recall"] = hit / k
    return res


# ════════════════════════════════════════════════════════════════════════
#  Rendering
# ════════════════════════════════════════════════════════════════════════

def render(name, xmax, ymax, cheap_maps, dgr_maps, cugr2_gcell, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cheap_g = edges_to_gcell(cheap_maps["of_h"], cheap_maps["of_v"],
                             xmax, ymax)
    dgr_g = edges_to_gcell(dgr_maps["of_h"], dgr_maps["of_v"], xmax, ymax)
    panels = [("Cheap pattern-routing overflow\n(uniform candidate dist.)",
               cheap_g),
              ("Post-DGR optimized overflow", dgr_g)]
    if cugr2_gcell is not None:
        panels.append(("CUGR2 routed overflow", cugr2_gcell))

    ncol = len(panels)
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.6))
    if ncol == 1:
        axes = [axes]
    # shared scale across the two DGR-side panels for honest comparison
    vmax = max(cheap_g.max(), dgr_g.max(), 1e-9)
    for ax, (title, g) in zip(axes, panels):
        im = ax.imshow(g.T, origin="lower", cmap="gray_r", aspect="auto",
                       vmin=0.0, vmax=vmax if "CUGR2" not in title else None)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x gcell")
        ax.set_ylabel("y gcell")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Congestion (overflow) — {name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf)
    plt.close(fig)
    log(f"heatmap written: {out_pdf}")


# ════════════════════════════════════════════════════════════════════════
#  Optional CUGR2 reference (ISOLATED route — copy of iso_compare.route_iso)
# ════════════════════════════════════════════════════════════════════════

def cugr2_overflow_gcell(chip, guide, xmax, ymax):
    """Run CUGR2 in an isolated tempdir and parse its per-gcell overflow
    (heatmap.txt / overflow_edges.txt) if the build emits one.  Returns an
    (xmax,ymax) array or None.  Never routes in the shared cu-gr-2/run."""
    import shutil
    import subprocess
    import tempfile
    try:
        import tune_cugr2 as T                    # read-only
    except Exception as e:
        log(f"CUGR2 ref skipped (tune_cugr2 import failed: {e})")
        return None
    lef, deff = T.find_input(chip, "lef"), T.find_input(chip, "def")
    if not (lef and deff):
        log(f"CUGR2 ref skipped (no lef/def for {chip})")
        return None
    cwd = tempfile.mkdtemp(prefix="cgviz_",
                           dir=os.environ.get("TMPDIR", "/tmp"))
    grid = None
    try:
        for dat in ("POWV9.dat", "POST9.dat"):
            src = os.path.join(T.RUN_DIR, dat)
            if os.path.exists(src):
                os.symlink(src, os.path.join(cwd, dat))
        og = os.path.join(cwd, "out.guide")
        lp = os.path.join(cwd, "route.log")
        cmd = [T.ROUTE, "-lef", lef, "-def", deff, "-output", og]
        for k, v in dict(T.BASE_POINT).items():
            cmd += [f"-{k}", str(v)]
        if guide:
            cmd += ["-dgr", guide]
        with open(lp, "w") as lf:
            subprocess.call(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
        # look for any per-edge overflow dump the fork may emit
        for fn in ("heatmap.txt", "overflow_edges.txt", "overflow.txt"):
            fp = os.path.join(cwd, fn)
            if os.path.exists(fp) and os.path.getsize(fp) > 0:
                grid = _parse_cugr2_overflow(fp, xmax, ymax)
                if grid is not None:
                    log(f"CUGR2 ref parsed from {fn}")
                    break
        if grid is None:
            log("CUGR2 ref: no per-edge overflow file emitted by this build")
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
    return grid


def _parse_cugr2_overflow(fp, xmax, ymax):
    """Best-effort parse of `x y [layer] overflow` rows into an (xmax,ymax)
    gcell grid.  Tolerant of unknown formats -> None on failure."""
    g = np.zeros((xmax, ymax), dtype=np.float64)
    cnt = np.zeros((xmax, ymax), dtype=np.float64)
    parsed = 0
    try:
        with open(fp) as fh:
            for line in fh:
                t = line.split()
                if len(t) < 3:
                    continue
                try:
                    x, y = int(float(t[0])), int(float(t[1]))
                    val = float(t[-1])
                except ValueError:
                    continue
                if 0 <= x < xmax and 0 <= y < ymax:
                    g[x, y] += val
                    cnt[x, y] += 1
                    parsed += 1
    except Exception:
        return None
    if parsed == 0:
        return None
    return g / np.maximum(cnt, 1)


# ════════════════════════════════════════════════════════════════════════
#  Driver
# ════════════════════════════════════════════════════════════════════════

def make_synth(out_pt, xmax, ymax, n_nets, utilization, seed):
    """Emit a tiny scratch .pt via generate_scratch_benchmarks (read-only)."""
    from generate_scratch_benchmarks import build_results
    rng = np.random.RandomState(seed * 7919 + 11)
    results, stats = build_results(rng, xmax, ymax, n_nets,
                                   utilization=utilization, num_layer=4,
                                   margin=0.10)
    torch.save(results, out_pt)
    log(f"synthetic benchmark -> {out_pt}  ({stats['n_nets']} nets, "
        f"{xmax}x{ymax}, util {utilization})")
    return out_pt


def run(args):
    fa = make_args(args.data_path, args.iter)
    (pre, name, candidate_pool, p_index, p_index_full, p2pat, hor_path,
     ver_path, wire_length_count, via_info) = dgr_fast.load_benchmark(fa)
    full = dgr_fast.build_full_problem(fa, pre, p_index, hor_path, ver_path,
                                       wire_length_count, via_info)
    xmax, ymax = full.xmax, full.ymax

    # (i) CHEAP: uniform candidate distribution (no optimization)
    p_uniform = uniform_distribution(full)
    d_h0, d_v0 = edge_demand(full, p_uniform)
    cheap_maps = maps_from_demand(full, d_h0, d_v0)
    log(f"cheap (uniform) demand: hor max {float(d_h0.max()):.2f} "
        f"ver max {float(d_v0.max()):.2f}; overflow edges hor "
        f"{int((cheap_maps['of_h'] > 0).sum())} ver "
        f"{int((cheap_maps['of_v'] > 0).sum())}")

    # (ii) DGR: optimize then take the noiseless distribution
    logits0 = dgr_fast.init_logits(full, fa.seed)
    ap, final, iters_run, opt_s, _ = dgr_fast.optimize(full, logits0, fa)
    p_dgr = ap.expand_full()
    d_h1, d_v1 = edge_demand(full, p_dgr)
    dgr_maps = maps_from_demand(full, d_h1, d_v1)
    log(f"DGR optimized demand: hor max {float(d_h1.max()):.2f} "
        f"ver max {float(d_v1.max()):.2f}; overflow edges hor "
        f"{int((dgr_maps['of_h'] > 0).sum())} ver "
        f"{int((dgr_maps['of_v'] > 0).sum())}")

    # PREDICTION PROBE: cheap demand field -> final demand field, per edge.
    # Concatenate hor+ver edges so it is one prediction problem over all edges.
    cheap_all = np.concatenate([cheap_maps["dem_h"].ravel(),
                                cheap_maps["dem_v"].ravel()])
    final_all = np.concatenate([dgr_maps["dem_h"].ravel(),
                                dgr_maps["dem_v"].ravel()])
    probe = prediction_probe(cheap_all, final_all, top_frac=args.top_frac)
    # also probe against final OVERFLOW (the thing we ultimately want hot-spots
    # of) using the cheap DEMAND as the predictor
    of_all = np.concatenate([dgr_maps["of_h"].ravel(),
                             dgr_maps["of_v"].ravel()])
    probe_of = prediction_probe(cheap_all, of_all, top_frac=args.top_frac)

    log("=" * 64)
    log(f"PREDICTION PROBE — does cheap pattern routing predict final?  "
        f"[{name}]")
    log(f"  cheap demand  vs  DGR demand    : "
        f"pearson {probe['pearson']:.4f}  spearman {probe['spearman']:.4f}")
    log(f"    top-{int(args.top_frac*100)}% congested edges     : "
        f"pearson {probe['pearson_top']:.4f}  "
        f"spearman {probe['spearman_top']:.4f}  "
        f"topk_recall {probe['topk_recall']:.4f}")
    log(f"  cheap demand  vs  DGR OVERFLOW  : "
        f"pearson {probe_of['pearson']:.4f}  "
        f"spearman {probe_of['spearman']:.4f}  "
        f"topk_recall {probe_of['topk_recall']:.4f}")
    log("=" * 64)

    # optional CUGR2 reference panel
    cugr2_g = None
    if args.cugr2:
        cugr2_g = cugr2_overflow_gcell(name, args.guide, xmax, ymax)

    # render
    out_pdf = args.out_pdf or os.path.join(ROOT, "reports", "figs",
                                           f"congestion_{name}.pdf")
    render(name, xmax, ymax, cheap_maps, dgr_maps, cugr2_g, out_pdf)

    # write correlation CSV
    csv_path = args.out_csv or os.path.join(ROOT, "congestion_corr.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["benchmark", "n_edges", "iters", "opt_s",
                        "pearson", "spearman",
                        "pearson_top", "spearman_top", "topk_recall",
                        "of_pearson", "of_spearman", "of_topk_recall",
                        "top_frac", "pdf"])
        w.writerow([name, probe["n_edges"], iters_run, round(opt_s, 2),
                    round(probe["pearson"], 4), round(probe["spearman"], 4),
                    round(probe["pearson_top"], 4),
                    round(probe["spearman_top"], 4),
                    round(probe["topk_recall"], 4),
                    round(probe_of["pearson"], 4),
                    round(probe_of["spearman"], 4),
                    round(probe_of["topk_recall"], 4),
                    args.top_frac, out_pdf])
    log(f"correlation row appended: {csv_path}")
    return {"name": name, "pdf": out_pdf, "csv": csv_path,
            "probe": probe, "probe_of": probe_of}


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--data_path", default=None,
                   help="benchmark .pt (results dict). Omit with --synth.")
    p.add_argument("--synth", action="store_true",
                   help="generate a tiny synthetic .pt first and use it")
    p.add_argument("--synth_xmax", type=int, default=40)
    p.add_argument("--synth_ymax", type=int, default=36)
    p.add_argument("--synth_nets", type=int, default=400)
    p.add_argument("--synth_util", type=float, default=0.55)
    p.add_argument("--synth_seed", type=int, default=0)
    p.add_argument("--synth_out", default=None)
    p.add_argument("--iter", type=int, default=400)
    p.add_argument("--top_frac", type=float, default=0.05,
                   help="fraction of edges treated as 'top-congested'")
    p.add_argument("--cugr2", action="store_true",
                   help="add CUGR2's own overflow as a 3rd panel (isolated)")
    p.add_argument("--guide", default=None,
                   help="optional -dgr guide for the CUGR2 reference route")
    p.add_argument("--out_pdf", default=None)
    p.add_argument("--out_csv", default=None)
    return p


def main():
    args = build_parser().parse_args()
    if args.synth:
        out = args.synth_out or os.path.join(
            ROOT, f"scratch_viz_x{args.synth_xmax}y{args.synth_ymax}"
                  f"n{args.synth_nets}_s{args.synth_seed}.pt")
        args.data_path = make_synth(out, args.synth_xmax, args.synth_ymax,
                                    args.synth_nets, args.synth_util,
                                    args.synth_seed)
    if not args.data_path:
        raise SystemExit("provide --data_path or --synth")
    run(args)


if __name__ == "__main__":
    main()
