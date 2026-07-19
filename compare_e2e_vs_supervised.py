#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_e2e_vs_supervised.py — head-to-head: train the SAME DeepDGR_GNN two
ways for a MATCHED step budget, then evaluate both on held-out instances.
===============================================================================

  (1) streaming-e2e   : physics objective, TEACHER-FREE.  Each step picks a
                        different pool instance (round-robin, via e2e_stream),
                        GNN->gumbel-softmax->DGR objective->backprop into the
                        shared GNN.  This is the e2e_stream.py training rule.

  (2) supervised      : for each instance a TEACHER per-subnet distribution is
                        obtained (from a saved *.npz target if available, else
                        by running a short plain-DGR optimization on that
                        instance's own logits — same FullProblem objective).
                        The GNN is trained to MATCH the teacher: loss =
                        KL(teacher || GNN_softmax) + mse_w * MSE(probs).

Both share: identical DeepDGR_GNN architecture/init, identical pool, identical
held-out split, identical step budget and optimizer/lr — so the only variable
is the training SIGNAL (physics vs teacher).

Evaluation on held-out instances reports, for each model:
  * physics objective (overflow + wl + via) under the GNN's noiseless init,
  * KL-to-teacher (lower = closer to the teacher distribution),
  * top-1 overlap (fraction of subnets whose argmax matches the teacher).

Outputs:
  reports/figs/e2e_vs_supervised.pdf  — academic B&W (no color) bar plots
  compare.csv                         — one row per (model, metric)

NEW file; imports e2e_stream (new) + DeepDGR_GNN / fastdgr_core read-only.

  python3 compare_e2e_vs_supervised.py --synthetic 4 --steps 60 --device -1
  python3 compare_e2e_vs_supervised.py \
      --instances 'cu-gr-2/run/ispd18_test*_metal5.pt' \
      --steps 3000 --device 0 --target_glob 'experiments/.../{name}*target*.npz'
"""

import argparse
import csv
import glob
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# reuse the streaming pool + instance machinery (NEW file, same dir)
from e2e_stream import (Instance, InstancePool, init_gnn, log,
                        make_synthetic_full, derive_graph_from_full)
from fastdgr_core import seg_softmax, seg_sum, seg_max     # read-only


# ════════════════════════════════════════════════════════════════════════
#  Teacher distribution per instance
# ════════════════════════════════════════════════════════════════════════

def teacher_from_npz(path, full):
    """Load a saved teacher distribution (dgr_fast --save_target format:
    'probabilities' / 'logits' / 'best_logits') if shape matches."""
    if not path or not os.path.isfile(path):
        return None
    try:
        z = np.load(path)
        if "probabilities" in z:
            p = torch.tensor(z["probabilities"]).float()
        elif "logits" in z:
            p = seg_softmax(torch.tensor(z["logits"]).float().to(full.device),
                            full.seg, full.S).cpu()
        elif "best_logits" in z:
            p = seg_softmax(torch.tensor(z["best_logits"]).float()
                            .to(full.device), full.seg, full.S).cpu()
        else:
            return None
        if p.shape[0] != full.n:
            return None
        return p.to(full.device)
    except Exception:                                    # pragma: no cover
        return None


def teacher_by_dgr(full, wl_coeff, via_coeff, iters=120, lr=0.8, seed=0):
    """Run a short plain-DGR optimization (RMSprop on per-candidate logits
    against full.objective_full) to produce a reference 'teacher' distribution.
    Same objective the e2e side uses; this is the supervised target when no
    saved teacher exists."""
    dev = full.device
    g = torch.Generator().manual_seed(seed)
    widths = (full.p_index[1:] - full.p_index[:-1]).float()
    u = torch.rand(full.n, generator=g).to(dev)
    logits = (torch.log(u.clamp_min(1e-30)) * widths.to(dev)[full.seg]) \
        .detach().requires_grad_(True)
    opt = torch.optim.RMSprop([logits], lr=lr)
    gen = (torch.Generator(device=dev).manual_seed(seed)
           if str(dev).startswith("cuda") else
           torch.Generator().manual_seed(seed))
    temp = 1.0
    for i in range(iters):
        if i and i % max(1, iters // 10) == 0:
            temp = max(temp * 0.9, 0.1)
        e = torch.empty_like(logits).exponential_(generator=gen)
        gp = -e.log()
        p = seg_softmax((logits + gp) / max(temp, 0.05), full.seg, full.S)
        of, via_c, wl_c, _ = full.objective_full(p)
        loss = of + wl_coeff * wl_c + via_coeff * via_c
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return seg_softmax(logits, full.seg, full.S).detach()


def get_teacher(inst, target_glob, iters, lr):
    """Teacher distribution for an instance: prefer a saved npz, else DGR."""
    if target_glob:
        pat = target_glob.replace("{name}", inst.name)
        for cand in sorted(glob.glob(pat)):
            t = teacher_from_npz(cand, inst.full)
            if t is not None:
                return t, os.path.basename(cand)
    t = teacher_by_dgr(inst.full, inst.wl_coeff, inst.via_coeff,
                       iters=iters, lr=lr)
    return t, "dgr"


# ════════════════════════════════════════════════════════════════════════
#  Metrics
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def metrics(gnn, inst, teacher):
    """(physics objective, KL(teacher||student), top-1 overlap)."""
    logits = gnn(inst.x_dict, inst.edge_index_dict)
    p = seg_softmax(logits, inst.seg, inst.S)
    of, via_c, wl_c, _ = inst.full.objective_full(p)
    phys = float(of + inst.wl_coeff * wl_c + inst.via_coeff * via_c)
    kl = float((teacher * (torch.log(teacher.clamp_min(1e-9))
                           - torch.log(p.clamp_min(1e-9)))).sum()
               / max(inst.S, 1))
    _, a_s = seg_max(p, inst.seg, inst.S)
    _, a_t = seg_max(teacher, inst.seg, inst.S)
    top1 = float((a_s == a_t).float().mean())
    return phys, kl, top1


# ════════════════════════════════════════════════════════════════════════
#  Training rules (matched budget)
# ════════════════════════════════════════════════════════════════════════

def train_e2e(gnn, pool, train_idx, steps, lr, seed, log_every):
    """Streaming-e2e: round-robin, physics objective (the e2e_stream rule)."""
    dev = next(gnn.parameters()).device
    opt = torch.optim.Adam(gnn.parameters(), lr=lr, weight_decay=1e-5)
    gen = (torch.Generator(device=str(dev)).manual_seed(seed)
           if str(dev).startswith("cuda") else
           torch.Generator().manual_seed(seed))
    temp = 1.0
    trace = []
    for t in range(steps):
        if t and t % max(1, steps // 10) == 0:
            temp = max(temp * 0.9, 0.1)
        inst = pool.get(train_idx[t % len(train_idx)])    # round-robin
        loss, of, _, _ = inst.loss(gnn, gen, temp)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 5.0)
        opt.step()
        trace.append(float(loss))
        if t % log_every == 0 or t == steps - 1:
            log(f"  [e2e ] step {t:5d} inst={inst.name:<22s} "
                f"loss={float(loss):.5g} of={float(of):.4g} T={temp:.3f}")
    return trace


def train_supervised(gnn, pool, train_idx, teachers, steps, lr, seed,
                     log_every, mse_w):
    """Supervised: round-robin, minimise KL+MSE to each instance's teacher."""
    dev = next(gnn.parameters()).device
    opt = torch.optim.Adam(gnn.parameters(), lr=lr, weight_decay=1e-5)
    trace = []
    for t in range(steps):
        ti = train_idx[t % len(train_idx)]
        inst = pool.get(ti)                               # round-robin
        teacher = teachers[ti].to(dev)
        logits = gnn(inst.x_dict, inst.edge_index_dict)
        logp = torch.log(seg_softmax(logits, inst.seg, inst.S)
                         .clamp_min(1e-9))
        p = logp.exp()
        kl = (teacher * (torch.log(teacher.clamp_min(1e-9)) - logp)).sum() \
            / max(inst.S, 1)
        mse = ((p - teacher) ** 2).sum() / max(inst.S, 1)
        loss = kl + mse_w * mse
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 5.0)
        opt.step()
        trace.append(float(loss))
        if t % log_every == 0 or t == steps - 1:
            log(f"  [sup ] step {t:5d} inst={inst.name:<22s} "
                f"loss={float(loss):.5g} (KL={float(kl):.4g} "
                f"MSE={float(mse):.4g})")
    return trace


# ════════════════════════════════════════════════════════════════════════
#  Plot (academic, black & white, NO color)
# ════════════════════════════════════════════════════════════════════════

def make_plot(rows, e2e_trace, sup_trace, out_pdf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics_order = ["phys", "kl", "top1"]
    titles = {"phys": "Held-out physics objective\n(lower better)",
              "kl": "Held-out KL to teacher\n(lower better)",
              "top1": "Held-out top-1 overlap\n(higher better)"}
    vals = {m: {} for m in metrics_order}
    for r in rows:
        vals[r["metric"]][r["model"]] = r["value"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    models = ["e2e", "supervised"]
    hatches = ["", "////"]                               # B&W differentiation
    for ax, m in zip(axes[:3], metrics_order):
        ys = [vals[m].get(md, 0.0) for md in models]
        bars = ax.bar(range(len(models)), ys, color="white",
                      edgecolor="black", linewidth=1.2)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models)
        ax.set_title(titles[m], fontsize=9)
        ax.grid(axis="y", color="0.7", linestyle=":", linewidth=0.6)
        for i, y in enumerate(ys):
            ax.text(i, y, f"{y:.3g}", ha="center", va="bottom", fontsize=8)

    # training-loss traces (B&W: solid vs dashed)
    ax = axes[3]
    if e2e_trace:
        ax.plot(range(len(e2e_trace)), e2e_trace, color="black",
                linestyle="-", linewidth=1.0, label="e2e (physics)")
    if sup_trace:
        ax.plot(range(len(sup_trace)), sup_trace, color="black",
                linestyle="--", linewidth=1.0, label="supervised (KL+MSE)")
    ax.set_title("Training loss\n(different objectives)", fontsize=9)
    ax.set_xlabel("step")
    ax.set_yscale("log")
    ax.grid(color="0.7", linestyle=":", linewidth=0.6)
    ax.legend(fontsize=7, frameon=False)

    fig.suptitle("e2e (teacher-free physics) vs supervised (teacher KL) — "
                 "same GNN, matched budget", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"figure written: {out_pdf}")


# ════════════════════════════════════════════════════════════════════════
#  Driver
# ════════════════════════════════════════════════════════════════════════

def build_specs(args):
    specs = []
    if args.synthetic > 0:
        for k in range(args.synthetic):
            specs.append(("synth", 100 + k))
    if args.instances:
        for f in sorted(glob.glob(args.instances)):
            specs.append(("pt", f))
    if not specs:
        raise SystemExit("no instances: pass --instances <glob> or --synthetic N")
    return specs


def run(args):
    dev = (f"cuda:{args.device}" if torch.cuda.is_available() and
           args.device >= 0 else "cpu")
    torch.manual_seed(args.seed)
    specs = build_specs(args)
    n = len(specs)
    holdout = min(args.holdout, max(0, n - 1)) if n > 1 else 0
    train_idx = list(range(n - holdout)) if holdout else list(range(n))
    holdout_idx = list(range(n - holdout, n)) if holdout else list(range(n))
    log(f"pool {n}: {len(train_idx)} train, {len(holdout_idx)} held-out; "
        f"matched budget = {args.steps} steps each")

    pool = InstancePool(specs, dev, args.wl_coeff, args.via_coeff,
                        args.pattern_level, args.max_c, cache=args.cache)

    # teachers for EVERY index we will train on or evaluate against
    need = sorted(set(train_idx) | set(holdout_idx))
    teachers, t_src = {}, {}
    log("building teacher distributions ...")
    for i in need:
        inst = pool.get(i)
        t, src = get_teacher(inst, args.target_glob, args.teacher_iters,
                             args.teacher_lr)
        teachers[i] = t.cpu()                            # keep off-device
        t_src[i] = src
        log(f"  teacher[{inst.name}] <- {src}")

    # ---- two GNNs from the SAME init (same seed -> identical weights) ----
    torch.manual_seed(args.seed)
    gnn_e2e = init_gnn(pool, args.hidden, args.layers, dev)
    torch.manual_seed(args.seed)
    gnn_sup = init_gnn(pool, args.hidden, args.layers, dev)

    log("=" * 72)
    log("TRAIN (1) streaming-e2e (teacher-free physics)")
    log("=" * 72)
    e2e_trace = train_e2e(gnn_e2e, pool, train_idx, args.steps, args.lr,
                          args.seed, args.log_every)
    log("=" * 72)
    log("TRAIN (2) supervised (KL+MSE to teacher)")
    log("=" * 72)
    sup_trace = train_supervised(gnn_sup, pool, train_idx, teachers,
                                 args.steps, args.lr, args.seed,
                                 args.log_every, args.mse_w)

    # ---- evaluate both on held-out ----
    log("=" * 72)
    log("EVALUATE on held-out instances")
    log("=" * 72)
    agg = {"e2e": {"phys": [], "kl": [], "top1": []},
           "supervised": {"phys": [], "kl": [], "top1": []}}
    for i in holdout_idx:
        inst = pool.get(i)
        teach = teachers[i].to(dev)
        pe, ke, te = metrics(gnn_e2e, inst, teach)
        ps, ks, ts = metrics(gnn_sup, inst, teach)
        log(f"  {inst.name:<22s}  e2e[phys={pe:.5g} kl={ke:.4g} top1={te:.3f}]"
            f"  sup[phys={ps:.5g} kl={ks:.4g} top1={ts:.3f}]")
        for k, v in zip(("phys", "kl", "top1"), (pe, ke, te)):
            agg["e2e"][k].append(v)
        for k, v in zip(("phys", "kl", "top1"), (ps, ks, ts)):
            agg["supervised"][k].append(v)

    rows = []
    for model in ("e2e", "supervised"):
        for metric in ("phys", "kl", "top1"):
            rows.append({"model": model, "metric": metric,
                         "value": float(np.mean(agg[model][metric])),
                         "n_holdout": len(holdout_idx), "steps": args.steps})

    csv_path = os.path.join(ROOT, args.out_csv)
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "metric", "value",
                                           "n_holdout", "steps"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"csv written: {csv_path}")
    for r in rows:
        log(f"   {r['model']:<11s} {r['metric']:<5s} = {r['value']:.5g}")

    make_plot(rows, e2e_trace, sup_trace,
              os.path.join(ROOT, args.out_pdf))
    return rows


def build_parser():
    p = argparse.ArgumentParser(
        description="Compare streaming-e2e vs supervised training of the same "
                    "DeepDGR_GNN")
    p.add_argument("--instances", default=None, help="glob of benchmark .pt")
    p.add_argument("--synthetic", type=int, default=0,
                   help="N tiny in-file synthetic problems (CPU smoke test)")
    p.add_argument("--steps", type=int, default=2000,
                   help="MATCHED step budget for both training modes")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--via_coeff", type=float, default=4.0)
    p.add_argument("--wl_coeff", type=float, default=0.5)
    p.add_argument("--pattern_level", type=int, default=1)
    p.add_argument("--max_c", type=int, default=20)
    p.add_argument("--cache", type=int, default=3)
    p.add_argument("--holdout", type=int, default=1)
    p.add_argument("--mse_w", type=float, default=1.0,
                   help="weight on the MSE term of the supervised loss")
    p.add_argument("--target_glob", default=None,
                   help="teacher npz pattern; '{name}' is replaced by instance "
                        "name. If unset/no match, a short DGR run is the teacher")
    p.add_argument("--teacher_iters", type=int, default=120)
    p.add_argument("--teacher_lr", type=float, default=0.8)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out_csv", default="compare.csv")
    p.add_argument("--out_pdf", default="reports/figs/e2e_vs_supervised.pdf")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
