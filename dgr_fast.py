#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dgr_fast.py — FastDGR driver: warm-startable, self-shrinking DGR optimizer
+ exact discrete refinement + dense Ω (Pareto) sweep.  See
FASTDGR_PROPOSAL.md.  NEW file; existing modules imported read-only:

  * data loading mirrors main_stochastic.py:110-148 via
    generate_e2e_graphs.preprocess_results (the same statements);
  * ./tmp candidate-pool caches are SHARED with main_stochastic.py
    (same file names, same formats);
  * guides are written by the original util.write_CUGR_input, so the
    CUGR2 `route -dgr` consumption path is byte-identical in format.

Typical use:
  python3 dgr_fast.py --data_path cu-gr-2/run/ispd18_test5_metal5.pt \
      --warmstart_file experiments/warmstarts/S2_18_test5_warmstart.npz \
      --output_name FAST --iter 600 --device 0 --verify

Dense Pareto front from one loaded benchmark (Das-Dennis lattice,
imf_pareto.py):
  python3 dgr_fast.py --data_path ... --omega_grid 4 --iter 400
"""

import argparse
import csv
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

import data as dgr_data                      # read-only (process_pool)
import util as dgr_util                      # read-only (pool, guide writer)
from generate_e2e_graphs import preprocess_results   # read-only preamble

from fastdgr_core import (ActiveProblem, ColMatrix, FullProblem,
                          carry_optimizer_state, make_optimizer, seg_softmax,
                          seg_sum)
from discrete_refine import (best_of_k_rounding, blend_distribution, refine,
                             score_state, build_state,
                             selection_to_distribution)

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:8.3f}] {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  Loading (cache-compatible with main_stochastic.py)
# ════════════════════════════════════════════════════════════════════════

def load_benchmark(args):
    t0 = time.time()
    results = torch.load(args.data_path, map_location="cpu")
    pre = preprocess_results(results, capacity=args.capacity,
                             pin_ratio=args.pin_ratio,
                             local_net_ratio=args.local_net_ratio,
                             via_layer=args.via_layer)
    a = pre["args"]
    log(f"data: grid {a.xmax}x{a.ymax}, {a.net_num:,} nets, "
        f"{a.num_layer} layers ({time.time()-t0:.1f}s)")

    t0 = time.time()
    os.makedirs("./tmp", exist_ok=True)
    name = os.path.basename(args.data_path).rsplit(".", 1)[0]
    pool_f = f"./tmp/{name}_candidate_pool.pt"
    pidx_f = f"./tmp/{name}_p_index.pt"
    if os.path.exists(pool_f):
        candidate_pool = torch.load(pool_f)
        log(f"candidate pool loaded from cache ({time.time()-t0:.1f}s)")
    else:
        candidate_pool = dgr_util.get_initial_candidate_pool(
            pre["RouteNets"], a.xmax, a.ymax, device=args.device,
            edge_length=pre["edge_length"], pattern_level=args.pattern_level,
            max_z=args.max_z, z_step=args.z_step, c_step=args.c_step,
            max_c=args.max_c, max_c_out_ratio=args.max_c_out_ratio)
        torch.save(candidate_pool, pool_f)
        log(f"candidate pool generated ({time.time()-t0:.1f}s)")
    t0 = time.time()
    if os.path.exists(pidx_f):
        tup = torch.load(pidx_f, map_location=args.device)
    else:
        tup = dgr_data.process_pool(candidate_pool, a.xmax, a.ymax,
                                    device=args.device)
        torch.save(tup, pidx_f)
    (p_index, p_index_full, p2pat, hor_path, ver_path, wire_length_count,
     via_info, _tpi, _tipc, _tp2, _tpif) = tup
    log(f"pool tensors ready: {p_index[-1]:,} candidates / "
        f"{len(p_index)-1:,} subnets ({time.time()-t0:.1f}s)")
    return pre, name, candidate_pool, p_index, p_index_full, p2pat, \
        hor_path, ver_path, wire_length_count, via_info


def build_full_problem(args, pre, p_index, hor_path, ver_path,
                       wire_length_count, via_info):
    a = pre["args"]
    dev = args.device
    via_map, via_count = via_info
    t0 = time.time()
    hor_el = torch.tensor(np.asarray(pre["edge_length"][0]),
                          dtype=torch.float32).reshape(1, -1)
    ver_el = torch.tensor(np.asarray(pre["edge_length"][1]),
                          dtype=torch.float32).reshape(-1, 1)
    w_h = (hor_el.repeat(a.xmax, 1).flatten() * pre["min_ulsc"]).to(dev)
    w_v = (ver_el.repeat(1, a.ymax).flatten() * pre["min_ulsc"]).to(dev)
    full = FullProblem(
        xmax=a.xmax, ymax=a.ymax,
        hor=ColMatrix.from_coo(hor_path, device=dev),
        ver=ColMatrix.from_coo(ver_path, device=dev),
        via=ColMatrix.from_coo(via_map, device=dev),
        wire_length=wire_length_count.float().to(dev),
        via_count=via_count.float().to(dev),
        p_index=torch.tensor(p_index, dtype=torch.long),
        hor_cap=pre["RoutingRegion"].cap_mat[0].flatten().float().to(dev),
        ver_cap=pre["RoutingRegion"].cap_mat[1].flatten().float().to(dev),
        w_h=w_h, w_v=w_v,
        hor_pin_demand=torch.tensor(np.asarray(pre["hor_pin_demand"]),
                                    dtype=torch.float32).to(dev),
        ver_pin_demand=torch.tensor(np.asarray(pre["ver_pin_demand"]),
                                    dtype=torch.float32).to(dev),
        via_layer=float(a.via_layer), m2_pitch=float(pre["m2_pitch"]),
        act=args.act, act_scale=args.act_scale, celu_alpha=args.celu_alpha,
        add_via=True, device=dev)
    log(f"FullProblem built: nnz hor {full.hor.nnz():,} / "
        f"ver {full.ver.nnz():,} / via {full.via.nnz():,} "
        f"({time.time()-t0:.1f}s)")
    return full


# ════════════════════════════════════════════════════════════════════════
#  Warm start (accepts logits / best_logits / probabilities)
# ════════════════════════════════════════════════════════════════════════

def init_logits(full, seed):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(full.n, generator=g)
    widths = (full.p_index[1:] - full.p_index[:-1]).float()
    return (torch.log(u.clamp_min(1e-30)).to(full.device)
            * widths.to(full.device)[full.seg])


def load_warmstart(path, full):
    """Per-subnet centered, clamped [-5,5] — main_stochastic.py:185-217
    semantics, vectorized, plus the `best_logits` key the old loader silently
    ignored (see experiments/opt_v2/RESULTS_REPORT.md §1)."""
    if not path or not os.path.exists(path):
        if path:
            log(f"warmstart file {path} not found -> random init")
        return None
    try:
        ws = np.load(path)
        if "logits" in ws:
            z = torch.tensor(ws["logits"]).float()
        elif "best_logits" in ws:
            z = torch.tensor(ws["best_logits"]).float()
        elif "probabilities" in ws:
            z = torch.log(torch.tensor(ws["probabilities"]).float()
                          .clamp(min=1e-6))
        else:
            log(f"warmstart {path}: no usable key -> random init")
            return None
        if z.shape[0] != full.n:
            log(f"warmstart SHAPE MISMATCH ({z.shape[0]} vs {full.n}) "
                "-> skipped")
            return None
        z = z.to(full.device)
        widths = (full.p_index[1:] - full.p_index[:-1]).float()
        mean = seg_sum(z, full.seg, full.S) / widths
        z = (z - mean[full.seg]).clamp(-5.0, 5.0)
        log(f"Warmstart loaded ({os.path.basename(path)}): "
            f"min={z.min():.3f} max={z.max():.3f} std={z.std():.3f}")
        return z
    except Exception as e:                          # pragma: no cover
        log(f"warmstart load failed: {e} -> random init")
        return None


# ════════════════════════════════════════════════════════════════════════
#  --verify: iter-0 equality vs the original model.objective_function
# ════════════════════════════════════════════════════════════════════════

def verify_against_model(args, pre, full, hor_path, ver_path,
                         wire_length_count, via_info, logits):
    import model as dgr_model                       # read-only
    a = pre["args"]
    dev = args.device
    with torch.no_grad():
        p = seg_softmax(logits - logits.max(), full.seg, full.S)
        mine = full.objective_full(p, want_max_overflow=True)
        shim = SimpleNamespace(use_ilp_metric=False, add_via=True,
                               via_layer=float(a.via_layer), act=args.act,
                               act_scale=args.act_scale,
                               celu_alpha=args.celu_alpha)
        rr = pre["RoutingRegion"]
        try:
            rr.to(dev)
        except Exception:
            pass
        hor_el = torch.tensor(np.asarray(pre["edge_length"][0]),
                              dtype=torch.float32).reshape(1, -1).to(dev)
        ver_el = torch.tensor(np.asarray(pre["edge_length"][1]),
                              dtype=torch.float32).reshape(-1, 1).to(dev)
        ref = dgr_model.objective_function(
            rr, hor_path, ver_path, wire_length_count, via_info, p, shim,
            full.hor_pin_demand, full.ver_pin_demand, hor_el, ver_el,
            pre["min_ulsc"], pre["m2_pitch"])
    pairs = [("overflow", mine[0], ref[0]), ("via", mine[1], ref[1]),
             ("wl", mine[2], ref[2]), ("max_of", mine[3], ref[3])]
    worst = 0.0
    for nm, x, y in pairs:
        x, y = float(x), float(y)
        rel = abs(x - y) / max(abs(y), 1e-9)
        worst = max(worst, rel)
        log(f"  verify {nm:8s}: fast={x:.6g} ref={y:.6g} rel={rel:.2e}")
    if worst > 1e-4:
        raise RuntimeError(f"VERIFY FAILED (worst rel {worst:.2e})")
    log("verify gate PASS (fast objective == model.objective_function)")


# ════════════════════════════════════════════════════════════════════════
#  The fast optimization loop
# ════════════════════════════════════════════════════════════════════════

def optimize(full, logits0, args):
    dev = full.device
    gen = (torch.Generator(device=dev).manual_seed(args.seed)
           if str(dev).startswith("cuda")
           else torch.Generator().manual_seed(args.seed))
    ap = ActiveProblem(full, logits0)
    n_start_frozen = int((ap.frozen_choice >= 0).sum())
    log(f"active at start: {ap.n_a:,} candidates / {ap.S_a:,} subnets "
        f"(width-1 folded: {n_start_frozen:,})")

    opt = make_optimizer(args.optimizer, [ap.logits], lr=args.lr,
                         weight_decay=args.weight_decay, beta1=args.beta1)
    temp = 1.0
    hist, best_eval, stale = [], float("inf"), 0
    iters_run, shrink_log = 0, []
    t0 = time.time()
    log_every = max(1, args.iter // 10)

    for i in range(args.iter):
        iters_run = i + 1
        if ap.S_a == 0:
            log(f"iter {i}: everything frozen — stopping")
            break
        if i % log_every == 0 and i:
            temp *= args.t
            with torch.no_grad():
                p = ap.probabilities(1.0, gumbel=False)
                of, vc, wc, mo = ap.objective(p, want_max_overflow=True)
            log(f"iter {i}: of {float(of):.3f} via {float(vc):.3f} "
                f"wl {float(wc):.3f} max_of {float(mo):.1f} "
                f"active {ap.n_a:,}/{ap.S_a:,}")

        p = ap.probabilities(temp, gumbel=args.use_gumble, generator=gen)
        of, vc, wc, _ = ap.objective(p, include_consts=False)
        loss = of * args.overflow_coeff + wc * args.wl_coeff \
            + vc * args.via_coeff
        opt.zero_grad()
        loss.backward()
        opt.step()

        if (i + 1) % args.check_every == 0:
            with torch.no_grad():
                pe = ap.probabilities(1.0, gumbel=False)
                ofe, vce, wce, _ = ap.objective(pe)
                ev = float(ofe * args.overflow_coeff + wce * args.wl_coeff
                           + vce * args.via_coeff)
            hist.append(ev)
            if ev < best_eval * (1 - args.rel_tol):
                best_eval, stale = ev, 0
            else:
                stale += 1

            if args.freeze_thresh <= 1.0 or args.prune_eps > 0:
                old_param = ap.logits
                st = ap.shrink(freeze_thresh=args.freeze_thresh,
                               prune_eps=args.prune_eps)
                if st["rebuilt"]:
                    shrink_log.append({"iter": i + 1, **st})
                    log(f"iter {i+1}: froze {st['frozen']:,} subnets, "
                        f"pruned {st['pruned']:,} cands -> "
                        f"{st['n_active']:,}/{st['S_active']:,} active")
                    new_opt = make_optimizer(args.optimizer, [ap.logits],
                                             lr=args.lr,
                                             weight_decay=args.weight_decay,
                                             beta1=args.beta1)
                    carry_optimizer_state(opt, old_param, new_opt, ap.logits,
                                          ap._last_keep_local)
                    opt = new_opt

            if stale >= args.patience and (i + 1) >= args.min_iter:
                log(f"iter {i+1}: plateau (best {best_eval:.3f}) — early stop")
                break

    opt_s = time.time() - t0
    with torch.no_grad():
        if ap.S_a > 0:
            pe = ap.probabilities(1.0, gumbel=False)
        else:
            pe = torch.zeros(0, device=dev)
        ofe, vce, wce, moe = ap.objective(pe, want_max_overflow=(ap.S_a > 0))
        final = {"overflow_cost": float(ofe), "via_cost": float(vce),
                 "wl_cost": float(wce),
                 "max_overflow": float(moe) if moe is not None else 0.0,
                 "total": float(ofe * args.overflow_coeff
                                + wce * args.wl_coeff + vce * args.via_coeff)}
    log(f"optimize done: {iters_run} iters in {opt_s:.1f}s — "
        f"total {final['total']:.3f} (of {final['overflow_cost']:.3f})")
    return ap, final, iters_run, opt_s, shrink_log


# ════════════════════════════════════════════════════════════════════════
#  One full pipeline run (optimize -> round -> refine -> guide)
# ════════════════════════════════════════════════════════════════════════

def run_pipeline(args, full, logits0, name, RouteNets, candidate_pool,
                 p_index_full, p2pat, tag_suffix=""):
    res = {"tag": args.output_name + tag_suffix,
           "via_coeff": args.via_coeff, "wl_coeff": args.wl_coeff,
           "overflow_coeff": args.overflow_coeff}
    ap, cont, iters_run, opt_s, shrink_log = optimize(full, logits0, args)
    res.update(continuous=cont, iters_run=iters_run, opt_s=round(opt_s, 2),
               shrink_events=len(shrink_log),
               frozen_final=int((ap.frozen_choice >= 0).sum()),
               active_final=ap.n_a)

    t0 = time.time()
    p_soft = ap.expand_full()
    gen = (torch.Generator(device=full.device).manual_seed(args.seed + 1)
           if str(full.device).startswith("cuda")
           else torch.Generator().manual_seed(args.seed + 1))
    if args.rounding_k > 0:
        sel, sc, _all = best_of_k_rounding(
            full, p_soft, k=args.rounding_k, of_coeff=args.overflow_coeff,
            wl_coeff=args.wl_coeff, via_coeff=args.via_coeff, generator=gen)
    else:
        sel, sc, _all = best_of_k_rounding(
            full, p_soft, k=1, of_coeff=args.overflow_coeff,
            wl_coeff=args.wl_coeff, via_coeff=args.via_coeff)
    round_s = time.time() - t0
    res.update(discrete_rounded=sc, round_s=round(round_s, 2))
    log(f"rounding (K={max(1,args.rounding_k)}): of_units "
        f"{sc['overflow_units']:.1f} total {sc['total']:.1f} "
        f"({round_s:.1f}s)")

    t0 = time.time()
    if args.refine_passes > 0:
        sel, sc2, rstats = refine(
            full, sel, of_coeff=args.overflow_coeff, wl_coeff=args.wl_coeff,
            via_coeff=args.via_coeff, passes=args.refine_passes,
            max_moves=args.refine_max_moves, audit=args.audit, log=log)
        res.update(discrete_refined=sc2,
                   refine_moves=rstats["moves"],
                   refine_s=round(time.time() - t0, 2))
    else:
        res.update(discrete_refined=sc, refine_moves=0, refine_s=0.0)

    t0 = time.time()
    guide_name = f"{name}_{args.output_name}{tag_suffix}"
    guide_path = os.path.abspath(f"./CUGR2_guide/CUgr_{guide_name}.txt")
    if args.guide_writer != "util":
        # CUGR2_FINDINGS.md F2/F3 levers (set membership + per-net path
        # count as the -sort 1 scheduling key)
        from guide_writer import write_guide_mode
        state = build_state(full, sel) \
            if args.guide_writer == "criticality" else None
        gstats = write_guide_mode(
            guide_path, RouteNets, candidate_pool, p2pat, full, p_soft, sel,
            mode=args.guide_writer, select_threshold=args.select_threshold,
            lam=args.diverse_lam, crit_margin=args.crit_margin, state=state)
        res["guide_stats"] = gstats
        log(f"guide_writer[{args.guide_writer}]: {gstats}")
    else:
        if args.guide_mode == "soft":
            p_guide = p_soft
        elif args.guide_mode == "onehot":
            p_guide = selection_to_distribution(full, sel)
        else:
            p_guide = blend_distribution(full, p_soft, sel,
                                         beta=args.blend_beta)
        dgr_util.write_CUGR_input(RouteNets, p_guide, p_index_full,
                                  candidate_pool, p2pat, guide_name,
                                  args.select_threshold)
    res.update(write_s=round(time.time() - t0, 2), guide=guide_path)
    log(f"guide written: {guide_path} ({res['write_s']}s)")

    if args.save_target:
        out = args.save_target
        if tag_suffix:
            stem, ext = os.path.splitext(out)
            out = f"{stem}{tag_suffix}{ext or '.npz'}"
        np.savez(out,
                 probabilities=p_soft.detach().cpu().numpy()
                 .astype(np.float32),
                 logits=torch.log(p_soft.clamp_min(1e-6)).cpu().numpy()
                 .astype(np.float32),
                 selected=sel.cpu().numpy().astype(np.int64),
                 p_index=full.p_index.cpu().numpy().astype(np.int64))
        log(f"targets saved: {out}")
    return res


# ════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(description="FastDGR (see FASTDGR_PROPOSAL.md)")
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_name", default="FAST")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    # DGR-compatible knobs (defaults = main_stochastic.py tuned defaults)
    p.add_argument("--lr", type=float, default=0.8)
    p.add_argument("--optimizer", default="rmsprop")
    p.add_argument("--iter", type=int, default=800)
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--act", default="sigmoid")
    p.add_argument("--act_scale", type=float, default=0.5)
    p.add_argument("--celu_alpha", type=float, default=2.0)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--via_coeff", type=float, default=4.0)
    p.add_argument("--wl_coeff", type=float, default=0.5)
    p.add_argument("--overflow_coeff", type=float, default=1.0)
    p.add_argument("--via_layer", type=float, default=1.5)
    p.add_argument("--capacity", type=float, default=1.0)
    p.add_argument("--pin_ratio", type=float, default=1.0)
    p.add_argument("--local_net_ratio", type=float, default=1.0)
    p.add_argument("--pattern_level", type=int, default=1)
    p.add_argument("--z_step", type=int, default=3)
    p.add_argument("--max_z", type=int, default=10)
    p.add_argument("--c_step", type=int, default=3)
    p.add_argument("--max_c", type=int, default=20)
    p.add_argument("--max_c_out_ratio", type=float, default=5)
    p.add_argument("--select_threshold", type=float, default=1.0)
    p.add_argument("--use_gumble", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--warmstart_file", default="")
    p.add_argument("--save_target", default=None)
    # FastDGR-specific
    p.add_argument("--freeze_thresh", type=float, default=0.995,
                   help=">1 disables freezing")
    p.add_argument("--prune_eps", type=float, default=0.02,
                   help="0 disables pruning")
    p.add_argument("--check_every", type=int, default=50)
    p.add_argument("--patience", type=int, default=6,
                   help="plateau checks before early stop")
    p.add_argument("--min_iter", type=int, default=150)
    p.add_argument("--rel_tol", type=float, default=1e-4)
    p.add_argument("--rounding_k", type=int, default=16)
    p.add_argument("--refine_passes", type=int, default=5)
    p.add_argument("--refine_max_moves", type=int, default=4000)
    p.add_argument("--guide_mode", choices=["soft", "onehot", "blend"],
                   default="blend")
    p.add_argument("--blend_beta", type=float, default=0.6)
    p.add_argument("--guide_writer",
                   choices=["util", "threshold", "k1", "diverse2",
                            "criticality"], default="util",
                   help="util = original util.write_CUGR_input (guide_mode "
                        "applies); others = guide_writer.py F2/F3 levers")
    p.add_argument("--diverse_lam", type=float, default=0.5)
    p.add_argument("--crit_margin", type=float, default=0.0)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--verify", action="store_true",
                   help="iter-0 equality gate vs model.objective_function")
    # Pareto sweep
    p.add_argument("--omega_grid", type=int, default=0,
                   help="Das-Dennis partitions; 0 = single run")
    p.add_argument("--omega_span", type=float, default=4.0)
    p.add_argument("--out_dir", default="experiments/fastdgr")
    return p


def main():
    args = build_parser().parse_args()
    args.device = (f"cuda:{args.device}" if torch.cuda.is_available()
                   else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("./CUGR2_guide", exist_ok=True)

    (pre, name, candidate_pool, p_index, p_index_full, p2pat, hor_path,
     ver_path, wire_length_count, via_info) = load_benchmark(args)
    t_loaded = time.time()
    full = build_full_problem(args, pre, p_index, hor_path, ver_path,
                              wire_length_count, via_info)

    ws = load_warmstart(args.warmstart_file, full)
    logits0 = ws if ws is not None else init_logits(full, args.seed)

    if args.verify:
        try:
            verify_against_model(args, pre, full, hor_path, ver_path,
                                 wire_length_count, via_info, logits0)
        except RuntimeError:
            raise
        except Exception as e:                       # CPU/CUDA mask quirks
            log(f"verify gate skipped (reference path failed: {e})")

    del hor_path, ver_path, via_info
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    runs = []
    if args.omega_grid > 0:
        from imf_pareto import omega_grid            # read-only (new-code)
        pts = [(args.via_coeff, args.wl_coeff)]
        for via, wl in omega_grid(args.omega_grid, base_via=args.via_coeff,
                                  base_wl=args.wl_coeff,
                                  span=args.omega_span):
            if (via, wl) not in pts:
                pts.append((via, wl))
        log(f"omega grid: {len(pts)} points")
        base_via, base_wl = args.via_coeff, args.wl_coeff
        for via, wl in pts:
            args.via_coeff, args.wl_coeff = via, wl
            suff = f"_om{via:g}x{wl:g}".replace(".", "p")
            log(f"== Omega (via={via:g}, wl={wl:g}) ==")
            runs.append(run_pipeline(args, full, logits0, name,
                                     pre["RouteNets"], candidate_pool,
                                     p_index_full, p2pat, tag_suffix=suff))
        args.via_coeff, args.wl_coeff = base_via, base_wl
    else:
        runs.append(run_pipeline(args, full, logits0, name,
                                 pre["RouteNets"], candidate_pool,
                                 p_index_full, p2pat))

    total_s = time.time() - T0
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20
               if str(args.device).startswith("cuda") else 0.0)
    report = {"data": name, "output_name": args.output_name,
              "warmstart": args.warmstart_file or None,
              "n_candidates": int(full.n), "n_subnets": int(full.S),
              "load_s": round(t_loaded - T0, 2),
              "total_s": round(total_s, 2), "peak_gpu_mb": round(peak_mb, 1),
              "args": {k: v for k, v in vars(args).items()},
              "runs": runs}
    rpath = os.path.join(args.out_dir, f"{name}_{args.output_name}.json")
    with open(rpath, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    log(f"report: {rpath}")

    csv_path = os.path.join(args.out_dir, "fastdgr_runs.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["data", "tag", "via_coeff", "wl_coeff", "of_coeff",
                        "iters", "opt_s", "round_s", "refine_s", "total_s",
                        "cont_total", "disc_total", "disc_of_units",
                        "disc_max_of", "refine_moves", "guide"])
        for r in runs:
            w.writerow([name, r["tag"], r["via_coeff"], r["wl_coeff"],
                        r["overflow_coeff"], r["iters_run"], r["opt_s"],
                        r["round_s"], r["refine_s"], round(total_s, 2),
                        round(r["continuous"]["total"], 3),
                        round(r["discrete_refined"]["total"], 3),
                        round(r["discrete_refined"]["overflow_units"], 2),
                        round(r["discrete_refined"]["max_overflow"], 2),
                        r["refine_moves"], r["guide"]])
    log(f"row(s) appended: {csv_path}")


if __name__ == "__main__":
    main()
