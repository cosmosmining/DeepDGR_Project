#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
guide_writer.py — CUGR2 guide writing that exploits how the router ACTUALLY
consumes guides (CUGR2_FINDINGS.md F2/F3):

  * the guide is a candidate SET per (net, pin); CUGR2's stage-1 DP picks the
    min-cost member under the LIVE grid state — order irrelevant, count free;
  * with `-sort 1`, the per-NET total path count (`num_paths`) is the PRIMARY
    routing-order key (fewer paths -> routed earlier, on an emptier grid).

Modes (all produce the exact `util.write_CUGR_input` file format):

  threshold    cumulative-probability selection — parse-equivalent to the
               original writer at the same select_threshold (parity mode);
  k1           exactly the refined selection (max stickiness, earliest order);
  diverse2     refined pick + the strongest SPATIALLY DISJOINT alternative
               (overlap-penalized), so the escape hatch is a real detour
               instead of the near-identical next-best pattern;
  criticality  per-net mixing (F3): nets touching near-capacity edges in the
               refined discrete solution get 1 path/subnet (early + sticky),
               flexible nets get diverse2 (later + escape option).

NEW file; `util.write_CUGR_input` and CUGR2 stay untouched.
"""

from typing import Optional

import numpy as np
import torch

from fastdgr_core import FullProblem, seg_max


# ════════════════════════════════════════════════════════════════════════
#  Per-subnet candidate selection
# ════════════════════════════════════════════════════════════════════════

def _topk_per_subnet(full: FullProblem, p: torch.Tensor, k: int):
    """List[LongTensor] of up to k global candidate ids per subnet, by prob."""
    out = []
    for s in range(full.S):
        lo, hi = int(full.p_index[s]), int(full.p_index[s + 1])
        ps = p[lo:hi]
        kk = min(k, hi - lo)
        idx = torch.topk(ps, kk).indices + lo
        out.append(idx)
    return out


def _threshold_per_subnet(full: FullProblem, p: torch.Tensor, thresh: float):
    """Candidates per subnet, highest-prob first, until cumsum >= thresh
    (the original writer's semantics, util.py:880)."""
    out = []
    for s in range(full.S):
        lo, hi = int(full.p_index[s]), int(full.p_index[s + 1])
        ps, order = torch.sort(p[lo:hi], descending=True)
        c = torch.cumsum(ps, 0)
        n = int(torch.searchsorted(c, thresh - 1e-6).item()) + 1
        n = min(n, hi - lo)
        out.append(order[:n] + lo)
    return out


def _cells_of(full: FullProblem, cand: int) -> torch.Tensor:
    """Wire cells (hor edge ids offset +0, ver offset +Eh) of one candidate."""
    one = torch.ones(1, dtype=torch.long, device=full.device) * cand
    rh, _, _ = full.hor.gather_cols(one)
    rv, _, _ = full.ver.gather_cols(one)
    return torch.cat([rh, rv + full.hor.shape[0]])


def _diverse2_per_subnet(full: FullProblem, p: torch.Tensor,
                         sel: torch.Tensor, lam: float = 0.5):
    """[pick, best disjoint alternative] per subnet: alternative maximizes
    prob − lam·overlap(pick, alt); straight/width-1 subnets stay single."""
    out = []
    for s in range(full.S):
        lo, hi = int(full.p_index[s]), int(full.p_index[s + 1])
        pick = int(sel[s])
        if hi - lo <= 1:
            out.append(torch.tensor([pick], device=full.device))
            continue
        pick_cells = _cells_of(full, pick)
        pick_set = set(pick_cells.tolist())
        best_score, best_alt = -1e9, None
        for c in range(lo, hi):
            if c == pick:
                continue
            cells = _cells_of(full, c)
            if cells.numel() == 0:
                ov = 0.0
            else:
                ov = sum(1 for x in cells.tolist() if x in pick_set) \
                    / cells.numel()
            score = float(p[c]) - lam * ov
            if score > best_score:
                best_score, best_alt = score, c
        out.append(torch.tensor([pick, best_alt], device=full.device))
    return out


def critical_nets(full: FullProblem, state, p2pat: np.ndarray,
                  margin: float = 0.0):
    """Set of net indices touching an edge with demand >= cap − margin in the
    refined discrete state (discrete_refine.DiscreteState).  These are the
    nets whose DGR decision is at risk (F1) and which should be routed
    early & sticky (F3)."""
    hot_h = (state.tot_h - full.hor_cap) >= -margin
    hot_v = (state.tot_v - full.ver_cap) >= -margin
    rh, _, oh = full.hor.gather_cols(state.sel)
    rv, _, ov = full.ver.gather_cols(state.sel)
    subs = torch.cat([oh[hot_h[rh]], ov[hot_v[rv]]]).unique()
    nets = {int(p2pat[int(full.p_index[s])][0]) for s in subs.tolist()}
    return nets


# ════════════════════════════════════════════════════════════════════════
#  Writer (format identical to util.write_CUGR_input)
# ════════════════════════════════════════════════════════════════════════

def write_guide(path: str, nets, candidate_pool, p2pat: np.ndarray,
                full: FullProblem, selections) -> dict:
    """selections: List[LongTensor] of global candidate ids per subnet.
    Returns stats incl. the per-net path counts that drive `-sort 1`."""
    per_net_paths = {}
    n_lines = 0
    with open(path, "w") as f:
        for s, cands in enumerate(selections):
            for cand in cands.tolist():
                idx = p2pat[cand]          # (netidx, treeidx, pinidx, candidx)
                if (np.asarray(idx) == 0).all():
                    continue
                net = nets[idx[0]]
                f.writelines([net.net_name, ' ', str(net.net_ID), ' ',
                              str(idx[2])])
                f.write('\n')
                candidate = candidate_pool[idx[0]][idx[1]][idx[2]][1][idx[3]]
                via_map = candidate[1]
                for vi in range(via_map.shape[1] - 1, -1, -1):
                    f.writelines([str(via_map[0, vi]), ' ',
                                  str(via_map[1, vi])])
                    f.write('\n')
                per_net_paths[int(idx[0])] = \
                    per_net_paths.get(int(idx[0]), 0) + 1
                n_lines += 1
    counts = np.array(list(per_net_paths.values())) if per_net_paths \
        else np.zeros(1)
    return {"paths_written": n_lines, "nets_in_guide": len(per_net_paths),
            "mean_paths_per_net": float(counts.mean()),
            "max_paths_per_net": int(counts.max())}


def write_guide_mode(path: str, nets, candidate_pool, p2pat: np.ndarray,
                     full: FullProblem, p_soft: torch.Tensor,
                     sel: torch.Tensor, mode: str = "k1",
                     select_threshold: float = 1.0, lam: float = 0.5,
                     crit_margin: float = 0.0,
                     state=None) -> dict:
    """High-level entry used by dgr_fast.py --guide_writer."""
    if mode == "threshold":
        selections = _threshold_per_subnet(full, p_soft, select_threshold)
    elif mode == "k1":
        selections = [sel[s:s + 1].to(full.device) for s in range(full.S)]
    elif mode == "diverse2":
        selections = _diverse2_per_subnet(full, p_soft, sel, lam=lam)
    elif mode == "criticality":
        assert state is not None, "criticality mode needs the refined state"
        crit = critical_nets(full, state, p2pat, margin=crit_margin)
        div = _diverse2_per_subnet(full, p_soft, sel, lam=lam)
        selections = []
        for s in range(full.S):
            netid = int(p2pat[int(full.p_index[s])][0])
            if netid in crit:
                selections.append(sel[s:s + 1].to(full.device))
            else:
                selections.append(div[s])
        stats = write_guide(path, nets, candidate_pool, p2pat, full,
                            selections)
        stats["critical_nets"] = len(crit)
        return stats
    else:
        raise ValueError(f"unknown guide mode {mode}")
    return write_guide(path, nets, candidate_pool, p2pat, full, selections)
