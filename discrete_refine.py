#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discrete_refine.py — exact discrete rounding + hot-subnet local search
(the previously-untested "E10" lever; see FASTDGR_PROPOSAL.md §2).

Operates on a fastdgr_core.FullProblem and a converged candidate
distribution.  Two stages:

  1. best_of_k_rounding : K Gumbel discretizations of the distribution
     (argmax is always sample #0, so the result is never worse than argmax),
     each scored with the EXACT selected-pattern cost — true ReLU overflow
     (length x short-cost weighted, via-in-demand included), WL, via.

  2. refine             : greedy local search.  Find overflowed edges, find
     subnets whose SELECTED pattern (wire or via) touches them, evaluate
     every alternative candidate's exact Δcost via incremental demand-map
     updates, apply the best improving swap.  Rip-up-and-reroute restricted
     to the candidate pool, with a global exact metric.

Everything is torch; works on CPU (tests) and GPU (production).
NEW file; imports existing modules read-only.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from fastdgr_core import FullProblem, seg_max


# ════════════════════════════════════════════════════════════════════════
#  Exact discrete state & scoring
# ════════════════════════════════════════════════════════════════════════

@dataclass
class DiscreteState:
    sel: torch.Tensor          # [S] global candidate id per subnet
    tot_h: torch.Tensor        # [Eh] wire + via-in-overflow demand
    tot_v: torch.Tensor        # [Ev]
    V: torch.Tensor            # (xmax, ymax) via congestion map
    wl_sum: torch.Tensor       # scalar, raw wire_length units
    via_sum: torch.Tensor      # scalar, raw via count


def build_state(full: FullProblem, sel: torch.Tensor) -> DiscreteState:
    sel = sel.to(full.device).long()
    D_h = full.hor.sum_cols(sel)
    D_v = full.ver.sum_cols(sel)
    V = full.via.sum_cols(sel).view(full.xmax, full.ymax)
    if full.add_via:
        add_h, add_v = full.via_add(V)
        tot_h, tot_v = D_h + add_h, D_v + add_v
    else:
        tot_h, tot_v = D_h, D_v
    return DiscreteState(sel=sel, tot_h=tot_h, tot_v=tot_v, V=V,
                         wl_sum=full.wire_length[sel].sum(),
                         via_sum=full.via_count[sel].float().sum())


def score_state(full: FullProblem, st: DiscreteState,
                of_coeff=1.0, wl_coeff=0.5, via_coeff=4.0) -> dict:
    """Exact cost of the selection.  overflow_w uses the same edge weights as
    the continuous objective but with the TRUE ReLU (act_scale=1)."""
    over_h = torch.relu(st.tot_h - full.hor_cap)
    over_v = torch.relu(st.tot_v - full.ver_cap)
    of_w = (full.w_h * over_h).sum() + (full.w_v * over_v).sum()
    of_units = over_h.sum() + over_v.sum()
    max_of = torch.maximum(over_h.max(), over_v.max())
    wl_cost = st.wl_sum / full.m2_pitch
    via_cost = st.via_sum * full.via_layer
    total = of_coeff * of_w + wl_coeff * wl_cost + via_coeff * via_cost
    return {"total": float(total), "overflow_w": float(of_w),
            "overflow_units": float(of_units), "max_overflow": float(max_of),
            "wl_cost": float(wl_cost), "via_cost": float(via_cost)}


# ════════════════════════════════════════════════════════════════════════
#  Best-of-K rounding
# ════════════════════════════════════════════════════════════════════════

def best_of_k_rounding(full: FullProblem, p_full: torch.Tensor, k: int = 16,
                       of_coeff=1.0, wl_coeff=0.5, via_coeff=4.0,
                       generator: Optional[torch.Generator] = None):
    """Returns (sel_best [S], score_best dict, all_scores list).  Sample #0 is
    the per-subnet argmax, so the result can only match or beat argmax."""
    logp = torch.log(p_full.clamp_min(1e-12))
    sels, scores = [], []
    for i in range(max(1, k)):
        if i == 0:
            x = logp
        else:
            e = torch.empty_like(logp)
            if generator is not None:
                e.exponential_(generator=generator)
            else:
                e.exponential_()
            x = logp + (-e.log())
        _, arg = seg_max(x, full.seg, full.S)        # global positions
        st = build_state(full, arg)
        sc = score_state(full, st, of_coeff, wl_coeff, via_coeff)
        sels.append(arg)
        scores.append(sc)
    best = min(range(len(scores)), key=lambda j: scores[j]["total"])
    return sels[best], scores[best], scores


# ════════════════════════════════════════════════════════════════════════
#  Incremental Δ machinery
# ════════════════════════════════════════════════════════════════════════

def _uniq_delta(rows: torch.Tensor, dval: torch.Tensor):
    """Coalesce (rows, dval) so each edge appears once (ReLU is nonlinear)."""
    if rows.numel() == 0:
        return rows, dval
    u, inv = torch.unique(rows, return_inverse=True)
    dd = torch.zeros(u.shape[0], device=rows.device,
                     dtype=dval.dtype).scatter_add_(0, inv, dval)
    return u, dd


def _of_delta(tot: torch.Tensor, cap: torch.Tensor, w: torch.Tensor,
              idx: torch.Tensor, dval: torch.Tensor) -> torch.Tensor:
    """Σ w·[relu(new−cap) − relu(old−cap)] over the touched edges."""
    if idx.numel() == 0:
        return torch.zeros((), device=tot.device)
    old = tot[idx] - cap[idx]
    new = old + dval
    return (w[idx] * (torch.relu(new) - torch.relu(old))).sum()


class _MoveCtx:
    """Per-(cur,alt) edge/cell deltas, shared between evaluate and apply."""

    __slots__ = ("eh", "dh", "ev", "dv", "cells", "dV",
                 "vh_idx", "vh_val", "vv_idx", "vv_val", "dwl", "dvia")

    def __init__(self, full: FullProblem, cur: int, alt: int):
        one = torch.ones(1, dtype=torch.long, device=full.device)
        rc_h, vc_h, _ = full.hor.gather_cols(one * cur)
        ra_h, va_h, _ = full.hor.gather_cols(one * alt)
        self.eh, self.dh = _uniq_delta(torch.cat([rc_h, ra_h]),
                                       torch.cat([-vc_h, va_h]))
        rc_v, vc_v, _ = full.ver.gather_cols(one * cur)
        ra_v, va_v, _ = full.ver.gather_cols(one * alt)
        self.ev, self.dv = _uniq_delta(torch.cat([rc_v, ra_v]),
                                       torch.cat([-vc_v, va_v]))
        self.dwl = (full.wire_length[alt] - full.wire_length[cur])
        self.dvia = (full.via_count[alt] - full.via_count[cur]).float()

        self.cells = self.dV = None
        self.vh_idx = self.vh_val = self.vv_idx = self.vv_val = None
        if full.add_via:
            rc_g, vc_g, _ = full.via.gather_cols(one * cur)
            ra_g, va_g, _ = full.via.gather_cols(one * alt)
            cells, dV = _uniq_delta(torch.cat([rc_g, ra_g]),
                                    torch.cat([-vc_g, va_g]))
            nz = dV != 0
            self.cells, self.dV = cells[nz], dV[nz]
            if self.cells.numel():
                self._via_edge_deltas(full)

    def _via_edge_deltas(self, full: FullProblem):
        """ΔV at cells -> Δ(via-in-overflow demand) at the ≤4 touching edges.
        add_h[i,j] = (pin_h[i,j]·V[i,j] + pin_h[i,j+1]·V[i,j+1])·via_layer"""
        x = self.cells // full.ymax
        y = self.cells % full.ymax
        ph = full.hor_pin_demand.reshape(-1)[self.cells] * self.dV \
            * full.via_layer
        pv = full.ver_pin_demand.reshape(-1)[self.cells] * self.dV \
            * full.via_layer
        eh_idx, eh_val = [], []
        m = y > 0                                   # hor edge (x, y-1)
        eh_idx.append((x[m] * (full.ymax - 1) + y[m] - 1))
        eh_val.append(ph[m])
        m = y < full.ymax - 1                       # hor edge (x, y)
        eh_idx.append((x[m] * (full.ymax - 1) + y[m]))
        eh_val.append(ph[m])
        ev_idx, ev_val = [], []
        m = x > 0                                   # ver edge (x-1, y)
        ev_idx.append(((x[m] - 1) * full.ymax + y[m]))
        ev_val.append(pv[m])
        m = x < full.xmax - 1                       # ver edge (x, y)
        ev_idx.append((x[m] * full.ymax + y[m]))
        ev_val.append(pv[m])
        self.vh_idx, self.vh_val = _uniq_delta(torch.cat(eh_idx),
                                               torch.cat(eh_val))
        self.vv_idx, self.vv_val = _uniq_delta(torch.cat(ev_idx),
                                               torch.cat(ev_val))

    # combined per-direction edge deltas (wire + via-induced)
    def edge_deltas(self):
        eh, dh = self.eh, self.dh
        ev, dv = self.ev, self.dv
        if self.vh_idx is not None and self.vh_idx.numel():
            eh, dh = _uniq_delta(torch.cat([eh, self.vh_idx]),
                                 torch.cat([dh, self.vh_val]))
        if self.vv_idx is not None and self.vv_idx.numel():
            ev, dv = _uniq_delta(torch.cat([ev, self.vv_idx]),
                                 torch.cat([dv, self.vv_val]))
        return eh, dh, ev, dv


def _move_delta(full: FullProblem, st: DiscreteState, ctx: _MoveCtx,
                of_coeff, wl_coeff, via_coeff) -> float:
    eh, dh, ev, dv = ctx.edge_deltas()
    d = of_coeff * (_of_delta(st.tot_h, full.hor_cap, full.w_h, eh, dh)
                    + _of_delta(st.tot_v, full.ver_cap, full.w_v, ev, dv))
    d = d + wl_coeff * ctx.dwl / full.m2_pitch \
        + via_coeff * ctx.dvia * full.via_layer
    return float(d)


def _apply_move(full: FullProblem, st: DiscreteState, s: int, alt: int,
                ctx: _MoveCtx):
    eh, dh, ev, dv = ctx.edge_deltas()
    if eh.numel():
        st.tot_h.scatter_add_(0, eh, dh)
    if ev.numel():
        st.tot_v.scatter_add_(0, ev, dv)
    if ctx.cells is not None and ctx.cells.numel():
        st.V.reshape(-1).scatter_add_(0, ctx.cells, ctx.dV)
    st.wl_sum = st.wl_sum + ctx.dwl
    st.via_sum = st.via_sum + ctx.dvia
    st.sel[s] = alt


# ════════════════════════════════════════════════════════════════════════
#  Hot-subnet greedy local search
# ════════════════════════════════════════════════════════════════════════

def _hot_subnets(full: FullProblem, st: DiscreteState, tol=1e-6):
    """Subnets whose selected pattern (wire or via cell) touches an
    overflowed edge, ordered by their overflow contribution (desc)."""
    hot_h = (st.tot_h - full.hor_cap) > tol
    hot_v = (st.tot_v - full.ver_cap) > tol
    n_hot = int(hot_h.sum() + hot_v.sum())
    if n_hot == 0:
        return torch.zeros(0, dtype=torch.long, device=full.device), 0

    contrib = torch.zeros(full.S, device=full.device)
    r_h, v_h, o_h = full.hor.gather_cols(st.sel)
    m = hot_h[r_h]
    contrib.scatter_add_(0, o_h[m], v_h[m])
    r_v, v_v, o_v = full.ver.gather_cols(st.sel)
    m = hot_v[r_v]
    contrib.scatter_add_(0, o_v[m], v_v[m])

    if full.add_via:
        # cells adjacent to hot edges
        cell_hot = torch.zeros(full.xmax * full.ymax, dtype=torch.bool,
                               device=full.device)
        eh = hot_h.nonzero(as_tuple=True)[0]
        xh, yh = eh // (full.ymax - 1), eh % (full.ymax - 1)
        cell_hot[xh * full.ymax + yh] = True
        cell_hot[xh * full.ymax + yh + 1] = True
        ev = hot_v.nonzero(as_tuple=True)[0]
        xv, yv = ev // full.ymax, ev % full.ymax
        cell_hot[xv * full.ymax + yv] = True
        cell_hot[(xv + 1) * full.ymax + yv] = True
        r_g, v_g, o_g = full.via.gather_cols(st.sel)
        m = cell_hot[r_g]
        contrib.scatter_add_(0, o_g[m], v_g[m] * 0.1)   # weak via tiebreak

    subs = (contrib > 0).nonzero(as_tuple=True)[0]
    order = torch.argsort(contrib[subs], descending=True)
    return subs[order], n_hot


def refine(full: FullProblem, sel: torch.Tensor, of_coeff=1.0, wl_coeff=0.5,
           via_coeff=4.0, passes: int = 5, max_moves: int = 4000,
           tol: float = 1e-9, audit: bool = False, log=print):
    """Greedy exact local search from a discrete selection.  Returns
    (sel, score_dict, stats)."""
    st = build_state(full, sel.clone())
    s0 = score_state(full, st, of_coeff, wl_coeff, via_coeff)
    moves_total, evals_total = 0, 0
    widths = full.p_index[1:] - full.p_index[:-1]

    for pa in range(passes):
        subs, n_hot = _hot_subnets(full, st)
        if n_hot == 0 or subs.numel() == 0:
            break
        accepted = 0
        for s in subs.tolist():
            if moves_total >= max_moves:
                break
            if int(widths[s]) <= 1:
                continue
            cur = int(st.sel[s])
            lo, hi = int(full.p_index[s]), int(full.p_index[s + 1])
            best_d, best_alt, best_ctx = -tol, None, None
            for alt in range(lo, hi):
                if alt == cur:
                    continue
                ctx = _MoveCtx(full, cur, alt)
                d = _move_delta(full, st, ctx, of_coeff, wl_coeff, via_coeff)
                evals_total += 1
                if d < best_d:
                    best_d, best_alt, best_ctx = d, alt, ctx
            if best_alt is not None:
                _apply_move(full, st, s, best_alt, best_ctx)
                accepted += 1
                moves_total += 1
        if audit:
            ref = build_state(full, st.sel)
            for name in ("tot_h", "tot_v"):
                a, b = getattr(st, name), getattr(ref, name)
                assert torch.allclose(a, b, atol=1e-3), f"audit {name} drifted"
            assert torch.allclose(st.V, ref.V, atol=1e-3), "audit V drifted"
        if accepted == 0 or moves_total >= max_moves:
            break

    s1 = score_state(full, st, of_coeff, wl_coeff, via_coeff)
    stats = {"moves": moves_total, "evals": evals_total,
             "score_before": s0, "score_after": s1}
    if log:
        log(f"[refine] moves={moves_total} evals={evals_total} "
            f"of_units {s0['overflow_units']:.1f}->{s1['overflow_units']:.1f} "
            f"total {s0['total']:.1f}->{s1['total']:.1f}")
    return st.sel, s1, stats


# ════════════════════════════════════════════════════════════════════════
#  Guide-vector construction
# ════════════════════════════════════════════════════════════════════════

def selection_to_distribution(full: FullProblem,
                              sel: torch.Tensor) -> torch.Tensor:
    p = torch.zeros(full.n, device=full.device)
    p[sel.to(full.device).long()] = 1.0
    return p


def blend_distribution(full: FullProblem, p_soft: torch.Tensor,
                       sel: torch.Tensor, beta: float = 0.6) -> torch.Tensor:
    """β·onehot(sel) + (1−β)·p_soft.  Each subnet still sums to 1.0; the
    refined pick is guaranteed scatter_max winner, so util.write_CUGR_input
    writes it FIRST and then keeps emitting DGR runner-ups until the
    cumulative probability reaches the select threshold."""
    p = (1.0 - beta) * p_soft.to(full.device)
    p[sel.to(full.device).long()] += beta
    return p
