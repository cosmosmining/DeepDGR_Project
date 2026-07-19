#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fastdgr_core.py — exact problem-shrinking core for FastDGR.

Speeds up the DGR optimization loop (main_stochastic.py semantics) with three
*exact* reductions plus mechanical wins:

  1. fold-at-start   : width-1 subnets have softmax == 1 forever; their demand
                       is constant -> subtract from capacity, drop columns.
  2. freeze          : subnets with a stable argmax and max-prob >= thresh are
                       folded to that argmax (the vertex rounding would pick).
  3. prune           : candidates with negligible noiseless probability
                       (outside the per-subnet top-2) are deleted; tensors are
                       physically rebuilt; RMSprop state is carried over.

  + cached CSR/CSR^T sparse matmuls (custom autograd.Function; no per-iter
    transpose/coalesce), and a hoisted-constant objective that equals
    model.objective_function (verified by test_fastdgr.py and the dgr_fast.py
    --verify runtime gate).

NEW file; imports existing modules read-only.  See FASTDGR_PROPOSAL.md.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

try:
    from torch_scatter import scatter_softmax, scatter_max
    _HAVE_TS = True
except Exception:                                    # pragma: no cover
    _HAVE_TS = False


# ════════════════════════════════════════════════════════════════════════
#  scatter helpers (torch_scatter when available, pure-torch fallback)
# ════════════════════════════════════════════════════════════════════════

def seg_softmax(x: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Per-segment softmax (same semantics as torch_scatter.scatter_softmax)."""
    if _HAVE_TS:
        return scatter_softmax(x, seg)
    mx = torch.full((n_seg,), -float("inf"), device=x.device, dtype=x.dtype)
    mx = mx.scatter_reduce(0, seg, x, reduce="amax", include_self=True)
    ex = torch.exp(x - mx[seg])
    den = torch.zeros(n_seg, device=x.device, dtype=x.dtype).scatter_add_(0, seg, ex)
    return ex / den[seg]


def seg_max(x: torch.Tensor, seg: torch.Tensor, n_seg: int):
    """Per-segment (max, argmax-global-position)."""
    if _HAVE_TS:
        return scatter_max(x, seg, dim_size=n_seg)
    mx = torch.full((n_seg,), -float("inf"), device=x.device, dtype=x.dtype)
    mx = mx.scatter_reduce(0, seg, x, reduce="amax", include_self=True)
    is_max = x == mx[seg]
    pos = torch.arange(x.shape[0], device=x.device)
    arg = torch.full((n_seg,), x.shape[0], device=x.device, dtype=torch.long)
    # first max position per segment
    arg = arg.scatter_reduce(0, seg[is_max], pos[is_max], reduce="amin",
                             include_self=True)
    return mx, arg


def seg_sum(x: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    out = torch.zeros(n_seg, device=x.device, dtype=x.dtype)
    return out.scatter_add_(0, seg, x)


# ════════════════════════════════════════════════════════════════════════
#  Column-slicable sparse matrix (handmade CSC arrays)
# ════════════════════════════════════════════════════════════════════════

class ColMatrix:
    """Sparse matrix stored as CSC-style plain arrays, supporting:
       * fast column subsetting (rebuild for prune/freeze),
       * column gathers (discrete refinement deltas),
       * constant-column sums (folding),
       * conversion to a cached-CSR SpMV autograd op.

    Built from a COO sparse tensor; duplicates are coalesced (summed), which
    matches what torch.matmul(sparse, dense) computes."""

    def __init__(self, rows, col_ptr, vals, shape, device):
        self.rows = rows            # int64 [nnz], row index per entry
        self.col_ptr = col_ptr      # int64 [C+1]
        self.vals = vals            # float32 [nnz]
        self.shape = shape          # (R, C)
        self.device = device

    # -- constructors ------------------------------------------------------
    @staticmethod
    def from_coo(A: torch.Tensor, device=None) -> "ColMatrix":
        A = A.coalesce()
        device = device or A.device
        idx = A.indices().to(device)
        val = A.values().float().to(device)
        R, C = A.shape
        order = torch.argsort(idx[1] * R + idx[0])      # sort by (col, row)
        rows = idx[0][order].contiguous()
        cols = idx[1][order]
        vals = val[order].contiguous()
        counts = torch.bincount(cols, minlength=C)
        col_ptr = torch.zeros(C + 1, dtype=torch.long, device=device)
        col_ptr[1:] = torch.cumsum(counts, 0)
        return ColMatrix(rows, col_ptr, vals, (R, C), device)

    # -- ops ----------------------------------------------------------------
    def _gather_index(self, cols: torch.Tensor):
        """Flat nnz indices of the requested columns plus per-column counts."""
        cols = cols.to(self.device)
        counts = self.col_ptr[cols + 1] - self.col_ptr[cols]
        total = int(counts.sum())
        if total == 0:
            e = torch.zeros(0, dtype=torch.long, device=self.device)
            return e, counts
        owner = torch.repeat_interleave(torch.arange(cols.shape[0],
                                                     device=self.device), counts)
        offs = torch.cumsum(counts, 0) - counts          # start offset per col
        pos = torch.arange(total, device=self.device) - offs[owner]
        flat = self.col_ptr[cols][owner] + pos
        return flat, counts

    def gather_cols(self, cols: torch.Tensor):
        """(rows, vals, owner) of all entries in the given columns; `owner[k]`
        is the position in `cols` the k-th entry belongs to."""
        flat, counts = self._gather_index(cols)
        owner = torch.repeat_interleave(
            torch.arange(cols.shape[0], device=self.device), counts)
        return self.rows[flat], self.vals[flat], owner

    def sum_cols(self, cols: torch.Tensor) -> torch.Tensor:
        """Dense [R] = sum of the given columns (for constant folding)."""
        out = torch.zeros(self.shape[0], device=self.device)
        if cols.numel() == 0:
            return out
        r, v, _ = self.gather_cols(cols)
        return out.scatter_add_(0, r, v)

    def slice_cols(self, keep: torch.Tensor) -> "ColMatrix":
        """New ColMatrix containing only columns `keep` (in that order)."""
        flat, counts = self._gather_index(keep)
        col_ptr = torch.zeros(keep.shape[0] + 1, dtype=torch.long,
                              device=self.device)
        col_ptr[1:] = torch.cumsum(counts, 0)
        return ColMatrix(self.rows[flat].contiguous(), col_ptr,
                         self.vals[flat].contiguous(),
                         (self.shape[0], keep.shape[0]), self.device)

    def to_spmv(self) -> "SpMV":
        return SpMV(self)

    def nnz(self) -> int:
        return int(self.vals.shape[0])


# ════════════════════════════════════════════════════════════════════════
#  Cached-CSR SpMV with explicit backward
# ════════════════════════════════════════════════════════════════════════

def _csr_from_arrays(rows, cols_owner, vals, shape, device):
    coo = torch.sparse_coo_tensor(torch.stack([rows, cols_owner]), vals,
                                  shape, device=device).coalesce()
    return coo.to_sparse_csr()


def _mv(A_csr, x):
    try:
        return torch.mv(A_csr, x)
    except Exception:                                # pragma: no cover
        return torch.sparse.mm(A_csr, x.unsqueeze(1)).squeeze(1)


class _SpMVFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, A_csr, At_csr):
        ctx.At_csr = At_csr
        return _mv(A_csr, p)

    @staticmethod
    def backward(ctx, grad_out):
        return _mv(ctx.At_csr, grad_out.contiguous()), None, None


class SpMV:
    """y = A @ p with cached CSR for forward and CSR^T for backward."""

    def __init__(self, M: ColMatrix):
        R, C = M.shape
        dev = M.device
        counts = M.col_ptr[1:] - M.col_ptr[:-1]
        cols = torch.repeat_interleave(torch.arange(C, device=dev), counts)
        self.A_csr = _csr_from_arrays(M.rows, cols, M.vals, (R, C), dev)
        self.At_csr = _csr_from_arrays(cols, M.rows, M.vals, (C, R), dev)
        self.shape = (R, C)

    def __call__(self, p: torch.Tensor) -> torch.Tensor:
        return _SpMVFn.apply(p, self.A_csr, self.At_csr)


# ════════════════════════════════════════════════════════════════════════
#  Activation (parity with model.objective_function)
# ════════════════════════════════════════════════════════════════════════

def make_act(name: str, celu_alpha: float = 2.0):
    if name == "sigmoid":
        return torch.sigmoid
    if name == "relu":
        return torch.relu
    if name == "celu":
        return torch.nn.CELU(alpha=celu_alpha)
    if name == "leaky_relu":
        return torch.nn.LeakyReLU()
    if name == "exp":
        return torch.exp
    raise NotImplementedError(name)


# ════════════════════════════════════════════════════════════════════════
#  Full problem (immutable inputs)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class FullProblem:
    """Everything the DGR objective needs, in flat/hoisted form.  Mirrors what
    main_stochastic.py hands to model.objective_function (epoch_iter=1,
    use_ilp_metric=False, add_via=True path)."""
    xmax: int
    ymax: int
    hor: ColMatrix              # (xmax*(ymax-1), n)
    ver: ColMatrix              # ((xmax-1)*ymax, n)
    via: ColMatrix              # (xmax*ymax, n)
    wire_length: torch.Tensor   # [n]
    via_count: torch.Tensor     # [n]
    p_index: torch.Tensor       # int64 [S+1]
    hor_cap: torch.Tensor       # [Eh] flattened cap_mat[0]
    ver_cap: torch.Tensor       # [Ev]
    w_h: torch.Tensor           # [Eh] edge_length * min_unit_length_short_cost
    w_v: torch.Tensor           # [Ev]
    hor_pin_demand: torch.Tensor  # (xmax, ymax)
    ver_pin_demand: torch.Tensor  # (xmax, ymax)
    via_layer: float            # ALREADY scaled by sqrt(num_layer)
    m2_pitch: float
    act: str = "sigmoid"
    act_scale: float = 0.5
    celu_alpha: float = 2.0
    add_via: bool = True
    device: str = "cpu"
    seg: torch.Tensor = field(init=False)     # [n] subnet id per candidate
    n: int = field(init=False)
    S: int = field(init=False)

    def __post_init__(self):
        self.p_index = self.p_index.to(torch.long).to(self.device)
        widths = self.p_index[1:] - self.p_index[:-1]
        self.seg = torch.repeat_interleave(
            torch.arange(widths.shape[0], device=self.device), widths)
        self.n = int(self.p_index[-1])
        self.S = int(widths.shape[0])

    # -- via-in-overflow additive term (linear in the via map) -------------
    def via_add(self, V: torch.Tensor):
        """V: (xmax, ymax) via congestion map -> (add_h [Eh], add_v [Ev])."""
        hv = self.hor_pin_demand * V
        vv = self.ver_pin_demand * V
        add_h = (hv[:, :-1] + hv[:, 1:]).reshape(-1) * self.via_layer
        add_v = (vv[:-1, :] + vv[1:, :]).reshape(-1) * self.via_layer
        return add_h, add_v

    def objective_full(self, p: torch.Tensor, want_max_overflow=False):
        """Reference objective on the FULL candidate set (no folding) —
        numerically equals model.objective_function (test T2/verify gate)."""
        hor_spmv = getattr(self, "_hor_spmv", None)
        if hor_spmv is None:
            self._hor_spmv = self.hor.to_spmv()
            self._ver_spmv = self.ver.to_spmv()
            self._via_spmv = self.via.to_spmv()
        d_h = self._hor_spmv(p)
        d_v = self._ver_spmv(p)
        if self.add_via:
            V = self._via_spmv(p).view(self.xmax, self.ymax)
            add_h, add_v = self.via_add(V)
            d_h = d_h + add_h
            d_v = d_v + add_v
        a = make_act(self.act, self.celu_alpha)
        over_h = d_h - self.hor_cap
        over_v = d_v - self.ver_cap
        of = (self.w_h * a(over_h * self.act_scale)).sum() \
            + (self.w_v * a(over_v * self.act_scale)).sum()
        via_cost = (self.via_count * p).sum() * self.via_layer
        wl_cost = (self.wire_length * p).sum() / self.m2_pitch
        mo = None
        if want_max_overflow:
            mo = torch.maximum(torch.relu(over_h).max(), torch.relu(over_v).max())
        return of, via_cost, wl_cost, mo


# ════════════════════════════════════════════════════════════════════════
#  Active problem (shrinkable view + folded constants)
# ════════════════════════════════════════════════════════════════════════

class ActiveProblem:
    """The live optimization state: active candidates only, with all frozen
    subnets folded into effective capacities and cost constants.

    Invariant (exact, no approximation): for any active distribution p_a,
        objective_active(p_a) == objective_full(expand(p_a))
    where expand() places one-hot vectors on frozen subnets."""

    def __init__(self, full: FullProblem, logits_full: torch.Tensor):
        self.full = full
        dev = full.device
        widths = full.p_index[1:] - full.p_index[:-1]

        # start: every multi-candidate subnet is active; width-1 are frozen.
        self.frozen_choice = torch.full((full.S,), -1, dtype=torch.long,
                                        device=dev)  # global candidate id
        single = (widths == 1).nonzero(as_tuple=True)[0]
        self.frozen_choice[single] = full.p_index[single]

        active_sub = (widths > 1).nonzero(as_tuple=True)[0]
        mask = torch.zeros(full.n, dtype=torch.bool, device=dev)
        mask[self.frozen_choice[self.frozen_choice >= 0]] = True   # frozen cands
        cand_keep = (~mask).nonzero(as_tuple=True)[0]
        # but only candidates of active subnets (width-1 candidates are the
        # frozen ones, so cand_keep is exactly the active-subnet candidates)
        self.cand_id = cand_keep                       # global ids, sorted
        self.sub_id = active_sub                       # global subnet ids

        # folded constants from the width-1 candidates
        fr = self.frozen_choice[self.frozen_choice >= 0]
        self.cap_eff_h = full.hor_cap.clone()
        self.cap_eff_v = full.ver_cap.clone()
        self.wl_const = torch.tensor(0.0, device=dev)
        self.via_const = torch.tensor(0.0, device=dev)
        self._fold_const(fr)

        self._rebuild_tensors(logits_full[self.cand_id].clone())
        self.last_argmax = None                        # for freeze stability

    # ------------------------------------------------------------------ —
    def _fold_const(self, cand_ids: torch.Tensor):
        """Fold the (constant) demand + costs of the given global candidates
        into effective capacity / cost constants."""
        f = self.full
        if cand_ids.numel() == 0:
            return
        self.cap_eff_h -= f.hor.sum_cols(cand_ids)
        self.cap_eff_v -= f.ver.sum_cols(cand_ids)
        if f.add_via:
            Vc = f.via.sum_cols(cand_ids).view(f.xmax, f.ymax)
            add_h, add_v = f.via_add(Vc)
            self.cap_eff_h -= add_h
            self.cap_eff_v -= add_v
        self.wl_const = self.wl_const + f.wire_length[cand_ids].sum() / f.m2_pitch
        self.via_const = self.via_const + f.via_count[cand_ids].sum() * f.via_layer

    def _rebuild_tensors(self, logits_active: torch.Tensor):
        """(Re)build sliced matrices/vectors + segment index for cand_id."""
        f = self.full
        dev = f.device
        self.hor_a = f.hor.slice_cols(self.cand_id)
        self.ver_a = f.ver.slice_cols(self.cand_id)
        self.via_a = f.via.slice_cols(self.cand_id) if f.add_via else None
        self.spmv_h = self.hor_a.to_spmv()
        self.spmv_v = self.ver_a.to_spmv()
        self.spmv_via = self.via_a.to_spmv() if f.add_via else None
        self.wl_a = f.wire_length[self.cand_id]
        self.vc_a = f.via_count[self.cand_id]
        # local segments: candidates stay grouped per subnet (cand_id sorted)
        g_seg = f.seg[self.cand_id]
        uniq, seg_local = torch.unique_consecutive(g_seg, return_inverse=True)
        assert uniq.shape[0] == self.sub_id.shape[0], "subnet bookkeeping broke"
        self.seg_a = seg_local
        self.S_a = int(uniq.shape[0])
        self.n_a = int(self.cand_id.shape[0])
        self.logits = torch.nn.Parameter(logits_active.detach().clone())

    # ------------------------------------------------------------------ —
    def probabilities(self, temp: float = 1.0, gumbel: bool = False,
                      generator: Optional[torch.Generator] = None):
        if self.n_a == 0:
            return torch.zeros(0, device=self.full.device)
        x = self.logits
        if gumbel:
            e = torch.empty_like(x)
            e.exponential_(generator=generator) if generator is not None \
                else e.exponential_()
            x = x + (-e.log())
        x = x / temp
        return seg_softmax(x - x.max(), self.seg_a, self.S_a)

    def objective(self, p_a: torch.Tensor, want_max_overflow=False,
                  include_consts=True):
        """Exact objective on the active slice (frozen demand folded into
        cap_eff).  Constants only shift via/wl costs, never gradients."""
        f = self.full
        if self.n_a == 0:
            d_h = torch.zeros_like(self.cap_eff_h)
            d_v = torch.zeros_like(self.cap_eff_v)
        else:
            d_h = self.spmv_h(p_a)
            d_v = self.spmv_v(p_a)
            if f.add_via:
                V = self.spmv_via(p_a).view(f.xmax, f.ymax)
                add_h, add_v = f.via_add(V)
                d_h = d_h + add_h
                d_v = d_v + add_v
        a = make_act(f.act, f.celu_alpha)
        over_h = d_h - self.cap_eff_h
        over_v = d_v - self.cap_eff_v
        of = (f.w_h * a(over_h * f.act_scale)).sum() \
            + (f.w_v * a(over_v * f.act_scale)).sum()
        via_cost = (self.vc_a * p_a).sum() * f.via_layer
        wl_cost = (self.wl_a * p_a).sum() / f.m2_pitch
        if include_consts:
            via_cost = via_cost + self.via_const
            wl_cost = wl_cost + self.wl_const
        mo = None
        if want_max_overflow:
            mo = torch.maximum(torch.relu(over_h).max(), torch.relu(over_v).max())
        return of, via_cost, wl_cost, mo

    # ------------------------------------------------------------------ —
    @torch.no_grad()
    def shrink(self, freeze_thresh: float = 0.995, prune_eps: float = 0.02,
               keep_top: int = 2, min_change_frac: float = 0.02):
        """Freeze decided subnets + prune negligible candidates.  Returns a
        dict of stats; mutates self only if the change is worth a rebuild.
        Exactness: freezing happens at the subnet's argmax vertex; pruning
        removes candidates that then get exactly renormalized softmax
        (equivalent to sending their logits to -inf)."""
        p_hat = self.probabilities(temp=1.0, gumbel=False)
        mx, arg = seg_max(p_hat, self.seg_a, self.S_a)        # arg: global pos
        # freeze only after the argmax has been OBSERVED stable across two
        # consecutive checks (first call only records)
        if self.last_argmax is not None and \
                self.last_argmax.shape[0] == self.S_a:
            stable = self.cand_id[arg] == self.last_argmax
        else:
            stable = torch.zeros(self.S_a, dtype=torch.bool,
                                 device=p_hat.device)
        self.last_argmax = self.cand_id[arg].clone()

        freeze_sub = (mx >= freeze_thresh) & stable           # local subnet ids

        # prune mask: keep if top-2 of subnet or prob >= eps; subnets being
        # frozen contribute exactly their argmax candidate.
        keep = p_hat >= prune_eps
        tmp = p_hat.clone()
        arg1 = arg.clone()
        keep[arg1[arg1 < self.n_a]] = True
        if keep_top >= 2:
            tmp[arg1[arg1 < self.n_a]] = -1.0
            _, arg2 = seg_max(tmp, self.seg_a, self.S_a)
            keep[arg2[arg2 < self.n_a]] = True
        # frozen subnets: only the argmax survives (as a fold, not a column)
        in_frozen = freeze_sub[self.seg_a]
        keep &= ~in_frozen

        n_freeze = int(freeze_sub.sum())
        n_prune = int((~keep & ~in_frozen).sum())
        if (n_freeze + n_prune) < max(1, int(min_change_frac * self.n_a)):
            return {"frozen": 0, "pruned": 0, "n_active": self.n_a,
                    "S_active": self.S_a, "rebuilt": False}

        # fold the newly frozen subnets
        new_frozen_global_cand = self.cand_id[arg[freeze_sub]]
        new_frozen_global_sub = self.sub_id[freeze_sub]
        self.frozen_choice[new_frozen_global_sub] = new_frozen_global_cand
        self._fold_const(new_frozen_global_cand)

        # rebuild the active slice
        old_logits = self.logits.detach()
        keep_idx_local = keep.nonzero(as_tuple=True)[0]
        self.cand_id = self.cand_id[keep_idx_local]
        self.sub_id = self.sub_id[~freeze_sub]
        self._last_keep_local = keep_idx_local       # for optimizer-state carry
        self._rebuild_tensors(old_logits[keep_idx_local])
        if n_freeze > 0:
            # subnet set changed -> stability record invalid; prune-only
            # rebuilds keep it (global candidate ids are unchanged and the
            # argmax candidate is never pruned)
            self.last_argmax = None
        return {"frozen": n_freeze, "pruned": n_prune, "n_active": self.n_a,
                "S_active": self.S_a, "rebuilt": True}

    # ------------------------------------------------------------------ —
    @torch.no_grad()
    def expand_full(self, p_a: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full-size candidate distribution: one-hot on frozen subnets, the
        given (or current noiseless) distribution on active subnets.  Each
        subnet sums to exactly 1.0 -> safe for util.write_CUGR_input."""
        if p_a is None:
            p_a = self.probabilities(temp=1.0, gumbel=False)
        full_p = torch.zeros(self.full.n, device=self.full.device)
        fr = self.frozen_choice[self.frozen_choice >= 0]
        full_p[fr] = 1.0
        full_p[self.cand_id] = p_a
        return full_p

    @torch.no_grad()
    def selection_full(self, p_a: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-subnet selected GLOBAL candidate id (argmax on active)."""
        if p_a is None:
            p_a = self.probabilities(temp=1.0, gumbel=False)
        sel = self.frozen_choice.clone()
        if self.S_a > 0:
            _, arg = seg_max(p_a, self.seg_a, self.S_a)
            sel[self.sub_id] = self.cand_id[arg]
        return sel


# ════════════════════════════════════════════════════════════════════════
#  Optimizer-state carry (RMSprop & friends) across rebuilds
# ════════════════════════════════════════════════════════════════════════

def make_optimizer(name, params, lr, weight_decay=0.0, beta1=0.9):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay,
                                betas=(beta1, 0.999))
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "adagrad":
        return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    raise NotImplementedError(name)


def carry_optimizer_state(old_opt, old_param, new_opt, new_param,
                          keep_idx: torch.Tensor):
    """Slice per-element optimizer buffers to the kept candidate subset so a
    rebuild does not reset adaptivity."""
    old_state = old_opt.state.get(old_param, None)
    if not old_state:
        return
    new_state = {}
    for k, v in old_state.items():
        if torch.is_tensor(v) and v.shape == old_param.shape:
            new_state[k] = v[keep_idx].clone()
        elif torch.is_tensor(v):
            new_state[k] = v.clone()
        else:
            new_state[k] = v
    new_opt.state[new_param] = new_state
