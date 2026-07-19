#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imf_field.py — The DGR instantaneous-velocity field + e2e graph packs.

This is the piece that makes the iMF/MeanFlow adaptation *teacher-free*:
the instantaneous velocity of the optimization flow at ANY state z is one
reverse-mode autograd call on the differentiable DGR objective —

    v(z) = + S * normalize( d/dz  L_DGR(softmax_per_subnet(z); Omega) )

(in the paper's time convention t: 1 -> 0; stepping backward in t descends
the loss; see imf_core.py).  No 2000-iteration DGR teacher run is involved
anywhere.

`DGRField.objective` re-implements model.objective_function (model.py:218-312)
on top of the tensors stored in an "e2e pack" (see generate_e2e_graphs.py),
with identical math but:
  * full-batch (no per-iteration capacity minibatching / CPU masks),
  * flattened capacities / edge lengths precomputed once per graph,
  * deterministic softmax(z) at temperature 1 with NO Gumbel noise — the
    field must be the exact marginal velocity (iMF Sec. 4.1's central point);
    Gumbel would re-introduce a noisy conditional velocity.
Numerical parity with model.objective_function is asserted in
test_imf_pipeline.py (test_objective_parity).

E2E pack format ("e2e_v1", written by generate_e2e_graphs.py)
-------------------------------------------------------------
  grid_x        [Ng, 4]  f16   — same features as the compact format
  candidate_x   [Nc, 4]  f16
  edges         {4 hetero edge types -> [2, E] i32}   ('connects' may live in
                a shared per-template static pack)
  p_index       [S+1]    i32   — subnet CSR offsets (the compact files lack
                               this; e2e packs always carry it)
  field:
    hor_idx [2, nnz] i32, hor_val [nnz] f16     — hor_path sparse COO
    ver_idx, ver_val                            — ver_path
    via_idx, via_val                            — via_map
    via_count [Nc] i32, wire_length [Nc] f32
    hor_cap [xmax, ymax-1] f32, ver_cap [xmax-1, ymax] f32   (pin/local-net
                               demand already subtracted, exactly like
                               main_stochastic.py:124-138)
    hor_pin_demand / ver_pin_demand [xmax, ymax] f32
    hor_edge_length [ymax-1] f32, ver_edge_length [xmax-1] f32  (post-swap,
                               i.e. after main_stochastic.py:127)
    scalars: m2_pitch, min_unit_length_short_cost, via_layer (already
             sqrt(num_layer)-scaled), xmax, ymax, num_layer, hor_first
  metadata      dict

This file is NEW code; it does not modify any existing file.
"""

import os
import math
from typing import Dict, Optional, Tuple

import torch

from imf_core import (segment_softmax, segment_center, segment_count,
                      sample_init_logits)


# ════════════════════════════════════════════════════════════════════════
#  Pack IO
# ════════════════════════════════════════════════════════════════════════

def load_e2e_pack(path: str, map_location="cpu") -> dict:
    """Load an e2e pack and, if it references a shared static pack (the
    per-template 'connects' edges), merge it in transparently."""
    pack = torch.load(path, map_location=map_location, weights_only=False)
    meta = pack.get("metadata", {})
    static_rel = meta.get("static_pack")
    if static_rel and ("grid", "connects", "grid") not in pack["edges"]:
        static_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                                   static_rel)
        static = torch.load(static_path, map_location=map_location,
                            weights_only=False)
        pack["edges"][("grid", "connects", "grid")] = static["connects"]
    return pack


# ════════════════════════════════════════════════════════════════════════
#  The field
# ════════════════════════════════════════════════════════════════════════

class DGRField:
    """
    Holds one routing problem on `device` and exposes:
        objective(z, via_coeff, wl_coeff)  — differentiable DGR total cost
        velocity(z, ...)                   — v(z) of the iMF flow (Eq. v-map)
        rollout(t_target, ...)             — honest z_t sample: Euler-integrate
                                             the SAME field from a fresh init
                                             at t=1 down to t_target  (this is
                                             our analog of the paper's
                                             z_t = (1-t)x + t*e construction)
        evaluate(z, ...)                   — no-grad cost report
        init(generator)                    — DGR's exact init distribution

    scale S and normalization define the flow's time units: with norm="rms",
    one full jump t:1->0 displaces each logit by ~S on average (warm-starts
    are consumed clamped to [-5, 5] by main_stochastic.py:196, so S ~= 5
    keeps one-step jumps in-range).  The field choice plays the role the
    linear interpolation path plays in flow matching: it is part of the
    problem definition, and the MeanFlow identity holds for it exactly.
    """

    def __init__(self, pack: dict, device: str = "cuda:0",
                 scale: float = 5.0, norm: str = "rms",
                 act_scale: float = 0.5, add_via: bool = True,
                 overflow_coeff: float = 1.0):
        self.device = device
        self.scale = float(scale)
        self.norm = norm
        self.act_scale = float(act_scale)
        self.add_via = bool(add_via)
        self.overflow_coeff = float(overflow_coeff)

        f = pack["field"]
        self.xmax = int(f["xmax"])
        self.ymax = int(f["ymax"])
        self.m2_pitch = float(f["m2_pitch"])
        self.min_ulsc = float(f["min_unit_length_short_cost"])
        self.via_layer = float(f["via_layer"])   # already sqrt(num_layer)-scaled

        dev = device
        n_cand = pack["candidate_x"].shape[0]
        self.n_cand = n_cand

        def coo(idx, val, rows):
            return torch.sparse_coo_tensor(
                idx.long().to(dev), val.float().to(dev), (rows, n_cand),
                is_coalesced=True)

        nE_hor = self.xmax * (self.ymax - 1)
        nE_ver = (self.xmax - 1) * self.ymax
        self.hor_path = coo(f["hor_idx"], f["hor_val"], nE_hor)
        self.ver_path = coo(f["ver_idx"], f["ver_val"], nE_ver)
        self.via_map = coo(f["via_idx"], f["via_val"], self.xmax * self.ymax)
        self.via_count = f["via_count"].float().to(dev)
        self.wire_length = f["wire_length"].float().to(dev)

        # Flattened caps and physical edge lengths, precomputed once
        # (model.objective_function rebuilds these every call: model.py:244-249)
        self.hor_cap = f["hor_cap"].float().to(dev).flatten()
        self.ver_cap = f["ver_cap"].float().to(dev).flatten()
        hor_el = f["hor_edge_length"].float().to(dev).reshape(1, -1)
        ver_el = f["ver_edge_length"].float().to(dev).reshape(-1, 1)
        self.hor_len = hor_el.repeat(self.xmax, 1).flatten()
        self.ver_len = ver_el.repeat(1, self.ymax).flatten()
        self.hor_pin_demand = f["hor_pin_demand"].float().to(dev)
        self.ver_pin_demand = f["ver_pin_demand"].float().to(dev)

        # Subnet structure
        p_index = pack["p_index"].long()
        sizes = (p_index[1:] - p_index[:-1]).to(dev)
        self.n_seg = int(p_index.shape[0] - 1)
        self.seg = torch.repeat_interleave(
            torch.arange(self.n_seg, device=dev), sizes.long())
        self.sizes = sizes.float()
        self.seg_cnt = segment_count(self.seg, self.n_seg, device=dev)
        assert self.seg.shape[0] == n_cand, \
            f"p_index inconsistent: {self.seg.shape[0]} != {n_cand}"

    # ------------------------------------------------------------------
    def probabilities(self, z: torch.Tensor) -> torch.Tensor:
        """Deterministic per-subnet softmax at temperature 1 (no Gumbel)."""
        return segment_softmax(z, self.seg, self.n_seg)

    # ------------------------------------------------------------------
    def objective(self, z: torch.Tensor, via_coeff: float = 4.0,
                  wl_coeff: float = 0.5) -> Tuple[torch.Tensor, dict]:
        """
        Total DGR cost at logits z.  Mirrors, term by term:
          demand           model.py:252-264   (incl. add_via congestion)
          overflow_cost    model.py:308       (sigmoid activation, act_scale)
          via_cost         model.py:310
          wire_length_cost model.py:311
          total            main_stochastic.py:294 (epoch_iter = 1)
        """
        p = self.probabilities(z)
        hor_demand = torch.matmul(self.hor_path, p)
        ver_demand = torch.matmul(self.ver_path, p)
        if self.add_via:
            vm = torch.matmul(self.via_map, p).view(self.xmax, self.ymax)
            hvm = self.hor_pin_demand * vm
            vvm = self.ver_pin_demand * vm
            hor_demand = hor_demand + \
                ((hvm[:, :-1] + hvm[:, 1:]) * self.via_layer).flatten()
            ver_demand = ver_demand + \
                ((vvm[:-1, :] + vvm[1:, :]) * self.via_layer).flatten()

        hor_over = hor_demand - self.hor_cap
        ver_over = ver_demand - self.ver_cap
        overflow_cost = (
            (self.hor_len * self.min_ulsc *
             torch.sigmoid(hor_over * self.act_scale)).sum()
            + (self.ver_len * self.min_ulsc *
               torch.sigmoid(ver_over * self.act_scale)).sum())
        via_cost = (self.via_count * p).sum() * self.via_layer
        wl_cost = (self.wire_length * p).sum() / self.m2_pitch

        total = (overflow_cost * self.overflow_coeff
                 + wl_cost * wl_coeff + via_cost * via_coeff)
        parts = {
            "total": total,
            "overflow": overflow_cost,
            "via": via_cost,
            "wl": wl_cost,
            "max_overflow": torch.relu(
                torch.cat([hor_over, ver_over])).max(),
        }
        return total, parts

    # ------------------------------------------------------------------
    def velocity(self, z: torch.Tensor, via_coeff: float = 4.0,
                 wl_coeff: float = 0.5) -> torch.Tensor:
        """
        v(z) = +S * normalize(grad_z L).  Exact marginal velocity of the
        (normalized) gradient flow; per-subnet zero-sum by construction
        (each subnet's softmax gradient sums to 0).  Detached.
        """
        z = z.detach().requires_grad_(True)
        total, _ = self.objective(z, via_coeff, wl_coeff)
        (g,) = torch.autograd.grad(total, z)
        return self._shape_velocity(g)

    def _shape_velocity(self, g: torch.Tensor) -> torch.Tensor:
        if self.norm == "rms":
            g = g / (g.pow(2).mean().sqrt() + 1e-8)
        elif self.norm != "raw":
            raise ValueError(f"unknown field norm: {self.norm}")
        return (self.scale * g).detach()

    # ------------------------------------------------------------------
    def velocity_and_objective(self, z, via_coeff=4.0, wl_coeff=0.5):
        z = z.detach().requires_grad_(True)
        total, parts = self.objective(z, via_coeff, wl_coeff)
        (g,) = torch.autograd.grad(total, z)
        parts = {k: float(v.detach().item()) for k, v in parts.items()}
        return self._shape_velocity(g), parts

    # ------------------------------------------------------------------
    def init(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """z_1 ~ DGR's init distribution (model.py:48-51), gauge-centered."""
        return sample_init_logits(self.seg, self.n_seg, self.sizes,
                                  generator=generator, center=True)

    # ------------------------------------------------------------------
    def rollout(self, t_target: float, k_max: int = 8,
                via_coeff: float = 4.0, wl_coeff: float = 0.5,
                generator: Optional[torch.Generator] = None,
                z_start: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Construct an honest on-trajectory sample z_t by Euler-integrating the
        field backward in time from t=1 (a fresh init) to t=t_target:

            z <- z - h * v(z),   h = (1 - t_target) / K,  K = ceil(Delta*k_max)

        Cost: K + 1 field evaluations (each ~ one DGR iteration), with
        K <= k_max  —  i.e. ~ k_max/2000 of a teacher run.  Returns
        (z_t, v(z_t), parts_at_z_t); the final field evaluation doubles as
        Alg. 1's regression target/tangent v, so it is never recomputed.
        """
        z = self.init(generator) if z_start is None else z_start.detach()
        delta = max(0.0, 1.0 - float(t_target))
        K = max(0, math.ceil(delta * k_max))
        h = delta / K if K > 0 else 0.0
        for _ in range(K):
            v = self.velocity(z, via_coeff, wl_coeff)
            z = (z - h * v).detach()
        v_t, parts = self.velocity_and_objective(z, via_coeff, wl_coeff)
        return z, v_t, parts

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, z: torch.Tensor, via_coeff: float = 4.0,
                 wl_coeff: float = 0.5) -> dict:
        total, parts = self.objective(z, via_coeff, wl_coeff)
        return {k: float(v.item()) for k, v in parts.items()}


# ════════════════════════════════════════════════════════════════════════
#  GNN-side tensors of a pack (kept separate from the field so the probe
#  in generate_e2e_graphs.py can build a field without touching PyG-style
#  edge dicts, and vice versa).
# ════════════════════════════════════════════════════════════════════════

def pack_graph_tensors(pack: dict, device: str) -> dict:
    """x/edge/p_index tensors for u_theta, moved to device, with the
    per-edge-type mean-aggregation degree precomputed."""
    edges = {}
    for et, ei in pack["edges"].items():
        edges[et] = ei.long().to(device)
    x = {
        "grid": pack["grid_x"].float().to(device),
        "candidate": pack["candidate_x"].float().to(device),
    }
    n_nodes = {"grid": x["grid"].shape[0], "candidate": x["candidate"].shape[0]}
    deg = {}
    for et, ei in edges.items():
        dst_type = et[2]
        deg[et] = torch.bincount(
            ei[1], minlength=n_nodes[dst_type]).clamp(min=1).float()
    return {"x": x, "edges": edges, "deg": deg, "n_nodes": n_nodes}
