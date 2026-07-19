#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imf_pareto.py — Das–Dennis structured Pareto sampling of the DGR objective
weights Omega = (via_coeff, wl_coeff), adopted from ParetoRouter
(OpenReview 83EC2v6EsL, Oct 2025; see experiments/opt_v2/RESEARCH_NOTES.md).

ParetoRouter shows that a structured simplex grid over objective weights
(instead of random draws) yields an evenly-covered Pareto front from ONE
conditioned flow model.  Our iMF model is already Omega-conditioned
(imf_gnn / deepdgr_gnn_attn in-context conditioning); this module supplies:

  * das_dennis(n_partitions, dim) — the classic simplex lattice;
  * omega_grid(...)  — lattice points mapped to (via_coeff, wl_coeff)
    around the DGR defaults (4.0, 0.5), overflow weight normalized to 1
    (DGR convention: total = of*1 + wl*wl_coeff + via*via_coeff);
  * DasDennisOmegaSampler — drop-in replacement for imf_core.OmegaSampler
    (same __call__(rng) -> (via, wl) signature) cycling the lattice in a
    seeded random order: a training curriculum that covers the front
    uniformly instead of log-uniform random draws.

NEW file; nothing existing is modified.  CPU-tested in test_imf_pareto.py.
"""

from typing import List, Tuple

import torch


def das_dennis(n_partitions: int, dim: int = 3) -> torch.Tensor:
    """All weight vectors w >= 0, sum(w) = 1 on the (dim-1)-simplex with
    components multiples of 1/n_partitions.  [N, dim], N = C(H+d-1, d-1)."""
    if n_partitions <= 0:
        return torch.full((1, dim), 1.0 / dim)

    out: List[List[float]] = []

    def rec(prefix, left, slots):
        if slots == 1:
            out.append(prefix + [left])
            return
        for k in range(left + 1):
            rec(prefix + [k], left - k, slots - 1)

    rec([], n_partitions, dim)
    return torch.tensor(out, dtype=torch.float32) / n_partitions


def omega_grid(n_partitions: int = 6,
               base_via: float = 4.0, base_wl: float = 0.5,
               span: float = 4.0, min_of: float = 1e-3
               ) -> List[Tuple[float, float]]:
    """
    Map simplex weights (l_of, l_wl, l_via) to DGR coefficients.

    With the overflow weight normalized to 1, a balanced lattice point
    (1/3, 1/3, 1/3) maps exactly to the tuned defaults (via=4, wl=0.5);
    the ratio l_x / l_of scales each coefficient, clamped to
    [base/span, base*span] so the grid spans the same range as
    imf_core.OmegaSampler does (span=4 by default).
    """
    pts = das_dennis(n_partitions, 3)
    grid = []
    for l_of, l_wl, l_via in pts.tolist():
        if l_of < min_of:                  # pure-WL/via corner: degenerate
            continue
        via = base_via * (l_via / l_of)
        wl = base_wl * (l_wl / l_of)
        via = min(max(via, base_via / span), base_via * span)
        wl = min(max(wl, base_wl / span), base_wl * span)
        grid.append((round(via, 4), round(wl, 4)))
    # dedup (clamping collapses corners)
    seen, uniq = set(), []
    for p in grid:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


class DasDennisOmegaSampler:
    """Drop-in for imf_core.OmegaSampler: __call__(rng) -> (via, wl).

    Cycles the Das–Dennis grid in a seeded shuffled order, re-shuffling
    each epoch; with probability p_default returns the tuned defaults
    (mirroring OmegaSampler's behavior so the model stays sharpest at the
    operating point that is actually consumed downstream)."""

    def __init__(self, n_partitions: int = 6, p_default: float = 0.5,
                 base_via: float = 4.0, base_wl: float = 0.5,
                 span: float = 4.0, seed: int = 0):
        self.grid = omega_grid(n_partitions, base_via, base_wl, span)
        assert self.grid, "empty omega grid"
        self.p_default = float(p_default)
        self.base = (base_via, base_wl)
        self._g = torch.Generator().manual_seed(seed)
        self._order = torch.randperm(len(self.grid), generator=self._g)
        self._i = 0

    def __call__(self, rng: torch.Generator) -> Tuple[float, float]:
        if torch.rand(1, generator=rng).item() < self.p_default:
            return self.base
        if self._i >= len(self._order):
            self._order = torch.randperm(len(self.grid), generator=self._g)
            self._i = 0
        via, wl = self.grid[int(self._order[self._i])]
        self._i += 1
        return via, wl
