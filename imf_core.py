#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
imf_core.py — Improved MeanFlow (iMF) core machinery for DeepDGR.

Implements the method of:
    "Improved Mean Flows: On the Challenges of Fastforward Generative Models"
    (Geng, Lu, Wu, Shechtman, Kolter, He — arXiv:2512.02012v2)
adapted from image generation to the *DGR optimization flow*:

    The ODE we "fastforward" is (normalized) gradient flow of the DGR
    objective over candidate logits z.  In the paper's time convention,
    t = 1 is the initialization ("noise") and t = 0 is the converged
    solution ("data"); along a trajectory  dz/dt = v(z)  with

        v(z) = + S * normalize( dL_DGR/dz )                      (autonomous)

    so that stepping BACKWARD in t (1 -> 0) descends the loss.  Unlike the
    generative setting, v(z) here is EXACT and computable everywhere via one
    reverse-mode autograd call on the differentiable DGR objective
    (model.objective_function) — i.e. *the teacher (2000-iter DGR run) is
    never needed*.  This is precisely the regime iMF Sec. 4.1 argues for:
    the JVP tangent and regression target are the true marginal velocity,
    not a noisy conditional sample.

Paper ↔ code map
----------------
  Eq. (3)  u(z_t, r, t) average velocity        -> the GNN u_theta (imf_gnn.py)
  Eq. (8)  v = u + (t-r) du/dt                  -> meanflow_step()
  Eq. (12) V_theta = u + (t-r) JVP_sg(u; v)     -> meanflow_step()  [iMF objective]
  Alg. 1   jvp + stopgrad + metric              -> meanflow_step(), dudt via
                                                   torch.func.jvp (or finite diff)
  Tab. 4   t,r ~ logit-normal(-0.4, 1.0),
           "ratio of r!=t 50%"                  -> TimeSampler
  Sec. 4.2 guidance scale as conditioning       -> OmegaSampler: DGR loss
           (Alg. 2)                                coefficients (via_coeff,
                                                   wl_coeff) become the "CFG
                                                   scale" Omega; sampled at
                                                   train time, free at test
                                                   time  ->  Pareto knob.
  Sec. 4.3 in-context conditioning              -> condition embeddings are
                                                   concatenated to every node's
                                                   input features (imf_gnn.py);
                                                   no adaLN.
  MF adaptive weighting ("metric")              -> adaptive_subnet_loss()

All segment (per-subnet) ops below are written with NATIVE torch ops
(index_select / out-of-place index_add) instead of torch_scatter, because
torch.func.jvp (forward-mode AD) must flow through u_theta and torch_scatter
kernels do not implement forward-AD rules.  Equivalence with
torch_scatter.scatter_softmax is asserted in test_imf_pipeline.py.

This file is NEW code; it does not modify any existing file.
"""

import math
import dataclasses
from typing import Callable, Optional, Tuple

import torch


# ════════════════════════════════════════════════════════════════════════
#  Segment (per-subnet) primitives — forward-AD safe
#  seg:  LongTensor [N] mapping element i -> segment id (= DGR p_full_index)
# ════════════════════════════════════════════════════════════════════════

def segment_sum(x: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    """Sum of x per segment. x: [N] or [N, C]."""
    shape = (n_seg,) + tuple(x.shape[1:])
    out = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return out.index_add(0, seg, x)


def segment_count(seg: torch.Tensor, n_seg: int,
                  device=None, dtype=torch.float32) -> torch.Tensor:
    """Element count per segment (clamped to >= 1)."""
    cnt = torch.bincount(seg, minlength=n_seg).to(dtype)
    return cnt.clamp(min=1).to(device or seg.device)


def segment_mean(x: torch.Tensor, seg: torch.Tensor, n_seg: int,
                 cnt: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean of x per segment. cnt may be precomputed by segment_count()."""
    if cnt is None:
        cnt = segment_count(seg, n_seg, device=x.device, dtype=x.dtype)
    s = segment_sum(x, seg, n_seg)
    if x.dim() > 1:
        cnt = cnt.view(-1, *([1] * (x.dim() - 1)))
    return s / cnt


def segment_center(x: torch.Tensor, seg: torch.Tensor, n_seg: int,
                   cnt: Optional[torch.Tensor] = None) -> torch.Tensor:
    """x minus its per-segment mean (the softmax gauge projection)."""
    m = segment_mean(x, seg, n_seg, cnt)
    return x - m.index_select(0, seg)


def segment_softmax(z: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
    """
    softmax(z) within each segment.  Numerically stabilized by subtracting the
    per-segment max; the max is DETACHED, which is exact for both reverse- and
    forward-mode AD (softmax(z - c) == softmax(z) for any constant c).
    Equivalent to torch_scatter.scatter_softmax(z, seg) (asserted in tests).
    """
    with torch.no_grad():
        m = torch.full((n_seg,), float("-inf"), dtype=z.dtype, device=z.device)
        m.index_reduce_(0, seg, z.detach(), "amax", include_self=True)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    e = torch.exp(z - m.index_select(0, seg))
    s = segment_sum(e, seg, n_seg)
    return e / s.index_select(0, seg).clamp(min=1e-30)


# ════════════════════════════════════════════════════════════════════════
#  Initialization sampler — replicates DGR's Net.__init__ (model.py:48-51)
# ════════════════════════════════════════════════════════════════════════

def sample_init_logits(seg: torch.Tensor, n_seg: int, sizes: torch.Tensor,
                       generator: Optional[torch.Generator] = None,
                       center: bool = True) -> torch.Tensor:
    """
    DGR initializes per-subnet logits as  log(U(0,1)) * subnet_size
    (model.py lines 48-51, including the "truly random probability" note).
    We replicate that distribution exactly, vectorized:
        z_i = log(u_i) * size(subnet(i)),  u_i ~ U(0,1)

    center=True additionally removes the per-subnet mean.  This is a pure
    gauge choice (softmax is shift-invariant per subnet) and matches how
    main_stochastic.py consumes warm-starts (per-subnet centering at
    main_stochastic.py:190-196).  The DGR field has zero per-subnet mean
    (softmax Jacobian rows sum to 0), so trajectories stay on this slice.
    """
    u = torch.rand(seg.shape[0], device=seg.device, generator=generator)
    z = torch.log(u.clamp(min=1e-12)) * sizes.index_select(0, seg).to(u.dtype)
    if center:
        z = segment_center(z, seg, n_seg)
    return z


# ════════════════════════════════════════════════════════════════════════
#  (t, r) sampler — iMF Tab. 4
# ════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class TimeSampler:
    """
    t, r ~ logit-normal(mu, sigma): sample two, t = max, r = min
    (paper Tab. 4: sampler logit-normal(-0.4, 1.0)); then with probability
    (1 - ratio_r_neq_t) set r = t ("ratio of r != t 50%").

    p_endpoint (our domain knob, default small): with this probability return
    exactly (t, r) = (1, 0) — the query used at 1-NFE warm-start inference.
    Set to 0.0 for a strictly paper-faithful sampler.
    """
    mu: float = -0.4
    sigma: float = 1.0
    ratio_r_neq_t: float = 0.5
    p_endpoint: float = 0.1

    def __call__(self, rng: torch.Generator, device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.p_endpoint > 0 and torch.rand((), generator=rng).item() < self.p_endpoint:
            return (torch.ones((), device=device), torch.zeros((), device=device))
        a = torch.randn(2, generator=rng) * self.sigma + self.mu
        ab = torch.sigmoid(a)
        t = ab.max().to(device)
        r = ab.min().to(device)
        if torch.rand((), generator=rng).item() >= self.ratio_r_neq_t:
            r = t.clone()
        return t, r


# ════════════════════════════════════════════════════════════════════════
#  Omega sampler — iMF Sec. 4.2 "guidance as conditioning", DGR edition
# ════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class OmegaSampler:
    """
    In iMF the CFG scale omega is sampled at training time and conditioned on,
    so a single model supports any guidance at test time (Sec. 4.2 / Alg. 2).
    DGR's analog of the guidance scale is the loss-coefficient vector

        Omega = (via_coeff, wl_coeff)          (overflow_coeff = 1 anchors scale)

    which is exactly the Pareto trade-off knob between congestion, vias, and
    wirelength.  With probability p_default we use DGR's tuned defaults
    (main_stochastic.py argparser: via=4.0, wl=0.5) — analogous to the paper
    biasing its omega distribution toward small values — otherwise we sample
    log-uniform in [default/4, default*4].
    """
    via_default: float = 4.0
    wl_default: float = 0.5
    p_default: float = 0.5
    span: float = 4.0   # log-uniform half-range multiplier

    def __call__(self, rng: torch.Generator) -> Tuple[float, float]:
        if torch.rand((), generator=rng).item() < self.p_default:
            return self.via_default, self.wl_default
        u1 = (torch.rand((), generator=rng).item() * 2 - 1)  # [-1, 1]
        u2 = (torch.rand((), generator=rng).item() * 2 - 1)
        via = self.via_default * (self.span ** u1)
        wl = self.wl_default * (self.span ** u2)
        return via, wl


def omega_to_feature(via_coeff: float, wl_coeff: float, device) -> torch.Tensor:
    """Network conditioning features: log-ratios to the tuned defaults."""
    return torch.tensor(
        [math.log(via_coeff / 4.0), math.log(wl_coeff / 0.5)],
        dtype=torch.float32, device=device)


# ════════════════════════════════════════════════════════════════════════
#  du/dt — JVP (Alg. 1) with finite-difference fallback
# ════════════════════════════════════════════════════════════════════════

def jvp_dudt(fn: Callable, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor,
             v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Alg. 1:  u, dudt = jvp(fn, (z, r, t), (v, 0, 1))
    Total derivative along the trajectory: d/dt u = dz_u . v + dt_u   (Eq. 5)
    Returns (u, dudt); dudt is NOT detached here (caller applies stop-grad,
    see "About Stop-gradient" in iMF Sec. 4.1).
    """
    u, dudt = torch.func.jvp(
        fn, (z, r, t),
        (v, torch.zeros_like(r), torch.ones_like(t)))
    return u, dudt


def fd_dudt(fn: Callable, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor,
            v: torch.Tensor, eps: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Central finite-difference fallback for d/dt u along the trajectory
    direction (used when some op in u_theta lacks a forward-AD rule):
        dudt ≈ [u(z + eps*v, r, t + eps) - u(z - eps*v, r, t - eps)] / (2 eps)
    The two perturbed evaluations are detached (they only feed the
    stop-gradded dudt term), so this costs 2 extra no-grad forwards.
    """
    u = fn(z, r, t)
    with torch.no_grad():
        up = fn(z + eps * v, r, t + eps)
        um = fn(z - eps * v, r, t - eps)
        dudt = (up - um) / (2.0 * eps)
    return u, dudt


# ════════════════════════════════════════════════════════════════════════
#  Loss metric — MF/iMF adaptive weighting, at per-subnet granularity
# ════════════════════════════════════════════════════════════════════════

def adaptive_subnet_loss(err: torch.Tensor, seg: torch.Tensor, n_seg: int,
                         p: float = 1.0, c: float = 1e-3,
                         mode: str = "adaptive") -> torch.Tensor:
    """
    The "metric(error)" of Alg. 1.  In MeanFlow [12] the squared error of each
    sample is reweighted by  w = 1 / (||err||^2 + c)^p  with stop-gradient on
    w (adaptive l2).  Our batch of "samples" is the set of subnets of one
    graph: err is the per-candidate residual (V_theta - v), grouped by subnet.

    mode="l2" gives the plain mean-squared error (used for the paper's Fig. 3
    style loss-curve comparisons).
    """
    e2 = segment_sum(err * err, seg, n_seg)                  # [n_seg]
    if mode == "l2":
        return e2.mean()
    w = (e2.detach() + c).pow(-p)
    return (w * e2).mean()


# ════════════════════════════════════════════════════════════════════════
#  The iMF training objective — Eq. (12) / Alg. 1
# ════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class MeanFlowConfig:
    dudt_mode: str = "jvp"        # jvp | fd   (fd = finite difference)
    fd_eps: float = 1e-3
    loss_mode: str = "adaptive"   # adaptive | l2
    adp_p: float = 1.0
    adp_c: float = 1e-3


def meanflow_step(fn: Callable,
                  z_t: torch.Tensor,
                  v_t: torch.Tensor,
                  r: torch.Tensor,
                  t: torch.Tensor,
                  seg: torch.Tensor,
                  n_seg: int,
                  cfg: MeanFlowConfig) -> Tuple[torch.Tensor, dict]:
    """
    One iMF loss evaluation (no optimizer step) — the heart of Alg. 1:

        v   = exact field at z_t  (passed in as v_t — our setting has the true
              marginal; no conditional-velocity substitution is needed)
        u, dudt = jvp(fn, (z, r, t), (v, 0, 1))
        V   = u + (t - r) * stopgrad(dudt)          # Eq. (12), JVP_sg
        loss = metric(V - v)                         # v-loss, target indep. of theta

    When r == t the (t - r) factor vanishes and this reduces to pure flow
    matching on the instantaneous velocity (u_theta's boundary condition,
    iMF Sec. 4.1) — the JVP is skipped entirely.

    fn must be a pure function (z, r, t) -> u of the SAME shapes; graph,
    Omega-conditioning, and parameters are captured in its closure.
    """
    z_t = z_t.detach()
    v_t = v_t.detach()
    r_neq_t = bool((t - r).abs().item() > 1e-8)

    if not r_neq_t:
        u = fn(z_t, r, t)
        V = u
        dudt_norm = 0.0
    else:
        if cfg.dudt_mode == "jvp":
            u, dudt = jvp_dudt(fn, z_t, r, t, v_t)
        else:
            u, dudt = fd_dudt(fn, z_t, r, t, v_t, eps=cfg.fd_eps)
        dudt = dudt.detach()                      # JVP_sg — iMF Eq. (9)/(12)
        V = u + (t - r) * dudt
        dudt_norm = float(dudt.norm().item())

    err = V - v_t
    loss = adaptive_subnet_loss(err, seg, n_seg,
                                p=cfg.adp_p, c=cfg.adp_c, mode=cfg.loss_mode)
    logs = {
        "loss": float(loss.item()),
        "raw_mse": float((err * err).mean().item()),
        "u_norm": float(u.detach().norm().item()),
        "v_norm": float(v_t.norm().item()),
        "dudt_norm": dudt_norm,
        "t": float(t.item()),
        "r": float(r.item()),
        "r_neq_t": int(r_neq_t),
    }
    return loss, logs


# ════════════════════════════════════════════════════════════════════════
#  Sinusoidal features for scalar conditions (t, t-r, ...) — Sec. 4.3
# ════════════════════════════════════════════════════════════════════════

def fourier_features(x: torch.Tensor, n_freq: int = 8,
                     max_period: float = 4.0) -> torch.Tensor:
    """
    Standard positional embedding of a scalar condition (iMF appendix:
    "All continuous-valued conditions are processed by standard positional
    embedding").  x: scalar 0-dim tensor -> [2 * n_freq] feature vector.
    Built from x with differentiable ops only, so forward-mode AD propagates
    the t-tangent (0, ..., 1) of Alg. 1 through the conditioning pathway.
    """
    freqs = torch.exp(
        torch.linspace(math.log(1.0), math.log(2.0 ** n_freq), n_freq,
                       device=x.device)) * (2.0 * math.pi / max_period)
    ang = x.reshape(1) * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=0)
