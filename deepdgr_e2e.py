#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepDGR End-to-End Training
============================
The TRUE e2e mode.  GNN output LITERALLY replaces net.p inside DGR.

From your notes (Feb 17):
  "Now in DGR:
     self.p = torch.nn.Parameter(torch.cat(p_list), requires_grad=True)
   What we can do: replace self.p as output of GNN"

How this file does it
---------------------
  1. Build/load the hetero graph exactly like deepdgr_graph_from_dgr.py.
  2. Construct a GNN (same architecture as deepdgr_gnn_v2).
  3. Monkey-patch model.Net so that net.forward() calls GNN(graph) instead
     of gumbel_softmax(self.p).
  4. Run DGR's normal optimizer loop — net.parameters() now includes GNN
     weights, so every optimizer.step() updates the GNN.
  5. Gradient path:
       DGR objective_function  →  total_cost.backward()
       → gumbel_softmax(GNN_output)
       → GNN weights
  6. After training, export the GNN as a warm-start generator.

Why this matters
----------------
  In vanilla DGR, self.p starts random and takes 2000 iterations to converge.
  With GNN as p-generator, we hope the GNN learns the structure of the problem
  well enough to provide near-optimal initialization for every new benchmark
  via forward pass (no iterations needed).

Usage
-----
  cd /ocean/projects/cis260079p/ctsai4/Differentiable-Global-Router
  python3 deepdgr_e2e.py \\
      --data_path cu-gr-2/run/ispd18_test1_metal5.pt \\
      --graph_path ispd18_test1_graph.pt \\
      --dgr_iter 500 \\
      --gnn_hidden 128 --gnn_layers 4 \\
      --lr 0.01 --device cuda:0 \\
      --save_gnn ispd18_test1_e2e_gnn.pth \\
      --export_warmstart ispd18_test1_e2e_warmstart.npz
"""

import os
import sys
import argparse
import time
import math
import gc
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from torch_scatter import scatter_softmax, scatter_sum


# ============================================================================
# GNN (same as v2, kept self-contained so this file runs standalone)
# ============================================================================

class DeepDGR_GNN(nn.Module):
    """
    Matches HeteroGNN from deepdgr_gnn_multi.py exactly.
    Uses lazy input_proj, no LayerNorm, simple Linear output head.
    """
    def __init__(self, grid_in=4, cand_in=4, hidden=64, num_layers=3, dropout=0.1):
        super().__init__()
        self._hidden = hidden
        self._num_layers = num_layers
        self.dropout = dropout  # kept for interface compat but not used in forward

        # Input projections — built eagerly with known dims
        self.input_proj = nn.ModuleDict({
            'grid': nn.Linear(grid_in, hidden),
            'candidate': nn.Linear(cand_in, hidden),
        })

        # Convolutions (same structure as HeteroGNN)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ('grid',      'connects',       'grid'):      SAGEConv((-1, -1), hidden),
                ('candidate', 'passes_through', 'grid'):      SAGEConv((-1, -1), hidden),
                ('grid',      'influences',     'candidate'): SAGEConv((-1, -1), hidden),
                ('candidate', 'competes',       'candidate'): SAGEConv((-1, -1), hidden),
            }
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        # Output head (single linear, matching training script)
        self.out = Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):
        h = {nt: F.relu(self.input_proj[nt](x.float()))
             for nt, x in x_dict.items() if nt in self.input_proj}
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {nt: F.relu(feat) for nt, feat in h.items()}
        return self.out(h['candidate']).squeeze(-1)



# ============================================================================
# Patched Net: wraps DGR's Net so forward() calls GNN
# ============================================================================

class GNNBackedNet(nn.Module):
    """
    Drop-in replacement for model.Net whose self.p is produced by GNN.

    DGR's training loop calls:
        p, candidate_p, tree_p = net.forward(temperature, tree_temperature)

    We intercept this: instead of gumbel_softmax(self.p), we:
        1. Run GNN to get logits  (shape: [total_candidates])
        2. Apply gumbel_softmax(logits)  → p   (same interface as original Net)
        3. candidate_p = p (already per-candidate), tree_p = None

    Gradients from DGR's total_cost.backward() flow:
        total_cost → p → gumbel_softmax → GNN logits → GNN.parameters()
    """

    def __init__(self, original_net, gnn, x_dict, edge_index_dict,
                 p_index_full, device):
        super().__init__()
        self.original_net = original_net    # keep for tree_p handling
        self.gnn = gnn
        self.x_dict = x_dict
        self.edge_index_dict = edge_index_dict
        self.p_index_full = p_index_full
        self.device = device

        # Expose p as a property for DGR code that accesses net.p directly
        # (e.g., the add_CZ branch).  We register a buffer so shape checks pass.
        n = p_index_full.shape[0]
        self.register_buffer('_p_shape_ref', torch.zeros(n))

    @property
    def p(self):
        """Return current GNN output as logits (same interface as nn.Parameter)."""
        return self._last_logits if hasattr(self, '_last_logits') else self._p_shape_ref

    @p.setter
    def p(self, val):
        # DGR sometimes does: net.p = torch.nn.Parameter(new_p, ...)
        # We silently ignore these reassignments — GNN owns p now.
        pass

    def parameters(self, recurse=True):
        # Only GNN parameters are trainable; original_net.p is replaced
        return self.gnn.parameters(recurse=recurse)

    def forward(self, temperature, tree_temperature):
        """
        Called as: p, candidate_p, tree_p = net.forward(T, T_tree)

        Returns:
            p           — [total_candidates] gumbel-softmax probabilities
            candidate_p — same as p (DGR uses this for writing routing results)
            tree_p      — None  (we only support single-tree mode here)
        """
        # --- GNN forward pass ---
        logits = self.gnn(self.x_dict, self.edge_index_dict)   # [total_cand]
        self._last_logits = logits

        # --- Gumbel-softmax per subnet (same as DGR's Net.forward) ---
        # Add Gumbel noise for exploration (temperature annealing as in DGR)
        gumbel_noise = -torch.log(-torch.log(
            torch.rand_like(logits).clamp(min=1e-10)
        )).clamp(-10, 10)
        noisy_logits = (logits + gumbel_noise) / max(temperature, 0.05)
        p = scatter_softmax(noisy_logits, self.p_index_full)

        return p, p, None   # (p, candidate_p, tree_p)


# ============================================================================
# Main training
# ============================================================================

def run_e2e(args):
    device_str = args.device
    device = torch.device(device_str)

    # ------------------------------------------------------------------
    # 1. Load DGR modules (must be in the DGR source directory)
    # ------------------------------------------------------------------
    print(f"[E2E] DGR dir: {args.dgr_dir}")
    sys.path.insert(0, os.path.abspath(args.dgr_dir))
    import data as dgr_data_mod
    import util as dgr_util
    import model as dgr_model

    # ------------------------------------------------------------------
    # 2. Load .pt data (identical to main_stochastic.py lines 110-148)
    # ------------------------------------------------------------------
    print(f"[E2E] Loading data: {args.data_path}")
    results = torch.load(args.data_path, map_location='cpu')
    RouteNets = results['net']
    RoutingRegion = results['region']
    RoutingRegion3D = results['region3D']

    xmax = RoutingRegion.xmax
    ymax = RoutingRegion.ymax
    num_layer = (len(RoutingRegion3D.cap_mat_3D[0]) +
                 len(RoutingRegion3D.cap_mat_3D[1]))
    hor_first = RoutingRegion3D.hor_first

    # Reconstruct capacity (same as main_stochastic.py)
    RoutingRegion.cap_mat = [
        torch.stack(RoutingRegion3D.cap_mat_3D[0]).sum(0),
        torch.stack(RoutingRegion3D.cap_mat_3D[1]).sum(0)
    ]
    RoutingRegion.cap_mat = [c * args.capacity for c in RoutingRegion.cap_mat]
    results['edge_length'] = [results['edge_length'][1], results['edge_length'][0]]

    class _Args: pass
    dgr_args = _Args()
    dgr_args.xmax, dgr_args.ymax = xmax, ymax
    dgr_args.net_num = len(RouteNets)
    dgr_args.num_layer, dgr_args.hor_first = num_layer, hor_first
    dgr_args.num_hor_layer = len(RoutingRegion3D.cap_mat_3D[0])
    dgr_args.num_ver_layer = len(RoutingRegion3D.cap_mat_3D[1])
    dgr_args.via_layer = float(np.sqrt(num_layer)) * args.via_layer
    dgr_args.device = device_str
    dgr_args.local_net_ratio = 1.0
    dgr_args.pin_ratio = 1.0
    dgr_args.use_ilp_metric = False
    dgr_args.add_via = True
    dgr_args.pattern_level = args.pattern_level
    dgr_args.read_new_tree = False
    dgr_args.add_CZ = False
    dgr_args.use_gumble = True
    dgr_args.select_threshold = 1.0
    dgr_args.via_coeff = 4.0
    dgr_args.wl_coeff = 0.5
    dgr_args.overflow_coeff = 1.0
    dgr_args.act = 'sigmoid'
    dgr_args.act_scale = 0.5
    dgr_args.epoch_iter = 1
    dgr_args.celu_alpha = 2.0

    hor_ed, ver_ed, hor_pd, ver_pd = dgr_util.get_pin_demand(
        RouteNets, dgr_args, results['layers'], results['edge_length'])
    hor_ld, ver_ld = dgr_util.get_local_net(RouteNets, dgr_args)
    RoutingRegion.cap_mat[0] -= hor_ld
    RoutingRegion.cap_mat[1] -= ver_ld
    RoutingRegion.cap_mat[0] -= torch.tensor(hor_ed)
    RoutingRegion.cap_mat[1] -= torch.tensor(ver_ed)

    m2_pitch = results['m2_pitch']
    min_ulsc = min(l['unit_length_short_cost'] for l in results['layers'])

    RoutingRegion.to(device)
    hor_pin_demand = torch.tensor(hor_pd).to(device)
    ver_pin_demand = torch.tensor(ver_pd).to(device)
    hor_edge_length = torch.tensor(results['edge_length'][0]).to(device).reshape(1, -1)
    ver_edge_length = torch.tensor(results['edge_length'][1]).to(device).reshape(-1, 1)

    print(f"[E2E] Grid: {xmax}x{ymax}  |  Nets: {len(RouteNets):,}  |  Layers: {num_layer}")

    # ------------------------------------------------------------------
    # 3. Candidate pool (same as main_stochastic.py lines 154-170)
    # ------------------------------------------------------------------
    data_name = os.path.basename(args.data_path).replace('.pt', '')
    os.makedirs('./tmp', exist_ok=True)
    cache_cand = f'./tmp/{data_name}_candidate_pool.pt'
    cache_pidx = f'./tmp/{data_name}_p_index.pt'

    if os.path.exists(cache_cand):
        candidate_pool = torch.load(cache_cand)
        print("[E2E] Candidate pool loaded from cache.")
    else:
        candidate_pool = dgr_util.get_initial_candidate_pool(
            RouteNets, xmax, ymax, device=device_str,
            edge_length=results['edge_length'],
            pattern_level=args.pattern_level, max_z=10, z_step=3,
            c_step=3, max_c=20, max_c_out_ratio=5)
        torch.save(candidate_pool, cache_cand)

    if os.path.exists(cache_pidx):
        (p_index, p_index_full, p_index2pattern_index,
         hor_path, ver_path, wire_length_count, via_info,
         tree_p_index, tree_index_per_candidate,
         tree_p_index2pattern_index, tree_p_index_full) = \
            torch.load(cache_pidx, map_location=device_str)
        print("[E2E] p_index loaded from cache.")
    else:
        (p_index, p_index_full, p_index2pattern_index,
         hor_path, ver_path, wire_length_count, via_info,
         tree_p_index, tree_index_per_candidate,
         tree_p_index2pattern_index, tree_p_index_full) = \
            dgr_data_mod.process_pool(candidate_pool, xmax, ymax, device=device_str)
        torch.save((p_index, p_index_full, p_index2pattern_index,
                    hor_path, ver_path, wire_length_count, via_info,
                    tree_p_index, tree_index_per_candidate,
                    tree_p_index2pattern_index, tree_p_index_full), cache_pidx)

    tree_p_index = None   # single-tree mode
    total_cand = p_index[-1]
    num_subnets = len(p_index) - 1
    print(f"[E2E] Candidates: {total_cand:,}  |  Subnets: {num_subnets:,}")

    # ------------------------------------------------------------------
    # 4. Load hetero graph for GNN
    # ------------------------------------------------------------------
    print(f"[E2E] Loading graph: {args.graph_path}")
    gd = torch.load(args.graph_path, map_location='cpu')
    graph = gd['graph']

    if ('grid', 'influences', 'candidate') not in graph.edge_types:
        print("[E2E] ERROR: Graph missing ('grid','influences','candidate') edges.")
        print("  Rebuild with: python3 deepdgr_graph_from_dgr.py ...")
        sys.exit(1)

    if gd['total_candidates'] != total_cand:
        print(f"[E2E] ERROR: Graph has {gd['total_candidates']} candidates "
              f"but data has {total_cand}. Rebuild graph.")
        sys.exit(1)

    x_dict = {k: graph[k].x.to(device) for k in ['grid', 'candidate']}
    edge_index_dict = {et: graph[et].edge_index.to(device) for et in graph.edge_types}
    p_index_full_dev = p_index_full.to(device) if not isinstance(p_index_full, torch.Tensor) \
        else p_index_full.to(device)

    # ------------------------------------------------------------------
    # 5. Build GNN and wrap DGR's Net
    # ------------------------------------------------------------------
    gnn = DeepDGR_GNN(
        grid_in=graph['grid'].x.shape[1],
        cand_in=graph['candidate'].x.shape[1],
        hidden=args.gnn_hidden,
        num_layers=args.gnn_layers,
        dropout=args.gnn_dropout
    ).to(device)

    if args.load_gnn and os.path.exists(args.load_gnn):
        ckpt = torch.load(args.load_gnn, map_location=device)
        if 'gnn_state_dict' in ckpt:
            state_dict = ckpt['gnn_state_dict']
        elif 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            raise KeyError(f"Checkpoint has no known key. Keys: {list(ckpt.keys())}")
        # Map keys if needed (training script may use slightly different names)
        try:
            gnn.load_state_dict(state_dict, strict=False)
            print(f"[E2E] Loaded GNN weights from {args.load_gnn}")
        except RuntimeError as e:
            print(f"[E2E] WARNING: strict load failed: {e}")
            print("[E2E] Trying key remapping...")
            # Remap from HeteroGNN (training) to DeepDGR_GNN (e2e)
            new_sd = {}
            model_keys = set(gnn.state_dict().keys())
            for k, v in state_dict.items():
                if k in model_keys:
                    new_sd[k] = v
                else:
                    print(f"  SKIP key: {k}")
            missing = model_keys - set(new_sd.keys())
            if missing:
                print(f"  Missing keys (will use random init): {missing}")
            gnn.load_state_dict(new_sd, strict=False)
            print(f"[E2E] Loaded GNN weights (partial) from {args.load_gnn}")

    n_params = sum(p.numel() for p in gnn.parameters())
    print(f"[E2E] GNN params: {n_params:,}  |  hidden={args.gnn_hidden}  layers={args.gnn_layers}")

    # Create the original DGR Net (initialises self.p the normal way)
    original_net = dgr_model.Net(
        p_index, pattern_level=args.pattern_level, device=device_str,
        use_gumble=True, tree_p_index=None,
        tree_index_per_candidate=tree_index_per_candidate
    ).to(device)

    # Wrap it with GNN backend
    net = GNNBackedNet(original_net, gnn, x_dict, edge_index_dict,
                       p_index_full_dev, device)

    # ------------------------------------------------------------------
    # 6. Optimizer — only GNN parameters
    # ------------------------------------------------------------------
    optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # Learning rate: linear warmup then cosine decay (same strategy as v2)
    warmup = min(50, args.dgr_iter // 10)
    def lr_sched(it):
        if it < warmup:
            return (it + 1) / max(warmup, 1)
        prog = (it - warmup) / max(args.dgr_iter - warmup, 1)
        return 0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_sched)

    # DGR temperature schedule (same as main_stochastic.py)
    temperature = 1.0
    tree_temperature = 1.0
    t_scale = args.t_scale          # default 1.0 → same as DGR's default t=1
    tree_t_scale = args.tree_t_scale
    update_interval = max(1, args.dgr_iter // 10)

    hor_cap = RoutingRegion.cap_mat[0].flatten()
    ver_cap = RoutingRegion.cap_mat[1].flatten()
    import math as _math
    hor_batch = _math.ceil(hor_cap.numel() / dgr_args.epoch_iter)
    ver_batch = _math.ceil(ver_cap.numel() / dgr_args.epoch_iter)

    # CSV log
    csv_file = None
    if args.log_csv:
        csv_file = open(args.log_csv, 'w', newline='')
        cw = csv.DictWriter(csv_file, fieldnames=[
            'iter', 'total_cost', 'overflow_cost', 'via_cost', 'wl_cost',
            'max_overflow', 'grad_norm', 'lr', 'temperature', 'time_s'])
        cw.writeheader()

    print(f"\n[E2E] === TRAINING: {args.dgr_iter} DGR iterations ===")
    print("  Gradient path: DGR_objective → gumbel_softmax(GNN_logits) → GNN_weights")
    print("="*70)

    best_cost = float('inf')
    best_state = None
    cost_history = []

    for i in range(args.dgr_iter):
        t0 = time.time()

        # Temperature annealing (identical to main_stochastic.py)
        if i % update_interval == 0 and i != 0:
            temperature *= t_scale
            tree_temperature *= tree_t_scale

        hor_shuffled = torch.randperm(hor_cap.numel())
        ver_shuffled = torch.randperm(ver_cap.numel())

        # ---- Forward: GNN → p ----
        p, candidate_p, _ = net.forward(temperature, tree_temperature)

        # ---- DGR objective (Eq 3-6 in paper) ----
        (overflow_cost, via_cost, wire_length_cost,
         max_overflow, hor_overflow, ver_overflow) = \
            dgr_model.objective_function(
                RoutingRegion, hor_path, ver_path, wire_length_count,
                via_info, p, dgr_args,
                hor_pin_demand, ver_pin_demand,
                hor_edge_length, ver_edge_length,
                min_ulsc, m2_pitch,
                iteration=0,
                hor_batch_size=hor_batch, ver_batch_size=ver_batch,
                hor_shuffled_indices=hor_shuffled,
                ver_shuffled_indices=ver_shuffled
            )

        total_cost = (overflow_cost * dgr_args.overflow_coeff
                      + wire_length_cost * dgr_args.wl_coeff
                      + via_cost * dgr_args.via_coeff)

        # ---- Backward: DGR → GNN ----
        optimizer.zero_grad()
        total_cost.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(gnn.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()

        cost_val = total_cost.item()
        cost_history.append(cost_val)

        if cost_val < best_cost:
            best_cost = cost_val
            best_state = {k: v.clone() for k, v in gnn.state_dict().items()}

        elapsed = time.time() - t0

        if i % args.log_every == 0 or i == args.dgr_iter - 1:
            print(f"  iter {i:4d} | cost={cost_val:.4f}  "
                  f"OF={overflow_cost.item():.3f}  "
                  f"via={via_cost.item():.3f}  wl={wire_length_cost.item():.3f}  "
                  f"maxOF={max_overflow.item():.1f}  "
                  f"T={temperature:.3f}  gnorm={grad_norm:.3f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.5f}  {elapsed:.1f}s")

        if csv_file:
            cw.writerow({
                'iter': i, 'total_cost': cost_val,
                'overflow_cost': overflow_cost.item(),
                'via_cost': via_cost.item(), 'wl_cost': wire_length_cost.item(),
                'max_overflow': max_overflow.item(),
                'grad_norm': grad_norm.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'temperature': temperature, 'time_s': elapsed
            })
            if i % 50 == 0:
                csv_file.flush()

        if i % 50 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if csv_file:
        csv_file.close()

    print("="*70)
    print(f"\n[E2E] Best cost: {best_cost:.4f}")
    if len(cost_history) >= 20:
        print(f"[E2E] Convergence: first-20 avg={np.mean(cost_history[:20]):.4f}  "
              f"→  last-20 avg={np.mean(cost_history[-20:]):.4f}")

    # ------------------------------------------------------------------
    # 7. Save GNN
    # ------------------------------------------------------------------
    if args.save_gnn:
        save_dict = {
            'gnn_state_dict': best_state if best_state else gnn.state_dict(),
            'args': vars(args),
            'total_candidates': total_cand,
            'num_subnets': num_subnets,
        }
        torch.save(save_dict, args.save_gnn)
        print(f"[E2E] GNN saved: {args.save_gnn}")

    # ------------------------------------------------------------------
    # 8. Export warm-start probabilities
    # ------------------------------------------------------------------
    if args.export_warmstart:
        gnn.eval()
        if best_state:
            gnn.load_state_dict(best_state)
        with torch.no_grad():
            logits = gnn(x_dict, edge_index_dict).cpu().numpy()

        probs = np.zeros_like(logits)
        pi = p_index if not isinstance(p_index, torch.Tensor) else p_index.tolist()
        for s in range(len(pi) - 1):
            st, en = pi[s], pi[s+1]
            if en > st:
                c = logits[st:en]
                e = np.exp(c - c.max())
                probs[st:en] = e / e.sum()
        np.savez(args.export_warmstart, logits=logits, probabilities=probs,
                 p_index=np.array(pi))
        print(f"[E2E] Warm-start saved: {args.export_warmstart}")

    return gnn


def main():
    p = argparse.ArgumentParser(description="DeepDGR E2E: GNN replaces net.p in DGR")
    # Data
    p.add_argument('--data_path',   type=str, required=True, help='.pt benchmark file')
    p.add_argument('--graph_path',  type=str, required=True, help='hetero graph .pt file')
    p.add_argument('--dgr_dir',     type=str,
                   default=os.path.dirname(os.path.abspath(__file__)),
                   help='DGR source root (contains model.py, util.py, data.py)')
    # GNN
    p.add_argument('--gnn_hidden',  type=int,   default=64)
    p.add_argument('--gnn_layers',  type=int,   default=3)
    p.add_argument('--gnn_dropout', type=float, default=0.1)
    p.add_argument('--load_gnn',    type=str,   default=None,
                   help='Path to pre-trained GNN (e.g. from supervised training)')
    p.add_argument('--save_gnn',    type=str,   default=None)
    p.add_argument('--export_warmstart', type=str, default=None)
    # Training
    p.add_argument('--dgr_iter',    type=int,   default=500,
                   help='DGR iterations (vs. default 2000; GNN needs fewer)')
    p.add_argument('--lr',          type=float, default=0.01)
    p.add_argument('--weight_decay',type=float, default=1e-5)
    p.add_argument('--t_scale',     type=float, default=1.0,
                   help='Temperature multiplier per update_interval (DGR default: 1.0)')
    p.add_argument('--tree_t_scale',type=float, default=0.9)
    p.add_argument('--capacity',    type=float, default=1.0)
    p.add_argument('--via_layer',   type=float, default=1.5)
    p.add_argument('--pattern_level',type=int,  default=1)
    # Misc
    p.add_argument('--device',   type=str, default='cuda:0')
    p.add_argument('--log_every',type=int, default=10)
    p.add_argument('--log_csv',  type=str, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available() and 'cuda' in args.device:
        print("[E2E] No CUDA, using CPU")
        args.device = 'cpu'

    run_e2e(args)


if __name__ == '__main__':
    main()