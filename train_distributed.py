#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_distributed_original.py
==============================
Multi-GPU training using the ORIGINAL DeepDGR_GNN architecture.

GNN is IDENTICAL to deepdgr_e2e.py / deepdgr_gnn_multi.py:
  - SAGEConv-based HeteroConv
  - Same parameter names, same layer structure
  - No torch.compile, no gradient checkpointing
  - Checkpoint-compatible with all existing .pth files

Infrastructure uses standard PyTorch (not GNN modifications):
  - DistributedDataParallel across available GPUs
  - Automatic Mixed Precision (float16)
  - Out-of-core data loading (1 graph at a time from disk)
  - Gradient accumulation for effective large batch size

USAGE:
  cd /ocean/projects/cis260079p/ctsai4/Differentiable-Global-Router

  # Single GPU test
  python3 train_distributed_original.py \
      --data_root /ocean/projects/cis260079p/ctsai4/compact \
      --target ispd18_test5_metal5 \
      --hidden 64 --layers 3 --epochs 100 \
      --no_distributed

  # 8 GPU training
  torchrun --nproc_per_node=8 train_distributed_original.py \
      --data_root /ocean/projects/cis260079p/ctsai4/compact \
      --target ispd18_test5_metal5 \
      --hidden 64 --layers 3 --epochs 5000 --lr 3e-3 \
      --output_dir experiments/dist_18_test5
"""

import argparse
import gc
import glob
import math
import os
import sys
import time

print("[train] Loading torch...", flush=True)
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch_geometric.nn import SAGEConv, HeteroConv, Linear


# ════════════════════════════════════════════════════════════════════════
#  ORIGINAL DeepDGR_GNN — copied verbatim from deepdgr_e2e.py
#  DO NOT MODIFY — must match existing checkpoints exactly
# ════════════════════════════════════════════════════════════════════════

class DeepDGR_GNN(nn.Module):
    """
    Matches HeteroGNN from deepdgr_gnn_multi.py exactly.
    Uses lazy input_proj, no LayerNorm, simple Linear output head.
    """
    def __init__(self, grid_in=4, cand_in=4, hidden=64, num_layers=3, dropout=0.1):
        super().__init__()
        self._hidden = hidden
        self._num_layers = num_layers
        self.dropout = dropout

        # Input projections
        self.input_proj = nn.ModuleDict({
            'grid': nn.Linear(grid_in, hidden),
            'candidate': nn.Linear(cand_in, hidden),
        })

        # Convolutions
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                ('grid',      'connects',       'grid'):      SAGEConv((-1, -1), hidden),
                ('candidate', 'passes_through', 'grid'):      SAGEConv((-1, -1), hidden),
                ('grid',      'influences',     'candidate'): SAGEConv((-1, -1), hidden),
                ('candidate', 'competes',       'candidate'): SAGEConv((-1, -1), hidden),
            }
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        # Output head
        self.out = Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):
        h = {nt: F.relu(self.input_proj[nt](x.float()))
             for nt, x in x_dict.items() if nt in self.input_proj}
        for conv in self.convs:
            h = conv(h, edge_index_dict)
            h = {nt: F.relu(feat) for nt, feat in h.items()}
        return self.out(h['candidate']).squeeze(-1)


# ════════════════════════════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════════════════════════════

ALL_BENCHES = [
    "ispd18_test5_metal5",
    "ispd18_test8_metal5",
    "ispd18_test10_metal5",
    "ispd19_test7_metal5",
    "ispd19_test8_metal5",
    "ispd19_test9_metal5",
]

EDGE_TYPES = [
    ("grid", "connects", "grid"),
    ("candidate", "passes_through", "grid"),
    ("grid", "influences", "candidate"),
    ("candidate", "competes", "candidate"),
]


def log(msg, rank=0):
    if rank == 0:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  Dataset — lazy loading from compact .pt files
# ════════════════════════════════════════════════════════════════════════

class CompactGraphDataset(Dataset):
    """
    Loads one compact .pt file per __getitem__ call.
    Memory: O(1) — only one graph in RAM at a time per worker.
    """
    def __init__(self, file_paths, preload_edges=True):
        self.file_paths = sorted(file_paths)
        self._shared_edges = None
        if preload_edges and len(file_paths) > 0:
            try:
                first = torch.load(file_paths[0], map_location="cpu",
                                   weights_only=False)
                self._shared_edges = first.get("edges", None)
            except Exception:
                pass

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], map_location="cpu",
                          weights_only=False)
        edges = data.get("edges", self._shared_edges)
        edge_index_dict = {}
        for et, ei in edges.items():
            edge_index_dict[et] = ei.long()

        p_index = data.get("p_index", None)
        if p_index is not None:
            p_index = p_index.long()

        return {
            "grid_x": data["grid_x"].float(),
            "candidate_x": data["candidate_x"].float(),
            "edge_index_dict": edge_index_dict,
            "target": data["target"].float(),
            "p_index": p_index,
        }


def collate_single(batch):
    """One graph per batch — no batching across graphs."""
    assert len(batch) == 1
    return batch[0]


# ════════════════════════════════════════════════════════════════════════
#  Loss
# ════════════════════════════════════════════════════════════════════════

def subnet_kl_loss(logits, target_probs, p_index):
    """KL divergence per subnet, same as deepdgr_gnn_multi.py."""
    total_loss = 0.0
    n = 0
    for s in range(len(p_index) - 1):
        st, en = p_index[s].item(), p_index[s + 1].item()
        if en <= st:
            continue
        pred = F.log_softmax(logits[st:en], dim=0)
        tgt = target_probs[st:en].clamp(min=1e-8)
        tgt = tgt / tgt.sum()
        total_loss += F.kl_div(pred, tgt, reduction="sum")
        n += 1
    return total_loss / max(n, 1)


def mse_loss(logits, target):
    return F.mse_loss(logits, target)


# ════════════════════════════════════════════════════════════════════════
#  Training loop
# ════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, optimizer, scheduler, scaler,
                     device, grad_accum_steps=1, use_amp=True,
                     loss_fn="kl", rank=0):
    model.train()
    total_loss = 0.0
    n_samples = 0
    t0 = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    optimizer.zero_grad(set_to_none=True)

    for step, sample in enumerate(dataloader):
        x_dict = {
            "grid": sample["grid_x"].to(device, non_blocking=True),
            "candidate": sample["candidate_x"].to(device, non_blocking=True),
        }
        edge_index_dict = {
            et: ei.to(device, non_blocking=True)
            for et, ei in sample["edge_index_dict"].items()
        }
        target = sample["target"].to(device, non_blocking=True)
        p_index = sample.get("p_index")
        if p_index is not None:
            p_index = p_index.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x_dict, edge_index_dict)
            if loss_fn == "kl" and p_index is not None:
                loss = subnet_kl_loss(logits, target, p_index)
            else:
                loss = mse_loss(logits, target)
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * grad_accum_steps
        n_samples += 1

        del x_dict, edge_index_dict, target, logits, loss
        if step % 20 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    elapsed = time.time() - t0
    peak_mem = 0
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "avg_loss": total_loss / max(n_samples, 1),
        "n_samples": n_samples,
        "elapsed_s": elapsed,
        "peak_memory_mb": peak_mem,
    }


@torch.no_grad()
def validate(model, dataloader, device, use_amp=True, loss_fn="kl"):
    model.eval()
    total_loss = 0.0
    n = 0
    for sample in dataloader:
        x_dict = {
            "grid": sample["grid_x"].to(device, non_blocking=True),
            "candidate": sample["candidate_x"].to(device, non_blocking=True),
        }
        edge_index_dict = {
            et: ei.to(device, non_blocking=True)
            for et, ei in sample["edge_index_dict"].items()
        }
        target = sample["target"].to(device, non_blocking=True)
        p_index = sample.get("p_index")
        if p_index is not None:
            p_index = p_index.to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x_dict, edge_index_dict)
            if loss_fn == "kl" and p_index is not None:
                loss = subnet_kl_loss(logits, target, p_index)
            else:
                loss = mse_loss(logits, target)

        total_loss += loss.item()
        n += 1
        del x_dict, edge_index_dict, target, logits, loss

    return total_loss / max(n, 1)


# ════════════════════════════════════════════════════════════════════════
#  Warmstart export — same format as deepdgr_gnn_multi.py
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def export_warmstart(model, test_sample, output_path, device):
    model.eval()
    x_dict = {
        "grid": test_sample["grid_x"].to(device),
        "candidate": test_sample["candidate_x"].to(device),
    }
    edge_index_dict = {
        et: ei.to(device)
        for et, ei in test_sample["edge_index_dict"].items()
    }
    logits = model(x_dict, edge_index_dict).cpu().numpy()

    p_index = test_sample.get("p_index")
    probs = np.zeros_like(logits)
    if p_index is not None:
        pi = p_index.numpy()
        for s in range(len(pi) - 1):
            st, en = pi[s], pi[s + 1]
            if en > st:
                c = logits[st:en]
                e = np.exp(c - c.max())
                probs[st:en] = e / e.sum()
    else:
        e = np.exp(logits - logits.max())
        probs = e / e.sum()

    save_dict = {"logits": logits, "probabilities": probs}
    if p_index is not None:
        save_dict["p_index"] = p_index.numpy()
    np.savez(output_path, **save_dict)
    return output_path


# ════════════════════════════════════════════════════════════════════════
#  File discovery
# ════════════════════════════════════════════════════════════════════════

def discover_files(data_root, benchmark, max_files=None):
    bench_dir = os.path.join(data_root, benchmark)
    files = []
    for pat in ["*_compact.pt", "*.compact.pt", "compact_*.pt", "*.pt"]:
        files.extend(glob.glob(os.path.join(bench_dir, pat)))
    files = sorted(set(files))
    if max_files is not None:
        files = files[:max_files]
    return files


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Distributed training with ORIGINAL DeepDGR_GNN")

    # Data
    parser.add_argument("--data_root", required=True,
                        help="Root dir with benchmark subdirs of compact .pt files")
    parser.add_argument("--benchmarks", nargs="+", default=None,
                        help="Train benchmarks (default: all except target)")
    parser.add_argument("--target", required=True,
                        help="Target benchmark for validation + warmstart")
    parser.add_argument("--max_files_per_bench", type=int, default=None)

    # GNN (must match existing models)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--grid_in", type=int, default=4)
    parser.add_argument("--cand_in", type=int, default=4)

    # Training
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--loss_fn", choices=["kl", "mse"], default="kl")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_true")

    # Output
    parser.add_argument("--output_dir", default="experiments/distributed")
    parser.add_argument("--export_warmstart", default=None)
    parser.add_argument("--save_model", default=None)

    # DDP
    parser.add_argument("--no_distributed", action="store_true")

    args = parser.parse_args()
    if args.no_amp:
        args.use_amp = False

    # ── DDP init ────────────────────────────────────────────────────────
    use_ddp = not args.no_distributed and torch.cuda.is_available()
    if use_ddp:
        if "RANK" in os.environ:
            dist.init_process_group(backend="nccl")
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
        else:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            dist.init_process_group(backend="nccl", rank=0, world_size=1)
            rank, world_size, local_rank = 0, 1, 0
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    is_main = (rank == 0)
    log(f"Rank {rank}/{world_size}, device={device}", rank)

    # ── Discover data ───────────────────────────────────────────────────
    train_benchmarks = args.benchmarks
    if train_benchmarks is None:
        train_benchmarks = [b for b in ALL_BENCHES if b != args.target]

    train_files = []
    for bench in train_benchmarks:
        files = discover_files(args.data_root, bench, args.max_files_per_bench)
        log(f"  Train: {bench} → {len(files)} files", rank)
        train_files.extend(files)

    test_files = discover_files(args.data_root, args.target, args.max_files_per_bench)
    log(f"  Test:  {args.target} → {len(test_files)} files", rank)

    if not train_files:
        log("ERROR: No training files found!", rank)
        sys.exit(1)

    # ── DataLoaders ─────────────────────────────────────────────────────
    train_dataset = CompactGraphDataset(train_files, preload_edges=True)
    test_dataset = CompactGraphDataset(test_files, preload_edges=True)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if (use_ddp and world_size > 1) else None

    train_loader = DataLoader(
        train_dataset, batch_size=1,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_single,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=1, collate_fn=collate_single, pin_memory=True,
    )

    # ── Model (ORIGINAL DeepDGR_GNN) ───────────────────────────────────
    model = DeepDGR_GNN(
        grid_in=args.grid_in, cand_in=args.cand_in,
        hidden=args.hidden, num_layers=args.layers,
    ).to(device)

    if use_ddp and world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    base_model = model.module if hasattr(model, "module") else model
    n_params = sum(p.numel() for p in base_model.parameters())
    log(f"Original DeepDGR_GNN: {n_params:,} params, "
        f"hidden={args.hidden}, layers={args.layers}", rank)

    # ── Optimizer + scheduler ───────────────────────────────────────────
    optimizer = torch.optim.Adam(base_model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader) // args.grad_accum
    warmup = int(total_steps * 0.02)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = (step - warmup) / max(total_steps - warmup, 1)
        return 0.01 + 0.5 * 0.99 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── AMP scaler ──────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if (args.use_amp and torch.cuda.is_available()) else None

    # ── Output ──────────────────────────────────────────────────────────
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, "training_log.csv")
        log_file = open(log_path, "w")
        log_file.write("epoch,train_loss,val_loss,lr,elapsed_s,peak_mem_mb\n")

    # ── Training ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state = None
    t_start = time.time()

    log(f"\n{'='*70}", rank)
    log(f"  Training: {args.epochs} epochs, {len(train_loader)} batches/epoch", rank)
    log(f"  Grad accum: {args.grad_accum}, "
        f"Effective batch: {args.grad_accum * world_size}", rank)
    log(f"  AMP: {args.use_amp}", rank)
    log(f"{'='*70}\n", rank)

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, grad_accum_steps=args.grad_accum,
            use_amp=args.use_amp, loss_fn=args.loss_fn, rank=rank,
        )

        val_loss = None
        if (epoch + 1) % args.val_every == 0 and is_main:
            val_loss = validate(model, test_loader, device,
                                use_amp=args.use_amp, loss_fn=args.loss_fn)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone()
                              for k, v in base_model.state_dict().items()}

        if (epoch + 1) % args.log_every == 0 and is_main:
            lr_now = optimizer.param_groups[0]["lr"]
            log(f"  epoch {epoch+1:5d} | "
                f"train={stats['avg_loss']:.6f}  "
                f"val={val_loss if val_loss is not None else '?':>10}  "
                f"lr={lr_now:.5f}  "
                f"mem={stats['peak_memory_mb']:.0f}MB  "
                f"{stats['elapsed_s']:.1f}s", rank)

        if is_main:
            lr_now = optimizer.param_groups[0]["lr"]
            log_file.write(
                f"{epoch+1},{stats['avg_loss']:.6f},"
                f"{val_loss if val_loss is not None else ''},"
                f"{lr_now:.6f},{stats['elapsed_s']:.2f},"
                f"{stats['peak_memory_mb']:.0f}\n"
            )
            if (epoch + 1) % 100 == 0:
                log_file.flush()

        if (epoch + 1) % args.save_every == 0 and is_main:
            ckpt_path = os.path.join(args.output_dir,
                                      f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": base_model.state_dict(),
                "gnn_state_dict": base_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }, ckpt_path)
            log(f"  Checkpoint: {ckpt_path}", rank)

    total_time = time.time() - t_start

    # ── Save ────────────────────────────────────────────────────────────
    if is_main:
        log_file.close()

        final_state = best_state if best_state else {
            k: v.cpu().clone() for k, v in base_model.state_dict().items()
        }

        model_path = args.save_model or os.path.join(
            args.output_dir, f"gnn_h{args.hidden}_l{args.layers}_final.pth")
        torch.save({
            "gnn_state_dict": final_state,
            "model_state_dict": final_state,
            "args": vars(args),
            "best_val_loss": best_val_loss,
            "total_time_s": total_time,
        }, model_path)
        log(f"Model saved: {model_path}", rank)

        # Export warmstart
        if test_dataset and len(test_dataset) > 0:
            base_model.load_state_dict(final_state)
            test_sample = test_dataset[0]
            ws_path = args.export_warmstart or os.path.join(
                args.output_dir,
                f"DIST_{args.target.replace('ispd','').replace('_metal5','')}_warmstart.npz"
            )
            export_warmstart(base_model, test_sample, ws_path, device)
            log(f"Warmstart: {ws_path}", rank)

        log(f"\n{'='*70}", rank)
        log(f"  Training complete!", rank)
        log(f"  Best val loss: {best_val_loss:.6f}", rank)
        log(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)", rank)
        log(f"{'='*70}", rank)

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
