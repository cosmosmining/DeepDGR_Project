#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_scalable.py — out-of-core, BATCHED, TEACHER-FREE GNN trainer that ingests
the certified-routable synth_field_v1 packs from gen_synth_batch.py and, in ONE
run, emits a data-SCALING-LAW curve (held-out generalization vs #instances seen:
5,50,500,5000,...) against a FIXED held-out split.

Why this exists: the quick e2e_stream proxy can't answer the scaling question
(its held-out instance changes with N). This trainer fixes the held-out split
and streams hundreds of thousands of packs out-of-core, so the curve is valid.

Design (reuses e2e_stream.py + fastdgr_core READ-ONLY; existing code untouched):
  * pack(.npz synth_field_v1) -> FullProblem  [pack_to_full, the only new glue]
  * graph + GNN + loss          : e2e_stream.derive_graph_from_full / Instance /
                                  seg_softmax ; full.objective_full (DGR physics)
  * teacher-free                 : loss = of + wl_coeff*wl + via_coeff*via on the
                                  GNN's gumbel-softmax distribution. NO targets.
  * out-of-core                  : packs streamed by path; one materialized at a
                                  time; gradients ACCUMULATED over batch≈100
                                  instances per optimizer step (ragged sizes).
  * scaling curve                : at instances-seen milestones, eval the FIXED
                                  held-out set -> scaling_curve.csv + .pdf.
  * AMP + single-GPU default (DDP-ready). H100.

  # CPU smoke (self-contained, no packs needed):
  python3 train_scalable.py --smoke --device -1
  # real run on the generated packs (H100):
  python3 train_scalable.py --packs '/ocean/.../synthdata/v1/*.npz' \
      --batch 100 --milestones 5,50,500,5000 --device 0
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
from fastdgr_core import FullProblem, ColMatrix                 # read-only
import e2e_stream as ES                                          # read-only
from deepdgr_e2e import DeepDGR_GNN                              # read-only


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ════════════════════════════════════════════════════════════════════════
#  The ONLY new glue: synth_field_v1 pack -> FullProblem
# ════════════════════════════════════════════════════════════════════════

def pack_to_full(npz_path, device, wl_coeff=0.5, via_coeff=4.0):
    d = np.load(npz_path)
    xmax, ymax = int(d["xmax"]), int(d["ymax"])
    Eh, Ev, G = xmax * (ymax - 1), (xmax - 1) * ymax, xmax * ymax
    n = int(d["wire_length"].shape[0])
    t = lambda a: torch.as_tensor(np.asarray(a))

    def coo(pfx, rows):
        idx = t(d[f"{pfx}_idx"]).long()
        val = t(d[f"{pfx}_val"]).float()
        return torch.sparse_coo_tensor(idx, val, (rows, n)).coalesce()

    hor, ver, via = coo("hor", Eh), coo("ver", Ev), coo("via", G)

    def edge_w(key, full_n, strip_n, interleave):
        """pack may store the edge-length STRIP ([ymax-1] for hor, [xmax-1] for
        ver) or the full [Eh]/[Ev] vector — expand strip -> full to match the
        x-outer/y-inner edge flattening used by the incidence."""
        v = t(d[key]).float().reshape(-1)
        if v.numel() == full_n:
            return v
        if v.numel() == strip_n:
            return (v.repeat_interleave(ymax) if interleave else v.repeat(xmax))
        return v.reshape(-1)[:full_n]
    w_h = edge_w("hor_edge_length", Eh, ymax - 1, interleave=False)
    w_v = edge_w("ver_edge_length", Ev, xmax - 1, interleave=True)
    # pin-demand fields are per grid-cell (Ng=xmax*ymax); FullProblem wants 2D
    hpd = t(d["hor_pin_demand"]).float().reshape(-1)
    vpd = t(d["ver_pin_demand"]).float().reshape(-1)
    hpd = (hpd[:G].reshape(xmax, ymax) if hpd.numel() >= G
           else hpd.reshape(xmax, ymax))
    vpd = (vpd[:G].reshape(xmax, ymax) if vpd.numel() >= G
           else vpd.reshape(xmax, ymax))
    full = FullProblem(
        xmax=xmax, ymax=ymax,
        hor=ColMatrix.from_coo(hor, device=device),
        ver=ColMatrix.from_coo(ver, device=device),
        via=ColMatrix.from_coo(via, device=device),
        wire_length=t(d["wire_length"]).float().to(device),
        via_count=t(d["via_count"]).float().to(device),
        p_index=t(d["p_index"]).long(),
        hor_cap=t(d["hor_cap"]).float().reshape(-1).to(device),
        ver_cap=t(d["ver_cap"]).float().reshape(-1).to(device),
        w_h=w_h.to(device), w_v=w_v.to(device),
        hor_pin_demand=hpd.to(device), ver_pin_demand=vpd.to(device),
        via_layer=float(d["via_layer"]), m2_pitch=float(d["m2_pitch"]),
        act="sigmoid", act_scale=0.5, celu_alpha=2.0, add_via=True,
        device=device)
    name = os.path.basename(npz_path).rsplit(".", 1)[0]
    x_dict, ei, gin, cin = ES.derive_graph_from_full(full)
    return ES.Instance(name, full, x_dict, ei, gin, cin, wl_coeff, via_coeff)


def specs_from_packs(paths):
    return [("pack", p) for p in paths]


# ════════════════════════════════════════════════════════════════════════
#  Trainer: out-of-core stream, batch grad-accum, fixed held-out, milestones
# ════════════════════════════════════════════════════════════════════════

def materialize(spec, device):
    kind, payload = spec
    if kind == "pack":
        return pack_to_full(payload, device)
    if kind == "synth":                                   # smoke
        full = ES.make_synthetic_full(payload, device)
        x, ei, gin, cin = ES.derive_graph_from_full(full)
        return ES.Instance(f"synth{payload}", full, x, ei, gin, cin, 0.5, 4.0)
    raise ValueError(kind)


@torch.no_grad()
def heldout_eval(gnn, held, device):
    gnn.eval()
    tot, oftot = 0.0, 0.0
    for sp in held:
        inst = materialize(sp, device)
        l, of = inst.eval_loss(gnn)
        tot += l
        oftot += of
    gnn.train()
    return tot / max(len(held), 1), oftot / max(len(held), 1)


def train(specs, held, gnn, opt, device, batch, milestones, max_steps,
          out_csv, fig, amp=False):
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    rows = []
    seen = 0
    ms = sorted(set(milestones))
    mi = 0
    fh = open(out_csv, "w", newline="")
    w = csv.writer(fh)
    w.writerow(["instances_seen", "train_loss", "heldout_loss",
                "heldout_overflow"])
    step = 0
    run = True
    while run and step < max_steps:
        opt.zero_grad()
        acc = 0.0
        for b in range(batch):
            sp = specs[seen % len(specs)]
            seen += 1
            inst = materialize(sp, device)
            temp = max(0.1, 1.0 - step / max(max_steps, 1))
            with torch.cuda.amp.autocast(enabled=amp):
                loss, of, vc, wl = inst.loss(gnn, None, temp)
            scaler.scale(loss / batch).backward()
            acc += float(loss)
            # milestone check on #instances seen
            if mi < len(ms) and seen >= ms[mi]:
                hl, ho = heldout_eval(gnn, held, device)
                w.writerow([seen, round(acc / (b + 1), 3), round(hl, 3),
                            round(ho, 3)])
                fh.flush()
                rows.append((seen, hl, ho))
                log(f"  milestone {seen} instances: train~{acc/(b+1):.2f} "
                    f"held-out loss={hl:.2f} overflow={ho:.2f}")
                mi += 1
                if mi >= len(ms) and seen >= ms[-1]:
                    run = False
                    break
        scaler.step(opt)
        scaler.update()
        step += 1
    fh.close()
    if len(rows) >= 2:
        d = rows[0][1] - rows[-1][1]
        log(f"SCALING: held-out loss {rows[0][1]:.1f} (n={rows[0][0]}) -> "
            f"{rows[-1][1]:.1f} (n={rows[-1][0]}) = {d:+.1f} "
            f"({'scaling HELPS' if d > 0 else 'flat'})")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            xs = [r[0] for r in rows]
            ys = [r[1] for r in rows]
            f, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(xs, ys, "o-", color="black", mfc="none")
            ax.set_xscale("log")
            ax.set_xlabel("# training instances seen (log)")
            ax.set_ylabel("held-out generalization loss")
            ax.set_title("Data-scaling law (fixed held-out)")
            f.tight_layout()
            os.makedirs(os.path.dirname(fig), exist_ok=True)
            f.savefig(fig)
            log(f"figure -> {fig}")
        except Exception as e:
            log(f"plot skipped: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", default=None, help="glob of synth_field .npz")
    ap.add_argument("--smoke", action="store_true",
                    help="self-contained CPU smoke (in-file synthetic, no packs)")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--milestones", default="5,50,500,5000")
    ap.add_argument("--max_steps", type=int, default=100000)
    ap.add_argument("--holdout", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--out_csv", default=os.path.join(
        ROOT, "validation_out", "train_scalable_curve.csv"))
    ap.add_argument("--fig", default=os.path.join(
        ROOT, "reports", "figs", "scaling_law_proper.pdf"))
    a = ap.parse_args()
    dev = (f"cuda:{a.device}" if a.device >= 0 and torch.cuda.is_available()
           else "cpu")
    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    milestones = [int(x) for x in a.milestones.split(",")]

    if a.smoke:
        specs = [("synth", s) for s in range(64)]
        held = [("synth", 900 + s) for s in range(4)]
        milestones = [4, 16, 64]
        a.batch, a.max_steps = 4, 30
        log("SMOKE: 64 in-file synthetic train + 4 held-out (CPU ok)")
    else:
        if not a.packs:
            raise SystemExit("pass --packs <glob> or --smoke")
        paths = sorted(glob.glob(a.packs))
        if len(paths) < a.holdout + 1:
            raise SystemExit(f"need > {a.holdout} packs, found {len(paths)}")
        held = specs_from_packs(paths[:a.holdout])      # FIXED held-out
        specs = specs_from_packs(paths[a.holdout:])
        log(f"{len(specs)} train packs, {len(held)} FIXED held-out; "
            f"batch={a.batch} milestones={milestones}")

    # init GNN from the first instance's feature dims (same call as
    # e2e_stream.init_gnn: DeepDGR_GNN(grid_in=, cand_in=, hidden=, num_layers=))
    first = materialize(specs[0], dev)
    gnn = DeepDGR_GNN(grid_in=first.grid_in, cand_in=first.cand_in,
                      hidden=a.hidden, num_layers=a.layers).to(dev)
    with torch.no_grad():
        gnn(first.x_dict, first.edge_index_dict)         # init lazy SAGEConv
    n_params = sum(p.numel() for p in gnn.parameters())
    log(f"GNN ready: {n_params:,} params (grid_in={first.grid_in} "
        f"cand_in={first.cand_in})")
    opt = torch.optim.Adam(gnn.parameters(), lr=a.lr)
    train(specs, held, gnn, opt, dev, a.batch, milestones, a.max_steps,
          a.out_csv, a.fig, amp=a.amp)
    ckpt = os.path.join(ROOT, "train_scalable_gnn.pth")
    torch.save(gnn.state_dict(), ckpt)
    log(f"checkpoint -> {ckpt}  ;  curve -> {a.out_csv}")


if __name__ == "__main__":
    main()
