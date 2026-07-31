#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_synth_ispd.py — teacher-free synthetic data ACROSS THE 6 ISPD BENCHMARKS.

The from-scratch packs (gen_synth_batch.py) carry no ISPD family structure, so
a GNN trained only on them does not transfer to the real benchmarks. This
generator instead makes BOUNDED, locality-preserving variants of EACH of the 6
ISPD metal5 templates (so congestion still resembles the real chip), writing
the SAME teacher-free synth_field_v1 packs train_scalable.py consumes.

It swaps ONLY the instance source: instead of build_results (from scratch), it
loads an ISPD .pt and applies generate_compact_graphs._randomize_pins (the
bounded translate/jitter transform). Everything downstream — certify,
build_pack_from_results, pack_to_field_npz, the per-instance TMPDIR isolation —
is reused verbatim from gen_synth_batch.gen_one. All imports READ-ONLY; NEW file.

  python3 gen_synth_ispd.py --per 300 --workers 16 --certify --mode translate_jitter
"""
import argparse
import copy
import csv
import os
import shutil
import sys
import tempfile
import time
import traceback
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import gen_synth_batch as G                                  # read-only
from gen_synth_batch import pack_to_field_npz, MANIFEST_FIELDS

TEMPLATES = ["ispd18_test5_metal5", "ispd18_test8_metal5",
             "ispd18_test10_metal5", "ispd19_test7_metal5",
             "ispd19_test8_metal5", "ispd19_test9_metal5"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def template_path(name):
    for c in (os.path.join(ROOT, f"{name}.pt"),
              os.path.join(ROOT, "cu-gr-2", "run", f"{name}.pt")):
        if os.path.isfile(c):
            return c
    return None


def gen_one_ispd(job):
    """Load ISPD template -> bounded _randomize_pins -> (certify) -> teacher-free
    pack.  Mirrors gen_synth_batch.gen_one's isolation + pack/save path."""
    import torch
    from generate_compact_graphs import _randomize_pins
    from generate_scratch_benchmarks import certify
    from generate_e2e_graphs import build_pack_from_results

    fam, tpath, seed, idx, cfg = (job["fam"], job["tpath"], job["seed"],
                                  job["idx"], job["cfg"])
    t0 = time.time()
    tmp = tempfile.mkdtemp(prefix=f"ispdsyn_{os.getpid()}_{idx}_",
                           dir=cfg["scratch_tmp"])
    old = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = tmp
    row = {k: "" for k in MANIFEST_FIELDS}
    row.update(dict(grid=fam, seed=seed, certified=0))
    try:
        results = torch.load(tpath, map_location="cpu", weights_only=False)
        # BOUNDED, locality-preserving variant of the real ISPD chip
        _randomize_pins(results, seed, mode=cfg["mode"],
                        window_frac=cfg["window_frac"])
        reg = results["region"]
        row.update(dict(xmax=reg.xmax, ymax=reg.ymax,
                        n_nets=len(results["net"])))

        certified, max_of, wl_cost = 1, "", ""
        if cfg["certify"]:
            ok, info = certify(results, iters=cfg["certify_iters"], seed=seed)
            certified = int(ok)
            max_of, wl_cost = info.get("max_overflow", ""), info.get("wl_cost", "")
            if not ok and not cfg["keep_uncertified"]:
                row.update(dict(certified=0, max_overflow=max_of,
                                gen_s=round(time.time() - t0, 1)))
                return ("reject", row, idx)

        name = f"ispd_{fam}_s{seed:06d}"
        pack, _ = build_pack_from_results(
            copy.deepcopy(results), benchmark=name, pattern_level=1,
            extra_meta={"generator": "synth_ispd_v1", "family": fam,
                        "seed": seed, "teacher": "none"})
        del results
        flat = pack_to_field_npz(pack)
        del pack
        out_path = os.path.join(cfg["out_dir"], name + ".npz")
        tmp_npz = os.path.join(tmp, name + ".npz")
        np.savez_compressed(tmp_npz, **flat)
        shutil.move(tmp_npz, out_path)
        row.update(dict(
            path=os.path.relpath(out_path, cfg["out_root"]),
            n_cand=int(flat["n_candidates"]), n_subnets=int(flat["n_subnets"]),
            certified=certified, max_overflow=max_of, wl_cost=wl_cost,
            nnz_hor=int(flat["hor_idx"].shape[1]),
            nnz_ver=int(flat["ver_idx"].shape[1]),
            nnz_via=int(flat["via_idx"].shape[1]),
            size_mb=round(os.path.getsize(out_path) / 2**20, 3),
            gen_s=round(time.time() - t0, 1)))
        return ("ok", row, idx)
    except Exception as e:                                    # noqa: BLE001
        row.update(dict(certified=-1, gen_s=round(time.time() - t0, 1)))
        return ("error", {"row": row, "idx": idx,
                          "err": f"{type(e).__name__}: {e}",
                          "tb": traceback.format_exc()}, idx)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if old is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=300,
                    help="variants PER ISPD template (x6 families)")
    ap.add_argument("--templates", default=",".join(TEMPLATES))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mode", default="translate_jitter",
                    choices=["translate", "jitter", "translate_jitter"])
    ap.add_argument("--window_frac", type=float, default=0.04)
    ap.add_argument("--certify", action="store_true")
    ap.add_argument("--certify_iters", type=int, default=200)
    ap.add_argument("--keep_uncertified", action="store_true")
    ap.add_argument("--out_root", default=os.environ.get("DEEPDGR_DATA", "./synthdata"))
    ap.add_argument("--run", default="ispd")
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="skip (family,seed) whose pack already exists")
    ap.add_argument("--scratch_tmp", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()

    out_dir = os.path.join(a.out_root, a.run)
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(mode=a.mode, window_frac=a.window_frac, certify=a.certify,
               certify_iters=a.certify_iters,
               keep_uncertified=a.keep_uncertified, out_dir=out_dir,
               out_root=a.out_root, scratch_tmp=a.scratch_tmp)
    fams = a.templates.split(",")
    tps = {f: template_path(f) for f in fams}
    for f in fams:
        if tps[f] is None:
            log(f"WARN template missing: {f} — skipped")
    # INTERLEAVE: seed-outer, family-inner -> all 6 families covered from the
    # start (instead of finishing all of test5 before any other family).
    jobs, idx = [], 0
    for k in range(a.per):
        seed = a.base_seed + k
        for fam in fams:
            tp = tps[fam]
            if tp is None:
                continue
            if a.resume and os.path.isfile(
                    os.path.join(out_dir, f"ispd_{fam}_s{seed:06d}.npz")):
                continue
            jobs.append(dict(fam=fam, tpath=tp, seed=seed, idx=idx, cfg=cfg))
            idx += 1
    log(f"RUN '{a.run}': {len(jobs)} variants ({a.per}/family x "
        f"{len(set(j['fam'] for j in jobs))} families), mode={a.mode}, "
        f"certify={'on' if a.certify else 'OFF'}, workers={a.workers}")
    log(f"  -> {out_dir}")

    man = os.path.join(out_dir, "manifest.csv")
    new = not os.path.isfile(man)
    mh = open(man, "a", newline="")
    mw = csv.DictWriter(mh, fieldnames=MANIFEST_FIELDS)
    if new:
        mw.writeheader()
    n_ok = n_rej = n_err = 0
    t0 = time.time()
    with Pool(a.workers, initializer=G._worker_init,
              initargs=(1,)) as pool:
        for status, payload, _ in pool.imap_unordered(gen_one_ispd, jobs):
            if status == "ok":
                n_ok += 1
                mw.writerow(payload)
                mh.flush()
                if n_ok % 25 == 0:
                    log(f"  ok={n_ok} rej={n_rej} err={n_err} "
                        f"({n_ok/(time.time()-t0):.2f}/s)")
            elif status == "reject":
                n_rej += 1
                mw.writerow(payload)
                mh.flush()
            else:
                n_err += 1
                if n_err <= 3:
                    log(f"  ERR idx={payload['idx']}: {payload['err']}")
    mh.close()
    by_fam = {}
    for r in csv.DictReader(open(man)):
        if r.get("certified") == "1":
            by_fam[r["grid"]] = by_fam.get(r["grid"], 0) + 1
    log(f"DONE '{a.run}': ok={n_ok} reject={n_rej} err={n_err} in "
        f"{time.time()-t0:.0f}s")
    log(f"  certified per family: {by_fam}")
    log(f"  manifest: {man}")


if __name__ == "__main__":
    main()
