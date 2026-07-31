# HPC drivers (reference)

These are the SLURM submit scripts used on **PSC Bridges-2** (H100-80 /
RM-shared) to run the pipeline at scale. They are **cluster-specific**:
they hardcode `/ocean/...` paths, a conda env, and `module load cuda/12.1.1`.
Use them as a reference for how each stage was launched, then adapt the paths
to your own cluster. The plain `python3 ...` commands in
`../docs/REPRODUCE.md` are the portable way to reproduce every result.

| script | stage |
|---|---|
| `submit_gen_big.sh`      | from-scratch bounded-box synthetic data (certified) |
| `submit_gencong.sh`      | congested synthetic data (genuine overflow) |
| `submit_gen_ispd.sh`     | ISPD-template bounded pin randomization |
| `submit_train_fullsize.sh` | train the warm-start GNN on all six full-size designs |
| `submit_traincached.sh`  | cache-accelerated batch training / scaling study |
| `submit_final.sh`        | ensemble finishing -> final per-chip results |
