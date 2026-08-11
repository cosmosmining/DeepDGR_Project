# Verification evidence

Every number in `report/final_report.pdf` was re-derived from the shipped
measured CSVs. The checks below are reproducible with the shipped data alone
(no GPU, no benchmarks) — run `python3 verify_numbers.py` to regenerate this
table.

## 1. Table 1 (main result) — "ours" column vs `RESULTS/final_best.csv`

| chip | report ours (WL/via/of) | CSV (WL/via/of) | match |
|---|---|---|---|
| 18_t5  | 26.29M / 801k / 0   | 26.29M / 801k / 0   | ✅ |
| 18_t8  | 60.44M / 1.951M / 0 | 60.44M / 1.951M / 0 | ✅ |
| 18_t10 | 70.19M / 2.087M / 0 | 70.19M / 2.087M / 0 | ✅ |
| 19_t7  | 103.97M / 3.701M / 0| 103.97M / 3.701M / 0| ✅ |
| 19_t8  | 175.70M / 6.040M /10| 175.70M / 6.040M /10| ✅ |
| 19_t9  | 261.34M / 9.677M /28| 261.34M / 9.677M /28| ✅ |

## 2. Baselines (native CUGR2, DGR) vs measured routes

All six native and DGR rows match the report Table 1 exactly (WL to 2 d.p.,
overflow exactly). Source: the isolated-route measurements.

## 3. Δ% columns — recomputed from raw integers

`100·(ours−native)/native` reproduces every reported Δ to ±0.01 pp:
−1.06/−6.43, −1.42/−8.09, −3.53/−7.03, −0.65/−2.94, −0.68/−1.38,
−0.31/−5.01 (WL/via per chip). ✅

## 4. Headline claim — strict domination on **6/6**

For every chip, `ours ≤ native` AND `ours ≤ DGR` on **all three** metrics
(WL, via, overflow) simultaneously. Verified True on 6/6. ✅

## 5. Table 3 (entire-flow runtime) vs `RESULTS/flow_runtime.csv`

All six rows match (native route-only, DGR flow, ours flow, ours-vs-DGR
speedup 2.8×–4.8×). ✅

## 6. Figure 2 (overflow vs iteration) vs `RESULTS/overflow_curve_*.csv`

- 18_t5: warm@5it = 98,444 < cold@5it = 103,129; warm flat to 250it. ✅
- 19_t8: warm@5it = 377,285 < cold@5it = 419,415; warm flat to 250it. ✅

Confirms the "warm start begins below cold and converges in ~5–30 iters" claim.

## 7. Figure 1 / Pareto (parameter sweep) — data completeness

All three methods × 12 router settings × 6 chips = **216 measured operating
points** present (`RESULTS/scatter12.csv` + `experiments/cugr2_tune/iso_clean.csv`).
✅

---

## Honest caveat (also stated in the report §4.2, §4.4)

The 6/6 domination in Table 1 compares **ours at its per-benchmark tuned knob**
against **native/DGR at their standard settings**. Router-knob tuning helps
*every* method; the parameter-sweep / Pareto figures (Fig 1, `pareto_all`)
show all three methods across the **same** 12 settings so the comparison is
fair at every operating point. At matched settings the method clouds largely
overlap (the guide moves WL/via/overflow only slightly); the robust, defensible
wins are **speed** (few-iteration DGR, Table 2/3), the **one-model cross-
benchmark** warm start, and the **congested-chip overflow** reduction
(18→10 on 19_t8). This is stated plainly and not overclaimed.
