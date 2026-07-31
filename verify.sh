#!/bin/bash
# verify.sh — self-check that the environment and scripts are wired correctly.
# Runs the parts that need NO benchmarks and NO GPU: import check, byte-compile,
# and regeneration of the report figures from the shipped RESULTS/*.csv.
# For the full raw-benchmark -> routing reproduction, see docs/REPRODUCE.md
# (needs CUGR2 built via ./setup_cugr2.sh and the ISPD benchmarks in place).
set -uo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
pass=0; fail=0
ok(){ echo "  [PASS] $1"; pass=$((pass+1)); }
no(){ echo "  [FAIL] $1"; fail=$((fail+1)); }

echo "== 1. Python + heavy imports =="
$PY - <<'P' && ok "torch / torch_geometric / torch_scatter / numpy / matplotlib / seaborn / pptx import" || no "heavy imports"
import torch, torch_geometric, torch_scatter, numpy, matplotlib, seaborn, pptx
P

echo "== 2. byte-compile all scripts =="
$PY -m py_compile *.py && ok "all .py compile" || no "compile"

echo "== 3. regenerate report figures from RESULTS/*.csv (no benchmarks needed) =="
$PY plot_scatter12_pro.py       >/dev/null 2>&1 && ok "Fig 1  parameter-sweep scatter"        || no "scatter"
$PY plot_scatter12_pro.py --pareto >/dev/null 2>&1 && ok "Fig 1b Pareto-frontier variant"      || no "scatter-pareto"
$PY make_color_plots.py         >/dev/null 2>&1 && ok "Fig 2  overflow curves + scaling + runtime" || no "color plots"
$PY make_figset.py              >/dev/null 2>&1 && ok "Table 3 flow-runtime figure + csv"       || no "figset"

echo "== 4. outputs present =="
for f in RESULTS/figs/scatter12pro_all-1.png RESULTS/figs/runtime_speedup-1.png \
         RESULTS/figs/fig_flow_runtime-1.png RESULTS/flow_runtime.csv; do
  [ -s "$f" ] && ok "wrote $f" || no "missing $f"
done

echo
echo "== summary: $pass passed, $fail failed =="
[ "$fail" -eq 0 ] && echo "OK — environment and figure pipeline verified." || \
  echo "Some checks failed; see docs/SETUP.md."
exit $fail
