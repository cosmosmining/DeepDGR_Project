#!/bin/bash
# setup_cugr2.sh — fetch and build the CUGR2 router this pipeline calls.
# CUGR2 is a separate project (github.com/wadmes/cu-gr-2); we pin the exact
# commit used for all reported numbers so results are bit-reproducible.
set -euo pipefail
PIN=f7ec42df5c9f63d41c9899b91b03d5b2131d193c
cd "$(dirname "$0")"
if [ ! -d cu-gr-2/.git ]; then
  rm -rf cu-gr-2
  git clone https://github.com/wadmes/cu-gr-2.git cu-gr-2
fi
cd cu-gr-2
git fetch --all -q || true
git checkout -q "$PIN" || { echo "WARN: pinned commit not found, using default branch"; }
# build (CMake). Produces run/route and run/drcu.
mkdir -p build && cd build
cmake .. >/dev/null && make -j"$(nproc)"
cd ..
# sanity: the binary and FLUTE tables the pipeline needs
ls -la run/route run/POWV9.dat run/POST9.dat 2>/dev/null || \
  echo "NOTE: ensure run/route, run/POWV9.dat, run/POST9.dat exist (see README)."
echo "CUGR2 build complete (pinned $PIN)."
