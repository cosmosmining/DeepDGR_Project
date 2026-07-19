# CUGR2 goes here

This directory is a placeholder. Build the CUGR2 router from source into it:

    git clone https://github.com/wadmes/cu-gr-2 .    # (into this directory)
    # follow that repo's cmake/make build -> produces run/route

Expected after build (referenced by tune_cugr2.py):
    cu-gr-2/run/route            # the global router binary
    cu-gr-2/run/POWV9.dat        # FLUTE lookup tables
    cu-gr-2/run/POST9.dat
    cu-gr-2/benchmark/test/<bench>/<bench>.input.lef   # your benchmarks
    cu-gr-2/benchmark/test/<bench>/<bench>.input.def

See ../docs/SETUP.md and ../docs/BENCHMARKS.md.
