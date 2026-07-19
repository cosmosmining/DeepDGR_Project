# Benchmarks — where to get them and where to put them

All reported results use the six **ISPD 2018 & 2019 global-routing
benchmarks** (metal5 variants). They are **not redistributed here** (contest
license); download them from the official sources and place them as shown.

## The six benchmarks

```
ispd18_test5_metal5   ispd18_test8_metal5   ispd18_test10_metal5
ispd19_test7_metal5   ispd19_test8_metal5   ispd19_test9_metal5
```

## Where to get them

- **ISPD 2018** initial-detailed-routing contest benchmarks:
  http://www.ispd.cc/contests/18/  (LEF/DEF)
- **ISPD 2019** contest benchmarks:
  http://www.ispd.cc/contests/19/  (LEF/DEF)

The CUGR2 repo (`github.com/wadmes/cu-gr-2`) also documents its expected
benchmark format and provides conversion notes; the `.input.lef` / `.input.def`
naming below is what CUGR2 consumes.

## Where to put them

Each script resolves inputs via `tune_cugr2.py: find_input()`, which searches
these paths (first match wins):

```
cu-gr-2/benchmark/test/<bench>/<bench>.input.lef
cu-gr-2/benchmark/test/<bench>/<bench>.input.def
# fallbacks also tried:
cu-gr-2/benchmark/<bench>/<bench>.input.{lef,def}
cu-gr-2/benchmark/<bench>.input.{lef,def}
```

So the canonical layout is:

```
cu-gr-2/benchmark/test/
├── ispd18_test5_metal5/
│   ├── ispd18_test5_metal5.input.lef
│   └── ispd18_test5_metal5.input.def
├── ispd18_test8_metal5/
│   ├── ispd18_test8_metal5.input.lef
│   └── ispd18_test8_metal5.input.def
├── ispd18_test10_metal5/ ...
├── ispd19_test7_metal5/  ...
├── ispd19_test8_metal5/  ...
└── ispd19_test9_metal5/  ...
```

## Preprocessing to `.pt` (one-time per benchmark)

The DGR optimizer consumes a preprocessed `.pt` (nets, capacities, Steiner
trees, layer metadata) produced by CUGR2 + the parser:

```bash
python3 data_process_CUGR2.py cu-gr-2 cu-gr-2/benchmark/test/ispd18_test5_metal5
# writes ispd18_test5_metal5.pt   (loaded by dgr_fast.py / test_gnn_all6.py via bench_pt())
```

`bench_pt()` (in `test_gnn_all6.py`) looks for `<bench>.pt` at the repo root or
under `cu-gr-2/run/`. Put the generated `.pt` in either location.

## ISPD'24 (optional, industrial scale-out)

`ispd24_adapter.py` converts ISPD'24 `.cap` / `.net` contest files directly to
the pipeline's internal format (no LEF/DEF needed). The ISPD'24 benchmarks are
Google-Drive hosted by the contest organizers; download them, then:

```bash
python3 ispd24_adapter.py --cap <design>.cap --net <design>.net --out <design>.pt
python3 test_ispd24_adapter.py     # unit test of the adapter
```
