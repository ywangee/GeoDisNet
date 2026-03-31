# DN-Design: Distribution Network Design Toolbox

DN-Design is a Python toolbox for generating radial distribution feeders, from fully synthetic cases to GIS-informed real-street topologies.

## Overview

The project supports three construction paths and exports a shared CSV/YAML feeder format compatible with [pandapower](https://pandapower.readthedocs.io).

| Path | Method | Typical use |
|------|--------|-------------|
| **A — Synthetic** | MST on random bus coordinates | Rapid prototyping and large sweeps |
| **B — Layout** | Comb/herringbone-style feeder templates | Stylised archetype studies |
| **C — GIS MIP** | GeoJSON → candidate graph → constrained radial MIP | Real-geography feeder synthesis |

## Installation

```bash
conda create -n geodistnet python=3.11
conda activate geodistnet
pip install -r requirements.txt
```

Path C needs a MIP solver. The code tries `gurobi -> highs -> cbc -> glpk` in order.

## Quick Start

**Path A (synthetic feeder generation)**
```bash
python experiments/build_synthetic_network.py
```

**Path C (GIS end-to-end demo)**
```bash
python experiments/run_gis_demo.py
```

**Path C full pipeline (GIS -> MIP -> loading scenarios)**
```bash
python experiments/run_path_c_pipeline.py
```

## Minimal Repository Layout

```text
src/network/
  synthetic_feeder.py
  layout_feeder.py
  feeder_builder.py
  feeder_source.py
  gis_feeder.py
  gis_reader.py
  candidate_graph.py
  mip_feeder.py
  solver_utils.py

experiments/
  build_synthetic_network.py
  build_real_world_case.py
  run_gis_demo.py
  run_mst_vs_mip.py
  run_path_c_pipeline.py
  applications/run_loading_scenarios.py

data/examples/
  simple_lv_feeder.geojson
  lv_feeder_32bus.geojson
  osm_bethnal_green.geojson

data/network_test/   # minimal fixture used by tests
```

## Output Files

Generated feeders use:

| File | Meaning |
|------|---------|
| `buses.csv` | Bus metadata and household allocation |
| `lines.csv` | Edge connectivity and electrical parameters |
| `feeder_params.yaml` | Base values and solver metadata |

These files can be loaded by `src/network/feeder_builder.py`.

## Clean GitHub Policy

This repo keeps only source code and minimal input fixtures. Regeneratable outputs are intentionally excluded from version control, including:

- `data/network_*/`
- `data/results*/`
- `data/comparison/`

Run experiment scripts locally to regenerate them when needed.

## Running Tests

```bash
conda run -n geodistnet pytest tests/ -v
```

## Dependencies

See `requirements.txt`. Core packages include `pandapower`, `pyomo`, `networkx`, `pyproj`, `numpy`, `scipy`, and `pandas`.

## License

MIT
