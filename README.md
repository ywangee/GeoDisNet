# DN-Design

GeoDisNet is a toolbox for radial distribution network design with synthetic and GIS-informed workflows, including MIP-based feeder synthesis and pandapower validation.

## Installation

```bash
conda create -n geodistnet python=3.11
conda activate geodistnet
pip install -r requirements.txt
```

Path C requires an LP/MIP solver. The code attempts:
`gurobi -> highs -> cbc -> glpk`.

## Quick Start

### 1) Synthetic feeder (Path A)

```bash
python experiments/build_synthetic_network.py
```

### 2) GIS demo (Path C)

```bash
python experiments/run_gis_demo.py
```

### 3) Full Path C pipeline (GIS -> MIP -> loading scenarios)

```bash
python experiments/run_path_c_pipeline.py
```

## Core Workflows

### Build a real-world GeoJSON case

```bash
python experiments/build_real_world_case.py
```

### Run MST vs constrained-MIP comparison

```bash
python experiments/run_mst_vs_mip.py
```

### Run loading scenarios on exported feeder files

```bash
python experiments/applications/run_loading_scenarios.py \
  --data-dir data/network \
  --out-dir data/results
```

## Inputs

Path C expects GeoJSON `LineString` features in WGS-84.

Minimal fixtures in `data/examples/`:

- `simple_lv_feeder.geojson`
- `lv_feeder_32bus.geojson`
- `osm_bethnal_green.geojson`

## Outputs

Each generated feeder is exported as:

- `buses.csv`
- `lines.csv`
- `feeder_params.yaml`

Scenario validation additionally exports summary tables and figures
(voltage profile and branch loading).

## Testing

```bash
conda run -n geodistnet pytest tests/ -v
```

## Sample 

<img width="400" height="400" alt="gis_melb" src="https://github.com/user-attachments/assets/2483ece0-ac41-450a-bf02-89ffa520cdce" />


## License

MIT
