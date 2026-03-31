from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_GEOJSON    = REPO_ROOT / "data" / "examples" / "lv_feeder_32bus.geojson"
DEFAULT_ROOT_LON   = -0.10000
DEFAULT_ROOT_LAT   = 51.50000
DEFAULT_BASE_KV    = 11.0         # medium-voltage, consistent with Path B
DEFAULT_BASE_MVA   = 1.0
DEFAULT_HOUSEHOLDS = 155          # 5 per non-root bus (31 load buses)
DEFAULT_FEEDER_DIR  = "data/network_gis"
DEFAULT_RESULTS_DIR = "data/results_gis"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Path C: GIS → MIP feeder synthesis → loading scenario validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--geojson",     default=str(DEFAULT_GEOJSON),
                   help="GeoJSON file containing LineString cable features.")
    p.add_argument("--root-lon",    type=float, default=DEFAULT_ROOT_LON,
                   help="Substation longitude (WGS-84).")
    p.add_argument("--root-lat",    type=float, default=DEFAULT_ROOT_LAT,
                   help="Substation latitude (WGS-84).")
    p.add_argument("--snap-tol-m",  type=float, default=5.0,
                   help="Node-snapping tolerance [m].")
    p.add_argument("--max-edge-km", type=float, default=0.5,
                   help="Drop candidate edges longer than this [km].")
    p.add_argument("--c-max",       type=int, default=None,
                   help="MIP C_max (max downstream nodes per edge). None = MST.")
    p.add_argument("--households",  type=int, default=DEFAULT_HOUSEHOLDS,
                   help="Total households to allocate across non-root buses.")
    p.add_argument("--base-kv",     type=float, default=DEFAULT_BASE_KV,
                   help="Nominal voltage [kV].")
    p.add_argument("--base-mva",    type=float, default=DEFAULT_BASE_MVA,
                   help="MVA base [MVA].")
    p.add_argument("--solver",      default="gurobi",
                   help="MIP solver (gurobi / highs / cbc / glpk).")
    p.add_argument("--seed",        type=int, default=42,
                   help="Random seed for impedance perturbation and HH allocation.")
    p.add_argument("--feeder-dir",  default=DEFAULT_FEEDER_DIR,
                   help="Output directory for exported feeder CSV/YAML files.")
    p.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                   help="Output directory for scenario CSV tables and figures.")
    p.add_argument("--crs-metric-epsg", type=int, default=27700,
                   help="EPSG code for metric projection (27700=OSGB36/UK, 32755=UTM-55S/Melbourne).")
    p.add_argument("--no-basemap", action="store_true",
                   help="Disable city basemap tiles in GIS overlay figure.")
    p.add_argument("--basemap-provider", default="CartoDB.Positron",
                   help="Tile provider path under contextily.providers (e.g. OpenStreetMap.Mapnik).")
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    """Print and execute a command, raising on non-zero exit."""
    print("\n" + "=" * 62)
    print(">>> " + " ".join(cmd))
    print("=" * 62 + "\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Stage 1 — GIS → MIP → export feeder files
    # ------------------------------------------------------------------
    demo_cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "run_gis_demo.py"),
        "--geojson",     args.geojson,
        "--root-lon",    str(args.root_lon),
        "--root-lat",    str(args.root_lat),
        "--snap-tol-m",  str(args.snap_tol_m),
        "--max-edge-km", str(args.max_edge_km),
        "--households",  str(args.households),
        "--base-kv",     str(args.base_kv),
        "--base-mva",    str(args.base_mva),
        "--solver",      args.solver,
        "--seed",        str(args.seed),
        "--out-dir",     args.feeder_dir,
    ]
    if args.c_max is not None:
        demo_cmd += ["--c-max", str(args.c_max)]
    if args.crs_metric_epsg != 27700:
        demo_cmd += ["--crs-metric-epsg", str(args.crs_metric_epsg)]
    if args.no_basemap:
        demo_cmd += ["--no-basemap"]
    if args.basemap_provider != "CartoDB.Positron":
        demo_cmd += ["--basemap-provider", args.basemap_provider]

    _run(demo_cmd)

    # ------------------------------------------------------------------
    # Stage 2 — multi-scenario loading validation
    # ------------------------------------------------------------------
    scenarios_cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "run_loading_scenarios.py"),
        "--data-dir", args.feeder_dir,
        "--out-dir",  args.results_dir,
    ]

    _run(scenarios_cmd)

    print("\n" + "=" * 62)
    print(f"Path C pipeline complete.")
    print(f"  Feeder data : {args.feeder_dir}")
    print(f"  Results     : {args.results_dir}")
    print("=" * 62)


if __name__ == "__main__":
    main()
