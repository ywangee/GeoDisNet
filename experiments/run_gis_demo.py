from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on any OS
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandapower as pp
import networkx as nx
import pyproj

try:
    import contextily as ctx  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    ctx = None

from src.network.gis_reader import read_gis_graph
from src.network.candidate_graph import extract_candidate_graph, graph_summary
from src.network.mip_feeder import MIPFeederParams, solve_mip_feeder, compute_downstream_counts
from src.network.synthetic_feeder import validate_feeder, export_feeder
from src.network.feeder_builder import build_pp_network


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "data" / "examples" / "simple_lv_feeder.geojson"
ROOT_COORD = (-0.10000, 51.50000)   # WGS-84 lon/lat of substation (node A)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GeoDistNet end-to-end demo: GIS → MIP → pandapower.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--geojson", type=str, default=str(FIXTURE),
        help="Path to input GeoJSON file.",
    )
    p.add_argument(
        "--root-lon", type=float, default=ROOT_COORD[0],
        help="Substation longitude (WGS-84).",
    )
    p.add_argument(
        "--root-lat", type=float, default=ROOT_COORD[1],
        help="Substation latitude (WGS-84).",
    )
    p.add_argument(
        "--snap-tol-m", type=float, default=5.0,
        help="Node-snapping tolerance [m].",
    )
    p.add_argument(
        "--max-edge-km", type=float, default=0.5,
        help="Drop candidate edges longer than this [km].",
    )
    p.add_argument(
        "--c-max", type=int, default=None,
        help="MIP C_max (max downstream nodes per edge). None = unconstrained (MST).",
    )
    p.add_argument(
        "--households", type=int, default=90,
        help="Total households to allocate across non-root buses.",
    )
    p.add_argument(
        "--base-kv", type=float, default=0.4,
        help="Nominal voltage [kV].",
    )
    p.add_argument(
        "--base-mva", type=float, default=0.5,
        help="MVA base [MVA].",
    )
    p.add_argument(
        "--solver", type=str, default="gurobi",
        help="MIP solver (gurobi / highs / cbc / glpk).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for impedance perturbation and household allocation.",
    )
    p.add_argument(
        "--out-dir", type=str, default="data/network_gis",
        help="Output directory for CSV/YAML feeder files.",
    )
    p.add_argument(
        "--p-kw-per-hh", type=float, default=1.0,
        help="Placeholder load per household for power-flow check [kW].",
    )
    p.add_argument(
        "--crs-metric-epsg", type=int, default=27700,
        help="EPSG code for metric projection (27700=OSGB36/UK, 32755=UTM-55S/Melbourne).",
    )
    p.add_argument(
        "--no-basemap", action="store_true",
        help="Disable city basemap tiles in overlay figure.",
    )
    p.add_argument(
        "--basemap-provider", type=str, default="CartoDB.PositronNoLabels",
        help="Tile provider path under contextily.providers (e.g. OpenStreetMap.Mapnik).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _plot_overlay(
    G_cand: nx.Graph,
    selected_edges: list[tuple[int, int]],
    root_bus: int,
    out_dir: pathlib.Path,
    *,
    crs_metric_epsg: int,
    use_basemap: bool,
    basemap_provider: str,
) -> pathlib.Path:
    """
    Generate a 2D map overlay figure: candidate routes (grey) + selected
    feeder tree (red) + substation (star).

    Saves to <out_dir>/figures/gis_overlay.png and returns the path.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    transformer = pyproj.Transformer.from_crs(crs_metric_epsg, 3857, always_xy=True)

    def _xy_3857(bus_id: int) -> tuple[float, float]:
        x_m = G_cand.nodes[bus_id]["x_km"] * 1000.0
        y_m = G_cand.nodes[bus_id]["y_km"] * 1000.0
        return transformer.transform(x_m, y_m)

    # Background: all candidate edges (light grey — input GIS geometry)
    for u, v in G_cand.edges():
        xu, yu = _xy_3857(u)
        xv, yv = _xy_3857(v)
        ax.plot([xu, xv], [yu, yv], color="#cccccc", linewidth=1.0, zorder=1)

    # All candidate nodes (light grey dots)
    xs = [_xy_3857(n)[0] for n in G_cand.nodes()]
    ys = [_xy_3857(n)[1] for n in G_cand.nodes()]
    ax.scatter(xs, ys, s=12, color="#bbbbbb", zorder=2)

    feeder_nodes = {root_bus} | {n for e in selected_edges for n in e}
    feeder_x = [_xy_3857(n)[0] for n in feeder_nodes] if feeder_nodes else xs
    feeder_y = [_xy_3857(n)[1] for n in feeder_nodes] if feeder_nodes else ys
    x_min, x_max = min(feeder_x), max(feeder_x)
    y_min, y_max = min(feeder_y), max(feeder_y)
    dx = max(1.0, x_max - x_min)
    dy = max(1.0, y_max - y_min)
    # Very tight framing so feeder fills most of the map area.
    pad_frac = 0.01
    ax.set_xlim(x_min - dx * pad_frac, x_max + dx * pad_frac)
    ax.set_ylim(y_min - dy * pad_frac, y_max + dy * pad_frac)

    # Selected tree edges (red)
    for u, v in selected_edges:
        xu, yu = _xy_3857(u)
        xv, yv = _xy_3857(v)
        ax.plot([xu, xv], [yu, yv], color="#e63946", linewidth=1.54, zorder=3)

    # Selected tree nodes (red dots, excluding root)
    tree_nodes = {n for edge in selected_edges for n in edge} - {root_bus}
    for n in tree_nodes:
        xn, yn = _xy_3857(n)
        ax.scatter(xn, yn, s=30, color="#e63946", zorder=4)

    # Root / substation (teal star)
    xr, yr = _xy_3857(root_bus)
    ax.scatter(xr, yr, s=180, marker="*", color="#2a9d8f", zorder=5)

    if use_basemap:
        if ctx is None:
            raise ImportError(
                "contextily is required for city basemap tiles. "
                "Install with: pip install contextily"
            )
        provider = ctx.providers
        for part in basemap_provider.split("."):
            if not hasattr(provider, part):
                raise ValueError(
                    f"Invalid --basemap-provider '{basemap_provider}'. "
                    "Example: OpenStreetMap.Mapnik"
                )
            provider = getattr(provider, part)
        ctx.add_basemap(ax, source=provider, crs="EPSG:3857", reset_extent=False)

    # Legend
    h_cand = mpatches.Patch(color="#cccccc", label="Candidate")
    h_tree = mpatches.Patch(color="#e63946", label="Feeder")
    h_root = plt.Line2D(
        [0], [0], marker="*", color="w", markerfacecolor="#2a9d8f",
        markersize=12, label="Substation",
    )
    ax.legend(handles=[h_cand, h_tree, h_root], loc="upper right", fontsize=9)

    ax.set_xlabel("Web Mercator X [m]")
    ax.set_ylabel("Web Mercator Y [m]")
    ax.set_title(
        f"GIS candidate graph ({G_cand.number_of_nodes()} nodes, "
        f"{G_cand.number_of_edges()} edges)  →  "
        f"MIP feeder ({len(selected_edges)} lines)"
    )
    yfmt = mticker.ScalarFormatter(useMathText=False)
    yfmt.set_scientific(True)
    yfmt.set_powerlimits((6, 6))
    yfmt.set_useOffset(True)
    ax.yaxis.set_major_formatter(yfmt)
    # Put Y-axis scientific offset at upper-left (x-axis-like style),
    # but keep it attached to the axis rather than title area.
    y_off = ax.yaxis.get_offset_text()
    y_off.set_ha("left")
    y_off.set_va("bottom")
    y_off.set_position((0.0, 1.0))
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.tight_layout()

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fpath = fig_dir / "gis_overlay.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fpath


def _sep(char: str = "=", width: int = 62) -> None:
    print(char * width)


def main() -> None:
    args = parse_args()
    root_coord = (args.root_lon, args.root_lat)

    _sep()
    print("GeoDistNet — end-to-end demo")
    _sep()
    print(f"  GeoJSON  : {args.geojson}")
    print(f"  Root     : lon={args.root_lon}  lat={args.root_lat}  (WGS-84)")
    print(f"  Snap tol : {args.snap_tol_m} m")
    print(f"  max edge : {args.max_edge_km} km")
    print(f"  C_max    : {args.c_max if args.c_max is not None else 'unconstrained (MST)'}")
    print(f"  HH total : {args.households}")
    print(f"  Solver   : {args.solver}")
    print(f"  Seed     : {args.seed}")
    print(f"  Basemap  : {'ON' if not args.no_basemap else 'OFF'}")
    if not args.no_basemap:
        print(f"  Provider : {args.basemap_provider}")

    # ------------------------------------------------------------------
    # Step 1: GIS read + projection
    # ------------------------------------------------------------------
    _sep("-")
    print(f"Step 1/5  GIS read + projection (EPSG:{args.crs_metric_epsg})")
    t0 = time.perf_counter()
    G_raw, root = read_gis_graph(
        args.geojson,
        root_coord=root_coord,
        crs_input_epsg=4326,
        crs_metric_epsg=args.crs_metric_epsg,
        snap_tol_m=args.snap_tol_m,
    )
    print(f"  Raw graph      : {G_raw.number_of_nodes()} nodes, {G_raw.number_of_edges()} edges  "
          f"({time.perf_counter() - t0:.3f} s)")

    # ------------------------------------------------------------------
    # Step 2: Candidate graph extraction
    # ------------------------------------------------------------------
    _sep("-")
    print("Step 2/5  Candidate graph extraction")
    t0 = time.perf_counter()
    G_cand, root = extract_candidate_graph(G_raw, root_node=root, max_edge_km=args.max_edge_km)
    summary = graph_summary(G_cand, root)
    print(f"  Candidate graph: {summary['n_nodes']} nodes, {summary['n_edges']} edges  "
          f"({time.perf_counter() - t0:.3f} s)")
    print(f"  Edge lengths   : min={summary['length_km_min']:.4f} km  "
          f"max={summary['length_km_max']:.4f} km  "
          f"total={summary['length_km_total']:.4f} km")
    print(f"  Connected      : {summary['is_connected']}")

    n = G_cand.number_of_nodes()
    c_max = args.c_max if args.c_max is not None else (n - 1)

    # ------------------------------------------------------------------
    # Step 3: MIP feeder synthesis
    # ------------------------------------------------------------------
    _sep("-")
    print(f"Step 3/5  MIP feeder synthesis  (C_max={c_max}, n={n})")
    params = MIPFeederParams(
        C_max=c_max,
        total_households=args.households,
        base_kv=args.base_kv,
        base_mva=args.base_mva,
        solver=args.solver,
    )
    t0 = time.perf_counter()
    result = solve_mip_feeder(G_cand, root, params, seed=args.seed)
    elapsed = time.perf_counter() - t0

    print(f"  Status         : {result.status}")
    print(f"  Objective      : {result.objective_km:.4f} km  (total cable)")
    print(f"  Selected edges : {len(result.selected_edges)}")
    print(f"  Solve time     : {result.solve_time_s:.3f} s  (wall: {elapsed:.3f} s)")

    if result.status not in ("optimal", "feasible") or result.feeder is None:
        print("\nERROR: MIP did not find a feasible solution.  Exiting.")
        sys.exit(1)

    feeder = result.feeder

    # Downstream counts for C_max compliance report
    dc = compute_downstream_counts(result.selected_edges)
    max_downstream = max(dc.values()) if dc else 0
    print(f"  Max downstream : {max_downstream}  (C_max limit: {c_max})")

    # ------------------------------------------------------------------
    # Step 4: Validate + export
    # ------------------------------------------------------------------
    _sep("-")
    print("Step 4/5  Validate + export feeder")
    validate_feeder(feeder)
    print("  Structural validation: PASSED")

    out_dir = pathlib.Path(args.out_dir)
    export_feeder(feeder, out_dir)
    print(f"  Exported to    : {out_dir.resolve()}")
    print(f"    buses.csv  ({feeder.n_buses} rows)")
    print(f"    lines.csv  ({feeder.n_lines} rows)")
    print(f"    feeder_params.yaml")

    overlay_path = _plot_overlay(
        G_cand,
        result.selected_edges,
        feeder.root_bus,
        out_dir,
        crs_metric_epsg=args.crs_metric_epsg,
        use_basemap=not args.no_basemap,
        basemap_provider=args.basemap_provider,
    )
    print(f"  Overlay figure : {overlay_path}")

    hh_vals = [b.n_households for b in feeder.buses if b.id != feeder.root_bus]
    lengths  = [ln.length_km for ln in feeder.lines]
    print(f"\n  Feeder summary")
    print(f"    Buses      : {feeder.n_buses}  (root + {feeder.n_buses - 1} load buses)")
    print(f"    Lines      : {feeder.n_lines}")
    print(f"    Households : {feeder.total_households}  "
          f"(min={min(hh_vals)}  max={max(hh_vals)}  mean={sum(hh_vals)/len(hh_vals):.1f})")
    print(f"    Length km  : total={sum(lengths):.4f}  "
          f"min={min(lengths):.4f}  max={max(lengths):.4f}")
    print(f"    Base kV    : {feeder.base_kv} kV")
    print(f"    Z_base     : {feeder.Z_base_ohm:.4f} Ohm")

    # ------------------------------------------------------------------
    # Step 5: pandapower power flow
    # ------------------------------------------------------------------
    _sep("-")
    print("Step 5/5  pandapower Newton-Raphson power flow")
    net = build_pp_network(
        out_dir / "buses.csv",
        out_dir / "lines.csv",
        out_dir / "feeder_params.yaml",
        p_kw_per_hh=args.p_kw_per_hh,
        q_kvar_per_hh=0.1,
    )
    print(f"  Network        : {len(net.bus)} buses, {len(net.line)} lines, {len(net.load)} loads")
    print(f"  Total load     : {net.load.p_mw.sum()*1000:.2f} kW active, "
          f"{net.load.q_mvar.sum()*1000:.2f} kVAr reactive")

    pp.runpp(net, algorithm="nr", verbose=False)

    if net.converged:
        v_min = net.res_bus.vm_pu.min()
        v_max = net.res_bus.vm_pu.max()
        losses_kw = net.res_line.pl_mw.sum() * 1000.0
        print(f"  Converged      : YES")
        print(f"  Voltage [pu]   : min={v_min:.6f}  max={v_max:.6f}")
        print(f"  Line losses    : {losses_kw:.4f} kW")
    else:
        print("  Converged      : NO — power flow diverged.")
        sys.exit(1)

    _sep()
    print("Demo complete.")
    _sep()


if __name__ == "__main__":
    main()
