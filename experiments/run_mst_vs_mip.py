from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import pandas as pd
import pyproj

try:
    import contextily as ctx  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover
    ctx = None

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.network.gis_reader import read_gis_graph
from src.network.candidate_graph import extract_candidate_graph
from src.network.mip_feeder import MIPFeederParams, solve_mip_feeder, compute_downstream_counts
from src.network.synthetic_feeder import validate_feeder, export_feeder

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

DEFAULT_GEOJSON   = REPO_ROOT / "data" / "examples" / "lv_feeder_32bus.geojson"
# Centre of the 4×8 grid (r=1, c=4) — 4 grid neighbours, enables C_max=8
DEFAULT_ROOT_LON  = -0.09424
DEFAULT_ROOT_LAT  =  51.50090
DEFAULT_C_MAX     = 8


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GeoDistNet: MST baseline vs. C_max-constrained MIP comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--geojson",     default=str(DEFAULT_GEOJSON))
    p.add_argument("--root-lon",    type=float, default=DEFAULT_ROOT_LON)
    p.add_argument("--root-lat",    type=float, default=DEFAULT_ROOT_LAT)
    p.add_argument("--c-max",       type=int,   default=DEFAULT_C_MAX,
                   help="C_max for the constrained case (baseline always uses n-1).")
    p.add_argument("--households",  type=int,   default=155)
    p.add_argument("--base-kv",     type=float, default=11.0)
    p.add_argument("--base-mva",    type=float, default=1.0)
    p.add_argument("--solver",      default="gurobi")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--crs-metric-epsg", type=int, default=27700,
                   help="EPSG code for metric projection (27700=OSGB36/UK, 32755=UTM-55S/Melbourne).")
    p.add_argument("--no-basemap", action="store_true",
                   help="Disable city basemap tiles in comparison figure.")
    p.add_argument("--basemap-provider", default="CartoDB.PositronNoLabels",
                   help="Tile provider path under contextily.providers (e.g. OpenStreetMap.Mapnik).")
    p.add_argument("--out-dir",     default="data/comparison",
                   help="Root output directory for all results.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------

def _tree_metrics(
    directed_edges: list[tuple[int, int]],
    root: int,
    G_cand: nx.Graph,
) -> dict:
    """
    Structural metrics for a directed rooted tree.

    Returns
    -------
    dict with keys:
        total_length_km, root_branches, max_depth, mean_depth,
        n_leaves, n_branching, max_downstream, mean_downstream
    """
    children: dict[int, list[int]] = {}
    all_nodes: set[int] = {root}
    for u, v in directed_edges:
        children.setdefault(u, []).append(v)
        children.setdefault(v, [])
        all_nodes.update((u, v))

    # BFS depth from root
    depth: dict[int, int] = {root: 0}
    q: deque[int] = deque([root])
    while q:
        nd = q.popleft()
        for ch in children.get(nd, []):
            depth[ch] = depth[nd] + 1
            q.append(ch)

    dc = compute_downstream_counts(directed_edges)
    non_root = [nd for nd in all_nodes if nd != root]
    total_len = sum(G_cand[u][v]["length_km"] for u, v in directed_edges)

    return {
        "total_length_km":  round(total_len, 4),
        "root_branches":    len(children.get(root, [])),
        "max_depth":        max(depth.values()),
        "mean_depth":       round(sum(depth[nd] for nd in non_root) / max(1, len(non_root)), 2),
        "n_leaves":         sum(1 for nd in all_nodes if not children.get(nd)),
        "n_branching":      sum(1 for nd in all_nodes if len(children.get(nd, [])) >= 2),
        "max_downstream":   max(dc.values()) if dc else 0,
        "mean_downstream":  round(sum(dc.values()) / max(1, len(dc)), 2),
    }


# ---------------------------------------------------------------------------
# Side-by-side overlay figure
# ---------------------------------------------------------------------------

def _plot_comparison(
    G_cand: nx.Graph,
    edges_base: list[tuple[int, int]],
    edges_cons: list[tuple[int, int]],
    root: int,
    m_base: dict,
    m_cons: dict,
    c_max: int,
    out_path: pathlib.Path,
    *,
    crs_metric_epsg: int,
    use_basemap: bool,
    basemap_provider: str,
) -> None:
    """
    Two-panel figure: candidate graph (grey) + tree edges (coloured).
    Left panel  — MST baseline.
    Right panel — C_max-constrained MIP.
    """
    C_CAND = "#cccccc"
    C_BASE = "#457b9d"   # steel blue
    C_CONS = "#e63946"   # red
    C_ROOT = "#2a9d8f"   # teal
    LW_CAND, LW_TREE = 0.9, 1.68

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    transformer = pyproj.Transformer.from_crs(crs_metric_epsg, 3857, always_xy=True)

    node_xy_3857 = {
        n: transformer.transform(
            G_cand.nodes[n]["x_km"] * 1000.0,
            G_cand.nodes[n]["y_km"] * 1000.0,
        )
        for n in G_cand.nodes()
    }
    feeder_nodes = {root} | {n for e in edges_base for n in e} | {n for e in edges_cons for n in e}
    xs_all = [node_xy_3857[n][0] for n in feeder_nodes]
    ys_all = [node_xy_3857[n][1] for n in feeder_nodes]
    x_min, x_max = min(xs_all), max(xs_all)
    y_min, y_max = min(ys_all), max(ys_all)
    dx = max(1.0, x_max - x_min)
    dy = max(1.0, y_max - y_min)
    pad_frac = 0.01

    for ax, edges, color, title, metrics in [
        (ax_l, edges_base, C_BASE,
         f"MST baseline  (C_max = n−1 = {G_cand.number_of_nodes() - 1})",
         m_base),
        (ax_r, edges_cons, C_CONS,
         f"Constrained MIP  (C_max = {c_max})",
         m_cons),
    ]:
        # — candidate routes (grey background)
        for u, v in G_cand.edges():
            xu, yu = node_xy_3857[u]
            xv, yv = node_xy_3857[v]
            ax.plot([xu, xv], [yu, yv], color=C_CAND, lw=LW_CAND, zorder=1)

        # — all candidate nodes (light dots)
        ax.scatter(
            [node_xy_3857[n][0] for n in G_cand.nodes()],
            [node_xy_3857[n][1] for n in G_cand.nodes()],
            s=14, color="#bbbbbb", zorder=2,
        )

        # — selected tree edges
        for u, v in edges:
            xu, yu = node_xy_3857[u]
            xv, yv = node_xy_3857[v]
            ax.plot([xu, xv], [yu, yv], color=color, lw=LW_TREE, zorder=3)

        # — tree nodes (excluding root)
        tree_nodes = {n for e in edges for n in e} - {root}
        ax.scatter(
            [node_xy_3857[n][0] for n in tree_nodes],
            [node_xy_3857[n][1] for n in tree_nodes],
            s=30, color=color, zorder=4,
        )

        # — root (substation)
        ax.scatter(
            node_xy_3857[root][0], node_xy_3857[root][1],
            s=200, marker="*", color=C_ROOT, zorder=5,
        )

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

        # — metric annotation box (lower-right of axes)
        txt = (
            f"Root branches : {metrics['root_branches']}\n"
            f"Max downstream: {metrics['max_downstream']}\n"
            f"Max depth     : {metrics['max_depth']}\n"
            f"Total length  : {metrics['total_length_km']:.4f} km"
        )
        ax.text(
            0.97, 0.04, txt,
            transform=ax.transAxes, fontsize=8.5,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.85, ec="#aaaaaa"),
        )

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Web Mercator X [m]", fontsize=9)
        ax.set_xlim(x_min - dx * pad_frac, x_max + dx * pad_frac)
        ax.set_ylim(y_min - dy * pad_frac, y_max + dy * pad_frac)
        yfmt = mticker.ScalarFormatter(useMathText=False)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((6, 6))
        yfmt.set_useOffset(True)
        ax.yaxis.set_major_formatter(yfmt)
        y_off = ax.yaxis.get_offset_text()
        y_off.set_ha("left")
        y_off.set_va("bottom")
        y_off.set_position((0.0, 1.0))
        ax.set_aspect("equal")
        ax.grid(True, lw=0.35, alpha=0.4)

    ax_l.set_ylabel("Web Mercator Y [m]", fontsize=9)

    # Shared legend (below the subplots)
    handles = [
        mpatches.Patch(color=C_CAND, label="Candidate"),
        mpatches.Patch(color=C_BASE, label="MST"),
        mpatches.Patch(color=C_CONS, label=f"MIP C_max={c_max}"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=C_ROOT,
                   markersize=11, label="Substation"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"GeoDistNet — MST vs. capacity-constrained MIP  "
        f"({G_cand.number_of_nodes()} nodes, {G_cand.number_of_edges()} candidate edges)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "=", width: int = 64) -> None:
    print(char * width)


def _run_scenarios(data_dir: pathlib.Path, results_dir: pathlib.Path) -> None:
    """Call run_loading_scenarios.py via subprocess — no logic duplication."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "applications" / "run_loading_scenarios.py"),
        "--data-dir", str(data_dir),
        "--out-dir",  str(results_dir),
    ]
    subprocess.run(cmd, check=True)


def _read_stressed(results_dir: pathlib.Path) -> dict | None:
    """Read the stressed-scenario row from loading_scenarios_summary.csv."""
    csv = results_dir / "loading_scenarios_summary.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    row = df[df["scenario"] == "stressed"]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    out_dir = pathlib.Path(args.out_dir)

    _sep()
    print("GeoDistNet — MST baseline vs. constrained MIP comparison")
    _sep()
    print(f"  GeoJSON  : {args.geojson}")
    print(f"  Root     : lon={args.root_lon}  lat={args.root_lat}")
    print(f"  C_max    : {args.c_max}  (baseline uses n-1 = unconstrained)")
    print(f"  HH       : {args.households}   base_kv={args.base_kv} kV")
    print(f"  CRS      : EPSG:{args.crs_metric_epsg}")
    print(f"  Basemap  : {'ON' if not args.no_basemap else 'OFF'}")
    if not args.no_basemap:
        print(f"  Provider : {args.basemap_provider}")
    print(f"  Out dir  : {out_dir}")

    # ------------------------------------------------------------------
    # Step 1  Build candidate graph
    # ------------------------------------------------------------------
    _sep("-")
    print("Step 1/7  Read GIS + build candidate graph")
    G_raw, root = read_gis_graph(
        args.geojson,
        root_coord=(args.root_lon, args.root_lat),
        crs_metric_epsg=args.crs_metric_epsg,
    )
    G_cand, root = extract_candidate_graph(G_raw, root_node=root, max_edge_km=0.5)
    n = G_cand.number_of_nodes()
    root_deg = len(list(G_cand.neighbors(root)))
    print(f"  Candidate graph: {n} nodes, {G_cand.number_of_edges()} edges")
    print(f"  Root = {root},  root degree = {root_deg}")

    # ------------------------------------------------------------------
    # Step 2  Baseline: unconstrained MIP (C_max = n−1 → MST)
    # ------------------------------------------------------------------
    _sep("-")
    print(f"Step 2/7  Baseline: unconstrained MIP  (C_max = n−1 = {n - 1})")
    params_base = MIPFeederParams(
        C_max=n - 1,
        total_households=args.households,
        base_kv=args.base_kv,
        base_mva=args.base_mva,
        solver=args.solver,
    )
    res_base = solve_mip_feeder(G_cand, root, params_base, seed=args.seed)
    if res_base.status not in ("optimal", "feasible") or res_base.feeder is None:
        print(f"  ERROR: baseline failed with status '{res_base.status}'")
        sys.exit(1)
    print(f"  Status: {res_base.status}   objective: {res_base.objective_km:.4f} km")

    # ------------------------------------------------------------------
    # Step 3  Constrained MIP
    # ------------------------------------------------------------------
    _sep("-")
    print(f"Step 3/7  Constrained MIP  (C_max = {args.c_max})")
    params_cons = MIPFeederParams(
        C_max=args.c_max,
        total_households=args.households,
        base_kv=args.base_kv,
        base_mva=args.base_mva,
        solver=args.solver,
    )
    res_cons = solve_mip_feeder(G_cand, root, params_cons, seed=args.seed)
    if res_cons.status not in ("optimal", "feasible") or res_cons.feeder is None:
        print(f"  ERROR: constrained MIP failed with status '{res_cons.status}'.")
        print(f"  Hint: try a larger --c-max value (minimum feasible ≥ "
              f"ceil({n - 1}/{root_deg}) = {-(-(n - 1) // root_deg)})")
        sys.exit(1)
    print(f"  Status: {res_cons.status}   objective: {res_cons.objective_km:.4f} km")

    # ------------------------------------------------------------------
    # Step 4  Structural metrics + comparison table
    # ------------------------------------------------------------------
    _sep("-")
    print("Step 4/7  Structural metrics")

    m_base = _tree_metrics(res_base.selected_edges, root, G_cand)
    m_cons = _tree_metrics(res_cons.selected_edges, root, G_cand)

    LABELS = {
        "total_length_km":  "Total cable length [km]",
        "root_branches":    "Root branches",
        "max_depth":        "Max depth  [hops from root]",
        "mean_depth":       "Mean depth [hops]",
        "n_leaves":         "Leaf nodes",
        "n_branching":      "Branching nodes  (≥2 children)",
        "max_downstream":   "Max downstream count  (C_max bound)",
        "mean_downstream":  "Mean downstream count",
    }

    col_b = f"MST (C_max={n - 1})"
    col_c = f"MIP (C_max={args.c_max})"

    print(f"\n  {'Metric':<42} {col_b:>18} {col_c:>18}")
    print("  " + "-" * 80)

    rows = []
    for key, label in LABELS.items():
        vb = m_base[key]
        vc = m_cons[key]
        fmt = ".2f" if isinstance(vb, float) else "d"
        print(f"  {label:<42} {vb:>18{fmt}} {vc:>18{fmt}}")
        rows.append({"metric": label, "baseline_mst": vb, f"mip_cmax{args.c_max}": vc})

    tbl_path = out_dir / "structural_metrics.csv"
    tbl_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tbl_path, index=False)
    print(f"\n  Saved: {tbl_path}")

    # ------------------------------------------------------------------
    # Step 5  Export both feeders
    # ------------------------------------------------------------------
    _sep("-")
    print("Step 5/7  Export feeder data")

    dir_base = out_dir / "feeder_mst"
    dir_cons = out_dir / f"feeder_mip_c{args.c_max}"

    validate_feeder(res_base.feeder)
    export_feeder(res_base.feeder, dir_base)
    validate_feeder(res_cons.feeder)
    export_feeder(res_cons.feeder, dir_cons)

    print(f"  MST feeder   → {dir_base}")
    print(f"  MIP feeder   → {dir_cons}")

    # ------------------------------------------------------------------
    # Step 6  Side-by-side overlay figure
    # ------------------------------------------------------------------
    _sep("-")
    print("Step 6/7  Side-by-side overlay figure")

    fig_path = out_dir / "figures" / "mst_vs_mip_overlay.png"
    _plot_comparison(
        G_cand,
        res_base.selected_edges,
        res_cons.selected_edges,
        root,
        m_base, m_cons,
        args.c_max,
        fig_path,
        crs_metric_epsg=args.crs_metric_epsg,
        use_basemap=not args.no_basemap,
        basemap_provider=args.basemap_provider,
    )

    # ------------------------------------------------------------------
    # Step 7  Electrical validation (reuse run_loading_scenarios.py)
    # ------------------------------------------------------------------
    _sep("-")
    print("Step 7/7  Electrical validation  (loading scenarios × 2)")

    res_dir_base = out_dir / "results_mst"
    res_dir_cons = out_dir / f"results_mip_c{args.c_max}"

    for label, data_d, res_d in [
        (f"MST (C_max={n - 1})", dir_base, res_dir_base),
        (f"MIP (C_max={args.c_max})",      dir_cons, res_dir_cons),
    ]:
        print(f"\n  [{label}]")
        _run_scenarios(data_d, res_d)

    # Read and compare stressed-scenario electrical results
    s_base = _read_stressed(res_dir_base)
    s_cons = _read_stressed(res_dir_cons)

    if s_base and s_cons:
        _sep("-")
        print("Electrical comparison  (stressed scenario: 4 kW/hh, 1.3 kVAr/hh)")
        print(f"\n  {'Metric':<35} {f'MST':>14} {f'MIP C_max={args.c_max}':>14}")
        print("  " + "-" * 65)

        elec_rows = []
        for key, label, fmt in [
            ("v_min_pu",            "V_min [pu]",             ".6f"),
            ("v_max_pu",            "V_max [pu]",             ".6f"),
            ("total_losses_kw",     "Total losses [kW]",      ".4f"),
            ("max_branch_load_pct", "Max branch loading [%]", ".2f"),
            ("total_load_kw",        "Total active load [kW]", ".1f"),
        ]:
            vb = s_base.get(key, float("nan"))
            vc = s_cons.get(key, float("nan"))
            print(f"  {label:<35} {vb:>14{fmt}} {vc:>14{fmt}}")
            elec_rows.append({"metric": label, "baseline_mst": vb,
                               f"mip_cmax{args.c_max}": vc})

        elec_csv = out_dir / "electrical_comparison.csv"
        pd.DataFrame(elec_rows).to_csv(elec_csv, index=False)
        print(f"\n  Saved: {elec_csv}")

    _sep()
    print("Comparison complete.")
    print(f"  All outputs in: {out_dir.resolve()}")
    _sep()


if __name__ == "__main__":
    main()
