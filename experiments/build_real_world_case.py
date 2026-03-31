

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is required.  Install with: pip install requests")
    sys.exit(1)

import networkx as nx

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Default study-area constants
# ---------------------------------------------------------------------------

STUDY_AREA   = "Bethnal Green, East London (E2)"
# Bounding box: lon_min, lat_min, lon_max, lat_max
#   Width : 0.01° × ~69 500 m/°  ≈ 695 m
#   Height: 0.006° × 111 320 m/°  ≈ 668 m
DEFAULT_BBOX = (-0.0800, 51.5140, -0.0700, 51.5200)
# Substation: intersection near centroid of largest connected component
DEFAULT_ROOT = (-0.0760, 51.5169)
DEFAULT_OUT  = REPO_ROOT / "data" / "examples" / "osm_bethnal_green.geojson"

OVERPASS_URL  = "https://overpass-api.de/api/interpreter"
# Road types used as cable-route candidates in UK residential distribution
HIGHWAY_TAGS  = "residential|tertiary|unclassified|living_street"
# Snap tolerance in WGS-84 degrees (~4 m at London latitude)
SNAP_DEG      = 4e-5
# Warn if simplified graph exceeds this node count (MIP may be slow)
MAX_NODES_WARN = 120


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download OSM roads and export GeoJSON for GeoDistNet Path C.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bbox", nargs=4, type=float,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
                   default=list(DEFAULT_BBOX),
                   help="Bounding box in WGS-84.")
    p.add_argument("--root-lon", type=float, default=DEFAULT_ROOT[0])
    p.add_argument("--root-lat", type=float, default=DEFAULT_ROOT[1])
    p.add_argument("--output",   default=str(DEFAULT_OUT),
                   help="Output GeoJSON path.")
    p.add_argument("--voltage-kv", type=float, default=11.0,
                   help="voltage_kv property written to every feature.")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip final gis_reader validation step.")
    p.add_argument("--crs-metric-epsg", type=int, default=27700,
                   help="EPSG code for metric projection (27700=OSGB36/UK, 32755=UTM-55S/Melbourne).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1 — Overpass query
# ---------------------------------------------------------------------------

def _query_overpass(bbox: tuple[float, float, float, float]) -> list[dict]:
    """
    Query Overpass API for road ways in bbox.

    Returns the list of OSM 'way' elements with embedded geometry.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    query = (
        f"[out:json][timeout:30];\n"
        f'way["highway"~"^({HIGHWAY_TAGS})$"]\n'
        f"({lat_min},{lon_min},{lat_max},{lon_max});\n"
        f"out geom;"
    )
    print(f"  Querying Overpass API …", end="", flush=True)
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=60,
            headers={"User-Agent": "GeoDistNet/1.0 (research)"},
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"\n  ERROR: Overpass API request failed: {exc}")
        sys.exit(1)

    data = resp.json()
    elements = [e for e in data.get("elements", []) if e.get("type") == "way"]
    print(f"  {len(elements)} ways  ({time.perf_counter() - t0:.1f} s)")
    return elements


# ---------------------------------------------------------------------------
# Step 2 — Build raw graph
# ---------------------------------------------------------------------------

def _snap_coord(lon: float, lat: float) -> tuple[float, float]:
    """Snap to SNAP_DEG grid to merge near-coincident OSM nodes."""
    return (
        round(lon / SNAP_DEG) * SNAP_DEG,
        round(lat / SNAP_DEG) * SNAP_DEG,
    )


def _build_raw_graph(elements: list[dict]) -> nx.Graph:
    """
    Build a NetworkX graph from OSM way elements.

    Nodes are snapped (lon, lat) tuples; edges connect consecutive OSM
    nodes along each way.  Self-loops and duplicate edges are dropped.
    """
    G: nx.Graph = nx.Graph()

    for elem in elements:
        geom = elem.get("geometry", [])
        if len(geom) < 2:
            continue

        nodes = [_snap_coord(pt["lon"], pt["lat"]) for pt in geom]

        for coord in nodes:
            if coord not in G:
                G.add_node(coord, lon=coord[0], lat=coord[1])

        for a, b in zip(nodes[:-1], nodes[1:]):
            if a != b and not G.has_edge(a, b):
                G.add_edge(a, b)

    return G


# ---------------------------------------------------------------------------
# Step 3 — Simplify to intersections / dead-ends
# ---------------------------------------------------------------------------

def _simplify(G: nx.Graph) -> nx.Graph:
    """
    Remove degree-2 intermediate nodes, collapsing linear chains into
    single edges.

    Only degree ≠ 2 nodes are kept: junctions (degree ≥ 3) and
    dead-ends (degree = 1).  This produces a graph where every node is
    topologically meaningful, matching what gis_reader.py expects.
    """
    important: set = {n for n in G.nodes() if G.degree(n) != 2}

    if not important:
        # Degenerate: all nodes degree-2 (a single cycle).  Keep as-is.
        return G.copy()

    G_out: nx.Graph = nx.Graph()
    for n in important:
        G_out.add_node(n, **G.nodes[n])

    visited: set[frozenset] = set()

    for start in important:
        for nb in list(G.neighbors(start)):
            step_key = frozenset([start, nb])
            if step_key in visited:
                continue
            visited.add(step_key)

            # Trace path through degree-2 chain until next important node
            prev, cur = start, nb
            while cur not in important:
                nexts = [x for x in G.neighbors(cur) if x != prev]
                if not nexts:
                    break                      # dangling chain — stop here
                prev, cur = cur, nexts[0]

            end = cur
            if end == start:
                continue                       # collapsed cycle — skip

            if not G_out.has_node(end):
                G_out.add_node(end, **G.nodes[end])
            if not G_out.has_edge(start, end):
                G_out.add_edge(start, end)

    return G_out


# ---------------------------------------------------------------------------
# Step 4 — Largest connected component containing root
# ---------------------------------------------------------------------------

def _root_component(G: nx.Graph, root_coord: tuple[float, float]) -> nx.Graph:
    """
    Return the connected component that contains the node closest to
    root_coord.  If the graph is already connected, returns G unchanged.
    """
    if nx.is_connected(G):
        return G

    # Find nearest node to root_coord
    best, best_d2 = None, float("inf")
    for n, data in G.nodes(data=True):
        dx = data["lon"] - root_coord[0]
        dy = data["lat"] - root_coord[1]
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2, best = d2, n

    for comp in nx.connected_components(G):
        if best in comp:
            return G.subgraph(comp).copy()

    # Fallback: largest component
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


# ---------------------------------------------------------------------------
# Step 5 — GeoJSON export
# ---------------------------------------------------------------------------

def _to_geojson(
    G: nx.Graph,
    bbox: tuple[float, float, float, float],
    root_coord: tuple[float, float],
    voltage_kv: float,
) -> dict:
    """
    Serialise the simplified graph as a GeoJSON FeatureCollection.

    Each edge becomes a 2-point LineString between its endpoint nodes.
    The gis_reader uses only first/last coordinates of each LineString,
    so 2-point features are exact.
    """
    lon_min, lat_min, lon_max, lat_max = bbox
    features = []
    for idx, (u, v) in enumerate(G.edges()):
        lon1, lat1 = G.nodes[u]["lon"], G.nodes[u]["lat"]
        lon2, lat2 = G.nodes[v]["lon"], G.nodes[v]["lat"]
        features.append({
            "type":       "Feature",
            "id":         f"seg_{idx}",
            "properties": {"voltage_kv": voltage_kv},
            "geometry": {
                "type":        "LineString",
                "coordinates": [
                    [round(lon1, 7), round(lat1, 7)],
                    [round(lon2, 7), round(lat2, 7)],
                ],
            },
        })

    return {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "description": (
            f"GeoDistNet real-world candidate graph — {STUDY_AREA}.  "
            f"Derived from OpenStreetMap (© OpenStreetMap contributors, ODbL).  "
            f"Bounding box: lon [{lon_min}, {lon_max}], lat [{lat_min}, {lat_max}].  "
            f"Substation root approx: lon={root_coord[0]}, lat={root_coord[1]}.  "
            f"Highway types: {HIGHWAY_TAGS}.  "
            f"Nodes are intersections/dead-ends after degree-2 simplification."
        ),
        "features": features,
    }


# ---------------------------------------------------------------------------
# Haversine length helper (for stats only)
# ---------------------------------------------------------------------------

def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    bbox        = tuple(args.bbox)
    root_coord  = (args.root_lon, args.root_lat)
    output_path = pathlib.Path(args.output)

    print("=" * 62)
    print("GeoDistNet — real-world OSM candidate graph builder")
    print("=" * 62)
    print(f"  Study area : {STUDY_AREA}")
    print(f"  Bbox       : lon [{bbox[0]}, {bbox[2]}]  lat [{bbox[1]}, {bbox[3]}]")
    print(f"  Root coord : lon={root_coord[0]}  lat={root_coord[1]}")
    print(f"  Output     : {output_path}")

    # ------------------------------------------------------------------
    # Step 1: Download OSM roads
    # ------------------------------------------------------------------
    print("\nStep 1/5  Download OSM road network")
    elements = _query_overpass(bbox)
    if not elements:
        print("  ERROR: no OSM ways returned.  Check bbox and highway filter.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Build raw graph
    # ------------------------------------------------------------------
    print("Step 2/5  Build raw NetworkX graph")
    G_raw = _build_raw_graph(elements)
    print(f"  Raw graph  : {G_raw.number_of_nodes()} nodes, "
          f"{G_raw.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # Step 3: Simplify (remove degree-2 nodes)
    # ------------------------------------------------------------------
    print("Step 3/5  Simplify to intersections / dead-ends")
    G_simple = _simplify(G_raw)
    print(f"  Simplified : {G_simple.number_of_nodes()} nodes, "
          f"{G_simple.number_of_edges()} edges")

    if G_simple.number_of_nodes() < 3:
        print("  ERROR: fewer than 3 nodes after simplification.  "
              "Widen the bounding box or loosen highway filter.")
        sys.exit(1)

    if G_simple.number_of_nodes() > MAX_NODES_WARN:
        print(f"  WARNING: {G_simple.number_of_nodes()} nodes — MIP may be slow.  "
              "Consider reducing the bounding box.")

    # ------------------------------------------------------------------
    # Step 4: Keep root's connected component
    # ------------------------------------------------------------------
    print("Step 4/5  Extract root's connected component")
    n_comps = nx.number_connected_components(G_simple)
    G_conn  = _root_component(G_simple, root_coord)
    print(f"  Components : {n_comps}  →  kept component: "
          f"{G_conn.number_of_nodes()} nodes, {G_conn.number_of_edges()} edges")

    # Edge-length stats
    lengths = [
        _haversine_km(
            G_conn.nodes[u]["lon"], G_conn.nodes[u]["lat"],
            G_conn.nodes[v]["lon"], G_conn.nodes[v]["lat"],
        )
        for u, v in G_conn.edges()
    ]
    if lengths:
        print(f"  Edge length: min={min(lengths):.4f} km  "
              f"max={max(lengths):.4f} km  "
              f"mean={sum(lengths)/len(lengths):.4f} km")

    # ------------------------------------------------------------------
    # Step 5: Export GeoJSON
    # ------------------------------------------------------------------
    print("Step 5/5  Export GeoJSON")
    geojson = _to_geojson(G_conn, bbox, root_coord, args.voltage_kv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")
    print(f"  Written    : {output_path}  "
          f"({output_path.stat().st_size // 1024} KB, "
          f"{len(geojson['features'])} features)")

    # ------------------------------------------------------------------
    # Validation: run through gis_reader to confirm compatibility
    # ------------------------------------------------------------------
    if not args.no_validate:
        print("\nValidation  gis_reader.read_gis_graph compatibility check")
        try:
            from src.network.gis_reader import read_gis_graph
            from src.network.candidate_graph import extract_candidate_graph, graph_summary
            G_check, r = read_gis_graph(
                output_path, root_coord=root_coord,
                crs_metric_epsg=args.crs_metric_epsg,
                snap_tol_m=10.0,
            )
            G_cand, r = extract_candidate_graph(G_check, r, max_edge_km=0.5)
            s = graph_summary(G_cand, r)
            print(f"  read_gis_graph : {G_check.number_of_nodes()} nodes, "
                  f"{G_check.number_of_edges()} edges  ✓")
            print(f"  candidate graph: {s['n_nodes']} nodes, {s['n_edges']} edges  "
                  f"connected={s['is_connected']}  ✓")
            print(f"  Edge range     : {s['length_km_min']:.4f} – "
                  f"{s['length_km_max']:.4f} km")
        except Exception as exc:
            print(f"  WARNING: validation raised {type(exc).__name__}: {exc}")
            print("  The GeoJSON was written; check manually before running pipeline.")

    print("\n" + "=" * 62)
    print("Done.  Run Path C pipeline with:")
    print(f"  python experiments/run_path_c_pipeline.py \\")
    print(f"      --geojson {output_path} \\")
    print(f"      --root-lon {root_coord[0]} --root-lat {root_coord[1]} \\")
    print(f"      --base-kv 11.0 --base-mva 1.0")
    print("=" * 62)


if __name__ == "__main__":
    main()
