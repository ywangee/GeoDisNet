from __future__ import annotations

import json
import math
import pathlib
from typing import Any

import networkx as nx

try:
    import pyproj
    _PYPROJ_AVAILABLE = True
except ImportError:                    # pragma: no cover
    _PYPROJ_AVAILABLE = False

try:
    import geopandas as gpd
    _GEOPANDAS_AVAILABLE = True
except ImportError:
    _GEOPANDAS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def read_gis_graph(
    filepath: str | pathlib.Path,
    root_coord: tuple[float, float],
    *,
    crs_input_epsg: int = 4326,
    crs_metric_epsg: int = 27700,
    snap_tol_m: float = 5.0,
    max_edge_km: float | None = None,
) -> tuple[nx.Graph, int]:
    """
    Read a GIS file of LineString features and build a candidate graph.

    Parameters
    ----------
    filepath : str or Path
        Path to a GeoJSON file (always supported) or a shapefile/GeoPackage
        (requires geopandas).
    root_coord : (lon, lat) or (x, y)
        Coordinates of the substation/root node **in the input CRS**
        (e.g. WGS-84 longitude, latitude for EPSG:4326).
    crs_input_epsg : int
        EPSG code of the input coordinates.  Default 4326 (WGS-84).
    crs_metric_epsg : int
        EPSG code of the projected metric CRS used internally.
        Default 27700 (OSGB36 / British National Grid).
    snap_tol_m : float
        Grid resolution [m] used to snap near-coincident endpoints.
        Set to 0 to disable snapping.
    max_edge_km : float or None
        If given, edges longer than this value are dropped before returning.
        This mirrors the ``max_edge_km`` parameter of ``extract_candidate_graph``.

    Returns
    -------
    G : nx.Graph
        Candidate graph with ``x_km``, ``y_km`` on every node and
        ``length_km`` on every edge.  Node IDs are consecutive integers
        starting at 0; root is node 0.
    root : int
        Always 0.

    Raises
    ------
    ImportError
        If pyproj is not installed.
    ValueError
        If the file contains no usable LineString features, or if
        ``root_coord`` does not project to a finite metric coordinate.
    FileNotFoundError
        If ``filepath`` does not exist.
    """
    if not _PYPROJ_AVAILABLE:
        raise ImportError(
            "pyproj is required for read_gis_graph.  "
            "Install it with: pip install pyproj"
        )

    filepath = pathlib.Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"GIS file not found: {filepath}")

    # --- Load raw line endpoints ---
    suffix = filepath.suffix.lower()
    if suffix == ".geojson" or suffix == ".json":
        raw_segments = _load_geojson_features(filepath)
    elif _GEOPANDAS_AVAILABLE:
        raw_segments = _load_via_geopandas(filepath)
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'.  "
            "Install geopandas for shapefile/GeoPackage support."
        )

    if not raw_segments:
        raise ValueError(f"No usable LineString features found in {filepath}.")

    # --- Set up coordinate transformer ---
    transformer = pyproj.Transformer.from_crs(
        crs_input_epsg, crs_metric_epsg, always_xy=True
    )

    # --- Project + snap + build graph ---
    G = _build_graph_from_segments(raw_segments, transformer, snap_tol_m)

    # --- Identify root: node nearest to root_coord ---
    rx_m, ry_m = transformer.transform(root_coord[0], root_coord[1])
    if not math.isfinite(rx_m) or not math.isfinite(ry_m):
        raise ValueError(
            f"root_coord {root_coord!r} did not project to a finite metric "
            f"coordinate (got {rx_m}, {ry_m}).  Check crs_input_epsg."
        )

    root_node = _nearest_node(G, rx_m * 1e-3, ry_m * 1e-3)

    # --- Drop long edges if requested ---
    if max_edge_km is not None:
        to_remove = [
            (u, v) for u, v, d in G.edges(data=True)
            if d["length_km"] > max_edge_km
        ]
        G.remove_edges_from(to_remove)
        # Remove isolated nodes except the root (it anchors the feeder)
        isolated = [n for n in nx.isolates(G) if n != root_node]
        G.remove_nodes_from(isolated)

    # --- Re-index so that root → 0 ---
    G, root_out = _reindex_root_first(G, root_node)

    return G, root_out


# ---------------------------------------------------------------------------
# GeoJSON loader (no pyproj dependency — testable in isolation)
# ---------------------------------------------------------------------------

def _load_geojson_features(
    filepath: str | pathlib.Path,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Parse a GeoJSON FeatureCollection and return a list of
    ``(start_coord, end_coord)`` pairs for every LineString feature.

    Only the first and last coordinate of each LineString are used (the
    start and end nodes of the cable segment), matching how topology is
    typically encoded in LV feeder GIS data.

    Multi-part LineStrings with intermediate vertices are supported: only
    the first and last point are retained as graph nodes.

    Features whose geometry type is not ``"LineString"`` are silently skipped.

    Parameters
    ----------
    filepath : str or Path
        Path to a GeoJSON file.

    Returns
    -------
    list of ((lon_start, lat_start), (lon_end, lat_end))
    """
    filepath = pathlib.Path(filepath)
    with filepath.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    features = data.get("features", [])
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for feat in features:
        geom = feat.get("geometry") or {}
        if geom.get("type") != "LineString":
            continue
        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue
        start = (float(coords[0][0]), float(coords[0][1]))
        end   = (float(coords[-1][0]), float(coords[-1][1]))
        segments.append((start, end))

    return segments


# ---------------------------------------------------------------------------
# Snapping helper (no pyproj dependency — testable in isolation)
# ---------------------------------------------------------------------------

def _snap(x_m: float, y_m: float, snap_tol_m: float) -> tuple[float, float]:
    """
    Round metric coordinates to the nearest ``snap_tol_m`` grid.

    Parameters
    ----------
    x_m, y_m : float
        Metric coordinates [m].
    snap_tol_m : float
        Grid resolution [m].  If ≤ 0 the coordinates are returned unchanged.

    Returns
    -------
    (snapped_x_m, snapped_y_m)
    """
    if snap_tol_m <= 0.0:
        return (x_m, y_m)
    return (
        round(x_m / snap_tol_m) * snap_tol_m,
        round(y_m / snap_tol_m) * snap_tol_m,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_graph_from_segments(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    transformer: "pyproj.Transformer",
    snap_tol_m: float,
) -> nx.Graph:
    """
    Project, snap, and assemble a NetworkX graph from raw (lon, lat) segments.

    Each unique snapped metric coordinate becomes a node with attributes
    ``x_km`` and ``y_km``.  Parallel edges (same snapped endpoints) are
    silently deduplicated, keeping the shorter one.
    """
    # Map snapped metric coord → node id
    coord_to_node: dict[tuple[float, float], int] = {}
    node_xy: dict[int, tuple[float, float]] = {}   # node_id → (x_m, y_m)

    def _get_or_add(x_m: float, y_m: float) -> int:
        key = _snap(x_m, y_m, snap_tol_m)
        if key not in coord_to_node:
            nid = len(coord_to_node)
            coord_to_node[key] = nid
            node_xy[nid] = key
        return coord_to_node[key]

    edges: list[tuple[int, int, float]] = []  # (u, v, length_km)

    for (lon0, lat0), (lon1, lat1) in segments:
        x0_m, y0_m = transformer.transform(lon0, lat0)
        x1_m, y1_m = transformer.transform(lon1, lat1)

        u = _get_or_add(x0_m, y0_m)
        v = _get_or_add(x1_m, y1_m)

        if u == v:
            continue  # degenerate (snapped to same node)

        dx_m = node_xy[u][0] - node_xy[v][0]
        dy_m = node_xy[u][1] - node_xy[v][1]
        length_km = math.sqrt(dx_m ** 2 + dy_m ** 2) * 1e-3
        edges.append((u, v, length_km))

    G = nx.Graph()
    for nid, (x_m, y_m) in node_xy.items():
        G.add_node(nid, x_km=x_m * 1e-3, y_km=y_m * 1e-3)

    for u, v, length_km in edges:
        if G.has_edge(u, v):
            # Keep shorter parallel edge
            if length_km < G[u][v]["length_km"]:
                G[u][v]["length_km"] = length_km
        else:
            G.add_edge(u, v, length_km=length_km)

    return G


def _nearest_node(G: nx.Graph, x_km: float, y_km: float) -> int:
    """Return the node in G closest (Euclidean) to (x_km, y_km)."""
    best_node = None
    best_dist2 = float("inf")
    for nid, data in G.nodes(data=True):
        dx = data["x_km"] - x_km
        dy = data["y_km"] - y_km
        d2 = dx * dx + dy * dy
        if d2 < best_dist2:
            best_dist2 = d2
            best_node = nid
    if best_node is None:
        raise ValueError("Graph has no nodes.")
    return best_node


def _reindex_root_first(G: nx.Graph, root_node: int) -> tuple[nx.Graph, int]:
    """
    Relabel nodes so that ``root_node`` becomes 0 and all others are
    consecutive integers 1…n-1 (preserving relative order).

    Returns the relabelled graph and the new root index (always 0).
    """
    old_nodes = [root_node] + [n for n in sorted(G.nodes()) if n != root_node]
    mapping = {old: new for new, old in enumerate(old_nodes)}
    G_new = nx.relabel_nodes(G, mapping)
    return G_new, 0


def _load_via_geopandas(
    filepath: pathlib.Path,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Load line endpoints via geopandas (for .shp, .gpkg, etc.).

    Returns the same ``(start_coord, end_coord)`` list as
    ``_load_geojson_features``, with coordinates in the file's native CRS.
    """
    gdf = gpd.read_file(filepath)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for geom in gdf.geometry:
        if geom is None or geom.geom_type != "LineString":
            continue
        coords = list(geom.coords)
        if len(coords) < 2:
            continue
        start = (float(coords[0][0]), float(coords[0][1]))
        end   = (float(coords[-1][0]), float(coords[-1][1]))
        segments.append((start, end))
    return segments
