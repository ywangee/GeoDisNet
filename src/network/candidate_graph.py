from __future__ import annotations

import math
from typing import Any

import networkx as nx


# ---------------------------------------------------------------------------
# Programmatic graph builder  (used for tests, demos, and paper figures)
# ---------------------------------------------------------------------------

def make_grid_candidate_graph(
    n_rows: int = 4,
    n_cols: int = 5,
    spacing_km: float = 0.10,
) -> tuple[nx.Graph, int]:
    """
    Build a regular 2-D grid candidate graph.

    Nodes are arranged in a (n_rows × n_cols) grid with 4-connectivity
    (up, down, left, right — no diagonals).  The root bus is node 0,
    placed at the top-left corner (row 0, col 0).

    This function does not require any GIS data and is suitable for
    unit tests, algorithm development, and paper illustration cases.

    Parameters
    ----------
    n_rows : int
        Number of rows (≥ 1).
    n_cols : int
        Number of columns (≥ 1).
    spacing_km : float
        Distance between adjacent nodes [km].

    Returns
    -------
    G : nx.Graph
        Grid graph with ``x_km``, ``y_km`` node attributes and
        ``length_km`` edge attributes.
    root_node : int
        Index of the root (substation) bus — always 0.

    Raises
    ------
    ValueError
        If n_rows < 1 or n_cols < 1.
    """
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must each be ≥ 1.")

    G: nx.Graph = nx.Graph()

    # ------------------------------------------------------------------
    # Add nodes  (row-major order: node id = row * n_cols + col)
    # x increases right (+col), y increases downward (+row)
    # Root = node 0 = top-left = (row=0, col=0)
    # ------------------------------------------------------------------
    for row in range(n_rows):
        for col in range(n_cols):
            node_id = row * n_cols + col
            G.add_node(
                node_id,
                x_km=round(col * spacing_km, 8),
                y_km=round(row * spacing_km, 8),
            )

    # ------------------------------------------------------------------
    # Add edges — horizontal and vertical neighbours only
    # ------------------------------------------------------------------
    for row in range(n_rows):
        for col in range(n_cols):
            u = row * n_cols + col
            # Right neighbour
            if col + 1 < n_cols:
                v = row * n_cols + (col + 1)
                G.add_edge(u, v, length_km=round(spacing_km, 8))
            # Down neighbour
            if row + 1 < n_rows:
                v = (row + 1) * n_cols + col
                G.add_edge(u, v, length_km=round(spacing_km, 8))

    return G, 0


# ---------------------------------------------------------------------------
# Graph preprocessing  (used after GIS ingest or programmatic construction)
# ---------------------------------------------------------------------------

def extract_candidate_graph(
    G: nx.Graph,
    root_node: int,
    max_edge_km: float = 0.5,
) -> tuple[nx.Graph, int]:
    """
    Prune and re-index a raw candidate graph for MIP input.

    Steps
    -----
    1. Remove all edges with ``length_km > max_edge_km``.
    2. Keep only the connected component that contains ``root_node``.
    3. Re-index nodes to 0 … n−1 with root → 0 and all other nodes in
       ascending order of their original IDs.

    Node attributes (``x_km``, ``y_km``) and edge attributes
    (``length_km``) are preserved exactly.

    Parameters
    ----------
    G : nx.Graph
        Input candidate graph.  Must have ``length_km`` on every edge.
    root_node : int
        Node ID of the substation (root) in ``G``.
    max_edge_km : float
        Edges longer than this threshold are removed before pruning.

    Returns
    -------
    G_clean : nx.Graph
        Pruned, re-indexed graph.  Root is always node 0.
    new_root : int
        Always 0 — provided for API symmetry with ``make_grid_candidate_graph``.

    Raises
    ------
    ValueError
        If ``root_node`` is not present in ``G``, or if it becomes
        isolated after edge removal (no component to return).
    """
    if root_node not in G:
        raise ValueError(
            f"root_node {root_node!r} is not a node in the supplied graph."
        )

    # ------------------------------------------------------------------
    # Step 1 — remove long edges
    # ------------------------------------------------------------------
    long_edges = [
        (u, v)
        for u, v, data in G.edges(data=True)
        if data.get("length_km", 0.0) > max_edge_km
    ]
    G_filtered = G.copy()
    G_filtered.remove_edges_from(long_edges)

    # ------------------------------------------------------------------
    # Step 2 — keep the connected component containing root_node
    # ------------------------------------------------------------------
    if root_node not in G_filtered or G_filtered.degree(root_node) == 0:
        # Root is isolated — check it still exists as a node
        if root_node not in G_filtered:
            raise ValueError(
                f"root_node {root_node!r} was removed during edge filtering."
            )
        # Root exists but has no edges — return a single-node graph
        G_root_only = nx.Graph()
        G_root_only.add_node(0, **G_filtered.nodes[root_node])
        return G_root_only, 0

    component_nodes = nx.node_connected_component(G_filtered, root_node)
    G_sub = G_filtered.subgraph(component_nodes).copy()

    # ------------------------------------------------------------------
    # Step 3 — re-index: root → 0, rest in sorted order
    # ------------------------------------------------------------------
    other_nodes = sorted(n for n in G_sub.nodes() if n != root_node)
    ordered = [root_node] + other_nodes           # root always first
    mapping = {old: new for new, old in enumerate(ordered)}
    G_clean = nx.relabel_nodes(G_sub, mapping)

    return G_clean, 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def graph_summary(G: nx.Graph, root_node: int = 0) -> dict[str, Any]:
    """
    Return a concise summary dict for a candidate graph.

    Useful for experiment scripts and paper result tables.
    """
    edge_lengths = [d["length_km"] for _, _, d in G.edges(data=True)]
    return {
        "n_nodes":        G.number_of_nodes(),
        "n_edges":        G.number_of_edges(),
        "root_node":      root_node,
        "is_connected":   nx.is_connected(G),
        "length_km_min":  round(min(edge_lengths), 4) if edge_lengths else None,
        "length_km_max":  round(max(edge_lengths), 4) if edge_lengths else None,
        "length_km_total":round(sum(edge_lengths), 4) if edge_lengths else None,
    }
