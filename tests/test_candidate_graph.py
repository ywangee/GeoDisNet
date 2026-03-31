from __future__ import annotations

import math
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import networkx as nx

from src.network.candidate_graph import (
    extract_candidate_graph,
    graph_summary,
    make_grid_candidate_graph,
)


# ===========================================================================
# make_grid_candidate_graph
# ===========================================================================

class TestMakeGridCandidateGraph:

    def test_node_count(self):
        G, _ = make_grid_candidate_graph(n_rows=3, n_cols=4)
        assert G.number_of_nodes() == 12

    def test_node_count_single_row(self):
        G, _ = make_grid_candidate_graph(n_rows=1, n_cols=5)
        assert G.number_of_nodes() == 5

    def test_node_count_single_col(self):
        G, _ = make_grid_candidate_graph(n_rows=4, n_cols=1)
        assert G.number_of_nodes() == 4

    def test_edge_count_3x4(self):
        # 4-connected grid: horizontal edges = n_rows*(n_cols-1),
        #                   vertical edges   = (n_rows-1)*n_cols
        n_rows, n_cols = 3, 4
        G, _ = make_grid_candidate_graph(n_rows=n_rows, n_cols=n_cols)
        expected = n_rows * (n_cols - 1) + (n_rows - 1) * n_cols
        assert G.number_of_edges() == expected

    def test_edge_count_1x1(self):
        G, _ = make_grid_candidate_graph(n_rows=1, n_cols=1)
        assert G.number_of_edges() == 0

    def test_root_is_zero(self):
        _, root = make_grid_candidate_graph(n_rows=4, n_cols=5)
        assert root == 0

    def test_root_node_coordinates(self):
        G, root = make_grid_candidate_graph(n_rows=3, n_cols=3, spacing_km=0.1)
        assert G.nodes[root]["x_km"] == pytest.approx(0.0)
        assert G.nodes[root]["y_km"] == pytest.approx(0.0)

    def test_node_coordinates_row_major(self):
        spacing = 0.15
        G, _ = make_grid_candidate_graph(n_rows=2, n_cols=3, spacing_km=spacing)
        # Node 1 = row 0, col 1
        assert G.nodes[1]["x_km"] == pytest.approx(spacing)
        assert G.nodes[1]["y_km"] == pytest.approx(0.0)
        # Node 3 = row 1, col 0
        assert G.nodes[3]["x_km"] == pytest.approx(0.0)
        assert G.nodes[3]["y_km"] == pytest.approx(spacing)
        # Node 5 = row 1, col 2
        assert G.nodes[5]["x_km"] == pytest.approx(2 * spacing)
        assert G.nodes[5]["y_km"] == pytest.approx(spacing)

    def test_all_edges_have_length_km(self):
        G, _ = make_grid_candidate_graph(n_rows=3, n_cols=3)
        for u, v, data in G.edges(data=True):
            assert "length_km" in data

    def test_edge_lengths_equal_spacing(self):
        spacing = 0.12
        G, _ = make_grid_candidate_graph(n_rows=3, n_cols=4, spacing_km=spacing)
        for _, _, data in G.edges(data=True):
            assert data["length_km"] == pytest.approx(spacing, rel=1e-6)

    def test_all_nodes_have_x_km(self):
        G, _ = make_grid_candidate_graph(n_rows=3, n_cols=3)
        for n, data in G.nodes(data=True):
            assert "x_km" in data
            assert "y_km" in data

    def test_connected(self):
        G, _ = make_grid_candidate_graph(n_rows=4, n_cols=5)
        assert nx.is_connected(G)

    def test_single_node_is_connected(self):
        G, root = make_grid_candidate_graph(n_rows=1, n_cols=1)
        assert G.number_of_nodes() == 1
        assert root == 0

    def test_raises_on_zero_rows(self):
        with pytest.raises(ValueError):
            make_grid_candidate_graph(n_rows=0, n_cols=3)

    def test_raises_on_zero_cols(self):
        with pytest.raises(ValueError):
            make_grid_candidate_graph(n_rows=3, n_cols=0)

    def test_default_params_run(self):
        G, root = make_grid_candidate_graph()
        assert G.number_of_nodes() == 4 * 5
        assert root == 0


# ===========================================================================
# extract_candidate_graph
# ===========================================================================

class TestExtractCandidateGraph:

    @pytest.fixture
    def small_grid(self) -> tuple[nx.Graph, int]:
        """3×3 grid with spacing 0.10 km — used across multiple tests."""
        return make_grid_candidate_graph(n_rows=3, n_cols=3, spacing_km=0.10)

    def test_root_always_zero(self, small_grid):
        G, root_in = small_grid
        G_clean, root_out = extract_candidate_graph(G, root_in)
        assert root_out == 0

    def test_root_exists_in_output(self, small_grid):
        G, root_in = small_grid
        G_clean, root_out = extract_candidate_graph(G, root_in)
        assert root_out in G_clean

    def test_node_count_unchanged_when_no_long_edges(self, small_grid):
        G, root_in = small_grid
        # max_edge_km >> spacing (0.10 km), so no edges are removed
        G_clean, _ = extract_candidate_graph(G, root_in, max_edge_km=1.0)
        assert G_clean.number_of_nodes() == G.number_of_nodes()

    def test_edge_count_unchanged_when_no_long_edges(self, small_grid):
        G, root_in = small_grid
        G_clean, _ = extract_candidate_graph(G, root_in, max_edge_km=1.0)
        assert G_clean.number_of_edges() == G.number_of_edges()

    def test_node_attributes_preserved(self, small_grid):
        G, root_in = small_grid
        G_clean, _ = extract_candidate_graph(G, root_in, max_edge_km=1.0)
        # Root (original node 0 → new node 0) should keep its coordinates
        assert G_clean.nodes[0]["x_km"] == pytest.approx(G.nodes[root_in]["x_km"])
        assert G_clean.nodes[0]["y_km"] == pytest.approx(G.nodes[root_in]["y_km"])

    def test_edge_attributes_preserved(self, small_grid):
        G, root_in = small_grid
        G_clean, _ = extract_candidate_graph(G, root_in, max_edge_km=1.0)
        for _, _, data in G_clean.edges(data=True):
            assert "length_km" in data
            assert data["length_km"] == pytest.approx(0.10, rel=1e-6)

    def test_long_edges_removed(self):
        # Build a graph where some edges are long
        G = nx.Graph()
        G.add_node(0, x_km=0.0, y_km=0.0)
        G.add_node(1, x_km=0.10, y_km=0.0)
        G.add_node(2, x_km=1.00, y_km=0.0)   # far node
        G.add_edge(0, 1, length_km=0.10)
        G.add_edge(1, 2, length_km=0.90)       # long edge
        G_clean, _ = extract_candidate_graph(G, root_node=0, max_edge_km=0.50)
        # Edge 1-2 should be gone; node 2 becomes isolated and not in root component
        assert 0 in G_clean
        assert 1 in G_clean
        assert G_clean.number_of_nodes() == 2
        assert G_clean.number_of_edges() == 1

    def test_disconnected_component_excluded(self):
        # Two separate components — only root's component should survive
        G = nx.Graph()
        # Component A (root)
        G.add_node(0, x_km=0.0, y_km=0.0)
        G.add_node(1, x_km=0.1, y_km=0.0)
        G.add_edge(0, 1, length_km=0.1)
        # Component B (isolated, no connection to root)
        G.add_node(2, x_km=5.0, y_km=5.0)
        G.add_node(3, x_km=5.1, y_km=5.0)
        G.add_edge(2, 3, length_km=0.1)

        G_clean, root_out = extract_candidate_graph(G, root_node=0, max_edge_km=1.0)
        assert root_out == 0
        assert G_clean.number_of_nodes() == 2
        # Nodes 2 and 3 must not appear
        all_node_ids = set(G_clean.nodes())
        assert all_node_ids == {0, 1}

    def test_reindexed_nodes_form_range(self, small_grid):
        G, root_in = small_grid
        G_clean, _ = extract_candidate_graph(G, root_in, max_edge_km=1.0)
        node_ids = sorted(G_clean.nodes())
        assert node_ids == list(range(G_clean.number_of_nodes()))

    def test_output_is_connected(self, small_grid):
        G, root_in = small_grid
        G_clean, _ = extract_candidate_graph(G, root_in, max_edge_km=1.0)
        assert nx.is_connected(G_clean)

    def test_isolated_root_returns_single_node(self):
        # Root with no edges should return a single-node graph
        G = nx.Graph()
        G.add_node(0, x_km=0.0, y_km=0.0)
        G.add_node(1, x_km=2.0, y_km=0.0)
        G.add_edge(0, 1, length_km=2.0)
        # max_edge_km=0.5 removes the only edge → root isolated
        G_clean, root_out = extract_candidate_graph(G, root_node=0, max_edge_km=0.5)
        assert root_out == 0
        assert G_clean.number_of_nodes() == 1
        assert G_clean.number_of_edges() == 0

    def test_raises_when_root_not_in_graph(self):
        G, _ = make_grid_candidate_graph(n_rows=2, n_cols=2)
        with pytest.raises(ValueError, match="root_node"):
            extract_candidate_graph(G, root_node=999)

    def test_different_original_root_still_maps_to_zero(self):
        # Build a path graph with original root = node 5
        G = nx.path_graph(6)
        for n in G.nodes():
            G.nodes[n]["x_km"] = float(n) * 0.1
            G.nodes[n]["y_km"] = 0.0
        for u, v in G.edges():
            G[u][v]["length_km"] = 0.1

        G_clean, root_out = extract_candidate_graph(G, root_node=5, max_edge_km=1.0)
        assert root_out == 0
        # The original root (node 5) should now be node 0 in the clean graph
        assert G_clean.nodes[0]["x_km"] == pytest.approx(5 * 0.1)

    def test_large_grid_preserves_structure(self):
        G, root_in = make_grid_candidate_graph(n_rows=10, n_cols=10, spacing_km=0.1)
        G_clean, root_out = extract_candidate_graph(G, root_in, max_edge_km=1.0)
        assert G_clean.number_of_nodes() == 100
        assert root_out == 0
        assert nx.is_connected(G_clean)


# ===========================================================================
# graph_summary
# ===========================================================================

class TestGraphSummary:

    def test_returns_dict(self):
        G, root = make_grid_candidate_graph(n_rows=2, n_cols=3)
        s = graph_summary(G, root)
        assert isinstance(s, dict)

    def test_node_and_edge_counts(self):
        G, root = make_grid_candidate_graph(n_rows=2, n_cols=3)
        s = graph_summary(G, root)
        assert s["n_nodes"] == 6
        # horizontal: 2*2=4, vertical: 1*3=3  → 7
        assert s["n_edges"] == 7

    def test_is_connected_true(self):
        G, root = make_grid_candidate_graph(n_rows=3, n_cols=3)
        s = graph_summary(G, root)
        assert s["is_connected"] is True

    def test_length_stats(self):
        spacing = 0.10
        G, root = make_grid_candidate_graph(n_rows=3, n_cols=3, spacing_km=spacing)
        s = graph_summary(G, root)
        assert s["length_km_min"] == pytest.approx(spacing, rel=1e-4)
        assert s["length_km_max"] == pytest.approx(spacing, rel=1e-4)
        n_edges = G.number_of_edges()
        assert s["length_km_total"] == pytest.approx(n_edges * spacing, rel=1e-4)

    def test_empty_graph_lengths_are_none(self):
        G = nx.Graph()
        G.add_node(0, x_km=0.0, y_km=0.0)
        s = graph_summary(G, root_node=0)
        assert s["length_km_min"] is None
        assert s["length_km_max"] is None
        assert s["length_km_total"] is None
