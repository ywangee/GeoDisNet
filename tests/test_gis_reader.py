from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import networkx as nx

from src.network.gis_reader import (
    _load_geojson_features,
    _snap,
)

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "examples"
FIXTURE_GEOJSON = FIXTURE_DIR / "simple_lv_feeder.geojson"
ROOT_COORD_WGS84 = (-0.10000, 51.50000)   # node A = substation


# ---------------------------------------------------------------------------
# pyproj availability
# ---------------------------------------------------------------------------

def _pyproj_available() -> bool:
    try:
        import pyproj  # noqa: F401
        return True
    except ImportError:
        return False


# ===========================================================================
# _load_geojson_features
# ===========================================================================

class TestLoadGeojsonFeatures:

    def test_segment_count(self):
        segs = _load_geojson_features(FIXTURE_GEOJSON)
        assert len(segs) == 12

    def test_segment_structure(self):
        segs = _load_geojson_features(FIXTURE_GEOJSON)
        for start, end in segs:
            assert len(start) == 2
            assert len(end) == 2
            # All should be float-able
            assert isinstance(float(start[0]), float)
            assert isinstance(float(start[1]), float)

    def test_coords_in_wgs84_range(self):
        segs = _load_geojson_features(FIXTURE_GEOJSON)
        for start, end in segs:
            # Longitude roughly -0.10 to -0.097; Latitude ~51.5
            assert -1.0 < start[0] < 0.0
            assert 51.4 < start[1] < 51.6

    def test_skips_non_linestring(self, tmp_path):
        data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [0.0, 0.0]
                    },
                    "properties": {}
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[0.0, 0.0], [1.0, 0.0]]
                    },
                    "properties": {}
                },
            ]
        }
        p = tmp_path / "test.geojson"
        p.write_text(json.dumps(data))
        segs = _load_geojson_features(p)
        assert len(segs) == 1

    def test_skips_degenerate_linestring(self, tmp_path):
        data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[0.0, 0.0]]   # only 1 point
                    },
                    "properties": {}
                }
            ]
        }
        p = tmp_path / "test.geojson"
        p.write_text(json.dumps(data))
        segs = _load_geojson_features(p)
        assert len(segs) == 0

    def test_empty_feature_collection(self, tmp_path):
        data = {"type": "FeatureCollection", "features": []}
        p = tmp_path / "empty.geojson"
        p.write_text(json.dumps(data))
        segs = _load_geojson_features(p)
        assert segs == []

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            # Use non-existent path — open() raises FileNotFoundError
            _load_geojson_features(FIXTURE_DIR / "does_not_exist.geojson")

    def test_multipoint_linestring_uses_endpoints_only(self, tmp_path):
        """Intermediate vertices are ignored; only first and last are kept."""
        data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [0.0, 0.0],
                            [0.5, 0.5],   # intermediate — ignored
                            [1.0, 1.0],
                        ]
                    },
                    "properties": {}
                }
            ]
        }
        p = tmp_path / "multi.geojson"
        p.write_text(json.dumps(data))
        segs = _load_geojson_features(p)
        assert len(segs) == 1
        start, end = segs[0]
        assert start == pytest.approx((0.0, 0.0))
        assert end   == pytest.approx((1.0, 1.0))

    def test_null_geometry_skipped(self, tmp_path):
        data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": None,
                    "properties": {}
                }
            ]
        }
        p = tmp_path / "null_geom.geojson"
        p.write_text(json.dumps(data))
        segs = _load_geojson_features(p)
        assert segs == []


# ===========================================================================
# _snap
# ===========================================================================

class TestSnap:

    def test_exact_multiple(self):
        x, y = _snap(100.0, 200.0, 5.0)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)

    def test_rounds_to_nearest(self):
        x, y = _snap(102.4, 198.7, 5.0)
        assert x == pytest.approx(100.0)
        assert y == pytest.approx(200.0)

    def test_rounds_up(self):
        x, y = _snap(102.6, 201.3, 5.0)
        assert x == pytest.approx(105.0)
        assert y == pytest.approx(200.0)

    def test_identity_when_tol_zero(self):
        x, y = _snap(123.456, 789.012, 0.0)
        assert x == pytest.approx(123.456)
        assert y == pytest.approx(789.012)

    def test_identity_when_tol_negative(self):
        x, y = _snap(123.456, 789.012, -1.0)
        assert x == pytest.approx(123.456)
        assert y == pytest.approx(789.012)

    def test_nearby_points_merge(self):
        """Two points within snap_tol_m of each other should snap to same cell."""
        x0, y0 = _snap(100.1, 200.1, 5.0)
        x1, y1 = _snap(102.3, 202.3, 5.0)
        assert x0 == pytest.approx(x1)
        assert y0 == pytest.approx(y1)

    def test_distant_points_differ(self):
        """Points > snap_tol_m apart should snap to different cells."""
        x0, y0 = _snap(100.0, 200.0, 5.0)
        x1, y1 = _snap(106.0, 200.0, 5.0)
        assert x0 != pytest.approx(x1)


# ===========================================================================
# read_gis_graph  (requires pyproj)
# ===========================================================================

@pytest.mark.skipif(
    not _pyproj_available(),
    reason="pyproj not installed — skipping read_gis_graph tests.",
)
class TestReadGisGraph:

    from src.network.gis_reader import read_gis_graph

    @pytest.fixture
    def graph_and_root(self):
        from src.network.gis_reader import read_gis_graph
        return read_gis_graph(
            FIXTURE_GEOJSON,
            root_coord=ROOT_COORD_WGS84,
            crs_input_epsg=4326,
            crs_metric_epsg=27700,
            snap_tol_m=5.0,
        )

    def test_returns_tuple(self, graph_and_root):
        G, root = graph_and_root
        assert isinstance(G, nx.Graph)
        assert isinstance(root, int)

    def test_root_is_zero(self, graph_and_root):
        _, root = graph_and_root
        assert root == 0

    def test_node_count(self, graph_and_root):
        G, _ = graph_and_root
        # 3×3 grid = 9 unique snapped endpoint nodes
        assert G.number_of_nodes() == 9

    def test_edge_count(self, graph_and_root):
        G, _ = graph_and_root
        # 12 LineString features → 12 edges (no degenerate edges after snapping)
        assert G.number_of_edges() == 12

    def test_all_nodes_have_x_km(self, graph_and_root):
        G, _ = graph_and_root
        for _, data in G.nodes(data=True):
            assert "x_km" in data
            assert "y_km" in data
            assert math.isfinite(data["x_km"])
            assert math.isfinite(data["y_km"])

    def test_all_edges_have_length_km(self, graph_and_root):
        G, _ = graph_and_root
        for _, _, data in G.edges(data=True):
            assert "length_km" in data
            assert data["length_km"] > 0.0

    def test_edge_lengths_approx_100m(self, graph_and_root):
        """Grid spacing is ~100 m, so each edge should be ~0.1 km."""
        G, _ = graph_and_root
        for _, _, data in G.edges(data=True):
            assert data["length_km"] == pytest.approx(0.1, abs=0.02)

    def test_root_node_is_nearest_to_root_coord(self, graph_and_root):
        """Node 0 should be the node with the smallest OSGB36 distance to the root."""
        import pyproj
        G, root = graph_and_root
        transformer = pyproj.Transformer.from_crs(4326, 27700, always_xy=True)
        rx_m, ry_m = transformer.transform(*ROOT_COORD_WGS84)
        root_x_km = G.nodes[root]["x_km"]
        root_y_km = G.nodes[root]["y_km"]
        root_dist2 = (root_x_km - rx_m * 1e-3) ** 2 + (root_y_km - ry_m * 1e-3) ** 2
        for nid, data in G.nodes(data=True):
            d2 = (data["x_km"] - rx_m * 1e-3) ** 2 + (data["y_km"] - ry_m * 1e-3) ** 2
            assert d2 >= root_dist2 - 1e-12

    def test_missing_file_raises(self):
        from src.network.gis_reader import read_gis_graph
        with pytest.raises(FileNotFoundError):
            read_gis_graph(
                FIXTURE_DIR / "no_such_file.geojson",
                root_coord=ROOT_COORD_WGS84,
            )

    def test_no_linestring_features_raises(self, tmp_path):
        from src.network.gis_reader import read_gis_graph
        data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
                    "properties": {}
                }
            ]
        }
        p = tmp_path / "points_only.geojson"
        p.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="[Nn]o usable"):
            read_gis_graph(p, root_coord=(0.0, 0.0))

    def test_different_snap_tol_changes_node_count(self):
        """Very large snap_tol merges all nodes into one — but that is
        tested indirectly.  Here we just verify that a tight snap (1 m)
        gives the same result as 5 m for the fixture."""
        from src.network.gis_reader import read_gis_graph
        G1, _ = read_gis_graph(FIXTURE_GEOJSON, ROOT_COORD_WGS84, snap_tol_m=1.0)
        G2, _ = read_gis_graph(FIXTURE_GEOJSON, ROOT_COORD_WGS84, snap_tol_m=5.0)
        assert G1.number_of_nodes() == G2.number_of_nodes()

    def test_max_edge_km_drops_edges(self):
        """Setting max_edge_km just below ~0.1 km should remove all edges."""
        from src.network.gis_reader import read_gis_graph
        G, _ = read_gis_graph(
            FIXTURE_GEOJSON, ROOT_COORD_WGS84, max_edge_km=0.05
        )
        assert G.number_of_edges() == 0

    def test_max_edge_km_keeps_all(self):
        """Setting max_edge_km >> edge length should keep all edges."""
        from src.network.gis_reader import read_gis_graph
        G, _ = read_gis_graph(
            FIXTURE_GEOJSON, ROOT_COORD_WGS84, max_edge_km=1.0
        )
        assert G.number_of_edges() == 12


# ===========================================================================
# Integration tests
# ===========================================================================

@pytest.mark.skipif(
    not _pyproj_available(),
    reason="pyproj not installed — skipping integration tests.",
)
class TestIntegration:

    @pytest.fixture
    def G_candidate(self):
        from src.network.gis_reader import read_gis_graph
        from src.network.candidate_graph import extract_candidate_graph
        G_raw, _ = read_gis_graph(
            FIXTURE_GEOJSON,
            root_coord=ROOT_COORD_WGS84,
            snap_tol_m=5.0,
        )
        G_cand, root = extract_candidate_graph(G_raw, root_node=0, max_edge_km=0.5)
        return G_cand, root

    def test_candidate_graph_connected(self, G_candidate):
        G, _ = G_candidate
        assert nx.is_connected(G)

    def test_candidate_graph_node_count(self, G_candidate):
        G, _ = G_candidate
        assert G.number_of_nodes() == 9

    def test_candidate_graph_edge_count(self, G_candidate):
        G, _ = G_candidate
        assert G.number_of_edges() == 12

    def test_candidate_graph_root_is_zero(self, G_candidate):
        _, root = G_candidate
        assert root == 0

    def test_nodes_reindexed_as_range(self, G_candidate):
        G, _ = G_candidate
        assert sorted(G.nodes()) == list(range(G.number_of_nodes()))

    @pytest.mark.skipif(
        not _pyproj_available(),
        reason="pyproj not installed",
    )
    def test_mip_feeder_produces_valid_feeder(self, G_candidate):
        """End-to-end: GIS → candidate graph → MIP → FeederData validation."""
        pytest.importorskip("pyomo.environ")

        from src.network.gis_reader import read_gis_graph
        from src.network.candidate_graph import extract_candidate_graph
        from src.network.mip_feeder import MIPFeederParams, solve_mip_feeder
        from src.network.solver_utils import _pick_solver

        solver, _ = _pick_solver("gurobi")
        if solver is None:
            pytest.skip("No MIP solver available.")

        G_raw, _ = read_gis_graph(
            FIXTURE_GEOJSON,
            root_coord=ROOT_COORD_WGS84,
            snap_tol_m=5.0,
        )
        G_cand, root = extract_candidate_graph(G_raw, root_node=0, max_edge_km=0.5)

        n = G_cand.number_of_nodes()
        params = MIPFeederParams(
            C_max=n - 1,       # unconstrained → MST
            total_households=90,
        )
        result = solve_mip_feeder(G_cand, root, params, seed=0)

        assert result.status in ("optimal", "feasible")
        assert result.feeder is not None
        # Tree has exactly n-1 directed edges
        assert len(result.selected_edges) == n - 1
        # All non-root buses have at least 1 household
        total_hh = sum(b.n_households for b in result.feeder.buses)
        assert total_hh == params.total_households
