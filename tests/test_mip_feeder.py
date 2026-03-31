from __future__ import annotations

import math
import sys
import pathlib
from collections import deque

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import networkx as nx
import pyomo.environ as pyo

from src.network.candidate_graph import make_grid_candidate_graph
from src.network.mip_feeder import (
    MIPFeederParams,
    MIPFeederResult,
    compute_downstream_counts,
    solve_mip_feeder,
)
from src.network.synthetic_feeder import validate_feeder


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _solver_available() -> bool:
    """Return True if at least one supported solver is installed."""
    for name in ("gurobi", "highs", "cbc", "glpk"):
        try:
            opt = pyo.SolverFactory(name)
            if opt.available():
                return True
        except Exception:
            pass
    return False


def _make_path_graph(n: int, spacing: float = 0.10) -> tuple[nx.Graph, int]:
    """Straight-line path graph: root=0, nodes 0-1-2-...(n-1)."""
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, x_km=round(i * spacing, 8), y_km=0.0)
    for i in range(n - 1):
        G.add_edge(i, i + 1, length_km=spacing)
    return G, 0


def _make_star_graph(n_arms: int, arm_length: float = 0.10) -> tuple[nx.Graph, int]:
    """Star graph: root=0 connected to n_arms leaf nodes, no other edges."""
    G = nx.Graph()
    G.add_node(0, x_km=0.0, y_km=0.0)
    for i in range(1, n_arms + 1):
        angle = 2.0 * math.pi * (i - 1) / n_arms
        G.add_node(
            i,
            x_km=round(arm_length * math.cos(angle), 8),
            y_km=round(arm_length * math.sin(angle), 8),
        )
        G.add_edge(0, i, length_km=arm_length)
    return G, 0


def _default_params(C_max: int, n_hh: int = 10) -> MIPFeederParams:
    return MIPFeederParams(C_max=C_max, total_households=n_hh)


def _all_reachable(directed_edges: list[tuple[int, int]], root: int, n: int) -> bool:
    """Return True if all n nodes are reachable from root via directed_edges."""
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    for u, v in directed_edges:
        children[u].append(v)
    visited: set[int] = set()
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        queue.extend(children[node])
    return len(visited) == n


def _mst_total_length(G: nx.Graph) -> float:
    """Total edge weight of the minimum spanning tree of G."""
    T = nx.minimum_spanning_tree(G, weight="length_km")
    return sum(d["length_km"] for _, _, d in T.edges(data=True))


# ===========================================================================
# compute_downstream_counts — no solver required
# ===========================================================================

class TestComputeDownstreamCounts:

    def test_single_edge(self):
        # Tree: 0→1 (root=0, one leaf)
        edges = [(0, 1)]
        counts = compute_downstream_counts(edges)
        assert counts[(0, 1)] == 1

    def test_path_two_edges(self):
        # Tree: 0→1→2
        # Edge 0→1: subtree = {1, 2} → 2
        # Edge 1→2: subtree = {2}    → 1
        edges = [(0, 1), (1, 2)]
        counts = compute_downstream_counts(edges)
        assert counts[(0, 1)] == 2
        assert counts[(1, 2)] == 1

    def test_path_three_edges(self):
        # Tree: 0→1→2→3
        edges = [(0, 1), (1, 2), (2, 3)]
        counts = compute_downstream_counts(edges)
        assert counts[(0, 1)] == 3
        assert counts[(1, 2)] == 2
        assert counts[(2, 3)] == 1

    def test_star_three_arms(self):
        # Tree: 0→1, 0→2, 0→3 (star, each arm = 1 node)
        edges = [(0, 1), (0, 2), (0, 3)]
        counts = compute_downstream_counts(edges)
        assert counts[(0, 1)] == 1
        assert counts[(0, 2)] == 1
        assert counts[(0, 3)] == 1

    def test_branching_tree(self):
        # Tree:  0→1→3
        #           0→2→4
        #               2→5
        edges = [(0, 1), (1, 3), (0, 2), (2, 4), (2, 5)]
        counts = compute_downstream_counts(edges)
        assert counts[(0, 1)] == 2    # subtree {1, 3}
        assert counts[(1, 3)] == 1    # subtree {3}
        assert counts[(0, 2)] == 3    # subtree {2, 4, 5}
        assert counts[(2, 4)] == 1
        assert counts[(2, 5)] == 1

    def test_all_counts_at_least_one(self):
        G, root = make_grid_candidate_graph(n_rows=3, n_cols=3)
        T = nx.minimum_spanning_tree(G, weight="length_km")
        # Orient MST via BFS from root
        adj = {n: list(T.neighbors(n)) for n in T.nodes()}
        directed = []
        visited = {root}
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    directed.append((node, nb))
                    queue.append(nb)
        counts = compute_downstream_counts(directed)
        assert all(v >= 1 for v in counts.values())

    def test_empty_edges_returns_empty(self):
        counts = compute_downstream_counts([])
        assert counts == {}


# ===========================================================================
# MIPFeederParams — no solver required
# ===========================================================================

class TestMIPFeederParams:

    def test_required_fields_only(self):
        p = MIPFeederParams(C_max=5, total_households=20)
        assert p.C_max == 5
        assert p.total_households == 20

    def test_defaults(self):
        p = MIPFeederParams(C_max=5, total_households=20)
        assert p.base_kv == 11.0
        assert p.base_mva == 1.0
        assert p.V_min_pu == 0.95
        assert p.V_max_pu == 1.05
        assert p.r_per_km == pytest.approx(0.642)
        assert p.x_per_km == pytest.approx(0.083)
        assert p.perturb_frac == pytest.approx(0.05)
        assert p.P_max_mw == 2.0
        assert p.solver == "gurobi"
        assert p.time_limit_s == 120.0
        assert p.mip_gap == pytest.approx(1e-3)

    def test_override_solver(self):
        p = MIPFeederParams(C_max=5, total_households=20, solver="highs")
        assert p.solver == "highs"


# ===========================================================================
# solve_mip_feeder — requires solver
# ===========================================================================

@pytest.mark.skipif(
    not _solver_available(),
    reason="No supported solver (Gurobi / HiGHS / CBC / GLPK) installed.",
)
class TestSolveMIPFeeder:

    # --- 3×3 grid, C_max = 8 (unconstrained: C_max ≥ n−1 = 8) ---

    @pytest.fixture(scope="class")
    def grid_result(self):
        """Solved 3×3 grid with C_max=8 (unconstrained)."""
        G, root = make_grid_candidate_graph(n_rows=3, n_cols=3, spacing_km=0.10)
        params = _default_params(C_max=8, n_hh=45)
        return solve_mip_feeder(G, root, params, seed=42), G, root

    def test_status_optimal(self, grid_result):
        result, G, root = grid_result
        assert result.status == "optimal"

    def test_validate_feeder_passes(self, grid_result):
        result, G, root = grid_result
        assert result.feeder is not None
        validate_feeder(result.feeder)          # raises AssertionError if invalid

    def test_n_buses(self, grid_result):
        result, G, root = grid_result
        assert result.n_buses == 9

    def test_exactly_n_minus_1_edges(self, grid_result):
        result, G, root = grid_result
        n = result.n_buses
        assert len(result.selected_edges) == n - 1

    def test_feeder_n_lines_equals_selected(self, grid_result):
        result, G, root = grid_result
        assert result.feeder.n_lines == len(result.selected_edges)

    def test_feeder_n_buses_correct(self, grid_result):
        result, G, root = grid_result
        assert result.feeder.n_buses == 9

    def test_all_nodes_reachable(self, grid_result):
        result, G, root = grid_result
        assert _all_reachable(result.selected_edges, root, result.n_buses)

    def test_objective_finite_positive(self, grid_result):
        result, G, root = grid_result
        assert result.objective_km > 0.0
        assert math.isfinite(result.objective_km)

    def test_objective_matches_mst_length(self, grid_result):
        """With C_max ≥ n−1 the MIP recovers the MST total cable length."""
        result, G, root = grid_result
        mst_len = _mst_total_length(G)
        assert result.objective_km == pytest.approx(mst_len, rel=1e-3)

    def test_cmax_compliance_unconstrained(self, grid_result):
        result, G, root = grid_result
        counts = compute_downstream_counts(result.selected_edges)
        for (u, v), cnt in counts.items():
            assert cnt <= 8, f"Edge ({u},{v}) has {cnt} downstream nodes > C_max=8"

    def test_solve_time_positive(self, grid_result):
        result, G, root = grid_result
        assert result.solve_time_s is not None
        assert result.solve_time_s >= 0.0

    def test_feeder_total_households(self, grid_result):
        result, G, root = grid_result
        assert result.feeder.total_households == 45

    def test_feeder_root_has_zero_households(self, grid_result):
        result, G, root = grid_result
        assert result.feeder.buses[root].n_households == 0

    def test_feeder_impedances_positive(self, grid_result):
        result, G, root = grid_result
        for ln in result.feeder.lines:
            assert ln.r_ohm > 0.0
            assert ln.x_ohm > 0.0

    def test_feeder_v_bounds_consistent(self, grid_result):
        result, G, root = grid_result
        p = MIPFeederParams(C_max=8, total_households=45)
        for b in result.feeder.buses:
            assert abs(b.V_min_pu2 - p.V_min_pu ** 2) < 1e-6
            assert abs(b.V_max_pu2 - p.V_max_pu ** 2) < 1e-6

    # --- 3×3 grid, C_max = 4 (binding for deep subtrees) ---

    def test_cmax_4_compliance(self):
        G, root = make_grid_candidate_graph(n_rows=3, n_cols=3, spacing_km=0.10)
        params = _default_params(C_max=4, n_hh=36)
        result = solve_mip_feeder(G, root, params, seed=0)
        assert result.status in ("optimal", "feasible")
        if result.selected_edges:
            counts = compute_downstream_counts(result.selected_edges)
            for (u, v), cnt in counts.items():
                assert cnt <= 4, f"Edge ({u},{v}): {cnt} downstream nodes > C_max=4"

    # --- 2-node graph: simplest possible feeder ---

    def test_two_node_graph_feasible(self):
        G = nx.Graph()
        G.add_node(0, x_km=0.0, y_km=0.0)
        G.add_node(1, x_km=0.10, y_km=0.0)
        G.add_edge(0, 1, length_km=0.10)
        params = _default_params(C_max=1, n_hh=5)
        result = solve_mip_feeder(G, root_node=0, params=params)
        assert result.status in ("optimal", "feasible")
        assert len(result.selected_edges) == 1
        assert result.feeder is not None
        validate_feeder(result.feeder)

    def test_two_node_objective(self):
        G = nx.Graph()
        G.add_node(0, x_km=0.0, y_km=0.0)
        G.add_node(1, x_km=0.20, y_km=0.0)
        G.add_edge(0, 1, length_km=0.20)
        params = _default_params(C_max=1, n_hh=3)
        result = solve_mip_feeder(G, root_node=0, params=params)
        assert result.objective_km == pytest.approx(0.20, rel=1e-4)

    # --- Star graph, C_max = 1 ---

    def test_star_cmax_1_feasible(self):
        """Star: each edge carries exactly 1 downstream node ≤ C_max=1."""
        G, root = _make_star_graph(n_arms=4, arm_length=0.10)
        params = _default_params(C_max=1, n_hh=12)
        result = solve_mip_feeder(G, root, params, seed=7)
        assert result.status in ("optimal", "feasible")
        assert result.feeder is not None
        validate_feeder(result.feeder)
        counts = compute_downstream_counts(result.selected_edges)
        for (u, v), cnt in counts.items():
            assert cnt == 1

    # --- Path(3), C_max = 1  →  infeasible ---

    def test_path3_cmax1_infeasible(self):
        """Path 0-1-2 with C_max=1: edge 0→1 must carry 2 units > C_max."""
        G, root = _make_path_graph(n=3, spacing=0.10)
        params = _default_params(C_max=1, n_hh=5)
        result = solve_mip_feeder(G, root, params)
        assert result.status in ("infeasible", "error")
        assert result.feeder is None
        assert result.selected_edges == []
        assert result.objective_km == float("inf")

    # --- Path(3), C_max = 2  →  feasible ---

    def test_path3_cmax2_feasible(self):
        """Path 0-1-2 with C_max=2: exactly at limit for edge 0→1."""
        G, root = _make_path_graph(n=3, spacing=0.10)
        params = _default_params(C_max=2, n_hh=6)
        result = solve_mip_feeder(G, root, params)
        assert result.status in ("optimal", "feasible")
        assert result.feeder is not None
        validate_feeder(result.feeder)
        counts = compute_downstream_counts(result.selected_edges)
        # Edge from root (0) to node 1 should carry exactly 2
        root_edges = [(u, v) for (u, v) in result.selected_edges if u == 0]
        assert len(root_edges) == 1
        assert counts[root_edges[0]] == 2

    def test_path3_cmax2_downstream_within_limit(self):
        G, root = _make_path_graph(n=3, spacing=0.10)
        params = _default_params(C_max=2, n_hh=6)
        result = solve_mip_feeder(G, root, params)
        counts = compute_downstream_counts(result.selected_edges)
        for (u, v), cnt in counts.items():
            assert cnt <= 2

    # --- Error handling ---

    def test_raises_on_single_node_graph(self):
        G = nx.Graph()
        G.add_node(0, x_km=0.0, y_km=0.0)
        params = _default_params(C_max=1, n_hh=1)
        with pytest.raises(ValueError, match="≥ 2"):
            solve_mip_feeder(G, root_node=0, params=params)

    def test_raises_on_missing_root(self):
        G, _ = make_grid_candidate_graph(n_rows=2, n_cols=2)
        params = _default_params(C_max=3, n_hh=5)
        with pytest.raises(ValueError, match="root_node"):
            solve_mip_feeder(G, root_node=999, params=params)

    def test_n_edges_candidate_reported(self):
        G, root = make_grid_candidate_graph(n_rows=2, n_cols=2, spacing_km=0.1)
        params = _default_params(C_max=3, n_hh=5)
        result = solve_mip_feeder(G, root, params)
        # 2×2 grid has 4 horizontal+vertical edges
        assert result.n_edges_candidate == G.number_of_edges()
