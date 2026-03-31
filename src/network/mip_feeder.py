from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pyomo.environ as pyo
import networkx as nx

from src.network.synthetic_feeder import BusRecord, FeederData, LineRecord
from src.network.solver_utils import solve_pyomo_model


# ---------------------------------------------------------------------------
# Parameter and result containers
# ---------------------------------------------------------------------------

@dataclass
class MIPFeederParams:
    """
    Parameters for MIP feeder synthesis.

    Attributes
    ----------
    C_max : int
        Maximum number of non-root nodes permitted downstream of any single
        cable segment.  Must be ≥ 1.  Set to n−1 (or higher) to disable.
    total_households : int
        Total households allocated across non-root buses after topology is
        fixed.  Must be ≥ 1.
    base_kv : float
        Nominal voltage [kV].
    base_mva : float
        MVA base [MVA].
    V_min_pu, V_max_pu : float
        Per-unit voltage limits applied uniformly to all buses.
    r_per_km, x_per_km : float
        Base conductor impedance [Ω/km]; perturbation applied per line.
    perturb_frac : float
        Fractional impedance perturbation (e.g. 0.05 → ±5 %).
    P_max_mw : float
        Thermal capacity applied uniformly to all lines [MW].
    solver : str
        Preferred solver name; falls back via solver_utils fallback chain.
    time_limit_s : float
        Wall-clock time limit [s].
    mip_gap : float
        Relative MIP optimality gap tolerance.
    """
    C_max: int
    total_households: int
    base_kv: float = 11.0
    base_mva: float = 1.0
    V_min_pu: float = 0.95
    V_max_pu: float = 1.05
    r_per_km: float = 0.642
    x_per_km: float = 0.083
    perturb_frac: float = 0.05
    P_max_mw: float = 2.0
    solver: str = "gurobi"
    time_limit_s: float = 120.0
    mip_gap: float = 1e-3


@dataclass
class MIPFeederResult:
    """
    Output of ``solve_mip_feeder``.

    Attributes
    ----------
    feeder : FeederData or None
        Fully-populated feeder, ready for ``validate_feeder`` and
        ``export_feeder``.  None when the problem is infeasible or an
        error occurred.
    selected_edges : list of (int, int)
        Directed edges (from_bus, to_bus) forming the selected tree,
        oriented away from root by BFS.  Empty when no solution found.
    objective_km : float
        Total installed cable length [km].  inf when no solution found.
    solve_time_s : float or None
        Wall-clock solve time reported by the solver.
    status : str
        ``"optimal"``, ``"feasible"``, ``"infeasible"``, or ``"error"``.
    n_buses : int
        Number of buses in the candidate graph.
    n_edges_candidate : int
        Number of candidate edges considered by the MIP.
    """
    feeder: FeederData | None
    selected_edges: list[tuple[int, int]]
    objective_km: float
    solve_time_s: float | None
    status: str
    n_buses: int
    n_edges_candidate: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve_mip_feeder(
    G: nx.Graph,
    root_node: int,
    params: MIPFeederParams,
    seed: int = 42,
) -> MIPFeederResult:
    """
    Synthesise a radial feeder topology from a candidate graph using the MIP.

    Parameters
    ----------
    G : nx.Graph
        Candidate graph from ``candidate_graph.extract_candidate_graph``.
        Must have ``x_km``, ``y_km`` on every node and ``length_km`` on
        every edge.  Root must be node 0 (after re-indexing).
    root_node : int
        Root bus index in G (should be 0 after re-indexing).
    params : MIPFeederParams
        Solver and network parameters.
    seed : int
        Random seed for impedance perturbation and household allocation.

    Returns
    -------
    MIPFeederResult
        Contains FeederData, selected edges, objective, and solve metadata.
        ``feeder`` is None when no feasible solution is found.

    Raises
    ------
    ValueError
        If G has fewer than 2 nodes, root_node not in G, or required node
        attributes are missing.
    """
    n = G.number_of_nodes()
    if n < 2:
        raise ValueError(f"Candidate graph must have ≥ 2 nodes; got {n}.")
    if root_node not in G:
        raise ValueError(f"root_node {root_node!r} is not in G.")

    edges = sorted(_canonical(u, v) for u, v in G.edges())
    arcs = [(u, v) for u, v in G.edges()] + [(v, u) for u, v in G.edges()]

    model = _build_mip_model(G, root_node, n, edges, arcs, params)
    solve_result = solve_pyomo_model(
        model,
        solver=params.solver,
        time_limit_s=params.time_limit_s,
        mip_gap=params.mip_gap,
    )

    if solve_result["status"] not in ("optimal", "feasible"):
        return MIPFeederResult(
            feeder=None,
            selected_edges=[],
            objective_km=float("inf"),
            solve_time_s=solve_result["solve_time_s"],
            status=solve_result["status"],
            n_buses=n,
            n_edges_candidate=len(edges),
        )

    selected_undirected = [
        e for e in edges
        if pyo.value(model.x[e]) > 0.5
    ]
    directed = _orient_tree(selected_undirected, root_node, n)
    feeder = _build_feeder_data(G, root_node, directed, params, seed)
    obj_km = solve_result["objective"] if solve_result["objective"] is not None else float("inf")

    return MIPFeederResult(
        feeder=feeder,
        selected_edges=directed,
        objective_km=obj_km,
        solve_time_s=solve_result["solve_time_s"],
        status=solve_result["status"],
        n_buses=n,
        n_edges_candidate=len(edges),
    )


def compute_downstream_counts(
    directed_edges: list[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """
    For each directed edge (u, v), count the nodes in the subtree rooted at v
    (including v itself).

    This equals the MIP flow value f_{uv} on the optimal solution, and is
    used to verify C_max compliance without accessing Pyomo internals.

    Parameters
    ----------
    directed_edges : list of (int, int)
        BFS-oriented directed edges of the synthesised tree.

    Returns
    -------
    dict mapping (u, v) → subtree size at v.
    """
    # Build children map
    all_nodes: set[int] = set()
    children: dict[int, list[int]] = {}
    for u, v in directed_edges:
        all_nodes.update((u, v))
        children.setdefault(u, []).append(v)
        children.setdefault(v, [])   # ensure v exists as a key

    counts: dict[tuple[int, int], int] = {}
    for u, v in directed_edges:
        # BFS from v counting all descendants
        subtree_size = 0
        queue: deque[int] = deque([v])
        while queue:
            node = queue.popleft()
            subtree_size += 1
            queue.extend(children.get(node, []))
        counts[(u, v)] = subtree_size

    return counts


# ---------------------------------------------------------------------------
# MIP model builder
# ---------------------------------------------------------------------------

def _build_mip_model(
    G: nx.Graph,
    root: int,
    n: int,
    edges: list[tuple[int, int]],
    arcs: list[tuple[int, int]],
    params: MIPFeederParams,
) -> pyo.ConcreteModel:
    """Build the Pyomo MIP model (C1–C4)."""
    model = pyo.ConcreteModel(name="GeoDistNet_MIP")

    # Decision variables
    model.x = pyo.Var(edges, domain=pyo.Binary)
    model.f = pyo.Var(arcs,  domain=pyo.NonNegativeReals)

    # Objective: minimise total cable length
    model.OBJ = pyo.Objective(
        expr=sum(G[u][v]["length_km"] * model.x[_canonical(u, v)]
                 for u, v in G.edges()),
        sense=pyo.minimize,
    )

    # (C1) Tree size: exactly n−1 edges selected
    model.c_tree = pyo.Constraint(
        expr=sum(model.x[e] for e in edges) == n - 1
    )

    # (C2) Flow balance at root: net outflow = n−1
    root_nbrs = list(G.neighbors(root))
    model.c_root_flow = pyo.Constraint(
        expr=(
            sum(model.f[root, j] for j in root_nbrs)
          - sum(model.f[i, root] for i in root_nbrs)
        ) == n - 1
    )

    # (C3) Flow balance at each non-root bus: net inflow = 1
    model.c_balance = pyo.ConstraintList()
    for b in range(n):
        if b == root:
            continue
        nbrs = list(G.neighbors(b))
        model.c_balance.add(
            sum(model.f[i, b] for i in nbrs)
          - sum(model.f[b, j] for j in nbrs)
          == 1
        )

    # (C4) Capacity: f_ij ≤ C_max · x_e  (also enforces flow only on selected edges)
    C_max = params.C_max
    model.c_cap = pyo.ConstraintList()
    for i, j in arcs:
        e = _canonical(i, j)
        model.c_cap.add(model.f[i, j] <= C_max * model.x[e])

    return model


# ---------------------------------------------------------------------------
# Post-solve helpers
# ---------------------------------------------------------------------------

def _canonical(u: int, v: int) -> tuple[int, int]:
    """Canonical (sorted) representation of an undirected edge."""
    return (min(u, v), max(u, v))


def _orient_tree(
    undirected_edges: list[tuple[int, int]],
    root: int,
    n_nodes: int,
) -> list[tuple[int, int]]:
    """
    Orient a spanning tree away from root via BFS.

    Returns a list of directed edges (parent, child) in BFS order.
    """
    adj: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
    for u, v in undirected_edges:
        adj[u].append(v)
        adj[v].append(u)

    directed: list[tuple[int, int]] = []
    visited: set[int] = {root}
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        for nb in sorted(adj[node]):       # sorted for determinism
            if nb not in visited:
                visited.add(nb)
                directed.append((node, nb))
                queue.append(nb)
    return directed


def _build_feeder_data(
    G: nx.Graph,
    root: int,
    directed_edges: list[tuple[int, int]],
    params: MIPFeederParams,
    seed: int,
) -> FeederData:
    """
    Build a FeederData from the MIP-selected directed tree.

    Impedances are assigned using the same IEEE-33-style perturbed formula
    used throughout this project.  Households are allocated via Dirichlet
    draw with largest-remainder rounding.
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    Z_base = params.base_kv ** 2 / params.base_mva

    # --- Buses (coordinates from candidate graph) ---
    buses: list[BusRecord] = []
    for node_id in range(n):
        nd = G.nodes[node_id]
        buses.append(BusRecord(
            id=node_id,
            geo_x_km=round(float(nd["x_km"]), 8),
            geo_y_km=round(float(nd["y_km"]), 8),
            n_households=0,            # filled below
            V_min_pu2=round(params.V_min_pu ** 2, 8),
            V_max_pu2=round(params.V_max_pu ** 2, 8),
        ))

    # --- Lines (perturbed IEEE-33-style impedances) ---
    lines: list[LineRecord] = []
    for line_id, (from_bus, to_bus) in enumerate(directed_edges):
        length_km = float(G[from_bus][to_bus]["length_km"])
        r_scale = float(rng.uniform(1.0 - params.perturb_frac, 1.0 + params.perturb_frac))
        x_scale = float(rng.uniform(1.0 - params.perturb_frac, 1.0 + params.perturb_frac))
        r_ohm = params.r_per_km * r_scale * length_km
        x_ohm = params.x_per_km * x_scale * length_km
        lines.append(LineRecord(
            id=line_id,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=round(length_km, 8),
            r_ohm=round(r_ohm, 8),
            x_ohm=round(x_ohm, 8),
            r_pu=round(r_ohm / Z_base, 10),
            x_pu=round(x_ohm / Z_base, 10),
            P_max_mw=params.P_max_mw,
        ))

    # --- Household allocation (Dirichlet + largest-remainder rounding) ---
    non_root = [b for b in range(n) if b != root]
    weights = rng.dirichlet(np.ones(len(non_root)))
    raw = weights * params.total_households
    hh_floor = np.floor(raw).astype(int)
    remainder = params.total_households - int(hh_floor.sum())
    fracs = raw - hh_floor
    top_idx = np.argsort(fracs)[::-1][:remainder]
    hh_floor[top_idx] += 1

    assert int(hh_floor.sum()) == params.total_households

    hh_map: dict[int, int] = {non_root[i]: int(hh_floor[i]) for i in range(len(non_root))}
    for b in buses:
        b.n_households = hh_map.get(b.id, 0)

    return FeederData(
        buses=buses,
        lines=lines,
        root_bus=root,
        base_kv=params.base_kv,
        base_mva=params.base_mva,
        Z_base_ohm=round(Z_base, 8),
        V_ref_pu2=1.0,
        V_min_pu=params.V_min_pu,
        V_max_pu=params.V_max_pu,
        r_per_km_base=params.r_per_km,
        x_per_km_base=params.x_per_km,
    )
