from __future__ import annotations

import pathlib
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BusRecord:
    """Single feeder bus."""
    id: int
    geo_x_km: float
    geo_y_km: float
    n_households: int
    V_min_pu2: float   # squared voltage lower bound [pu²]  (V_min_pu)²
    V_max_pu2: float   # squared voltage upper bound [pu²]  (V_max_pu)²


@dataclass
class LineRecord:
    """Single directed feeder line (from_bus → to_bus)."""
    id: int
    from_bus: int
    to_bus: int
    length_km: float
    r_ohm: float
    x_ohm: float
    r_pu: float
    x_pu: float
    P_max_mw: float


@dataclass
class FeederData:
    """Complete synthetic feeder dataset."""
    buses: List[BusRecord]
    lines: List[LineRecord]
    root_bus: int
    base_kv: float
    base_mva: float
    Z_base_ohm: float
    V_ref_pu2: float
    V_min_pu: float
    V_max_pu: float
    r_per_km_base: float
    x_per_km_base: float

    @property
    def n_buses(self) -> int:
        return len(self.buses)

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def total_households(self) -> int:
        return sum(b.n_households for b in self.buses)

    def adjacency(self) -> Dict[int, List[int]]:
        """Undirected adjacency list derived from line set."""
        adj: Dict[int, List[int]] = {b.id: [] for b in self.buses}
        for ln in self.lines:
            adj[ln.from_bus].append(ln.to_bus)
            adj[ln.to_bus].append(ln.from_bus)
        return adj


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_synthetic_feeder(
    n_buses: int = 33,
    area_km: float = 2.0,
    total_households: int = 252,
    base_kv: float = 11.0,
    base_mva: float = 1.0,
    V_min_pu: float = 0.9,
    V_max_pu: float = 1.1,
    r_per_km_base: float = 0.642,    # Ω/km  (IEEE-33 style)
    x_per_km_base: float = 0.083,    # Ω/km
    perturb_frac: float = 0.10,      # ±10 % applied independently to r and x
    P_max_mw: float = 2.0,
    seed: int = 42,
) -> FeederData:
    """
    Generate a synthetic radial distribution feeder.

    Steps
    -----
    1. Draw n_buses random coordinates uniformly in [0, area_km]².
    2. Build the complete Euclidean distance graph and compute the MST.
    3. Orient MST edges away from root bus 0 via BFS → directed radial tree.
    4. Assign line impedances (Ω):
           r_ij = r_per_km_base × r_scale × length_km
           x_ij = x_per_km_base × x_scale × length_km
       where r_scale, x_scale ~ Uniform(1−perturb_frac, 1+perturb_frac).
    5. Compute per-unit values:
           Z_base = base_kv² / base_mva   [Ω]
           r_pu = r_ohm / Z_base
           x_pu = x_ohm / Z_base
    6. Allocate total_households across non-root buses using a
       Dirichlet-weighted draw with largest-remainder rounding.

    Parameters
    ----------
    n_buses : int
        Total bus count including root (substation, bus 0).
    area_km : float
        Side length of the square service area [km].
    total_households : int
        Households to allocate across non-root buses.
    base_kv : float
        Nominal voltage [kV].
    base_mva : float
        MVA base [MVA].
    V_min_pu, V_max_pu : float
        Voltage magnitude limits [pu].
    r_per_km_base, x_per_km_base : float
        Base conductor impedance per km [Ω/km].
    perturb_frac : float
        Fractional perturbation bound (e.g. 0.10 → ±10 %).
    P_max_mw : float
        Thermal capacity applied uniformly to all lines [MW].
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    FeederData
    """
    if n_buses < 2:
        raise ValueError("n_buses must be ≥ 2.")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Random bus coordinates  [km]
    # ------------------------------------------------------------------
    coords = rng.uniform(0.0, area_km, size=(n_buses, 2))

    # ------------------------------------------------------------------
    # 2. Full Euclidean distance matrix  [km]
    # ------------------------------------------------------------------
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (n, n, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))                      # (n, n)

    # ------------------------------------------------------------------
    # 3. Minimum Spanning Tree → directed radial tree from root 0
    # ------------------------------------------------------------------
    mst = minimum_spanning_tree(csr_matrix(dist))   # lower-triangular CSR

    # Build undirected adjacency list from MST edges
    adj: Dict[int, List[int]] = {i: [] for i in range(n_buses)}
    rows, cols = mst.nonzero()
    for r, c in zip(rows.tolist(), cols.tolist()):
        adj[r].append(c)
        adj[c].append(r)

    # BFS from root to orient edges (parent → child)
    root_bus = 0
    directed_edges: List[Tuple[int, int]] = []
    visited: set = {root_bus}
    queue: deque = deque([root_bus])
    while queue:
        node = queue.popleft()
        for nb in sorted(adj[node]):       # sorted for deterministic ordering
            if nb not in visited:
                visited.add(nb)
                directed_edges.append((node, nb))
                queue.append(nb)

    assert len(directed_edges) == n_buses - 1, (
        f"Expected {n_buses - 1} edges, got {len(directed_edges)}."
    )
    assert len(visited) == n_buses, "MST does not span all buses."

    # ------------------------------------------------------------------
    # 4–5. Line impedances (Ω and pu)
    #      r_ij = r_base × r_scale × length,   r_pu = r_ohm / Z_base
    # ------------------------------------------------------------------
    Z_base = base_kv ** 2 / base_mva    # [Ω]

    lines: List[LineRecord] = []
    for line_id, (i, j) in enumerate(directed_edges):
        length_km = float(dist[i, j])

        r_scale = float(rng.uniform(1.0 - perturb_frac, 1.0 + perturb_frac))
        x_scale = float(rng.uniform(1.0 - perturb_frac, 1.0 + perturb_frac))

        r_ohm = r_per_km_base * r_scale * length_km
        x_ohm = x_per_km_base * x_scale * length_km
        r_pu  = r_ohm / Z_base
        x_pu  = x_ohm / Z_base

        lines.append(LineRecord(
            id=line_id,
            from_bus=i,
            to_bus=j,
            length_km=round(length_km, 8),
            r_ohm=round(r_ohm, 8),
            x_ohm=round(x_ohm, 8),
            r_pu=round(r_pu, 10),
            x_pu=round(x_pu, 10),
            P_max_mw=P_max_mw,
        ))

    # ------------------------------------------------------------------
    # 6. Household allocation
    #    Dirichlet proportional weights + largest-remainder rounding
    # ------------------------------------------------------------------
    non_root = [b for b in range(n_buses) if b != root_bus]
    weights = rng.dirichlet(np.ones(len(non_root)))
    raw = weights * total_households
    hh_floor = np.floor(raw).astype(int)

    remainder = total_households - int(hh_floor.sum())
    fracs = raw - hh_floor
    top_idx = np.argsort(fracs)[::-1][:remainder]
    hh_floor[top_idx] += 1

    assert int(hh_floor.sum()) == total_households

    hh_map: Dict[int, int] = {non_root[i]: int(hh_floor[i]) for i in range(len(non_root))}

    buses: List[BusRecord] = []
    for b in range(n_buses):
        buses.append(BusRecord(
            id=b,
            geo_x_km=round(float(coords[b, 0]), 8),
            geo_y_km=round(float(coords[b, 1]), 8),
            n_households=hh_map.get(b, 0),
            V_min_pu2=round(V_min_pu ** 2, 8),
            V_max_pu2=round(V_max_pu ** 2, 8),
        ))

    return FeederData(
        buses=buses,
        lines=lines,
        root_bus=root_bus,
        base_kv=base_kv,
        base_mva=base_mva,
        Z_base_ohm=round(Z_base, 8),
        V_ref_pu2=1.0,
        V_min_pu=V_min_pu,
        V_max_pu=V_max_pu,
        r_per_km_base=r_per_km_base,
        x_per_km_base=x_per_km_base,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_feeder(data: FeederData) -> None:
    """
    Assert structural correctness of a FeederData object.

    Checks
    ------
    - Radiality      : n_lines == n_buses − 1
    - Connectivity   : all buses reachable from root via undirected BFS
    - Positive r, x  : r_ohm > 0 and x_ohm > 0 for every line
    - PU consistency : |r_pu − r_ohm / Z_base| < 1e-6  (and same for x)
    - HH sum         : sum of n_households == total_households > 0
    - Root HH = 0    : root bus carries no households

    Raises
    ------
    AssertionError on any violation.
    """
    n, m = data.n_buses, data.n_lines

    # Radiality
    assert m == n - 1, (
        f"Radiality violated: {m} lines for {n} buses (expected {n - 1})."
    )

    # Connectivity (undirected BFS from root)
    adj = data.adjacency()
    visited: set = {data.root_bus}
    queue: deque = deque([data.root_bus])
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    assert len(visited) == n, (
        f"Connectivity violated: {len(visited)} of {n} buses reachable from root."
    )

    # Positive impedances
    for ln in data.lines:
        assert ln.r_ohm > 0, f"Line {ln.id}: r_ohm={ln.r_ohm} is not positive."
        assert ln.x_ohm > 0, f"Line {ln.id}: x_ohm={ln.x_ohm} is not positive."

    # Per-unit consistency
    Z = data.Z_base_ohm
    for ln in data.lines:
        err_r = abs(ln.r_pu - ln.r_ohm / Z)
        err_x = abs(ln.x_pu - ln.x_ohm / Z)
        assert err_r < 1e-6, (
            f"Line {ln.id}: r_pu={ln.r_pu} vs r_ohm/Z={ln.r_ohm / Z:.10f}  "
            f"(diff={err_r:.2e})"
        )
        assert err_x < 1e-6, (
            f"Line {ln.id}: x_pu={ln.x_pu} vs x_ohm/Z={ln.x_ohm / Z:.10f}  "
            f"(diff={err_x:.2e})"
        )

    # Household sum
    total = data.total_households
    assert total > 0, "Total households must be positive."

    # Root bus has zero households
    root_hh = data.buses[data.root_bus].n_households
    assert root_hh == 0, (
        f"Root bus {data.root_bus} should have 0 households, got {root_hh}."
    )


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_feeder(data: FeederData, output_dir) -> None:
    """
    Export FeederData to disk.

    Files written
    -------------
    {output_dir}/buses.csv          – bus coordinates, households, voltage bounds
    {output_dir}/lines.csv          – line parameters (Ω and pu)
    {output_dir}/feeder_params.yaml – scalar network parameters
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # buses.csv
    buses_df = pd.DataFrame([
        {
            "id": b.id,
            "geo_x_km": b.geo_x_km,
            "geo_y_km": b.geo_y_km,
            "n_households": b.n_households,
            "V_min_pu2": b.V_min_pu2,
            "V_max_pu2": b.V_max_pu2,
        }
        for b in data.buses
    ])
    buses_df.to_csv(out / "buses.csv", index=False)

    # lines.csv
    lines_df = pd.DataFrame([
        {
            "id": ln.id,
            "from_bus": ln.from_bus,
            "to_bus": ln.to_bus,
            "length_km": ln.length_km,
            "r_ohm": ln.r_ohm,
            "x_ohm": ln.x_ohm,
            "r_pu": ln.r_pu,
            "x_pu": ln.x_pu,
            "P_max_mw": ln.P_max_mw,
        }
        for ln in data.lines
    ])
    lines_df.to_csv(out / "lines.csv", index=False)

    # feeder_params.yaml
    params = {
        "root_bus":             int(data.root_bus),
        "base_kv":              float(data.base_kv),
        "base_mva":             float(data.base_mva),
        "Z_base_ohm":           float(data.Z_base_ohm),
        "V_ref_pu2":            float(data.V_ref_pu2),
        "V_min_pu":             float(data.V_min_pu),
        "V_max_pu":             float(data.V_max_pu),
        "r_per_km_base_ohm":    float(data.r_per_km_base),
        "x_per_km_base_ohm":    float(data.x_per_km_base),
        "n_buses":              int(data.n_buses),
        "n_lines":              int(data.n_lines),
        "total_households":     int(data.total_households),
    }
    with open(out / "feeder_params.yaml", "w") as f:
        yaml.dump(params, f, default_flow_style=False, sort_keys=False)
