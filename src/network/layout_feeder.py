from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.network.synthetic_feeder import (
    BusRecord,
    FeederData,
    LineRecord,
)


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

def _perp_unit(dx: float, dy: float, side: int) -> Tuple[float, float]:
    """
    Return a unit vector perpendicular to (dx, dy).

    side = +1 → rotate 90° counter-clockwise (left of forward direction)
    side = -1 → rotate 90° clockwise          (right of forward direction)
    """
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    if side == 1:
        return -uy, ux      # CCW: (−dy, dx)
    else:
        return uy, -ux      # CW:  (dy, −dx)


# ---------------------------------------------------------------------------
# generate_layout_feeder
# ---------------------------------------------------------------------------

def generate_layout_feeder(
    n_trunk: int = 8,
    lateral_interval: int = 1,
    lateral_depth: int = 3,
    trunk_spacing_km: float = 0.15,
    lateral_spacing_km: float = 0.08,
    total_households: int = 200,
    base_kv: float = 11.0,
    base_mva: float = 1.0,
    V_min_pu: float = 0.95,
    V_max_pu: float = 1.05,
    r_per_km_base: float = 0.642,
    x_per_km_base: float = 0.083,
    perturb_frac: float = 0.05,
    P_max_mw: float = 2.0,
    trunk_waypoints: Optional[List[Tuple[float, float]]] = None,
    seed: int = 42,
) -> FeederData:
    """
    Generate a layout-informed radial feeder with comb / herringbone topology.

    Parameters
    ----------
    n_trunk : int
        Number of trunk buses (not counting root, bus 0).
        Trunk buses are numbered 1 … n_trunk.
    lateral_interval : int
        Every *lateral_interval*-th trunk bus sprouts a lateral branch.
        Default 1 → every trunk bus has a lateral.
    lateral_depth : int
        Number of load buses per lateral branch (≥ 1).
    trunk_spacing_km : float
        Distance between consecutive trunk buses [km].
    lateral_spacing_km : float
        Distance between consecutive buses within a lateral [km].
    total_households : int
        Total households allocated across all non-root load buses.
    base_kv, base_mva : float
        System base values.
    V_min_pu, V_max_pu : float
        Per-unit voltage bounds.
    r_per_km_base, x_per_km_base : float
        Base conductor impedance [Ω/km].
    perturb_frac : float
        Impedance perturbation fraction (e.g. 0.05 → ±5 %).
    P_max_mw : float
        Thermal capacity applied uniformly to all lines [MW].
    trunk_waypoints : List[(x_km, y_km)] or None
        Optional explicit coordinates for trunk buses (including root at
        index 0).  Length must equal n_trunk + 1.  When provided,
        trunk_spacing_km is ignored for coordinate placement (but line
        lengths are computed from Euclidean distances).
        This parameter is the GIS integration hook.
    seed : int
        Random seed for impedance perturbation and household allocation.

    Returns
    -------
    FeederData
        Fully populated feeder dataset compatible with all downstream
        optimization and ADMM modules.

    Raises
    ------
    ValueError
        If n_trunk < 1, lateral_depth < 1, or trunk_waypoints has wrong length.
    """
    if n_trunk < 1:
        raise ValueError("n_trunk must be ≥ 1.")
    if lateral_depth < 1:
        raise ValueError("lateral_depth must be ≥ 1.")
    if trunk_waypoints is not None and len(trunk_waypoints) != n_trunk + 1:
        raise ValueError(
            f"trunk_waypoints must have n_trunk + 1 = {n_trunk + 1} entries, "
            f"got {len(trunk_waypoints)}."
        )

    rng = np.random.default_rng(seed)
    Z_base = base_kv ** 2 / base_mva

    # ------------------------------------------------------------------
    # 1. Place trunk bus coordinates
    # ------------------------------------------------------------------
    # trunk_coords[i] = (x_km, y_km) for bus i, i = 0 (root) … n_trunk
    if trunk_waypoints is not None:
        trunk_coords: List[Tuple[float, float]] = list(trunk_waypoints)
    else:
        # Straight spine: root at origin, trunk goes up (+y)
        trunk_coords = [
            (0.0, i * trunk_spacing_km)
            for i in range(n_trunk + 1)
        ]

    # ------------------------------------------------------------------
    # 2. Build bus and line lists
    #    Bus numbering:
    #      0            — root (substation)
    #      1 … n_trunk  — trunk buses
    #      n_trunk+1 …  — lateral buses (depth-first per branch)
    # ------------------------------------------------------------------
    buses: List[BusRecord] = []
    lines: List[LineRecord] = []
    coords: Dict[int, Tuple[float, float]] = {}

    # Root bus
    rx, ry = trunk_coords[0]
    buses.append(BusRecord(
        id=0,
        geo_x_km=round(rx, 8),
        geo_y_km=round(ry, 8),
        n_households=0,
        V_min_pu2=round(V_min_pu ** 2, 8),
        V_max_pu2=round(V_max_pu ** 2, 8),
    ))
    coords[0] = (rx, ry)

    next_bus_id = 1

    # Trunk buses and trunk lines
    for i in range(1, n_trunk + 1):
        bx, by = trunk_coords[i]
        buses.append(BusRecord(
            id=next_bus_id,
            geo_x_km=round(bx, 8),
            geo_y_km=round(by, 8),
            n_households=0,          # will be set during allocation
            V_min_pu2=round(V_min_pu ** 2, 8),
            V_max_pu2=round(V_max_pu ** 2, 8),
        ))
        coords[next_bus_id] = (bx, by)

        parent_id = next_bus_id - 1
        px, py = trunk_coords[i - 1]
        length_km = math.hypot(bx - px, by - py)

        lines.append(_make_line(
            len(lines), parent_id, next_bus_id, length_km,
            r_per_km_base, x_per_km_base, perturb_frac, Z_base, P_max_mw, rng,
        ))
        next_bus_id += 1

    # Lateral branches
    lateral_count = 0
    for trunk_idx in range(1, n_trunk + 1):   # trunk bus indices 1..n_trunk
        if (trunk_idx % lateral_interval) != 0:
            continue

        # Direction of trunk segment arriving at trunk_idx
        px, py = trunk_coords[trunk_idx - 1]
        cx, cy = trunk_coords[trunk_idx]
        dx, dy = cx - px, cy - py

        # Alternate left / right
        side = 1 if (lateral_count % 2 == 0) else -1
        ux, uy = _perp_unit(dx, dy, side)
        lateral_count += 1

        # Build lateral chain attached to trunk bus `trunk_idx`
        parent_id = trunk_idx   # trunk bus id equals trunk_idx (bus 0 is root)
        for depth in range(lateral_depth):
            ox, oy = coords[parent_id]
            bx = ox + ux * lateral_spacing_km
            by = oy + uy * lateral_spacing_km

            buses.append(BusRecord(
                id=next_bus_id,
                geo_x_km=round(bx, 8),
                geo_y_km=round(by, 8),
                n_households=0,
                V_min_pu2=round(V_min_pu ** 2, 8),
                V_max_pu2=round(V_max_pu ** 2, 8),
            ))
            coords[next_bus_id] = (bx, by)

            lines.append(_make_line(
                len(lines), parent_id, next_bus_id, lateral_spacing_km,
                r_per_km_base, x_per_km_base, perturb_frac, Z_base, P_max_mw, rng,
            ))
            parent_id = next_bus_id
            next_bus_id += 1

    # ------------------------------------------------------------------
    # 3. Household allocation (Dirichlet + largest-remainder)
    #    Allocated only to non-root buses; trunk buses get households too.
    # ------------------------------------------------------------------
    n_buses_total = len(buses)
    load_bus_ids = [b.id for b in buses if b.id != 0]
    n_load = len(load_bus_ids)

    weights = rng.dirichlet(np.ones(n_load))
    raw = weights * total_households
    hh_floor = np.floor(raw).astype(int)
    remainder = total_households - int(hh_floor.sum())
    fracs = raw - hh_floor
    top_idx = np.argsort(fracs)[::-1][:remainder]
    hh_floor[top_idx] += 1

    assert int(hh_floor.sum()) == total_households

    hh_map: Dict[int, int] = {
        load_bus_ids[i]: int(hh_floor[i]) for i in range(n_load)
    }
    for b in buses:
        b.n_households = hh_map.get(b.id, 0)

    return FeederData(
        buses=buses,
        lines=lines,
        root_bus=0,
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
# Internal: line record factory
# ---------------------------------------------------------------------------

def _make_line(
    line_id: int,
    from_bus: int,
    to_bus: int,
    length_km: float,
    r_per_km: float,
    x_per_km: float,
    perturb_frac: float,
    Z_base: float,
    P_max_mw: float,
    rng: np.random.Generator,
) -> LineRecord:
    """Create a LineRecord with perturbed impedance."""
    r_scale = float(rng.uniform(1.0 - perturb_frac, 1.0 + perturb_frac))
    x_scale = float(rng.uniform(1.0 - perturb_frac, 1.0 + perturb_frac))
    r_ohm = r_per_km * r_scale * length_km
    x_ohm = x_per_km * x_scale * length_km
    return LineRecord(
        id=line_id,
        from_bus=from_bus,
        to_bus=to_bus,
        length_km=round(length_km, 8),
        r_ohm=round(r_ohm, 8),
        x_ohm=round(x_ohm, 8),
        r_pu=round(r_ohm / Z_base, 10),
        x_pu=round(x_ohm / Z_base, 10),
        P_max_mw=P_max_mw,
    )
