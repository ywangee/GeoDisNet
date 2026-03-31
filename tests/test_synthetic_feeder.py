import sys
import pathlib
from collections import deque

import pytest

# ---------------------------------------------------------------------------
# Make sure project root is on the path when running without conftest.py
# ---------------------------------------------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.network.synthetic_feeder import (
    generate_synthetic_feeder,
    validate_feeder,
    FeederData,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

N_BUSES         = 33
TOTAL_HH        = 252
DEFAULT_SEED    = 42
PERTURB         = 0.10


@pytest.fixture(scope="module")
def default_feeder() -> FeederData:
    return generate_synthetic_feeder(
        n_buses=N_BUSES,
        total_households=TOTAL_HH,
        seed=DEFAULT_SEED,
        perturb_frac=PERTURB,
    )


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

class TestRadiality:
    def test_edge_count(self, default_feeder):
        """A radial tree on n buses has exactly n-1 edges."""
        assert default_feeder.n_lines == default_feeder.n_buses - 1

    def test_n_buses_matches_request(self, default_feeder):
        assert default_feeder.n_buses == N_BUSES

    def test_n_lines_is_32(self, default_feeder):
        assert default_feeder.n_lines == N_BUSES - 1


class TestConnectivity:
    def test_all_buses_reachable(self, default_feeder):
        """BFS from root must reach every bus in the undirected tree."""
        data = default_feeder
        adj = data.adjacency()
        visited: set = {data.root_bus}
        queue: deque = deque([data.root_bus])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        assert len(visited) == data.n_buses

    def test_root_bus_is_zero(self, default_feeder):
        assert default_feeder.root_bus == 0

    def test_no_self_loops(self, default_feeder):
        for ln in default_feeder.lines:
            assert ln.from_bus != ln.to_bus


class TestImpedances:
    def test_r_ohm_positive(self, default_feeder):
        for ln in default_feeder.lines:
            assert ln.r_ohm > 0, f"Line {ln.id}: r_ohm={ln.r_ohm}"

    def test_x_ohm_positive(self, default_feeder):
        for ln in default_feeder.lines:
            assert ln.x_ohm > 0, f"Line {ln.id}: x_ohm={ln.x_ohm}"

    def test_r_pu_positive(self, default_feeder):
        for ln in default_feeder.lines:
            assert ln.r_pu > 0, f"Line {ln.id}: r_pu={ln.r_pu}"

    def test_x_pu_positive(self, default_feeder):
        for ln in default_feeder.lines:
            assert ln.x_pu > 0, f"Line {ln.id}: x_pu={ln.x_pu}"

    def test_r_pu_consistency(self, default_feeder):
        """r_pu must equal r_ohm / Z_base within floating-point tolerance."""
        Z = default_feeder.Z_base_ohm
        for ln in default_feeder.lines:
            expected = ln.r_ohm / Z
            assert abs(ln.r_pu - expected) < 1e-6, (
                f"Line {ln.id}: r_pu={ln.r_pu:.10f} vs r_ohm/Z={expected:.10f}"
            )

    def test_x_pu_consistency(self, default_feeder):
        """x_pu must equal x_ohm / Z_base within floating-point tolerance."""
        Z = default_feeder.Z_base_ohm
        for ln in default_feeder.lines:
            expected = ln.x_ohm / Z
            assert abs(ln.x_pu - expected) < 1e-6, (
                f"Line {ln.id}: x_pu={ln.x_pu:.10f} vs x_ohm/Z={expected:.10f}"
            )

    def test_z_base_formula(self):
        """Z_base must equal base_kv² / base_mva."""
        data = generate_synthetic_feeder(seed=DEFAULT_SEED, base_kv=11.0, base_mva=1.0)
        assert abs(data.Z_base_ohm - (11.0 ** 2 / 1.0)) < 1e-6

    def test_z_base_formula_alt(self):
        """Check Z_base for a different base."""
        data = generate_synthetic_feeder(seed=DEFAULT_SEED, base_kv=33.0, base_mva=10.0)
        assert abs(data.Z_base_ohm - (33.0 ** 2 / 10.0)) < 1e-6

    def test_perturbation_bounds_r(self, default_feeder):
        """r_ohm must lie within ±10 % of the nominal value."""
        data = default_feeder
        r_base = data.r_per_km_base
        for ln in data.lines:
            r_nom = r_base * ln.length_km
            assert ln.r_ohm >= r_nom * (1.0 - PERTURB) - 1e-7, (
                f"Line {ln.id}: r_ohm={ln.r_ohm:.6f} below r_nom*0.9={r_nom*0.9:.6f}"
            )
            assert ln.r_ohm <= r_nom * (1.0 + PERTURB) + 1e-7, (
                f"Line {ln.id}: r_ohm={ln.r_ohm:.6f} above r_nom*1.1={r_nom*1.1:.6f}"
            )

    def test_perturbation_bounds_x(self, default_feeder):
        """x_ohm must lie within ±10 % of the nominal value."""
        data = default_feeder
        x_base = data.x_per_km_base
        for ln in data.lines:
            x_nom = x_base * ln.length_km
            assert ln.x_ohm >= x_nom * (1.0 - PERTURB) - 1e-7
            assert ln.x_ohm <= x_nom * (1.0 + PERTURB) + 1e-7


# ---------------------------------------------------------------------------
# Household allocation tests
# ---------------------------------------------------------------------------

class TestHouseholds:
    def test_total_households(self, default_feeder):
        assert default_feeder.total_households == TOTAL_HH

    def test_root_has_zero_households(self, default_feeder):
        root = default_feeder.buses[default_feeder.root_bus]
        assert root.n_households == 0

    def test_all_non_root_have_nonneg_households(self, default_feeder):
        for bus in default_feeder.buses:
            if bus.id != default_feeder.root_bus:
                assert bus.n_households >= 0

    def test_at_least_one_non_root_bus_has_households(self, default_feeder):
        non_root_hh = [
            b.n_households for b in default_feeder.buses
            if b.id != default_feeder.root_bus
        ]
        assert sum(non_root_hh) == TOTAL_HH


# ---------------------------------------------------------------------------
# Voltage bound tests
# ---------------------------------------------------------------------------

class TestVoltageBounds:
    def test_v_min_squared(self, default_feeder):
        V_min = default_feeder.V_min_pu
        for bus in default_feeder.buses:
            assert abs(bus.V_min_pu2 - V_min ** 2) < 1e-6

    def test_v_max_squared(self, default_feeder):
        V_max = default_feeder.V_max_pu
        for bus in default_feeder.buses:
            assert abs(bus.V_max_pu2 - V_max ** 2) < 1e-6

    def test_v_min_lt_v_max(self, default_feeder):
        for bus in default_feeder.buses:
            assert bus.V_min_pu2 < bus.V_max_pu2

    def test_default_voltage_limits(self):
        data = generate_synthetic_feeder(seed=DEFAULT_SEED, V_min_pu=0.9, V_max_pu=1.1)
        for bus in data.buses:
            assert abs(bus.V_min_pu2 - 0.81) < 1e-6
            assert abs(bus.V_max_pu2 - 1.21) < 1e-6


# ---------------------------------------------------------------------------
# Reproducibility and seed sensitivity
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_coordinates(self):
        d1 = generate_synthetic_feeder(seed=42)
        d2 = generate_synthetic_feeder(seed=42)
        for b1, b2 in zip(d1.buses, d2.buses):
            assert b1.geo_x_km == b2.geo_x_km
            assert b1.geo_y_km == b2.geo_y_km

    def test_same_seed_same_impedances(self):
        d1 = generate_synthetic_feeder(seed=42)
        d2 = generate_synthetic_feeder(seed=42)
        for l1, l2 in zip(d1.lines, d2.lines):
            assert l1.r_ohm == l2.r_ohm
            assert l1.x_ohm == l2.x_ohm

    def test_different_seed_different_topology(self):
        d1 = generate_synthetic_feeder(seed=42)
        d2 = generate_synthetic_feeder(seed=99)
        coords_differ = any(
            d1.buses[i].geo_x_km != d2.buses[i].geo_x_km
            for i in range(len(d1.buses))
        )
        assert coords_differ, "Different seeds should produce different coordinates."


# ---------------------------------------------------------------------------
# validate_feeder() end-to-end
# ---------------------------------------------------------------------------

class TestValidation:
    def test_validate_passes_on_valid_data(self, default_feeder):
        """validate_feeder should not raise for a correctly generated feeder."""
        validate_feeder(default_feeder)   # must not raise

    def test_validate_custom_params(self):
        data = generate_synthetic_feeder(
            n_buses=10,
            total_households=50,
            base_kv=33.0,
            base_mva=10.0,
            seed=7,
        )
        validate_feeder(data)

    def test_small_feeder(self):
        """Minimal feeder: 2 buses, 1 line."""
        data = generate_synthetic_feeder(
            n_buses=2,
            total_households=5,
            seed=1,
        )
        validate_feeder(data)
        assert data.n_lines == 1
        assert data.total_households == 5
        assert data.buses[0].n_households == 0
