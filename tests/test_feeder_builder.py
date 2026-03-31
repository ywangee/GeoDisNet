import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandapower as pp

from src.network.feeder_builder import build_pp_network
from src.network.synthetic_feeder import generate_synthetic_feeder, export_feeder


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_feeder_files(tmp_path_factory):
    """10-bus feeder exported to a temp directory."""
    tmp = tmp_path_factory.mktemp("small_feeder")
    data = generate_synthetic_feeder(n_buses=10, total_households=50, seed=42)
    export_feeder(data, tmp)
    return tmp


@pytest.fixture(scope="module")
def small_net(small_feeder_files):
    """pandapower network built from the 10-bus feeder."""
    d = small_feeder_files
    return build_pp_network(
        d / "buses.csv",
        d / "lines.csv",
        d / "feeder_params.yaml",
    )


@pytest.fixture(scope="module")
def small_net_solved(small_net):
    """10-bus network with runpp already executed."""
    pp.runpp(small_net, algorithm="nr", verbose=False)
    return small_net


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestNetworkConstruction:
    def test_returns_pandapower_net(self, small_net):
        assert isinstance(small_net, pp.pandapowerNet)

    def test_bus_count(self, small_net):
        assert len(small_net.bus) == 10

    def test_line_count(self, small_net):
        assert len(small_net.line) == 9   # radial: n_buses - 1

    def test_ext_grid_count(self, small_net):
        assert len(small_net.ext_grid) == 1

    def test_ext_grid_at_root(self, small_net):
        assert int(small_net.ext_grid.bus.iloc[0]) == 0

    def test_ext_grid_vm_pu(self, small_net):
        assert small_net.ext_grid.vm_pu.iloc[0] == pytest.approx(1.0)

    def test_bus_index_mapping(self, small_net):
        """Feeder bus id 0..9 must match pandapower bus indices 0..9."""
        assert set(small_net.bus.index.tolist()) == set(range(10))

    def test_no_load_at_root(self, small_net):
        root = int(small_net.ext_grid.bus.iloc[0])
        assert root not in small_net.load.bus.values

    def test_loads_proportional_to_households(self, small_feeder_files, small_net):
        import pandas as pd
        buses = pd.read_csv(small_feeder_files / "buses.csv")
        n_loaded_buses = int((buses["n_households"] > 0).sum()) - 0  # root always 0 hh
        # Every bus with n_households > 0 and not root should have a load
        assert len(small_net.load) == n_loaded_buses

    def test_total_load_proportional(self, small_feeder_files, small_net):
        import pandas as pd
        buses = pd.read_csv(small_feeder_files / "buses.csv")
        expected_p_mw = buses["n_households"].sum() * 1.0 / 1000.0   # 1 kW/hh
        assert small_net.load.p_mw.sum() == pytest.approx(expected_p_mw, rel=1e-6)


# ---------------------------------------------------------------------------
# Power-flow tests
# ---------------------------------------------------------------------------

class TestPowerFlow:
    def test_runpp_converges_small(self, small_net_solved):
        assert small_net_solved.converged is True

    def test_bus_voltages_exist(self, small_net_solved):
        assert len(small_net_solved.res_bus) == 10

    def test_bus_voltages_near_nominal(self, small_net_solved):
        """Under light placeholder load (1 kW/hh) voltages should be very close to 1 pu."""
        vm = small_net_solved.res_bus.vm_pu
        assert vm.min() >= 0.9
        assert vm.max() <= 1.1

    def test_line_results_exist(self, small_net_solved):
        assert len(small_net_solved.res_line) == 9

    def test_line_losses_nonneg(self, small_net_solved):
        assert small_net_solved.res_line.pl_mw.sum() >= 0.0

    def test_load_results_exist(self, small_net_solved):
        assert len(small_net_solved.res_load) == len(small_net_solved.load)

    def test_runpp_converges_default_33bus(self, tmp_path):
        """Full 33-bus feeder with 252 households must converge."""
        data = generate_synthetic_feeder(n_buses=33, total_households=252, seed=42)
        export_feeder(data, tmp_path)
        net = build_pp_network(
            tmp_path / "buses.csv",
            tmp_path / "lines.csv",
            tmp_path / "feeder_params.yaml",
        )
        pp.runpp(net, algorithm="nr", verbose=False)
        assert net.converged is True
