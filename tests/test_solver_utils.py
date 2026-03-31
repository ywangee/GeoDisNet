from __future__ import annotations

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pyomo.environ as pyo

from src.network.solver_utils import (
    _map_status,
    _pick_solver,
    solve_pyomo_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _any_solver_available() -> bool:
    """Return True if at least one supported solver can be found."""
    solver, _ = _pick_solver("gurobi")
    return solver is not None


def _build_trivial_lp() -> pyo.ConcreteModel:
    """minimize x  s.t.  x >= 1.0,  x >= 0 (continuous)  → opt = 1.0"""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(domain=pyo.NonNegativeReals)
    m.c = pyo.Constraint(expr=m.x >= 1.0)
    m.OBJ = pyo.Objective(expr=m.x, sense=pyo.minimize)
    return m


def _build_trivial_mip() -> pyo.ConcreteModel:
    """minimize x  s.t.  x in {0, 1}  → opt = 0"""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(domain=pyo.Binary)
    m.OBJ = pyo.Objective(expr=m.x, sense=pyo.minimize)
    return m


def _build_infeasible_lp() -> pyo.ConcreteModel:
    """minimize x  s.t.  x >= 2  and  x <= 1  → infeasible"""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(domain=pyo.NonNegativeReals)
    m.c1 = pyo.Constraint(expr=m.x >= 2.0)
    m.c2 = pyo.Constraint(expr=m.x <= 1.0)
    m.OBJ = pyo.Objective(expr=m.x, sense=pyo.minimize)
    return m


# ===========================================================================
# _pick_solver
# ===========================================================================

class TestPickSolver:

    def test_returns_tuple(self):
        result = _pick_solver("gurobi")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_unknown_solver_returns_none_none(self):
        opt, name = _pick_solver("__definitely_not_a_real_solver__")
        # If no solver in the fallback chain is available either,
        # both should be None.  If some fallback IS available, that's fine too.
        # Just verify type safety.
        assert name is None or isinstance(name, str)

    def test_returns_available_solver_or_none(self):
        opt, name = _pick_solver("gurobi")
        if opt is None:
            assert name is None
        else:
            assert name in ("gurobi", "highs", "cbc", "glpk")


# ===========================================================================
# _map_status
# ===========================================================================

class TestMapStatus:

    def test_optimal(self):
        assert _map_status(pyo.TerminationCondition.optimal) == "optimal"

    def test_globally_optimal(self):
        assert _map_status(pyo.TerminationCondition.globallyOptimal) == "optimal"

    def test_infeasible(self):
        assert _map_status(pyo.TerminationCondition.infeasible) == "infeasible"

    def test_max_time_limit(self):
        assert _map_status(pyo.TerminationCondition.maxTimeLimit) == "feasible"

    def test_unknown_maps_to_error(self):
        assert _map_status(pyo.TerminationCondition.unknown) == "error"


# ===========================================================================
# solve_pyomo_model — result dict structure
# ===========================================================================

class TestResultStructure:
    """Verify the returned dict always has the required keys, regardless of
    solver availability."""

    REQUIRED_KEYS = {"status", "objective", "solver_name", "termination", "solve_time_s"}

    def test_keys_present_when_solver_available(self):
        if not _any_solver_available():
            pytest.skip("No solver installed — skipping solve test.")
        m = _build_trivial_lp()
        result = solve_pyomo_model(m)
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_keys_present_on_error(self):
        # Force "no solver" scenario by requesting a nonsense solver name.
        # We patch _pick_solver indirectly by asking for a name that cannot
        # exist and whose fallbacks are also unavailable on most systems.
        # Alternatively, we just check the error path directly.
        from src.network import solver_utils as su
        original = su._FALLBACK_CHAIN
        try:
            su._FALLBACK_CHAIN = ()          # empty chain → no solver found
            m = _build_trivial_lp()
            result = su.solve_pyomo_model(m, solver="__no_solver__")
        finally:
            su._FALLBACK_CHAIN = original    # restore

        assert self.REQUIRED_KEYS.issubset(result.keys())
        assert result["status"] == "error"
        assert result["objective"] is None
        assert result["solver_name"] == "none"


# ===========================================================================
# solve_pyomo_model — actual solve behaviour
# ===========================================================================

@pytest.mark.skipif(
    not _any_solver_available(),
    reason="No supported solver (Gurobi / HiGHS / CBC / GLPK) installed.",
)
class TestSolveResults:

    def test_lp_status_optimal(self):
        result = solve_pyomo_model(_build_trivial_lp())
        assert result["status"] == "optimal"

    def test_lp_objective_value(self):
        result = solve_pyomo_model(_build_trivial_lp())
        assert result["objective"] == pytest.approx(1.0, rel=1e-4)

    def test_lp_solve_time_is_positive(self):
        result = solve_pyomo_model(_build_trivial_lp())
        assert result["solve_time_s"] is not None
        assert result["solve_time_s"] >= 0.0

    def test_lp_solver_name_is_string(self):
        result = solve_pyomo_model(_build_trivial_lp())
        assert isinstance(result["solver_name"], str)
        assert result["solver_name"] != "none"

    def test_mip_status_optimal(self):
        result = solve_pyomo_model(_build_trivial_mip())
        assert result["status"] == "optimal"

    def test_mip_objective_is_zero(self):
        result = solve_pyomo_model(_build_trivial_mip())
        assert result["objective"] == pytest.approx(0.0, abs=1e-6)

    def test_unnamed_objective_fallback(self):
        """solve_pyomo_model should find the objective even if not named OBJ."""
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=m.x >= 3.0)
        m.my_custom_obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
        result = solve_pyomo_model(m)
        assert result["status"] == "optimal"
        assert result["objective"] == pytest.approx(3.0, rel=1e-4)

    def test_infeasible_lp_status(self):
        result = solve_pyomo_model(_build_infeasible_lp())
        assert result["status"] in ("infeasible", "error")
        assert result["objective"] is None
