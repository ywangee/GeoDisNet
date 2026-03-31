from __future__ import annotations

import time
from typing import Any

import pyomo.environ as pyo


# Fallback chain — tried in order when the requested solver is unavailable.
_FALLBACK_CHAIN: tuple[str, ...] = ("gurobi", "highs", "cbc", "glpk")

# Pyomo TerminationConditions that indicate an optimal solution.
_OPTIMAL = frozenset({
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.globallyOptimal,
})

# Conditions that indicate a feasible (but possibly non-optimal) solution.
_FEASIBLE = frozenset({
    pyo.TerminationCondition.feasible,
    pyo.TerminationCondition.maxTimeLimit,
    pyo.TerminationCondition.maxIterations,
})


def solve_pyomo_model(
    model: pyo.ConcreteModel,
    solver: str = "gurobi",
    time_limit_s: float = 120.0,
    mip_gap: float = 1e-3,
    tee: bool = False,
) -> dict[str, Any]:
    """
    Solve a Pyomo model and return a standardised result dictionary.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Fully built Pyomo model with at least one active Objective.
    solver : str
        Preferred solver (e.g. ``"gurobi"``, ``"highs"``).
        Falls back automatically if unavailable.
    time_limit_s : float
        Wall-clock time limit [s].
    mip_gap : float
        Relative MIP optimality gap tolerance.
    tee : bool
        Stream solver log to stdout when True.

    Returns
    -------
    dict with keys:
        ``status``       — ``"optimal"``, ``"feasible"``, ``"infeasible"``,
                           or ``"error"``
        ``objective``    — float objective value, or None if no solution
        ``solver_name``  — name of the solver actually used
        ``termination``  — Pyomo TerminationCondition as a string
        ``solve_time_s`` — wall-clock elapsed time [s], or None on hard error
    """
    opt, used_solver = _pick_solver(solver)

    if opt is None:
        return {
            "status": "error",
            "objective": None,
            "solver_name": "none",
            "termination": "no_solver_available",
            "solve_time_s": None,
        }

    _set_options(opt, used_solver, time_limit_s, mip_gap)

    t0 = time.perf_counter()
    try:
        results = opt.solve(model, tee=tee, load_solutions=True)
    except Exception as exc:
        return {
            "status": "error",
            "objective": None,
            "solver_name": used_solver,
            "termination": str(exc),
            "solve_time_s": time.perf_counter() - t0,
        }
    elapsed = time.perf_counter() - t0

    tc = results.solver.termination_condition
    status = _map_status(tc)
    obj_val = _read_objective(model) if status in ("optimal", "feasible") else None

    return {
        "status": status,
        "objective": obj_val,
        "solver_name": used_solver,
        "termination": str(tc),
        "solve_time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_solver(preferred: str) -> tuple[Any | None, str | None]:
    """Return the first available (solver_instance, name) from the fallback chain."""
    chain = [preferred] + [s for s in _FALLBACK_CHAIN if s != preferred]
    for name in chain:
        try:
            candidate = pyo.SolverFactory(name)
            if candidate.available():
                return candidate, name
        except Exception:
            continue
    return None, None


def _set_options(
    opt: Any,
    solver_name: str,
    time_limit_s: float,
    mip_gap: float,
) -> None:
    """Apply time limit and MIP gap to a solver instance (best-effort)."""
    try:
        if solver_name == "gurobi":
            opt.options["TimeLimit"] = time_limit_s
            opt.options["MIPGap"] = mip_gap
        elif solver_name == "highs":
            opt.options["time_limit"] = time_limit_s
            opt.options["mip_rel_gap"] = mip_gap
        elif solver_name == "cbc":
            opt.options["sec"] = int(time_limit_s)
            opt.options["ratio"] = mip_gap
        elif solver_name == "glpk":
            opt.options["tmlim"] = int(time_limit_s)
            opt.options["mipgap"] = mip_gap
    except Exception:
        pass  # option failures are non-fatal; solver uses its defaults


def _map_status(tc: pyo.TerminationCondition) -> str:
    """Map a Pyomo TerminationCondition to a simple status string."""
    if tc in _OPTIMAL:
        return "optimal"
    if tc in _FEASIBLE:
        return "feasible"
    if tc == pyo.TerminationCondition.infeasible:
        return "infeasible"
    return "error"


def _read_objective(model: pyo.ConcreteModel) -> float | None:
    """Extract the objective value from a solved model."""
    # Try the conventional name used throughout this project first.
    try:
        return float(pyo.value(model.OBJ))
    except Exception:
        pass
    # Fallback: find any active Objective component.
    try:
        for obj in model.component_objects(pyo.Objective, active=True):
            return float(pyo.value(obj))
    except Exception:
        pass
    return None
