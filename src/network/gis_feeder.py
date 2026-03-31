from __future__ import annotations

import pathlib

from src.network.synthetic_feeder import FeederData
from src.network.gis_reader import read_gis_graph
from src.network.candidate_graph import extract_candidate_graph
from src.network.mip_feeder import MIPFeederParams, solve_mip_feeder


class GISFeederSource:
    """
    Build a FeederData from a GIS file via the MIP radial synthesis pipeline.

    Parameters
    ----------
    filepath : str or Path
        GeoJSON file containing LineString features (cable segments).
        Shapefile / GeoPackage also accepted when geopandas is installed.
    root_coord : (float, float)
        Coordinates of the substation in the *input* CRS.
        Default: WGS-84 (longitude, latitude).
    crs_input_epsg : int
        EPSG code of ``root_coord`` and the GIS file.  Default 4326 (WGS-84).
    crs_metric_epsg : int
        EPSG code of the projected metric CRS used internally.
        Default 27700 (OSGB36 / British National Grid).
    snap_tol_m : float
        Grid resolution [m] for merging near-coincident endpoints.  Default 5 m.
    max_edge_km : float
        Candidate edges longer than this are removed before the MIP.
        Default 0.5 km.
    C_max : int or None
        Maximum non-root nodes permitted downstream of any single cable.
        ``None`` (default) sets C_max = n−1, which recovers the MST.
    total_households : int
        Households to allocate across non-root buses.  Default 100.
    base_kv : float
        Nominal voltage [kV].  Default 0.4 (LV feeder).
    base_mva : float
        MVA base [MVA].  Default 0.5.
    V_min_pu, V_max_pu : float
        Per-unit voltage bounds.  Defaults 0.94 / 1.06.
    r_per_km : float
        Base conductor resistance [Ω/km].  Default 0.642 (IEEE-33 style).
    x_per_km : float
        Base conductor reactance [Ω/km].  Default 0.083.
    perturb_frac : float
        Fractional impedance perturbation (e.g. 0.05 → ±5 %).  Default 0.05.
    P_max_mw : float
        Thermal capacity applied to every line [MW].  Default 0.5.
    solver : str
        Preferred MIP solver name (Gurobi primary, HiGHS/CBC/GLPK fallback).
    time_limit_s : float
        Solver wall-clock limit [s].  Default 120.
    mip_gap : float
        Relative MIP gap tolerance.  Default 1e-3.
    seed : int
        Random seed for impedance perturbation and household allocation.
    """

    def __init__(
        self,
        filepath: str | pathlib.Path,
        root_coord: tuple[float, float],
        *,
        crs_input_epsg: int = 4326,
        crs_metric_epsg: int = 27700,
        snap_tol_m: float = 5.0,
        max_edge_km: float = 0.5,
        C_max: int | None = None,
        total_households: int = 100,
        base_kv: float = 0.4,
        base_mva: float = 0.5,
        V_min_pu: float = 0.94,
        V_max_pu: float = 1.06,
        r_per_km: float = 0.642,
        x_per_km: float = 0.083,
        perturb_frac: float = 0.05,
        P_max_mw: float = 0.5,
        solver: str = "gurobi",
        time_limit_s: float = 120.0,
        mip_gap: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.filepath        = pathlib.Path(filepath)
        self.root_coord      = root_coord
        self.crs_input_epsg  = crs_input_epsg
        self.crs_metric_epsg = crs_metric_epsg
        self.snap_tol_m      = snap_tol_m
        self.max_edge_km     = max_edge_km
        self.C_max           = C_max
        self.total_households = total_households
        self.base_kv         = base_kv
        self.base_mva        = base_mva
        self.V_min_pu        = V_min_pu
        self.V_max_pu        = V_max_pu
        self.r_per_km        = r_per_km
        self.x_per_km        = x_per_km
        self.perturb_frac    = perturb_frac
        self.P_max_mw        = P_max_mw
        self.solver          = solver
        self.time_limit_s    = time_limit_s
        self.mip_gap         = mip_gap
        self.seed            = seed

    def build(self) -> FeederData:
        """
        Execute the GIS → MIP → FeederData pipeline.

        Returns
        -------
        FeederData
            Fully populated feeder ready for ``validate_feeder`` and
            ``export_feeder``.

        Raises
        ------
        FileNotFoundError
            If the GIS file does not exist.
        ImportError
            If pyproj is not installed.
        RuntimeError
            If the MIP solver returns infeasible or error status.
        """
        # Step 1 — read and project GIS graph
        G_raw, root = read_gis_graph(
            self.filepath,
            self.root_coord,
            crs_input_epsg=self.crs_input_epsg,
            crs_metric_epsg=self.crs_metric_epsg,
            snap_tol_m=self.snap_tol_m,
        )

        # Step 2 — extract candidate subgraph
        G_cand, root = extract_candidate_graph(
            G_raw, root_node=root, max_edge_km=self.max_edge_km
        )

        n = G_cand.number_of_nodes()
        c_max = self.C_max if self.C_max is not None else (n - 1)

        # Step 3 — MIP synthesis
        params = MIPFeederParams(
            C_max=c_max,
            total_households=self.total_households,
            base_kv=self.base_kv,
            base_mva=self.base_mva,
            V_min_pu=self.V_min_pu,
            V_max_pu=self.V_max_pu,
            r_per_km=self.r_per_km,
            x_per_km=self.x_per_km,
            perturb_frac=self.perturb_frac,
            P_max_mw=self.P_max_mw,
            solver=self.solver,
            time_limit_s=self.time_limit_s,
            mip_gap=self.mip_gap,
        )

        result = solve_mip_feeder(G_cand, root, params, seed=self.seed)

        if result.status not in ("optimal", "feasible"):
            raise RuntimeError(
                f"MIP feeder synthesis failed with status '{result.status}'. "
                f"Graph had {result.n_buses} buses and {result.n_edges_candidate} "
                f"candidate edges.  Try increasing C_max or max_edge_km."
            )

        return result.feeder  # type: ignore[return-value]  # never None here
