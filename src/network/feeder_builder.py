from __future__ import annotations

import math
import pathlib
from typing import Union

import pandas as pd
import pandapower as pp
import yaml


def build_pp_network(
    buses_csv: Union[str, pathlib.Path],
    lines_csv: Union[str, pathlib.Path],
    params_yaml: Union[str, pathlib.Path],
    p_kw_per_hh: float = 1.0,
    q_kvar_per_hh: float = 0.1,
) -> pp.pandapowerNet:
    """
    Build a pandapower network from synthetic feeder data files.

    Steps
    -----
    1. Read buses.csv, lines.csv, feeder_params.yaml.
    2. Create one pandapower bus per row (feeder bus id == pp bus index).
    3. Create ext_grid (slack) at root bus 0 with vm_pu = 1.0.
    4. Create lines via create_line_from_parameters using exported r_ohm / x_ohm.
       Capacitance is set to 0 (lossless approximation, consistent with LinDistFlow).
    5. Add a proportional placeholder load to every non-root bus with n_households > 0.

    Parameters
    ----------
    buses_csv : path-like
        Path to data/network/buses.csv.
    lines_csv : path-like
        Path to data/network/lines.csv.
    params_yaml : path-like
        Path to data/network/feeder_params.yaml.
    p_kw_per_hh : float
        Placeholder active load per household [kW].  Default 1.0 kW.
    q_kvar_per_hh : float
        Placeholder reactive load per household [kVAr].  Default 0.1 kVAr.

    Returns
    -------
    pp.pandapowerNet
        Fully constructed pandapower network (power flow not yet run).
    """
    buses  = pd.read_csv(buses_csv)
    lines  = pd.read_csv(lines_csv)
    params = yaml.safe_load(pathlib.Path(params_yaml).read_text())

    base_kv  = float(params["base_kv"])
    base_mva = float(params["base_mva"])
    root_bus = int(params["root_bus"])

    # ------------------------------------------------------------------
    # Empty network
    # ------------------------------------------------------------------
    net = pp.create_empty_network(f_hz=50.0, sn_mva=base_mva)

    # ------------------------------------------------------------------
    # Buses  (feeder bus id used directly as pandapower bus index)
    # ------------------------------------------------------------------
    for _, row in buses.iterrows():
        bid = int(row["id"])
        pp.create_bus(
            net,
            vn_kv=base_kv,
            name=f"bus_{bid}",
            index=bid,
        )

    # ------------------------------------------------------------------
    # External grid (slack bus) at root
    # ------------------------------------------------------------------
    pp.create_ext_grid(net, bus=root_bus, vm_pu=1.0, name="substation")

    # ------------------------------------------------------------------
    # Lines
    # r_ohm_per_km = r_ohm / length_km  (inverts the assignment in the feeder generator)
    # c_nf_per_km  = 0  (no shunt capacitance, consistent with lossless LinDistFlow)
    # max_i_ka derived from P_max_mw
    # ------------------------------------------------------------------
    for _, row in lines.iterrows():
        from_bus  = int(row["from_bus"])
        to_bus    = int(row["to_bus"])
        length_km = float(row["length_km"])
        r_ohm     = float(row["r_ohm"])
        x_ohm     = float(row["x_ohm"])
        P_max_mw  = float(row["P_max_mw"])

        r_ohm_per_km = r_ohm / length_km
        x_ohm_per_km = x_ohm / length_km
        max_i_ka     = P_max_mw / (math.sqrt(3) * base_kv)

        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=length_km,
            r_ohm_per_km=r_ohm_per_km,
            x_ohm_per_km=x_ohm_per_km,
            c_nf_per_km=0.0,
            max_i_ka=max_i_ka,
            name=f"line_{int(row['id'])}",
        )

    # ------------------------------------------------------------------
    # Placeholder loads proportional to n_households
    # (sanity check only — not the VPP optimisation loads)
    # ------------------------------------------------------------------
    for _, row in buses.iterrows():
        bid  = int(row["id"])
        n_hh = int(row["n_households"])
        if bid == root_bus or n_hh == 0:
            continue
        pp.create_load(
            net,
            bus=bid,
            p_mw=n_hh * p_kw_per_hh / 1000.0,
            q_mvar=n_hh * q_kvar_per_hh / 1000.0,
            name=f"load_bus_{bid}",
        )

    return net
