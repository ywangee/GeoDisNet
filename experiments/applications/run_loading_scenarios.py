from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import pandapower as pp
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on any OS
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from src.network.feeder_builder import build_pp_network


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name":        "sanity",
        "label":       "Sanity",
        "p_kw_per_hh": 1.0,
        "q_kvar_per_hh": 0.10,
        "description": "Structural check — baseline operating point",
    },
    {
        "name":        "representative",
        "label":       "Representative",
        "p_kw_per_hh": 2.0,
        "q_kvar_per_hh": 0.65,
        "description": "UK residential annual average; pf ≈ 0.951",
    },
    {
        "name":        "stressed",
        "label":       "Stressed",
        "p_kw_per_hh": 4.0,
        "q_kvar_per_hh": 1.30,
        "description": "UK winter evening peak (after-diversity max demand); pf ≈ 0.951",
    },
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run steady-state loading scenarios on the synthetic feeder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data/network",
                   help="Directory containing buses.csv / lines.csv / feeder_params.yaml")
    p.add_argument("--out-dir",  default="data/results",
                   help="Directory for CSV outputs and figures")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Single scenario runner
# ---------------------------------------------------------------------------

def _run_scenario(
    scenario: dict,
    data_dir: pathlib.Path,
) -> dict:
    """
    Build, solve, and extract results for one loading scenario.

    Returns a flat dict of scalar summary metrics plus the full pandapower net.
    """
    net = build_pp_network(
        data_dir / "buses.csv",
        data_dir / "lines.csv",
        data_dir / "feeder_params.yaml",
        p_kw_per_hh=scenario["p_kw_per_hh"],
        q_kvar_per_hh=scenario["q_kvar_per_hh"],
    )

    pp.runpp(net, algorithm="nr", verbose=False)

    if not net.converged:
        return {
            "scenario":          scenario["name"],
            "converged":         False,
            "p_kw_per_hh":       scenario["p_kw_per_hh"],
            "q_kvar_per_hh":     scenario["q_kvar_per_hh"],
            "total_load_mw":     float("nan"),
            "total_load_mvar":   float("nan"),
            "total_losses_kw":   float("nan"),
            "v_min_pu":          float("nan"),
            "v_max_pu":          float("nan"),
            "max_v_dev_pu":      float("nan"),
            "max_branch_load_pct": float("nan"),
            "mean_branch_load_pct": float("nan"),
            "net":               net,
        }

    v_pu   = net.res_bus.vm_pu
    v_min  = v_pu.min()
    v_max  = v_pu.max()
    # voltage deviation measured from 1.0 pu reference
    max_dev = max(abs(v_min - 1.0), abs(v_max - 1.0))

    loading = net.res_line.loading_percent

    return {
        "scenario":             scenario["name"],
        "converged":            True,
        "p_kw_per_hh":          scenario["p_kw_per_hh"],
        "q_kvar_per_hh":        scenario["q_kvar_per_hh"],
        "total_load_mw":        net.res_load.p_mw.sum(),
        "total_load_mvar":      net.res_load.q_mvar.sum(),
        "total_losses_kw":      net.res_line.pl_mw.sum() * 1000.0,
        "v_min_pu":             v_min,
        "v_max_pu":             v_max,
        "max_v_dev_pu":         max_dev,
        "max_branch_load_pct":  loading.max(),
        "mean_branch_load_pct": loading.mean(),
        "net":                  net,
    }


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _voltage_profile_df(net: pp.pandapowerNet) -> pd.DataFrame:
    df = net.res_bus[["vm_pu", "va_degree"]].copy()
    df.index.name = "bus_id"
    df.reset_index(inplace=True)
    return df


def _branch_loading_df(net: pp.pandapowerNet) -> pd.DataFrame:
    df = net.res_line[["loading_percent", "pl_mw", "i_ka"]].copy()
    df.index.name = "line_id"
    df.reset_index(inplace=True)
    # Merge from_bus / to_bus for readability
    df["from_bus"] = net.line["from_bus"].values
    df["to_bus"]   = net.line["to_bus"].values
    return df[["line_id", "from_bus", "to_bus", "loading_percent", "pl_mw", "i_ka"]]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_voltage_profiles(results: list[dict], fig_dir: pathlib.Path, v_lim: float) -> None:
    """Voltage magnitude per bus for all scenarios on one axes."""
    fig, ax = plt.subplots(figsize=(9, 4))

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    markers = ["o", "s", "^"]

    for res, sc, col, mk in zip(results, SCENARIOS, colors, markers):
        if not res["converged"]:
            continue
        v_pu = res["net"].res_bus["vm_pu"].values
        bus_ids = res["net"].res_bus.index.tolist()
        ax.plot(bus_ids, v_pu, color=col, marker=mk, markersize=4,
                linewidth=1.2, label=sc["label"])

    # Voltage limits
    ax.axhline(v_lim, color="red",   linestyle="--", linewidth=1.0, label="V_min")
    ax.axhline(1.0,   color="black", linestyle=":",  linewidth=0.8, label="V_ref")

    ax.set_xlabel("Bus ID")
    ax.set_ylabel("Voltage magnitude [pu]")
    ax.set_title("Voltage profile — synthetic 33-bus feeder")
    ax.legend(fontsize=8.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.98), framealpha=0.9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, linestyle=":", alpha=0.5)
    # Zoom to show inter-scenario differences; keep V_min limit visible
    all_v = [r["net"].res_bus.vm_pu.min() for r in results if r["converged"]]
    y_lo = max(v_lim, min(all_v) - 0.002)
    ax.set_ylim(y_lo, 1.0015)
    fig.tight_layout()

    path = fig_dir / "voltage_profile.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path.resolve()}")


def _plot_branch_loading(result: dict, scenario_name: str, fig_dir: pathlib.Path) -> None:
    """Bar chart of branch loading % for one scenario."""
    net = result["net"]
    loading = net.res_line["loading_percent"].values
    line_ids = net.res_line.index.tolist()

    fig, ax = plt.subplots(figsize=(11, 4))

    bar_colors = ["#C44E52" if v > 80 else "#55A868" if v < 50 else "#DD8452"
                  for v in loading]
    ax.bar(line_ids, loading, color=bar_colors, edgecolor="none", width=0.7)

    ax.axhline(100.0, color="red",    linestyle="--", linewidth=1.0, label="Thermal limit 100 %")
    ax.axhline(80.0,  color="orange", linestyle=":",  linewidth=0.8, label="Warning level 80 %")

    ax.set_xlabel("Line ID")
    ax.set_ylabel("Branch loading [%]")
    ax.set_title(f"Branch loading — {scenario_name} scenario")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_ylim(0, max(110, loading.max() * 1.1))
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    path = fig_dir / f"branch_loading_{scenario_name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _sep(ch: str = "=", n: int = 62) -> None:
    print(ch * n)


def main() -> None:
    args = _parse_args()
    data_dir = pathlib.Path(args.data_dir)
    out_dir  = pathlib.Path(args.out_dir)
    fig_dir  = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    _sep()
    print("Steady-state loading scenarios — synthetic 33-bus feeder")
    _sep()
    print(f"  Input : {data_dir.resolve()}")
    print(f"  Output: {out_dir.resolve()}")

    # -----------------------------------------------------------------
    # Run all scenarios
    # -----------------------------------------------------------------
    results: list[dict] = []
    for sc in SCENARIOS:
        _sep("-")
        print(f"Scenario: {sc['name']}  "
              f"({sc['p_kw_per_hh']} kW/hh, {sc['q_kvar_per_hh']} kVAr/hh)")
        res = _run_scenario(sc, data_dir)
        results.append(res)

        if res["converged"]:
            v_warn = " *** BELOW LIMIT" if res["v_min_pu"] < 0.9 else ""
            bl_warn = " *** OVERLOADED"  if res["max_branch_load_pct"] > 100 else (
                      " ** near limit"   if res["max_branch_load_pct"] > 80  else "")
            print(f"  Converged           : YES")
            print(f"  Total load          : {res['total_load_mw']*1000:.1f} kW active"
                  f"  {res['total_load_mvar']*1000:.1f} kVAr reactive")
            print(f"  Total losses        : {res['total_losses_kw']:.3f} kW  "
                  f"({res['total_losses_kw'] / (res['total_load_mw']*1000)*100:.2f} % of load)")
            print(f"  Voltage min [pu]    : {res['v_min_pu']:.6f}{v_warn}")
            print(f"  Voltage max [pu]    : {res['v_max_pu']:.6f}")
            print(f"  Max V deviation     : {res['max_v_dev_pu']:.6f} pu  "
                  f"({res['max_v_dev_pu']*100:.3f} %)")
            print(f"  Max branch load     : {res['max_branch_load_pct']:.2f} %{bl_warn}")
            print(f"  Mean branch load    : {res['mean_branch_load_pct']:.2f} %")
        else:
            print(f"  Converged           : NO — power flow diverged.")

    # -----------------------------------------------------------------
    # Save per-scenario CSVs
    # -----------------------------------------------------------------
    _sep("-")
    print("Saving CSV outputs...")

    for sc, res in zip(SCENARIOS, results):
        if res["converged"]:
            vdf = _voltage_profile_df(res["net"])
            vdf.to_csv(out_dir / f"voltage_profile_{sc['name']}.csv", index=False)

            bdf = _branch_loading_df(res["net"])
            bdf.to_csv(out_dir / f"branch_loading_{sc['name']}.csv", index=False)

    # Summary table
    summary_rows = []
    for sc, res in zip(SCENARIOS, results):
        summary_rows.append({
            "scenario":             sc["name"],
            "description":          sc["description"],
            "p_kw_per_hh":          sc["p_kw_per_hh"],
            "q_kvar_per_hh":        sc["q_kvar_per_hh"],
            "total_load_kw":        round(res["total_load_mw"] * 1000, 2) if res["converged"] else None,
            "total_load_kvar":      round(res["total_load_mvar"] * 1000, 2) if res["converged"] else None,
            "total_losses_kw":      round(res["total_losses_kw"], 3) if res["converged"] else None,
            "loss_pct_of_load":     round(res["total_losses_kw"] / (res["total_load_mw"]*1000)*100, 3)
                                    if res["converged"] else None,
            "converged":            res["converged"],
            "v_min_pu":             round(res["v_min_pu"], 6) if res["converged"] else None,
            "v_max_pu":             round(res["v_max_pu"], 6) if res["converged"] else None,
            "max_v_dev_pu":         round(res["max_v_dev_pu"], 6) if res["converged"] else None,
            "max_branch_load_pct":  round(res["max_branch_load_pct"], 2) if res["converged"] else None,
            "mean_branch_load_pct": round(res["mean_branch_load_pct"], 2) if res["converged"] else None,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "loading_scenarios_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Summary CSV   : {summary_path.resolve()}")

    converged = [r for r in results if r["converged"]]
    for sc, res in zip(SCENARIOS, results):
        if res["converged"]:
            print(f"  voltage_profile_{sc['name']}.csv  "
                  f"branch_loading_{sc['name']}.csv")

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    _sep("-")
    print("Generating plots...")

    # Read V_min from feeder_params
    import yaml
    params = yaml.safe_load((pathlib.Path(args.data_dir) / "feeder_params.yaml").read_text())
    v_min_limit = float(params.get("V_min_pu", 0.9))

    _plot_voltage_profiles(results, fig_dir, v_min_limit)

    # Branch loading for stressed scenario (last in list)
    stressed_res = results[-1]
    if stressed_res["converged"]:
        _plot_branch_loading(stressed_res, "stressed", fig_dir)

    # -----------------------------------------------------------------
    # Final summary table to stdout
    # -----------------------------------------------------------------
    _sep()
    print("Summary table")
    _sep()
    print(f"{'Scenario':<16} {'Load kW':>8} {'Load kVAr':>10} {'Losses kW':>10} "
          f"{'V_min':>8} {'V_max':>8} {'MaxBr%':>8} {'Conv':>5}")
    print("-" * 78)
    for sc, res in zip(SCENARIOS, results):
        if res["converged"]:
            print(f"{sc['name']:<16} "
                  f"{res['total_load_mw']*1000:>8.1f} "
                  f"{res['total_load_mvar']*1000:>10.1f} "
                  f"{res['total_losses_kw']:>10.3f} "
                  f"{res['v_min_pu']:>8.5f} "
                  f"{res['v_max_pu']:>8.5f} "
                  f"{res['max_branch_load_pct']:>8.2f} "
                  f"{'YES':>5}")
        else:
            print(f"{sc['name']:<16} {'—':>8} {'—':>10} {'—':>10} "
                  f"{'—':>8} {'—':>8} {'—':>8} {'NO':>5}")
    _sep()
    print("Done.")


if __name__ == "__main__":
    main()
