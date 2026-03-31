import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.network.synthetic_feeder import (
    generate_synthetic_feeder,
    validate_feeder,
    export_feeder,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a synthetic radial distribution feeder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-buses",        type=int,   default=33,    help="Total bus count (including root bus 0)")
    p.add_argument("--total-households", type=int, default=252,   help="Households to allocate across non-root buses")
    p.add_argument("--area-km",        type=float, default=2.0,   help="Side length of square service area [km]")
    p.add_argument("--seed",           type=int,   default=42,    help="Random seed for reproducibility")
    p.add_argument("--r-per-km",       type=float, default=0.642, help="Base conductor resistance [Ohm/km]")
    p.add_argument("--x-per-km",       type=float, default=0.083, help="Base conductor reactance [Ohm/km]")
    p.add_argument("--perturbation",   type=float, default=0.10,  help="Fractional impedance perturbation bound (e.g. 0.10 = +/-10%%)")
    p.add_argument("--p-max-mw",       type=float, default=2.0,   help="Thermal line capacity [MW]")
    p.add_argument("--out-dir",        type=str,   default="data/network", help="Output directory for CSV/YAML files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Synthetic feeder generation")
    print("=" * 60)

    data = generate_synthetic_feeder(
        n_buses=args.n_buses,
        area_km=args.area_km,
        total_households=args.total_households,
        base_kv=11.0,
        base_mva=1.0,
        V_min_pu=0.9,
        V_max_pu=1.1,
        r_per_km_base=args.r_per_km,
        x_per_km_base=args.x_per_km,
        perturb_frac=args.perturbation,
        P_max_mw=args.p_max_mw,
        seed=args.seed,
    )

    print("\nValidating feeder structure...")
    validate_feeder(data)
    print("  All checks passed.")

    out_dir = pathlib.Path(args.out_dir)
    export_feeder(data, out_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    r_vals = [ln.r_ohm for ln in data.lines]
    x_vals = [ln.x_ohm for ln in data.lines]
    r_pu   = [ln.r_pu  for ln in data.lines]
    x_pu   = [ln.x_pu  for ln in data.lines]
    hh     = [b.n_households for b in data.buses if b.id != data.root_bus]

    print(f"\nFeeder summary")
    print(f"  Buses            : {data.n_buses}")
    print(f"  Lines            : {data.n_lines}")
    print(f"  Total households : {data.total_households}")
    print(f"  Base kV          : {data.base_kv} kV")
    print(f"  Base MVA         : {data.base_mva} MVA")
    print(f"  Z_base           : {data.Z_base_ohm:.4f} Ohm")

    print(f"\nLine lengths [km]")
    lengths = [ln.length_km for ln in data.lines]
    print(f"  min={min(lengths):.4f}  max={max(lengths):.4f}  "
          f"mean={sum(lengths)/len(lengths):.4f}")

    print(f"\nLine impedances [Ohm]")
    print(f"  r_ohm  min={min(r_vals):.4f}  max={max(r_vals):.4f}  "
          f"mean={sum(r_vals)/len(r_vals):.4f}")
    print(f"  x_ohm  min={min(x_vals):.4f}  max={max(x_vals):.4f}  "
          f"mean={sum(x_vals)/len(x_vals):.4f}")

    print(f"\nLine impedances [pu]  (Z_base = {data.Z_base_ohm:.2f} Ohm)")
    print(f"  r_pu   min={min(r_pu):.6f}  max={max(r_pu):.6f}")
    print(f"  x_pu   min={min(x_pu):.6f}  max={max(x_pu):.6f}")

    print(f"\nHousehold allocation (non-root buses)")
    print(f"  min={min(hh)}  max={max(hh)}  "
          f"mean={sum(hh)/len(hh):.1f}  total={sum(hh)}")

    print(f"\nFiles written to: {out_dir.resolve()}")
    print(f"  buses.csv  ({data.n_buses} rows)")
    print(f"  lines.csv  ({data.n_lines} rows)")
    print(f"  feeder_params.yaml")
    print("=" * 60)


if __name__ == "__main__":
    main()
