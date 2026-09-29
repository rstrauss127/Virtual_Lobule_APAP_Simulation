# -*- coding: utf-8 -*-
"""Split COMSOL_Porosity_Fields.csv into function-specific COMSOL files.

Input master file columns:
    node_id,x,y,z,porosity,inertial_resistance_1_per_m,
    viscous_resistance_1_per_m2,permeability_m2_from_viscous,status

Output files:
    COMSOL_FunctionFiles/porosity_xyz.csv
    COMSOL_FunctionFiles/permeability_xyz.csv
    COMSOL_FunctionFiles/viscous_resistance_xyz.csv
    COMSOL_FunctionFiles/status_xyz.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def load_rows(path: Path):
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "x", "y", "z", "porosity",
            "viscous_resistance_1_per_m2",
            "permeability_m2_from_viscous",
            "status",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        return list(reader)


def write_file(path: Path, rows, value_col: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "value"])
        for row in rows:
            writer.writerow([row["x"], row["y"], row["z"], row[value_col]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", default=str(PROJECT_ROOT / "COMSOL_Porosity_Fields.csv"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "COMSOL_FunctionFiles"))
    args = parser.parse_args()

    rows = load_rows(Path(args.master))
    outdir = Path(args.outdir)
    write_file(outdir / "porosity_xyz.csv", rows, "porosity")
    write_file(outdir / "permeability_xyz.csv", rows, "permeability_m2_from_viscous")
    write_file(outdir / "viscous_resistance_xyz.csv", rows, "viscous_resistance_1_per_m2")
    write_file(outdir / "status_xyz.csv", rows, "status")
    print(f"Wrote {len(rows)} rows to {outdir}")


if __name__ == "__main__":
    main()
