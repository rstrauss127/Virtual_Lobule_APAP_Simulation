# -*- coding: utf-8 -*-
"""Validate COMSOL interpolation function CSV files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def count_field_nodes(path: Path) -> int:
    count = 0
    with path.open("r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def read_values(path: Path):
    values = []
    coords_bad = 0
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"x", "y", "z", "value"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        for row_no, row in enumerate(reader, start=2):
            try:
                x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
                value = float(row["value"])
            except Exception as exc:
                raise ValueError(f"{path}:{row_no} cannot parse numeric x,y,z,value") from exc
            if not all(math.isfinite(v) for v in (x, y, z, value)):
                raise ValueError(f"{path}:{row_no} contains non-finite number")
            if abs(x) > 0.1 or abs(y) > 0.1 or abs(z) > 0.1:
                coords_bad += 1
            values.append(value)
    if not values:
        raise ValueError(f"{path} has no data rows")
    return values, coords_bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function-dir", default=str(PROJECT_ROOT / "COMSOL_FunctionFiles"))
    parser.add_argument("--field-nodes", default=str(PROJECT_ROOT / "Mesh_Volume_Calculations" / "Field_Nodes.csv"))
    args = parser.parse_args()

    expected = count_field_nodes(Path(args.field_nodes))
    files = [
        "porosity_xyz.csv",
        "permeability_xyz.csv",
        "viscous_resistance_xyz.csv",
        "status_xyz.csv",
    ]
    base = Path(args.function_dir)
    for name in files:
        path = base / name
        values, coords_bad = read_values(path)
        if len(values) != expected:
            raise ValueError(f"{path}: row count {len(values)} != Field_Nodes count {expected}")
        print(f"{path}: rows={len(values)} min={min(values):.12g} max={max(values):.12g}")
        if coords_bad:
            print(f"  WARNING: {coords_bad} coordinates exceed 0.1 m; check whether coordinates are in mm instead of m.")

    print("Interpolation file validation passed")


if __name__ == "__main__":
    main()
