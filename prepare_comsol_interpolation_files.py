# -*- coding: utf-8 -*-
"""Create COMSOL interpolation function files from Field_Nodes.txt.

This is a first-run helper. It writes uniform initial porosity/permeability files so
COMSOL interpolation functions can be created and tested before the MALD loop runs.

Expected Field_Nodes.txt format, no header:
    node_id,x,y,z

Coordinates are assumed to be meters.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def read_field_nodes(path: Path):
    nodes = []
    with path.open("r", newline="") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_no} expected node_id,x,y,z")
            nodes.append((int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3])))
    if not nodes:
        raise ValueError(f"No nodes found in {path}")
    return nodes


def write_xyz_value(path: Path, nodes, value_getter):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "value"])
        for node_id, x, y, z in nodes:
            writer.writerow([x, y, z, value_getter(node_id, x, y, z)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-nodes", default=str(PROJECT_ROOT / "Mesh_Volume_Calculations" / "Field_Nodes.csv"))
    parser.add_argument("--outdir", default=str(PROJECT_ROOT / "COMSOL_FunctionFiles"))
    parser.add_argument("--porosity", type=float, default=0.143)
    parser.add_argument("--viscous-resistance", type=float, default=2.74e13)
    parser.add_argument("--status", type=float, default=0.0)
    args = parser.parse_args()

    field_nodes = Path(args.field_nodes)
    outdir = Path(args.outdir)
    nodes = read_field_nodes(field_nodes)
    permeability = 1.0 / args.viscous_resistance

    write_xyz_value(outdir / "porosity_xyz.csv", nodes, lambda *_: args.porosity)
    write_xyz_value(outdir / "viscous_resistance_xyz.csv", nodes, lambda *_: args.viscous_resistance)
    write_xyz_value(outdir / "permeability_xyz.csv", nodes, lambda *_: permeability)
    write_xyz_value(outdir / "status_xyz.csv", nodes, lambda *_: args.status)

    # Also create an evaluation-point file for APAP export.
    with (PROJECT_ROOT / "COMSOL_Eval_Points.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "z"])
        writer.writerows(nodes)

    print(f"Wrote {len(nodes)} rows to {outdir}")
    print(f"Initial permeability = {permeability:.12g} m^2")
    print("Also wrote COMSOL_Eval_Points.csv")


if __name__ == "__main__":
    main()
