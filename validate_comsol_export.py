# -*- coding: utf-8 -*-
"""Validate that a COMSOL APAP export can be consumed by the MALD loop."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from comsol_backend import read_field_nodes, read_apap_export

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--field-nodes", default=str(PROJECT_ROOT / "Mesh_Volume_Calculations" / "Field_Nodes.csv"))
    parser.add_argument("--apap-export", required=True)
    args = parser.parse_args()

    nodes = read_field_nodes(Path(args.field_nodes))
    values = read_apap_export(Path(args.apap_export), expected_rows=len(nodes))

    print("COMSOL export validation passed")
    print(f"rows: {len(values)}")
    print(f"min:  {min(values):.12g}")
    print(f"mean: {mean(values):.12g}")
    print(f"max:  {max(values):.12g}")


if __name__ == "__main__":
    main()
