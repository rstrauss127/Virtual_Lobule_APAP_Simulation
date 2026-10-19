#!/usr/bin/env python3

"""
****This description is outdated. Need to update for new outputs and usage.*******

Calculate per-tetrahedron volumes and node control volumes from a COMSOL
sectionwise mesh export like test_mesh.txt.

Outputs:
  1) element_volumes.csv
     One row per tetrahedral element.

  2) node_control_volumes.csv
     One row per mesh node. Each tetrahedron volume is split V/4 to its
     four corner nodes. Use this for node-based MALD calculations.

  3) mesh_volume_summary.txt
     Basic volume and mesh-quality checks.

Usage:
  python calculate_comsol_mesh_volumes.py Mesh_From_Comsol.txt
Optional:
  python calculate_comsol_mesh_volumes.py test_mesh.txt --outdir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


MM3_TO_M3 = 1e-9


def parse_comsol_sectionwise_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Parse a COMSOL sectionwise mesh text file with:
      % Coordinates
      ... x y z rows ...
      % Elements (tetrahedra)
      ... n1 n2 n3 n4 rows ...

    Returns
    -------
    coords : np.ndarray, shape (n_nodes, 3)
        Node coordinates in the file's native length unit.
    tets : np.ndarray, shape (n_tets, 4)
        Tetrahedral connectivity using COMSOL's 1-based node numbering.
    meta : dict
        Metadata parsed from % header lines.
    """
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    lines = path.read_text(errors="replace").splitlines()

    meta: dict[str, str | int] = {}
    for line in lines:
        if not line.startswith("%"):
            continue
        text = line[1:].strip()
        if ":" in text:
            key, value = text.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            value = value.strip()
            if re.fullmatch(r"[-+]?\d+", value):
                value = int(value)
            meta[key] = value

    try:
        coord_start = next(i for i, line in enumerate(lines) if line.strip() == "% Coordinates") + 1
    except StopIteration as exc:
        raise ValueError("Could not find '% Coordinates' section.") from exc

    try:
        elem_start = next(i for i, line in enumerate(lines) if line.strip().startswith("% Elements ("))
    except StopIteration as exc:
        raise ValueError("Could not find '% Elements' section.") from exc

    element_header = lines[elem_start].strip().lower()
    if "tetrahedra" not in element_header:
        raise ValueError(
            f"This script expects tetrahedra, but found section header: {lines[elem_start]!r}"
        )

    coord_lines = [line for line in lines[coord_start:elem_start] if line.strip() and not line.startswith("%")]
    elem_lines = [line for line in lines[elem_start + 1 :] if line.strip() and not line.startswith("%")]

    coords = np.array([[float(v) for v in line.split()] for line in coord_lines], dtype=float)
    tets = np.array([[int(v) for v in line.split()] for line in elem_lines], dtype=np.int64)

    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coordinate array with shape (n, 3); got {coords.shape}")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"Expected tetrahedron array with shape (n, 4); got {tets.shape}")

    n_nodes_header = meta.get("nodes")
    n_elements_header = meta.get("elements")
    if isinstance(n_nodes_header, int) and n_nodes_header != coords.shape[0]:
        raise ValueError(f"Header says {n_nodes_header} nodes, parsed {coords.shape[0]} nodes.")
    if isinstance(n_elements_header, int) and n_elements_header != tets.shape[0]:
        raise ValueError(f"Header says {n_elements_header} elements, parsed {tets.shape[0]} elements.")

    if tets.min() < 1 or tets.max() > coords.shape[0]:
        raise ValueError(
            f"Element connectivity has node ids outside valid range 1..{coords.shape[0]}."
        )

    return coords, tets, meta


def tetra_volumes(coords_native: np.ndarray, tets_one_based: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate tetrahedron volumes and centroids in native units.

    Returns
    -------
    volumes_native3 : np.ndarray, shape (n_tets,)
        Absolute tetrahedron volumes in native length unit cubed.
    centroids_native : np.ndarray, shape (n_tets, 3)
        Tetrahedron centroids in native length unit.
    """
    # Convert COMSOL 1-based node ids to Python 0-based indices.
    tet_idx = tets_one_based - 1

    p0 = coords_native[tet_idx[:, 0]]
    p1 = coords_native[tet_idx[:, 1]]
    p2 = coords_native[tet_idx[:, 2]]
    p3 = coords_native[tet_idx[:, 3]]

    # Signed volume = dot((p1-p0), cross((p2-p0), (p3-p0))) / 6
    signed = np.einsum("ij,ij->i", p1 - p0, np.cross(p2 - p0, p3 - p0)) / 6.0
    volumes = np.abs(signed)
    centroids = (p0 + p1 + p2 + p3) / 4.0

    return volumes, centroids


def node_control_volumes(n_nodes: int, tets_one_based: np.ndarray, element_volumes_native3: np.ndarray) -> np.ndarray:
    """
    Build a node control volume by assigning V/4 from each tetrahedron to each
    of that tetrahedron's four vertices.
    """
    node_volumes = np.zeros(n_nodes, dtype=float)
    tet_idx = tets_one_based - 1

    # Add V/4 to each of the four vertices.
    share = element_volumes_native3 / 4.0
    for local_node in range(4):
        np.add.at(node_volumes, tet_idx[:, local_node], share)

    return node_volumes


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute COMSOL tetrahedral mesh volumes.")
    parser.add_argument("mesh_file", type=Path, help="Path to COMSOL sectionwise mesh export, e.g. test_mesh.txt")
    parser.add_argument("--outdir", type=Path, default=None, help="Output directory. Default: same folder as mesh_file")

    args = parser.parse_args()

    outdir = args.outdir if args.outdir is not None else args.mesh_file.parent
    outdir.mkdir(parents=True, exist_ok=True)

    coords, tets, meta = parse_comsol_sectionwise_mesh(args.mesh_file)
    volumes_native3, centroids_native = tetra_volumes(coords, tets)
    node_volumes_native3 = node_control_volumes(coords.shape[0], tets, volumes_native3)

    # Convert to SI units (m^3) for the volume columns
    element_volumes_m3 = volumes_native3 * MM3_TO_M3
    node_volumes_m3 = node_volumes_native3 * MM3_TO_M3

    length_unit = str(meta.get("length_unit", "unknown")).strip().lower()

    # This workflow currently assumes that COMSOL exported coordinates in mm.
    # Stop instead of silently applying the wrong conversion.
    if length_unit != "mm":
        raise ValueError(
            f"Expected COMSOL mesh coordinates in mm, but header reports "
            f"{length_unit!r}. Update the conversion before continuing."
        )

    # Coordinates remain in mm because the PV wall, CV wall, and COMSOL
    # interpolation coordinates use the same native coordinate system.
    # Volumes are converted to SI units for the QSP calculations.
    element_volumes_m3 = volumes_native3 * MM3_TO_M3
    node_control_volumes_m3 = node_volumes_native3 * MM3_TO_M3

    element_df = pd.DataFrame(
        {
            "element_id": np.arange(1, tets.shape[0] + 1),
            "n1": tets[:, 0],
            "n2": tets[:, 1],
            "n3": tets[:, 2],
            "n4": tets[:, 3],
            "centroid_x_mm": centroids_native[:, 0],
            "centroid_y_mm": centroids_native[:, 1],
            "centroid_z_mm": centroids_native[:, 2],
            "element_volume_mm3": volumes_native3,
            "element_volume_m3": element_volumes_m3,
        }
    )

    # IMPORTANT:
    # Field_Nodes.csv has no header because the current initializer and COMSOL
    # backend expect every row to begin with a numeric node ID.
    #
    # Column order:
    # node_id, x_mm, y_mm, z_mm, node_control_volume_m3
    node_df = pd.DataFrame(
        {
            "node_id": np.arange(1, coords.shape[0] + 1),
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "node_control_volume_m3": node_control_volumes_m3,
        }
    )

    element_csv = outdir / "Element_Volumes.csv"
    node_csv = outdir / "Field_Nodes.csv"
    summary_txt = outdir / "mesh_volume_summary.txt"

    # Element_Volumes.csv is a diagnostic file and can retain its header.
    element_df.to_csv(element_csv, index=False)
    # The downstream code currently expects a headerless numeric file.
    # The resulting Field_Nodes.csv must look like this:
    #   1,x_mm,y_mm,z_mm,node_control_volume_m3
    #   2,x_mm,y_mm,z_mm,node_control_volume_m3
    node_df.to_csv(node_csv, index=False, header=False)

    total_element_mm3 = float(volumes_native3.sum())
    total_node_mm3 = float(node_volumes_native3.sum())

    total_element_m3 = float(element_volumes_m3.sum())
    total_node_m3 = float(node_control_volumes_m3.sum())

    # Detect zero and nearly degenerate tetrahedra.
    volume_tolerance_mm3 = max(
        float(np.median(volumes_native3)) * 1e-12,
        np.finfo(float).tiny,
    )

    degenerate_count = int(
        np.count_nonzero(volumes_native3 <= volume_tolerance_mm3)
    )

    summary = f"""
        Input mesh file: {args.mesh_file}
        Length unit from COMSOL header: {meta.get('length_unit', 'unknown')}
        Nodes parsed: {coords.shape[0]}
        Tetrahedral elements parsed: {tets.shape[0]}

        Total element volume: {total_element_mm3:.12e} mm^3
        Total node control volume: {total_node_mm3:.12e} mm^3
        Node sum - element sum: {(total_node_mm3 - total_element_mm3):.12e} mm^3

        Total element volume: {total_element_m3:.12e} m^3
        Total node control volume: {total_node_m3:.12e} m^3
        Node sum - element sum: {(total_node_m3 - total_element_m3):.12e} m^3

        Minimum element volume: {float(element_volumes_m3.min()):.12e} m^3
        Median element volume: {float(np.median(element_volumes_m3)):.12e} m^3
        Mean element volume: {float(element_volumes_m3.mean()):.12e} m^3
        Maximum element volume: {float(element_volumes_m3.max()):.12e} m^3
        Degenerate or nearly degenerate elements: {degenerate_count}

        Files written:
        - {element_csv}
        - {node_csv}
        - {summary_txt}
    """.strip()

    summary_txt.write_text(summary + "\n")
    print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
