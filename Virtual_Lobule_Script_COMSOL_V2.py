# -*- coding: utf-8 -*-
"""Replay a patient-specific multi-time COMSOL APAP export through MALD.

This is strict one-way coupling: COMSOL is run separately and writes one file
containing x, y, z and a concentration column for every 1200-s output time.
MALD reads those fields sequentially; no MALD state is passed back to COMSOL.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
from time import time
from scipy.spatial import cKDTree

import MALD_model_Scaled_V5 as MALD
import Initialize_field_Files_V5 as Init

PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)

FIELD_NODES_PATH = project_path(r"Mesh_Volume_Calculations\Field_Nodes.csv")
FIELD_NODE_VOLUME_COLUMN = 4 # Zero-based column index in the headerless Field_Nodes.csv: node_id, x_mm, y_mm, z_mm, node_control_volume_m3
RESULT_BASENAME = "Lobule-Flow-Drug_Simulation_3D_V3-1200.txt"

# The supplied COMSOL export has no header or explicit time values. It contains
# x, y, z followed by one concentration column per stored output time. The
# observed first column is treated as the field at 1200 s. Change this to 0.0
# only if the COMSOL study was actually exported starting at t=0.
COMSOL_FIRST_OUTPUT_TIME_S = 1200.0

# After the final exported field, continue MALD with zero extracellular APAP.
# This is allowed only when the final COMSOL field is negligible relative to
# the peak exported field; otherwise the code raises instead of silently
# discarding a material APAP concentration.
CONTINUE_WITH_ZERO_AFTER_EXPORT = True
FINAL_FIELD_RELATIVE_TOLERANCE = 1.0e-3
COORDINATE_MATCH_RELATIVE_TOLERANCE = 1.0e-8

# Pinto-compatible biological volume used for QSP scaling.
REFERENCE_HEPATOCYTE_VOLUME_M3 = 3.4e-15

# Hydraulic diameter retained from the Pinto resistance calculation.
# This is not used to calculate biological hepatocyte volume.
HYDRAULIC_CELL_DIAMETER_M = 25e-6

# Pinto-compatible assumption: all non-fluid tissue is treated as
# hepatocyte-equivalent tissue.
HEPATOCYTE_FRACTION_OF_SOLID = 1.0

# Ghosh-compatible implicit cellular APAP uptake coefficient.
APAP_UPTAKE_RATE_S = 8.33e-5

# Store full spatial results only at this interval stride. The final interval
# is always stored even when it is not divisible by the stride.
OUTPUT_EVERY_N_INTERVALS = 6

# COMSOL Transport of Diluted Species exports concentration c [mol/m^3].
# For a concentration export, APAP amount added to MALD compartment is:
#     c_apap * porosity * element_volume

def _read_hepatocyte_state(path: Path):
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")
    hepatocytes_array = np.zeros((len(lines), 10))
    hepatocytesZonation_array = np.zeros((len(lines), 3))
    hepatocytesRandom_array = np.zeros(len(lines))

    for i, line in enumerate(lines):
        splitLines = line.split(",")
        if len(splitLines) < 15:
            raise ValueError(
                f"{path}:{i + 1} has {len(splitLines)} columns; expected 15."
            )
        for x in range(10):
            hepatocytes_array[i][x] = float(splitLines[x + 1])
        hepatocytesZonation_array[i][0] = float(splitLines[-4])
        hepatocytesZonation_array[i][1] = float(splitLines[-3])
        hepatocytesZonation_array[i][2] = float(splitLines[-2])
        hepatocytesRandom_array[i] = float(splitLines[-1])
    if np.any(~np.isfinite(hepatocytes_array)):
        raise ValueError(f"{path} contains non-finite MALD state values")
    return lines, hepatocytes_array, hepatocytesZonation_array, hepatocytesRandom_array


def _read_previous_status(path: Path, count: int):
    if not path.exists():
        return np.array(["0.0"] * count, dtype=object)
    lines = path.read_text().splitlines()
    if len(lines) != count:
        raise ValueError(f"{path} has {len(lines)} status rows, expected {count}")
    status = np.array([line.strip() for line in lines], dtype=object)
    return status


def _write_hepatocyte_state(
    path: Path,
    states: np.ndarray,
    zonation: np.ndarray,
    random_values: np.ndarray,
):
    """Write the existing 15-column Field_Hepatocytes-compatible format."""
    node_ids = np.arange(1, len(states) + 1, dtype=int)
    output = np.column_stack(
        (node_ids, states, zonation, random_values)
    )
    np.savetxt(
        path,
        output,
        delimiter=",",
        fmt=["%d"] + ["%.17g"] * 14,
    )


def _classify_status(
    normal: float,
    damaged: float,
    lysed: float,
    regen: float,
    random_value: float,
    previous_status: int,
) -> int:
    """Preserve the current Pinto-style categorical transition rules."""
    if random_value < regen:
        return 3

    total = damaged + lysed + normal
    if (
        total > 0.0
        and random_value < (lysed / total)
        and previous_status != 3
    ):
        return 2

    total = normal + damaged
    if (
        total > 0.0
        and random_value < (damaged / total)
        and previous_status not in (2, 3)
    ):
        return 1

    return previous_status


def _read_field_nodes_and_volumes(path: Path, volume_col: int = FIELD_NODE_VOLUME_COLUMN):
    """Read node coordinates and control volumes from Field_Nodes.csv.

    Expected Field_Nodes.txt format with volume as 5th column:
        element_id,x,y,z,volume_m3

    Extra columns after volume_m3 are allowed and ignored.
    Coordinates must use the same unit as the COMSOL export (currently mm).
    Control volumes must be in m^3.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] <= volume_col:
        raise ValueError(
            f"{path} must contain at least {volume_col + 1} columns: "
            "node_id,x,y,z,node_control_volume_m3"
        )

    ids = data[:, 0].astype(int)
    coordinates = data[:, 1:4]
    volumes = data[:, volume_col]

    if np.any(~np.isfinite(coordinates)):
        raise ValueError(f"{path} contains non-finite coordinates")
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError(f"{path} contains non-finite or non-positive element volumes")
    expected_ids = np.arange(1, len(ids) + 1)
    if not np.array_equal(ids, expected_ids):
        raise ValueError(
            f"{path} node ids must be sequential 1..{len(ids)}; "
            "this prevents APAP/volume/hepatocyte row mismatch."
        )
    return coordinates, volumes


def _read_comsol_apap_time_series(
    path: Path,
    field_coordinates: np.ndarray,
    time_iteration_s: float,
):
    """Load and align a headerless COMSOL multi-time concentration export.

    Expected columns are x, y, z, c(t0), c(t1), ... . Concentration columns
    are returned in Field_Nodes order, even when COMSOL exported a different
    row order.
    """
    if not path.exists():
        raise FileNotFoundError(f"COMSOL APAP export not found: {path}")

    export = np.loadtxt(path, ndmin=2)
    expected_rows = len(field_coordinates)
    if export.shape[0] != expected_rows:
        raise ValueError(
            f"{path} has {export.shape[0]} spatial rows; Field_Nodes has "
            f"{expected_rows}. Export at the same Lagrange/evaluation points."
        )
    if export.shape[1] < 4:
        raise ValueError(
            f"{path} has {export.shape[1]} columns; expected x,y,z and at "
            "least one APAP concentration column."
        )
    if np.any(~np.isfinite(export)):
        raise ValueError(f"{path} contains NaN or infinite values")

    export_coordinates = export[:, :3]
    concentrations = export[:, 3:]

    coordinate_scale = max(float(np.ptp(field_coordinates, axis=0).max()), 1.0)
    coordinate_tolerance = COORDINATE_MATCH_RELATIVE_TOLERANCE * coordinate_scale

    if not np.allclose(
        export_coordinates,
        field_coordinates,
        rtol=COORDINATE_MATCH_RELATIVE_TOLERANCE,
        atol=coordinate_tolerance,
    ):
        distances, export_indices = cKDTree(export_coordinates).query(
            field_coordinates, k=1
        )
        if float(distances.max()) > coordinate_tolerance:
            raise ValueError(
                "COMSOL export coordinates do not match Field_Nodes. "
                f"Maximum nearest-point distance is {distances.max():.6e}; "
                f"allowed tolerance is {coordinate_tolerance:.6e}."
            )
        if len(np.unique(export_indices)) != expected_rows:
            raise ValueError(
                "COMSOL-to-Field_Nodes coordinate matching was not one-to-one."
            )
        concentrations = concentrations[export_indices, :]

    negative_count = int(np.count_nonzero(concentrations < 0.0))
    if negative_count:
        minimum = float(concentrations.min())
        peak = float(np.max(np.abs(concentrations)))
        if minimum < -1.0e-2 * max(peak, 1.0e-30):
            raise ValueError(
                "COMSOL produced materially negative APAP concentrations: "
                f"minimum={minimum:.6e}, peak magnitude={peak:.6e}."
            )
        print(
            f"Warning: clipping {negative_count} small negative COMSOL APAP "
            f"values to zero (minimum={minimum:.6e})."
        )
        concentrations = np.maximum(concentrations, 0.0)

    output_times_s = COMSOL_FIRST_OUTPUT_TIME_S + (
        np.arange(concentrations.shape[1], dtype=float) * float(time_iteration_s)
    )
    return output_times_s, concentrations

def run(
    firstIteration,
    totalIterations,
    timeIteration,
    fileNamePrefixe,
):
    # Record time for runtime stats
    time0 = time()
    # Create folder for the simulation results
    output_dir = project_path(fileNamePrefixe)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Porosity constants retained from the original Pinto workflow.
    porosity_normal = 0.143
    porosity_lysed = 0.246
    cell_sphericity = 0.95
    cell_normal_diameter = 25e-6 # From Pinto
    # No separate damaged-cell diameter was supplied; retaining the same
    # diameter makes damage status neutral in the resistance calculation.
    cell_damaged_diameter = cell_normal_diameter

    field_coordinates, node_control_volumes_m3 = _read_field_nodes_and_volumes(
        FIELD_NODES_PATH
    )
    node_count = len(field_coordinates)

    apap_export = output_dir / RESULT_BASENAME
    output_times_s, exported_concentrations = _read_comsol_apap_time_series(
        apap_export,
        field_coordinates,
        timeIteration,
    )
    exported_field_count = exported_concentrations.shape[1]
    print(
        f"Loaded {exported_field_count} COMSOL APAP fields spanning "
        f"{output_times_s[0]:g} to {output_times_s[-1]:g} s."
    )

    if totalIterations > exported_field_count:
        final_peak = float(np.max(exported_concentrations[:, -1]))
        overall_peak = float(np.max(exported_concentrations))
        final_relative = final_peak / max(overall_peak, 1.0e-30)
        if not CONTINUE_WITH_ZERO_AFTER_EXPORT or final_relative > FINAL_FIELD_RELATIVE_TOLERANCE:
            raise ValueError(
                f"MALD requests {totalIterations} intervals, but the COMSOL "
                f"file contains only {exported_field_count} fields. The final "
                f"field is {final_relative:.6e} of the exported peak, so it "
                "cannot safely be replaced by zero. Extend the COMSOL run or "
                "change the explicit continuation policy."
            )
        print(
            f"COMSOL ends after field {exported_field_count}; remaining MALD "
            f"intervals will use zero extracellular APAP. Final/peak ratio="
            f"{final_relative:.6e}."
        )
    # Calculate the statisical weight of each hepatocyte
# Each numerical node represents a weighted number of hepatocytes.
# The node control volume comes from the tetrahedral mesh, while the
# biological volume per hepatocyte is the Pinto-compatible reference value.
    cell_volume_m3 = REFERENCE_HEPATOCYTE_VOLUME_M3

    hepatocyte_weights = (
        (1.0 - porosity_normal)
        * HEPATOCYTE_FRACTION_OF_SOLID
        * node_control_volumes_m3 
        / cell_volume_m3
    )

    if np.any(~np.isfinite(hepatocyte_weights)):
        raise ValueError("Non-finite hepatocyte weights were calculated")

    if np.any(hepatocyte_weights <= 0.0):
        raise ValueError("All hepatocyte weights must be positive")

    print(
        "Equivalent hepatocytes represented by mesh: "
        f"{hepatocyte_weights.sum():.6e}"
    )
    print(
        "Node hepatocyte-weight range: "
        f"{hepatocyte_weights.min():.6e} to "
        f"{hepatocyte_weights.max():.6e}"
    )

        # Only initialize the state files if this is the first iteration of the simulation.
    if firstIteration == 0:
        # Initialize state files from Field_Nodes.txt.
        Init.Initialize_All_Files()
    # Resume mode: check that the required restart files exist.
    else:
        required_restart_files = [
            project_path("Field_Hepatocytes.txt"),
            project_path("hepatocytes_status_Hepatocytes.txt"),
            project_path(r"Mesh_Volume_Calculations\Field_Zonation.txt"),
        ]
        
        for restart_file in required_restart_files:
            if not restart_file.exists():
                raise FileNotFoundError(
                    f"Cannot resume at iteration {firstIteration}: "
                    f"{restart_file} is missing"
                )

    # Read the restart state once. It remains in memory for the entire run.
    (
        hepatocytesLines,
        hepatocytes_array,
        hepatocytesZonation_array,
        hepatocytesRandom_array,
    ) = _read_hepatocyte_state(project_path("Field_Hepatocytes.txt"))

    expected_rows = len(hepatocytesLines)
    if expected_rows != node_count:
        raise ValueError(
            f"Field_Hepatocytes contains {expected_rows} rows, but "
            f"Field_Nodes contains {node_count} rows."
        )

    previous_status_text = _read_previous_status(
        project_path("hepatocytes_status_Hepatocytes.txt"),
        expected_rows,
    )
    previous_status = np.asarray(
        [int(float(value)) for value in previous_status_text],
        dtype=np.uint8,
    )

    remaining_intervals = totalIterations - firstIteration
    if remaining_intervals <= 0:
        raise ValueError(
            "totalIterations must be greater than firstIteration."
        )

    # Relative MALD times for this invocation. Global iteration n represents
    # the state after interval n, at relative time (n-firstIteration+1)*dt.
    relative_output_times_s = (
        np.arange(1, remaining_intervals + 1, dtype=float)
        * float(timeIteration)
    )
    forcing_times_s = np.arange(
        remaining_intervals + 1,
        dtype=float,
    ) * float(timeIteration)

    save_iterations = [
        n
        for n in range(firstIteration, totalIterations)
        if n % OUTPUT_EVERY_N_INTERVALS == 0
        or n == totalIterations - 1
    ]
    save_local_indices = np.asarray(
        [n - firstIteration for n in save_iterations],
        dtype=int,
    )
    save_slot_by_local_index = {
        local_index: slot
        for slot, local_index in enumerate(save_local_indices)
    }

    saved_states = np.empty(
        (len(save_iterations), expected_rows, 10),
        dtype=np.float64,
    )
    saved_status = np.empty(
        (len(save_iterations), expected_rows),
        dtype=np.uint8,
    )

    print(
        f"Solving {expected_rows} independent MALD histories across "
        f"{remaining_intervals} intervals; saving iterations "
        f"{save_iterations}."
    )

    for k in range(expected_rows):
        node_concentration = np.zeros(
            remaining_intervals + 1,
            dtype=float,
        )

        # At a resumed run, start the interpolated forcing at the preceding
        # COMSOL field. A new run begins at zero extracellular APAP at t=0.
        if firstIteration > 0 and firstIteration - 1 < exported_field_count:
            node_concentration[0] = exported_concentrations[
                k,
                firstIteration - 1,
            ]

        available_start = firstIteration
        available_stop = min(totalIterations, exported_field_count)
        available_count = max(0, available_stop - available_start)
        if available_count:
            node_concentration[1 : available_count + 1] = (
                exported_concentrations[
                    k,
                    available_start:available_stop,
                ]
            )

        solution = MALD.RunMALDHistory(
            hepatocytes_array[k],
            relative_output_times_s,
            forcing_times_s,
            node_concentration,
            hepatocytesZonation_array[k, 0],
            hepatocytesZonation_array[k, 1],
            hepatocytesZonation_array[k, 2],
            APAP_UPTAKE_RATE_S,
            REFERENCE_HEPATOCYTE_VOLUME_M3,
            timeIteration,
        )
        node_history = solution.y.T
        hepatocytes_array[k] = node_history[-1]

        # Status must still be evaluated at every biological interval because
        # the existing categorical transition logic depends on prior status.
        status = int(previous_status[k])
        for local_index in range(remaining_intervals):
            state = node_history[local_index]
            status = _classify_status(
                normal=float(state[3]),
                damaged=float(state[4]),
                lysed=float(state[5]),
                regen=float(state[6]),
                random_value=float(hepatocytesRandom_array[k]),
                previous_status=status,
            )
            save_slot = save_slot_by_local_index.get(local_index)
            if save_slot is not None:
                saved_status[save_slot, k] = status

        saved_states[:, k, :] = node_history[save_local_indices, :]
        previous_status[k] = status

        if (k + 1) % 1000 == 0 or k + 1 == expected_rows:
            print(f"Completed MALD histories for {k + 1}/{expected_rows} nodes")

    total_equivalent_hepatocytes = float(np.sum(hepatocyte_weights))

    # Write only the selected visualization outputs after all node histories
    # are complete. These files contain the post-MALD state at each interval.
    for save_slot, n in enumerate(save_iterations):
        states = saved_states[save_slot]
        statuses = saved_status[save_slot]

        _write_hepatocyte_state(
            output_dir / f"_Used-Field_Hepatocytes-{n}.txt",
            states,
            hepatocytesZonation_array,
            hepatocytesRandom_array,
        )
        np.savetxt(
            output_dir / f"_hepatocytes_status_Hepatocytes-{n}.txt",
            statuses.astype(float),
            fmt="%.1f",
        )

        expected_damaged_cells = float(
            np.sum(hepatocyte_weights * states[:, 4])
        )
        expected_lysed_cells = float(
            np.sum(hepatocyte_weights * states[:, 5])
        )
        damaged_fraction_global = (
            expected_damaged_cells / total_equivalent_hepatocytes
        )
        lysed_fraction_global = (
            expected_lysed_cells / total_equivalent_hepatocytes
        )

        print(
            f"Iteration {n} weighted injury: "
            f"damaged_fraction={damaged_fraction_global:.6e}, "
            f"lysed_fraction={lysed_fraction_global:.6e}, "
            f"expected_damaged_cells={expected_damaged_cells:.6e}, "
            f"expected_lysed_cells={expected_lysed_cells:.6e}"
        )

    # Write restart files once, using the final state from this invocation.
    _write_hepatocyte_state(
        project_path("Field_Hepatocytes.txt"),
        hepatocytes_array,
        hepatocytesZonation_array,
        hepatocytesRandom_array,
    )
    np.savetxt(
        project_path("hepatocytes_status_Hepatocytes.txt"),
        previous_status.astype(float),
        fmt="%.1f",
    )

    time1 = time()
    print(f"Total simulation time (s) = {time1 - time0} for simulation: {fileNamePrefixe}")
    return
