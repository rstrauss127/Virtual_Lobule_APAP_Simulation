# -*- coding: utf-8 -*-
"""
Backup file as of 10/25/2026

Replay a patient-specific multi-time COMSOL APAP export through MALD.

This is strict one-way coupling: COMSOL is run separately and writes one file
containing x, y, z and a concentration column for every 1200-s output time.
MALD reads those fields sequentially; no MALD state is passed back to COMSOL.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import csv
import json
import hashlib
import numpy as np
from time import time
from scipy.spatial import cKDTree
from scipy import __version__ as SCIPY_VERSION

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
# APAP_UPTAKE_RATE_S = 8.33e-5
# Pinto-compatible implicit cellular APAP uptake coefficient.
APAP_UPTAKE_RATE_S = 1.0

# Store full spatial results only at this interval stride. The final interval
# is always stored even when it is not divisible by the stride.
OUTPUT_EVERY_N_INTERVALS = 6

# Optional sequential Field_Nodes IDs for fixed, same-mesh comparison probes.
# None selects three distinct low/middle/high initial-GSH parameter examples.
# These automatic examples are NOT anatomical PV/CV location assignments.
COMPARISON_PROBE_IDS = None


class _ComparisonRecorder:
    """Compact diagnostics for discrete APAP additions and unforced MALD.

    Stores weighted means, not additional full spatial snapshots. Every
    invocation gets its own interval-range directory, including restarts.
    Extrema are sampled at initial/post-addition/end states; no AUC is inferred
    from those sparse values across the discontinuous APAP additions.
    """

    state_columns = [
        "mean_APAP_mol_per_cell", "mean_NAPQI_mol_per_cell",
        "mean_GSH_mol_per_cell", "healthy_fraction", "damaged_fraction",
        "lysed_fraction", "mean_regeneration_state", "mean_AST_IU_L",
        "mean_ALT_IU_L", "mean_clotting_state",
    ]
    probe_columns = [
        "APAP_mol_per_cell", "NAPQI_mol_per_cell", "GSH_mol_per_cell",
        "H", "Z", "Lys", "Regen", "AST_IU_L", "ALT_IU_L", "Clot",
    ]

    def __init__(self, output_dir, first, total, dt, coordinates, volumes,
                 weights, initial_states, zonation, random_values,
                 source_times, probe_ids, settings):
        self.first, self.total, self.dt = int(first), int(total), float(dt)
        self.count = self.total - self.first
        self.run_id = output_dir.name
        self.directory = output_dir / "Comparison" / (
            f"intervals_{self.first:06d}_{self.total - 1:06d}"
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.weights = weights / float(np.sum(weights))
        self.volumes = volumes / float(np.sum(volumes))
        self.total_weight = float(np.sum(weights))
        self.source_times = source_times
        self.coordinates = coordinates
        self.initial_mean = self.weights @ initial_states
        self.post_mean = np.zeros((self.count, 10))
        self.end_mean = np.zeros((self.count, 10))
        self.external_mean = np.zeros(self.count)
        self.added_mean = np.zeros(self.count)
        self.initial_hzl = initial_states[:, 3:6].sum(axis=1)
        self.max_hzl_drift = 0.0

        if probe_ids is None:
            order = np.argsort(initial_states[:, 2], kind="stable")
            chosen = {}
            for label, position in (("low_initial_GSH", 0),
                                    ("middle_initial_GSH", len(order) // 2),
                                    ("high_initial_GSH", len(order) - 1)):
                chosen.setdefault(int(order[position]), label)
        else:
            ids = list(probe_ids)
            if any(isinstance(i, bool) or int(i) != i or
                   not 1 <= int(i) <= len(weights) for i in ids):
                raise ValueError("Comparison probe IDs must be integers in 1..node_count")
            if len(set(ids)) != len(ids):
                raise ValueError("Comparison probe IDs must be unique")
            chosen = {int(i) - 1: "fixed_node" for i in ids}
        self.probes = chosen
        self.probe_rows = []
        start_phase = "initial" if self.first == 0 else "restart_start"
        self.start_phase = start_phase
        for k in self.probes:
            self._probe(k, self.first - 1, self.first * self.dt,
                        start_phase, initial_states[k])

        def digest(array):
            return hashlib.sha256(
                memoryview(np.ascontiguousarray(array)).cast("B")
            ).hexdigest()

        self.settings = dict(settings)
        self.settings.update({
            "schema_version": 1, "status": "running", "run_id": self.run_id,
            "first_iteration": self.first, "total_iterations": self.total,
            "start_time_s": self.first * self.dt,
            "end_time_s": self.total * self.dt, "interval_s": self.dt,
            "node_count": len(weights), "total_tissue_volume_m3": float(sum(volumes)),
            "equivalent_hepatocytes": self.total_weight,
            "initial_states_sha256": digest(initial_states),
            "zonation_sha256": digest(zonation),
            "random_values_sha256": digest(random_values),
            "coordinates_sha256": digest(coordinates),
            "control_volumes_sha256": digest(volumes),
            "probe_selection": "initial_GSH_examples" if probe_ids is None else "fixed_ids",
            "probes": [{"node_id": k + 1, "label": label,
                        "xyz_mm": coordinates[k].tolist()} for k, label in chosen.items()],
            "time_convention": "APAP addition at interval start; COMSOL sample time recorded separately",
            "cumulative_input_scope": "this invocation only; restarts exclude earlier additions",
            "extrema_scope": "recorded initial, post_addition, interval_end states only; not continuous extrema",
        })
        self._write_settings()

    def _write_settings(self):
        (self.directory / "run_settings.json").write_text(
            json.dumps(self.settings, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    def _probe(self, k, interval, t, phase, state):
        self.probe_rows.append([
            self.run_id, self.probes[k], k + 1,
            *self.coordinates[k].tolist(), interval, t, phase, *state.tolist(),
        ])

    def addition(self, k, j, concentration, added, state):
        self.post_mean[j] += self.weights[k] * state
        self.external_mean[j] += self.volumes[k] * concentration
        self.added_mean[j] += self.weights[k] * added
        if k in self.probes:
            n = self.first + j
            self._probe(k, n, n * self.dt, "post_addition", state)

    def end(self, k, j, state):
        self.end_mean[j] += self.weights[k] * state
        self.max_hzl_drift = max(
            self.max_hzl_drift,
            abs(float(state[3:6].sum()) - self.initial_hzl[k]),
        )
        if k in self.probes:
            n = self.first + j
            self._probe(k, n, (n + 1) * self.dt, "interval_end", state)

    def _csv(self, name, columns, rows):
        with (self.directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)

    def finish(self, nfev, njev, nlu, runtime_s):
        rows = [[self.run_id, self.first - 1, self.first * self.dt,
                 self.start_phase, *self.initial_mean.tolist()]]
        inputs = []
        cumulative = 0.0
        for j, n in enumerate(range(self.first, self.total)):
            rows.extend([
                [self.run_id, n, n * self.dt, "post_addition", *self.post_mean[j].tolist()],
                [self.run_id, n, (n + 1) * self.dt, "interval_end", *self.end_mean[j].tolist()],
            ])
            added_total = self.added_mean[j] * self.total_weight
            cumulative += added_total
            from_export = n < len(self.source_times)
            inputs.append([
                self.run_id, n, n * self.dt, (n + 1) * self.dt,
                float(self.source_times[n]) if from_export else "",
                "COMSOL_export" if from_export else "zero_continuation",
                self.external_mean[j], self.added_mean[j], added_total, cumulative,
            ])
        self._csv("comparison_timeseries.csv",
                  ["run_id", "interval", "time_s", "phase", *self.state_columns], rows)
        self._csv("comparison_inputs.csv", [
            "run_id", "interval", "interval_start_s", "interval_end_s",
            "comsol_sample_time_s", "input_source",
            "mean_extracellular_APAP_mol_m3", "mean_added_APAP_mol_per_cell",
            "total_added_APAP_mol", "cumulative_added_APAP_mol_this_invocation",
        ], inputs)
        self.probe_rows.sort(key=lambda row: (row[2], row[6], row[7]))
        self._csv("comparison_probes.csv", [
            "run_id", "probe_label", "node_id", "x_mm", "y_mm", "z_mm",
            "interval", "time_s", "phase", *self.probe_columns,
        ], self.probe_rows)
        sampled = np.vstack((self.initial_mean[None, :], self.post_mean, self.end_mean))
        metrics = [
            ["sampled_max_mean_APAP_mol_per_cell", float(sampled[:, 0].max())],
            ["sampled_max_mean_NAPQI_mol_per_cell", float(sampled[:, 1].max())],
            ["sampled_min_mean_GSH_mol_per_cell", float(sampled[:, 2].min())],
            ["sampled_max_damaged_fraction", float(sampled[:, 4].max())],
            ["sampled_max_lysed_fraction", float(sampled[:, 5].max())],
            ["total_added_APAP_mol_this_invocation", cumulative],
            ["max_node_H_Z_Lys_drift_from_start", self.max_hzl_drift],
        ]
        metrics.extend(["final_" + column, float(value)]
                       for column, value in zip(self.state_columns, self.end_mean[-1]))
        self._csv("comparison_metrics.csv", ["metric", "value"], metrics)
        self.settings.update({"status": "completed", "solver_nfev": int(nfev),
                              "solver_njev": int(njev), "solver_nlu": int(nlu),
                              "runtime_s_before_comparison_write": float(runtime_s)})
        self._write_settings()
        print(f"Comparison outputs: {self.directory}")

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
    mald_rtol=None,
    mald_atol=None,
    comparison_probe_ids=None,
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
    effective_rtol = (
    MALD.MALD_RTOL if mald_rtol is None else float(mald_rtol)
    )

    effective_atol = np.asarray(
        MALD.MALD_ATOL if mald_atol is None else mald_atol,
        dtype=float,
    )

    print(
        f"Radau tolerances: rtol={effective_rtol:.3e}; "
        f"atol={effective_atol.tolist()}"
    )

    solver_nfev = 0
    solver_njev = 0
    solver_nlu = 0

    # Logging only: the discrete additions, solver calls and state updates below
    # retain their original numerical behavior.
    comparison = _ComparisonRecorder(
        output_dir, firstIteration, totalIterations, timeIteration,
        field_coordinates, node_control_volumes_m3, hepatocyte_weights,
        hepatocytes_array, hepatocytesZonation_array, hepatocytesRandom_array,
        output_times_s,
        COMPARISON_PROBE_IDS if comparison_probe_ids is None else comparison_probe_ids,
        {
            "mald_rtol": float(effective_rtol), "mald_atol": effective_atol.tolist(),
            "mald_method": "Radau", "mald_max_step": "infinity (RunMALD default)",
            "apap_update": "A += uptake_rate_s * C * reference_cell_volume_m3 * interval_s",
            "apap_uptake_rate_s_inv": APAP_UPTAKE_RATE_S,
            "reference_hepatocyte_volume_m3": REFERENCE_HEPATOCYTE_VOLUME_M3,
            "normal_porosity": porosity_normal,
            "hepatocyte_fraction_of_solid": HEPATOCYTE_FRACTION_OF_SOLID,
            "comsol_first_output_time_s": COMSOL_FIRST_OUTPUT_TIME_S,
            "comsol_export": str(apap_export.resolve()),
            "comsol_export_size_bytes": apap_export.stat().st_size,
            "comsol_export_mtime_ns": apap_export.stat().st_mtime_ns,
            "comsol_export_field_count": exported_field_count,
            "field_nodes_path": str(FIELD_NODES_PATH.resolve()),
            "continue_with_zero_after_export": CONTINUE_WITH_ZERO_AFTER_EXPORT,
            "final_field_relative_tolerance": FINAL_FIELD_RELATIVE_TOLERANCE,
            "spatial_output_stride": OUTPUT_EVERY_N_INTERVALS,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "mald_source_sha256": hashlib.sha256(Path(MALD.__file__).read_bytes()).hexdigest(),
            "numpy_version": np.__version__,
            "scipy_version": SCIPY_VERSION,
        },
    )

    for k in range(expected_rows):
        state = hepatocytes_array[k].copy()

        node_history = np.empty(
            (remaining_intervals, 10),
            dtype=float,
        )

        for local_index, n in enumerate(
            range(firstIteration, totalIterations)
        ):
            # Use this interval's exported COMSOL field.
            # The existing continuation check permits zero only after
            # the exported APAP field has become negligible.
            if n < exported_field_count:
                extracellular_apap = float(
                    exported_concentrations[k, n]
                )
            else:
                extracellular_apap = 0.0

            # Add APAP ONCE before solving this interval.
            added_apap_mol = (
                APAP_UPTAKE_RATE_S
                * extracellular_apap
                * REFERENCE_HEPATOCYTE_VOLUME_M3
                * float(timeIteration)
            )
            state[0] += added_apap_mol
            comparison.addition(
                k, local_index, extracellular_apap, added_apap_mol, state
            )

            # Unforced MALD: metabolism proceeds from the updated state.
            solution = MALD.RunMALD(
                state,
                float(timeIteration),
                hepatocytesZonation_array[k, 0],
                hepatocytesZonation_array[k, 1],
                hepatocytesZonation_array[k, 2],
                rtol=effective_rtol,
                atol=effective_atol,
            )

            solver_nfev += int(solution.nfev)
            solver_njev += int(solution.njev)
            solver_nlu += int(solution.nlu)

            # Carry all ten states into the next interval.
            state = solution.y[:, -1].copy()
            node_history[local_index] = state
            comparison.end(k, local_index, state)

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

    comparison.finish(solver_nfev, solver_njev, solver_nlu, time() - time0)
    time1 = time()
    print(f"Total simulation time (s) = {time1 - time0} for simulation: {fileNamePrefixe}")
    return
