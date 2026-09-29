"""Compare patient MALD histories under progressively tighter solver tolerances.

Example:
    python tests/run_patient_convergence.py \
        --results-dir "C:/path/to/Results-6" "C:/path/to/Results-60"

The analysis restarts from each run's saved iteration-0 MALD state and reuses
its COMSOL APAP history. It writes new files under each results directory's
Convergence_Study folder; existing simulation outputs are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import MALD_model_Scaled_V5 as MALD
import Virtual_Lobule_Script_COMSOL_V2 as simulation


STATE_NAMES = (
    "APAP",
    "NAPQI",
    "GSH",
    "Healthy",
    "Damaged",
    "Necrosed",
    "Regeneration",
    "AST",
    "ALT",
    "Clotting",
)
DEFAULT_RTOLS = (1.0e-6, 1.0e-7, 1.0e-8)


def _parse_rtol_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rtol values must be comma-separated numbers") from exc
    if not values or any(not np.isfinite(item) or item <= 0.0 for item in values):
        raise argparse.ArgumentTypeError("rtol values must be finite and positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("rtol values must not contain duplicates")
    return tuple(sorted(values, reverse=True))


def _load_state(path: Path, expected_rows: int | None = None) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] != 15:
        raise ValueError(f"{path} has {data.shape[1]} columns; expected 15.")
    if expected_rows is not None and data.shape[0] != expected_rows:
        raise ValueError(
            f"{path} has {data.shape[0]} nodes; expected {expected_rows}."
        )
    if np.any(~np.isfinite(data)):
        raise ValueError(f"{path} contains non-finite state or parameter values.")
    expected_ids = np.arange(1, data.shape[0] + 1)
    if not np.array_equal(data[:, 0], expected_ids):
        raise ValueError(f"{path} node IDs must be sequential and ordered.")
    return data


def _load_saved_history(
    results_dir: Path,
) -> tuple[list[int], np.ndarray, dict[int, np.ndarray]]:
    iterations = sorted(
        int(path.stem.rsplit("-", 1)[1])
        for path in results_dir.glob("_Used-Field_Hepatocytes-*.txt")
    )
    if not iterations:
        raise FileNotFoundError(
            f"No _Used-Field_Hepatocytes-*.txt snapshots in {results_dir}"
        )
    if iterations[0] != 0:
        raise ValueError(
            "Patient convergence requires the saved iteration-0 state as its "
            f"restart point; first available snapshot is {iterations[0]}."
        )

    states = {}
    node_count = None
    initial_data = None
    for iteration in iterations:
        path = results_dir / f"_Used-Field_Hepatocytes-{iteration}.txt"
        data = _load_state(path, node_count)
        node_count = data.shape[0]
        if iteration == 0:
            initial_data = data
        states[iteration] = data[:, 1:11].copy()
    if initial_data is None:
        raise RuntimeError("Iteration-0 MALD snapshot was not loaded.")
    return iterations, initial_data, states


def _make_forcing(
    result_dir: Path,
    node_coordinates: np.ndarray,
    last_iteration: int,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    export_path = result_dir / simulation.RESULT_BASENAME
    _export_times_s, exported = simulation._read_comsol_apap_time_series(
        export_path,
        node_coordinates,
        dt_s,
    )
    field_count = exported.shape[1]
    required_field_count = last_iteration + 1

    if field_count < required_field_count:
        final_relative = float(np.max(exported[:, -1])) / max(
            float(np.max(exported)),
            1.0e-30,
        )
        if (
            not simulation.CONTINUE_WITH_ZERO_AFTER_EXPORT
            or final_relative > simulation.FINAL_FIELD_RELATIVE_TOLERANCE
        ):
            raise ValueError(
                f"{export_path} contains {field_count} APAP fields, but "
                f"iteration {last_iteration} needs {required_field_count}. "
                f"The final field is {final_relative:.6e} of the peak; zero "
                "continuation is not permitted for this export."
            )
        print(
            f"  COMSOL fields end at column {field_count - 1}; verified "
            "zero-APAP continuation will cover the remaining intervals."
        )

    forcing_values = np.zeros(
        (exported.shape[0], required_field_count),
        dtype=float,
    )
    copied_fields = min(field_count, required_field_count)
    forcing_values[:, :copied_fields] = exported[:, :copied_fields]
    forcing_times_s = np.arange(required_field_count, dtype=float) * dt_s
    return forcing_times_s, forcing_values


def _normalized_node_errors(
    candidate: np.ndarray,
    reference: np.ndarray,
    state_scales: np.ndarray,
) -> np.ndarray:
    return np.max(
        np.abs(candidate - reference) / state_scales[None, :],
        axis=1,
    )


def _summarize_by_iteration(
    iterations: list[int],
    dt_s: float,
    rtol_values: tuple[float, ...],
    node_ids: np.ndarray,
    errors_by_rtol: np.ndarray,
    saved_errors: np.ndarray,
    output_dir: Path,
) -> Path:
    summary_path = output_dir / "patient_convergence_errors.csv"
    columns = (
        "rtol",
        "iteration",
        "time_hours",
        "nodes_analyzed",
        "median_normalized_error_vs_reference",
        "p95_normalized_error_vs_reference",
        "p99_normalized_error_vs_reference",
        "max_normalized_error_vs_reference",
        "worst_node_vs_reference",
        "median_saved_run_error_vs_reference",
        "p95_saved_run_error_vs_reference",
        "p99_saved_run_error_vs_reference",
        "max_saved_run_error_vs_reference",
        "worst_saved_run_node",
    )
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for rtol_index, rtol in enumerate(rtol_values):
            for time_index, iteration in enumerate(iterations):
                errors = errors_by_rtol[rtol_index, time_index]
                saved = saved_errors[time_index]
                writer.writerow(
                    {
                        "rtol": f"{rtol:.12g}",
                        "iteration": iteration,
                        "time_hours": f"{(iteration + 1) * dt_s / 3600.0:.12g}",
                        "nodes_analyzed": len(node_ids),
                        "median_normalized_error_vs_reference": float(np.median(errors)),
                        "p95_normalized_error_vs_reference": float(np.percentile(errors, 95)),
                        "p99_normalized_error_vs_reference": float(np.percentile(errors, 99)),
                        "max_normalized_error_vs_reference": float(np.max(errors)),
                        "worst_node_vs_reference": int(node_ids[np.argmax(errors)]),
                        "median_saved_run_error_vs_reference": float(np.median(saved)),
                        "p95_saved_run_error_vs_reference": float(np.percentile(saved, 95)),
                        "p99_saved_run_error_vs_reference": float(np.percentile(saved, 99)),
                        "max_saved_run_error_vs_reference": float(np.max(saved)),
                        "worst_saved_run_node": int(node_ids[np.argmax(saved)]),
                    }
                )
    return summary_path


def _plot_summary(summary_path: Path, output_dir: Path) -> Path:
    with summary_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rtol_values = sorted({float(row["rtol"]) for row in rows}, reverse=True)
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for index, rtol in enumerate(rtol_values):
        selected = [row for row in rows if float(row["rtol"]) == rtol]
        hours = np.array([float(row["time_hours"]) for row in selected])
        p95 = np.array(
            [float(row["p95_normalized_error_vs_reference"]) for row in selected]
        )
        maximum = np.array(
            [float(row["max_normalized_error_vs_reference"]) for row in selected]
        )
        p95[p95 <= 0.0] = np.nan
        maximum[maximum <= 0.0] = np.nan
        color = cmap(index / max(len(rtol_values) - 1, 1))
        ax.plot(
            hours,
            p95,
            marker="o",
            markersize=3,
            linewidth=1.8,
            color=color,
            label=f"rtol={rtol:.0e}, node p95",
        )
        ax.plot(
            hours,
            maximum,
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            color=color,
            label=f"rtol={rtol:.0e}, node max",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Maximum state-wise normalized error vs tight reference")
    ax.set_title(f"{summary_path.parent.parent.name}: patient MALD convergence")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.17)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.text(
        0.01,
        0.015,
        "Spatial p95 and maximum at each saved snapshot; dashed lines are maxima.\n"
        "This continuation study starts from the saved iteration-0 state.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
    )
    output = output_dir / "patient_convergence_errors.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return output


def run_patient(
    result_dir: Path,
    dt_s: float,
    rtol_values: tuple[float, ...],
    reference_rtol: float,
    max_nodes: int | None,
) -> None:
    result_dir = result_dir.resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Patient results directory does not exist: {result_dir}")
    iterations, initial_data, saved_states = _load_saved_history(result_dir)
    first_iteration = iterations[0]
    last_iteration = iterations[-1]
    node_count = initial_data.shape[0]

    if max_nodes is not None:
        if max_nodes < 1:
            raise ValueError("--max-nodes must be at least 1.")
        selected_rows = np.unique(
            np.linspace(0, node_count - 1, min(max_nodes, node_count), dtype=int)
        )
    else:
        selected_rows = np.arange(node_count, dtype=int)
    node_ids = initial_data[selected_rows, 0].astype(int)

    node_coordinates, _ = simulation._read_field_nodes_and_volumes(
        simulation.FIELD_NODES_PATH
    )
    if len(node_coordinates) != node_count:
        raise ValueError(
            f"{node_count} saved MALD nodes do not match "
            f"{len(node_coordinates)} Field_Nodes rows."
        )
    forcing_times_s, forcing_by_node = _make_forcing(
        result_dir,
        node_coordinates,
        last_iteration,
        dt_s,
    )
    output_times_s = (
        np.asarray(iterations, dtype=float) - first_iteration
    ) * dt_s
    atol_by_rtol = {
        rtol: MALD.MALD_ATOL * (rtol / MALD.MALD_RTOL)
        for rtol in rtol_values
    }
    reference_atol = MALD.MALD_ATOL * (
        reference_rtol / MALD.MALD_RTOL
    )

    study_name = (
        "Convergence_Study"
        if len(selected_rows) == node_count
        else f"Convergence_Study_pilot_{len(selected_rows)}_nodes"
    )
    output_dir = result_dir / study_name
    output_dir.mkdir(parents=True, exist_ok=True)
    errors_by_rtol = np.empty(
        (len(rtol_values), len(iterations), len(selected_rows)),
        dtype=np.float32,
    )
    saved_errors = np.empty((len(iterations), len(selected_rows)), dtype=np.float32)

    print(
        f"\n{result_dir.name}: {len(iterations)} saved times "
        f"({first_iteration}..{last_iteration}), "
        f"{len(selected_rows)}/{node_count} nodes."
    )
    print(
        f"  Reintegrating from saved iteration {first_iteration}; "
        f"saved-run time origin is {(first_iteration + 1) * dt_s / 3600.0:.3f} h."
    )
    print(
        f"  rtol sweep={','.join(f'{value:.0e}' for value in rtol_values)}; "
        f"tight reference rtol={reference_rtol:.0e}; max_step={dt_s:g} s."
    )

    started = perf_counter()
    progress_every = max(1, len(selected_rows) // 20)
    for selected_index, row_index in enumerate(selected_rows):
        row = initial_data[row_index]
        initial_state = row[1:11]
        v1, v2, v3 = row[11:14]
        concentration = forcing_by_node[row_index]

        reference = MALD.RunMALDHistory(
            initial_state,
            output_times_s,
            forcing_times_s,
            concentration,
            v1,
            v2,
            v3,
            simulation.APAP_UPTAKE_RATE_S,
            simulation.REFERENCE_HEPATOCYTE_VOLUME_M3,
            dt_s,
            rtol=reference_rtol,
            atol=reference_atol,
        )
        reference_states = reference.y.T
        state_scales = np.maximum(
            np.max(np.abs(reference_states), axis=0),
            MALD.MALD_ATOL,
        )
        saved_states_for_node = np.stack(
            [saved_states[iteration][row_index] for iteration in iterations]
        )
        saved_errors[:, selected_index] = _normalized_node_errors(
            saved_states_for_node,
            reference_states,
            state_scales,
        )

        for rtol_index, rtol in enumerate(rtol_values):
            candidate = MALD.RunMALDHistory(
                initial_state,
                output_times_s,
                forcing_times_s,
                concentration,
                v1,
                v2,
                v3,
                simulation.APAP_UPTAKE_RATE_S,
                simulation.REFERENCE_HEPATOCYTE_VOLUME_M3,
                dt_s,
                rtol=rtol,
                atol=atol_by_rtol[rtol],
            )
            errors_by_rtol[rtol_index, :, selected_index] = (
                _normalized_node_errors(
                    candidate.y.T,
                    reference_states,
                    state_scales,
                )
            )

        if (
            (selected_index + 1) % progress_every == 0
            or selected_index + 1 == len(selected_rows)
        ):
            elapsed = perf_counter() - started
            print(
                f"  Completed {selected_index + 1}/{len(selected_rows)} nodes "
                f"({elapsed / 60.0:.1f} min elapsed)."
            )

    summary_path = _summarize_by_iteration(
        iterations,
        dt_s,
        rtol_values,
        node_ids,
        errors_by_rtol,
        saved_errors,
        output_dir,
    )
    plot_path = _plot_summary(summary_path, output_dir)
    metadata = {
        "results_directory": str(result_dir),
        "start_snapshot_iteration": first_iteration,
        "end_snapshot_iteration": last_iteration,
        "saved_iterations": iterations,
        "time_step_s": dt_s,
        "time_origin_hours": (first_iteration + 1) * dt_s / 3600.0,
        "reference_rtol": reference_rtol,
        "reference_atol": reference_atol.tolist(),
        "candidate_rtols": list(rtol_values),
        "candidate_atols": {
            f"{rtol:.12g}": atol_by_rtol[rtol].tolist()
            for rtol in rtol_values
        },
        "max_step_s": dt_s,
        "nodes_in_source_run": node_count,
        "nodes_analyzed": len(selected_rows),
        "node_selection": (
            "all nodes"
            if len(selected_rows) == node_count
            else "evenly spaced row sample; exploratory, not full spatial coverage"
        ),
        "reference_method": "Radau",
        "compared_states": list(STATE_NAMES),
        "error_metric": (
            "At each node and saved time, maximum over states of absolute "
            "candidate-reference difference divided by the maximum absolute "
            "reference value for that node/state over the analyzed trajectory, "
            "floored by MALD_ATOL."
        ),
        "apap_forcing_file": str(result_dir / simulation.RESULT_BASENAME),
        "start_state_file": str(
            result_dir / f"_Used-Field_Hepatocytes-{first_iteration}.txt"
        ),
        "limitation": (
            "The original pre-iteration-0 initial condition is not present in "
            "the saved patient results. This study therefore restarts from "
            "the saved post-iteration-0 state and assesses the remaining run."
        ),
        "outputs": [str(summary_path), str(plot_path)],
    }
    metadata_path = output_dir / "patient_convergence_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote {summary_path}")
    print(f"  Wrote {plot_path}")
    print(f"  Wrote {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run MALD tolerance-refinement studies against saved patient "
            "COMSOL forcing and iteration-0 MALD states."
        )
    )
    parser.add_argument(
        "--results-dir",
        nargs="+",
        type=Path,
        required=True,
        help="One or more patient result directories containing saved MALD snapshots.",
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=1200.0,
        help="Saved MALD/COMSOL interval in seconds (default: 1200).",
    )
    parser.add_argument(
        "--rtols",
        type=_parse_rtol_list,
        default=DEFAULT_RTOLS,
        help="Comma-separated candidate rtol values (default: 1e-6,1e-7,1e-8).",
    )
    parser.add_argument(
        "--reference-rtol",
        type=float,
        default=1.0e-10,
        help="Tight-reference rtol; must be lower than every candidate rtol.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help=(
            "Analyze this many evenly spaced nodes per patient as an exploratory "
            "pilot. Omit to analyze every node."
        ),
    )
    args = parser.parse_args()

    if not np.isfinite(args.dt_s) or args.dt_s <= 0.0:
        parser.error("--dt-s must be finite and positive.")
    if (
        not np.isfinite(args.reference_rtol)
        or args.reference_rtol <= 0.0
        or args.reference_rtol >= min(args.rtols)
    ):
        parser.error("--reference-rtol must be positive and tighter than every candidate rtol.")
    if args.max_nodes is not None and args.max_nodes < 1:
        parser.error("--max-nodes must be at least 1.")

    for result_dir in args.results_dir:
        run_patient(
            result_dir,
            args.dt_s,
            args.rtols,
            args.reference_rtol,
            args.max_nodes,
        )


if __name__ == "__main__":
    main()
