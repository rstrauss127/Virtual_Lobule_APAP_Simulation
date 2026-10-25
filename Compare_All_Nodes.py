"""Compare two COMPLETE runs on the SAME mesh and coupling time grid.

Without --limits, reports discrepancies without declaring convergence.
With --limits, requires every node to satisfy each variable's absolute +
relative allowance. The finer run is a numerical reference, not ground truth.
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
from All_Node_Comparison import STATE_NAMES, CHEMICAL_NAMES, atomic_json


def compare(candidate_dir, reference_dir, output_dir, limits_path=None):
    candidate_dir, reference_dir, output_dir = map(Path, (candidate_dir, reference_dir, output_dir))
    if output_dir.resolve() in (candidate_dir.resolve(), reference_dir.resolve()):
        raise ValueError("Choose a separate directory for the comparison report")
    manifests = [json.loads((p / "checkpoint.json").read_text()) for p in (candidate_dir, reference_dir)]
    for m in manifests:
        if m["status"] != "completed" or m["completed_nodes"] != m["identity"]["node_count"]:
            raise ValueError("Both runs must be complete; resume unfinished runs first")
    ci, ri = [m["identity"] for m in manifests]
    invariants = ["node_count", "first_iteration", "total_iterations", "interval_s",
                  "coordinates_sha256", "volumes_sha256", "initial_states_sha256",
                  "zonation_sha256", "random_values_sha256", "initial_status_sha256",
                  "aligned_comsol_fields_sha256", "apap_uptake_rate_s_inv",
                  "reference_hepatocyte_volume_m3", "normal_porosity", "hepatocyte_fraction_of_solid",
                  "comsol_first_output_time_s", "mald_source_sha256", "driver_sha256",
                  "recorder_sha256", "numpy_version", "scipy_version",
                  "continue_with_zero_after_export", "final_field_relative_tolerance"]
    for key in invariants:
        if ci[key] != ri[key]:
            raise ValueError(f"Not a controlled same-mesh tolerance comparison: {key} differs")
    n = ci["node_count"]
    t = ci["total_iterations"] - ci["first_iteration"]
    shapes = {"states": (n, t + 1, 10), "added_apap": (n, t),
              "status": (n, t), "diagnostics": (n, 15)}
    shapes.update({name: (n, 3) for name in ["auc", "minima", "maxima", "min_times", "max_times"]})
    arrays = []
    for folder in (candidate_dir, reference_dir):
        item = {name: np.load(folder / (name + ".npy"), mmap_mode="r", allow_pickle=False) for name in shapes}
        if any(item[name].shape != shape for name, shape in shapes.items()):
            raise ValueError(f"Invalid saved array dimensions in {folder}")
        arrays.append(item)
    with np.load(reference_dir / "initial_snapshot.npz", allow_pickle=False) as snap:
        coordinates = snap["coordinates"]
        weights = snap["weights"]
    weights = weights / weights.sum()
    labels = (["state_" + s for s in STATE_NAMES] + ["post_addition_APAP"]
              + ["auc_" + s for s in CHEMICAL_NAMES]
              + ["minimum_" + s for s in CHEMICAL_NAMES]
              + ["maximum_" + s for s in CHEMICAL_NAMES])
    limits = None
    if limits_path is not None:
        limits = json.loads(Path(limits_path).read_text())
        for key, length in [("state_atol", 10), ("state_rtol", 10), ("auc_atol", 3), ("auc_rtol", 3)]:
            a = np.asarray(limits[key], dtype=float)
            if a.shape != (length,) or not np.isfinite(a).all() or np.any(a < 0):
                raise ValueError(f"{key} must contain {length} finite, nonnegative numbers")
            limits[key] = a.tolist()
        aa = np.array(limits["state_atol"] + limits["state_atol"][:1]
                      + limits["auc_atol"] + limits["state_atol"][:3]*2)
        rr = np.array(limits["state_rtol"] + limits["state_rtol"][:1]
                      + limits["auc_rtol"] + limits["state_rtol"][:3]*2)
        if np.any((aa == 0) & (rr == 0)):
            raise ValueError("Every comparison variable needs a positive absolute or relative limit")
    output_dir.mkdir(parents=True, exist_ok=True)
    error_map = np.lib.format.open_memmap(output_dir / "node_max_abs_errors.npy", mode="w+",
                                          dtype="float64", shape=(n, len(labels)))
    fail_map = None if limits is None else np.lib.format.open_memmap(
        output_dir / "node_failed.npy", mode="w+", dtype="bool", shape=(n, len(labels)))
    maxima = np.full(len(labels), -1.)
    max_ref = np.zeros(len(labels))
    worst_nodes = np.zeros(len(labels), dtype=int)
    worst_indices = np.zeros(len(labels), dtype=int)
    failed_counts = np.zeros(len(labels), dtype=int)
    mean_differences = [np.zeros(t + 1) for _ in range(10)] + [np.zeros(t)] + [np.zeros(1) for _ in range(9)]
    status_disagreement_nodes = 0
    for first in range(0, n, 256):
        last = min(first + 256, n)
        c, r = [{key: value[first:last] for key, value in a.items()} for a in arrays]
        if any(not np.isfinite(v).all() for a in (c, r) for v in a.values()):
            raise ValueError(f"Nonfinite records in nodes {first + 1}..{last}")
        if any(np.any(a["diagnostics"][:, 4] != t) for a in (c, r)):
            raise ValueError("Incomplete interval coverage")
        if not np.array_equal(c["added_apap"], r["added_apap"]):
            raise ValueError("APAP additions differ between runs")
        status_disagreement_nodes += int(np.any(c["status"] != r["status"], axis=1).sum())
        groups = []
        for a in (c, r):
            g = [a["states"][:, :, v] for v in range(10)]
            g.append(a["states"][:, :-1, 0] + a["added_apap"])
            g += [a[key][:, v:v + 1] for key in ("auc", "minima", "maxima") for v in range(3)]
            groups.append(g)
        for v, (cv, rv) in enumerate(zip(*groups)):
            error = np.abs(cv - rv)
            per_node = error.max(axis=1)
            error_map[first:last, v] = per_node
            flat_index = int(error.argmax())
            node, ti = np.unravel_index(flat_index, error.shape)
            value = float(error[node, ti])
            if value > maxima[v]:
                maxima[v], worst_nodes[v], worst_indices[v] = value, first + node, ti
            max_ref[v] = max(max_ref[v], float(np.abs(rv).max()))
            mean_differences[v] += weights[first:last] @ (cv - rv)
            if limits is not None:
                failed = np.any(error > aa[v] + rr[v]*np.abs(rv), axis=1)
                fail_map[first:last, v] = failed
                failed_counts[v] += int(failed.sum())
    error_map.flush()
    if fail_map is not None:
        fail_map.flush()
    rows = []
    for v, label in enumerate(labels):
        k = int(worst_nodes[v])
        time_s = ((ci["first_iteration"] + int(worst_indices[v]))*ci["interval_s"] if v <= 10 else "")
        rows.append({"metric": label, "max_absolute_difference": float(maxima[v]),
                     "reference_max_abs_scale": float(max_ref[v]),
                     "max_difference_divided_by_reference_scale": float(maxima[v]/max_ref[v]) if max_ref[v] else None,
                     "worst_node_id": k + 1, "x_mm": float(coordinates[k, 0]),
                     "y_mm": float(coordinates[k, 1]), "z_mm": float(coordinates[k, 2]),
                     "worst_time_s": time_s,
                     "max_abs_weighted_mean_difference": float(np.max(np.abs(mean_differences[v]))),
                     "nodes_exceeding_limits": int(failed_counts[v]) if limits is not None else None,
                     "fraction_exceeding_limits": float(failed_counts[v]/n) if limits is not None else None})
    with (output_dir / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    report = {"candidate": str(candidate_dir.resolve()), "reference": str(reference_dir.resolve()),
              "nodes_compared": n, "intervals_compared_per_node": t,
              "all_nodes_within_limits": bool(not failed_counts.any()) if limits is not None else None,
              "limits": limits, "node_error_columns": labels,
              "nodes_with_any_categorical_status_difference": status_disagreement_nodes,
              "candidate_rtol": ci["mald_rtol"], "reference_rtol": ri["mald_rtol"],
              "note": "Same-mesh MALD tolerance comparison only. No limits means no pass/fail. Saved-time trajectories and chemical AUC/extrema are checked; this is not a uniform-in-time error bound or a mesh/coupling-interval test.",
              "metrics": rows}
    atomic_json(output_dir / "comparison_report.json", report)
    np.save(output_dir / "node_coordinates_mm.npy", coordinates)
    print(f"Compared ALL {n} nodes over {t} intervals. Report: {output_dir}")
    print("Pass/fail not assessed: no acceptance limits supplied." if limits is None else
          f"Every node within supplied limits: {report['all_nodes_within_limits']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limits", help="JSON with ten state_atol/state_rtol and three auc_atol/auc_rtol values")
    a = parser.parse_args()
    compare(a.candidate, a.reference, a.output, a.limits)
