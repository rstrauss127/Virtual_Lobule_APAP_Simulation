# -*- coding: utf-8 -*-
"""Graph COMSOL -> MALD acetaminophen simulation results.

Matches the 15-column files written by Virtual_Lobule_Script_COMSOL_V2.py:
id, APAP, NAPQI, GSH, Healthy, Damaged, Necrosed, Regeneration, AST,
ALT, Clotting, GSH_prod_rate, GSH_bind_rate, Regeneration_rate,
Random_number.

Field_Nodes.csv must be headerless:
id, x_mm, y_mm, z_mm, node_control_volume_m3.

Create horizontal img panel:
python "Read_Results_Pinto_Style.py" --results-dir "Results_v0/Results-61" --field-nodes "Mesh_Volume_Calculations/Field_Nodes.csv" --horizontal-only 
"""

from __future__ import annotations
import MALD_model_Scaled_V5 as MALD

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parent
MALD_COLUMNS = [
    "id", "APAP", "NAPQI", "GSH", "Healthy", "Damaged", "Necrosed",
    "Regeneration", "AST", "ALT", "Clotting", "GSH_prod_rate",
    "GSH_bind_rate", "Regeneration_rate", "Random_number",
]
STATE_AMOUNT_COLUMNS = ("APAP", "NAPQI", "GSH")
CELL_STATE_COLUMNS = ("Healthy", "Damaged", "Necrosed", "Regeneration")
# Pinto's visualization suite includes spatial APAP, NAPQI, GSH, AST, and
# ALT. COMSOL extracellular APAP is handled separately below.
HEATMAP_COLUMNS = (*STATE_AMOUNT_COLUMNS, "AST", "ALT")
STATUS_LABELS = {0.0: "Healthy", 1.0: "Damaged", 2.0: "Necrosed", 3.0: "Regenerated"}
STATUS_COLORS = {0.0: "#2ca02c", 1.0: "#ff7f0e", 2.0: "#202020", 3.0: "#7f7f7f"}

# Must match RESULT_BASENAME and COMSOL_FIRST_OUTPUT_TIME_S in
# Virtual_Lobule_Script_COMSOL_V2.py.
COMSOL_MULTITIME_BASENAME = "Lobule-Flow-Drug_Simulation_3D_V3-1200.txt"
COMSOL_FIRST_OUTPUT_TIME_S = 1200.0
COORDINATE_MATCH_RELATIVE_TOLERANCE = 1.0e-8

# Must match Virtual_Lobule_Script_COMSOL_V2.py.
REFERENCE_HEPATOCYTE_VOLUME_M3 = 3.4e-15
NORMAL_POROSITY = 0.143
HEPATOCYTE_FRACTION_OF_SOLID = 1.0
DEFAULT_HORIZONTAL_ITERATIONS = (0, 18, 36, 84, 138, 198)


def _resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_field_nodes(path: str | Path) -> pd.DataFrame:
    """Read and validate headerless Field_Nodes.csv."""
    path = Path(path)
    raw = pd.read_csv(path, header=None, comment="#", skip_blank_lines=True)
    expected = ["id", "x", "y", "z", "volume_m3"]
    if raw.shape[1] != len(expected):
        first = raw.iloc[0].tolist() if len(raw) else "EMPTY FILE"
        raise ValueError(
            f"{path} has {raw.shape[1]} columns; expected exactly 5: "
            f"id,x_mm,y_mm,z_mm,node_control_volume_m3. First row: {first}"
        )
    raw.columns = expected
    raw["id"] = pd.to_numeric(raw["id"], errors="raise").astype(int)
    for col in expected[1:]:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
    if raw.empty:
        raise ValueError(f"{path} is empty")
    if raw["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate node IDs")
    if not np.array_equal(raw["id"].to_numpy(), np.arange(1, len(raw) + 1)):
        raise ValueError(f"{path} IDs must be sequential and ordered from 1 to {len(raw)}")
    numeric = raw[["x", "y", "z", "volume_m3"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    if (raw["volume_m3"] <= 0.0).any():
        raise ValueError(f"{path} contains non-positive node control volumes")
    print(
        f"Read {path.name}: {len(raw):,} nodes; total control volume "
        f"{raw['volume_m3'].sum():.12e} m^3"
    )
    return raw


def read_mald_file(path: str | Path, *, verbose: bool = True) -> pd.DataFrame:
    """Read the exact 15-column S-free MALD state schema."""
    path = Path(path)
    raw = pd.read_csv(path, header=None, comment="#", skip_blank_lines=True)
    if raw.shape[1] != len(MALD_COLUMNS):
        first = raw.iloc[0].tolist() if len(raw) else "EMPTY FILE"
        raise ValueError(
            f"{path} has {raw.shape[1]} columns; expected {len(MALD_COLUMNS)}.\n"
            f"Expected: {', '.join(MALD_COLUMNS)}\nFirst row: {first}"
        )
    raw.columns = MALD_COLUMNS
    raw["id"] = pd.to_numeric(raw["id"], errors="raise").astype(int)
    for col in MALD_COLUMNS[1:]:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
    if raw.empty:
        raise ValueError(f"{path} is empty")
    if raw["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate node IDs")
    if not np.isfinite(raw[MALD_COLUMNS[1:]].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains NaN or infinite MALD values")
    if verbose:
        print(f"Read {path.name}: {len(raw):,} rows x {raw.shape[1]} columns")
    return raw


def read_status_file(path: str | Path, expected_rows: int) -> pd.DataFrame:
    path = Path(path)
    raw = pd.read_csv(path, header=None, comment="#", skip_blank_lines=True)
    if raw.shape[1] != 1 or len(raw) != expected_rows:
        raise ValueError(
            f"{path} must have one column and {expected_rows} rows; "
            f"found {len(raw)} rows x {raw.shape[1]} columns"
        )
    status = pd.to_numeric(raw.iloc[:, 0], errors="raise").astype(float)
    invalid = sorted(set(status) - set(STATUS_LABELS))
    if invalid:
        raise ValueError(f"{path} contains invalid status values: {invalid}")
    return pd.DataFrame({"id": np.arange(1, expected_rows + 1), "Status": status})


def _numeric_rows(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("%", "#", "@")):
                continue
            tokens = [token for token in re.split(r"[,\s]+", stripped) if token]
            try:
                rows.append([float(token) for token in tokens])
            except ValueError:
                continue
    return rows


def read_comsol_apap_export(path: str | Path, nodes: pd.DataFrame) -> pd.DataFrame:
    """Read x,y,z,c or id,x,y,z,c COMSOL concentration output [mol/m^3]."""
    path = Path(path)
    rows = _numeric_rows(path)
    if not rows:
        raise ValueError(f"No numeric COMSOL data rows found in {path}")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"{path} has inconsistent numeric row widths: {sorted(widths)}")
    exported = pd.DataFrame(rows)
    if len(exported) != len(nodes):
        raise ValueError(f"{path} has {len(exported)} rows; Field_Nodes has {len(nodes)}")
    if exported.shape[1] == 4:
        exported.columns = ["x", "y", "z", "COMSOL_APAP"]
        exported.insert(0, "id", nodes["id"].to_numpy())
    elif exported.shape[1] == 5:
        exported.columns = ["id", "x", "y", "z", "COMSOL_APAP"]
        exported["id"] = exported["id"].astype(int)
        if not np.array_equal(exported["id"].to_numpy(), nodes["id"].to_numpy()):
            raise ValueError(f"{path} IDs do not match Field_Nodes in the same order")
    else:
        raise ValueError(f"{path} has {exported.shape[1]} columns; expected x,y,z,c or id,x,y,z,c")
    if not np.isfinite(exported["COMSOL_APAP"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains non-finite APAP concentrations")
    return exported[["id", "COMSOL_APAP"]].copy()


def find_multitime_comsol_apap_file(results_dir: str | Path) -> Optional[Path]:
    """Find the single x,y,z,c(t1),c(t2),... COMSOL export."""
    results_dir = Path(results_dir)
    preferred = results_dir / COMSOL_MULTITIME_BASENAME
    if preferred.is_file():
        return preferred

    candidates = sorted(
        path
        for path in results_dir.glob("*Lobule-Flow-Drug*1200*.txt")
        if path.is_file() and not path.name.startswith("_Used-")
    )
    return candidates[0] if candidates else None


def read_multitime_comsol_apap_export(
    path: str | Path,
    nodes: pd.DataFrame,
) -> np.ndarray:
    """Load and align x,y,z,c(t1),c(t2),... to Field_Nodes order."""
    path = Path(path)
    exported = np.loadtxt(path, ndmin=2)
    expected_rows = len(nodes)

    if exported.shape[0] != expected_rows:
        raise ValueError(
            f"{path} has {exported.shape[0]} rows; "
            f"Field_Nodes has {expected_rows}"
        )
    if exported.shape[1] < 4:
        raise ValueError(
            f"{path} has {exported.shape[1]} columns; expected "
            "x,y,z and at least one concentration column"
        )
    if not np.isfinite(exported).all():
        raise ValueError(f"{path} contains NaN or infinite values")

    node_coordinates = nodes[["x", "y", "z"]].to_numpy(dtype=float)
    export_coordinates = exported[:, :3]
    concentrations = exported[:, 3:]

    coordinate_scale = max(
        float(np.ptp(node_coordinates, axis=0).max()),
        1.0,
    )
    coordinate_tolerance = (
        COORDINATE_MATCH_RELATIVE_TOLERANCE * coordinate_scale
    )

    if not np.allclose(
        export_coordinates,
        node_coordinates,
        rtol=COORDINATE_MATCH_RELATIVE_TOLERANCE,
        atol=coordinate_tolerance,
    ):
        distances, export_indices = cKDTree(export_coordinates).query(
            node_coordinates,
            k=1,
        )
        if float(distances.max()) > coordinate_tolerance:
            raise ValueError(
                "COMSOL coordinates do not match Field_Nodes. "
                f"Maximum nearest-point distance={distances.max():.6e}; "
                f"tolerance={coordinate_tolerance:.6e}"
            )
        if len(np.unique(export_indices)) != expected_rows:
            raise ValueError(
                "COMSOL-to-Field_Nodes coordinate matching was not one-to-one"
            )
        concentrations = concentrations[export_indices, :]

    negative_count = int(np.count_nonzero(concentrations < 0.0))
    if negative_count:
        minimum = float(concentrations.min())
        peak = float(np.max(np.abs(concentrations)))
        if minimum < -1.0e-2 * max(peak, 1.0e-30):
            raise ValueError(
                "COMSOL produced materially negative APAP concentrations: "
                f"minimum={minimum:.6e}, peak magnitude={peak:.6e}"
            )
        print(
            f"Warning: clipping {negative_count} small negative COMSOL "
            f"values to zero (minimum={minimum:.6e})."
        )
        concentrations = np.maximum(concentrations, 0.0)

    print(
        f"Read {path.name}: {concentrations.shape[0]:,} nodes x "
        f"{concentrations.shape[1]} concentration time columns"
    )
    return concentrations


def discover_iterations(results_dir: str | Path) -> List[int]:
    pattern = re.compile(r"_Used-Field_Hepatocytes-(\d+)\.txt$")
    found = []
    for path in Path(results_dir).glob("_Used-Field_Hepatocytes-*.txt"):
        match = pattern.fullmatch(path.name)
        if match:
            found.append(int(match.group(1)))
    return sorted(set(found))


def mald_file(results_dir: str | Path, iteration: int) -> Path:
    return Path(results_dir) / f"_Used-Field_Hepatocytes-{iteration}.txt"


def status_file(results_dir: str | Path, iteration: int) -> Path:
    return Path(results_dir) / f"_hepatocytes_status_Hepatocytes-{iteration}.txt"


def status_for_state_snapshot(
    results_dir: Path,
    iteration: int,
    expected_rows: int,
) -> Tuple[Optional[pd.DataFrame], str]:
    """Return status aligned with the post-MALD state at this iteration.

    The full-history simulator writes both _Used-Field_Hepatocytes-n and
    _hepatocytes_status_Hepatocytes-n from the same post-MALD snapshot.
    """
    current = status_file(results_dir, iteration)
    if current.is_file():
        return read_status_file(current, expected_rows), str(current)
    return None, "missing_same_iteration_status"


def find_comsol_apap_file(results_dir: str | Path, iteration: int) -> Optional[Path]:
    """Find only concentration exports; never return an arbitrary log/text file."""
    results_dir = Path(results_dir)
    iteration_dir = results_dir / f"comsol_iteration_{iteration:04d}"
    candidates: List[Path] = []
    if iteration_dir.is_dir():
        candidates.extend(sorted(iteration_dir.glob("*Lobule-Flow-Drug*")))
    candidates.extend(sorted(results_dir.glob(f"_Used-*Lobule-Flow-Drug*-{iteration}")))
    candidates.extend(sorted(results_dir.glob(f"_Used-*Lobule-Flow-Drug*{iteration}.txt")))
    candidates.extend(sorted(results_dir.glob(f"_Used-*Lobule-Flow-Drug*{iteration}.csv")))
    return next((path for path in candidates if path.is_file()), None)


def merge_nodes_and_mald(nodes: pd.DataFrame, mald: pd.DataFrame) -> pd.DataFrame:
    """Join by validated sequential ID; never silently rely on row order."""
    if len(nodes) != len(mald):
        raise ValueError(f"Field_Nodes has {len(nodes)} rows but MALD has {len(mald)} rows")
    if not np.array_equal(nodes["id"].to_numpy(), mald["id"].to_numpy()):
        raise ValueError("Field_Nodes and MALD IDs/order differ. Regenerate them before graphing.")
    return nodes.merge(mald, on="id", how="inner", validate="one_to_one")


def volume_weighted_mean(df: pd.DataFrame, column: str) -> float:
    return float(np.average(df[column].to_numpy(dtype=float), weights=df["volume_m3"]))


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hepatocyte_weight"] = (
        (1.0 - NORMAL_POROSITY) * HEPATOCYTE_FRACTION_OF_SOLID
        * out["volume_m3"] / REFERENCE_HEPATOCYTE_VOLUME_M3
    )
    # These are the exact node-wise probabilities used by _classify_status in
    # Virtual_Lobule_Script_COMSOL_V2.py. Keeping them in the visualization
    # output makes an all-zero categorical status directly auditable.
    healthy = out["Healthy"].to_numpy(dtype=float)
    damaged = out["Damaged"].to_numpy(dtype=float)
    necrosed = out["Necrosed"].to_numpy(dtype=float)
    random_number = out["Random_number"].to_numpy(dtype=float)
    out["damage_probability"] = np.divide(
        damaged,
        healthy + damaged,
        out=np.zeros(len(out), dtype=float),
        where=(healthy + damaged) > 0.0,
    )
    out["necrosis_probability"] = np.divide(
        necrosed,
        healthy + damaged + necrosed,
        out=np.zeros(len(out), dtype=float),
        where=(healthy + damaged + necrosed) > 0.0,
    )
    out["regeneration_probability"] = out["Regeneration"]
    out["damage_triggered"] = random_number < out["damage_probability"]
    out["necrosis_triggered"] = random_number < out["necrosis_probability"]
    out["regeneration_triggered"] = random_number < out["regeneration_probability"]

    # Pinto divided the idealized 2D lobule into three radial zones. For the
    # realistic geometry, recover the portal-to-central coordinate from the
    # stored, linearly zonated GSH production rate and use anatomical thirds.
    # This preserves the meaning of Pinto Zones 1-3 without assuming that the
    # geometric centroid is the central vein.
    production_rate = out["GSH_prod_rate"].to_numpy(dtype=float)
    rate_min = float(np.min(production_rate))
    rate_max = float(np.max(production_rate))
    if rate_max > rate_min:
        laue_xi = (rate_max - production_rate) / (rate_max - rate_min)
    else:
        laue_xi = np.full(len(out), 0.5)
    out["laue_xi"] = np.clip(laue_xi, 0.0, 1.0)
    out["Pinto_zone"] = pd.cut(
        out["laue_xi"],
        bins=[-np.inf, 1.0 / 3.0, 2.0 / 3.0, np.inf],
        labels=["Zone 1", "Zone 2", "Zone 3"],
    ).astype(str)
    return out


def summarize_iteration(df: pd.DataFrame, iteration: int, iteration_seconds: float) -> Dict[str, float]:
    result: Dict[str, float] = {
        "iteration": float(iteration),
        # Snapshot n is written after biological interval n, so iteration 0
        # is at one interval, not at t=0.
        "time_hours": float((iteration + 1) * iteration_seconds / 3600.0),
        "total_volume_m3": float(df["volume_m3"].sum()),
        "equivalent_hepatocytes": float(df["hepatocyte_weight"].sum()),
    }
    weights = df["hepatocyte_weight"].to_numpy(dtype=float)
    for col in STATE_AMOUNT_COLUMNS:
        result[f"mean_{col}_mol_per_cell"] = volume_weighted_mean(df, col)
        result[f"total_{col}_mol"] = float(np.sum(df[col].to_numpy(dtype=float) * weights))
    for col in (*CELL_STATE_COLUMNS, "AST", "ALT", "Clotting"):
        result[f"vw_mean_{col}"] = volume_weighted_mean(df, col)
    if "COMSOL_APAP" in df:
        result["vw_mean_COMSOL_APAP_mol_m3"] = volume_weighted_mean(df, "COMSOL_APAP")
    if "Status" in df:
        total_volume = float(df["volume_m3"].sum())
        for value, label in STATUS_LABELS.items():
            mask = df["Status"] == value
            result[f"status_{label}_volume_pct"] = float(100 * df.loc[mask, "volume_m3"].sum() / total_volume)
            result[f"status_{label}_count_pct"] = float(100 * mask.mean())
    total_volume = float(df["volume_m3"].sum())
    for transition in ("damage", "necrosis", "regeneration"):
        mask = df[f"{transition}_triggered"]
        result[f"{transition}_trigger_volume_pct"] = float(
            100.0 * df.loc[mask, "volume_m3"].sum() / total_volume
        )
        result[f"{transition}_trigger_count_pct"] = float(100.0 * mask.mean())
    return result


def _sample_df(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    return df if len(df) <= max_points else df.sample(max_points, random_state=0)


def _safe_limits(limits: Tuple[float, float]) -> Tuple[float, float]:
    low, high = limits
    if not np.isfinite(low) or not np.isfinite(high):
        raise ValueError(f"Non-finite color limits: {limits}")
    if high < low:
        low, high = high, low
    if high == low:
        padding = max(abs(high) * 1e-6, np.finfo(float).eps)
        return low - padding, high + padding
    return low, high


def _spatial_color_limits(
    values: pd.Series | np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Tuple[float, float]:
    """Robust within-iteration limits for revealing spatial structure.

    The percentile limits prevent a few extreme mesh nodes from compressing
    the rest of the lobule into one color. The absolute values remain visible
    on the colorbar; this is not a normalization of the state itself.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("Cannot calculate color limits from an empty field")
    low, high = np.percentile(array, [lower_percentile, upper_percentile])
    return _safe_limits((float(low), float(high)))


def _field_diagnostics(values: pd.Series | np.ndarray) -> Dict[str, float]:
    """Return range statistics that distinguish uniform biology from plotting."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "min": np.nan, "p01": np.nan, "mean": np.nan,
            "p99": np.nan, "max": np.nan, "std": np.nan,
            "cv": np.nan, "p01_p99_range_pct_of_mean": np.nan,
        }
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    mean = float(np.mean(array))
    std = float(np.std(array))
    p01, p99 = (float(value) for value in np.percentile(array, [1.0, 99.0]))
    denominator = max(abs(mean), np.finfo(float).tiny)
    return {
        "min": minimum,
        "p01": p01,
        "mean": mean,
        "p99": p99,
        "max": maximum,
        "std": std,
        "cv": std / denominator,
        "p01_p99_range_pct_of_mean": 100.0 * (p99 - p01) / denominator,
    }


def _plot_scale_specs(
    values: pd.Series | np.ndarray,
    global_limits: Tuple[float, float],
    color_scale: str,
) -> List[Tuple[str, Tuple[float, float], str]]:
    specs: List[Tuple[str, Tuple[float, float], str]] = []
    if color_scale in {"global", "both"}:
        specs.append(("global", _safe_limits(global_limits), "global absolute scale"))
    if color_scale in {"spatial", "both"}:
        specs.append((
            "spatial",
            _spatial_color_limits(values),
            "within-iteration spatial scale (1st-99th percentile)",
        ))
    return specs


def plot_3d_scatter(
    df: pd.DataFrame, value_col: str, output_path: str | Path, title: str,
    color_limits: Tuple[float, float], colorbar_label: str,
    max_points: int = 150000,
) -> None:
    plot_df = _sample_df(df.dropna(subset=[value_col]), max_points)
    if plot_df.empty:
        return
    vmin, vmax = _safe_limits(color_limits)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        plot_df["x"], plot_df["y"], plot_df["z"], c=plot_df[value_col],
        cmap="viridis", vmin=vmin, vmax=vmax, s=1.5, alpha=0.8, linewidths=0,
    )
    fig.colorbar(scatter, ax=ax, label=colorbar_label, shrink=0.7, pad=0.08)
    ax.set(title=title, xlabel="x [mm]", ylabel="y [mm]", zlabel="z [mm]")
    spans = np.ptp(plot_df[["x", "y", "z"]].to_numpy(dtype=float), axis=0)
    if np.all(spans > 0):
        ax.set_box_aspect(spans)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_slab(
    df: pd.DataFrame, quantile: float = 0.5,
    slab_fraction: float = 0.03, min_points: int = 800,
) -> Tuple[pd.DataFrame, float]:
    if not 0 <= quantile <= 1:
        raise ValueError("slice quantiles must be between 0 and 1")
    values = df["z"].to_numpy(dtype=float)
    target = float(np.quantile(values, quantile))
    half_width = max(float(np.ptp(values)) * slab_fraction / 2, np.finfo(float).eps)
    slab = df[np.abs(df["z"] - target) <= half_width].copy()
    if len(slab) < min_points:
        nearest = np.argsort(np.abs(values - target))[:min(min_points, len(df))]
        slab = df.iloc[nearest].copy()
    return slab, target


def plot_slice_heatmap(
    df: pd.DataFrame, value_col: str, output_path: str | Path, title: str,
    color_limits: Tuple[float, float], colorbar_label: str,
    quantile: float = 0.5, resolution: int = 500,
) -> None:
    grid, extent, target = _interpolate_slice_grid(
        df,
        value_col,
        quantile=quantile,
        resolution=resolution,
    )
    x_min, x_max, y_min, y_max = extent
    vmin, vmax = _safe_limits(color_limits)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(
        grid, origin="lower", extent=extent,
        aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, label=colorbar_label)
    ax.set_title(f"{title}\nz quantile {quantile:.2f}; z approximately {target:.4g} mm")
    ax.set(xlabel="x [mm]", ylabel="y [mm]", xlim=(x_min, x_max), ylim=(y_min, y_max))
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _interpolate_slice_grid(
    df: pd.DataFrame,
    value_col: str,
    quantile: float = 0.5,
    resolution: int = 500,
) -> Tuple[np.ma.MaskedArray | np.ndarray, Tuple[float, float, float, float], float]:
    """Interpolate one supported 3D slab onto an x-y image grid."""
    slab, target = select_slab(df.dropna(subset=[value_col]), quantile)
    projected = slab.groupby(["x", "y"], as_index=False)[value_col].mean()
    if len(projected) < 3:
        raise ValueError(f"Cannot plot {value_col}: fewer than three unique x-y points")
    x, y = projected["x"].to_numpy(), projected["y"].to_numpy()
    values = projected[value_col].to_numpy(dtype=float)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), resolution),
        np.linspace(y.min(), y.max(), resolution),
    )
    points = np.column_stack([x, y])
    try:
        linear = griddata(points, values, (grid_x, grid_y), method="linear")
        nearest = griddata(points, values, (grid_x, grid_y), method="nearest")
        grid = np.where(np.isnan(linear), nearest, linear)
    except Exception as exc:
        print(f"WARNING: linear interpolation failed for {value_col}: {exc}; using nearest")
        grid = griddata(points, values, (grid_x, grid_y), method="nearest")

    # Keep blank areas where no nearby mesh point supports the interpolation.
    tree = cKDTree(points)
    if len(points) > 1:
        spacing_values = tree.query(points, k=2)[0][:, 1]
        positive = spacing_values[spacing_values > 0]
        if len(positive):
            spacing = float(np.median(positive))
            distance = tree.query(np.column_stack([grid_x.ravel(), grid_y.ravel()]), k=1)[0]
            grid = np.ma.masked_where(distance.reshape(grid.shape) > 3 * spacing, grid)

    extent = (float(x.min()), float(x.max()), float(y.min()), float(y.max()))
    return grid, extent, target


def plot_status_slice(
    df: pd.DataFrame, output_path: str | Path, title: str, quantile: float = 0.5,
) -> None:
    if "Status" not in df:
        return
    slab, target = select_slab(df.dropna(subset=["Status"]), quantile)
    fig, ax = plt.subplots(figsize=(8, 7))
    for value, label in STATUS_LABELS.items():
        part = slab[slab["Status"] == value]
        if not part.empty:
            ax.scatter(part["x"], part["y"], s=2, label=label,
                       c=STATUS_COLORS[value], alpha=0.8, linewidths=0)
    ax.set_title(f"{title}\nz quantile {quantile:.2f}; z approximately {target:.4g} mm")
    ax.set(xlabel="x [mm]", ylabel="y [mm]")
    ax.set_aspect("equal")
    ax.legend(markerscale=4)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_horizontal_heatmap_comparisons(
    results_dir: Path,
    nodes: pd.DataFrame,
    iterations: Sequence[int],
    color_limits: Dict[str, Tuple[float, float]],
    output_dir: Path,
    iteration_seconds: float,
    quantile: float = 0.5,
    resolution: int = 500,
) -> None:
    """Arrange selected APAP, GSH, and NAPQI slices in horizontal rows."""
    requested = list(dict.fromkeys(int(value) for value in iterations))
    if not requested:
        return
    missing = [value for value in requested if not mald_file(results_dir, value).is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot create the requested horizontal comparisons because MALD "
            f"files are missing for iterations: {missing}"
        )

    fields = ("APAP", "GSH", "NAPQI")
    grids: Dict[str, Dict[int, Tuple[np.ma.MaskedArray | np.ndarray, Tuple[float, float, float, float]]]] = {
        field: {} for field in fields
    }
    for iteration in requested:
        mald = read_mald_file(mald_file(results_dir, iteration), verbose=False)
        df = add_analysis_columns(merge_nodes_and_mald(nodes, mald))
        #
        for field in fields:
            grid, extent, _ = _interpolate_slice_grid(
                df,
                field,
                quantile=quantile,
                resolution=resolution,
            )
            grids[field][iteration] = (grid, extent)

    labels = {
        "APAP": "Intracellular APAP [mol/cell]",
        "GSH": "Intracellular GSH [mol/cell]",
        "NAPQI": "Intracellular NAPQI [mol/cell]",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    iteration_tag = "_".join(f"{value:04d}" for value in requested)

    for field in fields:
        vmin, vmax = _safe_limits(color_limits[field])
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("white")
        fig, axes = plt.subplots(
            1,
            len(requested),
            figsize=(3.4 * len(requested) + 1.0, 4.1),
            squeeze=False,
            layout="constrained",
        )
        image = None
        for ax, iteration in zip(axes[0], requested):
            grid, extent = grids[field][iteration]
            image = ax.imshow(
                grid,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            time_hours = (iteration + 1) * iteration_seconds / 3600.0
            ax.set_title(f"Iteration {iteration}\n{time_hours:.2f} h")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{field} evolution")
        if image is not None:
            colorbar = fig.colorbar(image, ax=axes[0].tolist(), shrink=0.82, pad=0.015)
            colorbar.set_label(labels[field])
        fig.savefig(
            output_dir / f"horizontal_{field.lower()}_{iteration_tag}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def _save_line_plot(
    summary: pd.DataFrame, columns: Sequence[str], labels: Sequence[str],
    output_path: Path, title: str, ylabel: str,
) -> None:
    available = [(column, label) for column, label in zip(columns, labels) if column in summary]
    if not available:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for column, label in available:
        ax.plot(summary["time_hours"], summary[column], marker="o", markersize=2.5, label=label)
    ax.set(xlabel="Time [hours]", ylabel=ylabel, title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_state_panel_plot(
    summary: pd.DataFrame,
    columns: Sequence[str],
    labels: Sequence[str],
    output_path: Path,
    title: str,
    ylabel: str,
    x_column: str = "time_hours",
) -> None:
    """Plot APAP, NAPQI, and GSH on independent axes.

    Combining these molar states on one y-axis can make GSH recovery look like
    zero whenever APAP or NAPQI is one or more orders of magnitude larger.
    """
    available = [
        (column, label)
        for column, label in zip(columns, labels)
        if column in summary
    ]
    if not available:
        return

    fig, axes = plt.subplots(
        len(available), 1,
        figsize=(9, 3.2 * len(available)),
        sharex=True,
        squeeze=False,
    )
    colors = {"APAP": "#1f77b4", "NAPQI": "#d62728", "GSH": "#2ca02c"}
    for ax, (column, label) in zip(axes[:, 0], available):
        values = summary[column].to_numpy(dtype=float)
        ax.plot(
            summary[x_column], values,
            color=colors.get(label, "#1f77b4"),
            marker="o", markersize=2.5,
        )
        ax.set_ylabel(f"{label}\n{ylabel}")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.25)
        ax.set_title(
            f"{label}: start={values[0]:.4e}, "
            f"peak={np.max(values):.4e}, end={values[-1]:.4e}"
        )

    axes[-1, 0].set_xlabel("Time [hours]")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_ast_alt_panel(summary: pd.DataFrame, output_path: Path) -> None:
    """Reproduce Pinto's two-panel AST/ALT history with corrected units."""
    required = ("vw_mean_AST", "vw_mean_ALT")
    if not all(column in summary for column in required):
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, column, label, color in zip(
        axes,
        required,
        ("AST", "ALT"),
        ("#1f77b4", "#d62728"),
    ):
        ax.plot(summary["time_hours"], summary[column], color=color, marker="o", markersize=2.5)
        ax.set_ylabel(f"{label} [IU/L]")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [hours]")
    fig.suptitle("Mean AST and ALT in the lobule")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_representative_nodes(df: pd.DataFrame) -> Dict[str, int]:
    """Select anatomical counterparts to Pinto's three example nodes."""
    targets = {
        "Periportal": 0.0,
        "Midzonal": 0.5,
        "Pericentral": 1.0,
    }
    selected: Dict[str, int] = {}
    used: set[int] = set()
    for label, target in targets.items():
        candidates = df.loc[~df["id"].isin(used), ["id", "laue_xi"]].copy()
        row_index = (candidates["laue_xi"] - target).abs().idxmin()
        node_id = int(candidates.loc[row_index, "id"])
        selected[label] = node_id
        used.add(node_id)
    return selected


def plot_representative_node_histories(
    history: pd.DataFrame,
    representative_nodes: Dict[str, int],
    output_dir: Path,
) -> None:
    """Write Pinto-equivalent histories for three anatomical locations."""
    if history.empty:
        return
    labels = {
        "APAP": "Intracellular APAP [mol/cell]",
        "NAPQI": "Intracellular NAPQI [mol/cell]",
        "GSH": "Intracellular GSH [mol/cell]",
        "Healthy": "Healthy state",
        "Damaged": "Damaged state",
        "Necrosed": "Necrosed state",
        "Regeneration": "Regeneration state",
    }
    for column, ylabel in labels.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        for anatomical_label, node_id in representative_nodes.items():
            part = history[history["id"] == node_id].sort_values("time_hours")
            ax.plot(
                part["time_hours"],
                part[column],
                marker="o",
                markersize=2.5,
                label=f"{anatomical_label} (node {node_id})",
            )
        ax.set(
            xlabel="Time [hours]",
            ylabel=ylabel,
            title=f"Pinto-equivalent representative-node {column} history",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"pinto_representative_nodes_{column.lower()}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def pinto_zone_status_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return Pinto-shaped status counts using geometry-appropriate zones."""
    table = pd.crosstab(df["Status"], df["Pinto_zone"])
    table = table.reindex(index=list(STATUS_LABELS), columns=["Zone 1", "Zone 2", "Zone 3"], fill_value=0)
    table.index = [STATUS_LABELS[value] for value in table.index]
    table.index.name = "Status"
    return table


def write_pinto_zone_workbook(tables: Dict[int, pd.DataFrame], output_path: Path) -> None:
    """Write one zone/status sheet per available iteration, as Pinto did."""
    if not tables:
        return
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for iteration in sorted(tables):
            tables[iteration].to_excel(writer, sheet_name=f"Iteration {iteration}")


def plot_summary_timeseries(summary: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_state_panel_plot(
        summary, [f"mean_{x}_mol_per_cell" for x in STATE_AMOUNT_COLUMNS], STATE_AMOUNT_COLUMNS,
        output_dir / "mald_mean_apap_napqi_gsh_timeseries.png",
        "Mean intracellular APAP, NAPQI, and GSH (independent axes)", "[mol/cell]",
    )
    _save_line_plot(
        summary, [f"vw_mean_{x}" for x in CELL_STATE_COLUMNS], CELL_STATE_COLUMNS,
        output_dir / "mald_cell_state_timeseries.png", "Continuous MALD hepatocyte states",
        "Volume-weighted state",
    )
    _save_ast_alt_panel(summary, output_dir / "mald_ast_alt_timeseries.png")
    _save_line_plot(
        summary, ["vw_mean_Clotting"], ["Clotting"],
        output_dir / "mald_clotting_timeseries.png", "Clotting-factor state",
        "Volume-weighted state",
    )
    _save_line_plot(
        summary, [f"status_{x}_volume_pct" for x in STATUS_LABELS.values()], list(STATUS_LABELS.values()),
        output_dir / "mald_status_volume_percent.png", "Categorical hepatocyte status",
        "Tissue volume [%]",
    )
    _save_line_plot(
        summary, [f"status_{x}_count_pct" for x in STATUS_LABELS.values()], list(STATUS_LABELS.values()),
        output_dir / "mald_status_count_percent_pinto_equivalent.png",
        "Categorical hepatocyte status (Pinto-equivalent node counts)",
        "Numerical nodes [%]",
    )
    _save_line_plot(
        summary,
        [
            "damage_trigger_volume_pct",
            "necrosis_trigger_volume_pct",
            "regeneration_trigger_volume_pct",
        ],
        ["Damage", "Necrosis", "Regeneration"],
        output_dir / "status_transition_trigger_volume_percent.png",
        "Nodes satisfying Pinto-style status thresholds",
        "Tissue volume satisfying threshold [%]",
    )
    _save_line_plot(
        summary, ["vw_mean_COMSOL_APAP_mol_m3"], ["COMSOL APAP"],
        output_dir / "comsol_apap_timeseries.png", "Extracellular APAP from COMSOL",
        "Volume-weighted concentration [mol/m^3]",
    )


def _global_color_limits(results_dir: Path, iterations: Sequence[int]) -> Dict[str, Tuple[float, float]]:
    limits: Dict[str, Tuple[float, float]] = {}
    print("Scanning MALD files for global color limits...")
    for iteration in iterations:
        data = read_mald_file(mald_file(results_dir, iteration), verbose=False)
        for col in HEATMAP_COLUMNS:
            values = data[col].to_numpy(dtype=float)
            low = 0.0 if col in {"APAP", "NAPQI"} else float(np.min(values))
            high = float(np.max(values))
            old_low, old_high = limits.get(col, (low, high))
            limits[col] = (min(old_low, low), max(old_high, high))
    for col, (low, high) in limits.items():
        print(f"  {col}: {low:.8e} to {high:.8e}")
    return limits


def _plot_iterations(iterations: Sequence[int], every: int) -> List[int]:
    if every <= 0:
        raise ValueError("every must be at least 1")
    # Pinto plotted iteration numbers 0, 6, 12, ... . Select by iteration
    # number rather than taking every sixth available file; the latter would
    # incorrectly plot 0, 36, 72, ... when the simulator already saves every
    # sixth interval.
    selected = [iteration for iteration in iterations if iteration % every == 0]
    if iterations[-1] not in selected:
        selected.append(iterations[-1])
    return sorted(set(selected))


def visualize_results(
    results_dir: str | Path,
    field_nodes: str | Path = PROJECT_ROOT / "Mesh_Volume_Calculations" / "Field_Nodes.csv",
    iterations: Optional[Sequence[int]] = None,
    every: int = 6,
    slice_quantiles: Sequence[float] = (0.5,),
    max_scatter_points: int = 150000,
    iteration_seconds: float = 1200.0,
    resolution: int = 500,
    make_3d: bool = False,
    make_slices: bool = True,
    color_scale: str = "global",
    horizontal_iterations: Optional[Sequence[int]] = None,
    horizontal_only: bool = False,
) -> Path:
    results_dir = _resolve_project_path(results_dir)
    field_nodes = _resolve_project_path(field_nodes)

    if color_scale not in {"global", "spatial", "both"}:
        raise ValueError("color_scale must be 'global', 'spatial', or 'both'")

    nodes = read_field_nodes(field_nodes)
    available = discover_iterations(results_dir)

    if not available:
        raise FileNotFoundError(
            f"No _Used-Field_Hepatocytes-*.txt files found in {results_dir}"
        )

    if iterations is None:
        requested = available
    else:
        requested = sorted(set(int(value) for value in iterations))
    available_set = set(available)

    selected_iterations = [value for value in requested if value in available_set]

    if not selected_iterations:
        raise ValueError(
            "None of the requested iterations are available. "
            f"Available iterations: {available}"
        )

    plot_iterations = (
        selected_iterations
        if iterations is not None
        else _plot_iterations(selected_iterations, every)
    )
    plot_iteration_set = set(plot_iterations)
    color_limits = _global_color_limits(results_dir, selected_iterations)

    out_dir = results_dir / "Visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, float]] = []
    diagnostics: List[Dict[str, object]] = []
    stiffness_rows: List[Dict[str, float]] = [] #ode diagnositics
    representative_nodes: Optional[Dict[str, int]] = None
    representative_history_rows: List[Dict[str, float]] = []
    zone_status_tables: Dict[int, pd.DataFrame] = {}

    # Load the single multi-time COMSOL export once. Column 0 corresponds to
    # post-MALD iteration 0 at COMSOL_FIRST_OUTPUT_TIME_S, column 1 to
    # iteration 1, and so forth.
    multitime_comsol_path = find_multitime_comsol_apap_file(results_dir)
    multitime_comsol = None
    if multitime_comsol_path is not None:
        try:
            multitime_comsol = read_multitime_comsol_apap_export(
                multitime_comsol_path,
                nodes,
            )
        except Exception as exc:
            print(
                f"WARNING: could not read multi-time COMSOL export "
                f"{multitime_comsol_path}: {exc}"
            )
    else:
        print(
            "INFO: no single multi-time COMSOL APAP export was found; "
            "checking for legacy per-iteration exports."
        )

    for iteration in selected_iterations:
        mpath = mald_file(results_dir, iteration)
        mald = read_mald_file(mpath)
        df = add_analysis_columns(merge_nodes_and_mald(nodes, mald))

        stiffness_states = df[
            [
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
            ]
        ].to_numpy(dtype=float)

        # calculate stiffness metrics for the current iteration
        stiffness = MALD.CalculateStiffness(
            states=stiffness_states,
            gamma_local=df["GSH_bind_rate"].to_numpy(dtype=float),
            regen_rate_local=df["Regeneration_rate"].to_numpy(dtype=float),
            interval_s=iteration_seconds,
        )

        full_ratio = stiffness["full_ratio"]
        core_ratio = stiffness["core_ratio"]

        worst_local_index = int(np.nanargmax(full_ratio))
        worst_node_id = int(df.iloc[worst_local_index]["id"])

        fastest_rate = float(
            np.nanmax(stiffness["fastest_rate_s"])
        )
        slowest_rate = float(
            np.nanmin(stiffness["slowest_rate_s"])
        )

        stiffness_rows.append(
            {
                "iteration": iteration,
                "time_hours": (
                    (iteration + 1)
                    * iteration_seconds
                    / 3600.0
                ),
                "full_stiffness_median": float(
                    np.nanmedian(full_ratio)
                ),
                "full_stiffness_p95": float(
                    np.nanpercentile(full_ratio, 95)
                ),
                "full_stiffness_p99": float(
                    np.nanpercentile(full_ratio, 99)
                ),
                "full_stiffness_max": float(
                    np.nanmax(full_ratio)
                ),
                "core_stiffness_median": float(
                    np.nanmedian(core_ratio)
                ),
                "core_stiffness_p95": float(
                    np.nanpercentile(core_ratio, 95)
                ),
                "core_stiffness_p99": float(
                    np.nanpercentile(core_ratio, 99)
                ),
                "core_stiffness_max": float(
                    np.nanmax(core_ratio)
                ),
                "maximum_spectral_radius_per_s": float(
                    np.nanmax(
                        stiffness["spectral_radius_s"]
                    )
                ),
                "maximum_interval_spectral_radius": float(
                    np.nanmax(
                        stiffness[
                            "interval_spectral_radius"
                        ]
                    )
                ),
                "fastest_timescale_s": (
                    1.0 / fastest_rate
                ),
                "slowest_timescale_s": (
                    1.0 / slowest_rate
                ),
                "nodes_with_unstable_modes": int(
                    np.count_nonzero(
                        stiffness["unstable_modes"] > 0
                    )
                ),
                "worst_node_id": worst_node_id,
            }
        )

        if representative_nodes is None:
            representative_nodes = select_representative_nodes(df)
            print(
                "Pinto-equivalent representative nodes: "
                + ", ".join(
                    f"{label}={node_id}"
                    for label, node_id in representative_nodes.items()
                )
            )

        status, status_source = status_for_state_snapshot(results_dir, iteration, len(nodes))
        status_loaded = status is not None

        if status is not None:
            df = df.merge(
                status, on="id", how="left", validate="one_to_one"
            )
            zone_status_tables[iteration] = pinto_zone_status_table(df)

        comsol_loaded = False
        comsol_source = ""

        if multitime_comsol is not None:
            if iteration < multitime_comsol.shape[1]:
                comsol_values = multitime_comsol[:, iteration]
                physical_time_s = (
                    COMSOL_FIRST_OUTPUT_TIME_S
                    + iteration * iteration_seconds
                )
                comsol_source = (
                    f"{multitime_comsol_path} "
                    f"[concentration column {iteration}; "
                    f"t={physical_time_s:g} s]"
                )
            else:
                # Match the full-history simulator's explicit zero-APAP
                # continuation after the last exported COMSOL field.
                comsol_values = np.zeros(len(nodes), dtype=float)
                comsol_source = (
                    f"zero continuation after {multitime_comsol_path.name}"
                )

            df = df.merge(
                pd.DataFrame({
                    "id": nodes["id"].to_numpy(dtype=int),
                    "COMSOL_APAP": comsol_values,
                }),
                on="id",
                how="left",
                validate="one_to_one",
            )
            comsol_loaded = True
        else:
            # Backward-compatible fallback for the older workflow that wrote
            # one COMSOL concentration file per iteration.
            comsol_path = find_comsol_apap_file(results_dir, iteration)
            if comsol_path is not None:
                try:
                    df = df.merge(
                        read_comsol_apap_export(comsol_path, nodes),
                        on="id",
                        how="left",
                        validate="one_to_one",
                    )
                    comsol_loaded = True
                    comsol_source = str(comsol_path)
                except Exception as exc:
                    print(
                        f"WARNING: could not read COMSOL export "
                        f"{comsol_path}: {exc}"
                    )

        summary_rows.append(summarize_iteration(df, iteration, iteration_seconds))

        if representative_nodes is not None:
            representative_ids = set(representative_nodes.values())
            for _, row in df[df["id"].isin(representative_ids)].iterrows():
                representative_history_rows.append({
                    "iteration": float(iteration),
                    "time_hours": float((iteration + 1) * iteration_seconds / 3600.0),
                    "id": int(row["id"]),
                    **{
                        column: float(row[column])
                        for column in (*STATE_AMOUNT_COLUMNS, *CELL_STATE_COLUMNS)
                    },
                })
        diagnostic_row: Dict[str, object] = {
            "iteration": iteration, "mald_rows": len(mald),
            "status_loaded": status_loaded, "comsol_apap_loaded": comsol_loaded,
            "status_source": status_source,
            "mald_file": str(mpath), "comsol_file": comsol_source,
        }
        diagnostic_columns = list(HEATMAP_COLUMNS)
        if "COMSOL_APAP" in df:
            diagnostic_columns.append("COMSOL_APAP")
        for diagnostic_column in diagnostic_columns:
            for statistic, value in _field_diagnostics(df[diagnostic_column]).items():
                diagnostic_row[f"{diagnostic_column}_{statistic}"] = value
        diagnostics.append(diagnostic_row)

        if iteration not in plot_iteration_set or horizontal_only:
            continue

        time_hours = (iteration + 1) * iteration_seconds / 3600.0

        iteration_dir = out_dir / f"iteration_{iteration:04d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)

        labels = {
            "APAP": "Intracellular APAP [mol/cell]",
            "NAPQI": "Intracellular NAPQI [mol/cell]",
            "GSH": "Intracellular GSH [mol/cell]",
            "AST": "AST [IU/L]",
            "ALT": "ALT [IU/L]",
        }

        for col in HEATMAP_COLUMNS:
            statistics = _field_diagnostics(df[col])
            print(
                f"Iteration {iteration} {col}: "
                f"min={statistics['min']:.6e}, max={statistics['max']:.6e}, "
                f"CV={statistics['cv']:.6e}, "
                f"P01-P99/mean={statistics['p01_p99_range_pct_of_mean']:.6e}%"
            )

            for scale_tag, limits, scale_description in _plot_scale_specs(
                df[col], color_limits[col], color_scale
            ):
                title = (
                    f"MALD {col} state at {time_hours:.2f} h "
                    f"(end of iteration {iteration})\n{scale_description}"
                )

                if make_3d:
                    plot_3d_scatter(
                        df, col,
                        iteration_dir / f"mald_{col}_{scale_tag}_3d_iter_{iteration:04d}.png",
                        title, limits, labels[col], max_scatter_points,
                    )

                if make_slices:
                    for quantile in slice_quantiles:
                        suffix = f"zq{int(round(quantile * 100)):02d}"
                        plot_slice_heatmap(
                            df, col,
                            iteration_dir / f"mald_{col}_{scale_tag}_slice_{suffix}_iter_{iteration:04d}.png",
                            title, limits, labels[col], quantile, resolution,
                        )

        # Plot the extracellular COMSOL field separately from intracellular
        # MALD APAP. This is the fastest way to determine whether a missing
        # gradient originates in transport or in the cellular model.
        if "COMSOL_APAP" in df:
            comsol_statistics = _field_diagnostics(df["COMSOL_APAP"])
            print(
                f"Iteration {iteration} COMSOL_APAP: "
                f"min={comsol_statistics['min']:.6e}, "
                f"max={comsol_statistics['max']:.6e}, "
                f"CV={comsol_statistics['cv']:.6e}, "
                f"P01-P99/mean="
                f"{comsol_statistics['p01_p99_range_pct_of_mean']:.6e}%"
            )
            comsol_global_limits = (
                0.0,
                float(np.max(multitime_comsol))
                if multitime_comsol is not None else float(df["COMSOL_APAP"].max()),
            )
            for scale_tag, limits, scale_description in _plot_scale_specs(
                df["COMSOL_APAP"], comsol_global_limits, color_scale
            ):
                title = (
                    f"Extracellular COMSOL APAP at {time_hours:.2f} h\n"
                    f"{scale_description}"
                )
                if make_3d:
                    plot_3d_scatter(
                        df, "COMSOL_APAP",
                        iteration_dir / f"comsol_APAP_{scale_tag}_3d_iter_{iteration:04d}.png",
                        title, limits, "Extracellular APAP [mol/m^3]",
                        max_scatter_points,
                    )
                if make_slices:
                    for quantile in slice_quantiles:
                        suffix = f"zq{int(round(quantile * 100)):02d}"
                        plot_slice_heatmap(
                            df, "COMSOL_APAP",
                            iteration_dir / f"comsol_APAP_{scale_tag}_slice_{suffix}_iter_{iteration:04d}.png",
                            title, limits, "Extracellular APAP [mol/m^3]",
                            quantile, resolution,
                        )

        # Pinto wrote a status image at every selected time, including uniform
        # fields. Preserve that behavior in the comparison output.
        if "Status" in df and make_slices:
            for quantile in slice_quantiles:
                suffix = f"zq{int(round(quantile * 100)):02d}"
                plot_status_slice(
                    df, iteration_dir / f"status_slice_{suffix}_iter_{iteration:04d}.png",
                    f"Hepatocyte status at {time_hours:.2f} h (iteration {iteration})", quantile,
                )

    summary = pd.DataFrame(summary_rows).sort_values("iteration").reset_index(drop=True)
    summary.to_csv(out_dir / "mald_summary.csv", index=False)

    pd.DataFrame(diagnostics).to_csv(out_dir / "graphing_diagnostics.csv", index=False)

    # save stiffness diagnostics to a CSV file for further analysis
    stiffness_summary = (
        pd.DataFrame(stiffness_rows)
        .sort_values("iteration")
        .reset_index(drop=True)
    )

    stiffness_output = (
        out_dir / "mald_stiffness_summary.csv"
    )

    stiffness_summary.to_csv(
        stiffness_output,
        index=False,
    )

    if not stiffness_summary.empty:
        worst_row = stiffness_summary.loc[
            stiffness_summary[
                "full_stiffness_max"
            ].idxmax()
        ]

        print(
            "Maximum sampled patient stiffness: "
            f"{worst_row['full_stiffness_max']:.6e}; "
            f"iteration={int(worst_row['iteration'])}; "
            f"time={worst_row['time_hours']:.3f} h; "
            f"node={int(worst_row['worst_node_id'])}"
        ) # end of stiffness calculation 

    if not horizontal_only:
        plot_summary_timeseries(summary, out_dir)

        representative_history = pd.DataFrame(representative_history_rows)
        if representative_nodes is not None and not representative_history.empty:
            representative_history.to_csv(
                out_dir / "pinto_representative_node_history.csv",
                index=False,
            )
            pd.DataFrame(
                [
                    {"anatomical_location": label, "node_id": node_id}
                    for label, node_id in representative_nodes.items()
                ]
            ).to_csv(out_dir / "pinto_representative_nodes.csv", index=False)
            plot_representative_node_histories(
                representative_history,
                representative_nodes,
                out_dir,
            )

        write_pinto_zone_workbook(
            zone_status_tables,
            out_dir / "hepatocyte_zone_stats_pinto_equivalent.xlsx",
        )

    if horizontal_iterations is not None:
        horizontal_set = set(int(value) for value in horizontal_iterations)
        unavailable = sorted(horizontal_set - set(selected_iterations))
        if unavailable:
            raise ValueError(
                "Horizontal comparison iterations must also be included in "
                f"--iterations. Missing from the selected set: {unavailable}"
            )
        plot_horizontal_heatmap_comparisons(
            results_dir=results_dir,
            nodes=nodes,
            iterations=horizontal_iterations,
            color_limits=color_limits,
            output_dir=out_dir,
            iteration_seconds=iteration_seconds,
            quantile=slice_quantiles[0],
            resolution=resolution,
        )

    print(f"Done. Visualizations: {out_dir}")
    print(f"Summary: {out_dir / 'mald_summary.csv'}")
    print(f"Diagnostics: {out_dir / 'graphing_diagnostics.csv'}")
    print(
        "NOTE: _Used-Field_Hepatocytes-n and status-n are aligned "
        "post-MALD snapshots from the end of interval n."
    )
    
    return out_dir


def _parse_iterations(value: Optional[str]) -> Optional[List[int]]:
    if not value:
        return None
    output: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(item) for item in part.split("-", 1))
            if end < start:
                raise ValueError(f"Invalid iteration range: {part}")
            output.extend(range(start, end + 1))
        else:
            output.append(int(part))
    return sorted(set(output))


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph COMSOL -> MALD APAP simulation results")
    parser.add_argument("--results-dir", required=True, help="For example Results/Results-1")
    parser.add_argument(
        "--field-nodes",
        default=str(PROJECT_ROOT / "Mesh_Volume_Calculations" / "Field_Nodes.csv"),
        help="Headerless id,x_mm,y_mm,z_mm,node_control_volume_m3 file",
    )
    parser.add_argument("--iterations", default=None, help="For example 0-20 or 0,6,12,20")
    parser.add_argument(
        "--every",
        type=int,
        default=6,
        help="Plot iteration numbers divisible by this value; summaries use all available files",
    )
    parser.add_argument("--slice-quantiles", default="0.5", help="For example 0.25,0.5,0.75")
    parser.add_argument("--max-scatter-points", type=int, default=150000)
    parser.add_argument("--iteration-seconds", type=float, default=1200.0)
    parser.add_argument("--resolution", type=int, default=500)
    parser.add_argument(
        "--color-scale",
        choices=("spatial", "global", "both"),
        default="global",
        help=(
            "spatial reveals within-iteration gradients using 1st-99th "
            "percentiles; global preserves absolute cross-time comparison; "
            "both writes both versions"
        ),
    )
    parser.add_argument(
        "--make-3d",
        action="store_true",
        help="Also write 3D point clouds (off by default; slices are clearer and much smaller)",
    )
    parser.add_argument("--skip-slices", action="store_true")
    parser.add_argument(
        "--horizontal-iterations",
        default=None,
        help="Comma-separated iterations for APAP, GSH, and NAPQI horizontal comparison rows",
    )
    parser.add_argument(
        "--horizontal-only",
        action="store_true",
        help="Skip individual spatial images and write only the horizontal comparison rows",
    )
    args = parser.parse_args()

    quantiles = [float(item) for item in args.slice_quantiles.split(",") if item.strip()]
    requested_iterations = _parse_iterations(args.iterations)
    horizontal_iterations = _parse_iterations(args.horizontal_iterations)
    if args.horizontal_only and horizontal_iterations is None:
        horizontal_iterations = list(DEFAULT_HORIZONTAL_ITERATIONS)
    if args.horizontal_only and requested_iterations is None:
        requested_iterations = list(horizontal_iterations)
    visualize_results(
        results_dir=args.results_dir,
        field_nodes=args.field_nodes,
        iterations=requested_iterations,
        every=args.every,
        slice_quantiles=quantiles,
        max_scatter_points=args.max_scatter_points,
        iteration_seconds=args.iteration_seconds,
        resolution=args.resolution,
        make_3d=args.make_3d,
        make_slices=not args.skip_slices,
        color_scale=args.color_scale,
        horizontal_iterations=horizontal_iterations,
        horizontal_only=args.horizontal_only,
    )


if __name__ == "__main__":
    main()
