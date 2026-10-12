"""COMSOL backend for the lobule APAP -> MALD workflow.

This file is aligned to the uploaded COMSOL report:
    "3d - in MM Darcy and Drug Transport.mph"

Report-based defaults:
- Darcy's Law interface tag expected: dl
- Transport of Diluted Species in Porous Media expected variable: c / comp1.c
- Current transport study is Study 2, usually tag std2 in COMSOL-generated models
- Current parameter names: inlet_drug, inlet_pressure, outlet_pressure,
  p_cv_ext, porosity, viscous_resistance, relative_tolerance
- Current model uses scalar porosity and scalar viscous_resistance. The code still
  writes a spatial CSV, but also writes a scalar summary so the current report
  model can be driven without first adding interpolation functions.

The backend intentionally replaces only the Fluent-specific solver call.
The surrounding workflow remains:
    Excel APAP schedule -> COMSOL solve/export -> MALD update -> porosity update
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import math
import os
import shutil
import subprocess
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(value: Path | str) -> Path:
    """Resolve project-relative paths independently of the process cwd."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_comsol_batch() -> Path | str:
    """Find COMSOL batch, allowing an override for non-default installations."""
    override = os.environ.get("COMSOL_BATCH_PATH")
    if override:
        return Path(override).expanduser()

    on_path = shutil.which("comsolbatch")
    if on_path:
        return Path(on_path)

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates = sorted(
        program_files.glob(r"COMSOL\COMSOL*\Multiphysics\bin\win64\comsolbatch.exe"),
        reverse=True,
    )
    return candidates[0] if candidates else Path("comsolbatch")


@dataclass
class ComsolConfig:
    """Configuration for one COMSOL batch solve.

    Important: COMSOL uses feature *tags* internally. In a typical model with two
    studies, Study 1 is tag ``std1`` and Study 2 is tag ``std2``. Verify this in
    COMSOL Desktop by enabling *View > Model Builder Node Label > Show Name and Tag*.
    """

    model_template: Path = field(
        default_factory=lambda: project_path(r"COMSOL_Files\Darcy and Drug Transport.mph")
    )
    comsol_command: Tuple[str, ...] = field(
        default_factory=lambda: (str(find_comsol_batch()),)
    )

    #  Study 2 = time-dependent drug transport.
    study_tag: str = "std2"

    # Output names retained from the Fluent workflow where useful.
    apap_export_name: str = "Lobule-Flow-Drug_Simulation_3D_V3-1200.txt"
    fixed_comsol_export_path: Optional[Path] = field(
        default_factory=lambda: project_path(
            r"COMSOL_Files\Lobule-Flow-Drug_Simulation_3D_V3-1200.txt"
        )
    )

    output_model_name: str = "COMSOL_Lobule_Iteration.mph"

    # The COMSOL report uses these parameter names.
    parameter_names: Dict[str, Optional[str]] = field(
        default_factory=lambda: {
            "inlet_drug": "inlet_drug",
            "inlet_pressure": "inlet_pressure",
            "outlet_pressure": "outlet_pressure",
            "neighbor_pressure": "p_cv_ext",
            "porosity": "porosity",
            "viscous_resistance": "viscous_resistance",
            "relative_tolerance": "relative_tolerance",
            # The uploaded report does NOT currently show these parameters.
            # Leave them as None unless you add them to the .mph model.
            "t_end": None,
            "porosity_file": None,
            "apap_export_file": None,
        }
    )

    # Units expected by the current report. The old Fluent file used a dimensionless
    # species/mass-fraction expression; the COMSOL report uses mol/m^3.
    drug_unit: str = "mol/m^3"
    pressure_unit: str = "Pa"
    time_unit: str = "s"

    neighbor_pressure: str = "650[Pa]"
    relative_tolerance: str = "1e-3"

    # Current report mode: scalar porosity/resistance. Advanced mode can use the
    # spatial CSV if you add interpolation functions in COMSOL.
    porosity_update_mode: str = "scalar"  # "scalar" or "field_csv"
    interpolation_function_dir: Path = field(
        default_factory=lambda: project_path("COMSOL_FunctionFiles")
    )

    # Extra model parameters to send through COMSOL batch, e.g.
    # {"my_param": "1.23[unit]"}
    extra_parameters: Dict[str, str] = field(default_factory=dict)

    extra_batch_args: Tuple[str, ...] = ("-nosave",) # don't save model after comsol run
    strict_export_check: bool = True
    dry_run: bool = False


def _as_path(value) -> Path:
    return project_path(value)


def comsol_expr(value, unit: Optional[str] = None) -> str:
    """Return a COMSOL parameter expression.

    Examples:
        800, "Pa"       -> "800[Pa]"
        "800[Pa]", None -> "800[Pa]"
        "1e-3", None    -> "1e-3"
    """
    if isinstance(value, (int, float)):
        s = f"{value:g}"
    else:
        s = str(value).strip().replace(" ", "")
    return s


def read_field_nodes(field_nodes_path: Path) -> List[Tuple[int, float, float, float]]:
    """Read Field_Nodes.txt as (node_id, x, y, z).

    Expected format from your existing initialization code:
        node_id,x,y,z
    with no header.
    """
    nodes: List[Tuple[int, float, float, float]] = []
    path = _as_path(field_nodes_path)
    with path.open("r", newline="") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_no} expected at least 4 comma-separated columns")
            try:
                node_id = int(float(parts[0]))
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no} could not parse node_id,x,y,z from {parts[:4]}") from exc
            nodes.append((node_id, x, y, z))
    if not nodes:
        raise ValueError(f"No nodes found in {path}")
    return nodes


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("cannot compute mean of empty sequence")
    return sum(vals) / len(vals)



def write_xyz_value_file(path: Path, rows: Sequence[Tuple[float, float, float, float]]) -> Path:
    """Write a COMSOL interpolation function file with columns x,y,z,value."""
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "value"])
        writer.writerows(rows)
    return path


def write_comsol_function_files_from_master(master_csv_path: Path, function_dir: Path = Path("COMSOL_FunctionFiles")) -> Dict[str, Path]:
    """Split COMSOL_Porosity_Fields.csv into function-specific CSVs.

    COMSOL interpolation functions are simpler and less error-prone when each file
    contains only x,y,z,value. This avoids column-index mistakes caused by the
    node_id and multiple property columns in the master coupling file.
    """
    master_csv_path = _as_path(master_csv_path)
    function_dir = _as_path(function_dir)
    with master_csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "x", "y", "z", "porosity", "viscous_resistance_1_per_m2",
            "permeability_m2_from_viscous", "status",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{master_csv_path} missing required columns: {sorted(missing)}")
        porosity_rows = []
        permeability_rows = []
        viscous_rows = []
        status_rows = []
        for row in reader:
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            porosity_rows.append((x, y, z, float(row["porosity"])))
            permeability_rows.append((x, y, z, float(row["permeability_m2_from_viscous"])))
            viscous_rows.append((x, y, z, float(row["viscous_resistance_1_per_m2"])))
            status_rows.append((x, y, z, float(row["status"])))

    return {
        "porosity": write_xyz_value_file(function_dir / "porosity_xyz.csv", porosity_rows),
        "permeability": write_xyz_value_file(function_dir / "permeability_xyz.csv", permeability_rows),
        "viscous_resistance": write_xyz_value_file(function_dir / "viscous_resistance_xyz.csv", viscous_rows),
        "status": write_xyz_value_file(function_dir / "status_xyz.csv", status_rows),
    }

def write_comsol_porosity_csv(
    *,
    output_path: Path,
    field_nodes_path: Path,
    porosity_array,
    summary_path: Optional[Path] = None,
    function_dir: Optional[Path] = Path("COMSOL_FunctionFiles"),
) -> Dict[str, float]:
    """Write COMSOL porosity/resistance files.

    ``porosity_array`` must use the original MALD loop convention:
        column 0 = porosity
        column 1 = inertial resistance, 1/m
        column 2 = viscous resistance, 1/m^2
        column 3 = status

    The current COMSOL report uses scalar ``porosity`` and scalar
    ``viscous_resistance`` with ``k = 1/viscous_resistance``. Therefore this
    function also writes a scalar summary CSV that can be passed through
    COMSOL batch as global parameters.
    """
    output_path = _as_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nodes = read_field_nodes(_as_path(field_nodes_path))
    if len(nodes) != len(porosity_array):
        raise ValueError(
            f"Porosity row count ({len(porosity_array)}) does not match Field_Nodes count ({len(nodes)})"
        )

    porosities = []
    inertial = []
    viscous = []
    statuses = []

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "node_id",
            "x",
            "y",
            "z",
            "porosity",
            "inertial_resistance_1_per_m",
            "viscous_resistance_1_per_m2",
            "permeability_m2_from_viscous",
            "status",
        ])
        for (node_id, x, y, z), row in zip(nodes, porosity_array):
            por = float(row[0])
            inert = float(row[1])
            visc = float(row[2])
            status = float(row[3])
            permeability = 1.0 / visc if visc != 0 else math.nan
            porosities.append(por)
            inertial.append(inert)
            viscous.append(visc)
            statuses.append(status)
            writer.writerow([node_id, x, y, z, por, inert, visc, permeability, status])

    summary = {
        "mean_porosity": _mean(porosities),
        "mean_inertial_resistance_1_per_m": _mean(inertial),
        "mean_viscous_resistance_1_per_m2": _mean(viscous),
        "mean_permeability_m2_from_viscous": 1.0 / _mean(viscous),
        "healthy_or_regen_count_status_3": float(sum(1 for s in statuses if s == 3.0)),
        "damaged_count_status_1": float(sum(1 for s in statuses if s == 1.0)),
        "lysed_count_status_2": float(sum(1 for s in statuses if s == 2.0)),
        "row_count": float(len(statuses)),
    }

    if summary_path is not None:
        summary_path = _as_path(summary_path)
        with summary_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "value"])
            for key, value in summary.items():
                writer.writerow([key, value])

    if function_dir is not None:
        write_comsol_function_files_from_master(output_path, function_dir)

    return summary


def read_porosity_summary(summary_path: Path) -> Dict[str, float]:
    path = _as_path(summary_path)
    if not path.exists():
        raise FileNotFoundError(path)
    data: Dict[str, float] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "name" not in row or "value" not in row:
                continue
            data[row["name"]] = float(row["value"])
    required = {"mean_porosity", "mean_viscous_resistance_1_per_m2"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Missing required summary keys in {path}: {sorted(missing)}")
    return data


def read_apap_export(export_path: Path, expected_rows: Optional[int] = None) -> List[float]:
    """Read COMSOL APAP export as one concentration value per Field_Nodes row.

    Accepted formats:
    - CSV with an ``apap`` or ``c`` column.
    - COMSOL text/table export where the last numeric column is concentration.

    Do not use a COMSOL mesh-node export unless the rows exactly match
    Field_Nodes.txt. The MALD coupling depends on row order.
    """
    path = _as_path(export_path)
    if not path.exists():
        raise FileNotFoundError(f"APAP export not found: {path}")

    values: List[float] = []
    with path.open("r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        # Headered CSV path.
        if "," in sample.splitlines()[0]:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                lower_names = {name.strip().lower(): name for name in reader.fieldnames}
                value_col = None
                for candidate in ("apap", "c", "comp1.c", "concentration", "molar concentration"):
                    if candidate in lower_names:
                        value_col = lower_names[candidate]
                        break
                if value_col is not None:
                    for row in reader:
                        if row.get(value_col, "").strip() == "":
                            continue
                        values.append(float(row[value_col]))
                    if expected_rows is not None and len(values) != expected_rows:
                        raise ValueError(
                            f"APAP export row count {len(values)} != expected Field_Nodes count {expected_rows}"
                        )
                    return values
                f.seek(0)

        # Generic fallback: last numeric token on each non-comment row.
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%") or stripped.startswith("#"):
                continue
            tokens = stripped.replace(",", " ").split()
            numeric = []
            for token in tokens:
                try:
                    numeric.append(float(token))
                except ValueError:
                    pass
            if numeric:
                values.append(numeric[-1])

    if expected_rows is not None and len(values) != expected_rows:
        raise ValueError(
            f"APAP export row count {len(values)} != expected Field_Nodes count {expected_rows}. "
            "Check that COMSOL exported concentration at COMSOL_Eval_Points.csv, not mesh nodes."
        )
    return values


class ComsolBatchRunner:
    """Run one COMSOL iteration and return the APAP export file path."""

    def __init__(self, config: ComsolConfig):
        self.config = config

    def _build_parameters(
        self,
        *,
        drug_concentration,
        inlet_pressure,
        outlet_pressure,
        neighbor_pressure,
        time_iteration_s,
        porosity_summary_path: Path,
        porosity_csv: Path,
        apap_export_path: Path,
    ) -> Dict[str, str]:
        cfg = self.config
        summary = read_porosity_summary(porosity_summary_path)
        params: Dict[str, str] = {}

        def add(logical_name: str, value: str):
            tag = cfg.parameter_names.get(logical_name)
            if tag:
                params[tag] = value

        add("inlet_drug", f"{drug_concentration},{cfg.drug_unit}")
        add("inlet_pressure", comsol_expr(inlet_pressure, cfg.pressure_unit))
        add("outlet_pressure", comsol_expr(outlet_pressure, cfg.pressure_unit))
        add("neighbor_pressure", comsol_expr(neighbor_pressure, cfg.pressure_unit))
        add("relative_tolerance", str(cfg.relative_tolerance))
        add("porosity", f"{summary['mean_porosity']:.12g}")
        add("viscous_resistance", f"{summary['mean_viscous_resistance_1_per_m2']:.12g}")
        add("t_end", comsol_expr(time_iteration_s, cfg.time_unit))
        add("porosity_file", str(_as_path(porosity_csv).resolve()))
        add("apap_export_file", str(_as_path(apap_export_path).resolve()))

        params.update(cfg.extra_parameters)
        return params

    def build_command(self, *, input_model: Path, output_model: Path, params: Dict[str, str]) -> List[str]:
        cfg = self.config
        cmd: List[str] = list(cfg.comsol_command)
        cmd += [
            "-inputfile",
            str(input_model),
            "-outputfile",
            str(output_model),
        ]
        if cfg.study_tag:
            cmd += ["-study", cfg.study_tag]
        if params:
            cmd += ["-pname", ",".join(params.keys())]
            cmd += ["-plist", ",".join(params.values())]
        cmd.extend(cfg.extra_batch_args)
        return cmd

    def run_iteration(
        self,
        *,
        iteration: int,
        run_directory: Path,
        drug_concentration,
        inlet_pressure,
        outlet_pressure,
        time_iteration_s: int,
        porosity_csv: Path,
        porosity_summary_csv: Path,
        neighbor_pressure: Optional[str] = None,
    ) -> Path:
        cfg = self.config
        run_directory = _as_path(run_directory)
        run_directory.mkdir(parents=True, exist_ok=True)

        input_model = _as_path(cfg.model_template)
        if not input_model.exists() and not cfg.dry_run:
            raise FileNotFoundError(
                f"COMSOL model template not found: {input_model}. "
                "Put the .mph file in the working directory or update ComsolConfig.model_template."
            )

        output_model = run_directory / f"iteration_{iteration:04d}_{cfg.output_model_name}"
        # This is where Python will store a per-iteration copy.
        apap_export_path = run_directory / cfg.apap_export_name
        neighbor_pressure = neighbor_pressure if neighbor_pressure is not None else cfg.neighbor_pressure

        params = self._build_parameters(
            drug_concentration=drug_concentration,
            inlet_pressure=inlet_pressure,
            outlet_pressure=outlet_pressure,
            neighbor_pressure=neighbor_pressure,
            time_iteration_s=time_iteration_s,
            porosity_summary_path=porosity_summary_csv,
            porosity_csv=porosity_csv,
            apap_export_path=apap_export_path,
        )
        cmd = self.build_command(input_model=input_model, output_model=output_model, params=params)

        log_path = run_directory / f"iteration_{iteration:04d}_comsol_batch.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write("Command:\n")
            log.write(" ".join(cmd) + "\n\n")
            log.write("Parameters:\n")
            for k, v in params.items():
                log.write(f"  {k} = {v}\n")

        if cfg.dry_run:
            return apap_export_path

        executable = Path(cmd[0])
        if executable.parent != Path(".") and not executable.exists():
            raise FileNotFoundError(
                f"COMSOL batch executable was not found: {executable}. "
                "Set COMSOL_BATCH_PATH to the full path to comsolbatch.exe."
            )
        if executable.parent == Path(".") and shutil.which(cmd[0]) is None:
            raise FileNotFoundError(
                "COMSOL batch executable 'comsolbatch' was not found on PATH. "
                "Set COMSOL_BATCH_PATH to the full path to comsolbatch.exe."
            )

        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)

        # COMSOL may write to a fixed hardcoded export path inside the .mph model.
        # If so, copy that file into the per-iteration results folder.
        if cfg.fixed_comsol_export_path is not None:
            fixed_export = _as_path(cfg.fixed_comsol_export_path)

            if not fixed_export.exists():
                raise FileNotFoundError(
                    f"COMSOL finished, but the fixed COMSOL export file was not found:\n"
                    f"{fixed_export}\n\n"
                    f"Check the Results > Export filename in COMSOL."
                )

            apap_export_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixed_export, apap_export_path)

        if cfg.strict_export_check and not apap_export_path.exists():
            raise FileNotFoundError(
                f"COMSOL finished but APAP export was not found:\n"
                f"{apap_export_path}\n\n"
                f"Also checked fixed COMSOL export path:\n"
                f"{cfg.fixed_comsol_export_path}"
            )

        return apap_export_path

def copy_if_exists(src: Path, dst: Path) -> None:
    src = _as_path(src)
    dst = _as_path(dst)
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
