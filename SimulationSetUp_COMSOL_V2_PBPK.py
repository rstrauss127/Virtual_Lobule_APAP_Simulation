# -*- coding: utf-8 -*-
"""Top-level setup for the COMSOL-based lobule simulation workflow.

Patient ID, time since ingestion, and APAP dose are read from the Excel file.
For each patient, the PBPK model generates the time-dependent APAP inlet file.
The already-computed multi-time COMSOL export is then replayed through MALD.

Excel layout when read with pandas' default header behavior:
    csvfile[0][c]  -> Excel row 2: patient ID
    csvfile[1][c]  -> Excel row 3: time since ingestion [days]
    csvfile[23][c] -> Excel row 25: APAP dose [g]

Patients begin in Excel column C, so the patient loop begins at column index 2.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import Virtual_Lobule_Script_COMSOL_V2 as sim
from pbpk_apap import simulate_apap_pbpk, apply_pinto_compatible_cfd_cutoff
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(*parts):
    return PROJECT_ROOT.joinpath(*parts)

# -----------------------------------------------------------------------------
# Simulation settings
# -----------------------------------------------------------------------------
# One COMSOL-output/MALD interval, in seconds.
timeIteration = 1200

input_excel = project_path("InputValues_Temp.xlsx")
results_root = project_path("Results")

# Set to an integer while testing, or None to run every patient iteration.
iteration_cap = 12
#PATIENT_START_COLUMN_INDEX = 3

def main():
    csvfile = pd.read_excel(input_excel).values

    for c in (4, 2, 5, 3): # Iterate over the specified columns
    #for c in range(PATIENT_START_COLUMN_INDEX, csvfile.shape[1]):    
        if pd.isna(csvfile[0][c]):
            continue
        patient_num = str(int(csvfile[0][c]))
        #time_since_ingestion_days = float(csvfile[1][c])
        total_mass_g = float(csvfile[23][c])

        total_iterations  = int(float(csvfile[1][c]) * 72)

        #if not np.isclose(raw_iterations, total_iterations, rtol=0.0, atol=1.0e-9):
        #    raise ValueError(
        #        f"Patient {patient_num}: time since ingestion "
        #        f"({time_since_ingestion_days} days) is not an integer multiple "
        #        f"of the {timeIteration}-s MALD interval."
        #    )



        duration_s = total_iterations * timeIteration
        pbpk = simulate_apap_pbpk(
            dose_g=total_mass_g,
            duration_s=duration_s,
            dt_s=timeIteration,
        )

        # Boundary samples are written at their real PBPK times. Do not label
        # interval averages as values at the interval starts.
        pbpk_time_s = np.asarray(pbpk["time_s"], dtype=float)

        boundary_concentrations = np.asarray(
            pbpk["C_boundary_mol_m3"], dtype=float
        )

        
        expected_samples = total_iterations + 1
        if len(boundary_concentrations) != expected_samples:
            raise RuntimeError(
                f"Patient {patient_num}: PBPK returned "
                f"{len(boundary_concentrations)} boundary samples; "
                f"expected {expected_samples}."
            )

        directory = results_root / f"Results-{patient_num}"
        directory.mkdir(parents=True, exist_ok=True)

        pinto_concentration, simulation_end_s, cfd_stop_index = (
            apply_pinto_compatible_cfd_cutoff(
                pbpk_time_s,
                boundary_concentrations,
                relative_cutoff=1.0e-1,
                consecutive_below=3,
            )
        )

        np.savetxt(
            directory / "drug_inlet.csv",
            np.column_stack((pbpk_time_s, boundary_concentrations)),
            delimiter=",",
            fmt=["%d", "%.12e"],
        )
        np.savetxt(
            directory / "Pinto_drug_inlet.csv",
            np.column_stack((pbpk_time_s, pinto_concentration)),
            delimiter=",",
            fmt=["%d", "%.12e"],
        )
        print(
            f"Patient {patient_num}: cutoff index={cfd_stop_index}, "
            f"cutoff time={simulation_end_s:g} s"
        )


        firstIteration = 0

        sim.run(
            firstIteration,
            totalIterations=total_iterations,
            timeIteration=timeIteration,
            fileNamePrefixe=str(directory),
        )


if __name__ == "__main__":
    main()
