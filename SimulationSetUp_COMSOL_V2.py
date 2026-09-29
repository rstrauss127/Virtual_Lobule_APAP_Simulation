# -*- coding: utf-8 -*-
"""Top-level setup for the COMSOL-based lobule simulation workflow.

This replaces SimulationSetUp_V2.py. It no longer imports pyfluent and no
longer launches a solver directly. It reads the Excel dosing schedule, builds
patient APAP input arrays, then delegates each patient to
Virtual_Lobule_Script_COMSOL_V2.run().

Default COMSOL settings are aligned to the uploaded report:
    3d - in MM Darcy and Drug Transport.mph
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

import Virtual_Lobule_Script_COMSOL_V2 as sim #original simulation setup 
from comsol_backend import ComsolConfig, find_comsol_batch, project_path

# COMSOL report parameters.
inletPressure = "800[Pa]"
outletPressure = "500[Pa]"
neighborPressure = "650[Pa]"  # p_cv_ext / neighboring-lobule pressure

# The uploaded COMSOL report uses range(0,1,1200) s for Study 2.
timeIteration = 1200

input_excel = project_path("InputValues_Temp.xlsx")
results_root = project_path("Results")

# Keep the iteration cap from the original script while testing. Set to None to run all.
iteration_cap = None

# Check if files exist
comsol_exe = Path(find_comsol_batch())
model_file = project_path(r"COMSOL_Files\Darcy and Drug Transport.mph")

print("COMSOL exists:", comsol_exe.exists(), comsol_exe)
print("Model exists:", model_file.exists(), model_file)

comsol_config = ComsolConfig(
    model_template=model_file,
    # Command line for COMSOL batch execution. Adjust path if necessary.
    comsol_command=(str(comsol_exe),),
    # Typical COMSOL tag: Study 1 -> std1, Study 2 -> std2. Verify in your .mph file.
    study_tag="std2",
    
    apap_export_name="Lobule-Flow-Drug_Simulation_3D_V3-1200.txt",
    fixed_comsol_export_path=project_path(
        r"COMSOL_Files\Lobule-Flow-Drug_Simulation_3D_V3-1200.txt"
    ),
    # Current report uses inlet_drug [mol/m^3]. If your Excel values are still Fluent
    # mass fractions, convert them before passing them to COMSOL.
    drug_unit="mol/m^3",
    neighbor_pressure=neighborPressure,
)

csvfile = pd.read_excel(input_excel).values

## For each patient run a simulation and generate the results of this simulation

for c in range(2, len(csvfile[0])):
#for c in range(5, len(csvfile[0])): #Start at different patient ID
    patientNum = str(int(csvfile[0][c]))
    totalIterations = int(csvfile[1][c] * 72)
    totalMass = csvfile[23][c]  # retained for compatibility/logging

    concentration = ["0.0" for _ in range(totalIterations)]
    i = 1
    if totalIterations > 21:
        for r in range(2, 23):
            concentration[i] = str(csvfile[r][c])
            i += 1
    else:
        for r in range(2, totalIterations + 1):
            concentration[i] = str(csvfile[r][c])
            i += 1

    directory = results_root / f"Results-{patientNum}"
    directory.mkdir(parents=True, exist_ok=True)


    first_iteration = 0
    #first_iteration = 6
    max_iteration = totalIterations if iteration_cap is None else min(totalIterations, iteration_cap)

    print(f"Patient {patientNum}: totalMass={totalMass}; iterations={max_iteration}")
    sim.run(
        first_iteration,
        max_iteration,
        concentration,
        inletPressure,
        outletPressure,
        timeIteration,
        str(directory),
        runCFD=True,
        comsol_config=comsol_config,
        neighborPressure=neighborPressure,
    )
