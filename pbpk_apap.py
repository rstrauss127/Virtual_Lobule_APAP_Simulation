# -*- coding: utf-8 -*-
"""Sluka whole-body APAP PBPK model with the published Ghosh liver coupling.

This file preserves the existing simulation-setup interface::

    pbpk = simulate_apap_pbpk(
        dose_g=dose_g,
        duration_s=duration_s,
        dt_s=time_iteration,
    )

    concentration = pbpk["C_boundary_mol_m3"]

Implemented from available source material
------------------------------------------
1. Sluka et al. (2016), Table 1:
   whole-body APAP PBPK transfer-rate equations.
2. Sluka et al. (2016), Table 2:
   REFSIM APAP flow, volume, partition, binding, absorption, and renal
   filtration parameters for a 70 kg reference adult.

4. Ghosh et al. (2026), Equation 1:
   arterial and portal APAP delivery are combined and normalized by liver
   volume.

Not available in Ghosh
----------------------
Ghosh et al. do not publish their complete modified PBPK ODEs, complete
parameter set, source code, or the exact conversion from their liver-delivery
quantity to the CFD inlet boundary. This implementation therefore uses the
published Sluka whole-body APAP PBPK equations and parameters, calculates the
published Ghosh liver-delivery term, and additionally returns a flow-weighted
concentration for the single COMSOL concentration boundary. That final
flow-weighted concentration is a dimensional interface conversion, not an
extra equation reported by Ghosh.

Units
-----
Amounts: mol
Time: s
Volumes: L
Flows and clearances: L/s
Concentrations returned as both mol/L and mol/m^3
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path


@dataclass(frozen=True)
class APAPPBPKParams:
    """Sluka REFSIM parameters for APAP in a 70 kg reference adult.

    Parameter names are Python versions of the names in Sluka Table 2.
    Values reported in L/h or h^-1 are converted here to L/s or s^-1.
    """

    # APAP molecular weight. Used only to convert the oral dose from g to mol.
    MW_g_mol: float = 151.16

    # Reference body weight reported for REFSIM. The published parameter values
    # below are already scaled to this 70 kg reference individual.
    body_weight_kg: float = 70.0

    # Blood-flow rates [L/s] - Sluka Table 2.
    Q_cardiac_L_s: float = 363.01 / 3600.0
    Q_gut_L_s: float = 74.42 / 3600.0
    Q_liver_L_s: float = 19.42 / 3600.0
    Q_kidney_L_s: float = 80.37 / 3600.0
    Q_rest_L_s: float = 188.80 / 3600.0

    # Perfusable compartment volumes [L] - Sluka Table 2.
    V_art_L: float = 1.50
    V_ven_L: float = 3.41
    V_gut_L: float = 1.10
    V_liver_L: float = 1.71
    V_kidney_L: float = 0.29
    V_lung_L: float = 0.51
    V_rest_L: float = 33.47

    # APAP-dependent parameters - Sluka Table 2.
    k_gut_abs_s: float = 1.5 / 3600.0
    F_up: float = 0.8
    K_rest_to_plasma: float = 1.6
    K_kidney_to_plasma: float = 1.0
    K_liver_to_plasma: float = 1.0
    R_blood_to_plasma: float = 1.09
    Q_gfr_L_s: float = 0.714 / 3600.0

    # updated 8/26 RLS.
    k_metab_s_inv: float = 9.5 / 3600.0


DEFAULT_PARAMS = APAPPBPKParams()


def _validate_inputs(
    dose_g: float,
    duration_s: float,
    dt_s: float,
    p: APAPPBPKParams,
) -> None:
    """Validate simulation inputs and source parameters."""

    for name, value in {
        "dose_g": dose_g,
        "duration_s": duration_s,
        "dt_s": dt_s,
    }.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive; received {value!r}.")

    positive = {
        "MW_g_mol": p.MW_g_mol,
        "body_weight_kg": p.body_weight_kg,
        "Q_cardiac_L_s": p.Q_cardiac_L_s,
        "Q_gut_L_s": p.Q_gut_L_s,
        "Q_liver_L_s": p.Q_liver_L_s,
        "Q_kidney_L_s": p.Q_kidney_L_s,
        "Q_rest_L_s": p.Q_rest_L_s,
        "V_art_L": p.V_art_L,
        "V_ven_L": p.V_ven_L,
        "V_gut_L": p.V_gut_L,
        "V_liver_L": p.V_liver_L,
        "V_kidney_L": p.V_kidney_L,
        "V_lung_L": p.V_lung_L,
        "V_rest_L": p.V_rest_L,
        "k_gut_abs_s": p.k_gut_abs_s,
        "F_up": p.F_up,
        "K_rest_to_plasma": p.K_rest_to_plasma,
        "K_kidney_to_plasma": p.K_kidney_to_plasma,
        "K_liver_to_plasma": p.K_liver_to_plasma,
        "R_blood_to_plasma": p.R_blood_to_plasma,
        "Q_gfr_L_s": p.Q_gfr_L_s,
    }
    for name, value in positive.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive; received {value!r}.")

    if not np.isfinite(p.k_metab_s_inv) or p.k_metab_s_inv < 0.0:
        raise ValueError(
            "k_metab_s_inv must be finite and nonnegative; "
            f"received {p.k_metab_s_inv!r}."
        )

import numpy as np


def apply_pinto_compatible_cfd_cutoff(
    pbpk_time_s,
    drug_concentration,
    relative_cutoff=1.0e-4,
    consecutive_below=3,
):
    time_s = np.asarray(pbpk_time_s, dtype=float)
    concentration = np.asarray(
        drug_concentration,
        dtype=float,
    ).copy()

    if time_s.shape != concentration.shape:
        raise ValueError(
            "pbpk_time_s and drug_concentration must have "
            "the same shape."
        )

    if concentration.size == 0:
        raise ValueError("The PBPK schedule is empty.")

    concentration = np.maximum(concentration, 0.0)

    peak_index = int(np.argmax(concentration))
    peak_concentration = float(concentration[peak_index])

    if peak_concentration <= 0.0:
        concentration[:] = 0.0
        return concentration, float(time_s[0]), 0

    cutoff = relative_cutoff * peak_concentration
    below_count = 0
    stop_index = None

    for index in range(peak_index + 1, len(concentration)):
        if concentration[index] <= cutoff:
            below_count += 1
        else:
            below_count = 0

        if below_count >= consecutive_below:
            stop_index = index - consecutive_below + 1
            break

    if stop_index is None:
        simulation_end_s = float(time_s[-1])
    else:
        # Convert the PBPK tail into Pinto's exact-zero representation.
        concentration[stop_index:] = 0.0
        simulation_end_s = float(time_s[stop_index])

    return concentration, simulation_end_s, stop_index
    
def simulate_apap_pbpk(
    dose_g: float,
    duration_s: float,
    dt_s: float = 1200.0,
    params: APAPPBPKParams | None = None,
    *,
    rtol: float = 1.0e-8,
    atol: float | np.ndarray = 1.0e-12,
    max_step: float = np.inf,
) -> dict[str, np.ndarray]:
    """Run the Sluka whole-body APAP PBPK model and build the COMSOL schedule.

    Parameters
    ----------
    dose_g:
        Oral APAP dose placed initially in the gut-lumen compartment [g].
    duration_s:
        Simulation duration beginning at ingestion [s].
    dt_s:
        Output interval [s]. This can remain equal to the COMSOL/MALD coupling
        interval (currently 1200 s).
    params:
        Optional replacement parameter object. By default, the published
        Sluka REFSIM APAP values are used.
    rtol, atol, max_step:
        Optional SciPy solver controls. Defaults preserve the existing
        integration settings; these are useful for accuracy studies.

    State variables [mol]
    ---------------------
    A_gut_lumen, A_gut, A_liver, A_ven, A_lung, A_art, A_kidney,
    A_rest, A_tubules, A_metabolized.

    Main outputs
    ------------
    C_art_mol_L:
        Sluka arterial-compartment concentration, A_art / V_art.
    C_pv_mol_L:
        Portal concentration represented by the well-stirred gut-compartment
        outflow concentration, A_gut / V_gut.
    I_liver_mol_L_s:
        Ghosh Equation 1 liver-delivery rate:
        (Q_liver*C_art + Q_gut*C_pv) / V_liver.
    C_boundary_mol_m3:
        Flow-weighted arterial-plus-portal concentration for the single
        COMSOL concentration boundary, converted to mol/m^3.
    """

    p = DEFAULT_PARAMS if params is None else params
    _validate_inputs(dose_g, duration_s, dt_s, p)
    atol = np.asarray(atol, dtype=float)
    if (
        not np.isfinite(rtol)
        or rtol <= 0.0
        or atol.ndim > 1
        or atol.size not in (1, 10)
        or np.any(~np.isfinite(atol))
        or np.any(atol <= 0.0)
    ):
        raise ValueError("PBPK solver tolerances must be finite and positive.")
    if np.isnan(max_step) or max_step <= 0.0:
        raise ValueError("PBPK max_step must be positive.")

    dose_mol = dose_g / p.MW_g_mol

    # State order:
    # 0 gut lumen, 1 gut, 2 liver, 3 venous blood, 4 lung,
    # 5 arterial blood, 6 kidney, 7 rest, 8 kidney tubules, 9 metabolized.
    y0 = np.zeros(10, dtype=float)
    y0[0] = dose_mol

    def rhs(_t_s: float, y: np.ndarray) -> np.ndarray:
        (
            A_gut_lumen,
            A_gut,
            A_liver,
            A_ven,
            A_lung,
            A_art,
            A_kidney,
            A_rest,
            _A_tubules,
            _A_metabolized,
        ) = y

        # Sluka Table 1 transfer rates [mol/s].
        r_gut_lumen_to_gut = p.k_gut_abs_s * A_gut_lumen
        r_gut_to_liver = p.Q_gut_L_s * A_gut / p.V_gut_L

        r_liver_to_ven = (
            (p.Q_liver_L_s + p.Q_gut_L_s)
            * A_liver
            * p.R_blood_to_plasma
            / (p.K_liver_to_plasma * p.F_up * p.V_liver_L)
        )
        r_liver_to_metabolized = (
            p.k_metab_s_inv
            * A_liver
            / (p.K_liver_to_plasma * p.F_up)
        )

        r_ven_to_lung = p.Q_cardiac_L_s * A_ven / p.V_ven_L
        r_lung_to_art = p.Q_cardiac_L_s * A_lung / p.V_lung_L

        r_art_to_gut = p.Q_gut_L_s * A_art / p.V_art_L
        r_art_to_liver = p.Q_liver_L_s * A_art / p.V_art_L
        r_art_to_kidney = p.Q_kidney_L_s * A_art / p.V_art_L
        r_art_to_rest = p.Q_rest_L_s * A_art / p.V_art_L

        r_kidney_to_ven = (
            p.Q_kidney_L_s
            * A_kidney
            * p.R_blood_to_plasma
            / (p.K_kidney_to_plasma * p.F_up * p.V_kidney_L)
        )
        r_kidney_to_tubules = (
            p.Q_gfr_L_s
            * A_kidney
            / (p.K_kidney_to_plasma * p.V_kidney_L)
        )

        r_rest_to_ven = (
            p.Q_rest_L_s
            * A_rest
            * p.R_blood_to_plasma
            / (p.K_rest_to_plasma * p.F_up * p.V_rest_L)
        )

        dA_gut_lumen = -r_gut_lumen_to_gut
        dA_gut = r_gut_lumen_to_gut + r_art_to_gut - r_gut_to_liver
        dA_liver = (
            r_gut_to_liver
            + r_art_to_liver
            - r_liver_to_ven
            - r_liver_to_metabolized
        )
        dA_ven = (
            r_liver_to_ven
            + r_kidney_to_ven
            + r_rest_to_ven
            - r_ven_to_lung
        )
        dA_lung = r_ven_to_lung - r_lung_to_art
        dA_art = (
            r_lung_to_art
            - r_art_to_gut
            - r_art_to_liver
            - r_art_to_kidney
            - r_art_to_rest
        )
        dA_kidney = (
            r_art_to_kidney
            - r_kidney_to_ven
            - r_kidney_to_tubules
        )
        dA_rest = r_art_to_rest - r_rest_to_ven
        dA_tubules = r_kidney_to_tubules
        dA_metabolized = r_liver_to_metabolized

        return np.array(
            [
                dA_gut_lumen,
                dA_gut,
                dA_liver,
                dA_ven,
                dA_lung,
                dA_art,
                dA_kidney,
                dA_rest,
                dA_tubules,
                dA_metabolized,
            ],
            dtype=float,
        )

    # Include t=0. This matches the previous schedule convention where the
    # first inlet value is zero before oral APAP has entered the gut tissue.
    t_eval_s = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s, dtype=float)
    if t_eval_s[-1] < duration_s:
        t_eval_s = np.append(t_eval_s, duration_s)

    solution = solve_ivp(
        rhs,
        t_span=(0.0, duration_s),
        y0=y0,
        t_eval=t_eval_s,
        method="LSODA",
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not solution.success:
        raise RuntimeError("APAP PBPK integration failed: " + solution.message)

    # Tiny negative values can occur from numerical integration near zero.
    # Raise on material negativity; otherwise clip round-off only.
    min_amount = float(np.min(solution.y))
    if min_amount < -1.0e-9:
        raise RuntimeError(
            "APAP PBPK produced a materially negative compartment amount: "
            f"{min_amount:.6e} mol."
        )
    amounts = np.where(solution.y < 0.0, 0.0, solution.y)

    (
        A_gut_lumen,
        A_gut,
        A_liver,
        A_ven,
        A_lung,
        A_art,
        A_kidney,
        A_rest,
        A_tubules,
        A_metabolized,
    ) = amounts

    # Concentrations used by the published Sluka transfer equations.
    C_art_mol_L = A_art / p.V_art_L
    C_ven_mol_L = A_ven / p.V_ven_L
    C_lung_mol_L = A_lung / p.V_lung_L
    C_pv_mol_L = A_gut / p.V_gut_L
    C_liver_amount_mol_L = A_liver / p.V_liver_L
    C_kidney_amount_mol_L = A_kidney / p.V_kidney_L
    C_rest_amount_mol_L = A_rest / p.V_rest_L

    # Ghosh Equation 1, using Sluka's QLiver as the arterial liver inflow and
    # QGut as the portal inflow. Units: mol/(L*s).
    I_liver_mol_L_s = (
        p.Q_liver_L_s * C_art_mol_L
        + p.Q_gut_L_s * C_pv_mol_L
    ) / p.V_liver_L

    # COMSOL requires a concentration boundary, not concentration/time.
    # This flow-weighted value preserves the same arterial + portal molar
    # delivery in one effective inlet concentration. This conversion is not
    # specified by Ghosh; it is the explicit interface adaptation for COMSOL.
    Q_liver_total_L_s = p.Q_liver_L_s + p.Q_gut_L_s
    C_boundary_mol_L = (
        p.Q_liver_L_s * C_art_mol_L
        + p.Q_gut_L_s * C_pv_mol_L
    ) / Q_liver_total_L_s
    C_boundary_mol_m3 = 1000.0 * C_boundary_mol_L

    total_amount_mol = np.sum(amounts, axis=0)
    mass_balance_error_mol = total_amount_mol - dose_mol

    return {
        "time_s": solution.t,
        # Compartment amounts.
        "A_gut_lumen_mol": A_gut_lumen,
        "A_gut_mol": A_gut,
        "A_liver_mol": A_liver,
        "A_ven_mol": A_ven,
        "A_lung_mol": A_lung,
        "A_art_mol": A_art,
        "A_kidney_mol": A_kidney,
        "A_rest_mol": A_rest,
        "A_tubules_mol": A_tubules,
        "A_metabolized_mol": A_metabolized,
        # Concentrations.
        "C_art_mol_L": C_art_mol_L,
        "C_ven_mol_L": C_ven_mol_L,
        "C_lung_mol_L": C_lung_mol_L,
        "C_pv_mol_L": C_pv_mol_L,
        "C_liver_mol_L": C_liver_amount_mol_L,
        "C_kidney_mol_L": C_kidney_amount_mol_L,
        "C_rest_mol_L": C_rest_amount_mol_L,
        # Ghosh and COMSOL coupling outputs.
        "I_liver_mol_L_s": I_liver_mol_L_s,
        "I_liver_mol_L_h": 3600.0 * I_liver_mol_L_s,
        "C_boundary_mol_L": C_boundary_mol_L,
        "C_boundary_mol_m3": C_boundary_mol_m3,
        "C_pv_mol_m3": 1000.0 * C_pv_mol_L,
        # Diagnostics.
        "total_amount_mol": total_amount_mol,
        "mass_balance_error_mol": mass_balance_error_mol,
    }


def write_drug_inlet_file(
    output_path,
    time_s,
    c_pv_mol_m3,
):
    output_path = Path(output_path)

    time_s = np.asarray(time_s, dtype=float)
    concentration = np.asarray(c_pv_mol_m3, dtype=float)

    if time_s.shape != concentration.shape:
        raise ValueError("time_s and c_pv_mol_m3 must have the same shape.")

    if np.any(np.diff(time_s) <= 0):
        raise ValueError("PBPK time values must be strictly increasing.")

    if not np.all(np.isfinite(concentration)):
        raise ValueError("The PBPK schedule contains non-finite concentrations.")

    # Protect COMSOL from very small negative solver artifacts.
    concentration = np.maximum(concentration, 0.0)

    table = np.column_stack((time_s, concentration))

    np.savetxt(
        output_path,
        table,
        delimiter=",",
        fmt=["%.8f", "%.12e"],
    )

    return output_path