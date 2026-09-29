## MALD model from 
##


# %We consider y(0)=A, y(1)=N, y(2)=G, y(3)=H, y(4)=Z, y(5)=L, y(6)=R, with: 
#   State order
#   -----------
#   0 A      intracellular APAP amount [mol/cell]
#   1 N      intracellular NAPQI amount [mol/cell]
#   2 G      intracellular GSH amount [mol/cell]
#   3 H      healthy hepatocyte fraction
#   4 Z      damaged hepatocyte fraction
#   5 Lys    lysed/necrosed hepatocyte fraction
#   6 Regen  cumulative regeneration state
#   7 AST    serum AST [IU/L]
#   8 ALT    serum ALT [IU/L]
#   9 Clot   clotting-factor state

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Parameters
theta = 5.0                 # L             amount of blood

Hmax = 1.6e11               # cell         max hepatocytes
day = 8.64e4                # s


deltaZ = 5.0/8.64e4         # /s            damage hepatocyte lyse rate
eta = 5.12e13/8.64e4        # cell/mol/s    hepatocyte damage rate

alpha = 6.3/8.64e4          # /s            APAP clearing rate
deltaA = 0.33/8.64e4        # /s            APAP unconjugated clearing rate
p = 0.05                    # N/A           APAP fraction oxidized

deltaG = 2.0/8.64e4         # /s            GSH decay rate
deltaN = 1.0e-4/8.64e4      # /s

deltaS = 0.92/8.64e4        # /s            AST clearing rate
deltaL = 0.35/8.64e4        # /s            ALT clearing rate
betaS = 200000.0            # IU            total quantity of AST
betaL = 84800.0             # IU            total quantity of ALT
Smin = 12.0                 # IU/L          mini AST
Lmin = 9.0                  # IU/L          mini ALT
dz = 5.0/8.64e4             # N/A           Rate of ALT/AST generation

betaF = 5.0/8.64e4          # /s            clotting clearing rate
Fmin = 0.75                 # N/A           mini clothing factor


# Declare absolute and relative tolerances for the ODE solver. These are used to determine when the solver has converged to a solution at each time step.
MALD_ATOL = np.array( # Absolute Tolerance
    [
        1e-20,  # A: APAP [mol/cell]
        1e-20,  # N: NAPQI [mol/cell]
        1e-20,  # G: GSH [mol/cell]
        1e-10,  # H: healthy fraction
        1e-10,  # Z: damaged fraction
        1e-10,  # Lys: lysed fraction
        1e-10,  # Regen: regeneration state
        1e-6,   # AST [IU/L]
        1e-6,   # ALT [IU/L]
        1e-10,  # Clot: clotting-factor state
    ],
    dtype=float,
)

MALD_RTOL = 1e-6 # Relative Tolerance


# ODE system to be solved
def mald(t, y, kappa_local, gamma_local, regen_rate_local):
    
    ## APAP
    dy0= -(alpha*y[0]) - deltaA*y[0]
    
    ##############################################
    
    ## NAPQI
    dy1= (p*alpha*y[0]) - gamma_local*y[1]*y[2] - deltaN*y[1]
    ## GSH
    dy2= kappa_local - gamma_local*y[1]*y[2] - deltaG*y[2]
    
    ##############################################
    
    ## Hepatocytes status
    dy3= regen_rate_local*y[3]*(1-(y[3]+y[4])) - eta*y[1]*y[3]
    # dy3= - eta*y[1]*y[3]
    ## Hepatocyte damage
    dy4= eta*y[1]*y[3] - deltaZ*y[4]
    ## Hepatocyte Lyse
    dy5= deltaZ*y[4] - regen_rate_local*y[3]*(1-(y[3]+y[4]))
    ## Hepatocyte regeneration
    dy6= regen_rate_local*y[3]*(1-(y[3]+y[4]))
    
    ##############################################
    
    ## AST
    dy7= (dz*y[4]*betaS/theta) - (deltaS*(y[7]-Smin))
    ## ALT
    dy8= (dz*y[4]*betaL/theta) - (deltaL*(y[8]-Lmin))
    
    ##############################################
    
    ## Clotting
    dy9= betaF*((y[3]) - y[9])
    
    return [dy0,dy1,dy2,dy3,dy4,dy5,dy6,dy7,dy8,dy9]


def mald_with_apap_forcing(
    t,
    y,
    kappa_local,
    gamma_local,
    regen_rate_local,
    concentration_times_s,
    concentration_values_mol_m3,
    uptake_rate_s,
    hepatocyte_volume_m3,
):
    """MALD RHS with extracellular COMSOL APAP supplied continuously."""
    dydt = np.asarray(
        mald(t, y, kappa_local, gamma_local, regen_rate_local),
        dtype=float,
    )
    extracellular_apap = float(
        np.interp(
            t,
            concentration_times_s,
            concentration_values_mol_m3,
        )
    )
    dydt[0] += (
        uptake_rate_s
        * extracellular_apap
        * hepatocyte_volume_m3
    )
    return dydt

def _eigenvalues_2x2(a, b, c, d):
    """Vectorized eigenvalues of real 2x2 matrices."""
    trace = a + d
    discriminant = (a - d) ** 2 + 4.0 * b * c

    # Complex dtype safely handles negative discriminants.
    root = np.sqrt(discriminant.astype(np.complex128))

    lambda_1 = 0.5 * (trace + root)
    lambda_2 = 0.5 * (trace - root)

    return lambda_1, lambda_2


def CalculateStiffness(
    states,
    gamma_local,
    regen_rate_local,
    interval_s=1200.0,
    zero_tolerance=1.0e-12,
):
    """
    Calculate local Jacobian stiffness for one or more MALD states.

    Parameters
    ----------
    states
        Array shaped (n, 10), using MALD state order:
        A, N, G, H, Z, Lys, Regen, AST, ALT, Clot.
    gamma_local
        GSH-NAPQI binding rate for each state.
    regen_rate_local
        Regeneration rate for each state.
    interval_s
        Biological/output interval in seconds.
    zero_tolerance
        Eigenvalue threshold used to exclude neutral zero modes.

    Returns
    -------
    Dictionary containing one value per input state.
    """
    states = np.asarray(states, dtype=float)

    if states.ndim == 1:
        states = states.reshape(1, -1)

    if states.ndim != 2 or states.shape[1] != 10:
        raise ValueError("states must have shape (n, 10)")

    count = states.shape[0]

    gamma_local = np.broadcast_to(
        np.asarray(gamma_local, dtype=float),
        (count,),
    )

    regen_rate_local = np.broadcast_to(
        np.asarray(regen_rate_local, dtype=float),
        (count,),
    )

    N = states[:, 1]
    G = states[:, 2]
    H = states[:, 3]
    Z = states[:, 4]

    # APAP eigenvalue. APAP forcing is additive and therefore does not
    # contribute to the Jacobian.
    lambda_apap = np.full(
        count,
        -(alpha + deltaA),
        dtype=np.complex128,
    )

    # NAPQI-GSH Jacobian block.
    ng_11 = -(gamma_local * G + deltaN)
    ng_12 = -(gamma_local * N)
    ng_21 = -(gamma_local * G)
    ng_22 = -(gamma_local * N + deltaG)

    lambda_ng_1, lambda_ng_2 = _eigenvalues_2x2(
        ng_11,
        ng_12,
        ng_21,
        ng_22,
    )

    # Healthy-damaged Jacobian block.
    hz_11 = (
        regen_rate_local * (1.0 - 2.0 * H - Z)
        - eta * N
    )
    hz_12 = -(regen_rate_local * H)
    hz_21 = eta * N
    hz_22 = np.full(count, -deltaZ)

    lambda_hz_1, lambda_hz_2 = _eigenvalues_2x2(
        hz_11,
        hz_12,
        hz_21,
        hz_22,
    )

    core_eigenvalues = np.column_stack(
        [
            lambda_apap,
            lambda_ng_1,
            lambda_ng_2,
            lambda_hz_1,
            lambda_hz_2,
        ]
    )

    # Lysed and cumulative regeneration produce two exact zero modes.
    full_eigenvalues = np.column_stack(
        [
            core_eigenvalues,
            np.zeros(count, dtype=np.complex128),
            np.zeros(count, dtype=np.complex128),
            np.full(count, -deltaS, dtype=np.complex128),
            np.full(count, -deltaL, dtype=np.complex128),
            np.full(count, -betaF, dtype=np.complex128),
        ]
    )

    def calculate_metrics(eigenvalues):
        real_parts = eigenvalues.real
        active = np.abs(real_parts) > zero_tolerance

        decay_rates = np.where(
            active,
            np.abs(real_parts),
            np.nan,
        )

        fastest_rate = np.nanmax(decay_rates, axis=1)
        slowest_rate = np.nanmin(decay_rates, axis=1)

        stiffness_ratio = fastest_rate / slowest_rate
        spectral_radius = np.max(np.abs(eigenvalues), axis=1)
        unstable_modes = np.sum(
            real_parts > zero_tolerance,
            axis=1,
        )

        return {
            "stiffness_ratio": stiffness_ratio,
            "fastest_rate_s": fastest_rate,
            "slowest_rate_s": slowest_rate,
            "spectral_radius_s": spectral_radius,
            "interval_spectral_radius": interval_s * spectral_radius,
            "unstable_modes": unstable_modes,
        }

    full = calculate_metrics(full_eigenvalues)
    core = calculate_metrics(core_eigenvalues)

    return {
        "full_ratio": full["stiffness_ratio"],
        "core_ratio": core["stiffness_ratio"],
        "fastest_rate_s": full["fastest_rate_s"],
        "slowest_rate_s": full["slowest_rate_s"],
        "spectral_radius_s": full["spectral_radius_s"],
        "interval_spectral_radius": full["interval_spectral_radius"],
        "unstable_modes": full["unstable_modes"],
    }

# Call function to run model
def RunMALD(
    x,
    tMax,
    v1,
    v2,
    v3,
    *,
    rtol=MALD_RTOL,
    atol=MALD_ATOL,
    max_step=np.inf,
):
    initial_state = np.asarray(x, dtype=float)
    if initial_state.shape != (10,) or np.any(~np.isfinite(initial_state)):
        raise ValueError("MALD initial state must contain 10 finite values.")
    if not np.isfinite(tMax) or tMax <= 0.0:
        raise ValueError(f"MALD tMax must be finite and positive; received {tMax!r}.")
    if any(not np.isfinite(value) or value < 0.0 for value in (v1, v2, v3)):
        raise ValueError("MALD zonation rates must be finite and nonnegative.")
    atol = np.asarray(atol, dtype=float)
    if (
        not np.isfinite(rtol)
        or rtol <= 0.0
        or atol.ndim > 1
        or (atol.size not in (1, initial_state.size))
        or np.any(~np.isfinite(atol))
        or np.any(atol <= 0.0)
    ):
        raise ValueError("MALD solver tolerances must be finite and positive.")
    if np.isnan(max_step) or max_step <= 0.0:
        raise ValueError("MALD max_step must be positive.")

    # Pass zonated parameters explicitly. Mutating module-level globals for
    # every hepatocyte was unsafe and prevented parallel execution.
    solution = solve_ivp(
        lambda t, y: mald(t, y, v1, v2, v3),
        [0, tMax],
        initial_state,
        method='Radau',
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    # Check if the solver was successful
    if not solution.success:
        raise RuntimeError(
            "Radau MALD integration failed: "
            f"{solution.message}"
        )
        
    return solution


def RunMALDHistory(
    x,
    output_times_s,
    concentration_times_s,
    concentration_values_mol_m3,
    v1,
    v2,
    v3,
    uptake_rate_s,
    hepatocyte_volume_m3,
    max_step_s,
    *,
    rtol=MALD_RTOL,
    atol=MALD_ATOL,
):
    """Integrate one node across its complete remaining COMSOL history."""
    initial_state = np.asarray(x, dtype=float)
    output_times_s = np.asarray(output_times_s, dtype=float)
    concentration_times_s = np.asarray(concentration_times_s, dtype=float)
    concentration_values_mol_m3 = np.asarray(
        concentration_values_mol_m3,
        dtype=float,
    )

    if initial_state.shape != (10,) or np.any(~np.isfinite(initial_state)):
        raise ValueError("MALD initial state must contain 10 finite values.")
    if output_times_s.ndim != 1 or len(output_times_s) == 0:
        raise ValueError("output_times_s must be a non-empty 1D array.")
    if np.any(~np.isfinite(output_times_s)) or np.any(np.diff(output_times_s) <= 0.0):
        raise ValueError("output_times_s must be finite and strictly increasing.")
    if concentration_times_s.ndim != 1 or concentration_values_mol_m3.ndim != 1:
        raise ValueError("APAP forcing times and values must be 1D arrays.")
    if len(concentration_times_s) != len(concentration_values_mol_m3):
        raise ValueError("APAP forcing times and values must have equal length.")
    if concentration_times_s[0] != 0.0:
        raise ValueError("APAP forcing history must begin at relative time 0 s.")
    if np.any(np.diff(concentration_times_s) <= 0.0):
        raise ValueError("APAP forcing times must be strictly increasing.")
    if concentration_times_s[-1] < output_times_s[-1]:
        raise ValueError("APAP forcing history does not cover the MALD output times.")
    if np.any(~np.isfinite(concentration_values_mol_m3)) or np.any(
        concentration_values_mol_m3 < 0.0
    ):
        raise ValueError("APAP concentrations must be finite and nonnegative.")
    if any(not np.isfinite(value) or value < 0.0 for value in (v1, v2, v3)):
        raise ValueError("MALD zonation rates must be finite and nonnegative.")
    if not np.isfinite(uptake_rate_s) or uptake_rate_s < 0.0:
        raise ValueError("uptake_rate_s must be finite and nonnegative.")
    if not np.isfinite(hepatocyte_volume_m3) or hepatocyte_volume_m3 <= 0.0:
        raise ValueError("hepatocyte_volume_m3 must be finite and positive.")
    if not np.isfinite(max_step_s) or max_step_s <= 0.0:
        raise ValueError("max_step_s must be finite and positive.")
    atol = np.asarray(atol, dtype=float)
    if (
        not np.isfinite(rtol)
        or rtol <= 0.0
        or atol.ndim > 1
        or (atol.size not in (1, initial_state.size))
        or np.any(~np.isfinite(atol))
        or np.any(atol <= 0.0)
    ):
        raise ValueError("MALD solver tolerances must be finite and positive.")

    solution = solve_ivp(
        lambda t, y: mald_with_apap_forcing(
            t,
            y,
            v1,
            v2,
            v3,
            concentration_times_s,
            concentration_values_mol_m3,
            uptake_rate_s,
            hepatocyte_volume_m3,
        ),
        [0.0, float(output_times_s[-1])],
        initial_state,
        method="Radau",
        t_eval=output_times_s,
        max_step=float(max_step_s),
        rtol=rtol,
        atol=atol,
    )

    if not solution.success:
        raise RuntimeError(
            "Full-history Radau MALD integration failed: "
            f"{solution.message}"
        )

    return solution

# Call function to test model
def TestMALD():
    quantity = 20/151/160000000000
    days = 4
    # Plot results
    # x = [6.7e-13, 0, 0.88311e-14, 1, 0, 0, 1.18420e+04 ,6.731e+03, 0]
    x1 = [quantity, 0, 1.0e-14, 1, 0, 0, 0, 0 ,0, 0]
    x2 = [quantity, 0, 0.8e-14, 1, 0, 0, 0, 0 ,0, 0]
    x3 = [quantity, 0, 0.6e-14, 1, 0, 0, 0, 0 ,0, 0]
    x4 = [quantity, 0, 0.5e-14, 1, 0, 0, 0, 0 ,0, 0]
    solution1 = RunMALD(x1, 8.64e4*days, 1.575e-14/8.64e4, 1.10e18/8.64e4, 1.5/8.64e4)
    solution2 = RunMALD(x2, 8.64e4*days, 1.475e-14/8.64e4, 1.3e18/8.64e4, 1.0/8.64e4)
    solution3 = RunMALD(x3, 8.64e4*days, 1.275e-14/8.64e4, 1.6e18/8.64e4, 1.0/8.64e4)
    solution4 = RunMALD(x4, 8.64e4*days, 1.175e-14/8.64e4, 1.6e18/8.64e4, 0.5/8.64e4)

    
    variables1=solution1.y
    variables2=solution2.y
    variables3=solution3.y
    variables4=solution4.y
    temps1=solution1.t
    temps2=solution2.t
    temps3=solution3.t
    temps4=solution4.t        
    
    
    fig, axs = plt.subplots(2, 2,figsize=(20, 10))
    fig.suptitle('Model of Acetaminophen-induced Liver Disease')
    v=3
    axs[0, 0].plot(temps1,variables1[v], temps2, variables2[v], temps3, variables3[v], temps4, variables4[v])
    axs[0, 0].set_title('health')
    v=6
    axs[0, 1].plot(temps1,variables1[v], temps2, variables2[v], temps3, variables3[v], temps4, variables4[v])
    axs[0, 1].set_title('Regen')
    v=4
    axs[1, 0].plot(temps1,variables1[v], temps2, variables2[v], temps3, variables3[v], temps4, variables4[v])
    axs[1, 0].set_title('Damaged')
    v=5
    axs[1, 1].plot(temps1,variables1[v], temps2, variables2[v], temps3, variables3[v], temps4, variables4[v])
    axs[1, 1].set_title('Lysed')
    for ax in axs.flat:
        ax.set(xlabel='Time (s)', ylabel='')

    fig.tight_layout()
    return

# TestMALD()
