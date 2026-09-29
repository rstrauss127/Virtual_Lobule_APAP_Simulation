"""Ghosh-aligned MALD state-variables for APAP metabolism and hepatocyte injury:

state[0] = P_e : extracellular / lobular APAP pool
state[1] = A  : intracellular APAP
state[2] = S  : PAPS
state[3] = N  : NAPQI
state[4] = G  : GSH
state[5] = C  : NAPQI-cysteine / adduct burden
state[6] = H  : healthy hepatocyte fraction / count
state[7] = Z  : damaged hepatocyte fraction / count
state[8] = L  : lysed / necrosed hepatocyte fraction / count
state[9] = R  : regenerated hepatocyte fraction / count
"""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

SECOND = 8.64e4 # seconds in a day

DEFAULT_PARAMS = {
    "k_u": 8.33e-5,          # 1/s

    "k_S": 2.62e9,           # cell/mol/s
    "k_G": 3.46e-5,          # 1/s

    "k_CYP1A2": 2.60e-6,     # 1/s
    "k_CYP2E1": 2.80e-6,     # 1/s
    "k_CYP3A4": 5.60e-6,     # 1/s

    "k_N": 3.65e-7,          # 1/s

    "b_S": 3.07e-19,         # mol/cell/s
    "d_S": 2.31e-5,          # 1/s

    "k_GSH": 1.85e13,        # cell/mol/s
    "k_PSH": 1.27e-3,        # 1/s

    "b_G": 1.59e-19,         # mol/cell/s
    "d_G": 2.31e-5,          # 1/s

    "d_C": 3.00e-5,          # 1/s

    "eta": 6.02e8,           # cell/mol/s
    "delta_Z": 5.79e-5,      # 1/s
    "r": 1.16e-5,            # 1/s
}

CURRENT_PARAMS = DEFAULT_PARAMS.copy()


def _coerce_params(params=None):
    if params is None:
        params = {}
    p = DEFAULT_PARAMS.copy()
    p.update(params)
    return p


def mald(t, y):
    params = CURRENT_PARAMS
    y = np.maximum(np.asarray(y, dtype=float), 0.0)

    k_u = params["k_u"]
    k_S = params["k_S"]
    k_G = params["k_G"]
    k_CYP = params["k_CYP_total"]
    k_GSH = params["k_GSH"]
    k_PSH = params["k_PSH"]
    k_N = params["k_N"]
    b_S = params["b_S"]
    d_S = params["d_S"]
    b_G = params["b_G"]
    d_G = params["d_G"]
    d_C = params["d_C"]
    eta = params["eta"]
    delta_Z = params["delta_Z"]
    r = params["r"]

    P_e = y[0]
    A = y[1]
    S = y[2]
    N = y[3]
    G = y[4]
    C = y[5]
    H = y[6]
    Z = y[7]
    L = y[8]
    R = y[9]

    dP_e = 0 #-k_u * P_e
    dA = k_u * P_e - (k_S * S * A) - (k_G * A) - (k_CYP * A) + k_N * N
    dS = b_S - (k_S * S * A) - d_S * S
    dN = (k_CYP * A) - (k_GSH * N * G) - (k_PSH * N) - (k_N * N)
    dG = b_G - (k_GSH * N * G) - d_G * G
    dC = (k_PSH * N) - d_C * C

    # Hepatocyte dynamics: injury is driven by adduct burden and regeneration
    # follows a simple logistic-like recovery term.
    dH = r * H * (1.0 - (H + Z)) - eta * C * H
    dZ = eta * C * H - delta_Z * Z
    dL = delta_Z * Z - r * H * (1.0 - (H + Z))
    dR = r * H * (1.0 - (H + Z))

    return [dP_e, dA, dS, dN, dG, dC, dH, dZ, dL, dR]


def RunMALD(
    x,
    t_max_s,
    gsh_production_mol_s,
    gsh_binding_per_mol_s,
    regeneration_s_inv,
    method="Radau",
):
    """Run the Ghosh-aligned model
    """
    global CURRENT_PARAMS

    params = DEFAULT_PARAMS.copy()
    params["b_G"] = float(gsh_production_mol_s)
    params["k_GSH"] = float(gsh_binding_per_mol_s)
    params["r"] = float(regeneration_s_inv)
    CURRENT_PARAMS = params

    x = np.maximum(np.asarray(x, dtype=float), 0.0)

    sol = solve_ivp(mald, [0.0, t_max_s], x, method=method, rtol=1e-6, atol=1e-9)
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol


def TestMALD():
    quantity = 20.0 / 151.0 / 160000000000.0
    days = 4

    x1 = [quantity, 0.0, 1.0e-6, 0.0, 1.0e-14, 0.0, 1.0, 0.0, 0.0, 0.0]
    x2 = [quantity, 0.0, 1.0e-6, 0.0, 8.0e-15, 0.0, 1.0, 0.0, 0.0, 0.0]
    x3 = [quantity, 0.0, 1.0e-6, 0.0, 6.0e-15, 0.0, 1.0, 0.0, 0.0, 0.0]
    x4 = [quantity, 0.0, 1.0e-6, 0.0, 5.0e-15, 0.0, 1.0, 0.0, 0.0, 0.0]

    solution1 = RunMALD(x1, 8.64e4 * days, 1.0, 1.0, 1.0)
    solution2 = RunMALD(x2, 8.64e4 * days, 1.0, 1.0, 0.8)
    solution3 = RunMALD(x3, 8.64e4 * days, 1.0, 1.0, 0.6)
    solution4 = RunMALD(x4, 8.64e4 * days, 1.0, 1.0, 0.5)

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    fig.suptitle("Ghosh-aligned APAP hepatotoxicity model")

    axs[0, 0].plot(solution1.t, solution1.y[6], label="case 1")
    axs[0, 0].plot(solution2.t, solution2.y[6], label="case 2")
    axs[0, 0].plot(solution3.t, solution3.y[6], label="case 3")
    axs[0, 0].plot(solution4.t, solution4.y[6], label="case 4")
    axs[0, 0].set_title("Healthy hepatocytes")

    axs[0, 1].plot(solution1.t, solution1.y[7], label="case 1")
    axs[0, 1].plot(solution2.t, solution2.y[7], label="case 2")
    axs[0, 1].plot(solution3.t, solution3.y[7], label="case 3")
    axs[0, 1].plot(solution4.t, solution4.y[7], label="case 4")
    axs[0, 1].set_title("Damaged hepatocytes")

    axs[1, 0].plot(solution1.t, solution1.y[8], label="case 1")
    axs[1, 0].plot(solution2.t, solution2.y[8], label="case 2")
    axs[1, 0].plot(solution3.t, solution3.y[8], label="case 3")
    axs[1, 0].plot(solution4.t, solution4.y[8], label="case 4")
    axs[1, 0].set_title("Lysed hepatocytes")

    axs[1, 1].plot(solution1.t, solution1.y[4], label="case 1")
    axs[1, 1].plot(solution2.t, solution2.y[4], label="case 2")
    axs[1, 1].plot(solution3.t, solution3.y[4], label="case 3")
    axs[1, 1].plot(solution4.t, solution4.y[4], label="case 4")
    axs[1, 1].set_title("GSH")

    for ax in axs.flat:
        ax.set(xlabel="Time (s)", ylabel="")
        ax.legend(loc="best")

    fig.tight_layout()
    return

# TestMALD()
