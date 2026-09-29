"""Generate tolerance-refinement and time-resolved ODE convergence plots."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import MALD_model_Scaled_V5 as mald
from pbpk_apap import simulate_apap_pbpk


OUTPUT_DIR = Path(__file__).resolve().parent / "plots"
RELATIVE_TOLERANCES = np.array([1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8])
FORCED_TOLERANCES_TO_PLOT = (1.0e-3, 1.0e-5, 1.0e-6, 1.0e-8)


def _trajectory_error(solution, reference, state_atol):
    state_scales = np.maximum(
        np.max(np.abs(reference), axis=1),
        np.asarray(state_atol, dtype=float),
    )
    return np.max(np.abs(solution - reference) / state_scales[:, None], axis=0)


def _save_refinement_plot(refinement_errors):
    fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    colors = {
        "MALD (unforced)": "#2878B5",
        "MALD (APAP-forced)": "#E87500",
        "PBPK": "#25855A",
    }
    for label, errors in refinement_errors.items():
        ax.loglog(
            RELATIVE_TOLERANCES,
            errors,
            marker="o",
            linewidth=2,
            markersize=5,
            label=label,
            color=colors[label],
        )

    ax.invert_xaxis()
    ax.set_xlabel("Solver relative tolerance (rtol; tighter to the right)")
    ax.set_ylabel("Maximum normalized terminal-state error")
    ax.set_title("ODE convergence under solver-tolerance refinement")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.17)
    ax.legend(frameon=False)
    ax.text(
        0.02,
        0.02,
        "Each solver's atol was scaled with rtol; max_step was held fixed per model.\n"
        "Errors are relative to a tighter-tolerance reference integration.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
    )
    output = OUTPUT_DIR / "ode_convergence_refinement.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return output


def _save_forced_error_plot(times_hours, reference, trajectories, state_atol):
    fig, ax = plt.subplots(figsize=(10, 6.2), constrained_layout=True)
    colors = {
        1.0e-3: "#C43C39",
        1.0e-5: "#E87500",
        1.0e-6: "#2878B5",
        1.0e-8: "#25855A",
    }
    for rtol in FORCED_TOLERANCES_TO_PLOT:
        errors = _trajectory_error(
            trajectories[rtol],
            reference,
            state_atol,
        )
        ax.semilogy(
            times_hours,
            np.where(errors > 0.0, errors, np.nan),
            marker="o",
            markersize=2.5,
            linewidth=1.8,
            label=f"rtol={rtol:.0e}",
            color=colors[rtol],
        )

    ax.set_xlabel("Simulation time (hours)")
    ax.set_ylabel("Maximum normalized state error")
    ax.set_title("APAP-forced MALD: error versus tight-reference solution")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.17)
    ax.legend(title="Solver tolerance", frameon=False)
    ax.text(
        0.02,
        0.02,
        "At each output time, error is the largest state-wise absolute difference,\n"
        "normalized by that state's maximum magnitude in the reference trajectory.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#444444",
    )
    output = OUTPUT_DIR / "ode_convergence_forced_mald_error_over_time.png"
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)
    return output


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mald_initial = np.array(
        [
            20.0 / 151.0 / 1.6e11,
            0.0,
            0.8e-14,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    parameters = (
        1.475e-14 / 8.64e4,
        1.3e18 / 8.64e4,
        1.0 / 8.64e4,
    )
    duration_s = 86400.0

    unforced_reference = mald.RunMALD(
        mald_initial,
        duration_s,
        *parameters,
        rtol=1.0e-11,
        atol=mald.MALD_ATOL * 1.0e-5,
        max_step=120.0,
    )
    unforced_results = {}
    for rtol in RELATIVE_TOLERANCES:
        unforced_results[rtol] = mald.RunMALD(
            mald_initial,
            duration_s,
            *parameters,
            rtol=rtol,
            atol=mald.MALD_ATOL * (rtol / mald.MALD_RTOL),
            max_step=3600.0,
        )
    unforced_scale = np.maximum(
        np.max(np.abs(unforced_reference.y), axis=1),
        mald.MALD_ATOL,
    )
    unforced_errors = np.array(
        [
            np.max(
                np.abs(result.y[:, -1] - unforced_reference.y[:, -1])
                / unforced_scale
            )
            for result in unforced_results.values()
        ]
    )

    initial_state = np.array(
        [0.0, 0.0, 0.8e-14, 1.0, 0.0, 0.0, 0.0, 12.0, 9.0, 1.0]
    )
    output_times = np.arange(0.0, duration_s + 1200.0, 1200.0)
    forcing_times = np.arange(0.0, duration_s + 21600.0, 21600.0)
    forcing_values = np.array([0.0, 0.1, 0.5, 0.8, 0.4])
    forced_args = (
        initial_state,
        output_times,
        forcing_times,
        forcing_values,
        *parameters,
        8.33e-5,
        3.4e-15,
    )
    forced_reference = mald.RunMALDHistory(
        *forced_args,
        60.0,
        rtol=1.0e-11,
        atol=mald.MALD_ATOL * 1.0e-5,
    )
    forced_results = {}
    for rtol in RELATIVE_TOLERANCES:
        forced_results[rtol] = mald.RunMALDHistory(
            *forced_args,
            600.0,
            rtol=rtol,
            atol=mald.MALD_ATOL * (rtol / mald.MALD_RTOL),
        )
    forced_scale = np.maximum(
        np.max(np.abs(forced_reference.y), axis=1),
        mald.MALD_ATOL,
    )
    forced_errors = np.array(
        [
            np.max(
                np.abs(result.y[:, -1] - forced_reference.y[:, -1])
                / forced_scale
            )
            for result in forced_results.values()
        ]
    )

    pbpk_reference = simulate_apap_pbpk(
        1.0,
        duration_s,
        600.0,
        rtol=1.0e-12,
        atol=1.0e-16,
        max_step=60.0,
    )
    amount_keys = (
        "A_gut_lumen_mol",
        "A_gut_mol",
        "A_liver_mol",
        "A_ven_mol",
        "A_lung_mol",
        "A_art_mol",
        "A_kidney_mol",
        "A_rest_mol",
        "A_tubules_mol",
        "A_metabolized_mol",
    )
    pbpk_reference_states = np.stack(
        [pbpk_reference[key] for key in amount_keys]
    )
    pbpk_results = {}
    for rtol in RELATIVE_TOLERANCES:
        pbpk_results[rtol] = simulate_apap_pbpk(
            1.0,
            duration_s,
            600.0,
            rtol=rtol,
            atol=1.0e-12 * (rtol / 1.0e-8),
            max_step=1800.0,
        )
    pbpk_scale = np.maximum(
        np.max(np.abs(pbpk_reference_states), axis=1),
        1.0e-15,
    )
    pbpk_errors = np.array(
        [
            np.max(
                np.abs(
                    np.array([result[key][-1] for key in amount_keys])
                    - pbpk_reference_states[:, -1]
                )
                / pbpk_scale
            )
            for result in pbpk_results.values()
        ]
    )

    refinement_errors = {
        "MALD (unforced)": unforced_errors,
        "MALD (APAP-forced)": forced_errors,
        "PBPK": pbpk_errors,
    }
    outputs = [
        _save_refinement_plot(refinement_errors),
        _save_forced_error_plot(
            output_times / 3600.0,
            forced_reference.y,
            {rtol: result.y for rtol, result in forced_results.items()},
            mald.MALD_ATOL,
        ),
    ]

    for rtol, unforced_error, forced_error, pbpk_error in zip(
        RELATIVE_TOLERANCES,
        unforced_errors,
        forced_errors,
        pbpk_errors,
    ):
        print(
            f"rtol={rtol:.0e}: "
            f"unforced MALD={unforced_error:.6e}, "
            f"forced MALD={forced_error:.6e}, "
            f"PBPK={pbpk_error:.6e}"
        )
    for output in outputs:
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"Plot was not created: {output}")
        print(f"Created {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
