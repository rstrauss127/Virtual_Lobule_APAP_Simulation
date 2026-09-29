"""Numerical convergence checks for the MALD and APAP PBPK ODE systems."""

import unittest

import numpy as np

import MALD_model_Scaled_V5 as mald
from pbpk_apap import simulate_apap_pbpk


class ODEConvergenceTests(unittest.TestCase):
    def test_mald_solver_refines_to_tight_reference(self):
        initial_state = np.array(
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
        parameters = (1.475e-14 / 8.64e4, 1.3e18 / 8.64e4, 1.0 / 8.64e4)
        duration_s = 2.0 * 8.64e4

        coarse = mald.RunMALD(
            initial_state,
            duration_s,
            *parameters,
            rtol=1.0e-4,
            atol=mald.MALD_ATOL * 100.0,
            max_step=3600.0,
        )
        default = mald.RunMALD(initial_state, duration_s, *parameters)
        reference = mald.RunMALD(
            initial_state,
            duration_s,
            *parameters,
            rtol=1.0e-10,
            atol=mald.MALD_ATOL * 1.0e-4,
            max_step=300.0,
        )

        state_scales = np.maximum(np.max(np.abs(reference.y), axis=1), mald.MALD_ATOL)
        coarse_error = float(np.max(np.abs(coarse.y[:, -1] - reference.y[:, -1]) / state_scales))
        default_error = float(np.max(np.abs(default.y[:, -1] - reference.y[:, -1]) / state_scales))
        self.assertLess(default_error, coarse_error)
        self.assertLess(default_error, 1.0e-5)

    def test_mald_forced_history_refines_to_tight_reference(self):
        initial_state = np.array(
            [
                0.0,
                0.0,
                0.8e-14,
                1.0,
                0.0,
                0.0,
                0.0,
                12.0,
                9.0,
                1.0,
            ]
        )
        duration_s = 86400.0
        output_times = np.arange(0.0, duration_s + 1200.0, 1200.0)
        forcing_times = np.arange(0.0, duration_s + 21600.0, 21600.0)
        forcing_values = np.array([0.0, 0.1, 0.5, 0.8, 0.4])
        parameters = (1.475e-14 / 8.64e4, 1.3e18 / 8.64e4, 1.0 / 8.64e4)
        common = (
            initial_state,
            output_times,
            forcing_times,
            forcing_values,
            *parameters,
            8.33e-5,
            3.4e-15,
        )

        coarse = mald.RunMALDHistory(
            *common,
            1800.0,
            rtol=1.0e-4,
            atol=mald.MALD_ATOL * 100.0,
        )
        default = mald.RunMALDHistory(*common, 600.0)
        reference = mald.RunMALDHistory(
            *common,
            150.0,
            rtol=1.0e-10,
            atol=mald.MALD_ATOL * 1.0e-4,
        )

        state_scales = np.maximum(np.max(np.abs(reference.y), axis=1), mald.MALD_ATOL)
        coarse_error = float(np.max(np.abs(coarse.y - reference.y) / state_scales[:, None]))
        default_error = float(np.max(np.abs(default.y - reference.y) / state_scales[:, None]))
        self.assertLess(default_error, coarse_error)
        self.assertLess(default_error, 1.0e-5)

    def test_pbpk_solver_refines_to_tight_reference(self):
        common = (1.0, 2.0 * 86400.0, 600.0)

        coarse = simulate_apap_pbpk(
            *common,
            rtol=1.0e-5,
            atol=1.0e-9,
            max_step=1800.0,
        )
        default = simulate_apap_pbpk(*common)
        reference = simulate_apap_pbpk(
            *common,
            rtol=1.0e-11,
            atol=1.0e-15,
            max_step=120.0,
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
        coarse_amounts = np.stack([coarse[key] for key in amount_keys])
        default_amounts = np.stack([default[key] for key in amount_keys])
        reference_amounts = np.stack([reference[key] for key in amount_keys])
        scales = np.maximum(np.max(np.abs(reference_amounts), axis=1), 1.0e-15)

        coarse_error = float(
            np.max(np.abs(coarse_amounts - reference_amounts) / scales[:, None])
        )
        default_error = float(
            np.max(np.abs(default_amounts - reference_amounts) / scales[:, None])
        )
        self.assertLess(default_error, coarse_error)
        self.assertLess(default_error, 1.0e-6)
        self.assertLess(
            float(np.max(np.abs(default["mass_balance_error_mol"]))),
            1.0e-8,
        )


if __name__ == "__main__":
    unittest.main()
