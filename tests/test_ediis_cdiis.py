import unittest
from unittest.mock import patch

import numpy as np

from algo.ediis_cdiis import EDIISCDIIS
from algo.ediis import EDIISExtrapolate
from pyscf.scf.diis import ediis_minimize


class FakeMF:
    def energy_elec(self, dm, h1e, vhf):
        return float(np.sum(dm)), 0.0


def fake_ediis(energies, focks, densities, return_c=False):
    result = np.full_like(focks[-1], 10.0, dtype=float)
    coefficients = np.zeros(len(focks))
    coefficients[-1] = 1.0
    return (result, coefficients) if return_c else result


def fake_cdiis(focks, residuals, return_c=False):
    result = np.full_like(focks[-1], 20.0, dtype=float)
    coefficients = np.zeros(len(focks))
    coefficients[0] = 1.0
    return (result, coefficients) if return_c else result


class EDIISCDIISTest(unittest.TestCase):
    def setUp(self):
        self.s = np.eye(2)
        self.dm = np.eye(2)
        self.fock = np.eye(2)
        self.mf = FakeMF()

    def run_updates(self, errors, space=20, regression_factor=None):
        accelerator = EDIISCDIIS(
            space=space,
            regression_factor=regression_factor,
        )
        residuals = [np.array([error, -error]) for error in errors]
        with (
            patch("algo.ediis_cdiis.get_err_vec", side_effect=residuals),
            patch("algo.ediis_cdiis.EDIISExtrapolate", side_effect=fake_ediis),
            patch("algo.ediis_cdiis.DIISExtrapolate", side_effect=fake_cdiis),
        ):
            result = None
            for _ in errors:
                result = accelerator.update(
                    self.s,
                    self.dm,
                    self.fock,
                    self.mf,
                    self.fock,
                    self.fock,
                )
        return accelerator, result

    def test_high_error_uses_ediis(self):
        accelerator, result = self.run_updates([0.2, 0.15])
        self.assertEqual(accelerator.mode, "EDIIS")
        np.testing.assert_allclose(result, 10.0)

    def test_low_error_uses_cdiis(self):
        accelerator, result = self.run_updates([8e-5, 5e-5])
        self.assertEqual(accelerator.mode, "CDIIS")
        np.testing.assert_allclose(result, 20.0)

    def test_intermediate_error_blends_focks(self):
        accelerator, result = self.run_updates([0.06, 0.05])
        self.assertEqual(accelerator.mode, "EDIIS+CDIIS")
        np.testing.assert_allclose(result, 15.0)

    def test_error_regression_falls_back_to_ediis(self):
        accelerator, result = self.run_updates(
            [0.01, 0.012],
            regression_factor=1.1,
        )
        self.assertEqual(accelerator.mode, "EDIIS")
        np.testing.assert_allclose(result, 10.0)

    def test_regression_fallback_is_disabled_by_default(self):
        accelerator, result = self.run_updates([0.01, 0.012])
        self.assertEqual(accelerator.mode, "EDIIS+CDIIS")
        np.testing.assert_allclose(result, 18.8)

    def test_history_is_limited_to_space(self):
        accelerator, _ = self.run_updates([0.2, 0.15, 0.12], space=2)
        self.assertEqual(len(accelerator.fock_history), 2)
        self.assertEqual(len(accelerator.density_history), 2)
        self.assertEqual(len(accelerator.energy_history), 2)
        self.assertEqual(len(accelerator.residuals_history), 2)
        self.assertEqual(len(accelerator.error_max_history), 2)

    def test_generates_orthogonalizer_from_overlap(self):
        overlap = np.array([[1.0, 0.2], [0.2, 1.0]])
        accelerator = EDIISCDIIS()

        accelerator.update(
            overlap,
            self.dm,
            self.fock,
            self.mf,
            self.fock,
            self.fock,
        )

        np.testing.assert_allclose(
            accelerator.Corth.T @ overlap @ accelerator.Corth,
            np.eye(2),
            atol=1e-12,
        )

    def test_ediis_matches_pyscf_reference_functional(self):
        rng = np.random.default_rng(7)
        densities = rng.standard_normal((4, 2, 2))
        focks = rng.standard_normal((4, 2, 2))
        densities = densities + densities.transpose(0, 2, 1)
        focks = focks + focks.transpose(0, 2, 1)
        energies = np.array([-3.0, -2.8, -2.7, -2.5])

        _, coefficients = EDIISExtrapolate(
            energies,
            focks,
            densities,
            return_c=True,
        )
        _, reference_coefficients = ediis_minimize(
            energies,
            densities,
            focks,
        )

        np.testing.assert_allclose(
            coefficients,
            reference_coefficients,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
