import unittest
from unittest.mock import patch

import numpy as np

from stresscf import installFirstIterationDamping


class FakeSCF:
    def __init__(self):
        self.diis_arguments = []

    def get_fock(
        self,
        h1e=None,
        s1e=None,
        vhf=None,
        dm=None,
        cycle=-1,
        diis=None,
        diis_start_cycle=None,
        level_shift_factor=None,
        damp_factor=None,
        fock_last=None,
    ):
        self.diis_arguments.append(diis)
        return np.asarray(h1e) + np.asarray(vhf)


class FirstIterationDampingTest(unittest.TestCase):
    def test_damping_precedes_and_does_not_pollute_diis_history(self):
        mf = FakeSCF()
        state = installFirstIterationDamping(mf)
        accelerator = object()
        dm0 = np.eye(2)
        dm1 = 2.0 * dm0
        fock0 = np.eye(2)
        raw_fock1 = 3.0 * np.eye(2)
        damped_fock = 1.5 * np.eye(2)

        with patch(
            "stresscf.ODAInterpolate",
            return_value=(0.25, None, damped_fock),
        ) as interpolate:
            first = mf.get_fock(
                h1e=fock0,
                vhf=np.zeros((2, 2)),
                dm=dm0,
                cycle=0,
                diis=accelerator,
            )
            second = mf.get_fock(
                h1e=raw_fock1,
                vhf=np.zeros((2, 2)),
                dm=dm1,
                cycle=1,
                diis=accelerator,
            )
            third = mf.get_fock(
                h1e=raw_fock1,
                vhf=np.zeros((2, 2)),
                dm=dm1,
                cycle=2,
                diis=accelerator,
            )

        np.testing.assert_allclose(first, fock0)
        np.testing.assert_allclose(second, damped_fock)
        np.testing.assert_allclose(third, raw_fock1)
        self.assertEqual(mf.diis_arguments, [None, None, accelerator])
        interpolate.assert_called_once()
        self.assertTrue(state["complete"])
        self.assertAlmostEqual(state["damping_factor"], 0.75)


if __name__ == "__main__":
    unittest.main()
