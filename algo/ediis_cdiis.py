"""Gaussian-style EDIIS + CDIIS convergence acceleration.

This implements the default hybrid described by Garza and Scuseria,
J. Chem. Phys. 137, 054110 (2012).  A scalar measure of the current
orthogonal-basis CDIIS error selects EDIIS, CDIIS, or a linear blend.
"""

import numpy as np

from pyscf.scf.addons import canonical_orth_
from pyscf.scf.diis import CDIIS, get_err_vec

from .diis import DIISExtrapolate
from .ediis import EDIISExtrapolate


class EDIISCDIIS(CDIIS):
    """EDIIS + CDIIS with the Gaussian default switching rules.

    Parameters
    ----------
    space
        Maximum number of Fock, density, energy, and error vectors retained.
        The paper's default, denoted EDIIS+CDIIS(20), is 20.
    
    ediis_error
        Use pure EDIIS when ``errMax`` is greater than this value.
    
    cdiis_error
        Use pure CDIIS when ``errMax`` is less than this value.
    
    regression_factor
        Fall back to pure EDIIS when the newest ``errMax`` is more than this
        factor times the smallest error in the retained history.  ``None``
        disables this fallback.  The paper used 1.1, but the fallback is not
        enabled by default because it ejects the PySCF UF4 trajectory from the
        converged basin after EDIIS+CDIIS has already reached it.

    orthogonalize_error
        Generate a canonical orthogonalizer from the overlap matrix when
        ``Corth`` is not supplied.  The switching thresholds in the paper are
        defined for the commutator in an orthogonal basis.
    
    error_measure
        Scalar used with the Gaussian switching thresholds.  ``"rms"`` is
        the stable PySCF translation because AO matrix dimensions and density
        normalization make the largest raw matrix element non-portable.

    Notes
    -----
    Between the two error thresholds, the returned Fock matrix is

    ``10 * errMax * F_EDIIS + (1 - 10 * errMax) * F_CDIIS``.

    The residual is obtained with :func:`pyscf.scf.diis.get_err_vec`, so the
    class follows PySCF's handling of restricted, unrestricted,
    orthogonalized, complex, and symmetry-adapted calculations.
    """

    def __init__(
        self,
        mf=None,
        filename=None,
        Corth=None,
        space=20,
        ediis_error=0.2,
        cdiis_error=1e-4,
        regression_factor=None,
        orthogonalize_error=True,
        error_measure="rms",
        linear_dep_threshold=1e-7,
        **kwargs,
    ):
        super().__init__(mf=mf, filename=filename, Corth=Corth)

        if space < 1:
            raise ValueError("space must be at least 1")
        if not 0 <= cdiis_error < ediis_error:
            raise ValueError("require 0 <= cdiis_error < ediis_error")
        if regression_factor is not None and regression_factor <= 1:
            raise ValueError("regression_factor must be greater than 1")

        self.space = space
        self.ediis_error = ediis_error
        self.cdiis_error = cdiis_error
        self.regression_factor = regression_factor
        self.orthogonalize_error = orthogonalize_error

        if error_measure not in ("rms", "max"):
            raise ValueError("error_measure must be 'rms' or 'max'")

        self.error_measure = error_measure
        self.linear_dep_threshold = linear_dep_threshold

        self.fock_history = []
        self.density_history = []
        self.energy_history = []
        self.residuals_history = []
        self.error_max_history = []

        # Exposed for diagnostics and benchmark logging.
        self.errmax = np.inf
        self.ediis_weight = None
        self.iteration = 0

    def ensureOrthogonal(self, s):
        if self.Corth is not None or not self.orthogonalize_error:
            return

        overlap = np.asarray(s)

        if overlap.ndim == 2:
            self.Corth = canonical_orth_(overlap, thr=self.linear_dep_threshold)
        elif overlap.ndim == 3:
            self.Corth = np.asarray([canonical_orth_(block, thr=self.linear_dep_threshold) for block in overlap])
        else:
            raise ValueError(f"unsupported overlap shape {overlap.shape}")

    def update(self, s, dm, fock, mf, h1e, vhf, *args, **kwargs):
        self.ensureOrthogonal(s)
        residual = get_err_vec(s, dm, fock, self.Corth)

        if self.error_measure == "rms":
            errmax = float(np.sqrt(np.mean(np.abs(residual) ** 2)))
        else:
            errmax = float(np.max(np.abs(residual)))

        energy = float(np.real(mf.energy_elec(dm, h1e, vhf)[0]))

        self.fock_history.append(np.array(fock, copy=True))
        self.density_history.append(np.array(dm, copy=True))
        self.energy_history.append(energy)
        self.residuals_history.append(np.array(residual, copy=True))
        self.error_max_history.append(errmax)

        self.fock_history = self.fock_history[-self.space :]
        self.density_history = self.density_history[-self.space :]
        self.energy_history = self.energy_history[-self.space :]
        self.residuals_history = self.residuals_history[-self.space :]
        self.error_max_history = self.error_max_history[-self.space :]

        self.errmax = errmax
        self.ediis_weight = None

        if len(self.fock_history) < 2:
            return fock

        error_regressed = (self.regression_factor is not None and errmax > self.regression_factor * min(self.error_max_history))

        # EDIIS Mode
        if errmax > self.ediis_error or error_regressed:
            fock_ediis = EDIISExtrapolate(self.energy_history, self.fock_history, self.density_history)
            return fock_ediis

        # CDIIS Mode
        if errmax < self.cdiis_error:
            fock_cdiis = DIISExtrapolate(self.fock_history, self.residuals_history)
            return fock_cdiis

        # Use a mix of EDIIS & CDIIS
        fock_ediis, coefficients_ediis = EDIISExtrapolate(self.energy_history, self.fock_history, self.density_history, return_c=True)
        fock_cdiis, coefficients_cdiis = DIISExtrapolate(self.fock_history, self.residuals_history, return_c=True)

        # With the paper's default 1e-1 upper threshold this is exactly
        # 10 * errMax.  Expressing it relative to the threshold keeps a
        # user-supplied threshold well behaved too.

        ediis_weight = errmax / self.ediis_error
        self.ediis_weight = ediis_weight

        return ediis_weight * fock_ediis + (1.0 - ediis_weight) * fock_cdiis


# Readable alias for callers that prefer the method names to be separated.
EDIIS_CDIIS = EDIISCDIIS
