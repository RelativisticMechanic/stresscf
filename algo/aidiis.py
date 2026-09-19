import numpy as np
import os

from pyscf.scf.diis import CDIIS, get_err_vec

from scipy.optimize import brentq

import math
import numpy as np

from .diis import DIISExtrapolate

ALPHA_MIN = -1.0
ALPHA_MAX = 2.0
ALPHA_STEP = 0.05

def getAngle(v1, v2):
    cos_theta = np.vdot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    return theta * (180.0 / np.pi)

def getResNorm(residual):
    return np.sqrt(np.mean(residual ** 2))

def getResidual(f, p, s):
    return f @ p @ s - s @ p @ f
    
class AIDIIS(CDIIS):
    """
    Angle-Informed DIIS (AI-DIIS)
    """

    def __init__(
        self,
        mf=None,
        Corth=None,
        space=20,
        logging=False,
        candidate="CDIIS",
        **kwargs
    ):
        # PySCF CDIIS does not accept space/rollback in constructor
        super().__init__(mf=mf, Corth=Corth)
        self.space = space

        self.candidate = candidate

        self.history_fock = []
        self.history_residuals = []
        self.history_densities = []
        self.history_energies = []

        self.theta = 90

        self.cav = kwargs.get("custom_variable_access")
        self.iter = 0

        self.cav = kwargs["custom_variable_access"]
        self.file = kwargs.get("test_case_id", None)
        self.logging = logging
        self.diis_max = 0

        self.dm_diis, self.f_diis, = None, None

        if self.logging:
            os.makedirs(self.file, exist_ok=True)

        self.S_half, self.S_mhalf = None, None 

    def thetaFromAlpha(self, f_old : np.ndarray, dm_old : np.ndarray,
                        f_ext: np.ndarray, dm_ext : np.ndarray,
                        s : np.ndarray,
                        alpha : float):

        """
        Generate a theta for a particular alpha value.

        The theta is the angle between the Riemannian gradient
        (double commutator) and the proposed density step.
        """

        f_alpha = (1.0 - alpha) * f_old + alpha * f_ext
        dm_alpha = (1.0 - alpha) * dm_old + alpha * dm_ext
        density_step = dm_alpha - dm_old

        # Get the approximate residual
        res = getResidual(f_alpha, dm_alpha, s)

        # Get the double commutator i.e., the Riemannian gradient
        dc = getResidual(res, dm_alpha, s)

        # Get the angle between the double commutator and the density step
        theta = getAngle(-dc, density_step)

        return {
            "theta": theta, 
            "f": f_alpha, 
            "dm": dm_alpha
        }

    def optimizeAlpha(self, alpha, theta, f_old, dm_old, f_ext, dm_ext, s, target_theta=90):
        """
        Calculate alpha for a given value of theta.
        Returns the alpha with a positive slope.

        If no roots are found, it returns 1.0 (DIIS)
        If the slope at alpha is negative, it returns 1.0 (DIIS)
        """

        a = []
        t = []

        for ai, ti in zip(alpha, theta):
            if not (isinstance(ti, float) and math.isnan(ti)):
                a.append(ai)
                t.append(ti)

        a = np.asarray(a, dtype=float)
        t = np.asarray(t, dtype=float)

        if len(a) < 2:
            return 1.0

        # Sort by alpha: we are constructing theta(alpha)
        idx = np.argsort(a)
        alpha_sorted = a[idx]
        theta_sorted = t[idx]

        # Remove duplicate alpha values
        alpha_sorted, unique_idx = np.unique(alpha_sorted, return_index=True)
        theta_sorted = theta_sorted[unique_idx]

        if len(alpha_sorted) < 2:
            return 1.0

        # Function whose zeros correspond to theta = target_theta
        def f(x):
            data = self.thetaFromAlpha(f_old, dm_old, f_ext, dm_ext, s, x)
            return data["theta"] - target_theta

        roots = []
        slopes = []

        # Search across intervals
        for i in range(len(alpha_sorted) - 1):
            x1 = alpha_sorted[i]
            x2 = alpha_sorted[i + 1]

            y1 = f(x1)
            y2 = f(x2)

            # Exact endpoint
            if y1 == 0:
                roots.append(x1)

            # Crossing
            if y1 * y2 < 0:
                try:
                    root = brentq(f, x1, x2)
                    roots.append(root)
                    slopes.append((f(x2) - f(x1)) / (x2 - x1))
                except ValueError:
                    pass

        # Check final endpoint
        if f(alpha_sorted[-1]) == 0:
            roots.append(alpha_sorted[-1])

        if not roots:
            return 1.0

        slopes = np.asarray(slopes)
        roots = np.asarray(roots)

        # Keep only roots with positive slope
        try:
            positive_roots = roots[np.asarray(slopes) > 0]
        except:
            positive_roots = []

        if len(positive_roots) == 0:
            # No positive-slope root -> fall back to DIIS
            alpha_pred = 1.0
        else:
            # Largest alpha among positive-slope roots
            alpha_pred = float(np.max(positive_roots))
        
        if ALPHA_MIN < alpha_pred < ALPHA_MAX:
            return alpha_pred

        return 1.0

    def generateThetaAlphaCurve(self, mf, h1e, s, f_old, f_ext, dm_old, dm_ext):
        """
        Generate the theta-alpha curve between two endpoints (old & ext) given
        (F_old,P_old) & (F_ext,P_ext)
        """

        thetas = []
        energies_RH = []

        alphas = np.arange(ALPHA_MIN, ALPHA_MAX, ALPHA_STEP)

        for alpha in alphas:
            data = self.thetaFromAlpha(f_old, dm_old, f_ext, dm_ext, s, alpha)
            thetas.append(data["theta"])
            if self.logging:
                # Conduct a full RH iteration to get the
                # actual F and E corresponding to alpha.
                mo_energy, mo_coeff = mf.eig(data["f"], s)
                mo_occ = mf.get_occ(mo_energy, mo_coeff)
                dm_actual = mf.make_rdm1(mo_coeff, mo_occ)
                vhf_actual = mf.get_veff(dm=dm_actual)
                energy_RH = float(mf.energy_tot(dm=dm_actual, h1e=h1e, vhf=vhf_actual))
                energies_RH.append(energy_RH)
        
        return {
            "alpha": alphas, 
            "theta": thetas, 
            "energy": energies_RH
        }

    def update(self, s, dm, fock, mf, h1e, vhf, *args, **kwargs):  
        residual = getResidual(fock, dm, s)

        rnorm = np.sqrt(np.mean(residual ** 2))

        n_hist = len(self.history_fock)

        self.history_fock.append(fock)
        self.history_residuals.append(residual)
        self.history_densities.append(dm)
        self.history_energies.append(self.cav["e"]) 
        
        if len(self.history_fock) >= self.space:
            self.history_fock = self.history_fock[-self.space:]
            self.history_residuals = self.history_residuals[-self.space:]
            self.history_densities = self.history_densities[-self.space:]
            self.history_energies = self.history_energies[-self.space:]

        self.iter += 1

        if n_hist < 2:
            return fock

        # Last fock
        f_candidate = fock
        dm_candidate = dm

        self.f_diis, self.c_cdiis = DIISExtrapolate(self.history_fock, 
                        self.history_residuals, 
                        return_c=True)
        
        self.dm_diis = sum([c_i * dm_i for c_i, dm_i in zip(self.c_cdiis, self.history_densities)])

        if(self.iter >= self.diis_max):
            curve_data = self.generateThetaAlphaCurve(mf, 
                                                h1e, 
                                                s, 
                                                f_candidate, self.f_diis, 
                                                dm_candidate, self.dm_diis)
            
            alpha = self.optimizeAlpha(curve_data["alpha"], 
                                        curve_data["theta"], 
                                        f_candidate, dm_candidate, 
                                        self.f_diis, self.dm_diis, 
                                        s, target_theta=self.theta)
                

            e0 = self.history_energies[-1]

            dens = (1 - alpha) * dm_candidate + alpha * self.dm_diis
            result = (1 - alpha) * f_candidate + alpha * self.f_diis
            

            if self.logging:
                e_alpha = mf.energy_tot(dm=dens, h1e=h1e, vhf=result - h1e)
                print(f"SCF/RESIDUAL: {rnorm}, dE: {self.cav['de']}")
                print(f"--- STEP: {self.iter} ---")
                print(f"ENERGY0: {e0}")
                print(f"ALPHA: {alpha}")
                print(f"ENERGY ALPHA: {e_alpha}")

                if self.file is not None:
                    f = open(f"{self.file}/{self.iter}.json", "w")
                    json_data = {}

                    json_data["x:alpha"] = list(curve_data["alpha"])

                    json_data["y:e1"] = [e0]
                    json_data["y:e2"] = curve_data["energy"]

                    json_data["y:theta"] = curve_data["theta"]
                    
                    json_data["xline:alpha"] = alpha 

                    
                    import json
                    f.write(json.dumps(json_data, indent=4))
                    f.close()
        else:
            result = self.f_diis

        return result