# Optimal Damping Algorithm of Cances & Le Bris (1999)

import numpy as np

from pyscf.scf.diis import CDIIS

def ODAInterpolate(F_old, F_new, dm_old, dm_new):
    """
    Interpolate between two Fock matrices
    as per the ODA algorithm.

    Returns l, dm_ODA, F_ODA
    """
    s = np.vdot(F_old, dm_new - dm_old)
    c = np.vdot(F_new - F_old, dm_new - dm_old)
    l = np.min([1, np.max([0, -s/c])])

    return l, (1.0 - l) * dm_old + l * dm_new, (1.0 - l) * F_old + l * F_new

class ODA(CDIIS):
    def __init__(
        self,
        mf=None,
        Corth=None,
        history=8,
        **kwargs
    ):
        super().__init__(mf=mf, Corth=Corth)
        self.F_ODA = None

    def update(self, s, dm, fock, mf, h1e, vhf, *args, **kwargs):
        if self.F_ODA is None:
            self.F_ODA = fock
            self.dm_ODA = dm
            return fock
        else:
            l, self.dm_ODA, self.F_ODA = ODAInterpolate(self.F_ODA, fock, 
                                                        self.dm_ODA, dm)
            return self.F_ODA
    