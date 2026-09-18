import numpy as np

from pyscf.scf.diis import CDIIS

class PureRoothaanHall(CDIIS):
    def __init__(
        self,
        mf=None,
        Corth=None,
        space=8,
        **kwargs
    ):
        super().__init__(mf=mf, Corth=Corth)
        self.space = space

        self.residuals_history = []
        self.fock_history = []
        
    def update(self, s, dm, fock, mf, h1e, vhf, *args, **kwargs):
        self.fock_history.append(fock)
        self.residuals_history.append(fock @ dm @ s - s @ dm @ fock)

        if len(self.fock_history) > self.space:
            self.fock_history = self.fock_history[-self.space:]
            self.residuals_history = self.residuals_history[-self.space:]

        return fock
    