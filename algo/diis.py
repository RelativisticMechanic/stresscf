import numpy as np
from pyscf.scf.diis import CDIIS, get_err_vec

def DIISExtrapolate(xs, residuals, return_c=False, rcond=1e-12):
    n = len(xs)
    if n != len(residuals):
        raise ValueError("xs and residuals must have equal lengths")
    if n == 0:
        raise ValueError("DIIS requires at least one vector")

    B = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1):
            bij = np.vdot(residuals[i], residuals[j]).real
            B[i, j] = bij
            B[j, i] = bij

    # Make the singular-value cutoff insensitive to residual magnitude.
    scale = max(
        np.max(np.abs(np.diag(B))),
        np.finfo(np.float64).tiny,
    )
    B /= scale

    A = np.zeros((n + 1, n + 1), dtype=np.float64)
    A[:n, :n] = B
    A[:n, n] = -1.0
    A[n, :n] = -1.0

    rhs = np.zeros(n + 1, dtype=np.float64)
    rhs[n] = -1.0

    # Robust for redundant residuals.
    solution, *_ = np.linalg.lstsq(A, rhs, rcond=rcond)
    c = solution[:n]

    # Compensate for small constraint errors from the pseudoinverse.
    coefficient_sum = np.sum(c)
    if abs(coefficient_sum) < 1e-14:
        c = np.zeros(n)
        c[-1] = 1.0
    else:
        c /= coefficient_sum

    new_x = np.zeros_like(
        xs[0],
        dtype=np.result_type(*xs, np.float64),
    )
    for coefficient, x in zip(c, xs):
        new_x += coefficient * x

    if return_c:
        return new_x, c
    return new_x


class DIIS(CDIIS):
    def __init__(
        self,
        mf=None,
        filename=None,
        Corth=None,
        space=20,
        **kwargs,
    ):
        super().__init__(
            mf=mf,
            filename=filename,
            Corth=Corth,
        )
        self.space = space
        self.residuals_history = []
        self.fock_history = []

    def update(self, s, dm, fock, mf, h1e, vhf, *args, **kwargs):
        # Matches PySCF's RHF/UHF, orthogonalization, and symmetry handling.
        residual = fock @ dm @ s - s @ dm @ fock

        self.fock_history.append(fock)
        self.residuals_history.append(residual)

        if len(self.fock_history) > self.space:
            self.fock_history = self.fock_history[-self.space:]
            self.residuals_history = self.residuals_history[-self.space:]

        if len(self.fock_history) < 2:
            return fock
        
        return DIIS(self.fock_history, self.residuals_history)