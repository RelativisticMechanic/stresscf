# EDIIS Implementation
# Kudin et al. (2002)
# Siddharth Gautam, 2026

import numpy as np
import scipy.optimize

def EDIISExtrapolate(
    etot: list,
    focks: list,
    densities: list,
    return_c=False,
    tol=1e-10,
    maxiter=500,
):

    m = len(etot)

    assert(len(focks) == m)
    assert(len(densities) == m)

    etot = np.asarray(etot, dtype=float)
    focks = np.asarray(focks)
    densities = np.asarray(densities)

    nao = focks.shape[-1]

    fock_shape = focks.shape[1:]

    densities = densities.reshape(m, -1, nao, nao)
    focks = focks.reshape(m, -1, nao, nao)

    # Trick to calculate:
    # A_ij = Tr[(F_i - F_j)(D_i - D_j)]
    # Notice that:
    # A_ij = Tr(F_i D_i) + Tr(F_j D_j) - [Tr(F_i D_j) + Tr(F_j D_i)]
    
    # Steps:
    # 1. Calculate the matrix Tr(F_i D_j)
    # 2. The first part is the diagonal + transpose of the diagonal
    # 3. The second part is the matrix + its transpose
    # Note that the diagonal of A_ij = 0, which also happens here.

    # Tr(PQ) = \sum{P_pq Q_qp}
    # The n-index is spin A.O., which should be the same.
    # i & j remain
    FD = np.einsum('inpq,jnqp->ij', focks, densities).real

    # Numpy will automatically create a N x N matrix
    # by addition of two N x 1 diagonals.
    A = np.diag(FD) + np.diag(FD)[:, None]
    A -= FD + FD.T

    def fEDIIS(c):
        # EDIIS functional
        # This is the PySCF/Kudin EDIIS convention for the A matrix above.
        # It also keeps this standalone extrapolator consistent with
        # pyscf.scf.diis.ediis_minimize.
        # f^{EDIIS}(c) = c^T E - c^T A c
        return np.einsum('i,i', c, etot) - np.einsum('i,ij,j', c, A, c)

    def grad_fEDIIS(c):
        # The gradient of EDIIS functional in terms of c is:
        # df/dc = E - 2 A c
        return etot - 2.0 * np.einsum('i,ik->k', c, A)

    # Optimize c directly on the simplex.  Squaring unconstrained auxiliary
    # variables makes the problem non-convex in those variables and can trap
    # BFGS at a false stationary point.  Sample the vertices, center, and the
    # offset points proposed by OpenOrbitalOptimizer, then start SLSQP from
    # the best sampled coefficient vector.
    vertices = np.eye(m)
    center = np.full((1, m), 1.0 / m)
    offset = np.full((m, m), 1.0 / (m + 2.0))
    np.fill_diagonal(offset, 3.0 / (m + 2.0))
    candidates = np.vstack((vertices, center, offset))
    x0 = min(candidates, key=fEDIIS)

    result = scipy.optimize.minimize(
        fEDIIS,
        x0,
        method='SLSQP',
        jac=grad_fEDIIS,
        bounds=[(0.0, 1.0)] * m,
        constraints={
            'type': 'eq',
            'fun': lambda c: np.sum(c) - 1.0,
            'jac': lambda c: np.ones_like(c),
        },
        options={'ftol': tol, 'maxiter': maxiter},
    )

    c = np.clip(result.x, 0.0, 1.0)
    coefficient_sum = np.sum(c)
    if not np.isfinite(coefficient_sum) or coefficient_sum <= 0.0:
        c = np.array(x0, copy=True)
    else:
        c /= coefficient_sum
    f_ediis = np.einsum('i,inpq->npq', c, focks)
    f_ediis = f_ediis.reshape(fock_shape)

    if return_c:
        return f_ediis, c
    else:
        return f_ediis
