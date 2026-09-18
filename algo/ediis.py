# EDIIS Implementation
# Kudin et al. (2002)
# Siddharth Gautam, 2026

import numpy as np
import scipy.optimize

from scipy.optimize import OptimizeResult

def EDIISExtrapolate(etot : list, 
          focks : list, 
          densities : list, return_c=False):

    m = len(etot)

    assert(len(focks) == m)
    assert(len(densities) == m)

    focks = np.array(focks)
    densities = np.array(densities)

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
    FD = np.einsum('inpq,jnqp->ij', focks, densities)

    # Numpy will automatically create a N x N matrix
    # by addition of two N x 1 diagonals.
    A = np.diag(FD) + np.diag(FD)[:, None]
    A -= FD + FD.T

    def fEDIIS(x):
        # Trick: Instead of enforcing constraints via KKT, use
        # the idea that \sum{c_i} = 1 and c_i > 0 via squares
        # and norms.
        c = x ** 2 / np.sum(x ** 2)

        # EDIIS functional
        # f^{EDIIS}(c) = c^T E - 0.5 c^T A c
        return np.einsum('i,i', c, etot) - 0.5 * np.einsum('i,ij,j', c, A, c)

    def grad_fEDIIS(x):
        S = np.sum(x ** 2)
        c = x ** 2 / S
        # The gradient of EDIIS functional in terms of c is:
        # df/dc = E - A c
        dfdc = etot - np.einsum('i,ik->k', c, A)

        # However, since we are using x, we need df/dx.
        # df/dx =  [dc/dx]df/dc
        # c_i = x_i^2 / S (S = sum{x_k^2})
        # d(c_i)/d(x_j) = (2/S^2)((\delta_ij)(x_i)S - (x_i^2)(x_j))
        # Generate diagonal
        xS = np.diag(x * S)
        # Generate the second term by an outer-product
        xi2xj = np.einsum('k,n->kn', x**2, x)
        dcdx = (xS - xi2xj) * (2.0 / S ** 2)

        # Project the derivative
        # df/dx = [dc/dx]_ij [df/dc]_i
        dfdx = np.einsum('i,ij->j', dfdc, dcdx)

        return dfdx
    
    result : OptimizeResult = scipy.optimize.minimize(
        fEDIIS,
        np.ones(m),
        method='BFGS',
        jac=grad_fEDIIS,
        tol=1e-9
    )

    c = (result.x ** 2) / np.sum(result.x ** 2)
    f_ediis = np.einsum('i,inpq->npq', c, focks)
    f_ediis = f_ediis.reshape(fock_shape)
    
    if return_c:
        return f_ediis, c
    else:
        return f_ediis


     