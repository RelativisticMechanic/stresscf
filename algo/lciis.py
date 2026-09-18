# LCIIS Implementation
# Li & Yaron (2016)

# The Least-Squares Commutator in the Iterative Subspace (LCIIS)
# minimizes the commutator of the interpolated fock and density matrices.

# Given F1, F2, F3, ..., Fn and D1, D2, D3, ..., Dn, find c_i such that:
# F' = \sum{c_i F_i}
# D' = \sum{c_i D_i}
# Minimize the commutator: F'D'S - SD'F' s.t. \sum{c_i} = 1 

import numpy as np

from pyscf.scf.diis import CDIIS

from .diis import DIISExtrapolate

def LCIISExtrapolate(focks: list, densities: list, s : np.array, 
                      return_c : bool = False, maxiter : int = 100, tol : float = 1e-9):
    """
    Extrapolate an LCIIS Fock from a given list of fock and density matrices.
    
    - focks: list of Fock matrices
    - densities: list of density matrices
    - s: overlap matrix
    - return_c: Whether to return (f_lciis, c) or just f_lciis
    - maxiter: Number of minimizer iterations
    - tol: tolerance value for minizer
    """
    m = len(focks)

    assert(len(densities) == m)

    focks : np.ndarray = np.array(focks)
    densities : np.ndarray = np.array(densities)

    fock_shape = focks.shape[1:]
    nao = fock_shape[-1]

    focks = focks.reshape(m, -1, nao, nao)
    densities = densities.reshape(m, -1, nao, nao)

    # Calculate the commutator tensor
    # [F_i @ D_j] => (spin x n_AO x n_AO)_i @ (spin x n_AO x n_AO)_j @ (n_AO x n_AO) => (nxn)_ij.

    # F_i D_j S
    FD_ij = np.einsum('inpq,jnqr,rs->ijnps', focks, densities, s, optimize=True)
    # D_j F_i S
    DF_ij = np.einsum('pq,jnqr,inrs->ijnps', s, densities, focks, optimize=True)

    # F_i D_j S - D_j F_i S
    C_ij = FD_ij - DF_ij

    # Calculate T_ijkl as per LCIIS
    T = np.einsum('ijnpq,klnpq->ijkl', C_ij, C_ij, optimize=True)


    def fLCIIS(c):
        """
        LCIIS energy functional
        """
        return np.einsum('i,j,k,l,ijkl->', c, c, c, c, T, optimize=True)

    def gLCIIS(c):
        """
        Gradient of the LCIIS commutator with respect to c
        """
        return 2.0 * (np.einsum('j,k,l,ijkl->i', c, c, c, T, optimize=True) 
                      + np.einsum('j,k,l,jikl->i', c, c, c, T, optimize=True))

    def HLCIIS(c):
        """
        Hessian of the LCIIS commutators with respect to c
        """
        return 2.0 * (np.einsum("k,l,ijkl->ij", c, c, T, optimize=True) 
                + np.einsum("k,l,ikjl->ij", c, c, T, optimize=True)
                + np.einsum("k,l,iklj->ij", c, c, T, optimize=True)
                + np.einsum("k,l,jikl->ij", c, c, T, optimize=True)
                + np.einsum("k,l,kijl->ij", c, c, T, optimize=True)
                + np.einsum("k,l,kilj->ij", c, c, T, optimize=True))

    def KKTMatrix(c):
        """
        Returns KKT matrix of LCIIS
        [[H 1^T]
         [1  0]]
        """

        H = HLCIIS(c)
        kkt = np.block([
            [H, np.ones((len(c), 1))],
            [np.ones((1, len(c))), np.zeros((1, 1))]
            ])
        return kkt
    
    def RHS(c, l):
        """
        Return RHS of LCIIS equation
        [[g(c) + l]
         [0]]
        """
        rhs = np.block([
            gLCIIS(c) + l * np.ones_like(c),
            0
        ])
        return rhs


    # DIIS residuals = C_ii
    C_ii = [C_ij[i, i] for i in range(m)]
    C_ii = list(C_ii)

    # Lagrange multiplier
    l = 0.0

    # Start with DIIS residuals as an estimate.
    f_diis, c = DIISExtrapolate(focks, C_ii, return_c=True)

    # Solve the LCIIS iteration
    for iteration in range(maxiter):
        K = KKTMatrix(c)
        rhs = RHS(c, l)

        delta = np.linalg.solve(K, -rhs)

        dc = delta[:-1]
        dl = delta[-1]

        c += dc
        l += dl

        # Check convergence
        if np.linalg.norm(dc) < tol and abs(dl) < tol:
            break

    # Construct LCIIS matrix from linear combination
    f_lciis = np.einsum('i,inpq->npq', c, focks, optimize=True)
    f_lciis = f_lciis.reshape(fock_shape)

    if return_c:
        return f_lciis, c
    else:
        return f_lciis

class LCIIS(CDIIS):
    def __init__(
        self,
        mf=None,
        Corth=None,
        space=8,
        **kwargs
    ):
        super().__init__(mf=mf, Corth=Corth)
        self.space = space

        self.fock_history = []
        self.density_history = []
        
    def update(self, s, dm, fock, mf, h1e, vhf, *args, **kwargs):
        
        self.density_history.append(dm)
        self.fock_history.append(fock)

        self.fock_history = self.fock_history[-self.space:]
        self.density_history = self.density_history[-self.space:]

        if len(self.density_history) < 2:
            return fock
        
        f_lciis = LCIISExtrapolate(self.fock_history, self.density_history, s)

        return f_lciis

