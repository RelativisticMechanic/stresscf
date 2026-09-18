import numpy

from pyscf import scf

def harrisInitialGuess(mf):
    """
    Construct a Harris-style initial density for molecular PySCF SCF.

    Supports RHF/RKS, UHF/UKS, and ROHF/ROKS.

    Returns
    -------
    ndarray
        RHF/RKS:
            (nao, nao)
        UHF/UKS and ROHF/ROKS:
            (2, nao, nao), containing alpha and beta densities.
    """

    mol = mf.mol

    is_uhf = isinstance(mf, scf.uhf.UHF)
    is_rohf = isinstance(mf, scf.rohf.ROHF)

    # Neutral-atom, spherically averaged, occupancy-fractional atomic
    # HF densities in the molecular AO basis.
    dm_sad = scf.hf.init_guess_by_atom(mol)

    hcore = mf.get_hcore(mol)
    s1e = mf.get_ovlp(mol)

    if is_uhf:
        # Spin-unpolarized frozen promolecule potential.
        dm_frozen = numpy.stack((0.5 * dm_sad, 0.5 * dm_sad))
        veff = mf.get_veff(mol, dm_frozen)
        fock = hcore[None, :, :] + veff

        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)

        return mf.make_rdm1(mo_coeff, mo_occ)

    if is_rohf:
        # ROHF/ROKS get_veff returns alpha and beta potentials.
        # They should be equal for this spin-unpolarized frozen density;
        # average them to form the single Harris orbital potential.
        dm_frozen = numpy.stack((0.5 * dm_sad, 0.5 * dm_sad))
        veff = mf.get_veff(mol, dm_frozen)
        fock = hcore + 0.5 * (veff[0] + veff[1])

        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)

        return mf.make_rdm1(mo_coeff, mo_occ)

    # RHF/RKS
    veff = mf.get_veff(mol, dm_sad)
    fock = hcore + veff

    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)

    return mf.make_rdm1(mo_coeff, mo_occ)