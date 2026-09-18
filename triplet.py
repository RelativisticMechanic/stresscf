"""
Broken-symmetry singlet guesses obtained from a converged triplet SCF.

The construction follows the triplet-guess protocol used for the Ru4(CO) and
UF4 examples in the LCIIS paper: converge an unrestricted M_S=1 state, retain
its separate alpha and beta orbitals, and reoccupy those orbitals with the
electron counts of the target M_S=0 calculation.
"""

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyscf
from pyscf import scf

from harris import harrisInitialGuess


TRIPLET_CACHE_VERSION = 1

@dataclass(frozen=True)
class TripletGuessResult:
    """Density and diagnostics from the preparatory triplet calculation."""

    dm: np.ndarray
    triplet_energy: float
    triplet_cycles: int
    spin_square: float
    spin_multiplicity: float
    convergence_stage: str
    cache_hit: bool = False
    cache_file: str = ""


def reoccupyUnrestrictedOrbitals(mo_coeff, nelec):
    """Build an unrestricted density by Aufbau-reoccupying two MO manifolds.

    Parameters
    ----------
    mo_coeff
        Alpha and beta MO coefficient matrices, each ordered by its orbital
        energy.  Columns are molecular orbitals.
    nelec
        Desired ``(nalpha, nbeta)`` occupation, normally the target singlet's
        electron counts.
    """

    coeff = np.asarray(mo_coeff)
    if coeff.ndim != 3 or coeff.shape[0] != 2:
        raise ValueError(
            "Triplet reoccupation requires alpha and beta MO coefficients "
            "with shape (2, nao, nmo)."
        )

    nalpha, nbeta = (int(nelec[0]), int(nelec[1]))
    nmo = coeff.shape[-1]

    if min(nalpha, nbeta) < 0 or max(nalpha, nbeta) > nmo:
        raise ValueError(f"Invalid target occupation {nelec} for {nmo} orbitals.")

    densities = []
    for spin, nocc in enumerate((nalpha, nbeta)):
        occupied = coeff[spin, :, :nocc]
        densities.append(occupied @ occupied.conj().T)
    return np.asarray(densities)


def copySCFSettings(source, destination):
    """
    Copy settings that define the physical SCF model and its convergence.
    """

    for name in (
        "xc",
        "nlc",
        "disp",
        "conv_tol",
        "conv_tol_grad",
        "max_cycle",
        "max_memory",
        "direct_scf",
        "level_shift",
        "damp",
        "diis_space",
        "verbose",
    ):
        if hasattr(source, name) and hasattr(destination, name):
            setattr(destination, name, getattr(source, name))

    if hasattr(source, "grids") and hasattr(destination, "grids"):
        destination.grids.level = source.grids.level


def _callable_name(value):
    return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', repr(value))}"


def generateTripletCacheKey(target_mf, source_init_guess):
    """
    Return a stable fingerprint of every input that can affect the guess.
    """

    mol = target_mf.mol

    def optional_float(value):
        return None if value is None else float(value)

    payload = {
        "cache_version": TRIPLET_CACHE_VERSION,
        "pyscf_version": pyscf.__version__,
        "method": f"{target_mf.__class__.__module__}.{target_mf.__class__.__qualname__}",
        "atom_charges": mol.atom_charges().tolist(),
        "atom_coords_bohr": mol.atom_coords().tolist(),
        "charge": int(mol.charge),
        "target_spin": int(mol.spin),
        "cart": bool(mol.cart),
        "basis": mol._basis,
        "ecp": mol._ecp,
        "pseudo": mol._pseudo,
        "source_init_guess": source_init_guess,
        "xc": getattr(target_mf, "xc", None),
        "nlc": getattr(target_mf, "nlc", None),
        "disp": getattr(target_mf, "disp", None),
        "conv_tol": optional_float(target_mf.conv_tol),
        "conv_tol_grad": optional_float(target_mf.conv_tol_grad),
        "max_cycle": int(target_mf.max_cycle),
        "level_shift": float(target_mf.level_shift),
        "damp": float(target_mf.damp),
        "diis_space": int(target_mf.diis_space),
    }
    if hasattr(target_mf, "grids"):
        grids = target_mf.grids
        payload["grids"] = {
            "level": int(grids.level),
            "atom_grid": grids.atom_grid,
            "radi_method": _callable_name(grids.radi_method),
            "becke_scheme": _callable_name(grids.becke_scheme),
            "prune": _callable_name(grids.prune),
        }

    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=repr
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validateCacheResult(result, target_mf, max_spin_contamination):
    nao = target_mf.mol.nao
    if result.dm.shape != (2, nao, nao):
        raise ValueError(f"Cached density has invalid shape {result.dm.shape}.")
    if not np.all(np.isfinite(result.dm)):
        raise ValueError("Cached density contains non-finite values.")

    overlap = target_mf.get_ovlp()
    populations = np.einsum("sij,ji->s", result.dm, overlap).real
    if not np.allclose(populations, target_mf.nelec, atol=1e-7, rtol=0.0):
        raise ValueError(
            f"Cached density has electron counts {populations}, expected "
            f"{target_mf.nelec}."
        )

    contamination = result.spin_square - 2.0
    if (
        max_spin_contamination is not None
        and contamination > float(max_spin_contamination)
    ):
        raise RuntimeError(
            f"The cached M_S=1 source has <S^2>={result.spin_square:.8f} "
            f"(contamination {contamination:.8f}), exceeding the allowed "
            f"{float(max_spin_contamination):.8f}."
        )


def loadCachedGuess(path, cache_key, target_mf, max_spin_contamination):
    with np.load(path, allow_pickle=False) as cached:
        if int(cached["cache_version"].item()) != TRIPLET_CACHE_VERSION:
            raise ValueError("Unsupported triplet cache version.")

        if str(cached["cache_key"].item()) != cache_key:
            raise ValueError("Triplet cache fingerprint mismatch.")

        result = TripletGuessResult(
            dm=np.array(cached["dm"], copy=True),
            triplet_energy=float(cached["triplet_energy"].item()),
            triplet_cycles=int(cached["triplet_cycles"].item()),
            spin_square=float(cached["spin_square"].item()),
            spin_multiplicity=float(cached["spin_multiplicity"].item()),
            convergence_stage=str(cached["convergence_stage"].item()),
            cache_hit=True,
            cache_file=str(path),
        )
    validateCacheResult(result, target_mf, max_spin_contamination)
    return result


def saveTripletCache(path, cache_key, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".npz", dir=path.parent, delete=False) as stream:

            temporary_path = Path(stream.name)

            np.savez_compressed(
                stream,
                cache_version=np.asarray(TRIPLET_CACHE_VERSION),
                cache_key=np.asarray(cache_key),
                dm=result.dm,
                triplet_energy=np.asarray(result.triplet_energy),
                triplet_cycles=np.asarray(result.triplet_cycles),
                spin_square=np.asarray(result.spin_square),
                spin_multiplicity=np.asarray(result.spin_multiplicity),
                convergence_stage=np.asarray(result.convergence_stage),
            )

        os.replace(temporary_path, path)

    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def buildTripletGuess(target_mf, source_init_guess="harris", max_spin_contamination=None):
    """
    Construct an M_S=0 unrestricted density from a converged triplet.

    ``target_mf`` must be UHF or UKS on a singlet molecule.  The preparatory
    calculation uses the same mean-field class, basis/ECP, and XC functional,
    but a copied molecule with ``spin=2`` (multiplicity three).

    The frontier reoccupation is implicit and deterministic: relative to the
    converged triplet, the highest occupied alpha orbital is emptied and the
    lowest vacant beta orbital is filled.  Equivalently, the first N/2 alpha
    and first N/2 beta orbitals are occupied for the target singlet.
    """

    if not isinstance(target_mf, scf.uhf.UHF):
        raise TypeError("A triplet-broken-symmetry guess requires UHF or UKS.")

    if target_mf.mol.spin != 0 or target_mf.nelec[0] != target_mf.nelec[1]:
        raise ValueError("The target must be an even-electron M_S=0 singlet.")

    triplet_mol = target_mf.mol.copy(deep=True)
    triplet_mol.spin = 2
    triplet_mol.build(dump_input=False, parse_arg=False)

    def new_triplet_mf():
        mf = target_mf.__class__(triplet_mol)
        copySCFSettings(target_mf, mf)
        return mf

    # Near the initial guess, energy-based interpolation is much less sensitive
    # than Pulay DIIS to the frontier-level reorderings seen in UF4.  Once that
    # transient is past, CDIIS is normally faster.  A second-order final stage
    # makes the guess builder fail-safe without changing the target benchmark.
    
    total_cycles = max(int(target_mf.max_cycle), 2)
    startup_cycles = min(30, max(1, total_cycles // 4))

    triplet_mf = new_triplet_mf()
    triplet_mf.max_cycle = startup_cycles
    triplet_mf.diis = scf.diis.ADIIS()
    triplet_mf.diis.space = getattr(target_mf, "diis_space", 8)

    def run_kernel(mf, dm0=None):
        cycles = 0

        def count_cycles(envs):
            nonlocal cycles
            cycles = max(cycles, int(envs.get("cycle", cycles)) + 1)

        mf.callback = count_cycles
        energy = mf.kernel(dm0=dm0) if dm0 is not None else mf.kernel()
        return energy, cycles

    if source_init_guess == "harris":
        triplet_energy, triplet_cycles = run_kernel(triplet_mf, harrisInitialGuess(triplet_mf))
    else:
        triplet_mf.init_guess = source_init_guess
        triplet_energy, triplet_cycles = run_kernel(triplet_mf)

    convergence_stage = "ADIIS"

    if not triplet_mf.converged:
        dm0 = triplet_mf.make_rdm1()
        triplet_mf = new_triplet_mf()
        triplet_mf.max_cycle = max(1, total_cycles - startup_cycles)
        triplet_mf.diis = scf.diis.CDIIS()
        triplet_mf.diis.space = getattr(target_mf, "diis_space", 8)
        triplet_energy, stage_cycles = run_kernel(triplet_mf, dm0)
        triplet_cycles += stage_cycles
        convergence_stage = "ADIIS->CDIIS"

    if not triplet_mf.converged:
        dm0 = triplet_mf.make_rdm1()
        triplet_mf = triplet_mf.newton()
        triplet_mf.max_cycle = min(50, max(10, total_cycles // 2))
        triplet_energy, stage_cycles = run_kernel(triplet_mf, dm0)
        triplet_cycles += stage_cycles
        convergence_stage = "ADIIS->CDIIS->Newton"

    if not triplet_mf.converged:
        raise RuntimeError(
            f"The preparatory triplet SCF did not converge after "
            f"{triplet_cycles} staged cycles ({convergence_stage}); no "
            "broken-symmetry singlet guess was generated."
        )

    dm = reoccupyUnrestrictedOrbitals(triplet_mf.mo_coeff, target_mf.nelec)

    # Validate electron counts in the nonorthogonal AO basis.
    overlap = target_mf.get_ovlp()

    populations = np.einsum("sij,ji->s", dm, overlap).real

    if not np.allclose(populations, target_mf.nelec, atol=1e-7, rtol=0.0):
        raise RuntimeError(f"Reoccupied density has electron counts {populations}, expected " f"{target_mf.nelec}.")

    spin_square, multiplicity = triplet_mf.spin_square()
    spin_contamination = float(spin_square) - 2.0

    if (max_spin_contamination is not None and spin_contamination > float(max_spin_contamination)):
        raise RuntimeError(
            f"The converged M_S=1 source has <S^2>={spin_square:.8f} "
            f"(contamination {spin_contamination:.8f}), exceeding the allowed "
            f"{float(max_spin_contamination):.8f}."
        )

    return TripletGuessResult(
        dm=dm,
        triplet_energy=float(triplet_energy),
        triplet_cycles=triplet_cycles,
        spin_square=float(spin_square),
        spin_multiplicity=float(multiplicity),
        convergence_stage=convergence_stage,
    )


def fetchTripletGuess(
    target_mf,
    cache_dir="triplet_guess_cache",
    source_init_guess="harris",
    max_spin_contamination=None,
    force_rebuild=False,
):
    """Load a compatible triplet guess or build and atomically cache one.

    A per-key advisory lock prevents concurrent benchmark processes from
    launching the same expensive triplet calculation.  Invalid or incomplete
    cache files are ignored and replaced.
    """
    
    import fcntl

    cache_key = generateTripletCacheKey(target_mf, source_init_guess)
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"triplet-{cache_key}.npz"
    lock_path = cache_dir / f"triplet-{cache_key}.lock"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def try_load():
        if force_rebuild or not cache_path.is_file():
            return None
        try:
            return loadCachedGuess(cache_path, cache_key, target_mf, max_spin_contamination)
        except (KeyError, OSError, ValueError):
            return None

    cached_result = try_load()
    if cached_result is not None:
        return cached_result

    with open(lock_path, "a+b") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)

        # Another process may have populated the cache while this process was
        # waiting for the lock.

        cached_result = try_load()

        if cached_result is not None:
            return cached_result

        result = buildTripletGuess(
            target_mf,
            source_init_guess=source_init_guess,
            max_spin_contamination=max_spin_contamination,
        )
        saveTripletCache(cache_path, cache_key, result)

        return replace(result, cache_hit=False, cache_file=str(cache_path))
