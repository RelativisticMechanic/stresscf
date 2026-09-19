#!/usr/bin/env python3
"""
STRESSCF
Benchmark DIIS variants on test molecules.

Measures:
    - SCF convergence history
    - Final energy
    - Number of SCF cycles
    - dE vs iteration
    - Pulay commutator norm vs iteration

Outputs:
    ./results/results.csv
    ./results/histories.csv
    ./results/<molecule>_<quantity>.png
"""

import os
import sys

# Force underlying math libraries to use exactly 1 thread
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from pathlib import Path

from pyscf import gto, scf, dft
from pyscf.scf.hf import SCF

from pyscf.scf.diis import CDIIS, EDIIS, ADIIS

from logger import getLogger, ProgressBar
from triplet import fetchTripletGuess
from algo.oda import ODAInterpolate

from config import DIIS_METHODS, DIIS_M, DIIS_START, GRID_LEVEL, CONV_E, CONV_G
from config import DYNAMIC_DAMPING
from config import PLOT_LINE_WIDTH, PLOT_MARKER_SIZE
from config import SCF_MAX_ITER

# Support both HF and DFT inputs
METHOD_MAP = {
    "rhf": scf.RHF,
    "uhf": scf.UHF,
    "rks": dft.RKS,
    "uks": dft.UKS,
}

# Used by RDIIS to access C & epislon
CUSTOM_VARIABLE_ACCESSOR = {
    "mo_coeff": None,
    "eps": None,
    "mo_occ": None,
    "dm": None,
    "e": None,
    "de": None
}

logger = getLogger("STRESSCF")

def installFirstIterationDamping(mf):
    """Apply Gaussian-style ODA damping before DIIS starts collecting data.

    The first SCF Fock matrix is returned unchanged.  On the next cycle, ODA
    interpolates between the first and second Fock/density pairs.  Subsequent
    cycles are passed to PySCF unchanged, so the configured accelerator starts
    with a clean, post-damping history.

    Returns a small state dictionary for diagnostics and tests.
    """
    get_fock_without_driver_damping = mf.get_fock
    state = {
        "previous_fock": None,
        "previous_dm": None,
        "new_fraction": None,
        "damping_factor": None,
        "complete": False,
    }

    def get_fock(
        h1e=None,
        s1e=None,
        vhf=None,
        dm=None,
        cycle=-1,
        diis=None,
        diis_start_cycle=None,
        level_shift_factor=None,
        damp_factor=None,
        fock_last=None,
    ):
        common_arguments = {
            "h1e": h1e,
            "s1e": s1e,
            "vhf": vhf,
            "dm": dm,
            "cycle": cycle,
            "diis_start_cycle": diis_start_cycle,
            "level_shift_factor": level_shift_factor,
            "damp_factor": damp_factor,
            "fock_last": fock_last,
        }

        if diis is not None and cycle in (0, 1):
            # Suppress DIIS while constructing the two points used by ODA.
            fock = get_fock_without_driver_damping(
                diis=None,
                **common_arguments,
            )

            if cycle == 0:
                state["previous_fock"] = np.array(fock, copy=True)
                state["previous_dm"] = np.array(dm, copy=True)
                return fock

            if state["previous_fock"] is not None:
                new_fraction, _, damped_fock = ODAInterpolate(
                    state["previous_fock"],
                    fock,
                    state["previous_dm"],
                    dm,
                )
                new_fraction = float(np.real(new_fraction))
                state["complete"] = True
                if np.isfinite(new_fraction):
                    new_fraction = float(np.clip(new_fraction, 0.0, 1.0))
                    state["new_fraction"] = new_fraction
                    state["damping_factor"] = 1.0 - new_fraction
                    return damped_fock

            # A non-finite ODA model should not abort the SCF calculation.
            state["complete"] = True
            return fock

        return get_fock_without_driver_damping(
            diis=diis,
            **common_arguments,
        )

    mf.get_fock = get_fock
    return state

def convertToFloat(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def commutatorRMSNorm(fock, dm, s1e):
    """
    Compute the RMS norm of the commutator FDS - SDF.

    Handles RHF/UKS-style 2D arrays and UHF/UKS-style
    3D spin-separated arrays.
    """
    if fock is None or dm is None or s1e is None:
        return np.nan

    fock = np.asarray(fock)
    dm = np.asarray(dm)
    s1e = np.asarray(s1e)

    # Restricted / spin-summed case
    if fock.ndim == 2:
        r = fock @ dm @ s1e - s1e @ dm @ fock
        return float(np.sqrt(np.mean(r * r)))

    # Unrestricted / spin-separated case
    if fock.ndim == 3:
        total_sq = 0.0
        total_elements = 0

        for s in range(fock.shape[0]):
            r = fock[s] @ dm[s] @ s1e - s1e @ dm[s] @ fock[s]
            total_sq += np.sum(r * r)
            total_elements += r.size

        return float(np.sqrt(total_sq / total_elements))

    return np.nan

def maskPositiveFinite(y):
    y = np.asarray(y, dtype=float)
    return np.isfinite(y) & (y > 0)

def drawPlots(histories_by_method, molecule_name, output_dir, to_plot_quantities=["E", "dE", "comm"]):
    """
    Make one plot per quantity, with all methods overlaid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # quantity -> (ylabel, use_log_scale)
    quantities = {
        "E": ("E (Hartree)", False),
        "dE": (r"$\left|E_{n+1}-E_n\right|\;(\mathrm{Hartree})$", True),
        "comm": (r"$\left|FPS - SPF\right|$", True),
    }

    for quantity, (ylabel, use_log) in quantities.items():
        if quantity not in to_plot_quantities:
            continue
        
        plt.figure(figsize=(8, 6))

        for method_name, hist in histories_by_method.items():
            x = np.asarray(hist["cycle"], dtype=float)
            
            y = np.asarray(hist[quantity], dtype=float)

            if quantity == "E":
                y = np.abs(y)
                if not plt.gca().yaxis_inverted():
                    plt.gca().invert_yaxis()

            if quantity == "dE":
                plt.yscale('symlog', linthresh=1e-8)
                plt.plot(x, y, marker='.', 
                         linestyle='-', 
                         markersize=PLOT_MARKER_SIZE, 
                         linewidth=PLOT_LINE_WIDTH, 
                         label=method_name)
                
                continue

            mask = maskPositiveFinite(y)

            if np.any(mask):
                if use_log:
                    plt.semilogy(x[mask], y[mask], marker=".", 
                                 markersize=PLOT_MARKER_SIZE, 
                                 linewidth=PLOT_LINE_WIDTH, 
                                 label=method_name)
                else:
                    plt.plot(x[mask], y[mask], marker=".", 
                             markersize=PLOT_MARKER_SIZE, 
                             linewidth=PLOT_LINE_WIDTH, 
                             label=method_name)

        plt.xlabel("Iteration")
        plt.ylabel(ylabel)
        plt.title(f"{molecule_name} - {quantity}")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend(fontsize=8)
        plt.tight_layout()

        outfile = output_dir / f"{molecule_name}_{quantity}.png"
        plt.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close()

def testMoleculeDIISMethod(molecule_name: str, molecule: gto.Mole,
                       diis_obj, method=scf.UHF, xc=None, init_guess="hcore",
                       stability_analysis=False, initial_dm=None):
    """
    Run one SCF calculation and record convergence history.
    Returns:
        summary_dict, history_dict
    """

    mf : SCF = method(molecule)

    if xc is not None and hasattr(mf, "xc"):
        mf.xc = xc

    if init_guess not in ("gaussian", "triplet"):
        mf.init_guess = init_guess

    mf.diis_start_cycle = DIIS_START
    mf.conv_tol = CONV_E
    mf.conv_tol_grad = CONV_G
    mf.level_shift = 0
    mf.damp = 0.0
    mf.max_cycle = SCF_MAX_ITER

    try:
        mf.grids.level = GRID_LEVEL
    except:
        pass


    if initial_dm is None:
        CUSTOM_VARIABLE_ACCESSOR["dm"] = mf.get_init_guess()
    else:
        CUSTOM_VARIABLE_ACCESSOR["dm"] = initial_dm

    CUSTOM_VARIABLE_ACCESSOR["e"] = mf.energy_tot(dm=CUSTOM_VARIABLE_ACCESSOR["dm"])
    
    # DIIS object instance
    if diis_obj == scf.diis.CDIIS or diis_obj == scf.diis.ADIIS or diis_obj == scf.diis.EDIIS:
        mf.diis = diis_obj()
        mf.diis.space = DIIS_M
    else:
        data_dir = f"./trajectory_data/{molecule_name}_{diis_obj.__name__}"
        mf.diis = diis_obj(custom_variable_access=CUSTOM_VARIABLE_ACCESSOR, test_case_id=data_dir)
        mf.diis.space = DIIS_M

    # Gaussian 09 applies dynamic damping only to its first inter-iteration
    # step.  Keep this SCF policy in the benchmark driver, not in any DIIS
    # implementation, and begin the accelerator history afterwards.

    if DYNAMIC_DAMPING:
        installFirstIterationDamping(mf)

    history = {
        "cycle": [],
        "E": [],
        "dE": [],
        "comm": []
    }

    status = "ok"
    error_message = ""
    progress = ProgressBar(mf.max_cycle, name=logger.name)

    def callback(envs):
        cycle = int(envs.get("cycle", len(history["cycle"])))
        e = convertToFloat(envs.get("e_tot", np.nan))

        prev_e = history["E"][-1] if history["E"] else np.nan
        de = np.nan if not np.isfinite(prev_e) else np.abs(e - prev_e)

        s1e = envs.get("s1e", None)
        fock = envs.get("fock", None)
        dm = envs.get("dm", None)
        eps = envs.get("mo_energy", None)
        c = envs.get("mo_coeff", None)
        mc = envs.get("mo_occ", None)

        global CUSTOM_VARIABLE_ACCESSOR

        old_e = CUSTOM_VARIABLE_ACCESSOR["e"]
        old_dm = CUSTOM_VARIABLE_ACCESSOR["dm"]
        old_c = CUSTOM_VARIABLE_ACCESSOR["mo_coeff"]
        old_eps = CUSTOM_VARIABLE_ACCESSOR["eps"]

        CUSTOM_VARIABLE_ACCESSOR["mo_coeff"] = c
        CUSTOM_VARIABLE_ACCESSOR["dm"] = dm
        CUSTOM_VARIABLE_ACCESSOR["mo_occ"] = mc
        CUSTOM_VARIABLE_ACCESSOR["e"] = e
        CUSTOM_VARIABLE_ACCESSOR["eps"] =  np.eye(eps.shape[-1]) * eps[..., None]

        if old_e is not None:
            CUSTOM_VARIABLE_ACCESSOR["de"] = e - old_e
            
        pulay_norm = commutatorRMSNorm(fock, dm, s1e)

        # Add to history
        history["cycle"].append(cycle)
        history["E"].append(e)
        history["dE"].append(de)
        history["comm"].append(pulay_norm)
        progress.update(len(history["cycle"]), e - prev_e)

    mf.callback = callback

    import time
    start_time = time.perf_counter() * 1000.0
    progress.update(0, 0)
    try:
        try:
            if initial_dm is not None:
                # Give every accelerator an independent copy of the same guess.
                final_energy = mf.kernel(dm0=np.array(initial_dm, copy=True))
            else:
                final_energy = mf.kernel()
        finally:
            progress.finish()

            converged = bool(mf.converged)
            end_time = time.perf_counter() * 1000.0

            logger.info(f"Energy: {final_energy} ({'c' if converged else 'x'})")
            logger.info(f"Iterations: {len(history['cycle'])}")
            logger.info(f"Time taken: {int(end_time - start_time)}  ms")
            logger.info(f"Time/Iteration: {((end_time - start_time) / len(history['cycle'])):.2f} ms")

            if stability_analysis:
                if converged:
                    mo_i, mo_e, stable_i, stable_e = mf.stability(
                        internal=True,
                        external=True,
                        return_status=True
                    )

                    logger.info(f"Internal stability: {stable_i}")
                    logger.info(f"External stability: {stable_e}")
                else:
                    logger.info("Did not converge, will not conduct stability analysis.")

    except Exception as exc:
        final_energy = np.nan
        converged = False
        status = "failed"
        import traceback
        print(traceback.format_exc())
        error_message = repr(exc)

    summary = {
        "molecule": molecule_name,
        "label": diis_obj.__name__,
        "energy": convertToFloat(final_energy),
        "cycles": len(history["cycle"]),
        "converged": converged,
        "status": status,
        "error_message": error_message,
    }

    return summary, history

def testMolecule(name, mol, method, init_guess, xc=None, stability_analysis=False,
                 initial_dm=None):
    """
    Run all benchmark selected DIIS variants on one molecule.

    Returns:
        summaries (list of dict),
        histories_by_method (dict),
        long_history_rows (list of dict)
    """
    summaries = []
    histories_by_method = {}
    long_history_rows = []
    
    for diis_method in DIIS_METHODS:
        logger.info('=' * 50)
        logger.info(f'{diis_method.__name__} on {name}')
        summary, history = testMoleculeDIISMethod(name, mol, 
                                              diis_method, 
                                              method=method, 
                                              xc=xc, 
                                              init_guess=init_guess,
                                              stability_analysis=stability_analysis,
                                              initial_dm=initial_dm)

        summaries.append(summary)
        histories_by_method[diis_method.__name__] = history

        # Flatten the time series for CSV storage
        n = len(history["cycle"])

        for i in range(n):
            long_history_rows.append({
                    "molecule": name,
                    "method": diis_method.__name__,
                    "cycle": history["cycle"][i],
                    "E": history["E"][i],
                    "dE": history["dE"][i],
                    "comm": history["comm"][i],
                })

    return summaries, histories_by_method, long_history_rows

def runTestCase(test_case, molecule_data, test_dir_root):
    """
    Run a test case given molecule data.
    Calls testMolecule().
    """

    molecule = gto.Mole()

    molecule.verbose = 0

    # XYZ loading. Discard first two lines. Rest is passed
    # pyscf's Mole object.
    coords_file = test_dir_root / molecule_data["file"]
    with open(coords_file, "r") as f:
        lines = f.readlines()

    molecule_coords_data = "".join(lines[2:])
    molecule.atom = molecule_coords_data
    molecule.cart = molecule_data.get("cartesian", False)

    basis = molecule_data.get("basis", "6-31G")

    def parse_custom_field(prefix):
        custom_files = molecule_data.get(f"{prefix}_files", {})
        custom_data = {}
        for element in custom_files.keys():
            custom_field = custom_files.get(element)

            if custom_field.startswith("file:"):
                custom_file = custom_field.split(":")[1]
                from pyscf.gto.basis import parse_gaussian, parse_ecp
                with open(f'./basis/{custom_file}') as f:
                    if prefix == "basis":
                        gbs_data = parse_gaussian.parse(f.read())
                    elif prefix == "ecp":
                        # PySCF has exposed parse_ecp as either a function or
                        # a module across releases.
                        parser = getattr(parse_ecp, "parse", parse_ecp)
                        gbs_data = parser(f.read())
                custom_data[element] = gbs_data
            else:
                custom_data[element] = custom_field
        return custom_data

    if basis == "custom":
        molecule.basis = parse_custom_field("basis")
    else:
        molecule.basis = basis

    # Load infos like ecp, charge, mult.
    if "ecp" in molecule_data:
        ecp = molecule_data.get("ecp", "")
        if ecp == "custom":
            molecule.ecp = parse_custom_field("ecp")
        else:
            molecule.ecp = molecule_data["ecp"]
    
    if "charge" in molecule_data:
        molecule.charge = molecule_data["charge"]
    
    if "multiplicity" in molecule_data:
        molecule.spin = (molecule_data["multiplicity"] - 1)
    
    init_guess = molecule_data.get("init_guess", "hcore")


    method_name = molecule_data["method"].lower()

    if method_name not in METHOD_MAP:
        raise ValueError(f"Unsupported method '{method_name}'. "
                            f"Supported: {list(METHOD_MAP.keys())}")

    method = METHOD_MAP[method_name]

    # Optional XC for DFT tests
    xc = molecule_data.get("xc", None)

    stability_analysis = molecule_data.get("stability_analysis", False)
    molecule.build()
    logger.info(f"Total Gaussian basis functions: {molecule.nao}")

    initial_dm = None

    if init_guess == "triplet":
        guess_mf = method(molecule)
        if xc is not None and hasattr(guess_mf, "xc"):
            guess_mf.xc = xc

        guess_mf.conv_tol = molecule_data.get("triplet_conv_tol", 1e-10)
        guess_mf.conv_tol_grad = molecule_data.get("triplet_conv_tol_grad", 1e-6)
        guess_mf.max_cycle = molecule_data.get("triplet_max_cycle", 200)

        try:
            guess_mf.grids.level = GRID_LEVEL
        except AttributeError:
            pass

        triplet_guess = fetchTripletGuess(
            guess_mf,

            cache_dir=os.path.join(test_dir_root, molecule_data.get(
                "triplet_cache_dir", "guess_cache"
            )),

            source_init_guess=molecule_data.get("triplet_init_guess", "atom"),

            max_spin_contamination=molecule_data.get(
                "triplet_max_spin_contamination", None
            ),

            force_rebuild=molecule_data.get("triplet_force_rebuild", False),
        )

        initial_dm = triplet_guess.dm

        logger.info(
            "Triplet pre-SCF: E=%.6f Ha, cycles=%d, <S^2>=%.8f, "
            "2S+1=%.6f, stage=%s",
            triplet_guess.triplet_energy,
            triplet_guess.triplet_cycles,
            triplet_guess.spin_square,
            triplet_guess.spin_multiplicity,
            triplet_guess.convergence_stage,
        )

        logger.info(
            "Triplet guess cache: %s (%s)",
            "hit" if triplet_guess.cache_hit else "miss; saved",
            triplet_guess.cache_file,
        )

        if triplet_guess.spin_square > 2.5:
            logger.warning(
                "Triplet source is substantially spin contaminated: <S^2>=%.8f",
                triplet_guess.spin_square,
            )
    
    summaries, histories_by_method, long_history_rows = testMolecule(test_case, 
                                                                     molecule, method, 
                                                                     init_guess, xc=xc, 
                                                                     stability_analysis=stability_analysis,
                                                                     initial_dm=initial_dm)
    
    return summaries, histories_by_method, long_history_rows

def writeCSVAppend(df: pd.DataFrame, csv_file : Path):
    if os.path.isfile(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, header=True, index=False)


def main(json_file="tests.json", result_dir="results"):

    """
    Given a JSON file and a result directory, run DIIS
    variants on all cases, record performance history
    and save plots.
    """

    result_dir = Path(result_dir)
    json_file = Path(json_file)

    result_dir.mkdir(parents=True, exist_ok=True)
    
    if not json_file.exists():
        raise FileNotFoundError(f"No file found at {json_file}")

    with open(json_file, "r") as f:
        test_cases = json.loads(f.read())["tests"]

    summary_rows = []
    history_rows = []

    for test_case in test_cases.keys():
        logger.info("*" * 50)
        logger.info(f"Testing: {test_case}")
        molecule_data = test_cases[test_case]
        to_plot_quantities = molecule_data.get("outputs", ["E", "dE", "comm", "deviation"])

        # Check if this a test directory
        if molecule_data["file"].startswith("TESTDIR "):
            testset_dir = json_file.parent / molecule_data["file"].split(" ")[1]
            xyz_files = os.listdir(testset_dir)

            for file in xyz_files:
                if file.endswith(".xyz"):
                    test_case_internal = Path(file).stem
                    print(f"Testing: {test_case_internal}")
                    molecule_data["file"] = file
                    summaries, histories_by_method, long_history_rows = runTestCase(test_case_internal, 
                                                                                        molecule_data, 
                                                                                        testset_dir)
                    drawPlots(histories_by_method, test_case_internal, result_dir, to_plot_quantities)
                    summary_rows.extend(summaries)
                    history_rows.extend(long_history_rows)
        
        else:
            summaries, histories_by_method, long_history_rows = runTestCase(test_case, 
                                                                                molecule_data, 
                                                                                json_file.parent)
            drawPlots(histories_by_method, test_case, result_dir, to_plot_quantities)
            summary_rows.extend(summaries)
            history_rows.extend(long_history_rows)

    df_results = pd.DataFrame(summary_rows)
    df_histories = pd.DataFrame(history_rows)

    writeCSVAppend(df_results, result_dir / "results.csv")
    writeCSVAppend(df_histories, result_dir / "histories.csv")

    logger.info("*" * 50)
    logger.info("All tests completed.")

    logger.info(f"Saved summary results to: {result_dir / 'results.csv'}")
    logger.info(f"Saved iteration histories to: {result_dir / 'histories.csv'}")
    logger.info(f"Saved plots to: {result_dir.resolve()}")

if __name__ == "__main__":
    result_dir = Path("./results")

    if len(sys.argv) < 2:
        print("Usage: python ./stresscf.py [JSON] [Result Directory (optional)]")
        exit(-1)

    if len(sys.argv) >= 2:
        json_file = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        result_dir = Path(sys.argv[3])

    main(json_file, result_dir)
