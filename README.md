# STRESSCF 

This is a program to test an SCF DIIS implementation against standard benchmarks in literature. 

The benchmarks are taken from "Comparison of self-consistent field convergence acceleration techniques" by Alejandro J. Garza and Gustavo E. Scuseria, published in The Journal of Chemical Physics (2012). 

These include:

| Molecule                         | Method / Basis        | Guess  | 
| ---                              | ---                   | ---    |
| CH3CHO                           | UHF/6-31G(d)          | SAD    |
| CrC                              | RB3LYP/6-31G          | SAD    |
| CdIm2                            | RB3LYP/3-21G          | Core   |
| SiH4                             | RVWN5/VWN5            | SAD    |
| UF4                              | RB3LYP/LANL2DZ        | SAD    |
| UF(BS)                           | UB3LYP/LANL2DZ        | Triplet|
| Ru4CO                            | RB3LYP/LANL2DZ        | SAD    |
| Ru4CO(BS)                        | UB3LYP/LANL2DZ        | Triplet|
| Cr2                              | RHF/6-31G             | Core   |
| N-Methyl-(2-nitrovinyl)amine (E) | RHF/6-31G(d)          | Core   |


The LANL2DZ basis functions are taken from Basis Set Exchange (BSE) and are supplied under basis/ in Gaussian format
while the ECPs are in NWChem format.

In addition to this, the package also includes implementation of Pure Roothaan Hall, LCIIS (Li & Yaron, 2016) and ODA (Cances & Le Bris, 1999) that can be readily used with pyscf.

The program will also generate curves for change in energy per iteration, 
Example output (on UF4):

```
~/projects/stresscf ❯ python ./stresscf.py ./tests/uf4.json 
[18:33:43 STRESSCF]: **************************************************
[18:33:43 STRESSCF]: Testing: uf4
[18:33:43 STRESSCF]: Total Gaussian basis functions: 72
[18:33:43 STRESSCF]: ==================================================
[18:33:43 STRESSCF]: CDIIS on uf4
[18:33:49 STRESSCF]: Energy: -451.2187381039033 (c)
[18:33:49 STRESSCF]: Iterations: 36
[18:33:49 STRESSCF]: Time taken: 5802  ms
[18:33:49 STRESSCF]: Time/Iteration: 161.17 ms
[18:34:03 STRESSCF]: Internal stability: True
[18:34:03 STRESSCF]: External stability: False
[18:34:03 STRESSCF]: ==================================================
[18:34:03 STRESSCF]: EDIIS on uf4
[18:34:41 STRESSCF]: Energy: -430.57926777222167 (x)
[18:34:41 STRESSCF]: Iterations: 250
[18:34:41 STRESSCF]: Time taken: 36369  ms
[18:34:41 STRESSCF]: Time/Iteration: 145.48 ms
[18:34:41 STRESSCF]: Did not converge, will not conduct stability analysis.
[18:34:41 STRESSCF]: ==================================================
[18:34:41 STRESSCF]: ADIIS on uf4
[18:35:23 STRESSCF]: Energy: -451.21862216202237 (x)
[18:35:23 STRESSCF]: Iterations: 250
[18:35:23 STRESSCF]: Time taken: 42152  ms
[18:35:23 STRESSCF]: Time/Iteration: 168.61 ms
[18:35:23 STRESSCF]: Did not converge, will not conduct stability analysis.
[18:35:23 STRESSCF]: ==================================================
[18:35:23 STRESSCF]: LCIIS on uf4
[18:35:26 STRESSCF]: Energy: -451.1791089655013 (c)
[18:35:26 STRESSCF]: Iterations: 11
[18:35:26 STRESSCF]: Time taken: 2291  ms
[18:35:26 STRESSCF]: Time/Iteration: 208.28 ms
[18:35:38 STRESSCF]: Internal stability: False
[18:35:38 STRESSCF]: External stability: False
[18:35:39 STRESSCF]: **************************************************
[18:35:39 STRESSCF]: All tests completed.
[18:35:39 STRESSCF]: Saved summary results to: results/results.csv
[18:35:39 STRESSCF]: Saved iteration histories to: results/histories.csv
[18:35:39 STRESSCF]: Saved plots to: ./projects/stresscf/results
```