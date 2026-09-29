# tunnel-drawdown

Code, input files and results accompanying

> Kwon, Y. (2026). *Closed-form lining forces for an existing impermeable tunnel under groundwater drawdown.* Manuscript submitted.

This repository holds the code, input files and result files referred to by the manuscript's data availability statement. It contains no manuscript text. The snapshot corresponds to manuscript **v3.16**.

Everything in the paper — the inertness of the uniform mode, the reduction of the ovalizing mode to classical external loading, the third-harmonic correction, the exact solid-ring solution, the verification chain, the coupled hydro-mechanical (HM) finite element runs and the design chart — can be regenerated with open-source tools only ([OpenGeoSys 6](https://www.opengeosys.org), [ogstools](https://ogstools.opengeosys.org), [gmsh](https://gmsh.info), [SymPy](https://www.sympy.org), [PyVista](https://pyvista.org)).

## Quick start

```bash
# Linux / macOS / WSL
bash setup.sh          # creates .venv and installs pinned requirements (~1.5 GB of wheels, no compilation)
bash verify.sh         # runs the short verification chain (~5 min)
```

```powershell
# Windows (PowerShell); Python 3.11–3.14
powershell -ExecutionPolicy Bypass -File .\setup.ps1
powershell -ExecutionPolicy Bypass -File .\verify.ps1
```

All scripts resolve their inputs relative to the repository root, taken from the location of the
script itself; set `TD_ROOT` to override it if the tree is split up.

Every `*_results.json` and `*_sweep*.json` file is committed, so all figures and tables reproduce **without** re-running the finite element models. The OGS output folders (`runs/`) are not committed; each script recreates them.

## Where each result in the paper comes from

Section, figure and table numbers are those of manuscript v3.16.

| Paper item | Script / file | Runtime (2 cores) |
|---|---|---|
| §3.1 Uniform mode in an infinite medium, Eq. (4)–(5) | analytic; FE check `proj/drawdown_3b/drawdown_3b.py` → `dd_results.json` | ~7 min |
| §3.2 Proposition 1 (inertness), Eq. (6)–(8) | analytic; FE check `proj/drawdown_k0/drawdown_k0.py` → `k0_results.json` | ~5 min |
| §3.3 Proposition 2 (reduction), thin-shell coefficients Eq. (10)–(15) | `paper/theory/hoeg_check.py` → `hoeg_check.json` | < 1 min |
| Fig. 1 configurations (I)(II)(III); Fig. 2 theory map | `paper/figs/make_figs.py` (`fig_configs`, `fig_theory`) | < 1 min |
| Fig. 3, Table 1 drained uniform-mode response, (I) and (II) | `proj/drawdown_3b/`, `proj/drawdown_k0/` → `make_figs.py` (`fig_n0`) | – |
| §4.3.1 Fig. 4, Table 2 mode decomposition in the finite layer (III) | `proj/drawdown_3d/drawdown_3d.py` → `dd3d_results_nr3.json`, `n3_sweep_b.json` | ~4 min |
| §4.3.2, Table 4, Appendix A: exact *n* = 3 coefficients *a*<sub>T</sub>, *a*<sub>M</sub>, Eq. (16)–(17) | `paper/theory/n3_ring.py` → `n3_ring_sweep.json` | < 2 min |
| §4.3.1–4.3.2 Table 3 parameter sweep; coupled columns of Table 4 | `proj/drawdown_3d/drawdown_3d.py` → `n3_sweep_b.json`, `n3_sweep_zt10_zt20_R2p5.json` | ~40 min |
| Table 3 low-*F* row (*F* = 4.2, *E* = 8 MPa, *t* = 0.50 m) | `proj/drawdown_3d/run_lowF.py` → `lowF_result.json` | ~5 min |
| §4.3.3 Fig. 5 crown–invert pore-pressure difference | `make_figs.py` (`fig_pressure`) from `dd3d_results_nr3.json` | < 1 min |
| §4.4, Fig. 6 transient overshoot factor Θ; time-step convergence | `paper/sweeps/run_sweeps.py` → `sweep_results.json` (resumable) | ~45 min |
| §5 design procedure, Table 5 worked example (procedure column) | `paper/theory/exact_ring.py` (worked by hand from Eq. (10)–(17)) | < 1 min |
| Table 5 coupled column (metro tunnel, Busan) | `proj/drawdown_3d/run_example.py` → `example_result.json` | ~5 min |
| Fig. 7 design chart | `paper/theory/exact_ring.py` → `chart_data.json`; extended range by `extend_chart.py`; drawn by `make_figs.py` (`fig_chart`) | < 1 min |
| Appendix A exact plane-strain solid-ring solution; Table A.1 thin-shell error | `paper/theory/exact_ring.py` (SymPy) → `exact_ring_results.json` | < 1 min |
| Appendix B.1 mesh, time stepping, Taylor–Hood | `proj/*/` mesh generators and `.prj` templates | – |
| Appendix B.2 Terzaghi consolidation, mesh and time convergence | `proj/terzaghi/convergence.py` → `convergence_results.json` | ~3 min |
| Appendix B.2 Mandel–Cryer effect | `benchmarks/ogs/HydroMechanics/StaggeredScheme/MandelCryer/mandelcryer.py` | ~2 min |
| Appendix B.2, Table B.1 Einstein–Schwartz and external-loading checks | `proj/es_verif/es_verif.py` → `es_results.json`; thick-lining check `thick_check.py` | ~4 min |
| Appendix B.2 OGS HM regression set (10 cases) | `benchmarks/ogs/HydroMechanics/Verification/`, run by `verify.*` | ~1 min |
| All figures (vector PDF) | `paper/figs/make_figs.py` → `paper/figs/out/` | < 1 min |

## Layout

```
proj/            HM and dry finite element models (gmsh mesh generators, OGS .prj templates, post-processing)
paper/theory/    exact plane-strain solid-ring + medium solution (SymPy), thin-shell comparison, chart data
paper/sweeps/    time-step convergence and transient sweeps
paper/figs/      figure generation (all seven figures)
benchmarks/ogs/  OpenGeoSys HM benchmark inputs used in the verification chain (see licence note below)
```

## Software versions

Pinned in `requirements.txt`: OpenGeoSys 6.5.8 (pip wheel, MFront, serial), ogstools 0.8.2, gmsh 4.15.2, PyVista 0.49, SymPy 1.14, NumPy 2.x. Results in the paper were produced with these versions on Linux (x86-64) and reproduced on Windows 11 with Python 3.14.

## Modelling notes

* Taylor–Hood elements (quadratic displacement, linear pressure): meshes must be quadratic (QUAD8/9); the mesh generators do this.
* OGS HM offers only backward Euler in time. It is dissipative, so transient peaks are underestimated at coarse steps; the monotonicity result of §4.4 is therefore checked under time refinement (`run_sweeps.py`, P2 cases).
* Sign convention in the post-processing: depth enters as *z* = *z*<sub>t</sub> − *R* sin θ, so the invert sits at θ = −π/2. The third-harmonic ratios *T*<sub>3</sub>/*T*<sub>2</sub> and *M*<sub>3</sub>/*M*<sub>2</sub> are consequently negative, and the two modes add at the invert. See `lining_forces_half` in `proj/drawdown_3d/drawdown_3d.py`.
* Einstein & Schwartz (1979) solve a ring installed in a medium already under the in-situ stress; Burns & Richard (1964) / Höeg (1968) solve a stress-free ring loaded from infinity, which is the external-loading problem used throughout the paper. For the uniform mode the two differ by the factor 2(1−ν′). Using the wrong reference gives an apparent 27–70 % "error".
* The design chart is limited to *F* ≤ 10⁴. Above that, *C* = *F t*²/(12*R*²) rather than *F* governs the ovalizing thrust, and a single *F* no longer fixes the curve.

## Licence

Code and data in this repository: MIT Licence (see `LICENSE`).
The files under `benchmarks/ogs/` are copied from the OpenGeoSys test suite (`Tests/Data/HydroMechanics`), © OpenGeoSys Community, BSD 3-Clause licence.

## Citation

See `CITATION.cff`.
