"""Infinite-medium external-loading FE vs exact solid-ring solution for thicker linings,
with both extraction methods (linear / quadratic through-thickness fit) and two radial
element counts.  Settles whether the finite-layer T2 deficit is post-processing or physics."""
import json, sys, subprocess, shutil
from pathlib import Path
import numpy as np, pyvista as pv
sys.path.insert(0, f"{ROOT}/proj/es_verif"); sys.path.insert(0, f"{ROOT}/paper/theory")
import es_verif as ev
from exact_ring import solve_external
import os as _os, pathlib as _pl
# Repository root: override with TD_ROOT if the tree is moved.
ROOT = _os.environ.get("TD_ROOT") or str(_pl.Path(__file__).resolve().parents[2])

base = dict(E=48e6, nu=0.34, K=0.5152, P=100e3, El=25e9, nul=0.15, R=5.0, loading="external")
cases = [ev.Case("T_t0.25_nr3", **base, t=0.25, n_r=3),
         ev.Case("T_t0.25_nr6", **base, t=0.25, n_r=6),
         ev.Case("T_t0.50_nr3", **base, t=0.50, n_r=3),
         ev.Case("T_t0.50_nr6", **base, t=0.50, n_r=6)]


def extract(c, mesh, order):
    Ri = c.R - c.t / 2
    r = np.linspace(Ri + 0.03 * c.t, Ri + 0.70 * c.t, 15)
    th = np.linspace(0.0, np.pi / 2, 19); th[0] += 1e-4; th[-1] -= 1e-4
    TH, RR = np.meshgrid(th, r, indexing="ij")
    pts = np.column_stack([(RR * np.cos(TH)).ravel(), (RR * np.sin(TH)).ravel(), np.zeros(RR.size)])
    pr = pv.PolyData(pts).sample(mesh, tolerance=1e-4)
    s_ = np.asarray(pr.point_data["sigma"]); ct, st = np.cos(TH.ravel()), np.sin(TH.ravel())
    s_tt = (s_[:, 0] * st**2 + s_[:, 1] * ct**2 - 2 * s_[:, 3] * st * ct).reshape(TH.shape)
    cols = [np.ones_like(r), r - c.R] + ([(r - c.R) ** 2] if order == 2 else [])
    A = np.column_stack(cols)
    T = np.empty(len(th)); M = np.empty(len(th))
    for i in range(len(th)):
        co = np.linalg.lstsq(A, s_tt[i], rcond=None)[0]
        a, b = co[0], co[1]; cq = co[2] if order == 2 else 0.0
        T[i] = -(a * c.t + cq * c.t**3 / 12); M[i] = b * c.t**3 / 12
    A2 = np.column_stack([np.ones_like(th), np.cos(2 * th)])
    T0, T2 = np.linalg.lstsq(A2, T, rcond=None)[0]
    M2 = np.linalg.lstsq(np.cos(2 * th)[:, None], M, rcond=None)[0][0]
    return T0, T2, M2, s_tt


out = {}
for c in cases:
    d = ev.WORK / c.name
    if not (d / "out").exists():
        r = ev.run_case(c)
        if not r["ok"]:
            print(c.name, "FAILED", r["msg"][-300:]); continue
    vtu = sorted((d / "out").glob("es_ts_*_t_*.vtu"))[-1]; mesh = pv.read(vtu)
    ex = solve_external(c.E, c.nu, c.El, c.nul, c.R - c.t / 2, c.R + c.t / 2, -(1 + c.K) * c.P / 2, (1 - c.K) * c.P / 2)
    row = {}
    for order in (1, 2):
        T0, T2, M2, s_tt = extract(c, mesh, order)
        row[f"lin{order}"] = dict(T0=T0, T2=T2, M2=M2)
        print(f"{c.name:12s} fit order {order}:  T0 {T0/1e3:8.2f} ({100*(T0/ex['T0']-1):+5.1f}%)  T2 {T2/1e3:8.2f} ({100*(T2/ex['T2']-1):+5.1f}%)  M2 {M2/1e3:8.3f} ({100*(M2/ex['M2']-1):+5.1f}%)")
    # full-thickness integration by trapezoid over all 15 samples extrapolated? -> also report raw profile at springline
    row["exact"] = dict(T0=ex["T0"], T2=ex["T2"], M2=ex["M2"])
    out[c.name] = row
json.dump(out, open(Path(__file__).with_name("thick_check.json"), "w"), indent=1)
