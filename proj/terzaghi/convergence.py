"""
Terzaghi 1D consolidation: mesh + time-step convergence study against the exact series.

Verification chain items #1 (Terzaghi) and #5 (discretisation convergence).
Doubles as the prototype for the parametric sweep machinery of the tunnel study.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from lxml import etree

# ----------------------------------------------------------------------------
# problem definition (mirrors terzaghi.prj)
# ----------------------------------------------------------------------------
H, E, NU = 10.0, 10.0e6, 0.2
K, MU, SIG = 1.0e-14, 1.0e-3, 1.0e5

M = E * (1.0 - NU) / ((1.0 + NU) * (1.0 - 2.0 * NU))   # oedometric modulus
CV = (K / MU) * M                                      # consolidation coefficient
S_INF = SIG * H / M

ROOT = Path(__file__).parent
TEMPLATE = ROOT / "terzaghi.prj"
WORK = ROOT / "conv"

# compare only where the initial-condition singularity has been smoothed out
TV_MIN, TV_MAX = 0.01, 2.0


# ----------------------------------------------------------------------------
# analytical solution
# ----------------------------------------------------------------------------
def p_exact(zbar, Tv, n_terms: int = 500):
    """Excess pore pressure / p0.  zbar = z/H measured from the DRAINED surface."""
    Mm = (2 * np.arange(n_terms) + 1) * np.pi / 2.0
    zb = np.atleast_1d(zbar)[:, None]
    return ((2.0 / Mm) * np.sin(Mm * zb) * np.exp(-(Mm**2) * Tv)).sum(axis=1)


def U_exact(Tv, n_terms: int = 500):
    """Degree of consolidation."""
    Mm = (2 * np.arange(n_terms) + 1) * np.pi / 2.0
    return 1.0 - (2.0 / Mm**2 * np.exp(-(Mm**2) * np.atleast_1d(Tv)[:, None])).sum(axis=1)


# ----------------------------------------------------------------------------
# mesh
# ----------------------------------------------------------------------------
def make_mesh(n_cells: int, target: Path) -> None:
    """Quadratic (QUAD8) column mesh, n_cells elements over the height H."""
    from ogstools import Meshes
    from ogstools.gmsh_tools import rect

    target.mkdir(parents=True, exist_ok=True)
    msh = target / "column.msh"
    rect(lengths=(1.0, H), n_edge_cells=(1, n_cells), structured_grid=True,
         order=2, out_name=msh)
    ms = Meshes.from_gmsh(msh, dim=2, reindex=True, log=False)
    dom = ms["domain"]
    dom.cell_data["MaterialIDs"] = np.zeros(dom.n_cells, dtype=np.int32)
    ms.save(target, overwrite=True)


# ----------------------------------------------------------------------------
# project file templating
# ----------------------------------------------------------------------------
def write_prj(target: Path, time_refine: int) -> Path:
    """Copy the template, splitting every time step into `time_refine` substeps.

    Output is thinned by the same factor so every run reports at identical
    physical times, which makes the error metric comparable across runs.
    """
    tree = etree.parse(str(TEMPLATE))
    root = tree.getroot()

    for pair in root.findall(".//time_stepping/timesteps/pair"):
        rep = pair.find("repeat")
        dt = pair.find("delta_t")
        rep.text = str(int(rep.text) * time_refine)
        dt.text = repr(float(dt.text) / time_refine)

    for pair in root.findall(".//output/timesteps/pair"):
        pair.find("each_steps").text = str(time_refine)

    out = target / "terzaghi.prj"
    tree.write(str(out), xml_declaration=True, encoding="ISO-8859-1")
    return out


# ----------------------------------------------------------------------------
# run + error metrics
# ----------------------------------------------------------------------------
def corner_nodes(mesh: pv.UnstructuredGrid) -> np.ndarray:
    """Node ids carrying a pressure DOF (QUAD8 corners = first 4 of each cell)."""
    ids = set()
    for c in range(mesh.n_cells):
        ids.update(int(i) for i in mesh.get_cell(c).point_ids[:4])
    return np.array(sorted(ids))


def run_case(n_cells: int, time_refine: int) -> dict:
    tag = f"n{n_cells:04d}_r{time_refine}"
    d = WORK / tag
    if d.exists():
        shutil.rmtree(d)
    make_mesh(n_cells, d)
    prj = write_prj(d, time_refine)

    out = d / "out"
    out.mkdir(exist_ok=True)
    r = subprocess.run(["ogs", "-o", str(out), "-l", "error", prj.name],
                       cwd=d, capture_output=True, text=True)
    if r.returncode != 0:
        return {"tag": tag, "n": n_cells, "r": time_refine, "ok": False,
                "msg": (r.stdout + r.stderr)[-400:]}

    files = sorted(out.glob("terzaghi_ts_*_t_*.vtu"),
                   key=lambda f: float(re.search(r"_t_([0-9.]+)\.vtu", f.name).group(1)))
    m0 = pv.read(files[0])
    pts = m0.points
    base = int(np.argmin(np.linalg.norm(pts - np.array([0.0, 0.0, 0.0]), axis=1)))
    topn = int(np.argmin(np.linalg.norm(pts - np.array([0.0, H, 0.0]), axis=1)))
    corners = corner_nodes(m0)

    t, pb, se = [], [], []
    prof_err = 0.0
    for f in files:
        tt = float(re.search(r"_t_([0-9.]+)\.vtu", f.name).group(1))
        m = pv.read(f)
        t.append(tt)
        pb.append(float(m.point_data["pressure"][base]))
        se.append(-float(m.point_data["displacement"][topn, 1]))
        Tv = CV * tt / H**2
        if TV_MIN <= Tv <= TV_MAX:
            x, y = m.points[corners, 0], m.points[corners, 1]
            col = np.abs(x) < 1e-9
            zb = (H - y[col]) / H
            pn = m.point_data["pressure"][corners][col] / SIG
            o = np.argsort(zb)
            prof_err = max(prof_err, float(np.abs(pn[o] - p_exact(zb[o], Tv)).max()))

    t = np.array(t)
    Tv = CV * t / H**2
    sel = (Tv >= TV_MIN) & (Tv <= TV_MAX)

    ep = np.abs(np.array(pb)[sel] / SIG - np.array([p_exact(1.0, T)[0] for T in Tv[sel]]))
    eu = np.abs(np.array(se)[sel] / S_INF - U_exact(Tv[sel]))

    return {"tag": tag, "n": n_cells, "r": time_refine, "ok": True,
            "h": H / n_cells, "nsteps": len(files),
            "err_p_base": float(ep.max()), "err_U": float(eu.max()),
            "err_profile": prof_err,
            "s_final_ratio": float(se[-1] / S_INF)}


def order(errs, hs):
    """Observed convergence order between successive refinements."""
    e, h = np.asarray(errs, float), np.asarray(hs, float)
    return np.r_[np.nan, np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])]


if __name__ == "__main__":
    WORK.mkdir(exist_ok=True)
    print(f"M = {M:.6e} Pa    c_v = {CV:.6e} m2/s    H^2/c_v = {H**2/CV:.4e} s")
    print(f"s_inf = {S_INF:.6f} m    comparison window T_v in [{TV_MIN}, {TV_MAX}]\n")

    print("=" * 78)
    print("MESH REFINEMENT  (time refinement fixed at r = 4)")
    print("=" * 78)
    mesh_series = [run_case(n, 4) for n in (20, 40, 80, 160)]
    hs = [c["h"] for c in mesh_series if c["ok"]]
    ou = order([c["err_U"] for c in mesh_series if c["ok"]], hs)
    op = order([c["err_p_base"] for c in mesh_series if c["ok"]], hs)
    print(f"{'n':>5} {'h [m]':>8} {'steps':>6} {'max|dU|':>11} {'ord':>6} "
          f"{'max|dp/p0|':>12} {'ord':>6} {'max|dprofile|':>14}")
    for i, c in enumerate([c for c in mesh_series if c["ok"]]):
        print(f"{c['n']:5d} {c['h']:8.4f} {c['nsteps']:6d} {c['err_U']:11.3e} "
              f"{ou[i]:6.2f} {c['err_p_base']:12.3e} {op[i]:6.2f} {c['err_profile']:14.3e}")
    for c in mesh_series:
        if not c["ok"]:
            print(f"  FAILED {c['tag']}: {c['msg']}")

    print()
    print("=" * 78)
    print("TIME-STEP REFINEMENT  (mesh fixed at n = 160)")
    print("=" * 78)
    time_series = []
    for r in (1, 2, 4, 8):
        c = next((x for x in mesh_series if x["n"] == 160 and x["r"] == r), None)
        time_series.append(c if c else run_case(160, r))
    dts = [1.0 / c["r"] for c in time_series if c["ok"]]
    ou = order([c["err_U"] for c in time_series if c["ok"]], dts)
    op = order([c["err_p_base"] for c in time_series if c["ok"]], dts)
    print(f"{'r':>3} {'steps':>6} {'max|dU|':>11} {'ord':>6} "
          f"{'max|dp/p0|':>12} {'ord':>6} {'s_end/s_inf':>13}")
    for i, c in enumerate([c for c in time_series if c["ok"]]):
        print(f"{c['r']:3d} {c['nsteps']:6d} {c['err_U']:11.3e} {ou[i]:6.2f} "
              f"{c['err_p_base']:12.3e} {op[i]:6.2f} {c['s_final_ratio']:13.6f}")
    for c in time_series:
        if not c["ok"]:
            print(f"  FAILED {c['tag']}: {c['msg']}")

    import json
    (ROOT / "convergence_results.json").write_text(
        json.dumps({"mesh": mesh_series, "time": time_series}, indent=2))
    print("\nsaved convergence_results.json")
