"""
K0 ≠ 1 — laterally constrained (1D) far field under uniform drawdown Δu.

Same geometry as 3b, but the far lateral boundary is a roller (u_x = 0) so that the far
field is 1D vertical compression: Δσ'_v = Δu, Δσ'_h = K0' Δu, K0' = ν'/(1-ν').

Hand-derived predictions for the drained end state (impermeable lining, any stiffness):
    n=0 :  Δq0 = [2(1-ν') p_m - Δu] / (1+c*)  with p_m = Δu/(2(1-ν'))  ->  Δq0 = 0,  u_r0(R) = 0
    n=2 :  identical to the DRY external-loading problem with the same far field
           (water pressure is purely n=0), so T2, M2 must match a SMALL_DEFORMATION run.
Transient: record T(θ=0), T(θ=90), M2 history -> overshoot factor Θ (n=0 and n=2 diffuse
differently, so the total thrust at crown/springline may peak before the drained state).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "es_verif"))
sys.path.insert(0, str(HERE.parent / "drawdown_3b"))
import es_verif as ev
import drawdown_3b as dd

WORK = HERE / "runs"


def hm_prj_1d(c: dd.DDCase, t_end: float, pairs: str) -> str:
    """3b HM template with the far lateral boundary changed from free to roller."""
    s = dd.PRJ.format(E=repr(c.E), nu=repr(c.nu), El=repr(c.El), nul=repr(c.nul),
                      k=repr(c.k), mu=repr(c.mu), p_far=repr(-c.du),
                      t_end=repr(t_end), pairs=pairs)
    roller = ('                <boundary_condition><mesh>right</mesh><type>Dirichlet</type>'
              '<component>0</component><parameter>zero</parameter></boundary_condition>\n')
    anchor = '                <boundary_condition><mesh>sym_y</mesh><type>Dirichlet</type><component>1</component><parameter>zero</parameter></boundary_condition>\n'
    assert anchor in s
    return s.replace(anchor, anchor + roller)


def sd_prj_1d(c: dd.DDCase) -> str:
    """Dry external loading, same far field: top traction -Δu, right roller."""
    Ro = c.R + c.t / 2
    s = ev.PRJ.format(E=repr(c.E), nu=repr(c.nu), El=repr(c.El), nul=repr(c.nul),
                      trac_top=repr(-c.du), trac_right=repr(0.0),
                      initial_stress_tag="", Ro=repr(Ro), sxx="0.0", syy="0.0", szz="0.0")
    old = '<boundary_condition><mesh>right</mesh><type>Neumann</type><component>0</component><parameter>trac_right</parameter></boundary_condition>'
    new = '<boundary_condition><mesh>right</mesh><type>Dirichlet</type><component>0</component><parameter>zero</parameter></boundary_condition>'
    assert old in s
    return s.replace(old, new)


def lining_state(c: dd.DDCase, mesh: pv.UnstructuredGrid) -> dict:
    esc = c.as_es_case()
    th, T, M, _ = ev.lining_forces(esc, mesh)
    A = np.column_stack([np.ones_like(th), np.cos(2 * th)])
    T0, T2 = np.linalg.lstsq(A, T, rcond=None)[0]
    M2 = np.linalg.lstsq(np.cos(2 * th)[:, None], M, rcond=None)[0][0]

    def u_r(theta):
        pt = np.array([[c.R * np.cos(theta), c.R * np.sin(theta), 0.0]])
        uu = np.asarray(pv.PolyData(pt).sample(mesh, tolerance=1e-4).point_data["displacement"])[0]
        return float(uu[0] * np.cos(theta) + uu[1] * np.sin(theta))

    return dict(T0=float(T0), T2=float(T2), M2=float(M2),
                T_spring=float(T[0]), T_crown=float(T[-1]),
                u_r45=u_r(np.pi / 4), u_r0=u_r(1e-4), u_r90=u_r(np.pi / 2 - 1e-4))


def run_hm(c: dd.DDCase, d: Path, info: dict) -> dict:
    L = info["L"]
    T_ref = L**2 / c.c_v
    pairs, t_end = dd.time_pairs(T_ref)
    (d / "dd.prj").write_text(hm_prj_1d(c, t_end, pairs))
    out = d / "out_hm"
    out.mkdir()
    r = subprocess.run(["ogs", "-o", str(out.resolve()), "-l", "error", "dd.prj"],
                       cwd=str(d.resolve()), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr)[-800:])
    files = sorted(out.glob("dd_ts_*_t_*.vtu"),
                   key=lambda f: float(re.search(r"_t_([0-9.e+-]+)\.vtu", f.name).group(1)))
    hist = []
    for f in files:
        tt = float(re.search(r"_t_([0-9.e+-]+)\.vtu", f.name).group(1))
        st = lining_state(c, pv.read(f))
        st["t"] = tt
        st["t_over_Tref"] = tt / T_ref
        hist.append(st)
    # far-field check: pressure at corner nodes, sigma near (L/2, L/2)
    m = pv.read(files[-1])
    corners = set()
    for ci in range(m.n_cells):
        corners.update(int(i) for i in m.get_cell(ci).point_ids[:4])
    corners = np.array(sorted(corners))
    p_end = float(np.asarray(m.point_data["pressure"])[corners].mean())
    ff = pv.PolyData(np.array([[0.5 * L, 0.5 * L, 0.0]])).sample(m, tolerance=1e-4)
    s_ff = np.asarray(ff.point_data["sigma"])[0]     # effective stress [xx yy zz xy]
    return dict(T_ref=T_ref, hist=hist, p_end_over_du=p_end / c.du,
                ff_eff_xx_over_du=float(-s_ff[0] / c.du), ff_eff_yy_over_du=float(-s_ff[1] / c.du))


def run_sd(c: dd.DDCase, d: Path) -> dict:
    (d / "sd.prj").write_text(sd_prj_1d(c))
    out = d / "out_sd"
    out.mkdir()
    r = subprocess.run(["ogs", "-o", str(out.resolve()), "-l", "error", "sd.prj"],
                       cwd=str(d.resolve()), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr)[-800:])
    m = pv.read(sorted(out.glob("es_ts_*_t_*.vtu"))[-1])
    st = lining_state(c, m)
    L = 25.0 * (c.R + c.t / 2)
    ff = pv.PolyData(np.array([[0.5 * L, 0.5 * L, 0.0]])).sample(m, tolerance=1e-4)
    s_ff = np.asarray(ff.point_data["sigma"])[0]
    st["ff_xx_over_du"] = float(-s_ff[0] / c.du)
    st["ff_yy_over_du"] = float(-s_ff[1] / c.du)
    return st


def overshoot(hist, key):
    v = np.array([h[key] for h in hist])
    ref = v[-1]
    if abs(ref) < 1e-12:
        return np.nan, np.nan
    i = int(np.argmax(np.abs(v)))
    return float(abs(v[i]) / abs(ref)), float(hist[i]["t_over_Tref"])


def run_case(c: dd.DDCase) -> dict:
    d = WORK / c.name
    if d.exists():
        shutil.rmtree(d)
    info = ev.build_mesh(c.as_es_case(), d)
    hm = run_hm(c, d, info)
    sd = run_sd(c, d)
    end = hm["hist"][-1]
    K0p = c.nu / (1 - c.nu)
    s_dev = (1 - K0p) * c.du / 2                       # far-field deviatoric effective stress
    res = dict(name=c.name, C=c.C, c_star=c.c_star, F=c.as_es_case().F, K0p=K0p,
               T_ref=hm["T_ref"], p_end_over_du=hm["p_end_over_du"],
               ff_hm=dict(sxx=hm["ff_eff_xx_over_du"], syy=hm["ff_eff_yy_over_du"]),
               ff_sd=dict(sxx=sd["ff_xx_over_du"], syy=sd["ff_yy_over_du"]),
               n0=dict(dq0_over_du_hm=end["T0"] / c.R / c.du,
                       dq0_over_du_sd=sd["T0"] / c.R / c.du,
                       water_term=-1.0,                     # expected hm - sd  (loss of Δu on the lining)
                       u_r45_hm_over_uff=end["u_r45"] / c.u_ff_R,
                       u_r45_sd_over_uff=sd["u_r45"] / c.u_ff_R),
               n2=dict(T2_hm=end["T2"], T2_sd=sd["T2"], M2_hm=end["M2"], M2_sd=sd["M2"],
                       T2_over_sdevR_hm=end["T2"] / (s_dev * c.R),
                       M2_over_sdevR2_hm=end["M2"] / (s_dev * c.R**2)),
               hist=hm["hist"])
    for key in ("T_spring", "T_crown", "M2", "T2"):
        th_, tp = overshoot(hm["hist"], key)
        res[f"theta_{key}"] = th_
        res[f"tpeak_{key}"] = tp
    return res


if __name__ == "__main__":
    WORK.mkdir(exist_ok=True)
    cases = [dd.DDCase("El_25e9", El=25e9), dd.DDCase("El_2.5e9", El=2.5e9), dd.DDCase("El_5e8", El=5e8)]
    which = sys.argv[1:] or [c.name for c in cases]
    results = []
    for c in cases:
        if c.name not in which:
            continue
        r = run_case(c)
        results.append(r)
        n0, n2 = r["n0"], r["n2"]
        print(f"\n=== {c.name}  c*={r['c_star']:.3f}  F={r['F']:.0f}  K0'={r['K0p']:.3f}  p_end={r['p_end_over_du']:+.4f}Δu")
        print(f"  far field eff. stress /Δu   HM: sxx={r['ff_hm']['sxx']:+.4f} syy={r['ff_hm']['syy']:+.4f}   "
              f"SD: sxx={r['ff_sd']['sxx']:+.4f} syy={r['ff_sd']['syy']:+.4f}   (expect sxx=K0'={r['K0p']:.3f}, syy=1)")
        print(f"  n=0  Δq0/Δu   HM={n0['dq0_over_du_hm']:+.4f}  SD(dry)={n0['dq0_over_du_sd']:+.4f}  "
              f"HM−SD={n0['dq0_over_du_hm']-n0['dq0_over_du_sd']:+.4f} (expect −1: lost water)   pred HM=0")
        print(f"       u_r45/u_ff  HM={n0['u_r45_hm_over_uff']:+.4f}  SD={n0['u_r45_sd_over_uff']:+.4f}   pred HM=0")
        print(f"  n=2  T2 [kN/m]  HM={n2['T2_hm']/1e3:9.3f}  SD={n2['T2_sd']/1e3:9.3f}  ({100*(n2['T2_hm']-n2['T2_sd'])/n2['T2_sd']:+.2f}%)   "
              f"M2 [kNm/m] HM={n2['M2_hm']/1e3:8.4f} SD={n2['M2_sd']/1e3:8.4f} ({100*(n2['M2_hm']-n2['M2_sd'])/n2['M2_sd']:+.2f}%)")
        print(f"       normalised  T2/(s_dev R)={n2['T2_over_sdevR_hm']:.4f}   M2/(s_dev R²)={n2['M2_over_sdevR2_hm']:.4f}")
        print(f"  Θ   T_spring={r['theta_T_spring']:.4f} (t/Tref={r['tpeak_T_spring']:.3f})  "
              f"T_crown={r['theta_T_crown']:.4f} ({r['tpeak_T_crown']:.3f})  "
              f"M2={r['theta_M2']:.4f} ({r['tpeak_M2']:.3f})  T2={r['theta_T2']:.4f}", flush=True)
    (HERE / "k0_results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\nsaved k0_results.json")
