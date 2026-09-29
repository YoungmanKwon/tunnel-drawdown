"""
3d — clay layer over a pumped aquifer: depth-varying drawdown around an existing tunnel.

Half-plane model (x >= 0).  Clay layer 0 >= y >= -H, free drained surface (traction 0, p = 0),
rigid base with prescribed aquifer drawdown p = -Δu·r(t), lateral roller/no-flow at x = W.
Impermeable elastic lining (Biot 0) centred at (0, -z_t).  Final Δp(y) = Δu·y/H (linear):
the tunnel sees a pressure difference 2RΔu/H between crown and invert -> n = 1 mode.

A free-field run (lining material = ground, permeable) on the same mesh gives the
1D consolidation reference: s_ff(z_t) = Δu (H² - z_t²) / (2 H M).

Outputs per case: Fourier modes of lining thrust/moment (n = 0, 1, 2), tunnel settlement
vs free field (I_s), interaction/delay (Ψ) and transient overshoot Θ of crown/invert/springline
thrust and moment.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent
WORK = HERE / "runs"


@dataclass
class Case3d:
    name: str
    E: float = 48e6
    nu: float = 0.34
    k: float = 1e-12
    mu: float = 1e-3
    El: float = 25e9
    nul: float = 0.15
    t: float = 0.25
    R: float = 5.0            # lining centreline radius
    H: float = 40.0
    z_t: float = 15.0         # tunnel axis depth
    W: float = 120.0
    du: float = 100e3
    ramp: float = 0.0         # drawdown ramp duration t_d [s]; 0 = step
    free_field: bool = False  # lining -> ground material (no tunnel)
    n_theta: int = 72         # elements along the half arc (180°)
    n_r: int = 6             # elements through the lining thickness (3 under-resolves T2 by ~4 % at t/R = 0.05)
    h_far: float = 2.5
    n_decades: float = 3.5
    per_decade: int = 8

    @property
    def Ri(self): return self.R - self.t / 2
    @property
    def Ro(self): return self.R + self.t / 2
    @property
    def M_oed(self): return self.E * (1 - self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))
    @property
    def c_v(self): return self.k / self.mu * self.M_oed
    @property
    def T_ref(self): return self.H**2 / self.c_v
    @property
    def K0p(self): return self.nu / (1 - self.nu)
    @property
    def C(self): return self.E * self.R * (1 - self.nul**2) / (self.El * self.t * (1 - self.nu**2))
    @property
    def F(self): return self.E * self.R**3 * (1 - self.nul**2) / (self.El * self.t**3 / 12 * (1 - self.nu**2))
    @property
    def s_ff_surface(self): return self.du * self.H / (2 * self.M_oed)
    @property
    def s_ff_tunnel(self): return self.du * (self.H**2 - self.z_t**2) / (2 * self.H * self.M_oed)
    @property
    def du_local(self): return self.du * self.z_t / self.H      # Δu at tunnel axis depth (final)


# ----------------------------------------------------------------------------
# mesh
# ----------------------------------------------------------------------------
def build_mesh(c: Case3d, d: Path) -> dict:
    import gmsh
    from ogstools import Meshes

    Ri, Ro, zt, H, W = c.Ri, c.Ro, c.z_t, c.H, c.W
    h_lin = np.pi * Ro / c.n_theta

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("dd3d")
    g = gmsh.model.geo
    O = g.addPoint(0, -zt, 0)
    A, B, Cc, D = g.addPoint(0, 0, 0), g.addPoint(W, 0, 0), g.addPoint(W, -H, 0), g.addPoint(0, -H, 0)
    Eo, Fo, Go = g.addPoint(0, -zt + Ro, 0), g.addPoint(Ro, -zt, 0), g.addPoint(0, -zt - Ro, 0)
    Ei, Fi, Gi = g.addPoint(0, -zt + Ri, 0), g.addPoint(Ri, -zt, 0), g.addPoint(0, -zt - Ri, 0)

    top, right, bottom = g.addLine(A, B), g.addLine(B, Cc), g.addLine(Cc, D)
    gnd_sym_dn, gnd_sym_up = g.addLine(D, Go), g.addLine(Eo, A)
    lin_sym_up, lin_sym_dn = g.addLine(Ei, Eo), g.addLine(Go, Gi)
    a1o, a2o = g.addCircleArc(Eo, O, Fo), g.addCircleArc(Fo, O, Go)
    a1i, a2i = g.addCircleArc(Ei, O, Fi), g.addCircleArc(Fi, O, Gi)

    ground = g.addPlaneSurface([g.addCurveLoop([top, right, bottom, gnd_sym_dn, -a2o, -a1o, gnd_sym_up])])
    lining = g.addPlaneSurface([g.addCurveLoop([a1i, a2i, -lin_sym_dn, -a2o, -a1o, -lin_sym_up])])

    nq = c.n_theta // 2 + 1
    for a in (a1o, a2o, a1i, a2i):
        g.mesh.setTransfiniteCurve(a, nq)
    for l in (lin_sym_up, lin_sym_dn):
        g.mesh.setTransfiniteCurve(l, c.n_r + 1)
    g.mesh.setTransfiniteSurface(lining, cornerTags=[Ei, Gi, Go, Eo])
    g.mesh.setRecombine(2, lining)
    g.mesh.setRecombine(2, ground)
    g.synchronize()

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [a1o, a2o])
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 300)
    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", h_lin)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", c.h_far)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.6 * Ro)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax", 5.0 * Ro)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)
    for opt, v in (("Mesh.MeshSizeExtendFromBoundary", 0), ("Mesh.MeshSizeFromPoints", 0),
                   ("Mesh.MeshSizeFromCurvature", 0), ("Mesh.Algorithm", 8), ("Mesh.RecombinationAlgorithm", 1)):
        gmsh.option.setNumber(opt, v)

    gmsh.model.addPhysicalGroup(2, [ground], name="ground")
    gmsh.model.addPhysicalGroup(2, [lining], name="lining")
    gmsh.model.addPhysicalGroup(1, [top], name="top")
    gmsh.model.addPhysicalGroup(1, [right], name="right")
    gmsh.model.addPhysicalGroup(1, [bottom], name="bottom")
    gmsh.model.addPhysicalGroup(1, [gnd_sym_up, gnd_sym_dn, lin_sym_up, lin_sym_dn], name="sym_x")
    gmsh.model.addPhysicalGroup(1, [a1i, a2i], name="inner")
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    d.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(d / "dd3d.msh"))
    gmsh.finalize()

    ms = Meshes.from_gmsh(d / "dd3d.msh", dim=2, reindex=True, log=False)
    dom = ms["domain"]
    cc = dom.cell_centers().points
    rc = np.hypot(cc[:, 0], cc[:, 1] + zt)
    mid = np.where(rc < Ro + 1e-6, 1, 0).astype(np.int32)
    dom.cell_data["MaterialIDs"] = mid
    ms.save(d, overwrite=True)
    return dict(n_pts=int(dom.n_points), n_cells=int(dom.n_cells), n_lining=int((mid == 1).sum()),
                subdomains=sorted(ms.keys()))


# ----------------------------------------------------------------------------
# project file
# ----------------------------------------------------------------------------
PRJ = """<?xml version="1.0" encoding="ISO-8859-1"?>
<OpenGeoSysProject>
    <meshes>
        <mesh>domain.vtu</mesh><mesh>inner.vtu</mesh><mesh>sym_x.vtu</mesh>
        <mesh>top.vtu</mesh><mesh>right.vtu</mesh><mesh>bottom.vtu</mesh>
    </meshes>
    <processes>
        <process>
            <name>HM</name>
            <type>HYDRO_MECHANICS</type>
            <integration_order>3</integration_order>
            <constitutive_relation id="0">
                <type>LinearElasticIsotropic</type>
                <youngs_modulus>E_ground</youngs_modulus><poissons_ratio>nu_ground</poissons_ratio>
            </constitutive_relation>
            <constitutive_relation id="1">
                <type>LinearElasticIsotropic</type>
                <youngs_modulus>E_lining</youngs_modulus><poissons_ratio>nu_lining</poissons_ratio>
            </constitutive_relation>
            <process_variables><pressure>pressure</pressure><displacement>displacement</displacement></process_variables>
            <secondary_variables>
                <secondary_variable internal_name="sigma" output_name="sigma"/>
            </secondary_variables>
            <specific_body_force>0 0</specific_body_force>
        </process>
    </processes>
    <media>
        <medium id="0">
            <phases>
                <phase><type>AqueousLiquid</type><properties>
                    <property><name>viscosity</name><type>Constant</type><value>{mu}</value></property>
                    <property><name>density</name><type>Constant</type><value>1000</value></property></properties></phase>
                <phase><type>Solid</type><properties>
                    <property><name>density</name><type>Constant</type><value>2000</value></property></properties></phase>
            </phases>
            <properties>
                <property><name>permeability</name><type>Constant</type><value>{k}</value></property>
                <property><name>porosity</name><type>Constant</type><value>0</value></property>
                <property><name>biot_coefficient</name><type>Constant</type><value>1.0</value></property>
                <property><name>reference_temperature</name><type>Constant</type><value>293.15</value></property>
            </properties>
        </medium>
        <medium id="1">
            <phases>
                <phase><type>AqueousLiquid</type><properties>
                    <property><name>viscosity</name><type>Constant</type><value>{mu}</value></property>
                    <property><name>density</name><type>Constant</type><value>1000</value></property></properties></phase>
                <phase><type>Solid</type><properties>
                    <property><name>density</name><type>Constant</type><value>2500</value></property></properties></phase>
            </phases>
            <properties>
                <property><name>permeability</name><type>Constant</type><value>{k_lin}</value></property>
                <property><name>porosity</name><type>Constant</type><value>0</value></property>
                <property><name>biot_coefficient</name><type>Constant</type><value>{biot_lin}</value></property>
                <property><name>reference_temperature</name><type>Constant</type><value>293.15</value></property>
            </properties>
        </medium>
    </media>
    <time_loop>
        <processes>
            <process ref="HM">
                <nonlinear_solver>basic_newton</nonlinear_solver>
                <convergence_criterion>
                    <type>PerComponentDeltaX</type><norm_type>NORM2</norm_type>
                    <abstols>1e-3 1e-11 1e-11</abstols>
                </convergence_criterion>
                <time_discretization><type>BackwardEuler</type></time_discretization>
                <time_stepping>
                    <type>FixedTimeStepping</type>
                    <t_initial>0</t_initial><t_end>{t_end}</t_end>
                    <timesteps>
{pairs}
                    </timesteps>
                </time_stepping>
            </process>
        </processes>
        <output>
            <type>VTK</type><prefix>dd3d</prefix>
            <timesteps><pair><repeat>10000</repeat><each_steps>1</each_steps></pair></timesteps>
            <variables><variable>pressure</variable><variable>displacement</variable><variable>sigma</variable></variables>
            <suffix>_ts_{{:timestep}}_t_{{:time}}</suffix>
        </output>
    </time_loop>
    <parameters>
        <parameter><name>E_ground</name><type>Constant</type><value>{E}</value></parameter>
        <parameter><name>nu_ground</name><type>Constant</type><value>{nu}</value></parameter>
        <parameter><name>E_lining</name><type>Constant</type><value>{El}</value></parameter>
        <parameter><name>nu_lining</name><type>Constant</type><value>{nul}</value></parameter>
        <parameter><name>zero</name><type>Constant</type><value>0</value></parameter>
        <parameter><name>u0</name><type>Constant</type><values>0 0</values></parameter>
        <parameter><name>p0</name><type>Constant</type><value>0</value></parameter>
        <parameter>
            <name>p_base</name>
            <type>Function</type>
            <expression>{p_base_expr}</expression>
        </parameter>
    </parameters>
    <process_variables>
        <process_variable>
            <name>displacement</name><components>2</components><order>2</order>
            <initial_condition>u0</initial_condition>
            <boundary_conditions>
                <boundary_condition><mesh>sym_x</mesh><type>Dirichlet</type><component>0</component><parameter>zero</parameter></boundary_condition>
                <boundary_condition><mesh>right</mesh><type>Dirichlet</type><component>0</component><parameter>zero</parameter></boundary_condition>
                <boundary_condition><mesh>bottom</mesh><type>Dirichlet</type><component>1</component><parameter>zero</parameter></boundary_condition>
            </boundary_conditions>
        </process_variable>
        <process_variable>
            <name>pressure</name><components>1</components><order>1</order>
            <initial_condition>p0</initial_condition>
            <boundary_conditions>
                <boundary_condition><mesh>top</mesh><type>Dirichlet</type><component>0</component><parameter>zero</parameter></boundary_condition>
                <boundary_condition><mesh>bottom</mesh><type>Dirichlet</type><component>0</component><parameter>p_base</parameter></boundary_condition>
            </boundary_conditions>
        </process_variable>
    </process_variables>
    <nonlinear_solvers>
        <nonlinear_solver><name>basic_newton</name><type>Newton</type><max_iter>20</max_iter>
            <linear_solver>general_linear_solver</linear_solver></nonlinear_solver>
    </nonlinear_solvers>
    <linear_solvers>
        <linear_solver><name>general_linear_solver</name>
            <eigen><solver_type>SparseLU</solver_type><scaling>true</scaling></eigen></linear_solver>
    </linear_solvers>
</OpenGeoSysProject>
"""


def time_pairs(c: Case3d):
    """Geometric grid; if a ramp is present, resolve it with >= 10 steps first."""
    T = c.T_ref
    pairs = []
    t = 0.0
    if c.ramp > 0:
        n_ramp = 12
        pairs += [(n_ramp, c.ramp / n_ramp)]
        t += c.ramp
        dt = c.ramp / n_ramp
    else:
        dt = 1e-4 * T
    ratio = 10 ** (1.0 / c.per_decade)
    while t < c.n_decades and t < 10 ** (c.n_decades - 4) * T * 1e4:   # loop guard below
        break
    n = int(round(c.n_decades * c.per_decade))
    for _ in range(n):
        pairs.append((1, dt))
        t += dt
        dt *= ratio
    xml = "\n".join(f"                        <pair><repeat>{r}</repeat><delta_t>{d!r}</delta_t></pair>" for r, d in pairs)
    return xml, t


def write_prj(c: Case3d, d: Path):
    pairs, t_end = time_pairs(c)
    if c.ramp > 0:
        expr = f"if (t &lt; {c.ramp!r}) -{c.du!r}*t/{c.ramp!r}; else -{c.du!r}"
    else:
        expr = f"-{c.du!r}"
    if c.free_field:
        El, nul, k_lin, biot = c.E, c.nu, c.k, 1.0
    else:
        El, nul, k_lin, biot = c.El, c.nul, 1e-30, 0.0
    (d / "dd3d.prj").write_text(PRJ.format(E=repr(c.E), nu=repr(c.nu), El=repr(El), nul=repr(nul),
                                           k=repr(c.k), k_lin=repr(k_lin), biot_lin=repr(biot), mu=repr(c.mu),
                                           t_end=repr(t_end), pairs=pairs, p_base_expr=expr))
    return t_end


# ----------------------------------------------------------------------------
# post-processing
# ----------------------------------------------------------------------------
def corner_ids(mesh):
    ids = set()
    for ci in range(mesh.n_cells):
        ids.update(int(i) for i in mesh.get_cell(ci).point_ids[:4])
    return np.array(sorted(ids))


def lining_forces_half(c: Case3d, mesh, n_theta_samp: int = 37, n_r_samp: int = 15, fit_fraction: float = 0.70):
    """θ from the springline, θ∈(-90°, 90°): invert = -90°, crown = +90°.  T compression +,
    M + = inner fibre more compressed."""
    th = np.linspace(-np.pi / 2, np.pi / 2, n_theta_samp)
    th[0] += 1e-4
    th[-1] -= 1e-4
    r = np.linspace(c.Ri + 0.03 * c.t, c.Ri + fit_fraction * c.t, n_r_samp)
    TH, RR = np.meshgrid(th, r, indexing="ij")
    pts = np.column_stack([(RR * np.cos(TH)).ravel(), -c.z_t + (RR * np.sin(TH)).ravel(), np.zeros(RR.size)])
    probe = pv.PolyData(pts).sample(mesh, tolerance=1e-4)
    if not np.asarray(probe.point_data["vtkValidPointMask"]).all():
        raise RuntimeError("lining sample points outside mesh")
    s = np.asarray(probe.point_data["sigma"])
    ct, st = np.cos(TH.ravel()), np.sin(TH.ravel())
    s_tt = (s[:, 0] * st**2 + s[:, 1] * ct**2 - 2 * s[:, 3] * st * ct).reshape(TH.shape)
    # quadratic through-thickness fit (a linear fit extrapolated to r=R biases T by -2.7 % at t/R=0.05
    # and -12.6 % at t/R=0.10; see es_verif.lining_forces)
    A = np.column_stack([np.ones_like(r), r - c.R, (r - c.R) ** 2])
    T = np.empty(len(th)); M = np.empty(len(th))
    for i in range(len(th)):
        a, b, cq = np.linalg.lstsq(A, s_tt[i], rcond=None)[0]
        T[i] = -(a * c.t + cq * c.t**3 / 12)
        M[i] = b * c.t**3 / 12
    return th, T, M


def fourier(th, y):
    """y ≈ a0 + a1 sinθ + a2 cos2θ + a3 sin3θ + a4 cos4θ  (x-symmetric half model)."""
    A = np.column_stack([np.ones_like(th), np.sin(th), np.cos(2 * th), np.sin(3 * th), np.cos(4 * th)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def state(c: Case3d, mesh, corners) -> dict:
    out = {}
    if not c.free_field:
        th, T, M = lining_forces_half(c, mesh)
        cT, cM = fourier(th, T), fourier(th, M)
        i_inv, i_spr, i_crn = 0, len(th) // 2, len(th) - 1
        out.update(T0=cT[0], T1=cT[1], T2=cT[2], T3=cT[3], T4=cT[4],
                   M0=cM[0], M1=cM[1], M2=cM[2], M3=cM[3], M4=cM[4],
                   T_inv=T[i_inv], T_spr=T[i_spr], T_crn=T[i_crn],
                   M_inv=M[i_inv], M_spr=M[i_spr], M_crn=M[i_crn])
    # displacements: lining centreline crown/invert/springline, and centre (mean of crown & invert)
    def u_at(x, y):
        q = pv.PolyData(np.array([[x, y, 0.0]])).sample(mesh, tolerance=1e-4)
        return np.asarray(q.point_data["displacement"])[0]
    u_c, u_i, u_s = u_at(0.0, -c.z_t + c.R), u_at(0.0, -c.z_t - c.R), u_at(c.R, -c.z_t)
    out.update(uy_crown=float(u_c[1]), uy_invert=float(u_i[1]), ux_spring=float(u_s[0]),
               uy_centre=float(0.5 * (u_c[1] + u_i[1])),
               oval=float(0.5 * (u_c[1] - u_i[1])),                # vertical diameter change / 2 (neg = squash)
               uy_surface_x0=float(u_at(0.0, 0.0)[1]),
               uy_surface_far=float(u_at(0.9 * c.W, 0.0)[1]))
    # pore pressure at corner nodes: at tunnel axis depth far away, and just above crown / below invert
    P = mesh.points[corners]
    p = np.asarray(mesh.point_data["pressure"])[corners]
    def p_near(x, y, tol=0.6):
        sel = (np.abs(P[:, 0] - x) < tol) & (np.abs(P[:, 1] - y) < tol)
        return float(p[sel].mean()) if sel.any() else np.nan
    out.update(p_far_axis=p_near(0.8 * c.W, -c.z_t, 1.5),
               p_crown=p_near(0.0, -c.z_t + c.Ro + 0.3), p_invert=p_near(0.0, -c.z_t - c.Ro - 0.3))
    return out


def run(c: Case3d) -> dict:
    d = WORK / c.name
    if d.exists():
        shutil.rmtree(d)
    info = build_mesh(c, d)
    t_end = write_prj(c, d)
    out = d / "out"; out.mkdir()
    r = subprocess.run(["ogs", "-o", str(out.resolve()), "-l", "error", "dd3d.prj"],
                       cwd=str(d.resolve()), capture_output=True, text=True)
    if r.returncode != 0:
        return dict(name=c.name, ok=False, msg=(r.stdout + r.stderr)[-900:], mesh=info)
    files = sorted(out.glob("dd3d_ts_*_t_*.vtu"),
                   key=lambda f: float(re.search(r"_t_([0-9.e+-]+)\.vtu", f.name).group(1)))
    m0 = pv.read(files[0]); corners = corner_ids(m0)
    hist = []
    for f in files:
        tt = float(re.search(r"_t_([0-9.e+-]+)\.vtu", f.name).group(1))
        st = state(c, pv.read(f), corners); st["t"] = tt; st["t_over_Tref"] = tt / c.T_ref
        hist.append(st)
    return dict(name=c.name, ok=True, mesh=info, hist=hist, t_end_over_Tref=t_end / c.T_ref,
                params=dict(C=c.C, F=c.F, K0p=c.K0p, T_ref=c.T_ref, s_ff_surface=c.s_ff_surface,
                            s_ff_tunnel=c.s_ff_tunnel, du_local=c.du_local, ramp_over_Tref=c.ramp / c.T_ref))


def overshoot(hist, key):
    v = np.array([h[key] for h in hist]); ref = v[-1]
    if abs(ref) < 1e-12: return np.nan, np.nan
    i = int(np.argmax(np.abs(v))); return float(abs(v[i]) / abs(ref)), float(hist[i]["t_over_Tref"])


def t50(hist, key):
    v = np.array([h[key] for h in hist]); ref = v[-1]
    if abs(ref) < 1e-12: return np.nan
    f = np.abs(v / ref); i = int(np.argmax(f >= 0.5))
    if i == 0: return np.nan
    t0, t1 = hist[i - 1]["t_over_Tref"], hist[i]["t_over_Tref"]; f0, f1 = f[i - 1], f[i]
    return float(t0 + (0.5 - f0) / (f1 - f0) * (t1 - t0))


def report(r: dict, ff: dict | None):
    c = r["params"]; e = r["hist"][-1]
    print(f"\n=== {r['name']}   C={c['C']:.4f}  F={c['F']:.1f}  K0'={c['K0p']:.3f}  ramp/Tref={c['ramp_over_Tref']:.3f}  "
          f"mesh {r['mesh']['n_cells']} cells ({r['mesh']['n_lining']} lining)")
    print(f"  pore pressure end /Δu : far@axis {e['p_far_axis']/1e5:+.4f}  crown {e['p_crown']/1e5:+.4f}  invert {e['p_invert']/1e5:+.4f}"
          f"   (linear profile predicts axis {-c['du_local']/1e5:+.4f})")
    if "T0" in e:
        dul = c["du_local"]; R = 5.0
        print(f"  lining end state  [kN/m, kNm/m]:  T0={e['T0']/1e3:8.2f}  T1={e['T1']/1e3:8.2f}  T2={e['T2']/1e3:8.2f}   "
              f"M0={e['M0']/1e3:7.3f}  M1={e['M1']/1e3:7.3f}  M2={e['M2']/1e3:7.3f}")
        print(f"  normalised by local Δu·R:  T0/(Δu_l R)={e['T0']/(dul*R):+.4f}  T1/(Δu_l R)={e['T1']/(dul*R):+.4f}  T2/(Δu_l R)={e['T2']/(dul*R):+.4f}"
              f"   (3c theorem: T0→0;  1D n=2 ref T2/(s_dev R)≈1.16 → T2/(Δu_l R)≈{1.16*(1-c['K0p'])/2:+.3f})")
        print(f"  crown/spring/invert  T: {e['T_crn']/1e3:8.2f} {e['T_spr']/1e3:8.2f} {e['T_inv']/1e3:8.2f}   "
              f"M: {e['M_crn']/1e3:7.3f} {e['M_spr']/1e3:7.3f} {e['M_inv']/1e3:7.3f}")
    print(f"  displacements end [mm]: centre {e['uy_centre']*1e3:+.2f}  crown {e['uy_crown']*1e3:+.2f}  invert {e['uy_invert']*1e3:+.2f}  "
          f"spring ux {e['ux_spring']*1e3:+.2f}  oval {e['oval']*1e3:+.2f}   surface x=0 {e['uy_surface_x0']*1e3:+.2f}  far {e['uy_surface_far']*1e3:+.2f}")
    print(f"  free-field analytic [mm]: surface {-c['s_ff_surface']*1e3:+.2f}   at tunnel axis {-c['s_ff_tunnel']*1e3:+.2f}")
    if ff is not None:
        f_e = ff["hist"][-1]
        Is = e["uy_centre"] / f_e["uy_centre"]
        print(f"  I_s = s_tunnel/s_ff(z_t) = {Is:.4f}    (ff numeric centre {f_e['uy_centre']*1e3:+.2f} mm)")
        print(f"  Ψ = t50(tunnel centre)/t50(ff centre) = {t50(r['hist'],'uy_centre')/t50(ff['hist'],'uy_centre'):.3f}")
    if "T0" in e:
        cols = ["T_crn", "T_spr", "T_inv", "M_crn", "M_spr", "M_inv", "T1", "M1", "T2", "M2"]
        print("  Θ (max|·|/|end|):  " + "  ".join(f"{k}={overshoot(r['hist'],k)[0]:.3f}" for k in cols))
        print("  t_peak/Tref     :  " + "  ".join(f"{k}={overshoot(r['hist'],k)[1]:.3f}" for k in cols))


if __name__ == "__main__":
    WORK.mkdir(exist_ok=True)
    base = Case3d("tun_step")
    cases = {
        "ff_step": replace(base, name="ff_step", free_field=True),
        "tun_step": base,
        "tun_step_El2.5e9": replace(base, name="tun_step_El2.5e9", El=2.5e9),
    }
    which = sys.argv[1:] or list(cases)
    results = {}
    for nm in which:
        r = run(cases[nm]); results[nm] = r
        if not r["ok"]:
            print(f"{nm} FAILED:\n{r['msg']}"); continue
        report(r, results.get("ff_step") if nm != "ff_step" else None)
    (HERE / "dd3d_results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\nsaved dd3d_results.json")
