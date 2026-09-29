"""
Verification chain #3 — Einstein & Schwartz (1979) lined circular tunnel, no-slip.

Lined circular opening in an infinite linear-elastic medium, "external loading"
model: lining and ground are stress-free, then far-field stresses sigma_v = P,
sigma_h = K*P are applied.  Quarter symmetry, plane strain, lining as solid
elements (bonded = no-slip), OGS SMALL_DEFORMATION.

Reference for the closed form and the check parameter set:
  Einstein & Schwartz (1979) J. Geotech. Eng. Div. ASCE 105(GT4)
  Itasca FLAC3D verification "Lined Circular Tunnel in an Elastic Medium with
  Anisotropic Stresses" (achieves ~1.2 % on thrust)
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).parent
WORK = ROOT / "runs"


# ----------------------------------------------------------------------------
# closed form
# ----------------------------------------------------------------------------
@dataclass
class Case:
    name: str
    E: float        # ground Young's modulus [Pa]
    nu: float       # ground Poisson ratio
    K: float        # sigma_h / sigma_v
    P: float        # far-field vertical stress [Pa] (compression positive)
    El: float       # lining Young's modulus [Pa]
    nul: float      # lining Poisson ratio
    t: float        # lining thickness [m]
    R: float        # lining centreline radius [m]
    L_over_R: float = 25.0   # half-width of the square domain / R_out
    n_theta: int = 60        # elements along the quarter arc
    n_r: int = 3             # elements across the lining thickness
    h_far_over_R: float = 1.5
    loading: str = "excavation"   # "excavation" (Einstein & Schwartz) | "external" (Burns & Richard / Hoeg)

    def T0_external_exact(self):
        """Exact n=0 (uniform) thrust for EXTERNAL loading of a thin bonded ring in an
        infinite plane-strain medium: q = 2(1-nu) p_mean / (1 + C(1-nu)),  p_mean = (1+K)P/2."""
        pm = 0.5 * (1 + self.K) * self.P
        return 2 * (1 - self.nu) * pm * self.R / (1 + self.C * (1 - self.nu))

    # --- Einstein & Schwartz definitions ---
    @property
    def C(self):
        return self.E * self.R * (1 - self.nul**2) / (self.El * self.t * (1 - self.nu**2))

    @property
    def F(self):
        I = self.t**3 / 12.0
        return self.E * self.R**3 * (1 - self.nul**2) / (self.El * I * (1 - self.nu**2))

    def es_noslip(self, theta):
        C, F, nu, K, P, R = self.C, self.F, self.nu, self.K, self.P, self.R
        a0 = C * F * (1 - nu) / (C + F + C * F * (1 - nu))
        beta = ((6 + F) * C * (1 - nu) + 2 * F * nu) / (3 * F + 3 * C + 2 * C * F * (1 - nu))
        b2 = C * (1 - nu) / (2 * (C * (1 - nu) + 4 * nu - 6 * beta - 3 * beta * C * (1 - nu)))
        a2 = beta * b2
        th = np.asarray(theta, float)
        T = 0.5 * P * R * ((1 + K) * (1 - a0) + (1 - K) * (1 + 2 * a2) * np.cos(2 * th))
        M = 0.25 * P * R**2 * (1 - K) * (1 - 2 * a2 + 2 * b2) * np.cos(2 * th)
        return dict(a0=a0, a2=a2, b2=b2, beta=beta, T=T, M=M,
                    T0=0.5 * P * R * (1 + K) * (1 - a0),
                    T2=0.5 * P * R * (1 - K) * (1 + 2 * a2),
                    M2=0.25 * P * R**2 * (1 - K) * (1 - 2 * a2 + 2 * b2))


# ----------------------------------------------------------------------------
# mesh
# ----------------------------------------------------------------------------
def build_mesh(c: Case, d: Path) -> dict:
    import gmsh
    from ogstools import Meshes

    Ri, Ro = c.R - c.t / 2, c.R + c.t / 2
    L = c.L_over_R * Ro
    h_lin = (np.pi / 2 * Ro) / c.n_theta          # circumferential size at lining
    h_far = c.h_far_over_R * Ro

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("es")
    g = gmsh.model.geo

    O = g.addPoint(0, 0, 0)
    pRi_x, pRi_y = g.addPoint(Ri, 0, 0), g.addPoint(0, Ri, 0)
    pRo_x, pRo_y = g.addPoint(Ro, 0, 0), g.addPoint(0, Ro, 0)
    pL_x, pL_xy, pL_y = g.addPoint(L, 0, 0), g.addPoint(L, L, 0), g.addPoint(0, L, 0)

    arc_in = g.addCircleArc(pRi_x, O, pRi_y)
    arc_out = g.addCircleArc(pRo_x, O, pRo_y)
    lin_x = g.addLine(pRi_x, pRo_x)      # lining on y=0
    lin_y = g.addLine(pRo_y, pRi_y)      # lining on x=0
    gnd_x = g.addLine(pRo_x, pL_x)       # ground on y=0
    right = g.addLine(pL_x, pL_xy)
    top = g.addLine(pL_xy, pL_y)
    gnd_y = g.addLine(pL_y, pRo_y)       # ground on x=0

    lining = g.addPlaneSurface([g.addCurveLoop([arc_in, -lin_y, -arc_out, -lin_x])])
    ground = g.addPlaneSurface([g.addCurveLoop([arc_out, -gnd_y, -top, -right, -gnd_x])])

    # structured lining
    g.mesh.setTransfiniteCurve(arc_in, c.n_theta + 1)
    g.mesh.setTransfiniteCurve(arc_out, c.n_theta + 1)
    g.mesh.setTransfiniteCurve(lin_x, c.n_r + 1)
    g.mesh.setTransfiniteCurve(lin_y, c.n_r + 1)
    g.mesh.setTransfiniteSurface(lining)
    g.mesh.setRecombine(2, lining)
    g.mesh.setRecombine(2, ground)
    g.synchronize()

    # graded ground: size h_lin at the outer arc growing to h_far
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", [arc_out])
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)
    f_thr = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thr, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMin", h_lin)
    gmsh.model.mesh.field.setNumber(f_thr, "SizeMax", h_far)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMin", 0.5 * Ro)
    gmsh.model.mesh.field.setNumber(f_thr, "DistMax", 0.6 * L)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_thr)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 8)          # frontal-delaunay for quads
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 1)

    gmsh.model.addPhysicalGroup(2, [ground], name="ground")
    gmsh.model.addPhysicalGroup(2, [lining], name="lining")
    gmsh.model.addPhysicalGroup(1, [arc_in], name="inner")
    gmsh.model.addPhysicalGroup(1, [lin_x, gnd_x], name="sym_y")   # y = 0
    gmsh.model.addPhysicalGroup(1, [lin_y, gnd_y], name="sym_x")   # x = 0
    gmsh.model.addPhysicalGroup(1, [top], name="top")
    gmsh.model.addPhysicalGroup(1, [right], name="right")

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)
    d.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(d / "es.msh"))
    gmsh.finalize()

    ms = Meshes.from_gmsh(d / "es.msh", dim=2, reindex=True, log=False)
    dom = ms["domain"]
    rc = np.linalg.norm(dom.cell_centers().points[:, :2], axis=1)
    mid = np.where(rc < Ro + 1e-6, 1, 0).astype(np.int32)      # 0 ground, 1 lining
    dom.cell_data["MaterialIDs"] = mid
    ms.save(d, overwrite=True)

    CT = {23: "QUAD8", 28: "QUAD9", 22: "TRI6", 21: "LINE3", 9: "QUAD4", 5: "TRI3", 3: "LINE2"}
    types = {CT.get(int(x), int(x)) for x in dom.celltypes}
    return dict(n_pts=int(dom.n_points), n_cells=int(dom.n_cells),
                n_lining=int((mid == 1).sum()), types=sorted(types),
                subdomains=sorted(ms.keys()), Ri=Ri, Ro=Ro, L=L)


# ----------------------------------------------------------------------------
# project file
# ----------------------------------------------------------------------------
PRJ = """<?xml version="1.0" encoding="ISO-8859-1"?>
<OpenGeoSysProject>
    <meshes>
        <mesh>domain.vtu</mesh>
        <mesh>inner.vtu</mesh>
        <mesh>sym_x.vtu</mesh>
        <mesh>sym_y.vtu</mesh>
        <mesh>top.vtu</mesh>
        <mesh>right.vtu</mesh>
    </meshes>
    <processes>
        <process>
            <name>SD</name>
            <type>SMALL_DEFORMATION</type>
            <integration_order>3</integration_order>
            <constitutive_relation id="0">
                <type>LinearElasticIsotropic</type>
                <youngs_modulus>E_ground</youngs_modulus>
                <poissons_ratio>nu_ground</poissons_ratio>
            </constitutive_relation>
            <constitutive_relation id="1">
                <type>LinearElasticIsotropic</type>
                <youngs_modulus>E_lining</youngs_modulus>
                <poissons_ratio>nu_lining</poissons_ratio>
            </constitutive_relation>
            <specific_body_force>0 0</specific_body_force>
{initial_stress_tag}
            <process_variables>
                <process_variable>displacement</process_variable>
            </process_variables>
            <secondary_variables>
                <secondary_variable internal_name="sigma" output_name="sigma"/>
                <secondary_variable internal_name="epsilon" output_name="epsilon"/>
            </secondary_variables>
        </process>
    </processes>
    <time_loop>
        <processes>
            <process ref="SD">
                <nonlinear_solver>basic_newton</nonlinear_solver>
                <convergence_criterion>
                    <type>DeltaX</type>
                    <norm_type>NORM2</norm_type>
                    <abstol>1e-10</abstol>
                </convergence_criterion>
                <time_discretization><type>BackwardEuler</type></time_discretization>
                <time_stepping>
                    <type>FixedTimeStepping</type>
                    <t_initial>0</t_initial>
                    <t_end>1</t_end>
                    <timesteps><pair><repeat>1</repeat><delta_t>1</delta_t></pair></timesteps>
                </time_stepping>
            </process>
        </processes>
        <output>
            <type>VTK</type>
            <prefix>es</prefix>
            <timesteps><pair><repeat>1</repeat><each_steps>1</each_steps></pair></timesteps>
            <variables>
                <variable>displacement</variable>
                <variable>sigma</variable>
                <variable>epsilon</variable>
            </variables>
            <suffix>_ts_{{:timestep}}_t_{{:time}}</suffix>
        </output>
    </time_loop>
    <media>
        <medium id="0">
            <phases><phase><type>Solid</type><properties>
                <property><name>density</name><type>Constant</type><value>2000</value></property>
            </properties></phase></phases>
        </medium>
        <medium id="1">
            <phases><phase><type>Solid</type><properties>
                <property><name>density</name><type>Constant</type><value>2500</value></property>
            </properties></phase></phases>
        </medium>
    </media>
    <parameters>
        <parameter><name>E_ground</name><type>Constant</type><value>{E}</value></parameter>
        <parameter><name>nu_ground</name><type>Constant</type><value>{nu}</value></parameter>
        <parameter><name>E_lining</name><type>Constant</type><value>{El}</value></parameter>
        <parameter><name>nu_lining</name><type>Constant</type><value>{nul}</value></parameter>
        <parameter><name>zero</name><type>Constant</type><value>0</value></parameter>
        <parameter><name>u0</name><type>Constant</type><values>0 0</values></parameter>
        <parameter><name>trac_top</name><type>Constant</type><value>{trac_top}</value></parameter>
        <parameter><name>trac_right</name><type>Constant</type><value>{trac_right}</value></parameter>
        <parameter>
            <name>sigma0</name>
            <type>Function</type>
            <expression>if (sqrt(x^2+y^2) &gt; {Ro}) {sxx}; else 0.0</expression>
            <expression>if (sqrt(x^2+y^2) &gt; {Ro}) {syy}; else 0.0</expression>
            <expression>if (sqrt(x^2+y^2) &gt; {Ro}) {szz}; else 0.0</expression>
            <expression>0.0</expression>
        </parameter>
    </parameters>
    <process_variables>
        <process_variable>
            <name>displacement</name>
            <components>2</components>
            <order>2</order>
            <initial_condition>u0</initial_condition>
            <boundary_conditions>
                <boundary_condition><mesh>sym_x</mesh><type>Dirichlet</type><component>0</component><parameter>zero</parameter></boundary_condition>
                <boundary_condition><mesh>sym_y</mesh><type>Dirichlet</type><component>1</component><parameter>zero</parameter></boundary_condition>
                <boundary_condition><mesh>top</mesh><type>Neumann</type><component>1</component><parameter>trac_top</parameter></boundary_condition>
                <boundary_condition><mesh>right</mesh><type>Neumann</type><component>0</component><parameter>trac_right</parameter></boundary_condition>
            </boundary_conditions>
        </process_variable>
    </process_variables>
    <nonlinear_solvers>
        <nonlinear_solver>
            <name>basic_newton</name>
            <type>Newton</type>
            <max_iter>10</max_iter>
            <linear_solver>general_linear_solver</linear_solver>
        </nonlinear_solver>
    </nonlinear_solvers>
    <linear_solvers>
        <linear_solver>
            <name>general_linear_solver</name>
            <eigen>
                <solver_type>SparseLU</solver_type>
                <scaling>true</scaling>
            </eigen>
        </linear_solver>
    </linear_solvers>
</OpenGeoSysProject>
"""


def write_prj(c: Case, d: Path) -> Path:
    p = d / "es.prj"
    Ro = c.R + c.t / 2
    tag = "            <initial_stress>sigma0</initial_stress>" if c.loading == "excavation" else ""
    p.write_text(PRJ.format(E=repr(c.E), nu=repr(c.nu), El=repr(c.El), nul=repr(c.nul),
                            trac_top=repr(-c.P), trac_right=repr(-c.K * c.P),
                            initial_stress_tag=tag, Ro=repr(Ro),
                            sxx=repr(-c.K * c.P), syy=repr(-c.P),
                            szz=repr(-c.nu * (1 + c.K) * c.P)))
    return p


# ----------------------------------------------------------------------------
# post-processing: thrust and moment from sigma_theta_theta across the lining
# ----------------------------------------------------------------------------
def lining_forces(c: Case, mesh: pv.UnstructuredGrid, n_theta_samp: int = 19, n_r_samp: int = 15,
                  fit_fraction: float = 0.70, fit_order: int = 1):
    """Thrust T (compression +) and moment M (+ = inner fibre more compressed) per unit length.

    Nodal sigma at the lining/ground interface is an average over both materials, so the
    outermost samples are contaminated.  sigma_tt is linear through a thin ring (thrust +
    bending) to first order; a bonded ring of finite thickness also carries 1/r^2 and 1/r^4
    terms.  fit_order=1 (linear fit over the inner `fit_fraction` of the thickness, extrapolated
    to the centreline) is adequate for t/R = 0.025 with 3 elements through the thickness
    (verified against the exact solid-ring solution to 0.5 %; the profile bias is -0.4 % in T2).
    For t/R >= 0.05 use fit_order=2 together with >= 6 elements through the thickness: in the
    infinite-medium check (thick_check.py) this gives T2 within +0.4 % (t/R = 0.05) and +1.4 %
    (t/R = 0.10) of the exact solution, whereas 3 elements under-resolve T2 by 4 % and 17 %.
    T = -(a t + c t^3/12), M = b t^3/12 for sigma_tt = a + b (r-R) + c (r-R)^2.
    """
    Ri, Ro = c.R - c.t / 2, c.R + c.t / 2
    r = np.linspace(Ri + 0.03 * c.t, Ri + fit_fraction * c.t, n_r_samp)   # start inside: VTK cell search on curved QUAD9 edges
    th = np.linspace(0.0, np.pi / 2, n_theta_samp)
    th[0] += 1e-4
    th[-1] -= 1e-4
    TH, RR = np.meshgrid(th, r, indexing="ij")
    pts = np.column_stack([(RR * np.cos(TH)).ravel(), (RR * np.sin(TH)).ravel(), np.zeros(RR.size)])
    probe = pv.PolyData(pts).sample(mesh, tolerance=1e-4)   # curved QUAD9 edges need an explicit tolerance
    s_ = np.asarray(probe.point_data["sigma"])           # [xx, yy, zz, xy]
    valid = np.asarray(probe.point_data["vtkValidPointMask"]).astype(bool)
    if not valid.all():
        raise RuntimeError(f"{(~valid).sum()} sample points fell outside the mesh")
    ct, st = np.cos(TH.ravel()), np.sin(TH.ravel())
    s_tt = (s_[:, 0] * st**2 + s_[:, 1] * ct**2 - 2 * s_[:, 3] * st * ct).reshape(TH.shape)
    cols = [np.ones_like(r), r - c.R] + ([(r - c.R) ** 2] if fit_order == 2 else [])
    A = np.column_stack(cols)
    T = np.empty(len(th)); M = np.empty(len(th)); resid = np.empty(len(th))
    for i in range(len(th)):
        coef, res, *_ = np.linalg.lstsq(A, s_tt[i], rcond=None)
        a, b = coef[0], coef[1]; cq = coef[2] if fit_order == 2 else 0.0
        T[i] = -(a * c.t + cq * c.t**3 / 12.0)   # -∫ s_tt dr  over [Ri, Ro]
        M[i] = b * c.t**3 / 12.0                 #  ∫ s_tt (r-R) dr  (the quadratic term is odd about R)
        resid[i] = np.sqrt(res[0] / len(r)) / max(abs(a), 1e-30) if len(res) else 0.0
    return th, T, M, float(resid.max())


def run_case(c: Case, keep: bool = True) -> dict:
    d = WORK / c.name
    if d.exists():
        shutil.rmtree(d)
    info = build_mesh(c, d)
    prj = write_prj(c, d)
    out = d / "out"
    out.mkdir()
    r = subprocess.run(["ogs", "-o", str(out.resolve()), "-l", "error", prj.name],
                       cwd=str(d.resolve()), capture_output=True, text=True)
    if r.returncode != 0:
        return dict(name=c.name, ok=False, msg=(r.stdout + r.stderr)[-600:], mesh=info)

    vtu = sorted(out.glob("es_ts_*_t_*.vtu"))[-1]
    mesh = pv.read(vtu)
    th, T, M, fit_resid = lining_forces(c, mesh)
    es = c.es_noslip(th)

    # Fourier-fit the FE thrust/moment to T0 + T2 cos2θ (quarter arc, θ in [0, π/2])
    A = np.column_stack([np.ones_like(th), np.cos(2 * th)])
    T0_fe, T2_fe = np.linalg.lstsq(A, T, rcond=None)[0]
    M2_fe = np.linalg.lstsq(np.cos(2 * th)[:, None], M, rcond=None)[0][0]

    # crown displacement (θ = 90°) and springline (θ = 0) radial displacements at R
    u = np.asarray(mesh.point_data["displacement"])
    def u_r_at(theta):
        p = pv.PolyData(np.array([[c.R * np.cos(theta), c.R * np.sin(theta), 0.0]])).sample(mesh)
        uu = np.asarray(p.point_data["displacement"])[0]
        return uu[0] * np.cos(theta) + uu[1] * np.sin(theta)
    u_spring, u_crown = u_r_at(1e-4), u_r_at(np.pi / 2 - 1e-4)

    res = dict(
        name=c.name, ok=True, mesh=info, C=c.C, F=c.F, loading=c.loading,
        fit_resid=fit_resid, T0_external_exact=c.T0_external_exact(),
        es=dict(a0=es["a0"], a2=es["a2"], b2=es["b2"], T0=es["T0"], T2=es["T2"], M2=es["M2"]),
        fe=dict(T0=float(T0_fe), T2=float(T2_fe), M2=float(M2_fe)),
        err_pct=dict(
            T0=100 * (T0_fe - es["T0"]) / es["T0"],
            T2=100 * (T2_fe - es["T2"]) / es["T2"] if abs(es["T2"]) > 0 else np.nan,
            M2=100 * (M2_fe - es["M2"]) / es["M2"] if abs(es["M2"]) > 0 else np.nan,
            T_max_pointwise=float(100 * np.max(np.abs(T - es["T"])) / np.max(np.abs(es["T"]))),
            M_max_pointwise=float(100 * np.max(np.abs(M - es["M"])) / np.max(np.abs(es["M"]))) if np.max(np.abs(es["M"])) > 0 else np.nan,
        ),
        theta_deg=np.degrees(th).tolist(), T_fe=T.tolist(), T_es=es["T"].tolist(),
        M_fe=M.tolist(), M_es=es["M"].tolist(),
        u_r_spring=float(u_spring), u_r_crown=float(u_crown),
    )
    if not keep:
        shutil.rmtree(d)
    return res


def fmt_row(r):
    if not r["ok"]:
        return f"{r['name']:<14} FAILED: {r['msg'][-200:]}"
    e = r["err_pct"]
    if r["loading"] == "external":
        ex = r["T0_external_exact"]
        return (f"{r['name']:<14} [external ] C={r['C']:8.4f} | "
                f"T0 {r['fe']['T0']/1e3:9.2f} vs exact-n0 {ex/1e3:9.2f} kN/m ({100*(r['fe']['T0']-ex)/ex:+6.2f}%) | "
                f"(E&S excav. would be {r['es']['T0']/1e3:9.2f}) | fit resid {r['fit_resid']:.1e}")
    return (f"{r['name']:<14} [excav E&S] C={r['C']:8.4f} F={r['F']:10.1f} | "
            f"T0 {r['fe']['T0']/1e3:9.2f} vs {r['es']['T0']/1e3:9.2f} kN/m ({e['T0']:+6.2f}%) | "
            f"T2 {r['fe']['T2']/1e3:8.2f} vs {r['es']['T2']/1e3:8.2f} ({e['T2']:+6.2f}%) | "
            f"M2 {r['fe']['M2']/1e3:8.3f} vs {r['es']['M2']/1e3:8.3f} kNm/m ({e['M2']:+6.2f}%) | fit resid {r['fit_resid']:.1e}")


if __name__ == "__main__":
    WORK.mkdir(exist_ok=True)
    itasca = dict(E=48e6, nu=0.34, K=0.5, P=600e3, El=25e9, nul=0.15, t=0.125, R=5.0)

    cases = [
        # A — Itasca/FLAC3D reference set, excavation loading (= Einstein & Schwartz)
        Case("A_itasca", **itasca),
        # A' — same geometry, EXTERNAL loading: checked against the exact n=0 solution
        Case("A_external", **itasca, loading="external"),
        # B — compressibility sweep (lining stiffness / 10, / 100)
        Case("B_El_2.5e9", **{**itasca, "El": 2.5e9}),
        Case("B_El_2.5e8", **{**itasca, "El": 2.5e8}),
        # C — Poisson probe toward the incompressible (undrained) limit
        Case("C_nu0.45", **{**itasca, "nu": 0.45}),
        Case("C_nu0.49", **{**itasca, "nu": 0.49}),
        Case("C_nu0.499", **{**itasca, "nu": 0.499}),
        Case("C_ext_nu0.49", **{**itasca, "nu": 0.49}, loading="external"),
    ]
    which = sys.argv[1:] or [c.name for c in cases]
    results = []
    for c in cases:
        if c.name not in which:
            continue
        print(f"--- {c.name}: building & running ...", flush=True)
        r = run_case(c)
        results.append(r)
        if r["ok"]:
            m = r["mesh"]
            print(f"    mesh: {m['n_cells']} cells ({m['n_lining']} lining), types {m['types']}, L={m['L']:.1f} m")
        print("   ", fmt_row(r), flush=True)

    (ROOT / "es_results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\nsaved es_results.json")
