"""
Verification 3b — uniform far-field drawdown Δu around an impermeable lined tunnel.

Linear poroelastic ground (Biot α=1, incompressible constituents), impermeable elastic
lining (Biot α=0, k→0), plane strain, quarter symmetry.  Far-field TOTAL traction is held
(zero increment) while the far-field pore pressure is lowered by Δu.  Drained end state is
compared with the closed form derived in the design note:

    Δq/Δu = (1-2ν') / (1 + C(1-ν'))         Δq: change of total pressure on the lining
    I0    = C(1-ν') / (1 + C(1-ν'))         uniform-mode ground-following ratio
    u_ff(R) = -Δu R (1+ν')(1-2ν') / E       free-field radial displacement at r=R

The transient Δq(t) as the drawdown front arrives from the outer boundary is recorded too.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "es_verif"))
import es_verif as ev  # mesh builder + lining_forces (thrust/moment from sigma_tt)

ROOT = Path(__file__).parent
WORK = ROOT / "runs"


@dataclass
class DDCase:
    name: str
    E: float = 48e6
    nu: float = 0.34
    El: float = 25e9
    nul: float = 0.15
    t: float = 0.125
    R: float = 5.0
    k: float = 1e-12          # ground intrinsic permeability [m2]
    mu: float = 1e-3
    du: float = 100e3         # drawdown [Pa]
    L_over_R: float = 25.0
    n_theta: int = 60
    n_r: int = 3
    h_far_over_R: float = 1.5

    @property
    def C(self):  # Einstein & Schwartz compressibility ratio (drained ν')
        return self.E * self.R * (1 - self.nul**2) / (self.El * self.t * (1 - self.nu**2))

    @property
    def c_star(self):
        return self.C * (1 - self.nu)

    @property
    def dq_over_du_exact(self):
        return (1 - 2 * self.nu) / (1 + self.c_star)

    @property
    def I0_exact(self):
        return self.c_star / (1 + self.c_star)

    @property
    def u_ff_R(self):
        return -self.du * self.R * (1 + self.nu) * (1 - 2 * self.nu) / self.E

    @property
    def M_oed(self):
        return self.E * (1 - self.nu) / ((1 + self.nu) * (1 - 2 * self.nu))

    @property
    def c_v(self):
        return self.k / self.mu * self.M_oed

    def as_es_case(self) -> ev.Case:
        return ev.Case(self.name, E=self.E, nu=self.nu, K=1.0, P=self.du, El=self.El, nul=self.nul,
                       t=self.t, R=self.R, L_over_R=self.L_over_R, n_theta=self.n_theta,
                       n_r=self.n_r, h_far_over_R=self.h_far_over_R)


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
            <name>HM</name>
            <type>HYDRO_MECHANICS</type>
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
            <process_variables>
                <pressure>pressure</pressure>
                <displacement>displacement</displacement>
            </process_variables>
            <secondary_variables>
                <secondary_variable internal_name="sigma" output_name="sigma"/>
                <secondary_variable internal_name="epsilon" output_name="epsilon"/>
            </secondary_variables>
            <specific_body_force>0 0</specific_body_force>
        </process>
    </processes>
    <media>
        <medium id="0">
            <phases>
                <phase><type>AqueousLiquid</type><properties>
                    <property><name>viscosity</name><type>Constant</type><value>{mu}</value></property>
                    <property><name>density</name><type>Constant</type><value>1000</value></property>
                </properties></phase>
                <phase><type>Solid</type><properties>
                    <property><name>density</name><type>Constant</type><value>2000</value></property>
                </properties></phase>
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
                    <property><name>density</name><type>Constant</type><value>1000</value></property>
                </properties></phase>
                <phase><type>Solid</type><properties>
                    <property><name>density</name><type>Constant</type><value>2500</value></property>
                </properties></phase>
            </phases>
            <properties>
                <property><name>permeability</name><type>Constant</type><value>1e-30</value></property>
                <property><name>porosity</name><type>Constant</type><value>0</value></property>
                <property><name>biot_coefficient</name><type>Constant</type><value>0.0</value></property>
                <property><name>reference_temperature</name><type>Constant</type><value>293.15</value></property>
            </properties>
        </medium>
    </media>
    <time_loop>
        <processes>
            <process ref="HM">
                <nonlinear_solver>basic_newton</nonlinear_solver>
                <convergence_criterion>
                    <type>PerComponentDeltaX</type>
                    <norm_type>NORM2</norm_type>
                    <abstols>1e-3 1e-11 1e-11</abstols>
                </convergence_criterion>
                <time_discretization><type>BackwardEuler</type></time_discretization>
                <time_stepping>
                    <type>FixedTimeStepping</type>
                    <t_initial>0</t_initial>
                    <t_end>{t_end}</t_end>
                    <timesteps>
{pairs}
                    </timesteps>
                </time_stepping>
            </process>
        </processes>
        <output>
            <type>VTK</type>
            <prefix>dd</prefix>
            <timesteps><pair><repeat>1000</repeat><each_steps>1</each_steps></pair></timesteps>
            <variables>
                <variable>pressure</variable>
                <variable>displacement</variable>
                <variable>sigma</variable>
            </variables>
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
        <parameter><name>p_far</name><type>Constant</type><value>{p_far}</value></parameter>
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
            </boundary_conditions>
        </process_variable>
        <process_variable>
            <name>pressure</name>
            <components>1</components>
            <order>1</order>
            <initial_condition>p0</initial_condition>
            <boundary_conditions>
                <boundary_condition><mesh>top</mesh><type>Dirichlet</type><component>0</component><parameter>p_far</parameter></boundary_condition>
                <boundary_condition><mesh>right</mesh><type>Dirichlet</type><component>0</component><parameter>p_far</parameter></boundary_condition>
            </boundary_conditions>
        </process_variable>
    </process_variables>
    <nonlinear_solvers>
        <nonlinear_solver>
            <name>basic_newton</name>
            <type>Newton</type>
            <max_iter>20</max_iter>
            <linear_solver>general_linear_solver</linear_solver>
        </nonlinear_solver>
    </nonlinear_solvers>
    <linear_solvers>
        <linear_solver>
            <name>general_linear_solver</name>
            <eigen><solver_type>SparseLU</solver_type><scaling>true</scaling></eigen>
        </linear_solver>
    </linear_solvers>
</OpenGeoSysProject>
"""


def time_pairs(T_ref: float, n_decades: float = 4.0, per_decade: int = 8, t_start_frac: float = 1e-4):
    """Geometric time grid from t_start_frac*T_ref spanning n_decades, as FixedTimeStepping pairs."""
    t0 = t_start_frac * T_ref
    ratio = 10 ** (1.0 / per_decade)
    pairs, t, dt = [], 0.0, t0
    n = int(round(n_decades * per_decade))
    for _ in range(n):
        pairs.append((1, dt))
        t += dt
        dt *= ratio
    xml = "\n".join(f"                        <pair><repeat>{r}</repeat><delta_t>{d!r}</delta_t></pair>"
                    for r, d in pairs)
    return xml, t


def run(c: DDCase) -> dict:
    d = WORK / c.name
    if d.exists():
        shutil.rmtree(d)
    info = ev.build_mesh(c.as_es_case(), d)
    L = info["L"]
    T_ref = L**2 / c.c_v                     # diffusion time across the domain
    pairs, t_end = time_pairs(T_ref)
    (d / "dd.prj").write_text(PRJ.format(E=repr(c.E), nu=repr(c.nu), El=repr(c.El), nul=repr(c.nul),
                                         k=repr(c.k), mu=repr(c.mu), p_far=repr(-c.du),
                                         t_end=repr(t_end), pairs=pairs))
    out = d / "out"
    out.mkdir()
    r = subprocess.run(["ogs", "-o", str(out.resolve()), "-l", "error", "dd.prj"],
                       cwd=str(d.resolve()), capture_output=True, text=True)
    if r.returncode != 0:
        return dict(name=c.name, ok=False, msg=(r.stdout + r.stderr)[-800:])

    files = sorted(out.glob("dd_ts_*_t_*.vtu"),
                   key=lambda f: float(re.search(r"_t_([0-9.e+-]+)\.vtu", f.name).group(1)))
    esc = c.as_es_case()
    hist = []
    for f in files:
        tt = float(re.search(r"_t_([0-9.e+-]+)\.vtu", f.name).group(1))
        m = pv.read(f)
        th, T, M, _ = ev.lining_forces(esc, m)
        A = np.column_stack([np.ones_like(th), np.cos(2 * th)])
        T0, T2 = np.linalg.lstsq(A, T, rcond=None)[0]
        # radial displacement of the lining centreline at θ = 45° and pore pressure at the interface
        th45 = np.pi / 4
        pt = np.array([[c.R * np.cos(th45), c.R * np.sin(th45), 0.0]])
        q = pv.PolyData(pt).sample(m, tolerance=1e-4)
        uu = np.asarray(q.point_data["displacement"])[0]
        u_r = uu[0] * np.cos(th45) + uu[1] * np.sin(th45)
        pt_o = np.array([[(c.R + c.t / 2 + 0.02) * np.cos(th45), (c.R + c.t / 2 + 0.02) * np.sin(th45), 0.0]])
        p_if = float(np.asarray(pv.PolyData(pt_o).sample(m, tolerance=1e-4).point_data["pressure"])[0])
        hist.append(dict(t=tt, T0=float(T0), T2=float(T2), u_r=float(u_r), p_if=p_if))

    end = hist[-1]
    dq_fe = end["T0"] / c.R                     # thin-ring: uniform thrust = q R
    I0_fe = end["u_r"] / c.u_ff_R
    res = dict(name=c.name, ok=True, mesh=info, C=c.C, c_star=c.c_star, T_ref=T_ref,
               dq_over_du=dict(fe=dq_fe / c.du, exact=c.dq_over_du_exact),
               I0=dict(fe=I0_fe, exact=c.I0_exact),
               p_if_end_over_du=end["p_if"] / c.du,
               T2_end_over_T0=end["T2"] / end["T0"] if end["T0"] else np.nan,
               hist=hist)
    res["err_pct"] = dict(dq=100 * (res["dq_over_du"]["fe"] - c.dq_over_du_exact) / c.dq_over_du_exact,
                          I0=100 * (I0_fe - c.I0_exact) / max(c.I0_exact, 1e-12))
    # transient overshoot of the lining load relative to the drained end state
    T0s = np.array([h["T0"] for h in hist])
    res["theta_overshoot"] = float(T0s.max() / T0s[-1]) if T0s[-1] > 0 else np.nan
    res["t_peak_over_Tref"] = float(hist[int(np.argmax(T0s))]["t"] / T_ref)
    return res


if __name__ == "__main__":
    WORK.mkdir(exist_ok=True)
    cases = [
        DDCase("El_25e9", El=25e9),      # c* ≈ 0.056  (stiff)
        DDCase("El_2.5e9", El=2.5e9),    # c* ≈ 0.56
        DDCase("El_5e8", El=5e8),        # c* ≈ 2.8    (past the I0 = 0.5 transition)
        DDCase("El_1e8", El=1e8),        # c* ≈ 14     (flexible)
    ]
    which = sys.argv[1:] or [c.name for c in cases]
    results = []
    print(f"{'case':10s} {'c*=C(1-nu)':>10s} | {'dq/du FE':>9s} {'exact':>7s} {'err':>7s} | "
          f"{'I0 FE':>7s} {'exact':>7s} {'err':>7s} | {'p_if/du':>8s} {'T2/T0':>6s} | {'Θ':>6s} {'t_pk/Tref':>9s}")
    for c in cases:
        if c.name not in which:
            continue
        r = run(c)
        results.append(r)
        if not r["ok"]:
            print(f"{c.name:10s} FAILED: {r['msg'][-300:]}")
            continue
        e = r["err_pct"]
        print(f"{c.name:10s} {r['c_star']:10.4f} | {r['dq_over_du']['fe']:9.4f} {r['dq_over_du']['exact']:7.4f} "
              f"{e['dq']:+6.2f}% | {r['I0']['fe']:7.4f} {r['I0']['exact']:7.4f} {e['I0']:+6.2f}% | "
              f"{r['p_if_end_over_du']:8.4f} {r['T2_end_over_T0']:6.3f} | {r['theta_overshoot']:6.4f} {r['t_peak_over_Tref']:9.3f}",
              flush=True)
    (ROOT / "dd_results.json").write_text(json.dumps(results, indent=1, default=float))
    print("saved dd_results.json")
