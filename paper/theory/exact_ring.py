"""
Exact plane-strain elasticity solution: bonded elastic annulus (lining, Ri<r<Ro) inside an
infinite elastic medium, EXTERNAL loading (both stress-free, then far-field stress applied).

Modes:  n=0  far-field isotropic in-plane stress  p  (tension +)
        n=2  far-field deviator   sigma_xx = s, sigma_yy = -s  (tension +)

Displacement ansatz per region (theta from x-axis):
    n=0:  u_r = a r + b / r ,                        u_th = 0
    n=2:  u_r = sum_i a_i r^{p_i} cos 2th ,  u_th = sum_i b_i r^{p_i} sin 2th ,  p_i in {1,3,-1,-3}
Navier's equations fix b_i in terms of a_i (4 free constants per region for n=2).

Outputs: lining thrust T(theta) = -∫ sigma_tt dr (compression +) and moment
M = ∫ sigma_tt (r-R) dr  (+ = inner fibre more compressed), split into T0, T2, M2.
Compared with the OGS FE runs (es_verif A_external, drawdown_k0 SD, drawdown_3d).
"""
from __future__ import annotations
import json
from pathlib import Path
import sympy as sp
import os as _os, pathlib as _pl
# Repository root: override with TD_ROOT if the tree is moved.
ROOT = _os.environ.get("TD_ROOT") or str(_pl.Path(__file__).resolve().parents[2])

r, th = sp.symbols("r theta", positive=True)
G, nu = sp.symbols("G nu", positive=True)          # generic material symbols (substituted per region)
lam = 2 * G * nu / (1 - 2 * nu)


def strains_stresses(ur, ut):
    e_rr = sp.diff(ur, r)
    e_tt = ur / r + sp.diff(ut, th) / r
    e_rt = sp.Rational(1, 2) * (sp.diff(ur, th) / r + sp.diff(ut, r) - ut / r)
    tr = e_rr + e_tt
    s_rr = lam * tr + 2 * G * e_rr
    s_tt = lam * tr + 2 * G * e_tt
    s_rt = 2 * G * e_rt
    return s_rr, s_tt, s_rt


def equilibrium(s_rr, s_tt, s_rt):
    eq1 = sp.diff(s_rr, r) + sp.diff(s_rt, th) / r + (s_rr - s_tt) / r
    eq2 = sp.diff(s_rt, r) + sp.diff(s_tt, th) / r + 2 * s_rt / r
    return sp.simplify(eq1), sp.simplify(eq2)


# ---------------------------------------------------------------- general n=2 field
def general_n2(tag):
    a = sp.symbols(f"a1{tag} a3{tag} am1{tag} am3{tag}")
    b = sp.symbols(f"b1{tag} b3{tag} bm1{tag} bm3{tag}")
    pw = [1, 3, -1, -3]
    ur = sum(ai * r**p for ai, p in zip(a, pw)) * sp.cos(2 * th)
    ut = sum(bi * r**p for bi, p in zip(b, pw)) * sp.sin(2 * th)
    s_rr, s_tt, s_rt = strains_stresses(ur, ut)
    eq1, eq2 = equilibrium(s_rr, s_tt, s_rt)
    # collect powers of r in both equations (divide out the angular factor)
    e1 = sp.expand(sp.simplify(eq1 / sp.cos(2 * th)))
    e2 = sp.expand(sp.simplify(eq2 / sp.sin(2 * th)))
    conds = []
    for e in (e1, e2):
        num, _den = sp.fraction(sp.cancel(sp.together(e * r**5)))
        poly = sp.Poly(sp.expand(num), r)
        conds += [c for c in poly.coeffs()]
    sol = sp.solve(conds, list(b), dict=True)
    assert len(sol) == 1, sol
    sol = sol[0]
    ur, ut = ur.subs(sol), ut.subs(sol)
    s_rr, s_tt, s_rt = [sp.simplify(x.subs(sol)) for x in (s_rr, s_tt, s_rt)]
    # sanity: equilibrium satisfied identically
    q1, q2 = equilibrium(s_rr, s_tt, s_rt)
    assert sp.simplify(q1) == 0 and sp.simplify(q2) == 0
    return dict(a=a, ur=ur, ut=ut, s_rr=s_rr, s_tt=s_tt, s_rt=s_rt)


def general_n0(tag):
    a, b = sp.symbols(f"A{tag} B{tag}")
    ur = a * r + b / r
    ut = sp.Integer(0)
    s_rr, s_tt, s_rt = strains_stresses(ur, ut)
    q1, q2 = equilibrium(s_rr, s_tt, s_rt)
    assert sp.simplify(q1) == 0 and sp.simplify(q2) == 0
    return dict(a=(a, b), ur=ur, ut=ut, s_rr=s_rr, s_tt=s_tt, s_rt=s_rt)


def solve_external(E_s, nu_s, E_l, nu_l, Ri, Ro, p_iso, s_dev):
    """Return dict with lining T0, T2, M2 (per unit length) and interface data.
    E_s, nu_s: medium; E_l, nu_l: lining; p_iso: far-field isotropic in-plane stress (tension +);
    s_dev: far-field deviator (sigma_xx - sigma_yy)/2 (tension +)."""
    Gs, Gl = E_s / (2 * (1 + nu_s)), E_l / (2 * (1 + nu_l))
    Rm = (Ri + Ro) / 2
    out = {}

    # ---------------- n = 0
    L, M_ = general_n0("L"), general_n0("M")
    subL = {G: Gl, nu: nu_l}
    subM = {G: Gs, nu: nu_s}
    # far field: u_r -> p (1-2nu) r /(2G)  => A_M fixed ; medium r^1 coefficient
    AM, BM = M_["a"]
    AL, BL = L["a"]
    eqs = [
        L["s_rr"].subs(subL).subs(r, Ri),                                 # inner free
        (L["s_rr"].subs(subL) - M_["s_rr"].subs(subM)).subs(r, Ro),       # traction cont.
        (L["ur"] - M_["ur"]).subs(r, Ro),                                 # displacement cont.
        AM - p_iso * (1 - 2 * nu_s) / (2 * Gs),
    ]
    sol0 = sp.solve(eqs, [AL, BL, AM, BM], dict=True)[0]
    s_tt_L0 = L["s_tt"].subs(subL).subs(sol0)
    T0 = -sp.integrate(s_tt_L0, (r, Ri, Ro))
    M0 = sp.integrate(s_tt_L0 * (r - Rm), (r, Ri, Ro))
    q0 = -M_["s_rr"].subs(subM).subs(sol0).subs(r, Ro)                    # interface pressure (compression +)
    urR0 = L["ur"].subs(sol0).subs(r, Rm)
    out.update(T0=float(T0), M0=float(M0), q0=float(q0), ur0_R=float(urR0))

    # ---------------- n = 2
    L2, M2_ = general_n2("L"), general_n2("M")
    aL = L2["a"]; aM = M2_["a"]
    subL = {G: Gl, nu: nu_l}; subM = {G: Gs, nu: nu_s}
    # far field: u_r -> s r/(2G) cos2th  (a1 = s/2G) ; r^3 term zero
    def at(expr, rr): return expr.subs(r, rr)
    cos_, sin_ = sp.cos(2 * th), sp.sin(2 * th)
    eqs = [
        sp.simplify(at(L2["s_rr"].subs(subL), Ri) / cos_),
        sp.simplify(at(L2["s_rt"].subs(subL), Ri) / sin_),
        sp.simplify(at(L2["s_rr"].subs(subL) - M2_["s_rr"].subs(subM), Ro) / cos_),
        sp.simplify(at(L2["s_rt"].subs(subL) - M2_["s_rt"].subs(subM), Ro) / sin_),
        sp.simplify(at(L2["ur"].subs(subL) - M2_["ur"].subs(subM), Ro) / cos_),
        sp.simplify(at(L2["ut"].subs(subL) - M2_["ut"].subs(subM), Ro) / sin_),
        aM[0] - s_dev / (2 * Gs),
        aM[1],
    ]
    unknowns = list(aL) + list(aM)
    sol2 = sp.solve(eqs, unknowns, dict=True)
    assert len(sol2) == 1, "n=2 system not uniquely solvable"
    sol2 = sol2[0]
    s_tt_L2 = sp.simplify(L2["s_tt"].subs(subL).subs(sol2) / cos_)     # radial profile of the cos2θ amplitude
    T2 = -sp.integrate(s_tt_L2, (r, Ri, Ro))
    M2 = sp.integrate(s_tt_L2 * (r - Rm), (r, Ri, Ro))
    ur2_R = sp.simplify((L2["ur"].subs(subL).subs(sol2) / cos_).subs(r, Rm))
    ur2_ff = s_dev * Rm / (2 * Gs)
    out.update(T2=float(T2), M2=float(M2), ur2_R=float(ur2_R), ur2_ff=float(ur2_ff),
               I2=float(ur2_R / ur2_ff))
    return out


def es_thin_ring_n0_external(E_s, nu_s, E_l, nu_l, t, R, p_m):
    """thin-ring n=0 external loading (derived earlier): q = 2(1-nu) p_m /(1 + C(1-nu))"""
    C = E_s * R * (1 - nu_l**2) / (E_l * t * (1 - nu_s**2))
    return 2 * (1 - nu_s) * p_m * R / (1 + C * (1 - nu_s)), C


if __name__ == "__main__":
    here = Path(__file__).parent
    results = {}

    # ---- Case A_external (es_verif): E=48e6, nu=0.34, K=0.5, P=600 kPa, El=25e9, nul=0.15, t=0.125, R=5
    E_s, nu_s, K, P, E_l, nu_l, t, R = 48e6, 0.34, 0.5, 600e3, 25e9, 0.15, 0.125, 5.0
    p_iso = -(1 + K) * P / 2          # tension +
    s_dev = (1 - K) * P / 2           # sigma_xx - sigma_yy)/2 = (-KP + P)/2
    ex = solve_external(E_s, nu_s, E_l, nu_l, R - t / 2, R + t / 2, p_iso, s_dev)
    thinT0, C = es_thin_ring_n0_external(E_s, nu_s, E_l, nu_l, t, R, (1 + K) * P / 2)
    fe = {r_["name"]: r_ for r_ in json.load(open(f"{ROOT}/proj/es_verif/es_results.json"))}["A_external"]
    print("=== A_external  (C=%.4f) ===" % C)
    print(f"  T0  exact-solid {ex['T0']/1e3:9.2f}   thin-ring {thinT0/1e3:9.2f}   FE {fe['fe']['T0']/1e3:9.2f} kN/m"
          f"   -> FE/exact = {fe['fe']['T0']/ex['T0']:.4f},  thin/exact = {thinT0/ex['T0']:.4f}")
    print(f"  T2  exact-solid {ex['T2']/1e3:9.2f}   FE {fe['fe']['T2']/1e3:9.2f}   -> FE/exact = {fe['fe']['T2']/ex['T2']:.4f}")
    print(f"  M2  exact-solid {ex['M2']/1e3:9.3f}   FE {fe['fe']['M2']/1e3:9.3f}   -> FE/exact = {fe['fe']['M2']/ex['M2']:.4f}")
    results["A_external"] = dict(exact=ex, thin_T0=thinT0, fe=fe["fe"])

    # ---- 3c SD runs (dry external, 1D-constrained far field): P=du=1e5, K=K0'=nu/(1-nu)
    k0 = json.load(open(f"{ROOT}/proj/drawdown_k0/k0_results.json"))
    du = 1e5
    K0p = nu_s / (1 - nu_s)
    print(f"\n=== 3c dry SD  (P=Δu=100 kPa, K=K0'={K0p:.4f}, t=0.125, R=5) ===")
    for rr in k0:
        El = {"El_25e9": 25e9, "El_2.5e9": 2.5e9, "El_5e8": 5e8}[rr["name"]]
        ex = solve_external(E_s, nu_s, El, nu_l, R - t / 2, R + t / 2, -(1 + K0p) * du / 2, (1 - K0p) * du / 2)
        n2 = rr["n2"]; n0 = rr["n0"]
        print(f"  {rr['name']:9s}  T0 exact {ex['T0']/1e3:8.2f} vs SD {n0['dq0_over_du_sd']*du*R/1e3:8.2f} ({n0['dq0_over_du_sd']*du*R/ex['T0']:.4f}) | "
              f"T2 exact {ex['T2']/1e3:8.3f} vs SD {n2['T2_sd']/1e3:8.3f} ({n2['T2_sd']/ex['T2']:.4f}) vs HM {n2['T2_hm']/1e3:8.3f} | "
              f"M2 exact {ex['M2']/1e3:7.4f} vs SD {n2['M2_sd']/1e3:7.4f} ({n2['M2_sd']/ex['M2']:.4f})")
        results[f"k0_{rr['name']}"] = dict(exact=ex, sd=n2, n0=n0)

    # ---- 3d finite layer: local reduction test.  t=0.25, El=25e9, du_local = 0.375e5
    d3 = json.load(open(f"{ROOT}/proj/drawdown_3d/dd3d_results.json"))["tun_step"]
    e = d3["hist"][-1]; dul = d3["params"]["du_local"]
    t3 = 0.25
    ex = solve_external(E_s, nu_s, 25e9, nu_l, R - t3 / 2, R + t3 / 2, -(1 + K0p) * dul / 2, (1 - K0p) * dul / 2)
    print(f"\n=== 3d finite layer, local reduction (Δu_local={dul/1e3:.1f} kPa, t=0.25, F≈204) ===")
    print(f"  T2 exact(local ext. loading) {ex['T2']/1e3:8.3f} vs HM {e['T2']/1e3:8.3f} kN/m  ({e['T2']/ex['T2']:.4f})")
    print(f"  M2 exact                    {ex['M2']/1e3:8.3f} vs HM {e['M2']/1e3:8.3f} kNm/m ({e['M2']/ex['M2']:.4f})")
    print(f"  I2 (ovalisation / free-field) exact {ex['I2']:.4f}")
    results["d3_tun_step"] = dict(exact=ex, hm=dict(T2=e["T2"], M2=e["M2"]))

    (here / "exact_ring_results.json").write_text(json.dumps(results, indent=1, default=float))
    print("\nsaved exact_ring_results.json")
