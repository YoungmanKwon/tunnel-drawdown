"""
Scoping: exact plane-strain n=3 mode for a bonded elastic annulus in an infinite medium.

Far field (total stress increment, tension +) for a drawdown linear in depth:
    sigma_xx = 2 s (1 - y/z_t),  sigma_yy = 0,   y = r sin(theta) measured upward from the axis,
    s = (1 - K) du_l / 2.
The gradient part  sigma_xx = -2 s y / z_t  has Airy function  phi = -(s/3) y^3 / z_t, whose
sin(3 theta) component is
    phi_3 = (s / (12 z_t)) r^3 sin(3 theta),
a single Michell term: sigma_rr = -(s/2z_t) r sin3th, sigma_tt = +(s/2z_t) r sin3th,
sigma_rt = -(s/2z_t) r cos3th.  No pore-pressure term enters the n=3 mode.

Displacement ansatz for mode n (odd, sin):
    u_r = sum a_i r^{p_i} sin(n th),  u_th = sum b_i r^{p_i} cos(n th),  p_i in {n-1, n+1, -n+1, -n-1}.
Medium: r^{n-1} amplitude fixed by the far field, r^{n+1} excluded, two decaying terms free.
Ring: four free.  Six equations: inner surface traction-free (2), bonded interface (4).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import sympy as sp

sys.path.insert(0, str(Path(__file__).parent))
from exact_ring import r, th, G, nu, strains_stresses, equilibrium, solve_external


def general_n(n, tag, odd):
    """Navier-compatible displacement field of circumferential mode n with 4 free constants."""
    a = sp.symbols(" ".join(f"a{i}{tag}" for i in range(4)))
    b = sp.symbols(" ".join(f"b{i}{tag}" for i in range(4)))
    pw = [n - 1, n + 1, -n + 1, -n - 1]
    if odd:
        fr, ft = sp.sin(n * th), sp.cos(n * th)
    else:
        fr, ft = sp.cos(n * th), sp.sin(n * th)
    ur = sum(ai * r**p for ai, p in zip(a, pw)) * fr
    ut = sum(bi * r**p for bi, p in zip(b, pw)) * ft
    s_rr, s_tt, s_rt = strains_stresses(ur, ut)
    eq1, eq2 = equilibrium(s_rr, s_tt, s_rt)
    e1 = sp.expand(sp.simplify(eq1 / fr))
    e2 = sp.expand(sp.simplify(eq2 / ft))
    conds = []
    for e in (e1, e2):
        num, _ = sp.fraction(sp.cancel(sp.together(e * r ** (n + 3))))
        conds += sp.Poly(sp.expand(num), r).coeffs()
    sol = sp.solve(conds, list(b), dict=True)
    assert len(sol) == 1, sol
    sol = sol[0]
    ur, ut = ur.subs(sol), ut.subs(sol)
    s_rr, s_tt, s_rt = [sp.simplify(x.subs(sol)) for x in (s_rr, s_tt, s_rt)]
    q1, q2 = equilibrium(s_rr, s_tt, s_rt)
    assert sp.simplify(q1) == 0 and sp.simplify(q2) == 0
    return dict(a=a, ur=ur, ut=ut, s_rr=s_rr, s_tt=s_tt, s_rt=s_rt, fr=fr, ft=ft)


def solve_n3(E_s, nu_s, E_l, nu_l, Ri, Ro, s, z_t):
    """Lining T3, M3 (sin 3theta amplitudes) for the far field above."""
    Gs, Gl = E_s / (2 * (1 + nu_s)), E_l / (2 * (1 + nu_l))
    Rm = (Ri + Ro) / 2
    L, M_ = general_n(3, "L", odd=True), general_n(3, "M", odd=True)
    subL = {G: Gl, nu: nu_l}
    subM = {G: Gs, nu: nu_s}
    aL, aM = L["a"], M_["a"]
    fr, ft = L["fr"], L["ft"]

    # far field: the r^2 displacement term of the medium alone must give sigma_rr = -(s/2z_t) r sin3th
    ff = M_["s_rr"].subs(subM).subs({aM[1]: 0, aM[2]: 0, aM[3]: 0})
    ff_coef = sp.simplify(ff / (r * fr))            # linear in aM[0]
    a0_ff = sp.solve(sp.Eq(ff_coef, -s / (2 * z_t)), aM[0])[0]
    # consistency: sigma_tt and sigma_rt of that term must match the Michell term too
    chk_tt = sp.simplify(M_["s_tt"].subs(subM).subs({aM[0]: a0_ff, aM[1]: 0, aM[2]: 0, aM[3]: 0}) / (r * fr))
    chk_rt = sp.simplify(M_["s_rt"].subs(subM).subs({aM[0]: a0_ff, aM[1]: 0, aM[2]: 0, aM[3]: 0}) / (r * ft))
    assert abs(float(chk_tt) - s / (2 * z_t)) < 1e-9 * abs(s / z_t), chk_tt
    assert abs(float(chk_rt) + s / (2 * z_t)) < 1e-9 * abs(s / z_t), chk_rt

    def at(e, rr): return e.subs(r, rr)
    eqs = [
        sp.simplify(at(L["s_rr"].subs(subL), Ri) / fr),
        sp.simplify(at(L["s_rt"].subs(subL), Ri) / ft),
        sp.simplify(at(L["s_rr"].subs(subL) - M_["s_rr"].subs(subM), Ro) / fr),
        sp.simplify(at(L["s_rt"].subs(subL) - M_["s_rt"].subs(subM), Ro) / ft),
        sp.simplify(at(L["ur"].subs(subL) - M_["ur"].subs(subM), Ro) / fr),
        sp.simplify(at(L["ut"].subs(subL) - M_["ut"].subs(subM), Ro) / ft),
        aM[0] - a0_ff,
        aM[1],
    ]
    sol = sp.solve(eqs, list(aL) + list(aM), dict=True)
    assert len(sol) == 1, "n=3 system not uniquely solvable"
    sol = sol[0]
    s_tt_L = sp.simplify(L["s_tt"].subs(subL).subs(sol) / fr)
    T3 = -sp.integrate(s_tt_L, (r, Ri, Ro))
    M3 = sp.integrate(s_tt_L * (r - Rm), (r, Ri, Ro))
    return dict(T3=float(T3), M3=float(M3))


if __name__ == "__main__":
    # reference finite-layer case: E=48 MPa, nu=0.34, El=25 GPa, nul=0.15, t=0.25, R=5, z_t=15
    E_s, nu_s, E_l, nu_l = 48e6, 0.34, 25e9, 0.15
    R, z_t, du_l = 5.0, 15.0, 37.5e3
    K = nu_s / (1 - nu_s)
    s = (1 - K) * du_l / 2

    def run(t, E_s_=E_s, nu_s_=nu_s, z_t_=z_t, R_=R):
        K_ = nu_s_ / (1 - nu_s_); s_ = (1 - K_) * du_l / 2
        Ri, Ro = R_ - t / 2, R_ + t / 2
        ex2 = solve_external(E_s_, nu_s_, E_l, nu_l, Ri, Ro, -(1 + K_) * du_l / 2, s_)
        ex3 = solve_n3(E_s_, nu_s_, E_l, nu_l, Ri, Ro, s_, z_t_)
        F = E_s_ * (1 - nu_l**2) * R_**3 / (E_l * (t**3 / 12) * (1 - nu_s_**2))
        rT, rM = ex3["T3"] / ex2["T2"], ex3["M3"] / ex2["M2"]
        return dict(F=F, T2=ex2["T2"], M2=ex2["M2"], T3=ex3["T3"], M3=ex3["M3"],
                    T3_T2=rT, M3_M2=rM, aT=-rT / (R_ / z_t_), aM=-rM / (R_ / z_t_))

    print("reference case (t=0.25, F~204): measured T3/T2 = -0.110, M3/M2 = -0.181, aT=0.33, aM=0.54")
    ref = run(0.25)
    for k, v in ref.items(): print(f"  {k:6s} {v:+.4f}" if abs(v) < 1e3 else f"  {k:6s} {v:+.4e}")

    print("\nsweep over t (E=48 MPa): measured aT 0.32..0.36, aM 0.27..0.60")
    out = {}
    for t in (0.125, 0.25, 0.50):
        rr = run(t); out[f"t={t}"] = rr
        print(f"  t={t:5.3f} F={rr['F']:8.1f}  T3/T2={rr['T3_T2']:+.4f}  M3/M2={rr['M3_M2']:+.4f}  aT={rr['aT']:.3f}  aM={rr['aM']:.3f}")
    rr = run(0.50, E_s_=8e6); out["E=8,t=0.5"] = rr
    print(f"  E=8MPa t=0.5 F={rr['F']:8.1f}  T3/T2={rr['T3_T2']:+.4f}  M3/M2={rr['M3_M2']:+.4f}  aT={rr['aT']:.3f}  aM={rr['aM']:.3f}")
    Path(__file__).with_name("n3_ring_scoping.json").write_text(json.dumps(out, indent=1))
