"""
Bonded (no-slip) versus smooth (full-slip) interface for the exact thick-ring solution,
modes n = 2 and n = 3, same far fields as exact_ring.py (n = 2) and n3_ring.py (n = 3).

Smooth interface at r = Ro: sigma_rr continuous, u_r continuous, sigma_rt = 0 on both sides.
Bonded interface:           sigma_rr, sigma_rt, u_r, u_theta continuous.

Thin-shell forms for comparison:
  bonded    -- Hoeg (1968) coefficients C2, C3 (manuscript eqs C1-C3, dT2dM2);
  full slip -- inextensional thin shell in the same far field,
               M2 = 12 s R^2 (1-nu)/[(1-nu)F + 3(5-6nu)],  |T2| = M2/R,
               a_M = (2/3) [(1-nu)F + 3(5-6nu)] / [(1-nu)F + 8(7-8nu)].

Loading type: for every mode n >= 2 the external-loading forces are 4(1-nu)/(3-4nu) times the
excavation-loading (relative-stiffness) forces; checked here for n = 2, 3 and both interfaces.

Output: n3_interface.json with T2, M2, T3, M3 and a_T, a_M for both interface conditions.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import sympy as sp

sys.path.insert(0, str(Path(__file__).parent))
from exact_ring import r, G, nu
from n3_ring import general_n


def solve_mode(n, E_s, nu_s, E_l, nu_l, Ri, Ro, amp, interface, loading="external"):
    """Lining T_n, M_n for far-field sigma_rr amplitude `amp` of mode n.
    n = 2: far field sigma_rr = amp*cos2th from the r^1 term (uniform deviator s = amp).
    n = 3: far field sigma_rr = amp*r*sin3th from the r^2 term (amp = -s/(2 z_t)).
    loading = "external":   ring and ground stress-free before the far field is applied (drawdown).
    loading = "excavation": ground carries the far field before the ring is installed; the ring
                            sees only the displacement from releasing that stress at the opening."""
    odd = (n == 3)
    Gs, Gl = E_s / (2 * (1 + nu_s)), E_l / (2 * (1 + nu_l))
    Rm = (Ri + Ro) / 2
    L, M_ = general_n(n, "L", odd=odd), general_n(n, "M", odd=odd)
    subL, subM = {G: Gl, nu: nu_l}, {G: Gs, nu: nu_s}
    aL, aM = L["a"], M_["a"]
    fr, ft = L["fr"], L["ft"]

    # far field: medium r^{n-1} term only, fixed by sigma_rr amplitude; r^{n+1} term zero
    ff = M_["s_rr"].subs(subM).subs({aM[1]: 0, aM[2]: 0, aM[3]: 0})
    coef = sp.simplify(ff / (r ** (n - 2) * fr))
    a0_ff = sp.solve(sp.Eq(coef, amp), aM[0])[0]

    def at(e, rr): return e.subs(r, rr)
    sM = {k: M_[k].subs(subM) for k in ("s_rr", "s_rt", "ur", "ut")}
    if loading == "excavation":      # displacements measured from the in-situ state
        for k in ("ur", "ut"):
            sM[k] = sM[k].subs(aM[0], 0)
    elif loading != "external":
        raise ValueError(loading)
    eqs = [
        sp.simplify(at(L["s_rr"].subs(subL), Ri) / fr),
        sp.simplify(at(L["s_rt"].subs(subL), Ri) / ft),
        sp.simplify(at(L["s_rr"].subs(subL) - sM["s_rr"], Ro) / fr),
        sp.simplify(at(L["ur"].subs(subL) - sM["ur"], Ro) / fr),
    ]
    if interface == "bonded":
        eqs += [sp.simplify(at(L["s_rt"].subs(subL) - sM["s_rt"], Ro) / ft),
                sp.simplify(at(L["ut"].subs(subL) - sM["ut"], Ro) / ft)]
    elif interface == "smooth":
        eqs += [sp.simplify(at(L["s_rt"].subs(subL), Ro) / ft),
                sp.simplify(at(sM["s_rt"], Ro) / ft)]
    else:
        raise ValueError(interface)
    eqs += [aM[0] - a0_ff, aM[1]]
    sol = sp.solve(eqs, list(aL) + list(aM), dict=True)
    assert len(sol) == 1, f"n={n} {interface}: not uniquely solvable"
    sol = sol[0]
    s_tt = sp.simplify(L["s_tt"].subs(subL).subs(sol) / fr)
    T = -sp.integrate(s_tt, (r, Ri, Ro))
    M = sp.integrate(s_tt * (r - Rm), (r, Ri, Ro))
    return float(T), float(M)


def hoeg_thin(C, F, v):
    """Bonded thin shell: T2/(sR) = 1 - C3, M2/(sR^2) = (1 + C2 + C3)/2."""
    N2 = (1 - v)**2 * C + (1 - v) - 3 / F * ((1 - v) * C + 4)
    D2 = (1 - v)**2 * C + (1 - v) * (3 - 2 * v) + 3 / F * ((1 - v) * (5 - 6 * v) * C + 4 * (3 - 4 * v))
    C2 = -2 * N2 / D2
    C3 = ((1 - v) * C - 2 - C2 * ((1 - v) * C + 4 * v)) / (3 * ((1 - v) * C + 2))
    return 1 - C3, 0.5 * (1 + C2 + C3)


def slip_thin(F, v):
    """Full-slip inextensional thin shell: T2/(sR), M2/(sR^2), a_T = a_M."""
    M2 = 12 * (1 - v) / (F * (1 - v) + 3 * (5 - 6 * v))
    a = 2 * (F * (1 - v) + 3 * (5 - 6 * v)) / (3 * (F * (1 - v) + 8 * (7 - 8 * v)))
    return M2, M2, a


def case(t, E_s=48e6, nu_s=0.34, E_l=25e9, nu_l=0.15, R=5.0, z_t=15.0, du_l=37.5e3):
    K = nu_s / (1 - nu_s)
    s = (1 - K) * du_l / 2
    Ri, Ro = R - t / 2, R + t / 2
    F = E_s * (1 - nu_l**2) * R**3 / (E_l * (t**3 / 12) * (1 - nu_s**2))
    C = E_s * R * (1 - nu_l**2) / (E_l * t * (1 - nu_s**2))
    out = dict(t=t, E=E_s, F=F, C=C, R=R, z_t=z_t)
    hT, hM = hoeg_thin(C, F, nu_s)
    sT, sM, sa = slip_thin(F, nu_s)
    out["bonded_thin"] = dict(T2_sR=hT, M2_sR2=hM)
    out["smooth_thin"] = dict(T2_sR=sT, M2_sR2=sM, aT=sa, aM=sa)
    lam = 4 * (1 - nu_s) / (3 - 4 * nu_s)
    out["lambda"] = lam
    for itf in ("bonded", "smooth"):
        T2, M2 = solve_mode(2, E_s, nu_s, E_l, nu_l, Ri, Ro, s, itf)
        T3, M3 = solve_mode(3, E_s, nu_s, E_l, nu_l, Ri, Ro, -s / (2 * z_t), itf)
        T2x, M2x = solve_mode(2, E_s, nu_s, E_l, nu_l, Ri, Ro, s, itf, "excavation")
        T3x, M3x = solve_mode(3, E_s, nu_s, E_l, nu_l, Ri, Ro, -s / (2 * z_t), itf, "excavation")
        for ext, exc in ((T2, T2x), (M2, M2x), (T3, T3x), (M3, M3x)):
            assert abs(ext / exc / lam - 1) < 1e-8, (itf, ext / exc, lam)
        out[itf] = dict(T2_sR=T2 / (s * R), M2_sR2=M2 / (s * R**2),
                        aT=-(T3 / T2) / (R / z_t), aM=-(M3 / M2) / (R / z_t),
                        T2=T2, M2=M2, T3=T3, M3=M3)
    return out


if __name__ == "__main__":
    rows = [case(0.125), case(0.25), case(0.50), case(0.50, E_s=8e6)]
    hdr = f"{'t':>6} {'E/MPa':>6} {'F':>8} | {'T2/sR':>14} {'M2/sR2':>16} | {'a_T':>12} {'a_M':>12}"
    print(hdr); print("-" * len(hdr))
    for c in rows:
        b, s = c["bonded"], c["smooth"]
        print(f"{c['t']:6.3f} {c['E']/1e6:6.0f} {c['F']:8.1f} | "
              f"{b['T2_sR']:6.3f}/{s['T2_sR']:6.3f} {b['M2_sR2']:7.4f}/{s['M2_sR2']:7.4f} | "
              f"{b['aT']:5.3f}/{s['aT']:5.3f} {b['aM']:5.3f}/{s['aM']:5.3f}")
    print("(bonded / smooth, exact thick ring)")
    print(f"external/excavation = 4(1-nu)/(3-4nu) = {rows[0]['lambda']:.4f} for every T_n, M_n above")
    print()
    for c in rows:
        b, h, m = c["bonded"], c["bonded_thin"], c["smooth_thin"]
        print(f"F={c['F']:7.1f}  bonded thin T2/sR={h['T2_sR']:.3f} M2/sR2={h['M2_sR2']:.4f} | "
              f"slip thin T2/sR={m['T2_sR']:.3f} M2/sR2={m['M2_sR2']:.4f} a_M={m['aM']:.3f}")
    Path(__file__).with_name("n3_interface.json").write_text(json.dumps(rows, indent=1))
