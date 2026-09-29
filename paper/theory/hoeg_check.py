"""Check the thin-shell external-loading (no-slip) coefficients of Höeg (1968) / Burns & Richard (1964),
as reproduced in Bobet (2003, Appendix B, Eqs. A21-A24), against the exact solid-ring solution.

Bobet's convention: compression positive, theta from the springline (horizontal axis),
    T = (sv+sh)/2 (1+C1) ro + (sv-sh)/2 (1-C3) ro cos2th
    M = (sv-sh)/4 (1+C2+C3) ro^2 cos2th
    C = E(1-nul^2) R / (El t (1-nu^2)),  F = E(1-nul^2) R^3 / (El I (1-nu^2)),  I = t^3/12.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from exact_ring import solve_external


def bobet_coeffs(nu, C, F):
    C1 = (1 - 2 * nu - (1 - nu) * C + (1 - 2 * nu) * C / F) / (1 + (1 - nu) * C + C / F)
    C2 = -2 * ((1 - nu) ** 2 * C + (1 - nu) - ((1 - nu) * C + 4) * 3 / F) / (
        (1 - nu) ** 2 * C + (1 - nu) * (3 - 2 * nu) + ((1 - nu) * (5 - 6 * nu) * C + 4 * (3 - 4 * nu)) * 3 / F)
    C3 = (1 / 3) * ((1 - nu) * C - 2 - C2 * ((1 - nu) * C + 4 * nu)) / ((1 - nu) * C + 2)
    return C1, C2, C3


def thin_shell(nu, C, F, sv, sh, R):
    C1, C2, C3 = bobet_coeffs(nu, C, F)
    T0 = (sv + sh) / 2 * (1 + C1) * R
    T2 = (sv - sh) / 2 * (1 - C3) * R
    M2 = (sv - sh) / 4 * (1 + C2 + C3) * R ** 2
    return T0, T2, M2, (C1, C2, C3)


def compare(E, nu, El, nul, R, t_over_R, sv=1.0, sh=0.0):
    t = t_over_R * R
    Ri, Ro = R - t / 2, R + t / 2
    C = E * (1 - nul ** 2) * R / (El * t * (1 - nu ** 2))
    F = E * (1 - nul ** 2) * R ** 3 / (El * (t ** 3 / 12) * (1 - nu ** 2))
    # exact_ring: tension positive; p_iso = -(sv+sh)/2 ; s_dev = (sxx - syy)/2 = (sv - sh)/2
    ex = solve_external(E, nu, El, nul, Ri, Ro, -(sv + sh) / 2, (sv - sh) / 2)
    T0, T2, M2, cs = thin_shell(nu, C, F, sv, sh, R)
    return dict(t_over_R=t_over_R, C=C, F=F, C1=cs[0], C2=cs[1], C3=cs[2],
                T0_shell=T0, T0_exact=ex["T0"], T2_shell=T2, T2_exact=ex["T2"],
                M2_shell=M2, M2_exact=ex["M2"])


if __name__ == "__main__":
    E, El, nul, R = 48e6, 25e9, 0.15, 5.0
    rows = []
    print(f"{'nu':>5} {'t/R':>6} {'F':>8} | {'T0 sh':>9} {'T0 ex':>9} {'err%':>6} | {'T2 sh':>9} {'T2 ex':>9} {'err%':>6} | {'M2 sh':>9} {'M2 ex':>9} {'err%':>6}")
    for nu in (0.25, 0.34, 0.45):
        for tR in (0.05, 0.025, 0.0125, 0.00625):
            d = compare(E, nu, El, nul, R, tR)
            d["nu"] = nu
            rows.append(d)
            e = lambda a, b: 100 * (a - b) / abs(b)
            print(f"{nu:5.2f} {tR:6.4f} {d['F']:8.1f} | {d['T0_shell']:9.4f} {d['T0_exact']:9.4f} {e(d['T0_shell'], d['T0_exact']):6.2f} | "
                  f"{d['T2_shell']:9.4f} {d['T2_exact']:9.4f} {e(d['T2_shell'], d['T2_exact']):6.2f} | "
                  f"{d['M2_shell']:9.4f} {d['M2_exact']:9.4f} {e(d['M2_shell'], d['M2_exact']):6.2f}")
    # stiff-lining check (different modulus ratio) at the paper's reference geometry
    for El_ in (2.5e9, 250e9):
        d = compare(E, 0.34, El_, nul, R, 0.05); d["nu"] = 0.34; d["El"] = El_
        rows.append(d)
        print(f"El={El_:.1e} t/R=0.05  F={d['F']:.1f}  T2 sh/ex = {d['T2_shell']/d['T2_exact']:.4f}  M2 sh/ex = {d['M2_shell']/d['M2_exact']:.4f}")
    Path(__file__).with_name("hoeg_check.json").write_text(json.dumps(rows, indent=1))
