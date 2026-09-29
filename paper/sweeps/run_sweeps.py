"""P2 (time-step convergence of the monotonicity claim) + P3 (design-chart sweeps) on the 3d pipeline."""
import json, sys, time
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, f"{ROOT}/proj/drawdown_3d")
import drawdown_3d as d3
import os as _os, pathlib as _pl
# Repository root: override with TD_ROOT if the tree is moved.
ROOT = _os.environ.get("TD_ROOT") or str(_pl.Path(__file__).resolve().parents[2])

HERE = Path(__file__).parent
d3.WORK = HERE / "runs"; d3.WORK.mkdir(exist_ok=True, parents=True)
base = d3.Case3d("base", n_r=6)   # 6 elements through the lining thickness (3 under-resolves T2 by ~4 % at t/R=0.05)

cases = [
    # P2 — time refinement of the reference case (per_decade 8 -> 16 -> 32)
    replace(base, name="P2_r1", per_decade=8),
    replace(base, name="P2_r2", per_decade=16),
    replace(base, name="P2_r4", per_decade=32),
    # P3a — lining flexibility via thickness (F ~ 1630, 204, 25)
    replace(base, name="P3_t0.125", t=0.125),
    replace(base, name="P3_t0.50", t=0.50),
    # P3b — Poisson ratio -> K0'
    replace(base, name="P3_nu0.25", nu=0.25),
    replace(base, name="P3_nu0.45", nu=0.45),
    # P3c — tunnel depth ratio z_t/H  (0.25, 0.5; base is 0.375)
    replace(base, name="P3_zt10", z_t=10.0),
    replace(base, name="P3_zt20", z_t=20.0),
    # P3d — radius (obstacle amplification vs R/H)
    replace(base, name="P3_R2.5", R=2.5, t=0.125),
]
which = sys.argv[1:] or [c.name for c in cases]
res_file = HERE / "sweep_results.json"
out = json.load(open(res_file)) if res_file.exists() else {}
for c in cases:
    if c.name not in which:
        continue
    if c.name in out and out[c.name].get("ok"):
        print(f"{c.name:11s} skip (done)", flush=True); continue
    t0 = time.time()
    r = d3.run(c)
    r["wall_s"] = time.time() - t0
    r["case"] = {k: getattr(c, k) for k in ("t", "nu", "z_t", "R", "H", "El", "per_decade", "n_r")}
    out[c.name] = r
    e = r["hist"][-1] if r.get("ok") else {}
    if r.get("ok"):
        thetas = {k: d3.overshoot(r["hist"], k)[0] for k in ("T_crn", "T_spr", "T_inv", "M_crn", "M_spr", "M_inv", "T2", "M2")}
        print(f"{c.name:11s} ok  {r['wall_s']:5.0f}s  cells={r['mesh']['n_cells']}  T2={e['T2']/1e3:7.2f} M2={e['M2']/1e3:7.3f}  "
              f"uy_c={e['uy_centre']*1e3:+.2f}mm  p_crn={e['p_crown']/1e5:+.3f} p_inv={e['p_invert']/1e5:+.3f}  "
              f"Θmax={max(thetas.values()):.4f}", flush=True)
    else:
        print(f"{c.name:11s} FAILED: {r['msg'][-300:]}", flush=True)
    (HERE / "sweep_results.json").write_text(json.dumps(out, indent=1, default=float))
print("done")
