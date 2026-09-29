"""Coupled HM run of the Section 5 design example (Table 5), for an end-to-end check."""
import json, sys
from dataclasses import replace
from drawdown_3d import Case3d, run, WORK, HERE

WORK.mkdir(exist_ok=True)
base = Case3d("tun_step")
# Table 5: H=40, z_t=20, E=10 MPa, nu'=0.30, R=3.0, t=0.30, El=30 GPa, du=147 kPa
c = replace(base, name="example", E=10e6, nu=0.30, El=30e9, t=0.30, R=3.0, z_t=20.0, H=40.0, du=147e3)
print("F =", c.F, " C =", c.C, " t/R =", c.t / c.R)
r = run(c)
print("ok:", r["ok"])
if not r["ok"]:
    print(r["msg"][:2000]); sys.exit(1)
e = r["hist"][-1]; p = r["params"]
print({k: e[k] for k in ("T0","T1","T2","T3","T_inv","T_crn","T_spr","M2","M3","M_inv","M_crn","M_spr","uy_centre")})
print("dul =", p["du_local"])
(HERE / "example_result.json").write_text(json.dumps({"example": r}, indent=1, default=float))
print("saved")
