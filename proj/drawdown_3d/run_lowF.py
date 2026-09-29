import json, sys
from dataclasses import replace
from drawdown_3d import Case3d, run, WORK, HERE

WORK.mkdir(exist_ok=True)
base = Case3d("tun_step")
c = replace(base, name="softF4", E=8e6, t=0.50)
print("F =", c.F, " C =", c.C, " t/R =", c.t / c.R)
r = run(c)
print("ok:", r["ok"])
if not r["ok"]:
    print(r["msg"][:2000]); sys.exit(1)
e = r["hist"][-1]
p = r["params"]
print("T2 =", e["T2"], "M2 =", e["M2"], "dul =", p["du_local"])
out = HERE / "lowF_result.json"
out.write_text(json.dumps({"softF4": r}, indent=1, default=float))
print("saved", out)
