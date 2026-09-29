import numpy as np, json, sys
sys.path.insert(0,f"{ROOT}/paper/theory")
import exact_ring as ex
import os as _os, pathlib as _pl
# Repository root: override with TD_ROOT if the tree is moved.
ROOT = _os.environ.get("TD_ROOT") or str(_pl.Path(__file__).resolve().parents[2])
E, nul, R, t = 48e6, 0.15, 5.0, 0.25
nus=[0.25,0.30,0.34,0.40,0.45]
Fs=np.logspace(0,5,16)
data={}
for nu in nus:
    K0p=nu/(1-nu); rows=[]
    for F in Fs:
        El=E*R**3*(1-nul**2)/(F*t**3/12*(1-nu**2))
        s=ex.solve_external(E,nu,El,nul,R-t/2,R+t/2,-(1+K0p)/2*1.0,(1-K0p)/2*1.0)
        rows.append(dict(F=float(F),T2n=s["T2"]/R,M2n=s["M2"]/R**2,I2=s["I2"]))
    data[str(nu)]=rows
    json.dump(data, open(f"{ROOT}/paper/theory/chart_data_F1_partial.json","w"))
    print("done nu",nu, flush=True)
json.dump(data, open(f"{ROOT}/paper/theory/chart_data_F1.json","w"))
print("WROTE")
