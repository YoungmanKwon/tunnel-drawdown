#!/usr/bin/env bash
set -e; cd "$(dirname "$0")"; . .venv/bin/activate
echo "== [1] OGS regression (hm2_1Dbiot) =="; mkdir -p proj/_verif_out
( cd benchmarks/ogs/HydroMechanics/Verification && ogs -o "$PWD/../../../../proj/_verif_out" -r . -l error hm2_1Dbiot.prj && echo "hm2_1Dbiot PASS" )
echo "== [2] Terzaghi + convergence =="; python proj/terzaghi/convergence.py
echo "== [3] Einstein-Schwartz =="; python proj/es_verif/es_verif.py A_itasca A_external
echo "== [4] Exact solid ring vs FE =="; python paper/theory/exact_ring.py
echo; echo "Done. Full runs: proj/drawdown_3b, drawdown_k0, drawdown_3d (see README)."
