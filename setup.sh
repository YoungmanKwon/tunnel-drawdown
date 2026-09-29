#!/usr/bin/env bash
# tunnel-drawdown : Linux / WSL / macOS setup.   bash setup.sh
set -e; cd "$(dirname "$0")"
python3 --version
[ -d .venv ] || python3 -m venv .venv
. .venv/bin/activate
python -m pip install -q --upgrade pip wheel
pip install -q -r requirements.txt
# gmsh needs OpenGL libs on headless Linux
if [ "$(uname)" = "Linux" ] && ! python -c "import gmsh" 2>/dev/null; then
  echo "gmsh import failed - installing OpenGL runtime libs (sudo)"; sudo apt-get install -y -qq libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 libgl1 xvfb
fi
ogs --version
python -c "import ogstools, gmsh, pyvista, sympy; print('ogstools', ogstools.__version__, '| pyvista', pyvista.__version__, '| sympy', sympy.__version__)"
echo; echo "Setup complete.  Next: bash verify.sh"
