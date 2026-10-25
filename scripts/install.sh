#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
backend=cpu
if [[ "${1:-}" == "--backend" ]]; then backend="${2:?Choose cpu, cuda128 or existing}"; fi
python3 -m venv --system-site-packages "$repo_dir/.venv"
release_python="$repo_dir/.venv/bin/python"
"$release_python" -m pip install 'setuptools>=68' wheel
case "$backend" in
  cpu)
    "$release_python" -m pip install torch==2.8.0+cpu --index-url https://download.pytorch.org/whl/cpu
    pyg_index=https://data.pyg.org/whl/torch-2.8.0+cpu.html ;;
  cuda128)
    "$release_python" -m pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
    pyg_index=https://data.pyg.org/whl/torch-2.8.0+cu128.html ;;
  existing)
    pyg_index="$($release_python -c 'import torch; assert torch.__version__.split("+")[0]=="2.8.0" and torch.version.cuda in (None,"12.8"), "Use cpu/cuda128 installer or install matching PyG wheels manually"; print("https://data.pyg.org/whl/torch-2.8.0+"+("cu128" if torch.version.cuda=="12.8" else "cpu")+".html")')" ;;
  *) echo "Unknown backend: $backend" >&2; exit 2 ;;
esac
"$release_python" -m pip install torch-scatter==2.1.2 torch-cluster==1.6.3 --only-binary=:all: -f "$pyg_index"
"$release_python" -m pip install -r "$repo_dir/requirements.txt"
"$release_python" -m pip install --no-build-isolation --no-deps -e "$repo_dir"
