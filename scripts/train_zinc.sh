#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model="${1:?model: molalign3d or molflae}"
data="${2:?ZINC multi-conformer LMDB path}"
splits="${3:?Directory containing train_indices.pt, val_indices.pt and test_indices.pt}"
output="${4:?Output directory}"
shift 4
if [[ "$model" != molalign3d && "$model" != molflae ]]; then echo "Only the two released ZINC models are supported." >&2; exit 2; fi
"${MODEL_PYTHON:-python}" "$repo_dir/scripts/train.py" --model "$model" --data "$data" --splits "$splits" --output "$output" --accelerator gpu "$@"
