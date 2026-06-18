#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

mkdir -p datasets

python - <<'PY'
import torch
print("torch:", torch.__version__,
      "cuda available:", torch.cuda.is_available(),
      "device count:", torch.cuda.device_count())
PY

cat <<EOF

Done. Activate with:
  source "\$PWD/.venv/bin/activate"
EOF
