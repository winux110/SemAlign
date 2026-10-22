#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 CONFIG CHECKPOINT GPUS [test.py options...]" >&2
    exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3
python tools/preflight.py --config "$CONFIG"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="${PORT:-29500}" \
    test.py "$CONFIG" "$CHECKPOINT" --launcher pytorch "$@"
