#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 CONFIG GPUS WORK_DIR [train.py options...]" >&2
    exit 2
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CONFIG=$1
GPUS=$2
WORK_DIR=$3
shift 3
python tools/preflight.py --config "$CONFIG"
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python -m torch.distributed.launch --nproc_per_node="$GPUS" --master_port="${PORT:-29500}" \
    train.py "$CONFIG" --work-dir="$WORK_DIR" --launcher pytorch "$@"
