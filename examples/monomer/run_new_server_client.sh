#!/bin/bash
# Same role as ../../run_new_server_client.sh (portable tree) but paths assume this
# repo is cwd: fasta_dir, out/, and OPENFOLD_SERVER_URL.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

export OPENFOLD_HOST="${OPENFOLD_HOST:-127.0.0.1}"
export OPENFOLD_PORT="${OPENFOLD_PORT:-8000}"
export OPENFOLD_SERVER_URL="${OPENFOLD_SERVER_URL:-http://${OPENFOLD_HOST}:${OPENFOLD_PORT}}"

python3 client_openfold_server.py fasta_dir \
    --output_dir out \
    --config_preset seq_model_esm1b_ptm \
    --server_url "${OPENFOLD_SERVER_URL}"
