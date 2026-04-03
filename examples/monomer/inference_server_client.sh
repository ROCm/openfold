#!/bin/bash
# Drive a running serve_openfold instance the same way inference.sh drives
# run_pretrained_openfold.py: FASTA directory -> one POST /predict per sequence,
# structures under ${OUTPUT_DIR}/predictions/, timings in ${OUTPUT_DIR}/timings.json.
#
# Start the server separately, e.g.:
#   python3 serve_openfold.py "${MMCIF_DIR}" \
#     --use_precomputed_alignments "${PRECOMPUTED_ALIGNMENT_DIR}" \
#     --output_dir serve_state --model_device cuda:0 \
#     --config_preset model_1_ptm --openfold_checkpoint_path /path/to.ckpt

set -euo pipefail

export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LIBRARY_PATH:-}"

export FASTA_DIR="${FASTA_DIR:-./fasta_dir}"
export OUTPUT_DIR="${OUTPUT_DIR:-./}"
export CONFIG_PRESET="${CONFIG_PRESET:-model_1_ptm}"
export OPENFOLD_SERVER_URL="${OPENFOLD_SERVER_URL:-http://127.0.0.1:8000}"

python3 client_openfold_server.py "${FASTA_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --config_preset "${CONFIG_PRESET}" \
  --server_url "${OPENFOLD_SERVER_URL}"
