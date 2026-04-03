#!/bin/bash
# Same as ../../run_new_server.sh (portable tree) but assumes cwd is the openfoldv21
# repo root (templates/, embeddings_dir/, params/ next to this tree or adjust paths).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

export OPENFOLD_HOST="${OPENFOLD_HOST:-127.0.0.1}"
export OPENFOLD_PORT="${OPENFOLD_PORT:-8000}"

TORCH_COMPILE_ARGS=()
if [[ "${OPENFOLD_TORCH_COMPILE:-0}" == "1" ]]; then
  TORCH_COMPILE_ARGS+=(--torch_compile)
fi
if [[ "${OPENFOLD_TORCH_INDUCTOR_NO_STATIC_LAUNCHER:-0}" == "1" ]]; then
  TORCH_COMPILE_ARGS+=(--torch_inductor_no_static_launcher)
fi

PAD_ARGS=()
if [[ -n "${OPENFOLD_PAD_SEQLEN_MULTIPLE:-}" ]]; then
  PAD_ARGS+=(--pad_seqlen_multiple "${OPENFOLD_PAD_SEQLEN_MULTIPLE}")
fi
if [[ "${OPENFOLD_PAD_SEQLEN_TRACE_BUCKETS:-0}" == "1" ]]; then
  PAD_ARGS+=(--pad_seqlen_trace_buckets)
fi

WARMUP_ARGS=()
if [[ -n "${OPENFOLD_STARTUP_WARMUP_FASTA_DIR:-}" ]]; then
  WARMUP_ARGS+=(--startup_warmup_fasta_dir "${OPENFOLD_STARTUP_WARMUP_FASTA_DIR}")
fi
if [[ -n "${OPENFOLD_STARTUP_WARMUP_MAX_SEQUENCES:-}" ]]; then
  WARMUP_ARGS+=(--startup_warmup_max_sequences "${OPENFOLD_STARTUP_WARMUP_MAX_SEQUENCES}")
fi

python3 serve_openfold.py \
    templates \
    --use_precomputed_alignments embeddings_dir \
    --output_dir serve_out \
    --model_device cuda:0 \
    --config_preset seq_model_esm1b_ptm \
    --openfold_checkpoint_path params/openfold_soloseq_params/seq_model_esm1b_ptm.pt \
    --host "${OPENFOLD_HOST}" \
    --port "${OPENFOLD_PORT}" \
    "${TORCH_COMPILE_ARGS[@]}" \
    "${PAD_ARGS[@]}" \
    "${WARMUP_ARGS[@]}"
