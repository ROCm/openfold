# OpenFold HTTP inference server

This branch adds a **long-running inference path**: keep the model (and optional `torch.compile` wrapper) in memory and submit sequences over HTTP instead of one `run_pretrained_openfold.py` process per batch.

## Components

| File | Role |
|------|------|
| `serve_openfold.py` | FastAPI app: load weights once, `POST /predict` with `tag` + `sequence`. Dependencies: `pip install fastapi uvicorn`. Templates and `--use_precomputed_alignments` are configured on the server (same filesystem layout as batch inference). |
| `client_openfold_server.py` | CLI client: scan a `fasta_dir` like the batch driver, one request per target, write `output_dir/predictions/*_unrelaxed.pdb` and merge `output_dir/timings.json`. |

The server does **not** run Amber relaxation; the batch script’s `--skip_relaxation` behavior is the usual comparison.

## Example shell wrappers

Run scripts from the **repository root** and edit paths (`templates/`, `embeddings_dir/`, `fasta_dir/`, checkpoints) to match your machine.

- `examples/monomer/run_new_server.sh` — start `serve_openfold.py` (example SoloSeq-style flags).
- `examples/monomer/run_new_server_client.sh` — call the client against `OPENFOLD_SERVER_URL` (defaults to `http://$OPENFOLD_HOST:$OPENFOLD_PORT`).
- `examples/monomer/inference_server_client.sh` — generic client using `FASTA_DIR`, `OUTPUT_DIR`, `CONFIG_PRESET`, `OPENFOLD_SERVER_URL`.

Optional environment variables for the example server script: `OPENFOLD_HOST`, `OPENFOLD_PORT`, `OPENFOLD_TORCH_COMPILE=1`, `OPENFOLD_TORCH_INDUCTOR_NO_STATIC_LAUNCHER=1` (see script comments). For shape bucketing: `OPENFOLD_PAD_SEQLEN_MULTIPLE` (e.g. `128`), `OPENFOLD_PAD_SEQLEN_TRACE_BUCKETS=1` (50-residue buckets like batch `--trace_model`).

## Sequence-length padding (compile bucketing)

`serve_openfold.py` can pad the feature dict before inference:

- **`--pad_seqlen_multiple M`** — round the effective length up to the next multiple of `M` so more requests share the same tensor shapes (helps **`torch.compile`** / Inductor cache). `M=0` disables.
- **`--pad_seqlen_trace_buckets`** — also take the max with **`round_up_seqlen`** (50-residue steps), matching the bucketing used by **`run_pretrained_openfold.py --trace_model`**.

The effective padded length is `max(natural_length, multiple_bucket, trace_bucket_if_enabled)`. Extra padding adds compute; use with **`--torch_compile`** for the main benefit.

## Startup warmup (optional)

The server does not run inference until a request arrives unless you enable warmup:

- **`--startup_warmup_fasta_dir DIR`** — after loading the model, run one forward per monomer FASTA in `DIR` (same discovery rules as `client_openfold_server.py`), then start HTTP. Use the same `embeddings_dir/tag` layout as for normal requests.
- **`--startup_warmup_max_sequences N`** — if `N > 0`, only warm up the first `N` targets after shortest-first sorting (`0` = all).

Example env (see `examples/monomer/run_new_server.sh`): `OPENFOLD_STARTUP_WARMUP_FASTA_DIR=fasta_dir`, optionally `OPENFOLD_STARTUP_WARMUP_MAX_SEQUENCES=3`.

## `torch.compile`

Enable on the server with **`--torch_compile`** (and optional Inductor cache / strategy flags; **`serve_openfold.py --help`**). The wrapper is applied at load time; **Inductor may still compile kernels on the first forward** (or first new shape). Use the client’s `inference_warmup` or a dummy request if you want that cost before serving traffic.

## ROCm / HIP notes

- TensorRT-related imports in `script_utils` are **lazy** so inference does not require NVIDIA’s `cuda` Python package at import time.
- On HIP, the structure module avoids the NVIDIA-only inplace attention kernel (standard softmax path).
- `ChunkSizeTuner.tune_chunk_size` is excluded from Dynamo tracing so `torch.compile` can run the template stack.
- `torch.compile` + Inductor on ROCm can still be finicky; `--torch_inductor_no_static_launcher` may help in some setups.
