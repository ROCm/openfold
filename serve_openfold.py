#!/usr/bin/env python3
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Minimal HTTP server: load OpenFold once, run inference per request.
# Requires: pip install fastapi uvicorn
#
# Example:
#   python serve_openfold.py templates \\
#     --use_precomputed_alignments /path/to/embeddings_root \\
#     --output_dir serve_out \\
#     --model_device cuda:0 \\
#     --config_preset seq_model_esm1b_ptm \\
#     --openfold_checkpoint_path params/openfold_soloseq_params/seq_model_esm1b_ptm.pt \\
#     --host 127.0.0.1 --port 8000
#
# Optional compile (same knobs as batch tooling):
#   ... --torch_compile [--torch_compile_strategy full|submodules] \\
#       [--torch_inductor_cache_dir DIR] [--torch_inductor_cache_force_rebuild] \\
#       [--torch_inductor_no_static_launcher]
# Optional shape bucketing (fewer Inductor shapes with --torch_compile):
#   ... [--pad_seqlen_multiple 128] [--pad_seqlen_trace_buckets]
#
#   curl -s -X POST http://127.0.0.1:8000/predict \\
#     -H 'Content-Type: application/json' \\
#     -d '{"tag":"11ba_B","sequence":"ACDEFGHIKLMNPQRSTVWY"}'

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve_openfold")

torch.set_grad_enabled(False)

from openfold.config import model_config
from openfold.data import data_pipeline, feature_pipeline, templates
from openfold.np import protein
from openfold.utils.script_utils import load_models_from_command_line, parse_fasta, prep_output
from openfold.utils.tensor_utils import tensor_tree_map

from openfold.utils.trace_utils import pad_feature_dict_seq

from run_pretrained_openfold import generate_feature_dict, list_files_with_extensions, round_up_seqlen
from scripts.utils import add_data_args


def _round_up_seqlen_multiple(n: int, mult: int) -> int:
    if mult <= 0:
        return n
    return int(math.ceil(n / mult) * mult)


def _prepare_batch(st: AppState, tag: str, seq: str) -> tuple[dict, dict]:
    """Build feature_dict and device batch tensors (shared by /predict and startup warmup)."""
    args = st.args
    alignment_dir = args.use_precomputed_alignments
    feature_dict = generate_feature_dict(
        [tag],
        [seq],
        alignment_dir,
        st.data_processor,
        args,
    )
    n = int(feature_dict["aatype"].shape[-2])
    target_seqlen = n
    if args.pad_seqlen_multiple > 0:
        target_seqlen = max(target_seqlen, _round_up_seqlen_multiple(n, args.pad_seqlen_multiple))
    if args.pad_seqlen_trace_buckets:
        target_seqlen = max(target_seqlen, round_up_seqlen(n))
    if target_seqlen > n:
        logger.info("Padding sequence length %d -> %d for %s", n, target_seqlen, tag)
        feature_dict = pad_feature_dict_seq(feature_dict, target_seqlen)

    processed = st.feature_processor.process_features(
        feature_dict, mode="predict", is_multimer=st.is_multimer
    )
    processed = {
        k: torch.as_tensor(v, device=args.model_device) for k, v in processed.items()
    }
    return feature_dict, processed


def _startup_warmup(state: AppState) -> None:
    """Run inference once per FASTA (monomer) before HTTP, to prime compile/chunk tuning."""
    args = state.args
    fasta_dir = args.startup_warmup_fasta_dir
    if not fasta_dir:
        return
    if state.is_multimer:
        logger.warning("Startup warmup skipped (multimer not implemented on /predict).")
        return
    if not os.path.isdir(fasta_dir):
        raise ValueError(f"--startup_warmup_fasta_dir is not a directory: {fasta_dir}")

    tag_list: list[tuple[str, list[str]]] = []
    seq_list: list[list[str]] = []
    for fasta_file in list_files_with_extensions(fasta_dir, (".fasta", ".fa")):
        fasta_path = os.path.join(fasta_dir, fasta_file)
        with open(fasta_path, "r") as fp:
            data = fp.read()
        tags, seqs = parse_fasta(data)
        if len(tags) != 1:
            logger.warning(
                "Skipping %s for startup warmup (%d sequences; monomer only).",
                fasta_path,
                len(tags),
            )
            continue
        tag = "-".join(tags)
        tag_list.append((tag, tags))
        seq_list.append(seqs)

    seq_sort_fn = lambda target: sum(len(s) for s in target[1])
    sorted_targets = sorted(zip(tag_list, seq_list), key=seq_sort_fn)
    if args.startup_warmup_max_sequences > 0:
        sorted_targets = sorted_targets[: args.startup_warmup_max_sequences]

    logger.info(
        "Startup warmup: %d sequence(s) from %s",
        len(sorted_targets),
        fasta_dir,
    )
    for (tag, _tags), seqs in sorted_targets:
        seq = seqs[0]
        try:
            feature_dict, processed = _prepare_batch(state, tag, seq)
        except Exception as e:
            raise RuntimeError(f"Startup warmup failed for tag={tag}: {e}") from e
        _run_forward(state.model, processed, tag)
        del feature_dict, processed
    logger.info("Startup warmup finished.")


try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
except ImportError as e:
    raise SystemExit(
        "serve_openfold requires: pip install fastapi uvicorn\n"
        f"Import error: {e}"
    ) from e


@dataclass
class AppState:
    args: Any
    config: Any
    model: torch.nn.Module
    output_directory: str
    data_processor: Any
    feature_processor: Any
    is_multimer: bool


def _run_forward(model: torch.nn.Module, batch: dict, tag: str) -> tuple[Any, float]:
    """Same logic as script_utils.run_model, without writing timings.json."""
    with torch.no_grad():
        template_enabled = model.config.template.enabled
        model.config.template.enabled = template_enabled and any(
            "template_" in k for k in batch
        )
        logger.info("Running inference for %s...", tag)
        t0 = time.perf_counter()
        out = model(batch)
        inference_seconds = time.perf_counter() - t0
        logger.info("Inference time: %.4fs", inference_seconds)
        model.config.template.enabled = template_enabled
    return out, inference_seconds


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenFold inference HTTP server.")
    p.add_argument(
        "template_mmcif_dir",
        type=str,
        help="Template mmCIF directory (same as run_pretrained_openfold.py)",
    )
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--use_precomputed_alignments",
        type=str,
        default=None,
        help="Base directory for alignments/embeddings (per-target subdirs named by tag)",
    )
    p.add_argument("--output_dir", type=str, default="serve_output")
    p.add_argument("--model_device", type=str, default="cuda:0")
    p.add_argument("--config_preset", type=str, default="model_1")
    p.add_argument("--jax_param_path", type=str, default=None)
    p.add_argument("--openfold_checkpoint_path", type=str, default=None)
    p.add_argument("--use_single_seq_mode", action="store_true", default=False)
    p.add_argument("--long_sequence_inference", action="store_true", default=False)
    p.add_argument("--use_deepspeed_evoformer_attention", action="store_true", default=False)
    p.add_argument("--experiment_config_json", default="")
    p.add_argument("--multimer_ri_gap", type=int, default=200)
    p.add_argument("--subtract_plddt", action="store_true", default=False)
    p.add_argument("--cif_output", action="store_true", default=False)
    p.add_argument("--trace_model", action="store_true", default=False)
    p.add_argument(
        "--torch_compile",
        action="store_true",
        default=False,
        help="Wrap the model with torch.compile after load (Inductor cache under TORCHINDUCTOR_CACHE_DIR).",
    )
    p.add_argument(
        "--torch_inductor_cache_dir",
        type=str,
        default=None,
        help="Inductor on-disk cache (default: $XDG_CACHE_HOME/openfold/torch_inductor).",
    )
    p.add_argument(
        "--torch_inductor_cache_force_rebuild",
        action="store_true",
        default=False,
        help="Clear the Inductor cache directory before starting.",
    )
    p.add_argument(
        "--torch_compile_strategy",
        choices=("full", "submodules"),
        default="full",
        help="full: torch.compile(model); submodules: compile evoformer, structure_module, aux_heads.",
    )
    p.add_argument(
        "--torch_compile_evoformer_mode",
        type=str,
        default=None,
        help="torch.compile mode for strategy=full (whole model), or for evoformer when strategy=submodules.",
    )
    p.add_argument(
        "--torch_compile_structure_module_mode",
        type=str,
        default=None,
        help="torch.compile mode for structure_module when strategy=submodules.",
    )
    p.add_argument(
        "--torch_compile_submodule_mode",
        type=str,
        default=None,
        help="torch.compile mode for aux_heads when strategy=submodules.",
    )
    p.add_argument(
        "--torch_inductor_no_static_launcher",
        action="store_true",
        default=False,
        help="Set torch._inductor.config.use_static_triton_launcher=False (e.g. ROCm CUDA driver 301).",
    )
    p.add_argument(
        "--pad_seqlen_multiple",
        type=int,
        default=0,
        help="If >0, pad sequence length to the next multiple of this value (fewer distinct "
        "shapes under torch.compile). 0 disables.",
    )
    p.add_argument(
        "--pad_seqlen_trace_buckets",
        action="store_true",
        default=False,
        help="Also pad up to the next multiple of 50 residues (same bucketing as "
        "run_pretrained_openfold.py --trace_model). Combined with --pad_seqlen_multiple via max().",
    )
    p.add_argument(
        "--startup_warmup_fasta_dir",
        type=str,
        default=None,
        help="If set, run one forward per FASTA in this directory (monomer .fa/.fasta, same "
        "rules as client_openfold_server.py) after load and before accepting HTTP requests.",
    )
    p.add_argument(
        "--startup_warmup_max_sequences",
        type=int,
        default=0,
        help="If >0, only warm up the first N targets after shortest-first sort. 0 = all.",
    )
    add_data_args(p)
    return p


def _load_state(args: Any) -> AppState:
    if args.config_preset.startswith("seq"):
        args.use_single_seq_mode = True

    if args.use_precomputed_alignments is None:
        raise ValueError(
            "serve_openfold expects --use_precomputed_alignments "
            "(base dir with per-tag embedding/alignment subdirectories)."
        )

    if args.trace_model:
        raise ValueError(
            "Use run_pretrained_openfold.py for --trace_model; not supported in the server."
        )

    config = model_config(
        args.config_preset,
        long_sequence_inference=args.long_sequence_inference,
        use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
    )
    if args.experiment_config_json:
        import json

        with open(args.experiment_config_json, "r") as f:
            config.update_from_flattened_dict(json.load(f))

    is_multimer = "multimer" in args.config_preset
    if is_multimer and args.openfold_checkpoint_path:
        raise ValueError("Multimer + openfold_checkpoint_path is not supported here.")

    if is_multimer:
        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=config.data.predict.max_templates,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path,
        )
    else:
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=config.data.predict.max_templates,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path,
        )

    data_processor = data_pipeline.DataPipeline(template_featurizer=template_featurizer)
    if is_multimer:
        data_processor = data_pipeline.DataPipelineMultimer(
            monomer_data_pipeline=data_processor,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    random_seed = random.randrange(2**32)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    if args.jax_param_path is None and args.openfold_checkpoint_path is None:
        args.jax_param_path = os.path.join(
            "openfold", "resources", "params", "params_" + args.config_preset + ".npz"
        )

    gen = load_models_from_command_line(
        config,
        args.model_device,
        args.openfold_checkpoint_path,
        args.jax_param_path,
        args.output_dir,
        torch_compile=args.torch_compile,
        torch_inductor_cache_dir=args.torch_inductor_cache_dir,
        force_inductor_cache_rebuild=args.torch_inductor_cache_force_rebuild,
        torch_compile_strategy=args.torch_compile_strategy,
        torch_compile_evoformer_mode=args.torch_compile_evoformer_mode,
        torch_compile_structure_module_mode=args.torch_compile_structure_module_mode,
        torch_compile_submodule_mode=args.torch_compile_submodule_mode,
        torch_inductor_no_static_launcher=args.torch_inductor_no_static_launcher,
    )
    model, output_directory = next(iter(gen))

    if args.pad_seqlen_multiple > 0 and not args.torch_compile:
        logger.info(
            "--pad_seqlen_multiple is set without --torch_compile: bucketing mainly "
            "helps Inductor/torch.compile reuse; eager mode still pays extra compute for padding."
        )

    logger.info("Model loaded; prediction dir %s", output_directory)
    return AppState(
        args=args,
        config=config,
        model=model,
        output_directory=output_directory,
        data_processor=data_processor,
        feature_processor=feature_processor,
        is_multimer=is_multimer,
    )


class PredictBody(BaseModel):
    tag: str = Field(..., description="Target id; data expected under alignment_dir/tag")
    sequence: str = Field(..., description="Amino acid sequence (single chain)")
    inference_warmup: bool = Field(
        default=False,
        description="If true, run one untimed forward before the timed run.",
    )


def create_app(state: AppState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.openfold = state
        yield

    app = FastAPI(title="OpenFold inference", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "device": state.args.model_device,
            "torch_compile": bool(state.args.torch_compile),
            "torch_compile_strategy": state.args.torch_compile_strategy,
            "pad_seqlen_multiple": state.args.pad_seqlen_multiple,
            "pad_seqlen_trace_buckets": bool(state.args.pad_seqlen_trace_buckets),
        }

    @app.post("/predict")
    def predict(body: PredictBody):
        st: AppState = app.state.openfold
        args = st.args
        tag = body.tag
        seq = body.sequence

        if st.is_multimer:
            raise HTTPException(status_code=501, detail="Multimer /predict not implemented")

        try:
            feature_dict, processed = _prepare_batch(st, tag, seq)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"feature generation failed: {e}") from e

        if body.inference_warmup:
            _run_forward(st.model, processed, f"{tag}_warmup")

        out, inference_seconds = _run_forward(st.model, processed, tag)

        processed_cpu = tensor_tree_map(
            lambda x: np.array(x[..., -1].cpu()),
            processed,
        )
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        unrelaxed = prep_output(
            out,
            processed_cpu,
            feature_dict,
            st.feature_processor,
            args.config_preset,
            args.multimer_ri_gap,
            args.subtract_plddt,
        )
        if args.cif_output:
            text = protein.to_modelcif(unrelaxed)
            fmt = "cif"
        else:
            text = protein.to_pdb(unrelaxed)
            fmt = "pdb"

        return JSONResponse(
            {
                "tag": tag,
                "inference_seconds": inference_seconds,
                "structure_format": fmt,
                "structure": text,
            }
        )

    return app


def main() -> None:
    args = _build_parser().parse_args()
    state = _load_state(args)
    if args.startup_warmup_fasta_dir:
        _startup_warmup(state)
    app = create_app(state)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
