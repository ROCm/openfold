#!/usr/bin/env python3
"""
OpenFold v2.1.0 torch.compile benchmark harness.

Systematically applies torch.compile to individual modules and measures
inference time across all 50 samples. Supports multiple optimization
strategies controlled by CLI flags.

Usage:
    python benchmark_compile.py --strategy baseline
    python benchmark_compile.py --strategy compile_evoformer
    python benchmark_compile.py --strategy compile_all
    python benchmark_compile.py --strategy max_autotune
    python benchmark_compile.py --strategy reduce_overhead
    python benchmark_compile.py --strategy aggressive
"""

import argparse
import gc
import json
import logging
import os
import sys
import time

import numpy as np
import torch

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".torch_compile_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", CACHE_DIR)

torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("benchmark")

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.np import protein
from openfold.utils.script_utils import (
    load_models_from_command_line,
    parse_fasta,
)
from openfold.utils.tensor_utils import tensor_tree_map

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
FASTA_DIR = os.path.join(ROOT_DIR, "fasta_dir")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
EMBEDDINGS_DIR = os.path.join(ROOT_DIR, "embeddings_dir")
CHECKPOINT = os.path.join(
    ROOT_DIR,
    "openfold/params/openfold_soloseq_params/seq_model_esm1b_ptm.pt",
)
CONFIG_PRESET = "seq_model_esm1b_ptm"
DEVICE = "cuda:0"

STRATEGIES = [
    "baseline",
    "hip_graph",
    "compile_evoformer",
    "compile_all_default",
    "compile_all_autotune",
    "max_autotune",
    "reduce_overhead",
    "aggressive",
    "aggressive_fused",
    "aggressive_te",
    "aggressive_granular",
    "aggressive_te_graph",
    "aggressive_te_triton",
    "dynamic_default",
    "dynamic_autotune",
    "dynamic_aggressive",
]


def apply_compile_strategy(model, strategy):
    """Apply torch.compile to model submodules based on strategy."""

    if strategy == "baseline":
        log.info("Strategy: baseline (no torch.compile)")
        return model

    if strategy == "hip_graph":
        log.info("Strategy: HIP Graph (manual CUDA/HIP graph capture around evoformer)")
        from openfold.utils.hip_graph import GraphedEvoformer
        wrapper = GraphedEvoformer(model.evoformer, warmup_iters=3)
        object.__setattr__(model, 'evoformer', wrapper)
        return model

    if strategy == "compile_evoformer":
        log.info("Strategy: compile evoformer only (mode=default)")
        model.evoformer = torch.compile(model.evoformer, mode="default")
        return model

    if strategy == "compile_all_default":
        log.info("Strategy: compile all major modules (mode=default)")
        model.evoformer = torch.compile(model.evoformer, mode="default")
        model.structure_module = torch.compile(
            model.structure_module, mode="default"
        )
        model.aux_heads = torch.compile(model.aux_heads, mode="default")
        model.input_embedder = torch.compile(
            model.input_embedder, mode="default"
        )
        if hasattr(model, "extra_msa_stack"):
            model.extra_msa_stack = torch.compile(
                model.extra_msa_stack, mode="default"
            )
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(
                model.recycling_embedder, mode="default"
            )
        return model

    if strategy == "compile_all_autotune":
        log.info("Strategy: compile all modules (mode=max-autotune-no-cudagraphs)")
        model.evoformer = torch.compile(
            model.evoformer, mode="max-autotune"
        )
        model.structure_module = torch.compile(
            model.structure_module, mode="max-autotune-no-cudagraphs"
        )
        model.aux_heads = torch.compile(
            model.aux_heads, mode="max-autotune-no-cudagraphs"
        )
        model.input_embedder = torch.compile(
            model.input_embedder, mode="max-autotune-no-cudagraphs"
        )
        if hasattr(model, "extra_msa_stack"):
            model.extra_msa_stack = torch.compile(
                model.extra_msa_stack, mode="max-autotune-no-cudagraphs"
            )
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(
                model.recycling_embedder, mode="max-autotune-no-cudagraphs"
            )
        return model

    if strategy == "max_autotune":
        log.info("Strategy: max-autotune on evoformer, max-autotune-no-cudagraphs on rest")
        model.evoformer = torch.compile(
            model.evoformer, mode="max-autotune"
        )
        model.structure_module = torch.compile(
            model.structure_module, mode="max-autotune-no-cudagraphs"
        )
        model.aux_heads = torch.compile(
            model.aux_heads, mode="max-autotune-no-cudagraphs"
        )
        model.input_embedder = torch.compile(
            model.input_embedder, mode="max-autotune-no-cudagraphs"
        )
        if hasattr(model, "extra_msa_stack"):
            model.extra_msa_stack = torch.compile(
                model.extra_msa_stack, mode="max-autotune-no-cudagraphs"
            )
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(
                model.recycling_embedder, mode="max-autotune-no-cudagraphs"
            )
        return model

    if strategy == "reduce_overhead":
        log.info("Strategy: reduce-overhead on all modules")
        model.evoformer = torch.compile(
            model.evoformer, mode="reduce-overhead"
        )
        model.structure_module = torch.compile(
            model.structure_module, mode="reduce-overhead"
        )
        model.aux_heads = torch.compile(
            model.aux_heads, mode="reduce-overhead"
        )
        model.input_embedder = torch.compile(
            model.input_embedder, mode="reduce-overhead"
        )
        if hasattr(model, "extra_msa_stack"):
            model.extra_msa_stack = torch.compile(
                model.extra_msa_stack, mode="reduce-overhead"
            )
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(
                model.recycling_embedder, mode="reduce-overhead"
            )
        return model

    if strategy == "aggressive_fused":
        log.info("Strategy: aggressive + aggressive_fusion + permute_fusion + combo_kernels")
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 256
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
        torch._inductor.config.aggressive_fusion = True
        torch._inductor.config.permute_fusion = True
        torch._inductor.config.combo_kernels = True
        torch._inductor.config.max_autotune_allow_flexible_layouts = True

        model.evoformer = torch.compile(
            model.evoformer, mode="max-autotune"
        )
        model.structure_module = torch.compile(
            model.structure_module, mode="max-autotune-no-cudagraphs"
        )
        model.aux_heads = torch.compile(
            model.aux_heads, mode="max-autotune-no-cudagraphs"
        )
        model.input_embedder = torch.compile(
            model.input_embedder, mode="max-autotune-no-cudagraphs"
        )
        if hasattr(model, "extra_msa_stack"):
            model.extra_msa_stack = torch.compile(
                model.extra_msa_stack, mode="max-autotune-no-cudagraphs"
            )
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(
                model.recycling_embedder, mode="max-autotune-no-cudagraphs"
            )
        return model

    if strategy == "aggressive":
        log.info("Strategy: aggressive — max-autotune everywhere + inductor knobs (no template_embedder)")
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 256
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True

        model.evoformer = torch.compile(
            model.evoformer, mode="max-autotune"
        )
        model.structure_module = torch.compile(
            model.structure_module, mode="max-autotune-no-cudagraphs"
        )
        model.aux_heads = torch.compile(
            model.aux_heads, mode="max-autotune-no-cudagraphs"
        )
        model.input_embedder = torch.compile(
            model.input_embedder, mode="max-autotune-no-cudagraphs"
        )
        if hasattr(model, "extra_msa_stack"):
            model.extra_msa_stack = torch.compile(
                model.extra_msa_stack, mode="max-autotune-no-cudagraphs"
            )
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(
                model.recycling_embedder, mode="max-autotune-no-cudagraphs"
            )
        return model

    if strategy == "aggressive_te":
        log.info("Strategy: aggressive + template_embedder submodules (pair_embedder, pair_stack, single_embedder)")
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 256
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True

        model.evoformer = torch.compile(
            model.evoformer, mode="max-autotune"
        )
        model.structure_module = torch.compile(
            model.structure_module, mode="max-autotune-no-cudagraphs"
        )
        model.aux_heads = torch.compile(
            model.aux_heads, mode="max-autotune-no-cudagraphs"
        )
        model.input_embedder = torch.compile(
            model.input_embedder, mode="max-autotune-no-cudagraphs"
        )
        if hasattr(model, "extra_msa_stack"):
            model.extra_msa_stack = torch.compile(
                model.extra_msa_stack, mode="max-autotune-no-cudagraphs"
            )
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(
                model.recycling_embedder, mode="max-autotune-no-cudagraphs"
            )
        if hasattr(model, "template_embedder"):
            te = model.template_embedder
            te.template_pair_embedder = torch.compile(
                te.template_pair_embedder, mode="max-autotune-no-cudagraphs"
            )
            if hasattr(te, "template_single_embedder"):
                te.template_single_embedder = torch.compile(
                    te.template_single_embedder, mode="max-autotune-no-cudagraphs"
                )
            # Granular: compile individual layers inside each TemplatePairStackBlock
            for blk_idx, blk in enumerate(te.template_pair_stack.blocks):
                blk.tri_att_start = torch.compile(
                    blk.tri_att_start, mode="max-autotune-no-cudagraphs"
                )
                blk.tri_att_end = torch.compile(
                    blk.tri_att_end, mode="max-autotune-no-cudagraphs"
                )
                blk.tri_mul_out = torch.compile(
                    blk.tri_mul_out, mode="max-autotune-no-cudagraphs"
                )
                blk.tri_mul_in = torch.compile(
                    blk.tri_mul_in, mode="max-autotune-no-cudagraphs"
                )
                blk.pair_transition = torch.compile(
                    blk.pair_transition, mode="max-autotune-no-cudagraphs"
                )
                log.info(f"  pair_stack block {blk_idx}: compiled 5 submodules (mode=max-autotune-no-cudagraphs)")
            te.template_pair_stack.layer_norm = torch.compile(
                te.template_pair_stack.layer_norm, mode="max-autotune-no-cudagraphs"
            )
            # pointwise_att: use mode=default to avoid Triton binary compat issue
            te.template_pointwise_att = torch.compile(
                te.template_pointwise_att, mode="default"
            )
            log.info("  template_pointwise_att: compiled (mode=default)")
        return model

    if strategy == "aggressive_te_graph":
        log.info("Strategy: aggressive_te (no-cudagraphs) + manual HIP Graph on evoformer")
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 256
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True

        compiled_evo = torch.compile(
            model.evoformer, mode="max-autotune-no-cudagraphs"
        )
        model.evoformer = compiled_evo

        model.structure_module = torch.compile(
            model.structure_module, mode="max-autotune-no-cudagraphs"
        )
        model.aux_heads = torch.compile(
            model.aux_heads, mode="max-autotune-no-cudagraphs"
        )
        model.input_embedder = torch.compile(
            model.input_embedder, mode="max-autotune-no-cudagraphs"
        )
        if hasattr(model, "extra_msa_stack"):
            model.extra_msa_stack = torch.compile(
                model.extra_msa_stack, mode="max-autotune-no-cudagraphs"
            )
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(
                model.recycling_embedder, mode="max-autotune-no-cudagraphs"
            )
        if hasattr(model, "template_embedder"):
            te = model.template_embedder
            te.template_pair_embedder = torch.compile(
                te.template_pair_embedder, mode="max-autotune-no-cudagraphs"
            )
            if hasattr(te, "template_single_embedder"):
                te.template_single_embedder = torch.compile(
                    te.template_single_embedder, mode="max-autotune-no-cudagraphs"
                )
            for blk_idx, blk in enumerate(te.template_pair_stack.blocks):
                blk.tri_att_start = torch.compile(
                    blk.tri_att_start, mode="max-autotune-no-cudagraphs"
                )
                blk.tri_att_end = torch.compile(
                    blk.tri_att_end, mode="max-autotune-no-cudagraphs"
                )
                blk.tri_mul_out = torch.compile(
                    blk.tri_mul_out, mode="max-autotune-no-cudagraphs"
                )
                blk.tri_mul_in = torch.compile(
                    blk.tri_mul_in, mode="max-autotune-no-cudagraphs"
                )
                blk.pair_transition = torch.compile(
                    blk.pair_transition, mode="max-autotune-no-cudagraphs"
                )
            te.template_pair_stack.layer_norm = torch.compile(
                te.template_pair_stack.layer_norm, mode="max-autotune-no-cudagraphs"
            )
            te.template_pointwise_att = torch.compile(
                te.template_pointwise_att, mode="default"
            )

        from openfold.utils.hip_graph import GraphedEvoformer
        wrapper = GraphedEvoformer(compiled_evo, warmup_iters=3)
        object.__setattr__(model, 'evoformer', wrapper)
        return model

    if strategy == "aggressive_te_triton":
        log.info("Strategy: aggressive_te + force Triton-only for all GEMMs (no rocBLAS)")
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 256
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
        torch._inductor.config.max_autotune_gemm_backends = "TRITON"
        log.info("  max_autotune_gemm_backends = TRITON (rocBLAS disabled)")

        model.evoformer = torch.compile(
            model.evoformer, mode="max-autotune"
        )
        model.structure_module = torch.compile(
            model.structure_module, mode="max-autotune-no-cudagraphs"
        )
        model.aux_heads = torch.compile(
            model.aux_heads, mode="max-autotune-no-cudagraphs"
        )
        model.input_embedder = torch.compile(
            model.input_embedder, mode="max-autotune-no-cudagraphs"
        )
        if hasattr(model, "extra_msa_stack"):
            model.extra_msa_stack = torch.compile(
                model.extra_msa_stack, mode="max-autotune-no-cudagraphs"
            )
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(
                model.recycling_embedder, mode="max-autotune-no-cudagraphs"
            )
        if hasattr(model, "template_embedder"):
            te = model.template_embedder
            te.template_pair_embedder = torch.compile(
                te.template_pair_embedder, mode="max-autotune-no-cudagraphs"
            )
            if hasattr(te, "template_single_embedder"):
                te.template_single_embedder = torch.compile(
                    te.template_single_embedder, mode="max-autotune-no-cudagraphs"
                )
            mode = "max-autotune-no-cudagraphs"
            for blk_idx, blk in enumerate(te.template_pair_stack.blocks):
                blk.tri_att_start = torch.compile(blk.tri_att_start, mode=mode)
                blk.tri_att_end = torch.compile(blk.tri_att_end, mode=mode)
                blk.tri_mul_out = torch.compile(blk.tri_mul_out, mode=mode)
                blk.tri_mul_in = torch.compile(blk.tri_mul_in, mode=mode)
                blk.pair_transition = torch.compile(blk.pair_transition, mode=mode)
            te.template_pair_stack.layer_norm = torch.compile(te.template_pair_stack.layer_norm, mode=mode)
            te.template_pointwise_att = torch.compile(te.template_pointwise_att, mode="default")
        return model

    if strategy == "aggressive_granular":
        log.info("Strategy: granular compilation of ALL triangle ops across entire model")
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 256
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True

        mode = "max-autotune-no-cudagraphs"

        # ── Evoformer: compile each block's PairStack submodules individually ──
        # Instead of torch.compile(model.evoformer) which hits graph breaks
        # from checkpoint_blocks/partial(), compile triangle ops directly.
        evo = model.evoformer
        for blk_idx, blk in enumerate(evo.blocks):
            ps = blk.pair_stack
            ps.tri_mul_out = torch.compile(ps.tri_mul_out, mode=mode)
            ps.tri_mul_in = torch.compile(ps.tri_mul_in, mode=mode)
            ps.tri_att_start = torch.compile(ps.tri_att_start, mode=mode)
            ps.tri_att_end = torch.compile(ps.tri_att_end, mode=mode)
            ps.pair_transition = torch.compile(ps.pair_transition, mode=mode)
            blk.msa_att_row = torch.compile(blk.msa_att_row, mode=mode)
            blk.msa_transition = torch.compile(blk.msa_transition, mode=mode)
            blk.outer_product_mean = torch.compile(blk.outer_product_mean, mode=mode)
            if hasattr(blk, "msa_att_col"):
                blk.msa_att_col = torch.compile(blk.msa_att_col, mode=mode)
        evo.linear = torch.compile(evo.linear, mode=mode)
        log.info(f"  evoformer: compiled {len(evo.blocks)} blocks × 8-9 submodules each")

        # ── Extra MSA stack: same granular approach ──
        if hasattr(model, "extra_msa_stack"):
            ems = model.extra_msa_stack
            for blk_idx, blk in enumerate(ems.blocks):
                ps = blk.pair_stack
                ps.tri_mul_out = torch.compile(ps.tri_mul_out, mode=mode)
                ps.tri_mul_in = torch.compile(ps.tri_mul_in, mode=mode)
                ps.tri_att_start = torch.compile(ps.tri_att_start, mode=mode)
                ps.tri_att_end = torch.compile(ps.tri_att_end, mode=mode)
                ps.pair_transition = torch.compile(ps.pair_transition, mode=mode)
                blk.msa_att_row = torch.compile(blk.msa_att_row, mode=mode)
                blk.msa_transition = torch.compile(blk.msa_transition, mode=mode)
                blk.outer_product_mean = torch.compile(blk.outer_product_mean, mode=mode)
                if hasattr(blk, "msa_att_col"):
                    blk.msa_att_col = torch.compile(blk.msa_att_col, mode=mode)
            log.info(f"  extra_msa_stack: compiled {len(ems.blocks)} blocks × 8-9 submodules each")

        # ── Structure module, aux_heads, input_embedder ──
        model.structure_module = torch.compile(model.structure_module, mode=mode)
        model.aux_heads = torch.compile(model.aux_heads, mode=mode)
        model.input_embedder = torch.compile(model.input_embedder, mode=mode)
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(model.recycling_embedder, mode=mode)
        log.info("  structure_module, aux_heads, input_embedder, recycling_embedder: compiled")

        # ── Template embedder: same granular approach as aggressive_te ──
        if hasattr(model, "template_embedder"):
            te = model.template_embedder
            te.template_pair_embedder = torch.compile(te.template_pair_embedder, mode=mode)
            if hasattr(te, "template_single_embedder"):
                te.template_single_embedder = torch.compile(te.template_single_embedder, mode=mode)
            for blk_idx, blk in enumerate(te.template_pair_stack.blocks):
                blk.tri_att_start = torch.compile(blk.tri_att_start, mode=mode)
                blk.tri_att_end = torch.compile(blk.tri_att_end, mode=mode)
                blk.tri_mul_out = torch.compile(blk.tri_mul_out, mode=mode)
                blk.tri_mul_in = torch.compile(blk.tri_mul_in, mode=mode)
                blk.pair_transition = torch.compile(blk.pair_transition, mode=mode)
            te.template_pair_stack.layer_norm = torch.compile(te.template_pair_stack.layer_norm, mode=mode)
            te.template_pointwise_att = torch.compile(te.template_pointwise_att, mode="default")
            log.info(f"  template_embedder: compiled {len(te.template_pair_stack.blocks)} blocks granularly + pointwise_att")

        return model

    if strategy == "dynamic_default":
        log.info("Strategy: dynamic=True, mode=default on all modules")
        torch._dynamo.config.suppress_errors = True
        model.evoformer = torch.compile(
            model.evoformer, mode="default", dynamic=True
        )
        model.structure_module = torch.compile(
            model.structure_module, mode="default", dynamic=True
        )
        model.aux_heads = torch.compile(
            model.aux_heads, mode="default", dynamic=True
        )
        model.input_embedder = torch.compile(
            model.input_embedder, mode="default", dynamic=True
        )
        if hasattr(model, "extra_msa_stack"):
            model.extra_msa_stack = torch.compile(
                model.extra_msa_stack, mode="default", dynamic=True
            )
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(
                model.recycling_embedder, mode="default", dynamic=True
            )
        return model

    if strategy == "dynamic_autotune":
        log.info("Strategy: dynamic=True, max-autotune on evoformer, max-autotune-no-cudagraphs rest")
        torch._dynamo.config.suppress_errors = True
        model.evoformer = torch.compile(
            model.evoformer, mode="max-autotune-no-cudagraphs", dynamic=True
        )
        model.structure_module = torch.compile(
            model.structure_module, mode="max-autotune-no-cudagraphs", dynamic=True
        )
        model.aux_heads = torch.compile(
            model.aux_heads, mode="max-autotune-no-cudagraphs", dynamic=True
        )
        model.input_embedder = torch.compile(
            model.input_embedder, mode="max-autotune-no-cudagraphs", dynamic=True
        )
        if hasattr(model, "extra_msa_stack"):
            model.extra_msa_stack = torch.compile(
                model.extra_msa_stack, mode="max-autotune-no-cudagraphs", dynamic=True
            )
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(
                model.recycling_embedder, mode="max-autotune-no-cudagraphs", dynamic=True
            )
        return model

    if strategy == "dynamic_aggressive":
        log.info("Strategy: dynamic aggressive — all dynamo/inductor knobs + dynamic=True")
        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = 256
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True

        model.evoformer = torch.compile(
            model.evoformer, mode="max-autotune-no-cudagraphs", dynamic=True
        )
        model.structure_module = torch.compile(
            model.structure_module, mode="max-autotune-no-cudagraphs", dynamic=True
        )
        model.aux_heads = torch.compile(
            model.aux_heads, mode="max-autotune-no-cudagraphs", dynamic=True
        )
        model.input_embedder = torch.compile(
            model.input_embedder, mode="max-autotune-no-cudagraphs", dynamic=True
        )
        if hasattr(model, "extra_msa_stack"):
            model.extra_msa_stack = torch.compile(
                model.extra_msa_stack, mode="max-autotune-no-cudagraphs", dynamic=True
            )
        if hasattr(model, "recycling_embedder"):
            model.recycling_embedder = torch.compile(
                model.recycling_embedder, mode="max-autotune-no-cudagraphs", dynamic=True
            )
        if hasattr(model, "template_embedder"):
            model.template_embedder = torch.compile(
                model.template_embedder, mode="max-autotune-no-cudagraphs", dynamic=True
            )
        return model

    raise ValueError(f"Unknown strategy: {strategy}")


def load_fasta_files(fasta_dir):
    """Return sorted list of (tag, seq) from FASTA directory."""
    entries = []
    for fn in sorted(os.listdir(fasta_dir)):
        if not fn.endswith(".fasta"):
            continue
        tag = fn[:-6]
        with open(os.path.join(fasta_dir, fn)) as f:
            lines = f.readlines()
        seq = "".join(l.strip() for l in lines if not l.startswith(">"))
        entries.append((tag, seq))
    return entries


def main():
    parser = argparse.ArgumentParser(description="OpenFold torch.compile benchmark")
    parser.add_argument(
        "--strategy", default="baseline", choices=STRATEGIES,
        help="Compile strategy to apply",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for timings (default: compile_runs/<strategy>)",
    )
    parser.add_argument(
        "--warmup", type=int, default=2,
        help="Number of warmup samples before timing (default: 2)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=None,
        help="Limit number of samples to run (default: all 50)",
    )
    parser.add_argument(
        "--prewarm", action="store_true", default=False,
        help="Treat first occurrence of each unique seq_len as warmup (excludes recompilation from average)",
    )
    parser.add_argument(
        "--fasta-dir", default=FASTA_DIR,
        help="Path to FASTA directory",
    )
    parser.add_argument(
        "--embeddings-dir", default=EMBEDDINGS_DIR,
        help="Path to embeddings directory (default: embeddings_dir)",
    )
    parser.add_argument(
        "--passes", type=int, default=1,
        help="Number of full passes over the dataset. Only the LAST pass is measured. "
             "Pass 1 compiles all shapes. (default: 1)",
    )
    parser.add_argument(
        "--torch-profile", action="store_true", default=False,
        help="Enable torch.profiler on the measured pass and save chrome trace + stacks",
    )
    parser.add_argument(
        "--bf16", action="store_true", default=False,
        help="Run inference under torch.autocast with bfloat16",
    )
    parser.add_argument(
        "--save-predictions", action="store_true", default=False,
        help="Save per-sample predicted coordinates and pLDDT for validation",
    )
    parser.add_argument(
        "--no-chunk", action="store_true", default=False,
        help="Disable chunking: set globals.chunk_size=None and tune_chunk_size=False. "
             "Runs full tensors in one shot (requires more VRAM, faster on large-HBM GPUs)",
    )
    parser.add_argument(
        "--static-shapes", action="store_true", default=False,
        help="Set torch._dynamo.config.dynamic_shapes=False for fully static compilation",
    )
    parser.add_argument(
        "--triton-triangle", action="store_true", default=False,
        help="Enable Triton fused kernels for TriangleMultiplicativeUpdate "
             "(sets OPENFOLD_TRITON_TRIANGLE=1 before model load)",
    )
    args = parser.parse_args()

    if args.triton_triangle:
        os.environ["OPENFOLD_TRITON_TRIANGLE"] = "1"
        log.info("Triton triangle kernels ENABLED (OPENFOLD_TRITON_TRIANGLE=1)")

    if args.output_dir is None:
        args.output_dir = os.path.join("compile_runs", args.strategy)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config
    evo_backend = os.environ.get("OPENFOLD_EVO_BACKEND", "").lower()
    use_evo = evo_backend in ("deepspeed", "auto")
    config = model_config(CONFIG_PRESET, use_deepspeed_evoformer_attention=use_evo)
    if use_evo:
        log.info(f"Evoformer attention enabled (OPENFOLD_EVO_BACKEND={evo_backend})")

    if args.no_chunk:
        config.globals.chunk_size = None
        config.model.evoformer_stack.tune_chunk_size = False
        config.model.extra_msa.extra_msa_stack.tune_chunk_size = False
        config.model.template.template_pair_stack.tune_chunk_size = False
        log.info("Chunking DISABLED: chunk_size=None, tune_chunk_size=False")

    if args.static_shapes:
        torch._dynamo.config.dynamic_shapes = False
        log.info("Static shapes ENABLED: dynamic_shapes=False")

    # Load model
    log.info("Loading model...")
    model_gen = load_models_from_command_line(
        config, DEVICE, CHECKPOINT, None, args.output_dir
    )
    model, output_directory = next(model_gen)
    log.info(f"Model loaded on {DEVICE}")

    # Apply compile strategy
    model = apply_compile_strategy(model, args.strategy)

    # Set up data pipeline (same as run_pretrained_openfold.py)
    template_featurizer = templates.HhsearchHitFeaturizer(
        mmcif_dir=TEMPLATES_DIR,
        max_template_date="2026-03-04",
        max_hits=config.data.predict.max_templates,
        kalign_binary_path="kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None,
    )
    data_proc = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )
    feature_processor = feature_pipeline.FeaturePipeline(config.data)

    # Load FASTA entries
    fasta_entries = load_fasta_files(args.fasta_dir)
    if args.num_samples:
        fasta_entries = fasta_entries[:args.num_samples]
    log.info(f"Loaded {len(fasta_entries)} samples")

    # Sort by sequence length (matches original run_pretrained_openfold.py behavior)
    fasta_entries.sort(key=lambda x: len(x[1]))

    timings = {}
    all_times = []

    for pass_num in range(1, args.passes + 1):
        is_final_pass = (pass_num == args.passes)
        if args.passes > 1:
            log.info(f"{'='*60}")
            log.info(f"Pass {pass_num}/{args.passes} {'(MEASURED)' if is_final_pass else '(warmup/compile)'}")
            log.info(f"{'='*60}")

        if is_final_pass:
            timings = {}
            all_times = []
            main._compiled_shapes = set()

        torch_profiler = None
        if is_final_pass and args.torch_profile:
            from torch.profiler import profile, ProfilerActivity, schedule
            prof_dir = os.path.join(args.output_dir, "torch_profile")
            os.makedirs(prof_dir, exist_ok=True)
            torch_profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
                record_shapes=True,
                profile_memory=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(prof_dir),
            )
            torch_profiler.__enter__()
            log.info(f"Torch profiler enabled → {prof_dir}")

        for idx, (tag, seq) in enumerate(fasta_entries):
            tmp_fasta = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
            with open(tmp_fasta, "w") as f:
                f.write(f">{tag}\n{seq}")
            alignment_dir = os.path.join(args.embeddings_dir, tag)
            feature_dict = data_proc.process_fasta(
                fasta_path=tmp_fasta,
                alignment_dir=alignment_dir,
                seqemb_mode=True,
            )
            os.remove(tmp_fasta)

            processed = feature_processor.process_features(
                feature_dict, mode="predict", is_multimer=False
            )
            processed = {
                k: torch.as_tensor(v, device=DEVICE)
                for k, v in processed.items()
            }

            if not hasattr(main, '_compiled_shapes'):
                main._compiled_shapes = set()

            seq_len = len(seq)
            is_new_shape = seq_len not in main._compiled_shapes
            is_warmup = idx < args.warmup or (
                args.prewarm and is_new_shape
            )
            main._compiled_shapes.add(seq_len)

            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

            if not is_final_pass:
                marker_prefix = "compile"
            elif is_warmup:
                marker_prefix = "warmup"
            else:
                marker_prefix = "inference"
            torch.cuda.nvtx.range_push(f"{marker_prefix}_{tag}")
            t0 = time.perf_counter()
            try:
                if args.bf16:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(processed)
                else:
                    with torch.no_grad():
                        out = model(processed)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                torch.cuda.nvtx.range_pop()
                oom = False
            except torch.cuda.OutOfMemoryError:
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                torch.cuda.nvtx.range_pop()
                oom = True
                log.warning(
                    f"[OOM]     {idx+1:3d}/{len(fasta_entries)} {tag:12s} "
                    f"seq_len={seq_len:4d}  — skipped (out of memory)"
                )
                del processed
                gc.collect()
                torch.cuda.empty_cache()
                continue

            if not is_final_pass:
                label = "COMPILE"
            elif is_warmup and is_new_shape and args.prewarm:
                label = "PREWARM"
            elif is_warmup:
                label = "WARMUP"
            else:
                label = "TIMED"
            log.info(
                f"[{label}] {idx+1:3d}/{len(fasta_entries)} {tag:12s} "
                f"seq_len={seq_len:4d}  time={elapsed:.4f}s"
            )

            if is_final_pass:
                timings[tag] = {
                    "inference": elapsed,
                    "seq_len": seq_len,
                    "warmup": is_warmup,
                }
                if not is_warmup:
                    all_times.append(elapsed)

                if args.save_predictions and "final_atom_positions" in out:
                    pred_dir = os.path.join(args.output_dir, "predictions")
                    os.makedirs(pred_dir, exist_ok=True)
                    coords = out["final_atom_positions"].cpu().float()
                    plddt = out["plddt"].cpu().float() if "plddt" in out else None
                    pred = {"coords": coords, "seq_len": seq_len}
                    if plddt is not None:
                        pred["plddt"] = plddt
                    torch.save(pred, os.path.join(pred_dir, f"{tag}.pt"))

            del out, processed
            gc.collect()

        if torch_profiler is not None:
            torch_profiler.__exit__(None, None, None)
            stacks_path = os.path.join(args.output_dir, "torch_profile", "stacks.txt")
            torch_profiler.export_stacks(stacks_path, metric="self_cuda_time_total")
            log.info(f"Torch profiler stacks exported → {stacks_path}")
            table = torch_profiler.key_averages(group_by_stack_n=5).table(
                sort_by="self_cuda_time_total", row_limit=40
            )
            table_path = os.path.join(args.output_dir, "torch_profile", "key_averages.txt")
            with open(table_path, "w") as f:
                f.write(table)
            log.info(f"Torch profiler key averages → {table_path}")

    # Summary
    if all_times:
        avg = sum(all_times) / len(all_times)
        median = sorted(all_times)[len(all_times) // 2]
        p90 = sorted(all_times)[int(len(all_times) * 0.9)]
        log.info("=" * 60)
        log.info(f"Strategy:        {args.strategy}")
        log.info(f"Samples timed:   {len(all_times)}")
        log.info(f"Average:         {avg:.4f} s/sample")
        log.info(f"Median:          {median:.4f} s/sample")
        log.info(f"P90:             {p90:.4f} s/sample")
        log.info(f"Min:             {min(all_times):.4f} s/sample")
        log.info(f"Max:             {max(all_times):.4f} s/sample")
        log.info(f"Total:           {sum(all_times):.2f}s for {len(all_times)} samples")
        log.info("=" * 60)

        timings["_summary"] = {
            "strategy": args.strategy,
            "n_timed": len(all_times),
            "n_warmup": args.warmup,
            "avg_s_per_sample": avg,
            "median_s_per_sample": median,
            "p90_s_per_sample": p90,
            "min_s_per_sample": min(all_times),
            "max_s_per_sample": max(all_times),
            "total_s": sum(all_times),
        }

    # Save
    out_path = os.path.join(args.output_dir, "timings.json")
    with open(out_path, "w") as f:
        json.dump(timings, f, indent=2)
    log.info(f"Timings written to {out_path}")


if __name__ == "__main__":
    main()
