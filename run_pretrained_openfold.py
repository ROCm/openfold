# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import math
import numpy as np
import os
import pickle
import random
import time
import json
import datetime
import sys

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import torch
torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if (
    torch_major_version > 1 or
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.data.tools import hhsearch, hmmsearch
from openfold.np import protein
from openfold.utils.script_utils import (load_models_from_command_line, parse_fasta, run_model,
                                         prep_output, relax_protein)
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)

from scripts.precompute_embeddings import EmbeddingGenerator
from scripts.utils import add_data_args


TRACING_INTERVAL = 50


def precompute_alignments(tags, seqs, alignment_dir, args):
    for tag, seq in zip(tags, seqs):
        tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)

        if args.use_precomputed_alignments is None:
            logger.info(f"Generating alignments for {tag}...")

            os.makedirs(local_alignment_dir, exist_ok=True)

            if "multimer" in args.config_preset:
                template_searcher = hmmsearch.Hmmsearch(
                    binary_path=args.hmmsearch_binary_path,
                    hmmbuild_binary_path=args.hmmbuild_binary_path,
                    database_path=args.pdb_seqres_database_path,
                )
            else:
                template_searcher = hhsearch.HHSearch(
                    binary_path=args.hhsearch_binary_path,
                    databases=[args.pdb70_database_path],
                )

            # In seqemb mode, use AlignmentRunner only to generate templates
            if args.use_single_seq_mode:
                alignment_runner = data_pipeline.AlignmentRunner(
                    jackhmmer_binary_path=args.jackhmmer_binary_path,
                    uniref90_database_path=args.uniref90_database_path,
                    template_searcher=template_searcher,
                    no_cpus=args.cpus,
                )
                embedding_generator = EmbeddingGenerator()
                embedding_generator.run(tmp_fasta_path, alignment_dir)
            else:
                alignment_runner = data_pipeline.AlignmentRunner(
                    jackhmmer_binary_path=args.jackhmmer_binary_path,
                    hhblits_binary_path=args.hhblits_binary_path,
                    uniref90_database_path=args.uniref90_database_path,
                    mgnify_database_path=args.mgnify_database_path,
                    bfd_database_path=args.bfd_database_path,
                    uniref30_database_path=args.uniref30_database_path,
                    uniclust30_database_path=args.uniclust30_database_path,
                    uniprot_database_path=args.uniprot_database_path,
                    template_searcher=template_searcher,
                    use_small_bfd=args.bfd_database_path is None,
                    no_cpus=args.cpus
                )

            alignment_runner.run(
                tmp_fasta_path, local_alignment_dir
            )
        else:
            logger.info(
                f"Using precomputed alignments for {tag} at {alignment_dir}..."
            )

        # Remove temporary FASTA file
        os.remove(tmp_fasta_path)


def round_up_seqlen(seqlen):
    return int(math.ceil(seqlen / TRACING_INTERVAL)) * TRACING_INTERVAL


def round_up_seqlen_multiple(seqlen, multiple):
    """Round sequence length up to the next multiple of ``multiple`` (bucket padding)."""
    if multiple is None or multiple <= 0:
        return seqlen
    return int(math.ceil(seqlen / multiple)) * multiple


def write_run_metadata(args, output_dir):
    """
    Write run metadata to a JSON file alongside timings.json.
    Captures all command-line arguments and environment info for analysis.
    
    Args:
        args: Parsed command-line arguments
        output_dir: Output directory where metadata will be written
    """
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
        "arguments": {}
    }
    
    # Capture all argparse arguments
    # Convert args namespace to dict, filtering out non-serializable objects
    for key, value in vars(args).items():
        # Skip functions, modules, and other non-serializable objects
        if not callable(value) and not hasattr(value, '__module__'):
            try:
                # Test if it's JSON serializable
                json.dumps(value)
                metadata["arguments"][key] = value
            except (TypeError, ValueError):
                # If not serializable, convert to string
                metadata["arguments"][key] = str(value)
    
    # Add environment info
    metadata["environment"] = {
        "python_version": sys.version.split()[0],
        "pytorch_version": torch.__version__,
    }
    
    # Add ROCm version if available
    if hasattr(torch.version, 'hip'):
        metadata["environment"]["rocm_version"] = torch.version.hip
    
    # Write to same directory as timings.json
    metadata_file = os.path.join(output_dir, "run_metadata.json")
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Run metadata written to: {metadata_file}")
    except Exception as e:
        logger.warning(f"Failed to write run metadata: {e}")
    
    return metadata_file


def generate_feature_dict(
    tags,
    seqs,
    alignment_dir,
    data_processor,
    args,
):
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")

    if "multimer" in args.config_preset:
        with open(tmp_fasta_path, "w") as fp:
            fp.write(
                '\n'.join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)])
            )
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path, alignment_dir=alignment_dir,
        )
    elif len(seqs) == 1:
        tag = tags[0]
        seq = seqs[0]
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=local_alignment_dir,
            seqemb_mode=args.use_single_seq_mode,
        )
    else:
        with open(tmp_fasta_path, "w") as fp:
            fp.write(
                '\n'.join([f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)])
            )
        feature_dict = data_processor.process_multiseq_fasta(
            fasta_path=tmp_fasta_path, super_alignment_dir=alignment_dir,
        )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


def main(args):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Write run metadata (captures all command-line arguments and environment info)
    write_run_metadata(args, args.output_dir)

    if args.config_preset.startswith("seq"):
        args.use_single_seq_mode = True

    config = model_config(
        args.config_preset, 
        long_sequence_inference=args.long_sequence_inference,
        use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
        )

    if args.experiment_config_json: 
        with open(args.experiment_config_json, 'r') as f:
            custom_config_dict = json.load(f)
        config.update_from_flattened_dict(custom_config_dict)

    if args.experiment_config_json: 
        with open(args.experiment_config_json, 'r') as f:
            custom_config_dict = json.load(f)
        config.update_from_flattened_dict(custom_config_dict)

    if args.trace_model:
        if not config.data.predict.fixed_size:
            raise ValueError(
                "Tracing requires that fixed_size mode be enabled in the config"
            )

    if args.trace_model and args.torch_compile:
        logger.warning(
            "Both --trace_model and --torch_compile are enabled; combining "
            "TorchScript tracing with torch.compile can be unsupported or "
            "redundant. Prefer one or the other unless you have verified this stack."
        )

    if args.pad_seqlen_multiple > 0 and not args.torch_compile:
        logger.info(
            "--pad_seqlen_multiple is set without --torch_compile: shape bucketing "
            "mainly helps torch.compile/Inductor reuse; eager mode sees little "
            "kernel-cache benefit (extra padding still adds compute)."
        )

    is_multimer = "multimer" in args.config_preset

    if is_multimer:
        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=config.data.predict.max_templates,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path
        )
    else:
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=config.data.predict.max_templates,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path
        )

    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer,
    )

    if is_multimer:
        data_processor = data_pipeline.DataPipelineMultimer(
            monomer_data_pipeline=data_processor,
        )

    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2 ** 32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    if args.use_precomputed_alignments is None:
        alignment_dir = os.path.join(output_dir_base, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments

    tag_list = []
    seq_list = []
    for fasta_file in list_files_with_extensions(args.fasta_dir, (".fasta", ".fa")):
        # Gather input sequences
        fasta_path = os.path.join(args.fasta_dir, fasta_file)
        with open(fasta_path, "r") as fp:
            data = fp.read()

        tags, seqs = parse_fasta(data)

        if not is_multimer and len(tags) != 1:
            print(
                f"{fasta_path} contains more than one sequence but "
                f"multimer mode is not enabled. Skipping..."
            )
            continue

        # assert len(tags) == len(set(tags)), "All FASTA tags must be unique"
        tag = '-'.join(tags)

        tag_list.append((tag, tags))
        seq_list.append(seqs)

    seq_sort_fn = lambda target: sum([len(s) for s in target[1]])
    sorted_targets = sorted(zip(tag_list, seq_list), key=seq_sort_fn)
    feature_dicts = {}

    if is_multimer and args.openfold_checkpoint_path:
        raise ValueError(
            '`openfold_checkpoint_path` was specified, but no OpenFold checkpoints are available for multimer mode')

    model_generator = load_models_from_command_line(
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

    for model, output_directory in model_generator:
        cur_tracing_interval = 0
        for (tag, tags), seqs in sorted_targets:
            output_name = f'{tag}_{args.config_preset}'
            if args.output_postfix is not None:
                output_name = f'{output_name}_{args.output_postfix}'

            # Does nothing if the alignments have already been computed
            precompute_alignments(tags, seqs, alignment_dir, args)

            feature_dict = feature_dicts.get(tag, None)
            if feature_dict is None:
                feature_dict = generate_feature_dict(
                    tags,
                    seqs,
                    alignment_dir,
                    data_processor,
                    args,
                )

                n = feature_dict["aatype"].shape[-2]
                target_seqlen = n
                if args.pad_seqlen_multiple > 0:
                    target_seqlen = max(
                        target_seqlen,
                        round_up_seqlen_multiple(n, args.pad_seqlen_multiple),
                    )
                if args.trace_model:
                    target_seqlen = max(target_seqlen, round_up_seqlen(n))
                if target_seqlen > n:
                    logger.info(
                        f"Padded sequence length {n} -> {target_seqlen} for {tag}"
                    )
                    feature_dict = pad_feature_dict_seq(
                        feature_dict, target_seqlen,
                    )
                rounded_seqlen = target_seqlen

                feature_dicts[tag] = feature_dict

            processed_feature_dict = feature_processor.process_features(
                feature_dict, mode='predict', is_multimer=is_multimer
            )

            processed_feature_dict = {
                k: torch.as_tensor(v, device=args.model_device)
                for k, v in processed_feature_dict.items()
            }

            if args.trace_model:
                if rounded_seqlen > cur_tracing_interval:
                    logger.info(
                        f"Tracing model at {rounded_seqlen} residues..."
                    )
                    t = time.perf_counter()
                    trace_model_(model, processed_feature_dict)
                    tracing_time = time.perf_counter() - t
                    logger.info(
                        f"Tracing time: {tracing_time}"
                    )
                    cur_tracing_interval = rounded_seqlen

            if args.inference_warmup:
                run_model(
                    model,
                    processed_feature_dict,
                    tag,
                    args.output_dir,
                    record_timings=False,
                )
            out = run_model(model, processed_feature_dict, tag, args.output_dir)

            # Toss out the recycling dimensions --- we don't need them anymore
            processed_feature_dict = tensor_tree_map(
                lambda x: np.array(x[..., -1].cpu()),
                processed_feature_dict
            )
            out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

            unrelaxed_protein = prep_output(
                out,
                processed_feature_dict,
                feature_dict,
                feature_processor,
                args.config_preset,
                args.multimer_ri_gap,
                args.subtract_plddt
            )

            unrelaxed_file_suffix = "_unrelaxed.pdb"
            if args.cif_output:
                unrelaxed_file_suffix = "_unrelaxed.cif"
            unrelaxed_output_path = os.path.join(
                output_directory, f'{output_name}{unrelaxed_file_suffix}'
            )

            with open(unrelaxed_output_path, 'w') as fp:
                if args.cif_output:
                    fp.write(protein.to_modelcif(unrelaxed_protein))
                else:
                    fp.write(protein.to_pdb(unrelaxed_protein))

            logger.info(f"Output written to {unrelaxed_output_path}...")

            if not args.skip_relaxation:
                # Relax the prediction.
                logger.info(f"Running relaxation on {unrelaxed_output_path}...")
                relax_protein(config, args.model_device, unrelaxed_protein, output_directory, output_name,
                              args.cif_output)

            if args.save_outputs:
                output_dict_path = os.path.join(
                    output_directory, f'{output_name}_output_dict.pkl'
                )
                with open(output_dict_path, "wb") as fp:
                    pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

                logger.info(f"Model output written to {output_dict_path}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_dir", type=str,
        help="Path to directory containing FASTA files, one sequence per file"
    )
    parser.add_argument(
        "template_mmcif_dir", type=str,
    )
    parser.add_argument(
        "--use_precomputed_alignments", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--use_single_seq_mode", action="store_true", default=False,
        help="""Use single sequence embeddings instead of MSAs."""
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--config_preset", type=str, default="model_1",
        help="""Name of a model config preset defined in openfold/config.py"""
    )
    parser.add_argument(
        "--jax_param_path", type=str, default=None,
        help="""Path to JAX model parameters. If None, and openfold_checkpoint_path
             is also None, parameters are selected automatically according to 
             the model name from openfold/resources/params"""
    )
    parser.add_argument(
        "--openfold_checkpoint_path", type=str, default=None,
        help="""Path to OpenFold checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file"""
    )
    parser.add_argument(
        "--save_outputs", action="store_true", default=False,
        help="Whether to save all model outputs, including embeddings, etc."
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        "--preset", type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    parser.add_argument(
        "--output_postfix", type=str, default=None,
        help="""Postfix for output prediction filenames"""
    )
    parser.add_argument(
        "--data_random_seed", type=int, default=None
    )
    parser.add_argument(
        "--skip_relaxation", action="store_true", default=False,
    )
    parser.add_argument(
        "--multimer_ri_gap", type=int, default=200,
        help="""Residue index offset between multiple sequences, if provided"""
    )
    parser.add_argument(
        "--inference_warmup", action="store_true", default=False,
        help="""Run a warmup inference step before the main inference to improve
                steady-state performance."""
    )
    parser.add_argument(
        "--trace_model", action="store_true", default=False,
        help="""Whether to convert parts of each model to TorchScript.
                Significantly improves runtime at the cost of lengthy
                'compilation.' Useful for large batch jobs."""
    )
    parser.add_argument(
        "--pad_seqlen_multiple",
        type=int,
        default=0,
        help="""If > 0, pad each sequence length up to the next multiple of this
                value (bucket padding). Fewer distinct shapes help
                torch.compile/Inductor reuse kernels and on-disk cache across
                proteins with similar lengths; combines with --trace_model by
                taking the max of both paddings. Costs extra compute on padded
                residues (masks preserve correctness). Little benefit for
                eager-only runs without compile."""
    )
    parser.add_argument(
        "--torch_compile", action="store_true", default=False,
        help="""Wrap the loaded model with torch.compile (PyTorch 2+). Improves
                steady-state inference after the first compiled forward pass.
                Inductor cache defaults to $XDG_CACHE_HOME/openfold/torch_inductor
                (usually ~/.cache/openfold/torch_inductor); override with
                --torch_inductor_cache_dir."""
    )
    parser.add_argument(
        "--torch_inductor_cache_dir", type=str, default=None,
        help="""With --torch_compile: sets TORCHINDUCTOR_CACHE_DIR to this path.
                If omitted, uses $XDG_CACHE_HOME/openfold/torch_inductor (usually
                ~/.cache/openfold/torch_inductor). Clears the cache when stale or
                when --torch_inductor_cache_force_rebuild is set; writes
                .openfold_inductor_cache_meta.json for checkpoint/version checks."""
    )
    parser.add_argument(
        "--torch_inductor_cache_force_rebuild", action="store_true",
        default=False,
        help="""If set with --torch_compile, clears the Inductor cache directory
                (default or --torch_inductor_cache_dir) before compiling."""
    )
    parser.add_argument(
        "--torch_inductor_no_static_launcher",
        action="store_true",
        default=False,
        help="""With --torch_compile: disable Inductor's static Triton/CUDA kernel
                launcher (same idea as TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0).
                Try if you see InductorError / CUDA driver error (e.g. 301) during
                Triton kernel load; clear the Inductor cache when toggling this."""
    )
    parser.add_argument(
        "--torch_compile_strategy",
        type=str,
        choices=("full", "submodules"),
        default="full",
        help="""With --torch_compile: \"full\" wraps the entire AlphaFold model
                (single torch.compile, default mode). \"submodules\" compiles
                evoformer, structure_module, aux_heads, input_embedder, and
                recycling_embedder separately; use --torch_compile_evoformer_mode,
                --torch_compile_structure_module_mode, and
                --torch_compile_submodule_mode for their mode strings."""
    )
    parser.add_argument(
        "--torch_compile_evoformer_mode",
        type=str,
        default=None,
        help="""torch.compile mode for model.evoformer when
                --torch_compile_strategy=submodules. Default: PyTorch default
                mode (omit to use the same as a bare torch.compile).""",
    )
    parser.add_argument(
        "--torch_compile_structure_module_mode",
        type=str,
        default=None,
        help="""torch.compile mode for model.structure_module when
                --torch_compile_strategy=submodules. Default: PyTorch default.""",
    )
    parser.add_argument(
        "--torch_compile_submodule_mode",
        type=str,
        default=None,
        help="""torch.compile mode for aux_heads, input_embedder, and
                recycling_embedder when --torch_compile_strategy=submodules.
                Default: PyTorch default.""",
    )
    parser.add_argument(
        "--subtract_plddt", action="store_true", default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself"""
    )
    parser.add_argument(
        "--long_sequence_inference", action="store_true", default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details"""
    )
    parser.add_argument(
        "--cif_output", action="store_true", default=False,
        help="Output predicted models in ModelCIF format instead of PDB format (default)"
    )
    parser.add_argument(
        "--experiment_config_json", default="", help="Path to a json file with custom config values to overwrite config setting",
    )
    parser.add_argument(
        "--use_deepspeed_evoformer_attention", action="store_true", default=False, 
        help="Whether to use the DeepSpeed evoformer attention layer. Must have deepspeed installed in the environment.",
    )
    add_data_args(parser)
    args = parser.parse_args()

    if args.jax_param_path is None and args.openfold_checkpoint_path is None:
        args.jax_param_path = os.path.join(
            "openfold", "resources", "params",
            "params_" + args.config_preset + ".npz"
        )

    if args.model_device == "cpu" and torch.cuda.is_available():
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )

    main(args)
