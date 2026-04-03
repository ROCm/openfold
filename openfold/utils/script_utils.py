import json
import logging
import os
import re
import shutil
import time

import numpy
import torch

from openfold.model.model import AlphaFold
from openfold.np import residue_constants, protein
from openfold.np.relax import relax
from openfold.utils.import_weights import (
    import_jax_weights_,
    import_openfold_weights_
)

from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict
)

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

# Written alongside a dedicated TORCHINDUCTOR_CACHE_DIR to detect stale caches.
_INDUCTOR_CACHE_META_FILENAME = ".openfold_inductor_cache_meta.json"


def _default_torch_inductor_cache_dir():
    """
    Default Inductor cache: ``$XDG_CACHE_HOME/openfold/torch_inductor``, or
    ``$HOME/.cache/openfold/torch_inductor`` when ``XDG_CACHE_HOME`` is unset
    (XDG default).
    """
    cache_home = os.environ.get(
        "XDG_CACHE_HOME",
        os.path.join(os.path.expanduser("~"), ".cache"),
    )
    return os.path.join(cache_home, "openfold", "torch_inductor")


def _apply_torch_inductor_no_static_launcher():
    """
    Use Triton's own launcher instead of Inductor's static CUDA launcher.
    Helps some environments that hit CUDA driver errors inside
    ``static_triton_launcher.load_kernel`` (e.g. error 301).
    """
    try:
        import torch._inductor.config as inductor_config

        inductor_config.use_static_triton_launcher = False
        logger.info(
            "Inductor: static Triton/CUDA launcher disabled "
            "(equivalent to TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0)."
        )
    except Exception as e:
        logger.warning("Could not disable static Triton launcher: %s", e)


def _checkpoint_stat_signature(path):
    """Stable identity for a weight file or directory used for cache staleness."""
    if not os.path.exists(path):
        return None
    st = os.stat(path)
    sig = {"mtime_ns": st.st_mtime_ns, "is_dir": os.path.isdir(path)}
    if os.path.isfile(path):
        sig["size"] = st.st_size
    return sig


def _read_inductor_cache_meta(cache_dir):
    p = os.path.join(cache_dir, _INDUCTOR_CACHE_META_FILENAME)
    if not os.path.isfile(p):
        return None
    with open(p, "r") as f:
        return json.load(f)


def _write_inductor_cache_meta(cache_dir, checkpoint_path, stat_sig):
    os.makedirs(cache_dir, exist_ok=True)
    meta = {
        "torch_version": torch.__version__,
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_stat": stat_sig,
    }
    path = os.path.join(cache_dir, _INDUCTOR_CACHE_META_FILENAME)
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                existing = json.load(f)
            if existing == meta:
                return
        except (json.JSONDecodeError, OSError):
            pass
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def _is_inductor_cache_stale(meta, checkpoint_path):
    if meta is None:
        return True
    if meta.get("torch_version") != torch.__version__:
        return True
    if meta.get("checkpoint_path") != os.path.abspath(checkpoint_path):
        return True
    cur_sig = _checkpoint_stat_signature(checkpoint_path)
    if cur_sig is None:
        return True
    if meta.get("checkpoint_stat") != cur_sig:
        return True
    return False


def _clear_inductor_cache_dir_contents(cache_dir):
    """Remove cache entries but keep the directory. Use a dedicated cache_dir."""
    os.makedirs(cache_dir, exist_ok=True)
    for name in os.listdir(cache_dir):
        p = os.path.join(cache_dir, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except OSError as e:
            logger.warning("Could not remove %s: %s", p, e)


def _torch_compile_with_optional_mode(module, mode=None):
    """``torch.compile`` with PyTorch's default when ``mode`` is None."""
    if mode is None:
        return torch.compile(module)
    return torch.compile(module, mode=mode)


def _apply_torch_compile_submodules(
    model,
    evoformer_mode=None,
    structure_module_mode=None,
    embed_heads_mode=None,
):
    """
    Compile major AlphaFold trunk pieces separately so each can use a different
    ``mode=`` (e.g. Evoformer vs structure module vs embedders/heads).
    ``None`` uses PyTorch's default ``torch.compile`` mode for that submodule.
    """
    model.evoformer = _torch_compile_with_optional_mode(
        model.evoformer, evoformer_mode
    )
    model.structure_module = _torch_compile_with_optional_mode(
        model.structure_module, structure_module_mode
    )
    model.aux_heads = _torch_compile_with_optional_mode(
        model.aux_heads, embed_heads_mode
    )
    model.input_embedder = _torch_compile_with_optional_mode(
        model.input_embedder, embed_heads_mode
    )
    model.recycling_embedder = _torch_compile_with_optional_mode(
        model.recycling_embedder, embed_heads_mode
    )
    return model


def maybe_apply_torch_compile(
    model,
    torch_compile,
    torch_inductor_cache_dir,
    checkpoint_path_for_cache,
    force_inductor_cache_rebuild,
    torch_compile_strategy="full",
    torch_compile_evoformer_mode=None,
    torch_compile_structure_module_mode=None,
    torch_compile_submodule_mode=None,
    torch_inductor_no_static_launcher=False,
):
    """
    Optionally wrap the model with torch.compile and manage a dedicated
    Inductor cache directory with simple staleness checks (PyTorch version,
    checkpoint path, and file stats).

    When compiling: a dedicated cache directory is always used. If
    ``torch_inductor_cache_dir`` is ``None``, it defaults to
    ``$XDG_CACHE_HOME/openfold/torch_inductor`` (usually
    ``~/.cache/openfold/torch_inductor``). The directory is created on first
    use; Inductor writes artifacts there; OpenFold writes staleness metadata
    when the checkpoint path is valid.

    ``torch_compile_strategy``: ``\"full\"`` compiles the entire ``AlphaFold``
    module once (default PyTorch compile mode). ``\"submodules\"`` compiles
    ``evoformer``, ``structure_module``, ``aux_heads``, ``input_embedder``,
    and ``recycling_embedder`` separately. Mode arguments default to ``None``,
    i.e. PyTorch's default ``torch.compile`` mode per submodule; pass strings
    such as ``\"max-autotune\"`` when needed (availability depends on PyTorch).
    """
    if not torch_compile:
        return model

    major = int(torch.__version__.split(".")[0])
    if major < 2:
        logger.warning(
            "torch.compile requires PyTorch 2.0+; skipping torch.compile."
        )
        return model

    if torch.version.hip is not None:
        logger.warning(
            "ROCm (HIP) build detected: torch.compile/Inductor/Triton support "
            "varies by ROCm and PyTorch version; stack traces may still mention "
            "'CUDA' via the compatibility layer. If compilation fails (e.g. kernel "
            "load errors), run without --torch_compile, or try "
            "--torch_inductor_no_static_launcher and clear the Inductor cache "
            "under TORCHINDUCTOR_CACHE_DIR."
        )

    if torch_inductor_no_static_launcher:
        _apply_torch_inductor_no_static_launcher()

    effective_cache_dir = (
        torch_inductor_cache_dir
        if torch_inductor_cache_dir is not None
        else _default_torch_inductor_cache_dir()
    )
    cache_dir = os.path.abspath(effective_cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    if torch_inductor_cache_dir is None:
        logger.info(
            "Using default Inductor cache directory %s "
            "(override with --torch_inductor_cache_dir).",
            cache_dir,
        )
    meta = _read_inductor_cache_meta(cache_dir)
    stale = (
        _is_inductor_cache_stale(meta, checkpoint_path_for_cache)
        or force_inductor_cache_rebuild
    )
    if stale:
        if force_inductor_cache_rebuild:
            logger.info(
                "Clearing Inductor cache at %s (force rebuild).",
                cache_dir,
            )
        elif meta is None and not any(
            n != _INDUCTOR_CACHE_META_FILENAME
            for n in os.listdir(cache_dir)
        ):
            logger.info(
                "Using Inductor cache directory %s (created or empty; "
                "will populate on first compiled forward pass).",
                cache_dir,
            )
        else:
            logger.info(
                "Clearing Inductor cache at %s (stale or missing metadata).",
                cache_dir,
            )
        _clear_inductor_cache_dir_contents(cache_dir)
    else:
        logger.info(
            "Reusing Inductor cache at %s (checkpoint and PyTorch version "
            "match saved metadata).",
            cache_dir,
        )
    sig = _checkpoint_stat_signature(checkpoint_path_for_cache)
    if sig is None:
        logger.warning(
            "Checkpoint path %s not found; skipping torch.compile.",
            checkpoint_path_for_cache,
        )
        return model

    if torch_compile_strategy == "submodules":
        logger.info(
            "Applying torch.compile per submodule (evoformer=%r, "
            "structure_module=%r, aux_heads/embedders=%r).",
            torch_compile_evoformer_mode,
            torch_compile_structure_module_mode,
            torch_compile_submodule_mode,
        )
        model = _apply_torch_compile_submodules(
            model,
            evoformer_mode=torch_compile_evoformer_mode,
            structure_module_mode=torch_compile_structure_module_mode,
            embed_heads_mode=torch_compile_submodule_mode,
        )
    else:
        logger.info("Applying torch.compile to the full model (default mode).")
        model = torch.compile(model)

    if sig is not None:
        _write_inductor_cache_meta(cache_dir, checkpoint_path_for_cache, sig)
    return model


def count_models_to_evaluate(openfold_checkpoint_path, jax_param_path):
    model_count = 0
    if openfold_checkpoint_path:
        model_count += len(openfold_checkpoint_path.split(","))
    if jax_param_path:
        model_count += len(jax_param_path.split(","))
    return model_count


def get_model_basename(model_path):
    return os.path.splitext(
                os.path.basename(
                    os.path.normpath(model_path)
                )
            )[0]


def make_output_directory(output_dir, model_name, multiple_model_mode):
    if multiple_model_mode:
        prediction_dir = os.path.join(output_dir, "predictions", model_name)
    else:
        prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    return prediction_dir


def load_models_from_command_line(
    config,
    model_device,
    openfold_checkpoint_path,
    jax_param_path,
    output_dir,
    torch_compile=False,
    torch_inductor_cache_dir=None,
    force_inductor_cache_rebuild=False,
    torch_compile_strategy="full",
    torch_compile_evoformer_mode=None,
    torch_compile_structure_module_mode=None,
    torch_compile_submodule_mode=None,
    torch_inductor_no_static_launcher=False,
):
    # Create the output directory

    multiple_model_mode = count_models_to_evaluate(openfold_checkpoint_path, jax_param_path) > 1
    if multiple_model_mode:
        logger.info(f"evaluating multiple models")

    if jax_param_path:
        for path in jax_param_path.split(","):
            model_basename = get_model_basename(path)
            model_version = "_".join(model_basename.split("_")[1:])
            model = AlphaFold(config)
            model = model.eval()
            import_jax_weights_(
                model, path, version=model_version
            )
            model = model.to(model_device)
            model = maybe_apply_torch_compile(
                model,
                torch_compile,
                torch_inductor_cache_dir,
                path,
                force_inductor_cache_rebuild,
                torch_compile_strategy=torch_compile_strategy,
                torch_compile_evoformer_mode=torch_compile_evoformer_mode,
                torch_compile_structure_module_mode=torch_compile_structure_module_mode,
                torch_compile_submodule_mode=torch_compile_submodule_mode,
                torch_inductor_no_static_launcher=torch_inductor_no_static_launcher,
            )
            logger.info(
                f"Successfully loaded JAX parameters at {path}..."
            )
            output_directory = make_output_directory(output_dir, model_basename, multiple_model_mode)
            yield model, output_directory

    if openfold_checkpoint_path:
        for path in openfold_checkpoint_path.split(","):
            model = AlphaFold(config)
            model = model.eval()
            checkpoint_basename = get_model_basename(path)
            if os.path.isdir(path):
                # A DeepSpeed checkpoint
                ckpt_path = os.path.join(
                    output_dir,
                    checkpoint_basename + ".pt",
                )

                if not os.path.isfile(ckpt_path):
                    convert_zero_checkpoint_to_fp32_state_dict(
                        path,
                        ckpt_path,
                    )
                d = torch.load(ckpt_path)
                import_openfold_weights_(model=model, state_dict=d["ema"]["params"])
            else:
                ckpt_path = path
                d = torch.load(ckpt_path)

                if "ema" in d:
                    # The public weights have had this done to them already
                    d = d["ema"]["params"]
                import_openfold_weights_(model=model, state_dict=d)

            model = model.to(model_device)
            model = maybe_apply_torch_compile(
                model,
                torch_compile,
                torch_inductor_cache_dir,
                ckpt_path,
                force_inductor_cache_rebuild,
                torch_compile_strategy=torch_compile_strategy,
                torch_compile_evoformer_mode=torch_compile_evoformer_mode,
                torch_compile_structure_module_mode=torch_compile_structure_module_mode,
                torch_compile_submodule_mode=torch_compile_submodule_mode,
                torch_inductor_no_static_launcher=torch_inductor_no_static_launcher,
            )
            logger.info(
                f"Loaded OpenFold parameters at {path}..."
            )
            output_directory = make_output_directory(output_dir, checkpoint_basename, multiple_model_mode)
            yield model, output_directory

    if not jax_param_path and not openfold_checkpoint_path:
        raise ValueError(
            "At least one of jax_param_path or openfold_checkpoint_path must "
            "be specified."
        )


def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [re.split('\W| \|', t)[0] for t in tags]

    return tags, seqs


def update_timings(timing_dict, output_file=os.path.join(os.getcwd(), "timings.json")):
    """
    Write dictionary of one or more run step times to a file
    """
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            try:
                timings = json.load(f)
            except json.JSONDecodeError:
                logger.info(f"Overwriting non-standard JSON in {output_file}.")
                timings = {}
    else:
        timings = {}
    timings.update(timing_dict)
    with open(output_file, "w") as f:
        json.dump(timings, f)
    return output_file


def run_model(model, batch, tag, output_dir, record_timings=True):
    with torch.no_grad():
        # Temporarily disable templates if there aren't any in the batch
        template_enabled = model.config.template.enabled
        model.config.template.enabled = template_enabled and any([
            "template_" in k for k in batch
        ])


        if record_timings:
            logger.info(f"Running inference for {tag}...")
        else:
            logger.info(f"Warmup inference for {tag} (not recorded in timings.json)...")
        t = time.perf_counter()
        out = model(batch)
        inference_time = time.perf_counter() - t
        if record_timings:
            logger.info(f"Inference time: {inference_time}")
            update_timings({tag: {"inference": inference_time}}, os.path.join(output_dir, "timings.json"))
        else:
            logger.info(f"Warmup inference time: {inference_time} (not recorded)")

        model.config.template.enabled = template_enabled

    return out


def prep_output(out, batch, feature_dict, feature_processor, config_preset, multimer_ri_gap, subtract_plddt):
    plddt = out["plddt"]

    plddt_b_factors = numpy.repeat(
        plddt[..., None], residue_constants.atom_type_num, axis=-1
    )

    if subtract_plddt:
        plddt_b_factors = 100 - plddt_b_factors

    # Prep protein metadata
    template_domain_names = []
    template_chain_index = None
    if feature_processor.config.common.use_templates and "template_domain_names" in feature_dict:
        template_domain_names = [
            t.decode("utf-8") for t in feature_dict["template_domain_names"]
        ]

        # This works because templates are not shuffled during inference
        template_domain_names = template_domain_names[
                                :feature_processor.config.predict.max_templates
                                ]

        if "template_chain_index" in feature_dict:
            template_chain_index = feature_dict["template_chain_index"]
            template_chain_index = template_chain_index[
                                   :feature_processor.config.predict.max_templates
                                   ]

    no_recycling = feature_processor.config.common.max_recycling_iters
    remark = ', '.join([
        f"no_recycling={no_recycling}",
        f"max_templates={feature_processor.config.predict.max_templates}",
        f"config_preset={config_preset}",
    ])

    # For multi-chain FASTAs
    ri = feature_dict["residue_index"]
    chain_index = (ri - numpy.arange(ri.shape[0])) / multimer_ri_gap
    chain_index = chain_index.astype(numpy.int64)
    cur_chain = 0
    prev_chain_max = 0
    for i, c in enumerate(chain_index):
        if c != cur_chain:
            cur_chain = c
            prev_chain_max = i + cur_chain * multimer_ri_gap

        batch["residue_index"][i] -= prev_chain_max

    unrelaxed_protein = protein.from_prediction(
        features=batch,
        result=out,
        b_factors=plddt_b_factors,
        remove_leading_feature_dimension=False,
        remark=remark,
        parents=template_domain_names,
        parents_chain_index=template_chain_index,
    )

    return unrelaxed_protein


def relax_protein(config, model_device, unrelaxed_protein, output_directory, output_name, cif_output=False):
    amber_relaxer = relax.AmberRelaxation(
        use_gpu=(model_device != "cpu"),
        **config.relax,
    )

    t = time.perf_counter()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", default="")
    if "cuda" in model_device:
        device_no = model_device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device_no
    # the struct_str will contain either a PDB-format or a ModelCIF format string
    struct_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein, cif_output=cif_output)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    relaxation_time = time.perf_counter() - t

    logger.info(f"Relaxation time: {relaxation_time}")
    update_timings({"relaxation": relaxation_time}, os.path.join(output_directory, "timings.json"))

    # Save the relaxed PDB.
    suffix = "_relaxed.pdb"
    if cif_output:
        suffix = "_relaxed.cif"
    relaxed_output_path = os.path.join(
        output_directory, f'{output_name}{suffix}'
    )
    with open(relaxed_output_path, 'w') as fp:
        fp.write(struct_str)

    logger.info(f"Relaxed output written to {relaxed_output_path}...")
