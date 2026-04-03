#!/usr/bin/env python3
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Client for serve_openfold.py: reads the same FASTA layout as run_pretrained_openfold.py,
# calls POST /predict once per target, writes structures under output_dir/predictions/ and
# merges timings into output_dir/timings.json (the server does not record batch context).

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("client_openfold_server")

from openfold.utils.script_utils import parse_fasta

from run_pretrained_openfold import list_files_with_extensions


def _merge_timings_json(path: str, new_entries: dict[str, Any], replace: bool) -> None:
    if replace or not os.path.exists(path):
        merged = dict(new_entries)
    else:
        with open(path, "r") as f:
            try:
                merged = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Replacing invalid JSON in %s", path)
                merged = {}
        merged.update(new_entries)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(merged, f, indent=2)
    logger.info("Wrote %s", path)


def _post_predict(
    server_url: str,
    tag: str,
    sequence: str,
    inference_warmup: bool,
    timeout: float,
) -> dict[str, Any]:
    base = server_url.rstrip("/")
    url = f"{base}/predict"
    body = json.dumps(
        {
            "tag": tag,
            "sequence": sequence,
            "inference_warmup": inference_warmup,
        }
    ).encode()
    req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drive serve_openfold one sequence at a time; write predictions + timings.json.",
    )
    parser.add_argument(
        "fasta_dir",
        type=str,
        help="Directory of .fasta / .fa files (same as run_pretrained_openfold.py)",
    )
    parser.add_argument(
        "--server_url",
        type=str,
        default=os.environ.get("OPENFOLD_SERVER_URL", "http://127.0.0.1:8000"),
        help="Base URL of serve_openfold (default: env OPENFOLD_SERVER_URL or http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        help="Output root (same role as run_pretrained_openfold --output_dir)",
    )
    parser.add_argument(
        "--config_preset",
        type=str,
        default="model_1",
        help="Used only for output filenames; must match the running server config.",
    )
    parser.add_argument("--output_postfix", type=str, default=None)
    parser.add_argument(
        "--inference_warmup",
        action="store_true",
        default=False,
        help="Pass inference_warmup=true on every /predict call (extra untimed forward on server).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-request timeout in seconds (default: no timeout).",
    )
    parser.add_argument(
        "--timings_replace",
        action="store_true",
        default=False,
        help="Overwrite timings.json with this run only instead of merging.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.fasta_dir):
        logger.error("Not a directory: %s", args.fasta_dir)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    predictions_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    tag_list: list[tuple[str, list[str]]] = []
    seq_list: list[list[str]] = []

    for fasta_file in list_files_with_extensions(args.fasta_dir, (".fasta", ".fa")):
        fasta_path = os.path.join(args.fasta_dir, fasta_file)
        with open(fasta_path, "r") as fp:
            data = fp.read()

        tags, seqs = parse_fasta(data)
        if len(tags) != 1:
            logger.warning(
                "%s contains %d sequences; client supports monomer only. Skipping.",
                fasta_path,
                len(tags),
            )
            continue

        tag = "-".join(tags)
        tag_list.append((tag, tags))
        seq_list.append(seqs)

    seq_sort_fn = lambda target: sum(len(s) for s in target[1])
    sorted_targets = sorted(zip(tag_list, seq_list), key=seq_sort_fn)

    timings_path = os.path.join(args.output_dir, "timings.json")
    batch_timings: dict[str, dict[str, float]] = {}

    for (tag, _tags), seqs in sorted_targets:
        sequence = seqs[0]
        output_name = f"{tag}_{args.config_preset}"
        if args.output_postfix is not None:
            output_name = f"{output_name}_{args.output_postfix}"

        try:
            payload = _post_predict(
                args.server_url,
                tag,
                sequence,
                args.inference_warmup,
                args.timeout,
            )
        except HTTPError as e:
            logger.error("HTTP %s for tag=%s: %s", e.code, tag, e.read().decode()[:500])
            sys.exit(1)
        except URLError as e:
            logger.error("Request failed for tag=%s: %s", tag, e)
            sys.exit(1)

        inference_s = float(payload.get("inference_seconds", 0.0))
        batch_timings[tag] = {"inference": inference_s}

        fmt = payload.get("structure_format", "pdb")
        structure = payload.get("structure", "")
        suffix = "_unrelaxed.cif" if fmt == "cif" else "_unrelaxed.pdb"
        out_path = os.path.join(predictions_dir, f"{output_name}{suffix}")
        with open(out_path, "w") as fp:
            fp.write(structure)
        logger.info("Wrote %s (inference %.4fs server-reported)", out_path, inference_s)

    _merge_timings_json(timings_path, batch_timings, replace=args.timings_replace)


if __name__ == "__main__":
    main()
