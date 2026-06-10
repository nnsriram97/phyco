# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import re
from typing import Any, Dict, List

import torch

from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate T5 embeddings for every entry in a JSON batch file."
    )
    parser.add_argument("--json-path", type=str, required=True, help="Input JSON describing the items.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where per-sample embeddings (.pt) will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/net/acadia1a/data/sriram/cosmos/google-t5/t5-11b",
        help="Directory that contains or will cache the T5 model weights.",
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="prompt",
        help="JSON key that stores the text prompt to encode.",
    )
    parser.add_argument(
        "--fallback_prompt",
        type=str,
        default="",
        help="Fallback prompt text when a JSON item is missing the prompt key.",
    )
    parser.add_argument(
        "--name_key",
        type=str,
        default="output_video",
        help="JSON key that will be used to derive the embedding filename (before .pt).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum number of tokens for the T5 encoder.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate embeddings even if the output file already exists.",
    )
    parser.add_argument(
        "--write_json",
        action="store_true",
        help="Write updated JSON with t5_embeddings_path fields (overwrites input unless --json_out is set).",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Optional path to save the updated JSON. Defaults to --json_path when --write_json is set.",
    )
    parser.add_argument(
        "--write_prompts",
        action="store_true",
        help="Store each prompt next to its embedding as <name>.txt for easy inspection.",
    )
    return parser.parse_args()


def _sanitize_name(value: Any, fallback: str) -> str:
    if isinstance(value, str) and value:
        base = os.path.splitext(os.path.basename(value))[0]
    elif value is not None:
        base = str(value)
    else:
        base = fallback
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base.strip())
    return base or fallback


def _choose_name(item: Dict[str, Any], index: int, primary_key: str) -> str:
    candidates: List[Any] = []
    if primary_key:
        candidates.append(item.get(primary_key))
    candidates.extend(
        [
            item.get("t5_embeddings_name"),
            item.get("t5_filename"),
            item.get("output_video"),
            item.get("input_video"),
            item.get("input_frame"),
            item.get("id"),
        ]
    )
    for candidate in candidates:
        name = _sanitize_name(candidate, "")
        if name:
            return name
    return f"sample_{index:05d}"


def _encode_prompt(encoder: CosmosT5TextEncoder, prompt: str, max_length: int) -> torch.Tensor:
    encoded_text, mask_bool = encoder.encode_prompts([prompt], max_length=max_length, return_mask=True)
    attn_mask = mask_bool.long()
    lengths = attn_mask.sum(dim=1).cpu()
    encoded_text = encoded_text.cpu().to(torch.float16)
    trimmed = encoded_text[0][: lengths[0]]
    return trimmed


def main() -> None:
    args = parse_args()
    with open(args.json_path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    encoder = CosmosT5TextEncoder(cache_dir=args.cache_dir, local_files_only=True, use_8bit=False)

    saved = 0
    skipped = 0

    for idx, item in enumerate(items):
        prompt = item.get(args.prompt_key) or args.fallback_prompt
        if not prompt:
            print(f"[WARN] Item {idx} missing prompt key '{args.prompt_key}', skipping.")
            skipped += 1
            continue

        name = _choose_name(item, idx, args.name_key)
        embedding_path = os.path.join(args.output_dir, f"{name}.pt")
        if not args.overwrite and os.path.exists(embedding_path):
            print(f"[SKIP] {embedding_path} already exists.")
            item["t5_embeddings_path"] = embedding_path
            skipped += 1
            continue

        print(f"[ENCODE] idx={idx} name={name}")
        embedding = _encode_prompt(encoder, prompt, args.max_length)
        torch.save(embedding, embedding_path)
        if args.write_prompts:
            with open(os.path.join(args.output_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.write(prompt)
        item["t5_embeddings_path"] = embedding_path
        saved += 1

    print(f"Finished encoding prompts. saved={saved} skipped={skipped}")

    if args.write_json:
        json_out = args.json_out or args.json_path
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=4, ensure_ascii=False)
        print(f"Wrote updated JSON with t5_embeddings_path fields to: {json_out}")


if __name__ == "__main__":
    main()
