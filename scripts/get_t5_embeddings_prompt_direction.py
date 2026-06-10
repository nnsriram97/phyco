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
import os
import pickle
import torch

import numpy as np

from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder

"""example command
python -m scripts.get_t5_embeddings_prompt --prompt "A beautiful sunset over the ocean" --save_path "output/embedding.pickle"
"""

directions = ["left", "right", "up", "down", "up-left", "up-right", "down-left", "down-right"]
dir_replace_texts = ["left", "right", "upward", "downward", "top left", "top right", "bottom left", "bottom right"]

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute T5 embeddings for a single text prompt")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to encode")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the embedding (including filename)")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length of the text embedding")
    parser.add_argument(
        "--cache_dir", type=str, default="/net/acadia1a/data/sriram/cosmos/google-t5/t5-11b", 
        help="Directory to cache the T5 model"
    )
    return parser.parse_args()


def main(args) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    # Initialize T5 using CosmosT5TextEncoder
    encoder = CosmosT5TextEncoder(cache_dir=args.cache_dir, local_files_only=True, use_8bit=False)

    for direction, dir_replace_text in zip(directions, dir_replace_texts):
        # Prompt has a {direction} placeholder, replace it with the direction text
        prompt = args.prompt.replace("{direction}", dir_replace_text)
        print(f"Encoding prompt: {prompt}")
        
        encoded_text, mask_bool = encoder.encode_prompts(
            [prompt], max_length=args.max_length, return_mask=True
        )  # list of np.ndarray in (len, 1024)
        attn_mask = mask_bool.long()
        lengths = attn_mask.sum(dim=1).cpu()

        encoded_text = encoded_text.cpu().to(torch.float16)
        
        # trim zeros to save space
        encoded_text_trimmed = encoded_text[0][: lengths[0]]
        save_path = args.save_path.replace(".pt", f"_{direction}.pt")
        torch.save(encoded_text_trimmed, save_path)
        # Write the prompt to a text file
        with open(save_path.replace(".pt", ".txt"), "w") as f:
            f.write(prompt)
        print(f"Embedding saved to: {save_path}")
        print(f"Embedding shape: {[emb.shape for emb in encoded_text]}")


if __name__ == "__main__":
    args = parse_args()
    main(args) 