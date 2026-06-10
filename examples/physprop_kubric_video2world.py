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
import math
import os
import pdb
import pickle

import mediapy as mp
import numpy as np

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from decord import VideoReader, cpu
from megatron.core import parallel_state

from cosmos_predict2.configs.physprop_conditioned.config_physprop_conditioned import *
from cosmos_predict2.pipelines.physprop_v2w import PhyspropConditionedVideo2WorldPipeline
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video
import cv2
from examples.setup_utils import setup_lora_physprop_pipeline
from torchvision import transforms as T
from cosmos_predict2.data.dataset_utils import VIDEO_RES_SIZE_INFO
from cosmos_predict2.data.dataset_utils import Resize_Preprocess, ToTensorVideo, detect_aspect_ratio
import matplotlib.pyplot as plt
from cosmos_predict2.data.kubric_data.kubric_utils import (
    get_kubric_seg_frame_as_obj_index,
    get_physprop_as_spatial_data,
    get_physprop_as_fg_bg_vector,
    get_physprop_as_image_blob,
    save_physprop_as_image,
    save_physprop_as_text,
    save_physprop_as_image_blob,
    augment_input_image_with_move_dir,
)
from pycocotools import mask as mask_utils


def _run_name_from_dit(dit_path: str) -> str:
    """Derive an output subfolder name from a checkpoint path.

    Training checkpoints live at ``.../<run_name>/checkpoints/model/iter_*.pt`` so we
    use the 4th-from-last path component as the run name. For shorter/clean paths
    (e.g. ``checkpoints/phyco/phyco_wVLM.pt``) we fall back to the file stem.
    """
    parts = dit_path.rstrip("/").split("/")
    if len(parts) >= 4:
        return parts[-4]
    return os.path.splitext(parts[-1])[0]


# Default negative prompt: read from the bundled file so online encoding matches the
# precomputed embedding (assets/common_neg_prompt-v1.pt). Inline copy is a fallback.
_NEG_PROMPT_TXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "common_neg_prompt-v1.txt")
try:
    with open(_NEG_PROMPT_TXT) as _nf:
        _DEFAULT_NEGATIVE_PROMPT = _nf.read().strip()
except OSError:
    _DEFAULT_NEGATIVE_PROMPT = 'The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.'



def _parse_list_argument(arg: str | None, cast=lambda x: x):
    """Parse comma-separated or JSON list arguments."""
    if arg is None:
        return None
    if isinstance(arg, list):
        return [cast(item) for item in arg]
    arg = arg.strip()
    if not arg:
        return None
    if arg.startswith("["):
        try:
            data = json.loads(arg)
            return [cast(item) for item in data]
        except Exception as exc:
            raise ValueError(f"Could not parse list argument '{arg}': {exc}") from exc
    items = [item.strip() for item in arg.split(",") if item.strip()]
    return [cast(item) for item in items]


def _parse_mapping_argument(arg: str | None):
    """Parse mapping arguments provided as JSON, file path, or key=value pairs."""
    if arg is None:
        return None
    arg = arg.strip()
    if not arg:
        return None
    if os.path.isfile(arg):
        with open(arg, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"File '{arg}' does not contain a JSON object.")
        return {str(k): str(v) for k, v in data.items()}
    if arg.startswith("{"):
        try:
            data = json.loads(arg)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object.")
            return {str(k): str(v) for k, v in data.items()}
        except Exception as exc:
            raise ValueError(f"Could not parse mapping argument '{arg}': {exc}") from exc
    mapping: dict[str, str] = {}
    for pair in arg.split(","):
        if "=" not in pair:
            raise ValueError(
                f"Could not parse mapping argument '{arg}'. Expected key=value pairs separated by commas."
            )
        key, value = pair.split("=", 1)
        mapping[key.strip()] = value.strip()
    return mapping


def _parse_channel_groups_argument(arg: str | None):
    """Parse controlnet channel groups (expects JSON list of lists)."""
    if arg is None:
        return None
    arg = arg.strip()
    if not arg:
        return None
    try:
        data = json.loads(arg)
    except Exception as exc:
        raise ValueError(
            "controlnet_channel_groups must be provided as a JSON list of lists, "
            "e.g. '[[0,1,2],[3,4,5]]'."
        ) from exc
    if not isinstance(data, list):
        raise ValueError("controlnet_channel_groups must be a list of lists.")
    parsed_groups: list[list[int]] = []
    for group in data:
        if not isinstance(group, (list, tuple)):
            raise ValueError("Each controlnet channel group must itself be a list.")
        parsed_groups.append([int(idx) for idx in group])
    return parsed_groups

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video-to-World Generation with Cosmos Predict2")
    parser.add_argument(
        "--model_size",
        choices=["2B"],
        default="2B",
        help="Size of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="",
        help="Custom path to the DiT model checkpoint for post-trained models.",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=None,
        help="Path to Kubric data folder containing rgba.mp4, segmentation.mp4, and metadata.json",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Prompt for conditioning",
    )
    parser.add_argument(
        "--t5_embeddings_path",
        type=str,
        default="",
        help="Path to t5 embeddings",
    )
    parser.add_argument(
        "--num_conditional_frames",
        type=int,
        default=1,
        choices=[1],
        help="Number of frames to condition on (1 for single frame, 5 for multi-frame conditioning)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=12,
        help="Chunk size",
    )
    parser.add_argument(
        "--total_seconds",
        type=float,
        default=None,
        help="If set, duplicate the last frame so each video reaches this duration at the configured fps.",
    )
    parser.add_argument("--autoregressive", action="store_true", help="Use autoregressive mode")
    parser.add_argument("--guidance", type=float, default=7, help="Guidance value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated or JSON list of seeds to run sequentially.",
    )
    parser.add_argument(
        "--save_filename",
        type=str,
        default="",
        help="Filename to save the generated video (include file extension)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/",
        help="Path to save the generated video",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference (should be a divisor of the total frames)",
    )
    parser.add_argument("--disable_guardrail", action="store_true", help="Disable guardrail checks on prompts")
    parser.add_argument(
        "--disable_prompt_refiner", action="store_true", help="Disable prompt refiner that enhances short prompts"
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="256",
        help="Resolution of the generated video",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for conditioning",
    )
    parser.add_argument(
        "--neg_t5_embedding_path",
        type=str,
        default="assets/common_neg_prompt-v1.pt",
        help="Path to negative t5 embedding",
    )
    parser.add_argument(
        "--pipeline_config",
        type=str,
        default="",
        help="Name of the config to use for the pipeline",
    )

    parser.add_argument(
        "--use_text_encoder",
        action="store_true",
        default=False,
        help="Use text encoder",
    )
    parser.add_argument(
        "--physprop_type",
        type=str,
        default="all",
        help="Type of physprop to use",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS of the generated video",
    )
    parser.add_argument(
        "--batch_input_json",
        type=str,
        default=None,
        help="Path to batch input json",
    )
    parser.add_argument(
        "--prepend_fname",
        type=str,
        default="1_",
        help="Prepend filename to the output video",
    )
    parser.add_argument(
        "--conditioning_type",
        type=str,
        default="image",
        help="Type of conditioning to use",
        choices=["image", "fg_bg_vector", "image_blob"],
    )
    parser.add_argument(
        "--blob_type",
        type=str,
        default="circle",
        help="Type of blob to use",
        choices=["circle", "convex_hull", "ellipse"],
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        default=None,
        help="Comma-separated list (e.g. 0,1,2,3) or range interval (e.g. 0-3) or a combination of both (e.g. 0,1,3-5) of run ids to run in the batch",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=1,
        help="Split the batch JSON into this many equal parts for distributed execution.",
    )
    parser.add_argument(
        "--split_index",
        type=int,
        default=0,
        help="Zero-based index of the split to process when num_splits > 1.",
    )

    parser.add_argument(
        "--post_dit_subfolder",
        type=str,
        default=None,
        help="Subfolder of the DiT checkpoint to use for post-training",
    )
    parser.add_argument(
        "--dynamic_controlnet",
        action="store_true",
        default=False,
        help="Dynamic controlnet mode",
    )
    
    # LoRA-specific arguments
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Enable LoRA inference mode",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Rank of the LoRA adaptation",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Alpha parameter for LoRA",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="controlnet_blocks.0.self_attn.q_proj,controlnet_blocks.0.self_attn.k_proj,controlnet_blocks.0.self_attn.v_proj,controlnet_blocks.0.self_attn.output_proj,controlnet_blocks.0.mlp.layer1,controlnet_blocks.0.mlp.layer2,controlnet_blocks.0.cross_attn.q_proj,controlnet_blocks.0.cross_attn.k_proj,controlnet_blocks.0.cross_attn.v_proj,controlnet_blocks.0.cross_attn.output_proj",
        help="Comma-separated list of target modules for LoRA",
    )
    parser.add_argument(
        "--init_lora_weights",
        action="store_true",
        default=True,
        help="Whether to initialize LoRA weights",
    )

    # ControlNet multi-branch options
    parser.add_argument(
        "--controlnet_branch_names",
        type=str,
        default=None,
        help="Comma-separated or JSON list of ControlNet branch names (ordered).",
    )
    parser.add_argument(
        "--active_controlnets",
        type=str,
        default=None,
        help="Comma-separated or JSON list of ControlNet branches to enable during inference.",
    )
    parser.add_argument(
        "--controlnet_branch_scales",
        type=str,
        default=None,
        help="Comma-separated or JSON list of conditioning scales per active ControlNet branch.",
    )
    parser.add_argument(
        "--controlnet_branch_ckpts",
        type=str,
        default=None,
        help="Mapping of branch names to checkpoint paths. Accepts JSON, key=value pairs, or a JSON file path.",
    )
    parser.add_argument(
        "--controlnet_channels_per_controlnet",
        type=int,
        default=None,
        help="Override number of physprop channels routed to each ControlNet branch.",
    )
    parser.add_argument(
        "--controlnet_channel_groups",
        type=str,
        default=None,
        help="JSON list defining explicit physprop channel groups per ControlNet branch, e.g. '[[0,1,2],[3,4,5]]'.",
    )
    parser.add_argument(
        "--no_background_condition",
        action="store_true",
        default=False,
        help="No background condition",
    )

    parser.add_argument(
        "--no_regenerate_videos",
        action="store_true",
        default=False,
        help="No regenerate videos",
    )

    args = parser.parse_args()

    # Post-process complex arguments into convenient Python objects.
    args.controlnet_branch_names = _parse_list_argument(args.controlnet_branch_names)
    args.active_controlnets = _parse_list_argument(args.active_controlnets)
    args.controlnet_branch_scales = _parse_list_argument(args.controlnet_branch_scales, cast=float)
    args.controlnet_branch_ckpts = _parse_mapping_argument(args.controlnet_branch_ckpts)
    args.controlnet_channel_groups = _parse_channel_groups_argument(args.controlnet_channel_groups)
    args.seeds = _parse_list_argument(args.seeds, cast=int)

    return args


def _sample_frames(video_path, start_min=0, start_max=0, sampling_rate=1, desired_fps=10, num_frames=1):
    """Sample frames from video similar to kubric_dataset.py"""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
    n_frames = len(vr)
    start_frame = np.random.randint(start_min, start_max + 1)
    try:
        video_fps = vr.get_avg_fps()
    except Exception:  # failed to read FPS
        video_fps = 30
    if desired_fps is not None:
        sampling_rate = int(video_fps / desired_fps)
    
    frame_ids = list(range(start_frame, n_frames, sampling_rate))
    frame_ids = frame_ids[:num_frames]
    frames = vr.get_batch(frame_ids).asnumpy()
    frames = frames.astype(np.uint8)
    
    return frames, frame_ids, int(video_fps/sampling_rate)

def _get_depth_video(depth_path):
    depth_data = np.load(depth_path)
    depth_frames = depth_data["arr_0"].squeeze(axis=-1)
    depth_frames = depth_frames.astype(np.float32) # [T, H, W]
    # Normalize depth and convert to uint8 3 channel image 
    # Reverse normalization: max depth -> 0, min depth -> 1
    min_depth = depth_frames.min(axis=(1, 2), keepdims=True)
    max_depth = depth_frames.max(axis=(1, 2), keepdims=True)
    depth_frames = (max_depth - depth_frames) / (max_depth - min_depth)
    depth_frames = (depth_frames * 255).astype(np.uint8)
    depth_frames = np.stack([depth_frames, depth_frames, depth_frames], axis=-1) # [T, H, W, 3]

    return depth_frames

def load_kubric_data(data_folder, physprop_type="all", physprops_range = None, conditioning_type="image", force_magnitude=None, dir_angle=None, move_object_seg_id=None, fg_seg_ids=None, bg_seg_id=None, fg_seg_start_id=None, props_of_interest=None, blob_type=None):
    """Load data from Kubric data folder"""
    video_path = os.path.join(data_folder, "rgba.mp4")
    seg_video_path = os.path.join(data_folder, "segmentation.mp4")
    metadata_path = os.path.join(data_folder, "metadata.json")
    depth_path = os.path.join(data_folder, "depth.npz")
    
    # Check if required files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"rgba.mp4 not found in {data_folder}")
    if not os.path.exists(seg_video_path):
        raise FileNotFoundError(f"segmentation.mp4 not found in {data_folder}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {data_folder}")
    
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    # Sample first frame from rgba video
    frames, frame_ids, fps = _sample_frames(video_path, start_min=0, start_max=0, desired_fps=10, num_frames=5)
    video = torch.from_numpy(frames)
    video = video.permute(3, 0, 1, 2)  # Rearrange from [T, H, W, C] to [C, T, H, W]
    # Resize the first frame to 256x256
    aspect_ratio = detect_aspect_ratio((video.shape[2], video.shape[3]))  # expects (W, H)
    video_size = VIDEO_RES_SIZE_INFO[args.resolution][aspect_ratio]
    preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(video_size)])
    video = preprocess(video)
    
    video = torch.clamp(video * 255.0, 0, 255).to(torch.uint8)
    video_np = video.cpu().numpy()
    first_frame = video_np[:, 0]
    first_frame = first_frame.transpose(1, 2, 0)

    # Load depth frame
    if os.path.exists(depth_path):
        depth_frames = _get_depth_video(depth_path)
        depth_frame = depth_frames[0]
        depth_frame = cv2.resize(depth_frame, (video_size[1], video_size[0]))
    else:
        depth_frame = np.zeros((video_size[0], video_size[1], 3), dtype=np.uint8)
    
    # Sample first frame from segmentation video
    seg_frames, seg_frame_ids, seg_fps = _sample_frames(seg_video_path, start_min=0, start_max=0, desired_fps=10, num_frames=1)
    first_seg_frame = seg_frames[0]
    first_seg_frame = cv2.resize(first_seg_frame, (video_size[1], video_size[0]))
    
    first_seg_frame = get_kubric_seg_frame_as_obj_index(first_seg_frame, metadata["segmentation_color_map"])
    
    metadata_update = {}
    if "applied_forces_image" in metadata:
        force_information = metadata["applied_forces_image"][0]
        force_metadata = {}
        force_metadata["force_magnitude"] = force_information["force_magnitude"]
        force_metadata["dir_start_image_coordinates"] = force_information["image_coordinates"]
        force_metadata["dir_end_image_coordinates"] = force_information["force_end_image_coordinates"]
        force_metadata["force_magnitude_min"] = metadata["min_force"]
        force_metadata["force_magnitude_max"] = metadata["max_force"]
        force_sim_info = metadata["applied_forces_simulator"][0]
        force_object_name = force_sim_info["object_name"]
        object_name_idx = metadata["object_data"]["object_name"].index(force_object_name)
        move_object_seg_id = metadata["object_data"]["segmentation_id"][object_name_idx]
        force_metadata["move_object_seg_id"] = move_object_seg_id
        metadata_update.update(force_metadata)
    
    if "applied_velocities_image" in metadata:
        velocity_information = metadata["applied_velocities_image"][0]
        velocity_metadata = {}
        velocity_metadata["dir_start_image_coordinates"] = velocity_information["image_coordinates"]
        velocity_metadata["dir_end_image_coordinates"] = velocity_information["velocity_end_image_coordinates"]
        velocity_sim_info = metadata["applied_velocities_simulator"][0]
        velocity_object_name = velocity_sim_info["object_name"]
        object_name_idx = metadata["object_data"]["object_name"].index(velocity_object_name)
        move_object_seg_id = metadata["object_data"]["segmentation_id"][object_name_idx]
        velocity_metadata["move_object_seg_id"] = move_object_seg_id
        metadata_update.update(velocity_metadata)

    if force_magnitude is not None:
        metadata_update["force_magnitude"] = force_magnitude
    if dir_angle is not None:
        metadata_update["dir_angle"] = dir_angle
    if move_object_seg_id is not None:
        metadata_update["move_object_seg_id"] = move_object_seg_id
    
    if fg_seg_start_id is not None:
        fg_seg_ids = metadata["object_data"]["segmentation_id"][fg_seg_start_id:]

    
    # Create physprop frame
    if conditioning_type=="image":
        physprop_frame = get_physprop_as_spatial_data(metadata["object_data"], first_seg_frame, physprop_type, physprops_range)
    elif conditioning_type=="fg_bg_vector":
        physprop_frame = get_physprop_as_fg_bg_vector(metadata["object_data"], physprop_type, physprops_range)
    elif conditioning_type=="image_blob":
        sim_metadata = metadata["object_data"]
        sim_metadata.update(metadata_update)
        physprop_frame = get_physprop_as_image_blob(sim_metadata, first_seg_frame, physprop_type, physprops_range, fg_seg_ids=fg_seg_ids, bg_seg_id=bg_seg_id, props_of_interest=props_of_interest, blob_type=blob_type)
    else:
        raise ValueError(f"Invalid conditioning_type: {conditioning_type}")
    
    print(f"first_frame.shape: {first_frame.shape}")
    print(f"depth_frame.shape: {depth_frame.shape}")
    print(f"first_seg_frame.shape: {first_seg_frame.shape}")
    print(f"physprop_frame.shape: {physprop_frame.shape}")
    print(f"fg_seg_ids: {fg_seg_ids}")
    print(f"bg_seg_id: {bg_seg_id}")

    model_inputs = {
        "first_frame": first_frame,
        "physprop": physprop_frame,
        "neg_physprop": None, #neg_physprop_frame,
        "first_depth_frame": depth_frame,
    }
    
    return model_inputs

def rle_decode(rle_data, mask_shape):
    """
    Decode RLE data back to a binary mask.
    
    Args:
        rle_data: RLE encoded data (dict with 'counts' and 'size')
        mask_shape: Original shape of the mask
    
    Returns:
        numpy array: Decoded binary mask
    """
    # The rle_data should already be in the correct format from pycocotools
    # If it's just the counts, reconstruct the full RLE dict
    if isinstance(rle_data, dict) and 'counts' in rle_data:
        rle_dict = rle_data
    else:
        # If it's just the counts bytes, reconstruct the dict
        rle_dict = {
            'counts': rle_data,
            'size': [mask_shape[0], mask_shape[1]]  # height, width
        }
    
    # Decode using pycocotools
    decoded_mask = mask_utils.decode(rle_dict)
    
    return decoded_mask

def rle_to_mask(rle):
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order

def load_custom_seg_data(seg_pkl_path, needed_segmentation_ids, image_path, resolution="256"):
    # Load image
    img = mp.read_image(image_path)  # Returns (H, W, C) numpy array
    img = img[..., :3]
    
    # Resize the image to target size
    aspect_ratio = detect_aspect_ratio((img.shape[0], img.shape[1]))  # expects (W, H)
    video_size = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]

    # Load segmentation masks from pickle file
    with open(seg_pkl_path, 'rb') as f:
        seg_data = pickle.load(f)
    
    # Create composite segmentation mask
    seg_frame = np.zeros((video_size[0], video_size[1]), dtype=np.int32)
    
    # Process each needed segmentation ID
    for seg_id in needed_segmentation_ids:
        if seg_id < len(seg_data):
            mask_data = seg_data[seg_id]
            # Decode RLE mask
            try:
                rle_data = mask_data['segmentation_mask_rle']
                original_shape = rle_data['mask_shape']
                decoded_mask = rle_decode(rle_data['data'], original_shape)
            except Exception as e:
                mask_rle = mask_data['segmentation_mask_rle']
                print(f"Error decoding RLE mask for seg_id {seg_id}: {e}")
                decoded_mask = rle_to_mask(mask_rle['data'])
            
            # Resize mask to match image size
            resized_mask = cv2.resize(decoded_mask.astype(np.uint8), (video_size[1], video_size[0]), interpolation=cv2.INTER_NEAREST)
            
            # Add to composite segmentation frame with seg_id as the value
            seg_frame[resized_mask > 0] = seg_id + 1
    
    return seg_frame

def load_custom_data(image_path, seg_pkl_path, needed_segmentation_ids, resolution="256", physprop_type="all", depth_path=None, physprops_range=None, conditioning_type="image", fg_object_types=None, fg_seg_ids=None, bg_seg_id=None, force_magnitude=None, force_magnitude_max=None, force_magnitude_min=None, dir_angle=None, props_of_interest=None, blob_type=None, no_background_condition=False):
    """Load custom data (image + RLE segmentation) for inference"""
    # Load image
    img = mp.read_image(image_path)  # Returns (H, W, C) numpy array
    img = img[..., :3]
    print(f"img.shape: {img.shape}")

    # Resize the image to target size
    aspect_ratio = detect_aspect_ratio((img.shape[0], img.shape[1]))  # expects (W, H)
    video_size = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
    img_resized = cv2.resize(img, (video_size[1], video_size[0]))
    
    # Load segmentation frame
    seg_frame = load_custom_seg_data(seg_pkl_path, needed_segmentation_ids, image_path, resolution)

    # Generate dummy depth frame (all zeros) if not provided
    if depth_path is None:
        depth_frame = np.zeros((video_size[0], video_size[1], 3), dtype=np.uint8)
    else:
        # Load depth if provided (implement if needed)
        depth_frame = np.zeros((video_size[0], video_size[1], 3), dtype=np.uint8)
    
    needed_segmentation_ids = [seg_id + 1 for seg_id in needed_segmentation_ids]
    if fg_seg_ids is not None:
        fg_seg_ids = [seg_id + 1 for seg_id in fg_seg_ids]
    else:
        fg_seg_ids = [needed_segmentation_ids[1]]
    if bg_seg_id is not None:
        bg_seg_id = bg_seg_id + 1
    else:
        bg_seg_id = needed_segmentation_ids[0]
    move_object_seg_id = fg_seg_ids[-1]
    metadata = {
        "segmentation_id": needed_segmentation_ids,
        "mass": [1.0] * len(needed_segmentation_ids),
        "restitution": [0.0] * len(needed_segmentation_ids),
        "friction": [1.0] * len(needed_segmentation_ids),
        "neo_hookean_mu": None,
        "neo_hookean_lambda": None,
        "neo_hookean_damping": None,
        "force_start_img_coords": None,
        "force_end_img_coords": None,
        "force_magnitude": force_magnitude,
        "force_magnitude_max": force_magnitude_max,
        "force_magnitude_min": force_magnitude_min,
        "dir_angle": dir_angle,
        "move_object_seg_id": move_object_seg_id,
    }
    print(f"metadata: {metadata}")
    print(f"fg_object_types: {fg_object_types}")
    print(f"fg_seg_ids: {fg_seg_ids}")
    print(f"bg_seg_id: {bg_seg_id}")
    print(f"move_object_seg_id: {move_object_seg_id}")
    # Create dummy physical properties for custom data
    # For custom data without physics metadata, we'll use default values
    if conditioning_type=="image":
        physprop_frame = get_physprop_as_spatial_data(metadata, seg_frame, physprop_type, physprops_range)
    elif conditioning_type=="fg_bg_vector":
        physprop_frame = get_physprop_as_fg_bg_vector(metadata, physprop_type, physprops_range)
    elif conditioning_type=="image_blob":
        physprop_frame = get_physprop_as_image_blob(metadata, seg_frame, physprop_type, physprops_range, bg_seg_id=bg_seg_id, fg_object_types=fg_object_types, fg_seg_ids=fg_seg_ids, props_of_interest=props_of_interest, blob_type=blob_type, no_background_condition=no_background_condition)
    else:
        raise ValueError(f"Invalid conditioning_type: {conditioning_type}")
    
    # Debug visualization
    # plt.clf()
    # plt.imshow(seg_frame)
    # plt.colorbar()
    # plt.title("Custom Segmentation Frame")
    # plt.savefig("custom_seg_frame.png")
    # plt.close()
    
    
    # plt.clf()
    # plt.imshow(physprop_frame[:, :, 0], cmap="viridis", vmin=physprop_frame[:, :, 0].min(), vmax=physprop_frame[:, :, 0].max())
    # plt.colorbar()
    # plt.title("Custom Physprop Frame")
    # plt.savefig("custom_physprop_frame.png")
    # plt.close()
    
    print(f"img_resized.shape: {img_resized.shape}")
    print(f"depth_frame.shape: {depth_frame.shape}")
    print(f"seg_frame.shape: {seg_frame.shape}")
    print(f"physprop_frame.shape: {physprop_frame.shape}")

    model_inputs = {
        "first_frame": img_resized,
        "physprop": physprop_frame,
        "neg_physprop": None, #neg_physprop_frame,
        "first_depth_frame": depth_frame,
    }
    
    return model_inputs

def check_batch_requirements(batch_input):
    """Check batch input to determine if text encoder is needed"""
    needs_text_encoder = False
    for item in batch_input:
        if item.get("use_text_encoder", False):
            needs_text_encoder = True
            break
    return needs_text_encoder


def setup_pipeline(args: argparse.Namespace, needs_text_encoder: bool = None):
    """Setup pipeline with conditional text encoder loading based on batch requirements"""
    # If needs_text_encoder is not specified, fall back to args.use_text_encoder
    if needs_text_encoder is None:
        needs_text_encoder = args.use_text_encoder
    # If no precomputed negative-prompt embedding is supplied, the pipeline encodes
    # the (default) negative prompt on the fly, which requires the T5 text encoder.
    if (not args.neg_t5_embedding_path or not os.path.isfile(args.neg_t5_embedding_path)) and getattr(args, "negative_prompt", ""):
        if not needs_text_encoder:
            log.info(
                "No --neg_t5_embedding_path given; loading the T5 text encoder to "
                "encode the negative prompt online (pass --neg_t5_embedding_path to skip this)."
            )
        needs_text_encoder = True

    log.info(f"Using model size: {args.model_size}")
    log.info(f"Using pipeline config: {args.pipeline_config}")
    log.info(f"Using text encoder: {needs_text_encoder}")
    log.info(f"Using physprop_type: {args.physprop_type}")
    log.info(f"Using fps: {args.fps}")
    log.info(f"Using save_filename: {args.save_filename}")
    log.info(f"Using save_dir: {args.save_dir}")
    log.info(f"Using prepend_fname: {args.prepend_fname}")
    
    if args.model_size == "2B":
        MULTI = PHYS_PROP_CONTROLNET_PREDICT2_VIDEO2WORLD_PIPELINE_2B_MULTIPLE
        # state_t (latent temporal length) per requested frame count.
        _STATE_T = {
            "": 14,
            "controlnet_multi": 14,
            "controlnet_multi_24fps_33frames": 8,
            "controlnet_multi_24fps_57frames": 14,
            "controlnet_multi_24fps_73frames": 18,
            "controlnet_multi_24fps_120frames": 30,
        }
        if args.pipeline_config not in _STATE_T:
            raise ValueError(
                f"Unsupported pipeline_config '{args.pipeline_config}'. "
                f"Supported: {sorted(k for k in _STATE_T if k)}"
            )
        config = MULTI
        config.state_t = _STATE_T[args.pipeline_config]
        config.net.physprop_channels = 3
        if "deformable" in args.physprop_type:
            config.net.physprop_channels += 3
        if "force" in args.physprop_type:
            config.net.physprop_channels += 3
        print(config)
        dit_path = "checkpoints/cosmos_predict2/debug/phys_prop_conditioned_predict2_video2world_2b_training_2025-07-01_23-42-49/checkpoints/model/iter_000005000.pt"
    else:
        raise ValueError("Invalid model size. Choose either '2B' or '14B'.")
    if hasattr(args, "dit_path") and args.dit_path:
        dit_path = args.dit_path

    # Determine text encoder path based on batch requirements
    if needs_text_encoder:
        text_encoder_path = "./checkpoints/google-t5/t5-11b"
    else:
        text_encoder_path = ""

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize distributed environment for multi-GPU inference
    if hasattr(args, "num_gpus") and args.num_gpus > 1:
        log.info(f"Initializing distributed environment with {args.num_gpus} GPUs for context parallelism")
        distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.num_gpus)
        log.info(f"Context parallel group initialized with {args.num_gpus} GPUs")

    # Disable guardrail if requested
    if args.disable_guardrail:
        log.warning("Guardrail checks are disabled")
        config.guardrail_config.enabled = False

    # Disable prompt refiner if requested
    if args.disable_prompt_refiner:
        log.warning("Prompt refiner is disabled")
        config.prompt_refiner_config.enabled = False

    # Load models - for LoRA, we need to handle this differently
    log.info(f"Initializing Video2WorldPipeline with model size: {args.model_size}")
    if hasattr(config.net, "controlnet_branch_names") and args.controlnet_branch_names:
        config.net.controlnet_branch_names = args.controlnet_branch_names
    if hasattr(config.net, "active_controlnet_names") and args.active_controlnets is not None:
        config.net.active_controlnet_names = args.active_controlnets
    if hasattr(config.net, "controlnet_conditioning_scales") and args.controlnet_branch_scales:
        config.net.controlnet_conditioning_scales = args.controlnet_branch_scales
    if hasattr(config.net, "controlnet_branch_ckpt_paths") and args.controlnet_branch_ckpts:
        config.net.controlnet_branch_ckpt_paths = args.controlnet_branch_ckpts
    if (
        hasattr(config.net, "channels_per_controlnet")
        and args.controlnet_channels_per_controlnet is not None
    ):
        config.net.channels_per_controlnet = args.controlnet_channels_per_controlnet
    if hasattr(config.net, "controlnet_channel_groups") and args.controlnet_channel_groups:
        config.net.controlnet_channel_groups = args.controlnet_channel_groups

    if args.use_lora:
        # For LoRA inference, we need to add LoRA before loading the checkpoint
        log.info("LoRA inference mode detected - using custom pipeline loading")
        pipe = setup_lora_physprop_pipeline(config, dit_path, text_encoder_path, args)
    else:
        # Standard inference
        pipe = PhyspropConditionedVideo2WorldPipeline.from_config(
            config=config,
            dit_path=dit_path,
            text_encoder_path=text_encoder_path,
            device="cuda",
            torch_dtype=torch.bfloat16,
            load_prompt_refiner=not args.disable_prompt_refiner,
        )

    # Apply runtime ControlNet overrides if requested
    if (
        hasattr(pipe, "set_controlnet_mode")
        and (args.active_controlnets is not None or args.controlnet_branch_scales is not None)
    ):
        log.info(
            f"Setting ControlNet mode: active={args.active_controlnets}, scales={args.controlnet_branch_scales}"
        )
        pipe.set_controlnet_mode(
            active=args.active_controlnets,
            scales=args.controlnet_branch_scales,
        )

    if hasattr(pipe, "load_controlnet_branch_weights") and args.controlnet_branch_ckpts:
        log.info(f"Loading ControlNet branch checkpoints: {args.controlnet_branch_ckpts}")
        pipe.load_controlnet_branch_weights(args.controlnet_branch_ckpts, strict=False, map_location="cuda")

    return pipe


def read_first_frame(video_path):
    video = mp.read_video(video_path)  # Returns (T, H, W, C) numpy array
    return video[0]

def read_first_frame_image(image_path):
    image = mp.read_image(image_path)  # Returns (H, W, C) numpy array
    return image

def process_single_generation(
    pipe, args, item_config, physprops_range=None, seed=None, seed_prefix=""
):
    """Process single generation with per-item configuration overrides"""
    # Extract parameters from item config and args
    is_kubric_generated = item_config.get("is_kubric_generated", True)
    prompt = item_config.get("prompt", None)
    output_filename = item_config["output_video"]
    filename_prefix = ""
    if seed_prefix:
        filename_prefix += seed_prefix
    if args.prepend_fname and args.prepend_fname != "":
        filename_prefix += args.prepend_fname
    if filename_prefix:
        output_filename = filename_prefix + output_filename
    
    # Check if video already exists and should skip regeneration
    if args.no_regenerate_videos:
        output_dir = os.path.join(args.save_dir, _run_name_from_dit(args.dit_path)) if args.dit_path else args.save_dir
        if args.post_dit_subfolder:
            output_dir = os.path.join(output_dir, args.post_dit_subfolder)
        output_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_path):
            log.info(f"Video already exists, skipping regeneration: {output_path}")
            return True
    
    # Use item-specific T5 embeddings path or fall back to args
    t5_embeddings_path = item_config.get("t5_embeddings_path", None)
    
    force_magnitude = item_config.get("force_magnitude", None)
    dir_angle = item_config.get("dir_angle", None)
    if dir_angle is not None:
        dir_angle = np.radians(dir_angle)
    move_object_seg_id = item_config.get("move_object_seg_id", None)
    fg_seg_ids = item_config.get("fg_seg_ids", None)
    bg_seg_id = item_config.get("bg_seg_id", None)
    fg_seg_start_id = item_config.get("fg_seg_start_id", None)
    props_of_interest = item_config.get("props_of_interest", None)
        
    # Load data based on whether it's Kubric generated or custom data
    if is_kubric_generated:
        # Load data from Kubric folder
        data_folder = item_config["data_folder"]
        model_inputs = load_kubric_data(data_folder, physprop_type=args.physprop_type, physprops_range=physprops_range, conditioning_type=args.conditioning_type, force_magnitude=force_magnitude, dir_angle=dir_angle, move_object_seg_id=move_object_seg_id, fg_seg_ids=fg_seg_ids, bg_seg_id=bg_seg_id, fg_seg_start_id=fg_seg_start_id, props_of_interest=props_of_interest, blob_type=args.blob_type)
        log.info(f"Running Video2WorldPipeline (seed={seed if seed is not None else args.seed})\ndata_folder: {data_folder}")
    else:
        # Load custom data (image + RLE segmentation)
        input_video = item_config["input_video"]
        segmentation_pkl = item_config["segmentation_pkl"]
        needed_segmentation_ids = item_config["needed_segmentation_ids"]
        fg_object_types = item_config.get("fg_object_types", None)
        fg_seg_ids = item_config.get("fg_seg_ids", None)
        bg_seg_id = item_config.get("bg_seg_id", None)
        force_magnitude_max = item_config.get("force_magnitude_max", 1.0)
        force_magnitude_min = item_config.get("force_magnitude_min", 0.0)
        model_inputs = load_custom_data(
            input_video, 
            segmentation_pkl, 
            needed_segmentation_ids, 
            resolution=args.resolution, 
            physprop_type=args.physprop_type, 
            physprops_range=physprops_range,
            conditioning_type=args.conditioning_type,
            fg_object_types=fg_object_types,
            fg_seg_ids=fg_seg_ids,
            bg_seg_id=bg_seg_id,
            force_magnitude=force_magnitude,
            force_magnitude_max=force_magnitude_max,
            force_magnitude_min=force_magnitude_min,
            dir_angle=dir_angle,
            props_of_interest=props_of_interest,
            blob_type=args.blob_type,
            no_background_condition=args.no_background_condition
        )
        log.info(
            f"Running Video2WorldPipeline (seed={seed if seed is not None else args.seed})\n"
            f"input_video: {input_video}, segmentation_pkl: {segmentation_pkl}"
        )
    
    # Handle negative T5 embeddings (use global args)
    if args.neg_t5_embedding_path and os.path.isfile(args.neg_t5_embedding_path):
        neg_t5_text_embeddings = torch.load(args.neg_t5_embedding_path)
        neg_t5_text_embeddings = neg_t5_text_embeddings.contiguous()
        n_tokens = neg_t5_text_embeddings.shape[0]
        if n_tokens < 512:
            neg_t5_text_embeddings = torch.cat([neg_t5_text_embeddings, torch.zeros(512 - n_tokens, 1024)], dim=0)
        neg_t5_text_embeddings = neg_t5_text_embeddings.unsqueeze(0)
    else:
        neg_t5_text_embeddings = None
    
    # Handle T5 embeddings - use item-specific path if available
    if t5_embeddings_path is not None and t5_embeddings_path != "":
        log.info(f"Using item-specific T5 embeddings: {t5_embeddings_path}")
        t5_text_embeddings = torch.load(t5_embeddings_path)
        t5_text_embeddings = t5_text_embeddings.contiguous()
        n_tokens = t5_text_embeddings.shape[0]
        if n_tokens < 512:
            t5_text_embeddings = torch.cat([t5_text_embeddings, torch.zeros(512 - n_tokens, 1024)], dim=0)
        t5_text_embeddings = t5_text_embeddings.unsqueeze(0)
    else:
        t5_text_embeddings = None

    # If in controlnet mode and --dynamic_controlnet is set then I want to change the active controlnet based on props_of_interest if available
    if args.dynamic_controlnet and props_of_interest is not None:
        # Create a mapping from controlnet names to their default scales
        controlnet_scale_map = {}
        if args.controlnet_branch_names and args.controlnet_branch_scales:
            for name, scale in zip(args.controlnet_branch_names, args.controlnet_branch_scales):
                controlnet_scale_map[name] = scale
        
        # Build a dictionary of scales for dynamic controlnet - start with all at 0.0
        dynamic_scales = {name: 0.0 for name in args.controlnet_branch_names}
        
        # Enable specific controlnets based on props_of_interest
        # Check for friction or bounciness -> controlnet_1
        if ("friction" in props_of_interest or "bounciness" in props_of_interest) and "controlnet_1" in args.active_controlnets:
            dynamic_scales["controlnet_1"] = controlnet_scale_map.get("controlnet_1", 1.0)
        
        # Check for deformable -> controlnet_2
        if "deformable" in props_of_interest and "controlnet_2" in args.active_controlnets:
            dynamic_scales["controlnet_2"] = controlnet_scale_map.get("controlnet_2", 1.0)
        
        # Check for force -> controlnet_3
        if "force" in props_of_interest and "controlnet_3" in args.active_controlnets:
            dynamic_scales["controlnet_3"] = controlnet_scale_map.get("controlnet_3", 1.0)
        
        active_names = [name for name, scale in dynamic_scales.items() if scale > 0.0]
        log.info(f"Dynamic ControlNet: Setting scales={dynamic_scales} (active={active_names}) based on props_of_interest={props_of_interest}")
        pipe.set_controlnet_mode(active=args.active_controlnets, scales=dynamic_scales)
    elif props_of_interest is None and args.dynamic_controlnet:
        # Set all controlnets to inactive (scale 0.0) when no props_of_interest specified
        dynamic_scales = {name: 0.0 for name in args.controlnet_branch_names}
        log.info(f"Dynamic ControlNet: No props_of_interest, setting all controlnets to scale 0.0")
        pipe.set_controlnet_mode(active=args.active_controlnets, scales=dynamic_scales)

    current_seed = seed if seed is not None else args.seed
    video = pipe(
        **model_inputs,
        prompt=prompt,
        num_conditional_frames=1,
        guidance=args.guidance,
        seed=current_seed,
        negative_prompt=args.negative_prompt,
        neg_t5_text_embeddings=neg_t5_text_embeddings,
        t5_text_embeddings=t5_text_embeddings,
    )
    
    def pad_video_to_length(video_tensor, fps, total_seconds):
        if video_tensor is None or total_seconds is None or fps is None or fps <= 0:
            return video_tensor
        target_frames = int(math.ceil(total_seconds * fps))
        if target_frames <= 0:
            return video_tensor
        if video_tensor.ndim == 5:
            current_frames = video_tensor.shape[2]
            if current_frames >= target_frames:
                return video_tensor
            repeat = target_frames - current_frames
            last_frame = video_tensor[:, :, -1:, :, :]
            pad = last_frame.repeat(1, 1, repeat, 1, 1)
            return torch.cat([video_tensor, pad], dim=2)
        if video_tensor.ndim == 4:
            current_frames = video_tensor.shape[1]
            if current_frames >= target_frames:
                return video_tensor
            repeat = target_frames - current_frames
            last_frame = video_tensor[:, -1:, :, :]
            pad = last_frame.repeat(1, repeat, 1, 1)
            return torch.cat([video_tensor, pad], dim=1)
        return video_tensor
    
    video = pad_video_to_length(video, args.fps, args.total_seconds)
    
    if video is not None:
        # Determine output directory
        output_dir = os.path.join(args.save_dir, _run_name_from_dit(args.dit_path)) if args.dit_path else args.save_dir
        if args.post_dit_subfolder:
            output_dir = os.path.join(output_dir, args.post_dit_subfolder)
        # save the generated video
        if not output_filename:
            if is_kubric_generated:
                folder_name = os.path.basename(item_config["data_folder"].rstrip('/'))
                output_filename = f"{folder_name}_generated.mp4"
            else:
                image_name = os.path.splitext(os.path.basename(item_config["input_video"]))[0]
                output_filename = f"{image_name}_generated.mp4"
        
        output_path = os.path.join(output_dir, output_filename)
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Saving generated video to: {output_path}")
        save_image_or_video(video, output_path, fps=args.fps)
        log.success(f"Successfully saved video to: {output_path}")
        if args.conditioning_type == "image" or args.conditioning_type == "image_blob":
            save_physprop_as_image(
                model_inputs["physprop"], 
                save_path=os.path.join(output_dir, output_filename.replace(".mp4", "_physprop.png")), 
                physprop_type=args.physprop_type
            )
        elif args.conditioning_type == "fg_bg_vector":
            save_physprop_as_text(
                model_inputs["physprop"], 
                save_path=os.path.join(output_dir, output_filename.replace(".mp4", "_physprop.txt"))
            )
        # if "force" in args.physprop_type and force_magnitude is not None and dir_angle is not None:
        if dir_angle is not None:
            # Load segmentation frame
            seg_frame = load_custom_seg_data(segmentation_pkl, needed_segmentation_ids, input_video, args.resolution)
            if fg_seg_ids is not None:
                fg_seg_id = fg_seg_ids[-1] + 1
            else:
                fg_seg_id = needed_segmentation_ids[-1] + 1
            augment_input_image_with_move_dir(
                input_image=model_inputs["first_frame"],
                dir_angle=dir_angle,
                seg_frame=seg_frame,
                fg_seg_id=fg_seg_id,
                force_magnitude=force_magnitude,
                save_path=os.path.join(output_dir, output_filename.replace(".mp4", "_force.png"))
            )
        return True
    return False


def generate_video(args: argparse.Namespace, pipe: PhyspropConditionedVideo2WorldPipeline, batch_input: list[dict] = None) -> None:
    """Generate videos using batch input JSON as the primary mode"""
    if not args.batch_input_json and batch_input is None:
        log.error("batch_input_json is required. The script now operates primarily through batch JSON files.")
        log.info("Please provide a batch input JSON file using --batch_input_json")
        log.info("For single item processing, create a JSON file with one item.")
        return
    
    if batch_input is None:
        with open(args.batch_input_json, "r") as f:
            batch_input = json.load(f)
    
    log.info(f"Processing {len(batch_input)} items from batch input JSON")
    
    seeds_to_use = args.seeds if args.seeds else [args.seed]
    multi_seed_mode = bool(args.seeds)
    
    for seed_idx, seed in enumerate(seeds_to_use, start=1):
        log.info(f"=== Seed run {seed_idx}/{len(seeds_to_use)}: seed={seed} ===")
        misc.set_random_seed(seed=seed, by_rank=True)
        seed_prefix = f"seed{seed}_" if multi_seed_mode else ""
        
        for i, item in enumerate(batch_input):
            log.info(f"Processing item {i+1}/{len(batch_input)} for seed {seed}")
            
            # Extract physical properties range for this item
            physprops_range = {
                "friction": item.get("friction", "default"),
                "bounciness": item.get("bounciness", "default"),
                "mass": item.get("mass", "default"),
                "neo_hookean_mu": item.get("neo_hookean_mu", "default"),
                "neo_hookean_lambda": item.get("neo_hookean_lambda", "default"),
                "neo_hookean_damping": item.get("neo_hookean_damping", "default"),
            }
            
            # Process this item
            success = process_single_generation(
                pipe=pipe,
                args=args,
                item_config=item,
                physprops_range=physprops_range,
                seed=seed,
                seed_prefix=seed_prefix,
            )
            
            if success:
                log.success(
                    f"Successfully processed item {i+1} for seed {seed}: {item.get('output_video', 'unnamed')}"
                )
            else:
                log.error(f"Failed to process item {i+1} for seed {seed}: {item.get('output_video', 'unnamed')}")
    
    return


def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def parse_json_for_run_ids(batch_input, run_ids):
    """
    Parse the batch input JSON for the specified run ids and return the new batch input
    Run ids can be specified as a comma-separated list (e.g. 0,1,2,3) or range interval (e.g. 0-3) or a combination of both (e.g. 0,1,3-5)
    """
    run_ids_split = run_ids.split(",")
    individual_ids = [int(id.strip()) for id in run_ids_split if "-" not in id]
    range_ids = [range(int(id.split("-")[0].strip()), int(id.split("-")[1].strip()) + 1) for id in run_ids_split if "-" in id]
    all_run_ids = individual_ids + [id for sublist in range_ids for id in sublist]
    all_run_ids = sorted(list(set(all_run_ids)))
    new_batch_input = [item for i, item in enumerate(batch_input) if i in all_run_ids]
    log.info(f"Parsed batch input for the following run ids: {all_run_ids}")
    return new_batch_input


def slice_batch_inputs(batch_input: list[dict], num_splits: int, split_index: int) -> list[dict]:
    if num_splits <= 0:
        raise ValueError("--num_splits must be >= 1")
    if split_index < 0 or split_index >= num_splits:
        raise ValueError("--split_index must be in [0, num_splits)")
    if num_splits == 1 or not batch_input:
        return batch_input
    chunk = math.ceil(len(batch_input) / num_splits)
    start = chunk * split_index
    end = min(start + chunk, len(batch_input))
    sliced = batch_input[start:end]
    log.info(f"Sliced batch input for split {split_index + 1}/{num_splits}: {len(sliced)} items from indices [{start}, {end})")
    return sliced

if __name__ == "__main__":
    args = parse_args()
    print(f"args: {args}")
    if args.seeds:
        args.seed = args.seeds[0]
    
    try:
        # Check if batch input JSON is provided and determine text encoder requirements
        if args.batch_input_json:
            with open(args.batch_input_json, "r") as f:
                batch_input = json.load(f)
            if args.run_ids:
                batch_input = parse_json_for_run_ids(batch_input, args.run_ids)
            batch_input = slice_batch_inputs(batch_input, args.num_splits, args.split_index)
            needs_text_encoder = check_batch_requirements(batch_input)
            log.info(f"Batch processing mode: {len(batch_input)} items")
            log.info(f"Text encoder required: {needs_text_encoder}")
        else:
            # Fallback to args setting if no batch input (will show error in generate_video)
            needs_text_encoder = args.use_text_encoder
            batch_input = []
            log.warning("No batch input JSON provided. Script now requires batch input for operation.")
        
        # Setup pipeline with conditional text encoder loading
        pipe = setup_pipeline(args, needs_text_encoder=needs_text_encoder)
        
        # Generate videos
        generate_video(args, pipe, batch_input=batch_input)
        
    finally:
        # Make sure to clean up the distributed environment
        cleanup_distributed()
