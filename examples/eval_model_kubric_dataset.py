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
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set TOKENIZERS_PARALLELISM environment variable to avoid deadlocks with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from megatron.core import parallel_state

from cosmos_predict2.configs.physprop_conditioned.config_physprop_conditioned import *
from cosmos_predict2.data.kubric_data.kubric_dataset import KubricDataset_v2
from cosmos_predict2.pipelines.physprop_v2w import PhyspropConditionedVideo2WorldPipeline
from examples.setup_utils import setup_lora_physprop_pipeline
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.io import save_image_or_video

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. The video also seems to be not physically plausible, and the objects are not moving according to the physical laws of the world. The objects in the scene break the laws of physics, blend with the background and other objects in the scene. The objects break in non-physical ways, disappear from the scene. Overall, the video is of poor quality."


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
    parser = argparse.ArgumentParser(description="Evaluate Video2World Model on Kubric Dataset")
    parser.add_argument(
        "--model_size",
        choices=["2B"],
        default="2B",
        help="Size of the model to use for video-to-world generation",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        required=True,
        help="Path to the DiT model checkpoint",
    )
    parser.add_argument(
        "--dataset_json_path",
        type=str,
        required=True,
        help="Path to the Kubric dataset JSON file",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=37,
        help="Number of frames in the video",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="480",
        help="Resolution of the generated video",
    )
    parser.add_argument(
        "--conditioning_type",
        type=str,
        default="image_blob",
        help="Type of conditioning to use",
        choices=["image", "fg_bg_vector", "image_blob"],
    )
    parser.add_argument(
        "--desired_fps",
        type=int,
        default=12,
        help="Desired FPS of the video",
    )
    parser.add_argument(
        "--physprop_inputs",
        type=str,
        default="all",
        help="Type of physical properties to use",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default=None,
        help="Base path for the dataset (to replace paths in JSON)",
    )
    parser.add_argument(
        "--blob_type",
        type=str,
        default="circle",
        help="Type of blob to use",
        choices=["circle", "convex_hull", "ellipse"],
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=7,
        help="Guidance value for inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="output/eval_results/",
        help="Path to save the generated videos",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for context parallel inference",
    )
    parser.add_argument(
        "--disable_guardrail",
        action="store_true",
        help="Disable guardrail checks on prompts",
    )
    parser.add_argument(
        "--disable_prompt_refiner",
        action="store_true",
        help="Disable prompt refiner that enhances short prompts",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=_DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for conditioning",
    )
    parser.add_argument(
        "--pipeline_config",
        type=str,
        default="controlnet_24fps_73frames",
        help="Name of the config to use for the pipeline",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (keep at 1 for now)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (None for all)",
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

    parser.add_argument(
        "--regenerate_videos",
        action="store_true",
        help="Regenerate videos for the dataset",
    )

    parser.add_argument(
        "--append_fname",
        type=str,
        default="",
        help="Append filename to the output video",
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

    args = parser.parse_args()

    args.controlnet_branch_names = _parse_list_argument(args.controlnet_branch_names)
    args.active_controlnets = _parse_list_argument(args.active_controlnets)
    args.controlnet_branch_scales = _parse_list_argument(args.controlnet_branch_scales, cast=float)
    args.controlnet_branch_ckpts = _parse_mapping_argument(args.controlnet_branch_ckpts)
    args.controlnet_channel_groups = _parse_channel_groups_argument(args.controlnet_channel_groups)

    return args


def setup_pipeline(args: argparse.Namespace):
    """Setup the inference pipeline"""
    log.info(f"Using model size: {args.model_size}")
    log.info(f"Using pipeline config: {args.pipeline_config}")
    log.info(f"Using physprop_inputs: {args.physprop_inputs}")
    log.info(f"Using conditioning_type: {args.conditioning_type}")
    log.info(f"ControlNet branches: {args.controlnet_branch_names}")
    log.info(f"Active ControlNets: {args.active_controlnets}")

    if args.use_lora:
        log.info("LoRA inference mode enabled")

    # Select the appropriate config
    if args.model_size == "2B":
        if args.pipeline_config == "controlnet_12fps_37frames":
            config = PHYS_PROP_CONTROLNET_PREDICT2_VIDEO2WORLD_PIPELINE_2B
            config.state_t = 9
            config.net.physprop_channels = 3
            if "deformable" in args.physprop_inputs:
                config.net.physprop_channels += 3
            if "force" in args.physprop_inputs:
                config.net.physprop_channels += 3
        elif args.pipeline_config == "controlnet_24fps_73frames":
            config = PHYS_PROP_CONTROLNET_PREDICT2_VIDEO2WORLD_PIPELINE_2B
            config.state_t = 18
            config.net.physprop_channels = 3
            if "deformable" in args.physprop_inputs:
                config.net.physprop_channels += 3
            if "force" in args.physprop_inputs:
                config.net.physprop_channels += 3
        elif args.pipeline_config == "controlnet_24fps_49frames":
            config = PHYS_PROP_CONTROLNET_PREDICT2_VIDEO2WORLD_PIPELINE_2B
            config.state_t = 12
            config.net.physprop_channels = 3
            if "deformable" in args.physprop_inputs:
                config.net.physprop_channels += 3
            if "force" in args.physprop_inputs:
                config.net.physprop_channels += 3
        elif args.pipeline_config == "controlnet_multi_24fps_33frames":
            config = PHYS_PROP_CONTROLNET_PREDICT2_VIDEO2WORLD_PIPELINE_2B_MULTIPLE
            config.state_t = 8
            config.net.physprop_channels = 3
            if "deformable" in args.physprop_inputs:
                config.net.physprop_channels += 3
            if "force" in args.physprop_inputs:
                config.net.physprop_channels += 3
        elif args.pipeline_config == "controlnet_multi_24fps_57frames":
            config = PHYS_PROP_CONTROLNET_PREDICT2_VIDEO2WORLD_PIPELINE_2B_MULTIPLE
            config.state_t = 14
            config.net.physprop_channels = 3
            if "deformable" in args.physprop_inputs:
                config.net.physprop_channels += 3
            if "force" in args.physprop_inputs:
                config.net.physprop_channels += 3
        elif args.pipeline_config == "controlnet_multi_24fps_73frames":
            config = PHYS_PROP_CONTROLNET_PREDICT2_VIDEO2WORLD_PIPELINE_2B_MULTIPLE
            config.state_t = 18
            config.net.physprop_channels = 3
            if "deformable" in args.physprop_inputs:
                config.net.physprop_channels += 3
            if "force" in args.physprop_inputs:
                config.net.physprop_channels += 3
        else:
            raise ValueError(f"Unknown pipeline config: {args.pipeline_config}")
    else:
        raise ValueError("Invalid model size. Choose '2B'.")

    dit_path = args.dit_path
    text_encoder_path = ""  # No text encoder needed for this evaluation

    misc.set_random_seed(seed=args.seed, by_rank=True)
    # Initialize cuDNN
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # Floating-point precision settings
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

    # Load the pipeline
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


def evaluate_model(args: argparse.Namespace, pipe: PhyspropConditionedVideo2WorldPipeline) -> None:
    """Evaluate the model on the Kubric dataset"""
    # Create dataset
    log.info(f"Loading dataset from: {args.dataset_json_path}")
    dataset = KubricDataset_v2(
        dataset_json_path=args.dataset_json_path,
        num_frames=args.num_frames,
        resolution=args.resolution,
        conditioning_type=args.conditioning_type,
        desired_fps=args.desired_fps,
        prompt_path=None,
        is_train=False,  # Evaluation mode
        input_type="video",
        physprop_inputs=args.physprop_inputs,
        base_path=args.base_path,
        blob_type=args.blob_type,
    )

    log.info(f"Dataset contains {len(dataset)} samples")

    # Sort the scene list to ensure deterministic ordering (dataset uses set() which is unordered)
    dataset.scene_fname_gather = sorted(dataset.scene_fname_gather)
    log.info("Sorted scene list for deterministic evaluation order")

    # Limit the number of samples if specified
    if args.max_samples is not None:
        log.info(f"Limiting evaluation to {args.max_samples} samples")
        dataset.scene_fname_gather = dataset.scene_fname_gather[:args.max_samples]

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create output directory
    # output_dir = args.save_dir
    output_dir = os.path.join(args.save_dir, args.dit_path.split("/")[-4], f"eval_videos{args.append_fname}") if args.dit_path else args.save_dir
    os.makedirs(output_dir, exist_ok=True)
    log.info(f"Output directory: {output_dir}")

    # Run inference
    log.info("Starting evaluation...")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        try:
            # Extract data from batch
            video = batch["video"]
            physprop = batch["physprop"]
            first_depth_frame = batch["first_depth_frame"]
            t5_text_embeddings = batch["t5_text_embeddings"]
            t5_text_mask = batch["t5_text_mask"]
            neg_t5_text_embeddings = batch.get("neg_t5_text_embeddings", None)

            # Save the generated video
            scene_fname = dataset.scene_fname_gather[batch_idx]
            scene_name = os.path.basename(scene_fname.rstrip('/'))
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(scene_fname.rstrip('/'))))
            output_filename = f"{folder_name}_{scene_name}.mp4"
            log.info(f"Generating video for: {output_filename}, scene_name: {scene_name}, folder_name: {folder_name}")
            
            if os.path.exists(os.path.join(output_dir, output_filename)) and not args.regenerate_videos:
                log.info(f"Video already exists: {output_filename}")
                continue
            
            # Convert first frame to numpy
            first_frame = video[0, :, 0, :, :].permute(1, 2, 0).cpu().numpy()
            physprop_np = physprop[0].permute(1, 2, 0).cpu().numpy()
            first_depth_frame_np = first_depth_frame[0].permute(1, 2, 0).cpu().numpy()

            # Run inference
            generated_video = pipe(
                first_frame=first_frame,
                physprop=physprop_np,
                neg_physprop=None,
                first_depth_frame=first_depth_frame_np,
                prompt=None,
                num_conditional_frames=1,
                guidance=args.guidance,
                seed=args.seed,
                negative_prompt=args.negative_prompt,
                neg_t5_text_embeddings=neg_t5_text_embeddings,
                t5_text_embeddings=t5_text_embeddings,
            )

            if generated_video is not None:
                output_path = os.path.join(output_dir, output_filename)

                log.info(f"Saving generated video to: {output_path}")
                save_image_or_video(generated_video, output_path, fps=args.desired_fps)
                log.success(f"Successfully saved video: {output_filename}")
            else:
                log.error(f"Failed to generate video for batch {batch_idx}")

        except Exception as e:
            log.error(f"Error processing batch {batch_idx}: {e}")
            import traceback
            log.error(traceback.format_exc())
            continue

    log.success(f"Evaluation complete! Results saved to: {output_dir}")


def cleanup_distributed():
    """Clean up the distributed environment if initialized."""
    if parallel_state.is_initialized():
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    log.info(f"Arguments: {args}")

    if torch.cuda.is_available():
        # Get total and free memory for each GPU
        mem = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        log.info("==================================================")
        log.info(f"Memory for each GPU: {mem}")
        log.info("==================================================")
        device_id = int(torch.argmax(torch.tensor(mem)))  # GPU with most free memory
        torch.cuda.set_device(device_id)
        log.info("==================================================")
        log.info(f"Using cuda:{device_id}")
        log.info("==================================================")
        
    try:
        # Setup pipeline
        pipe = setup_pipeline(args)

        # Evaluate the model
        evaluate_model(args, pipe)

    finally:
        # Clean up distributed environment
        cleanup_distributed()
