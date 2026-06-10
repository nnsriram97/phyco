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

"""
Shared utility functions for pipeline setup, including LoRA support.
"""

import torch
from imaginaire.utils import log


def add_lora_to_model(
    model,
    lora_rank=16,
    lora_alpha=16,
    lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
    init_lora_weights=True,
):
    """
    Add LoRA to a model using PEFT library.
    
    Args:
        model: The model to add LoRA to
        lora_rank: Rank of the LoRA adaptation
        lora_alpha: Alpha parameter for LoRA
        lora_target_modules: Comma-separated list of target modules
        init_lora_weights: Whether to initialize LoRA weights
    
    Returns:
        Model with LoRA adapters injected
    """
    from peft import LoraConfig, inject_adapter_in_model

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights=init_lora_weights,
        target_modules=lora_target_modules.split(","),
    )
    model = inject_adapter_in_model(lora_config, model)
    
    # Upcast LoRA parameters to fp32 for better stability
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.to(torch.float32)
    
    return model


def setup_lora_physprop_pipeline(config, dit_path, text_encoder_path, args):
    """
    Set up a PhyspropConditionedVideo2WorldPipeline with LoRA support and optional ControlNet branches.
    
    This function creates the pipeline, adds LoRA to the DiT, loads the main checkpoint,
    and optionally loads separate ControlNet branch checkpoints. This allows combining
    LoRA fine-tuning on the DiT with independently trained ControlNet branches.
    
    Args:
        config: Pipeline configuration
        dit_path: Path to DiT checkpoint (with LoRA weights)
        text_encoder_path: Path to text encoder or empty string
        args: Command-line arguments with LoRA and ControlNet settings
              Expected LoRA args: lora_rank, lora_alpha, lora_target_modules, init_lora_weights
              Expected ControlNet args (optional): controlnet_branch_names, active_controlnets,
                                                   controlnet_branch_scales, controlnet_branch_ckpts,
                                                   controlnet_channels_per_controlnet, controlnet_channel_groups
    
    Returns:
        Pipeline with LoRA adapters on DiT and loaded ControlNet branch checkpoints
    """
    import numpy as np
    from megatron.core import parallel_state

    from cosmos_predict2.pipelines.physprop_v2w import PhyspropConditionedVideo2WorldPipeline
    from cosmos_predict2.auxiliary.text_encoder import CosmosT5TextEncoder
    from cosmos_predict2.models.utils import load_state_dict
    from cosmos_predict2.module.denoiser_scaling import RectifiedFlowScaling
    from cosmos_predict2.schedulers.rectified_flow_scheduler import RectifiedFlowAB2Scheduler
    from imaginaire.lazy_config import instantiate
    from imaginaire.utils.ema import FastEmaModelUpdater

    # Create pipeline
    pipe = PhyspropConditionedVideo2WorldPipeline(device="cuda", torch_dtype=torch.bfloat16)
    pipe.config = config
    pipe.precision = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[config.precision]
    pipe.tensor_kwargs = {"device": "cuda", "dtype": pipe.precision}
    log.warning(f"precision {pipe.precision}")
    
    # 1. Set data keys and data information
    pipe.sigma_data = config.sigma_data
    pipe.setup_data_key()
    
    # 2. Setup diffusion processing and scaling (pre-condition)
    pipe.scheduler = RectifiedFlowAB2Scheduler(
        sigma_min=config.timestamps.t_min,
        sigma_max=config.timestamps.t_max,
        order=config.timestamps.order,
        t_scaling_factor=config.rectified_flow_t_scaling_factor,
    )

    pipe.scaling = RectifiedFlowScaling(pipe.sigma_data, config.rectified_flow_t_scaling_factor)
    
    # 3. Set up tokenizer
    pipe.tokenizer = instantiate(config.tokenizer)
    assert pipe.tokenizer.latent_ch == pipe.config.state_ch, (
        f"latent_ch {pipe.tokenizer.latent_ch} != state_shape {pipe.config.state_ch}"
    )
    
    # 4. Load text encoder
    if text_encoder_path:
        log.info(f"Loading text encoder from {text_encoder_path}")
        pipe.text_encoder = CosmosT5TextEncoder(
            device="cuda", 
            cache_dir=text_encoder_path, 
            local_files_only=True, 
            use_8bit=True
        )
        pipe.text_encoder.to(device="cuda", dtype=torch.bfloat16)
        log.info(f"Text encoder loaded successfully")
    else:
        pipe.text_encoder = None
    
    # 5. Initialize conditioner
    pipe.conditioner = instantiate(config.conditioner)
    assert sum(p.numel() for p in pipe.conditioner.parameters() if p.requires_grad) == 0, (
        "conditioner should not have learnable parameters"
    )
    
    # 6. Set up guardrail (if enabled)
    if config.guardrail_config.enabled:
        from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

        pipe.text_guardrail_runner = guardrail_presets.create_text_guardrail_runner(
            config.guardrail_config.checkpoint_dir, 
            config.guardrail_config.offload_model_to_cpu
        )
        pipe.video_guardrail_runner = guardrail_presets.create_video_guardrail_runner(
            config.guardrail_config.checkpoint_dir, 
            config.guardrail_config.offload_model_to_cpu
        )
    else:
        pipe.text_guardrail_runner = None
        pipe.video_guardrail_runner = None
    
    # 7. Set up DiT WITHOUT loading checkpoint first
    log.info("Initializing DiT model...")
    dit_config = config.net
    # Set the physprop_tokenizer to the main tokenizer if not already set
    if hasattr(dit_config, 'physprop_tokenizer') and dit_config.physprop_tokenizer is None:
        dit_config.physprop_tokenizer = pipe.tokenizer
    
    # Configure ControlNet settings from args if provided
    if hasattr(dit_config, "controlnet_branch_names") and hasattr(args, 'controlnet_branch_names') and args.controlnet_branch_names:
        dit_config.controlnet_branch_names = args.controlnet_branch_names
        log.info(f"Setting controlnet_branch_names: {args.controlnet_branch_names}")
    if hasattr(dit_config, "active_controlnet_names") and hasattr(args, 'active_controlnets') and args.active_controlnets is not None:
        dit_config.active_controlnet_names = args.active_controlnets
        log.info(f"Setting active_controlnet_names: {args.active_controlnets}")
    if hasattr(dit_config, "controlnet_conditioning_scales") and hasattr(args, 'controlnet_branch_scales') and args.controlnet_branch_scales:
        dit_config.controlnet_conditioning_scales = args.controlnet_branch_scales
        log.info(f"Setting controlnet_conditioning_scales: {args.controlnet_branch_scales}")
    if hasattr(dit_config, "controlnet_branch_ckpt_paths") and hasattr(args, 'controlnet_branch_ckpts') and args.controlnet_branch_ckpts:
        dit_config.controlnet_branch_ckpt_paths = args.controlnet_branch_ckpts
        log.info(f"Setting controlnet_branch_ckpt_paths: {args.controlnet_branch_ckpts}")
    if (
        hasattr(dit_config, "channels_per_controlnet")
        and hasattr(args, 'controlnet_channels_per_controlnet')
        and args.controlnet_channels_per_controlnet is not None
    ):
        dit_config.channels_per_controlnet = args.controlnet_channels_per_controlnet
        log.info(f"Setting channels_per_controlnet: {args.controlnet_channels_per_controlnet}")
    if hasattr(dit_config, "controlnet_channel_groups") and hasattr(args, 'controlnet_channel_groups') and args.controlnet_channel_groups:
        dit_config.controlnet_channel_groups = args.controlnet_channel_groups
        log.info(f"Setting controlnet_channel_groups: {args.controlnet_channel_groups}")
    
    pipe.dit = instantiate(dit_config).eval()
    
    # 8. Add LoRA to the DiT model BEFORE loading checkpoint
    log.info("Adding LoRA to the DiT model...")
    log.info(
        f"LoRA parameters: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"target_modules={args.lora_target_modules}"
    )
    pipe.dit = add_lora_to_model(
        pipe.dit,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
    )
    
    # 9. Handle EMA model if enabled
    if config.ema.enabled:
        log.info("Setting up EMA model...")
        # Set the physprop_tokenizer for EMA model as well
        if hasattr(dit_config, 'physprop_tokenizer') and dit_config.physprop_tokenizer is None:
            dit_config.physprop_tokenizer = pipe.tokenizer
        pipe.dit_ema = instantiate(dit_config).eval()
        pipe.dit_ema.requires_grad_(False)
        
        # Add LoRA to EMA model
        log.info("Adding LoRA to the EMA DiT model...")
        pipe.dit_ema = add_lora_to_model(
            pipe.dit_ema,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules=args.lora_target_modules,
            init_lora_weights=args.init_lora_weights,
        )
        pipe.dit_ema_worker = FastEmaModelUpdater()
        s = config.ema.rate
        pipe.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
        
        # Copy weights from regular model to EMA
        pipe.dit_ema_worker.copy_to(src_model=pipe.dit, tgt_model=pipe.dit_ema)
    
    # 10. NOW load the LoRA checkpoint with strict=False
    if dit_path:
        log.info(f"Loading LoRA checkpoint from {dit_path}")
        state_dict = load_state_dict(dit_path)
        
        # Split state dict for regular and EMA models
        state_dict_dit_regular = dict()
        state_dict_dit_ema = dict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                state_dict_dit_regular[k[4:]] = v
            elif k.startswith("net_ema."):
                state_dict_dit_ema[k[8:]] = v
        
        # Load regular model with strict=False to allow LoRA weights
        log.info("Loading regular DiT model weights...")
        missing_keys = pipe.dit.load_state_dict(state_dict_dit_regular, strict=False, assign=True)
        if missing_keys.missing_keys:
            log.warning(f"Missing keys in regular model: {missing_keys.missing_keys[:10]}...")
        if missing_keys.unexpected_keys:
            log.warning(f"Unexpected keys in regular model: {missing_keys.unexpected_keys[:10]}...")
        
        # Load EMA model if enabled
        if config.ema.enabled and state_dict_dit_ema:
            log.info("Loading EMA DiT model weights...")
            missing_keys_ema = pipe.dit_ema.load_state_dict(
                state_dict_dit_ema, strict=False, assign=True
            )
            if missing_keys_ema.missing_keys:
                log.warning(f"Missing keys in EMA model: {missing_keys_ema.missing_keys[:10]}...")
            if missing_keys_ema.unexpected_keys:
                log.warning(f"Unexpected keys in EMA model: {missing_keys_ema.unexpected_keys[:10]}...")
        
        del state_dict, state_dict_dit_regular, state_dict_dit_ema
        log.success(f"Successfully loaded LoRA checkpoint from {dit_path}")
    else:
        log.warning("No checkpoint path provided, using random weights")
    
    # 11. Move models to device BEFORE loading controlnet branches
    pipe.dit = pipe.dit.to(device="cuda", dtype=torch.bfloat16)
    if config.ema.enabled:
        pipe.dit_ema = pipe.dit_ema.to(device="cuda", dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    
    # 11b. Load ControlNet branch checkpoints if provided (after moving to device)
    if hasattr(args, 'controlnet_branch_ckpts') and args.controlnet_branch_ckpts:
        log.info(f"Loading ControlNet branch checkpoints: {args.controlnet_branch_ckpts}")
        if hasattr(pipe.dit, 'load_controlnet_branch_checkpoints'):
            load_results = pipe.dit.load_controlnet_branch_checkpoints(
                args.controlnet_branch_ckpts, 
                strict=False, 
                map_location="cuda"
            )
            for branch_name, (missing_keys, unexpected_keys) in load_results.items():
                log.info(f"Loaded ControlNet branch '{branch_name}':")
                if missing_keys:
                    log.warning(f"  Missing keys: {missing_keys[:10]}...")
                if unexpected_keys:
                    log.warning(f"  Unexpected keys: {unexpected_keys[:10]}...")
            log.success("Successfully loaded all ControlNet branch checkpoints")
        else:
            log.warning("Model does not support ControlNet branch checkpoint loading")
    
    # 11c. Handle controlnet_branch_ckpt_paths from config if present
    branch_ckpts = getattr(pipe.dit, "controlnet_branch_ckpt_paths", None)
    if branch_ckpts:
        if hasattr(pipe.dit, "load_controlnet_branch_checkpoints"):
            log.info("Loading ControlNet branch checkpoints provided in configuration")
            load_results = pipe.dit.load_controlnet_branch_checkpoints(
                branch_ckpts,
                strict=False,
                map_location="cuda"
            )
            for branch_name, (missing_keys, unexpected_keys) in load_results.items():
                log.info(f"Loaded ControlNet branch '{branch_name}' from config")
                if missing_keys:
                    log.warning(f"  Missing keys: {missing_keys[:5]}...")
                if unexpected_keys:
                    log.warning(f"  Unexpected keys: {unexpected_keys[:5]}...")
        else:
            log.warning("controlnet_branch_ckpt_paths provided but model does not support branch checkpoint loading")
    if hasattr(pipe.dit, "controlnet_branch_ckpt_paths"):
        pipe.dit.controlnet_branch_ckpt_paths = {}
    
    # 12. Set up training states
    if parallel_state.is_initialized():
        pipe.data_parallel_size = parallel_state.get_data_parallel_world_size()
    else:
        pipe.data_parallel_size = 1
    
    # Print parameter counts
    total_params = sum(p.numel() for p in pipe.dit.parameters())
    trainable_params = sum(p.numel() for p in pipe.dit.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable LoRA parameters: {trainable_params:,}")
    log.info(f"LoRA parameter ratio: {trainable_params / total_params * 100:.2f}%")
    
    return pipe

