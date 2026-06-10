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
Physical Properties Conditioned Video Diffusion Transformer (DiT) Model.

This module provides ``PhyspropControlNetDiTMultiple`` — the multi-branch ControlNet
architecture used by the released PhyCo checkpoints. The frozen base DiT
(``MinimalV1LVGDiT``) is conditioned on physical-property maps via one or more
independently activatable/loadable ControlNet branches (``ControlNetBranch``), whose
zero-initialised outputs are added to the main features at each layer.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.video2world_dit import MinimalV1LVGDiT
from cosmos_predict2.models.text2image_dit import Block
from imaginaire.utils import log


class ZeroConv2d(nn.Module):
    """Zero-initialized 2D convolution layer"""
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        # Zero-initialize the weights and bias
        nn.init.zeros_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ControlNetBranch(nn.Module):
    """Encapsulates a single ControlNet branch used for physprop conditioning."""

    def __init__(
        self,
        name: str,
        latent_channels: int,
        model_channels: int,
        num_controlnet_blocks: int,
        context_dim: int,
        num_heads: int,
        use_adaln_lora: bool,
        adaln_lora_dim: int,
        atten_backend: str,
        conditioning_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.name = name
        self.num_controlnet_blocks = num_controlnet_blocks

        self.physprop_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, model_channels, kernel_size=3, padding=1),
            nn.GELU(approximate="tanh"),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.GELU(approximate="tanh"),
        )

        self.input_projection = nn.Sequential(
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
            nn.GELU(approximate="tanh"),
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    x_dim=model_channels,
                    context_dim=context_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    use_adaln_lora=use_adaln_lora,
                    adaln_lora_dim=adaln_lora_dim,
                    backend=atten_backend,
                )
                for _ in range(num_controlnet_blocks)
            ]
        )

        self.output_projections = nn.ModuleList(
            [ZeroConv2d(model_channels, model_channels, kernel_size=1) for _ in range(num_controlnet_blocks)]
        )

        self.register_buffer(
            "conditioning_scale", torch.tensor(float(conditioning_scale), dtype=torch.float32), persistent=False
        )

    def initialize_from_base_blocks(self, base_blocks: Sequence[Block]) -> None:
        """Copy weights from the frozen base model blocks."""
        for idx in range(min(len(self.blocks), len(base_blocks))):
            try:
                self.blocks[idx].load_state_dict(base_blocks[idx].state_dict(), strict=False)
            except Exception as exc:  # pragma: no cover - logging utility
                log.warning(f"Could not initialize ControlNet branch {self.name} block {idx} from base model: {exc}")

    def set_conditioning_scale(self, scale: float) -> None:
        self.conditioning_scale.fill_(float(scale))

    def get_conditioning_scale(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return self.conditioning_scale.to(device=device, dtype=dtype)

    def set_requires_grad(self, requires_grad: bool) -> None:
        for param in self.parameters():
            param.requires_grad = requires_grad

    def encode(self, latents_2d: torch.Tensor) -> torch.Tensor:
        return self.physprop_encoder(latents_2d)

    def project_input(self, spatial: torch.Tensor) -> torch.Tensor:
        return self.input_projection(spatial)

    def forward_block(
        self,
        block_idx: int,
        state_B_T_H_W_D: torch.Tensor,
        t_embedding_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        **block_kwargs,
    ) -> torch.Tensor:
        return self.blocks[block_idx](state_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, **block_kwargs)

    def project_output(self, block_idx: int, state_B_T_H_W_D: torch.Tensor) -> torch.Tensor:
        B_ctrl, T_ctrl, H_ctrl, W_ctrl, D_ctrl = state_B_T_H_W_D.shape
        state_2d = state_B_T_H_W_D.permute(0, 1, 4, 2, 3).reshape(B_ctrl * T_ctrl, D_ctrl, H_ctrl, W_ctrl)
        controlnet_output_2d = self.output_projections[block_idx](state_2d)
        return controlnet_output_2d.reshape(B_ctrl, T_ctrl, D_ctrl, H_ctrl, W_ctrl).permute(0, 1, 3, 4, 2)


class PhyspropControlNetDiTMultiple(MinimalV1LVGDiT):
    """Multi-branch ControlNet variant for physprop conditioning."""

    def __init__(self, *args, **kwargs):
        assert 'physprop_tokenizer' in kwargs, 'physprop_tokenizer must be provided'
        assert 'physprop_channels' in kwargs, 'physprop_channels must be provided'

        physprop_tokenizer = kwargs.pop('physprop_tokenizer')
        physprop_channels = int(kwargs.pop('physprop_channels'))
        num_controlnet_blocks = kwargs.pop('num_controlnet_blocks', None)
        freeze_physprop_encoder = kwargs.pop('freeze_physprop_encoder', True)
        controlnet_channel_groups = kwargs.pop('controlnet_channel_groups', None)
        channels_per_controlnet = kwargs.pop('channels_per_controlnet', 3)
        controlnet_branch_names = kwargs.pop('controlnet_branch_names', None)
        default_conditioning_scale = kwargs.pop('controlnet_conditioning_scale', 1.0)
        controlnet_conditioning_scales = kwargs.pop('controlnet_conditioning_scales', None)
        active_controlnet_names = kwargs.pop('active_controlnet_names', None)
        trainable_controlnet_names = kwargs.pop('trainable_controlnet_names', None)
        controlnet_branch_ckpt_paths = kwargs.pop('controlnet_branch_ckpt_paths', None)

        super().__init__(*args, **kwargs)

        self.physprop_tokenizer = physprop_tokenizer
        self.physprop_channels = physprop_channels
        self.freeze_physprop_encoder = freeze_physprop_encoder
        self.controlnet_branch_ckpt_paths = controlnet_branch_ckpt_paths or {}

        if controlnet_channel_groups is None:
            controlnet_channel_groups = []
            cursor = 0
            while cursor < self.physprop_channels:
                group = list(range(cursor, min(cursor + channels_per_controlnet, self.physprop_channels)))
                controlnet_channel_groups.append(group)
                cursor += channels_per_controlnet
        else:
            controlnet_channel_groups = [list(group) for group in controlnet_channel_groups]

        if not controlnet_channel_groups:
            raise ValueError('At least one controlnet channel group must be defined')

        self.controlnet_channel_groups: List[List[int]] = controlnet_channel_groups
        self.num_controlnets = len(self.controlnet_channel_groups)

        if controlnet_branch_names is None:
            controlnet_branch_names = [f'controlnet_{idx + 1}' for idx in range(self.num_controlnets)]
        if len(controlnet_branch_names) != self.num_controlnets:
            raise ValueError('controlnet_branch_names must match the number of channel groups')

        self.controlnet_branch_names = list(controlnet_branch_names)
        self.controlnet_branch_name_to_idx = {name: idx for idx, name in enumerate(self.controlnet_branch_names)}

        if controlnet_conditioning_scales is None:
            controlnet_conditioning_scales = [default_conditioning_scale] * self.num_controlnets
        elif isinstance(controlnet_conditioning_scales, (float, int)):
            controlnet_conditioning_scales = [float(controlnet_conditioning_scales)] * self.num_controlnets
        else:
            if len(controlnet_conditioning_scales) != self.num_controlnets:
                raise ValueError('controlnet_conditioning_scales must align with controlnet branches')
            controlnet_conditioning_scales = [float(scale) for scale in controlnet_conditioning_scales]

        self.physprop_latent_channels = getattr(self.physprop_tokenizer, "latent_ch", 16)
        requested_num_blocks = num_controlnet_blocks or self.num_blocks
        self.num_controlnet_blocks = min(requested_num_blocks, self.num_blocks)

        if hasattr(self.blocks[0], "cross_attn") and hasattr(self.blocks[0].cross_attn, "context_dim"):
            context_dim = self.blocks[0].cross_attn.context_dim
        else:
            context_dim = 1024

        self.controlnet_branches = nn.ModuleList()
        for idx in range(self.num_controlnets):
            branch = ControlNetBranch(
                name=self.controlnet_branch_names[idx],
                latent_channels=self.physprop_latent_channels,
                model_channels=self.model_channels,
                num_controlnet_blocks=self.num_controlnet_blocks,
                context_dim=context_dim,
                num_heads=self.num_heads,
                use_adaln_lora=getattr(self, 'use_adaln_lora', False),
                adaln_lora_dim=getattr(self, 'adaln_lora_dim', 256),
                atten_backend=getattr(self, 'atten_backend', 'transformer_engine'),
                conditioning_scale=controlnet_conditioning_scales[idx],
            )
            branch.initialize_from_base_blocks(self.blocks[: self.num_controlnet_blocks])
            self.controlnet_branches.append(branch)

        self._trainable_controlnet_indices = set(range(self.num_controlnets))
        self._active_controlnet_indices = set(range(self.num_controlnets))

        log.info(f"Number of controlnet branches: {self.num_controlnets}")
        log.info(f"Controlnet branch names: {self.controlnet_branch_names}")
        log.info(f"Controlnet channel groups: {self.controlnet_channel_groups}")

        self.freeze_base_model()

        if trainable_controlnet_names is not None:
            self.set_trainable_controlnets(trainable_controlnet_names)
        if active_controlnet_names is not None:
            self.set_active_controlnets(active_controlnet_names)

    def requires_grad_(self, requires_grad: bool = True):
        """
        Respect global requires_grad_ calls while keeping the base DiT frozen.

        Video2WorldModel.freeze_parameters() calls requires_grad_(True) on the
        denoising model, which would otherwise unfreeze the entire backbone.
        Re-apply the ControlNet freezing policy whenever gradients are enabled.
        """
        super().requires_grad_(requires_grad)
        if requires_grad:
            self.freeze_base_model()
        return self

    def _resolve_controlnet_indices(
        self, identifiers: Union[None, str, int, Sequence[Union[str, int]]]
    ) -> set[int]:
        if identifiers is None:
            return set(range(self.num_controlnets))
        if isinstance(identifiers, str):
            if identifiers.lower() == 'all':
                return set(range(self.num_controlnets))
            identifiers = [identifiers]
        elif isinstance(identifiers, Iterable):
            identifiers = list(identifiers)
        else:
            identifiers = [identifiers]

        resolved: set[int] = set()
        for item in identifiers:
            if isinstance(item, int):
                if item < 0 or item >= self.num_controlnets:
                    raise ValueError(f'Invalid controlnet index {item}')
                resolved.add(item)
            elif isinstance(item, str):
                if item not in self.controlnet_branch_name_to_idx:
                    raise KeyError(f'Unknown controlnet branch name {item}')
                resolved.add(self.controlnet_branch_name_to_idx[item])
            else:
                raise TypeError(f'Unsupported controlnet identifier type: {type(item)}')
        return resolved

    def _resolve_single_controlnet_index(self, identifier: Union[str, int]) -> int:
        indices = self._resolve_controlnet_indices(identifier)
        if len(indices) != 1:
            raise ValueError('Identifier must resolve to exactly one controlnet branch')
        return next(iter(indices))

    def _set_controlnet_branch_requires_grad(self, indices: Iterable[int]) -> None:
        active = set(indices)
        for idx, branch in enumerate(self.controlnet_branches):
            branch.set_requires_grad(idx in active)

    def _log_trainable_controlnets(self) -> None:
        total_params = sum(param.numel() for param in self.parameters())
        trainable_params = sum(param.numel() for param in self.parameters() if param.requires_grad)
        frozen_params = total_params - trainable_params
        trainable_names = [self.controlnet_branch_names[idx] for idx in sorted(self._trainable_controlnet_indices)]
        active_names = [self.controlnet_branch_names[idx] for idx in sorted(self._active_controlnet_indices)]
        log.info('ControlNet freezing applied:')
        log.info(f"  Total parameters: {total_params:,}")
        log.info(f"  Frozen parameters: {frozen_params:,}")
        log.info(f"  Trainable parameters: {trainable_params:,}")
        log.info(f"  Trainable ratio: {trainable_params / max(total_params, 1):.1%}")
        log.info(f"  Trainable ControlNets: {trainable_names}")
        log.info(f"  Active ControlNets: {active_names}")

    def freeze_base_model(self) -> None:
        for param in self.parameters():
            param.requires_grad = False
        self._set_controlnet_branch_requires_grad(self._trainable_controlnet_indices)

        if self.freeze_physprop_encoder and hasattr(self.physprop_tokenizer, "model"):
            try:
                for param in self.physprop_tokenizer.model.model.parameters():
                    param.requires_grad = False
            except AttributeError:
                log.warning('Could not freeze physprop tokenizer parameters; unexpected structure')

        self._log_trainable_controlnets()

    def unfreeze_base_model(self) -> None:
        for param in self.parameters():
            param.requires_grad = True
        if self.freeze_physprop_encoder and hasattr(self.physprop_tokenizer, "model"):
            try:
                for param in self.physprop_tokenizer.model.model.parameters():
                    param.requires_grad = False
            except AttributeError:
                log.warning('Could not access physprop tokenizer parameters during unfreeze')
        log.info('Unfrozen all model parameters.')

    def validate_controlnet_freezing(self) -> bool:
        base_trainable = [name for name, param in self.named_parameters() if 'controlnet_branches' not in name and param.requires_grad]
        issues = False
        if base_trainable:
            log.warning('Base model parameters unexpectedly trainable: ' + ', '.join(base_trainable[:5]) + ('...' if len(base_trainable) > 5 else ''))
            issues = True

        for idx in range(self.num_controlnets):
            prefix = f'controlnet_branches.{idx}.'
            params = [param.requires_grad for name, param in self.named_parameters() if name.startswith(prefix)]
            if not params:
                continue
            if idx in self._trainable_controlnet_indices:
                if not all(params):
                    log.warning(f'ControlNet branch {self.controlnet_branch_names[idx]} should be trainable but has frozen params')
                    issues = True
            else:
                if any(params):
                    log.warning(f'ControlNet branch {self.controlnet_branch_names[idx]} should be frozen but has trainable params')
                    issues = True

        if not issues:
            log.info('✓ ControlNet freezing validation passed')
            return True
        log.error('✗ ControlNet freezing validation failed')
        return False

    def _ensure_device_consistency(self, physprop: torch.Tensor) -> None:
        physprop_device = physprop.device
        if hasattr(self.physprop_tokenizer, "model"):
            try:
                tokenizer_param = next(self.physprop_tokenizer.model.model.parameters())
            except StopIteration:
                tokenizer_param = None
            if tokenizer_param is not None and tokenizer_param.device != physprop_device:
                self.physprop_tokenizer.model.model.to(physprop_device)

    def process_physprop_features(self, physprop: torch.Tensor) -> List[Optional[torch.Tensor]]:
        self._ensure_device_consistency(physprop)

        if physprop.dim() == 4:
            physprop = physprop.unsqueeze(2)

        physprop_image = torch.clamp(physprop * 255, 0, 255).to(torch.uint8)
        physprop_image = physprop_image.to(torch.bfloat16) / 127.5 - 1.0
        _, total_channels, _, _, _ = physprop_image.shape

        branch_features: List[Optional[torch.Tensor]] = [None] * self.num_controlnets
        for idx, channel_indices in enumerate(self.controlnet_channel_groups):
            if idx not in self._active_controlnet_indices:
                continue
            if not channel_indices:
                continue
            if max(channel_indices) >= total_channels:
                raise ValueError(f'ControlNet branch {self.controlnet_branch_names[idx]} expects physprop channel {max(channel_indices)} but input has only {total_channels} channels')

            physprop_slice = physprop_image[:, channel_indices, ...]
            with torch.set_grad_enabled(not self.freeze_physprop_encoder and idx in self._trainable_controlnet_indices):
                latents = self.physprop_tokenizer.encode(physprop_slice)

            if latents.shape[2] == 1:
                latents_2d = latents.squeeze(2)
            else:
                latents_2d = latents[:, :, 0, ...]

            branch_features[idx] = self.controlnet_branches[idx].encode(latents_2d)

        return branch_features

    def set_active_controlnets(self, identifiers: Union[None, str, int, Sequence[Union[str, int]]]) -> set[int]:
        self._active_controlnet_indices = self._resolve_controlnet_indices(identifiers)
        return self._active_controlnet_indices

    def set_trainable_controlnets(self, identifiers: Union[None, str, int, Sequence[Union[str, int]]]) -> set[int]:
        self._trainable_controlnet_indices = self._resolve_controlnet_indices(identifiers)
        self._set_controlnet_branch_requires_grad(self._trainable_controlnet_indices)
        self._log_trainable_controlnets()
        return self._trainable_controlnet_indices

    def set_controlnet_conditioning_scales(
        self, scales: Union[float, Sequence[float], Dict[Union[str, int], float]]
    ) -> None:
        if isinstance(scales, dict):
            for identifier, scale in scales.items():
                idx = self._resolve_single_controlnet_index(identifier)
                self.controlnet_branches[idx].set_conditioning_scale(scale)
            return

        if isinstance(scales, (float, int)):
            scales = [float(scales)] * self.num_controlnets
        elif len(scales) != self.num_controlnets:
            raise ValueError('Scale list must match the number of controlnet branches')

        for idx, scale in enumerate(scales):
            self.controlnet_branches[idx].set_conditioning_scale(scale)

    @staticmethod
    def _strip_known_prefixes(key: str) -> str:
        prefixes = ['state_dict.', 'model.', 'module.', 'net.']
        updated = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if updated.startswith(prefix):
                    updated = updated[len(prefix):]
                    changed = True
        return updated

    def _prepare_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        candidate = state_dict
        if 'state_dict' in candidate and isinstance(candidate['state_dict'], dict):
            candidate = candidate['state_dict']
        if 'net' in candidate and isinstance(candidate['net'], dict):
            candidate = candidate['net']
        cleaned: Dict[str, torch.Tensor] = {}
        for key, value in candidate.items():
            cleaned[self._strip_known_prefixes(key)] = value
        return cleaned

    def load_controlnet_branch_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        branch_identifier: Union[str, int],
        strict: bool = True,
    ) -> Tuple[List[str], List[str]]:
        cleaned = self._prepare_state_dict(state_dict)
        branch_idx = self._resolve_single_controlnet_index(branch_identifier)
        prefix = f'controlnet_branches.{branch_idx}.'
        branch = self.controlnet_branches[branch_idx]
        branch_state = {key[len(prefix):]: value for key, value in cleaned.items() if key.startswith(prefix)}

        if not branch_state:
            legacy_map = {'physprop_encoder.': 'physprop_encoder.', 'input_projection.': 'input_projection.', 'blocks.': 'blocks.', 'output_projections.': 'output_projections.'}
            for src_prefix, dst_prefix in legacy_map.items():
                for key, value in cleaned.items():
                    if key.startswith(src_prefix):
                        branch_state[f"{dst_prefix}{key[len(src_prefix):]}"] = value

        missing, unexpected = branch.load_state_dict(branch_state, strict=strict)
        if missing:
            log.warning(f'Missing keys when loading ControlNet branch {branch.name}: {missing}')
        if unexpected:
            log.warning(f'Unexpected keys when loading ControlNet branch {branch.name}: {unexpected}')
        return missing, unexpected

    def load_controlnet_branch_checkpoint(
        self,
        branch_identifier: Union[str, int],
        checkpoint_path: str,
        strict: bool = True,
        map_location: Union[str, torch.device] = 'cpu',
    ) -> Tuple[List[str], List[str]]:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        return self.load_controlnet_branch_state_dict(checkpoint, branch_identifier, strict=strict)

    def load_controlnet_branch_checkpoints(
        self, branch_ckpt_paths: Dict[Union[str, int], str], strict: bool = True, map_location: Union[str, torch.device] = "cpu"
    ) -> Dict[Union[str, int], Tuple[List[str], List[str]]]:
        results: Dict[Union[str, int], Tuple[List[str], List[str]]] = {}
        for identifier, path in branch_ckpt_paths.items():
            results[identifier] = self.load_controlnet_branch_checkpoint(identifier, path, strict=strict, map_location=map_location)
        return results

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        use_cuda_graphs: bool = False,
        physprop: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        del kwargs

        assert physprop is not None, 'physprop must be provided for ControlNet'

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1
            )

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        B, T, H, W, D = x_B_T_H_W_D.shape

        branch_spatials = self.process_physprop_features(physprop)
        controlnet_states: Dict[int, torch.Tensor] = {}
        for idx, spatial in enumerate(branch_spatials):
            if spatial is None:
                continue
            if spatial.shape[-2:] != (H, W):
                spatial = torch.nn.functional.interpolate(spatial, size=(H, W), mode="bilinear", align_corners=False)
            controlnet_input = self.controlnet_branches[idx].project_input(spatial)
            controlnet_state = controlnet_input.view(B, D, 1, H, W).expand(B, D, T, H, W)
            controlnet_states[idx] = controlnet_state.permute(0, 2, 3, 4, 1).contiguous()

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert (
                x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape
            ), f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"

        if use_cuda_graphs:
            use_cuda_graphs = False

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
        }

        for block_idx, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                **block_kwargs,
            )

            for branch_idx, state in list(controlnet_states.items()):
                branch = self.controlnet_branches[branch_idx]
                if block_idx >= branch.num_controlnet_blocks:
                    continue
                state = branch.forward_block(block_idx, state, t_embedding_B_T_D, crossattn_emb, **block_kwargs)
                controlnet_states[branch_idx] = state
                controlnet_output = branch.project_output(block_idx, state)
                scale = branch.get_conditioning_scale(x_B_T_H_W_D.dtype, x_B_T_H_W_D.device)
                if torch.count_nonzero(scale) == 0:
                    continue
                controlnet_output = controlnet_output.to(dtype=x_B_T_H_W_D.dtype)
                x_B_T_H_W_D = x_B_T_H_W_D + controlnet_output * scale.view(1, 1, 1, 1, 1)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)

        return x_B_C_Tt_Hp_Wp


