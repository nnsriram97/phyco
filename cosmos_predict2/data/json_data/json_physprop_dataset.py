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

from __future__ import annotations

import json
import math
import os
import pickle
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import mediapy as mp
import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_predict2.data.dataset_utils import VIDEO_RES_SIZE_INFO, detect_aspect_ratio
from cosmos_predict2.data.kubric_data.kubric_utils import (
    get_physprop_as_fg_bg_vector,
    get_physprop_as_image_blob,
    get_physprop_as_spatial_data,
)
from imaginaire.utils import log
from pycocotools import mask as mask_utils

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
_LABEL_TO_SCORE = {
    "low": 0.15,
    "medium": 0.5,
    "mid": 0.5,
    "neutral": 0.5,
    "high": 0.85,
    "yes": 0.85,
    "no": 0.15,
    "default": 0.5,
}


def _resolve_path(path: str, base_path: Optional[str]) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded) or expanded.startswith("s3://"):
        return expanded
    if base_path is None:
        return expanded
    return os.path.join(base_path, expanded)


def _range_to_score(value: Any, default: float = 0.5) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(np.clip(value, 0.0, 1.0))
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _LABEL_TO_SCORE:
            return _LABEL_TO_SCORE[lowered]
        try:
            return float(np.clip(float(lowered), 0.0, 1.0))
        except ValueError:
            return default
    return default


def _normalize_force(value: Any, min_val: float, max_val: float) -> Tuple[float, float, float]:
    if value is None:
        return 0.0, min_val, max_val
    if isinstance(value, str):
        normalized = _range_to_score(value, 0.5)
        return normalized, 0.0, 1.0
    if not isinstance(value, (int, float)):
        return 0.0, min_val, max_val
    if max_val <= min_val:
        max_val = min_val + 1.0
    norm = (float(value) - min_val) / (max_val - min_val)
    norm = float(np.clip(norm, 0.0, 1.0))
    return norm, min_val, max_val


def _decode_mask(rle_payload: Mapping[str, Any]) -> np.ndarray:
    data = rle_payload
    if "data" in rle_payload:
        data = rle_payload["data"]
    rle_dict = {
        "counts": data.get("counts"),
        "size": data.get("size") or data.get("mask_shape"),
    }
    counts = rle_dict["counts"]
    size = rle_dict["size"]
    if size is None:
        raise ValueError("RLE payload missing size information.")

    if isinstance(counts, str):
        rle_dict["counts"] = counts.encode("utf-8")
        mask = mask_utils.decode(rle_dict)
    elif isinstance(counts, bytes):
        mask = mask_utils.decode(rle_dict)
    elif isinstance(counts, list):
        # Uncompressed RLE (list of counts). Convert to compressed via frPyObjects.
        fr_obj = {"counts": counts, "size": size}
        compressed = mask_utils.frPyObjects(fr_obj, size[0], size[1])
        mask = mask_utils.decode(compressed)
    else:
        raise TypeError(f"Unsupported RLE 'counts' type: {type(counts)}")

    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


class JsonPhyspropDataset(Dataset):
    """Dataset that turns small JSON benchmarks into training samples for VLM supervision."""

    def __init__(
        self,
        json_path: str,
        num_frames: int = 61,
        resolution: int | str = 480,
        conditioning_type: str = "image_blob",
        physprop_inputs: str = "all",
        blob_type: str = "circle",
        base_path: Optional[str] = None,
        repeat_first_frame: bool = True,
        include_kubric_generated: bool = False,
        t5_embeddings_path: Optional[str] = None,
        neg_t5_embeddings_path: Optional[str] = None,
        default_prompt: str = "",
        fps: int = 10,
    ) -> None:
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)
        if not isinstance(raw_items, list):
            raise ValueError(f"JSON at {json_path} must contain a list of samples.")
        self.items: List[Dict[str, Any]] = [
            item
            for item in raw_items
            if include_kubric_generated or not item.get("is_kubric_generated", False)
        ]
        if not self.items:
            raise ValueError(
                "No usable samples found in JSON (all entries marked as Kubric-generated). "
                "Set include_kubric_generated=True if you want to keep them."
            )
        self.sequence_length = int(num_frames)
        self.resolution = str(resolution)
        if self.resolution not in VIDEO_RES_SIZE_INFO:
            raise ValueError(f"Resolution {self.resolution} is not supported by VIDEO_RES_SIZE_INFO.")
        self.conditioning_type = conditioning_type
        self.physprop_inputs = physprop_inputs
        self.blob_type = blob_type
        self.base_path = base_path
        self.repeat_first_frame = repeat_first_frame
        self.default_prompt = default_prompt
        self.fps = fps
        self.t5_embeddings_path = t5_embeddings_path
        self.neg_t5_embeddings_path = neg_t5_embeddings_path
        self._t5_cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.neg_t5_text_embeddings: Optional[torch.Tensor] = None
        if neg_t5_embeddings_path:
            neg_path = _resolve_path(neg_t5_embeddings_path, base_path)
            if os.path.exists(neg_path):
                neg_emb = torch.load(neg_path, map_location="cpu")
                neg_emb = self._pad_embeddings(neg_emb)
            else:
                log.warning(f"Negative T5 embedding path does not exist: {neg_path}")
                neg_emb = None
            self.neg_t5_text_embeddings = neg_emb
        log.info(f"Initialized JsonPhyspropDataset with {len(self.items)} samples.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]
        video_rgb = self._load_first_frame(item)
        video_size = self._compute_video_size(video_rgb)
        resized_frame = cv2.resize(video_rgb, (video_size[1], video_size[0]), interpolation=cv2.INTER_AREA)
        video_tensor = self._tile_video(resized_frame)

        seg_frame, seg_ids, fg_seg_ids, bg_seg_id = self._load_segmentation(item, video_size)

        props_of_interest = self._prepare_props_of_interest(item)
        physprops_range = self._build_physprops_range(item)
        force_magnitude = item.get("force_magnitude")
        force_min = float(item.get("force_magnitude_min", 0.0))
        force_max = float(item.get("force_magnitude_max", max(force_min + 1.0, 1.0)))
        norm_force, force_min, force_max = _normalize_force(force_magnitude, force_min, force_max)

        dir_angle = item.get("dir_angle")
        dir_angle_rad = None
        if dir_angle is not None:
            dir_angle_rad = math.radians(float(dir_angle)) if abs(dir_angle) > 2 * math.pi else float(dir_angle)

        metadata = self._build_metadata(
            seg_ids=seg_ids,
            fg_seg_ids=fg_seg_ids,
            bg_seg_id=bg_seg_id,
            move_seg_id=item.get("move_object_seg_id"),
            video_size=video_size,
            norm_force=norm_force,
            force_min=force_min,
            force_max=force_max,
            dir_angle_rad=dir_angle_rad,
        )

        physprop_tensor, physprop_text_labels = self._build_physprop_tensor(
            seg_frame=seg_frame,
            metadata=metadata,
            physprops_range=physprops_range,
            fg_seg_ids=fg_seg_ids,
            bg_seg_id=bg_seg_id,
            props_of_interest=props_of_interest,
            fg_object_types=item.get("fg_object_types"),
        )

        avg_props = self._compute_avg_props(item, norm_force)
        avg_props_tensor = {k: torch.tensor(v, dtype=torch.float32) for k, v in avg_props.items()}

        direction_annotation = self._build_direction_annotation(dir_angle_rad)
        if dir_angle_rad is not None and direction_annotation:
            start_norm = direction_annotation["overlay"]["start"]
            end_norm = direction_annotation["overlay"]["end"]
            metadata["dir_start_image_coordinates"] = (
                start_norm[0] * video_size[1],
                start_norm[1] * video_size[0],
            )
            metadata["dir_end_image_coordinates"] = (
                end_norm[0] * video_size[1],
                end_norm[1] * video_size[0],
            )

        embedding_path = self._resolve_item_t5_path(item, index)
        t5_embeddings, t5_mask = self._get_t5_embeddings(
            embedding_path,
            use_text_encoder=item.get("use_text_encoder", False),
        )

        sample = {
            "video": video_tensor,
            "physprop": physprop_tensor,
            "prompt": item.get("prompt", self.default_prompt),
            "negative_prompt": item.get("negative_prompt", ""),
            "props_of_interest": ",".join(props_of_interest) if props_of_interest else "",
            "chosen_question_type": item.get(
                "question_type",
                props_of_interest[0] if props_of_interest else "None",
            ),
            "avg_props": avg_props_tensor,
            "physprop_text_labels": physprop_text_labels or self._default_text_labels(item),
            "vlm_direction_annotation": direction_annotation or "",
            "fps": self.fps,
            "frame_start": 0,
            "frame_end": self.sequence_length,
            "num_frames": self.sequence_length,
            "image_size": torch.tensor(
                [video_size[0], video_size[1], video_size[0], video_size[1]], dtype=torch.int64
            ),
            "padding_mask": torch.zeros(1, video_size[0], video_size[1], dtype=torch.float32),
        }

        if t5_embeddings is not None:
            sample["t5_text_embeddings"] = t5_embeddings
            sample["t5_text_mask"] = t5_mask
        if self.neg_t5_text_embeddings is not None:
            sample["neg_t5_text_embeddings"] = self.neg_t5_text_embeddings
        sample["first_depth_frame"] = torch.zeros_like(video_tensor[:, 0])
        sample["first_seg_frame"] = torch.from_numpy(seg_frame.astype(np.int32))
        sample["sample_id"] = item.get("id", index)
        sample["conditioning_type"] = self.conditioning_type
        return sample

    def _load_first_frame(self, item: Mapping[str, Any]) -> np.ndarray:
        path = item.get("input_video")
        if not path:
            raise ValueError("JSON sample is missing 'input_video'")
        resolved = _resolve_path(path, self.base_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Input video or image not found: {resolved}")
        ext = os.path.splitext(resolved)[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            frame = mp.read_image(resolved)
        elif ext in _VIDEO_EXTENSIONS:
            video = mp.read_video(resolved)
            frame = video[0]
        else:
            raise ValueError(f"Unsupported input format: {resolved}")
        frame = np.asarray(frame)[..., :3]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _compute_video_size(self, frame: np.ndarray) -> Tuple[int, int]:
        aspect_ratio = detect_aspect_ratio((frame.shape[0], frame.shape[1]))
        return VIDEO_RES_SIZE_INFO[self.resolution][aspect_ratio]

    def _tile_video(self, frame: np.ndarray) -> torch.Tensor:
        if self.repeat_first_frame:
            video_np = np.repeat(frame[None, ...], self.sequence_length, axis=0)
        else:
            video_np = frame[None, ...]
        video_np = np.ascontiguousarray(video_np.astype(np.uint8))
        video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).contiguous()
        return video_tensor

    def _load_segmentation(
        self,
        item: Mapping[str, Any],
        video_size: Tuple[int, int],
    ) -> Tuple[np.ndarray, List[int], List[int], int]:
        seg_path = item.get("segmentation_pkl")
        if not seg_path:
            raise ValueError("JSON sample is missing 'segmentation_pkl'")
        resolved = _resolve_path(seg_path, self.base_path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"Segmentation pickle not found: {resolved}")
        needed_seg_ids = item.get("needed_segmentation_ids")
        if not needed_seg_ids:
            raise ValueError("JSON sample must define 'needed_segmentation_ids'")
        raw_seg_ids: List[int] = [int(seg_id) for seg_id in needed_seg_ids]
        converted_seg_ids = [seg_id + 1 for seg_id in raw_seg_ids]

        fg_seg_ids_field = item.get("fg_seg_ids")
        if fg_seg_ids_field is None:
            fg_seg_ids = [converted_seg_ids[-1]]
        else:
            fg_seg_ids = [int(seg_id) + 1 for seg_id in fg_seg_ids_field]
        bg_seg_id = item.get("bg_seg_id")
        if bg_seg_id is None:
            bg_seg_id = converted_seg_ids[0]
        else:
            bg_seg_id = int(bg_seg_id) + 1

        with open(resolved, "rb") as f:
            seg_data = pickle.load(f)

        seg_frame = np.zeros(video_size, dtype=np.int32)
        for seg_id in raw_seg_ids:
            if seg_id >= len(seg_data):
                log.warning(f"Segmentation id {seg_id} out of range for {resolved}")
                continue
            mask_record = seg_data[seg_id]
            rle_payload = mask_record.get("segmentation_mask_rle")
            if not rle_payload:
                continue
            decoded = _decode_mask(rle_payload)
            resized = cv2.resize(
                decoded, (video_size[1], video_size[0]), interpolation=cv2.INTER_NEAREST
            )
            seg_frame[resized > 0] = seg_id + 1
        return seg_frame, converted_seg_ids, fg_seg_ids, bg_seg_id

    def _build_metadata(
        self,
        seg_ids: Sequence[int],
        fg_seg_ids: Sequence[int],
        bg_seg_id: int,
        move_seg_id: Optional[int],
        video_size: Tuple[int, int],
        norm_force: float,
        force_min: float,
        force_max: float,
        dir_angle_rad: Optional[float],
    ) -> Dict[str, Any]:
        num_objects = len(seg_ids)
        mass = [1.0 for _ in range(num_objects)]
        bounciness = [0.1 for _ in range(num_objects)]
        friction = [0.5 for _ in range(num_objects)]
        metadata = {
            "segmentation_id": list(seg_ids),
            "mass": mass,
            "restitution": bounciness,
            "friction": friction,
            "neo_hookean_mu": None,
            "neo_hookean_lambda": None,
            "neo_hookean_damping": None,
            "force_magnitude": norm_force,
            "force_magnitude_min": force_min,
            "force_magnitude_max": force_max,
            "dir_angle": dir_angle_rad,
            "move_object_seg_id": int(move_seg_id) + 1 if move_seg_id is not None else fg_seg_ids[-1],
            "image_width": video_size[1],
            "image_height": video_size[0],
        }
        return metadata

    def _build_physprops_range(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "friction": item.get("friction", "default"),
            "bounciness": item.get("bounciness", "default"),
            "mass": item.get("mass", "default"),
            "neo_hookean_mu": item.get("neo_hookean_mu", "default"),
            "neo_hookean_lambda": item.get("neo_hookean_lambda", "default"),
            "neo_hookean_damping": item.get("neo_hookean_damping", "default"),
        }

    def _prepare_props_of_interest(self, item: Mapping[str, Any]) -> List[str]:
        props = item.get("props_of_interest")
        if props is None:
            inferred = []
            for key in ("friction", "bounciness", "mass", "neo_hookean_mu", "force_magnitude"):
                if key in item:
                    inferred.append("deformable" if key.startswith("neo_") else key)
            return inferred
        if isinstance(props, str):
            return [props]
        return list(props)

    def _build_physprop_tensor(
        self,
        seg_frame: np.ndarray,
        metadata: Dict[str, Any],
        physprops_range: Dict[str, Any],
        fg_seg_ids: Sequence[int],
        bg_seg_id: int,
        props_of_interest: Sequence[str],
        fg_object_types: Optional[Sequence[str]],
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if self.conditioning_type == "image":
            physprop_np = get_physprop_as_spatial_data(
                metadata, seg_frame, physprop_inputs=self.physprop_inputs, physprops_range=physprops_range
            )
            physprop_tensor = torch.from_numpy(physprop_np).permute(2, 0, 1).to(torch.float32)
            return physprop_tensor, None
        if self.conditioning_type == "fg_bg_vector":
            physprop_np = get_physprop_as_fg_bg_vector(
                metadata, physprop_inputs=self.physprop_inputs, physprops_range=physprops_range
            )
            physprop_tensor = torch.from_numpy(physprop_np).to(torch.float32)
            return physprop_tensor, None
        if self.conditioning_type != "image_blob":
            raise ValueError(f"Unsupported conditioning_type: {self.conditioning_type}")
        physprop_np, text_labels = get_physprop_as_image_blob(
            metadata,
            seg_frame,
            physprop_type=self.physprop_inputs,
            physprops_range=physprops_range,
            bg_seg_id=bg_seg_id,
            fg_seg_ids=list(fg_seg_ids),
            props_of_interest=props_of_interest,
            blob_type=self.blob_type,
            fg_object_types=fg_object_types,
            return_physprop_text_labels=True,
        )
        physprop_tensor = torch.from_numpy(physprop_np).permute(2, 0, 1).to(torch.float32)
        return physprop_tensor, text_labels

    def _compute_avg_props(self, item: Mapping[str, Any], norm_force: float) -> Dict[str, float]:
        return {
            "friction": _range_to_score(item.get("friction"), 0.5),
            "bounciness": _range_to_score(item.get("bounciness"), 0.5),
            "deformability": _range_to_score(item.get("neo_hookean_mu"), 0.5),
            "force_magnitude": norm_force,
        }

    def _build_direction_annotation(self, dir_angle_rad: Optional[float]) -> Optional[Dict[str, Any]]:
        if dir_angle_rad is None:
            return None
        start = [0.5, 0.7]
        length = 0.25
        end = [
            start[0] + length * math.cos(dir_angle_rad),
            start[1] - length * math.sin(dir_angle_rad),
        ]
        end[0] = float(np.clip(end[0], 0.0, 1.0))
        end[1] = float(np.clip(end[1], 0.0, 1.0))
        return {
            "answer": "Yes",
            "overlay": {
                "type": "arrow",
                "start": [float(start[0]), float(start[1])],
                "end": end,
                "normalized": True,
                "color": [255, 0, 0],
            },
        }

    def _get_t5_embeddings(
        self,
        path: Optional[str],
        use_text_encoder: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not use_text_encoder:
            return None, None
        resolved = _resolve_path(path or "", self.base_path) if path else self.t5_embeddings_path
        if not resolved:
            return self._zero_t5()
        resolved = _resolve_path(resolved, self.base_path)
        cache_key = resolved
        if cache_key in self._t5_cache:
            return self._t5_cache[cache_key]
        if not os.path.exists(resolved):
            log.warning(f"T5 embedding path does not exist: {resolved}")
            embeddings, mask = self._zero_t5()
            self._t5_cache[cache_key] = (embeddings, mask)
            return embeddings, mask
        emb = torch.load(resolved, map_location="cpu")
        emb = self._pad_embeddings(emb)
        tokens = emb.shape[0]
        mask = torch.zeros(emb.shape[0], dtype=torch.int64)
        mask[:tokens] = 1
        self._t5_cache[cache_key] = (emb, mask)
        return emb, mask

    def _pad_embeddings(self, emb: torch.Tensor, target_tokens: int = 512) -> torch.Tensor:
        emb = emb.contiguous().to(torch.bfloat16)
        if emb.shape[0] < target_tokens:
            pad = torch.zeros(target_tokens - emb.shape[0], emb.shape[1], dtype=emb.dtype)
            emb = torch.cat([emb, pad], dim=0)
        return emb

    def _zero_t5(self) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = torch.zeros(512, 1024, dtype=torch.bfloat16)
        mask = torch.zeros(512, dtype=torch.int64)
        return embeddings, mask

    def _resolve_item_t5_path(self, item: Mapping[str, Any], index: int) -> Optional[str]:
        explicit = item.get("t5_embeddings_path")
        if explicit:
            return explicit
        if not self.t5_embeddings_path:
            return None

        candidate_root = _resolve_path(self.t5_embeddings_path, self.base_path)
        if not os.path.isdir(candidate_root):
            return self.t5_embeddings_path

        candidate_names: List[str] = []
        for key in ("t5_embeddings_name", "t5_filename", "output_video", "input_video", "input_frame"):
            value = item.get(key)
            if isinstance(value, str) and value:
                candidate_names.append(os.path.splitext(os.path.basename(value))[0])
        if "id" in item:
            candidate_names.append(f"id_{item['id']}")
        candidate_names.append(f"sample_{index}")

        seen: set[str] = set()
        for name in candidate_names:
            if not name:
                continue
            base = os.path.splitext(os.path.basename(str(name)))[0]
            if base in seen:
                continue
            seen.add(base)
            candidate_file = os.path.join(candidate_root, base + ".pt")
            if os.path.exists(candidate_file):
                return candidate_file

        return self.t5_embeddings_path

    def _default_text_labels(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        labels = {}
        for key in ("friction", "bounciness", "mass", "neo_hookean_mu", "neo_hookean_lambda", "neo_hookean_damping"):
            if key in item:
                labels[key] = item[key]
        if "force_magnitude" in item:
            labels["force_magnitude"] = item["force_magnitude"]
        return labels
