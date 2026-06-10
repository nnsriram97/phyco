import os
import pickle
import traceback
import warnings
import json
import copy
import random
import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torchvision import transforms as T
from cosmos_predict2.data.dataset_utils import VIDEO_RES_SIZE_INFO
from cosmos_predict2.data.dataset_utils import Resize_Preprocess, ToTensorVideo, detect_aspect_ratio
from imaginaire.utils import log
from pycocotools import mask as mask_utils
import cv2
from cosmos_predict2.data.kubric_data.kubric_utils import *

class KubricDataset(Dataset):
    def __init__(
        self,
        dataset_json_path,
        num_frames,
        resolution,
        conditioning_type="image",
        desired_fps=10,
        prompt_path=None,
        is_train=True,
        vlm_question_config_path=None,
    ):
        super(KubricDataset, self).__init__()
        self.dataset_json_path = dataset_json_path
        self.sequence_length = num_frames
        self.is_train = is_train
        self.resolution = str(resolution)
        self.desired_fps = desired_fps
        self.conditioning_type = conditioning_type
        self.prompt_path = prompt_path
        self.t5_text_embedding = torch.load(self.prompt_path)
        self.vlm_question_config = {}
        if vlm_question_config_path is not None and os.path.exists(vlm_question_config_path):
            try:
                with open(vlm_question_config_path, "r") as f:
                    self.vlm_question_config = json.load(f)
                log.info(
                    f"Loaded VLM question configuration with {len(self.vlm_question_config)} entries from {vlm_question_config_path}"
                )
            except Exception as exc:
                log.warning(f"Failed to load VLM question configuration {vlm_question_config_path}: {exc}")
        
        assert conditioning_type in ["image"], "Invalid conditioning type"
        assert (
            str(resolution) in VIDEO_RES_SIZE_INFO.keys()
        ), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."

        with open(self.dataset_json_path, "r") as f:
            valid_data_folders = json.load(f)
        
        # Valid data folders is a list of strings, each string is a folder name stats.json file
        self.scene_fname_gather = []
        for folder_stats_json_path in valid_data_folders:
            with open(folder_stats_json_path, "r") as f:
                folder_stats = json.load(f)
            
            if is_train:
                self.scene_fname_gather.extend(folder_stats["scene_metadata"].keys())
            else:
                # Add only 10 scenes, but only add if the number of objects is greater than 10
                for key in folder_stats["scene_metadata"].keys():
                    if folder_stats["scene_metadata"][key]["num_objects"] > 10:
                        self.scene_fname_gather.append(key)
                    if len(self.scene_fname_gather) >= 10:
                        break
        
        self.scene_fname_gather = list(set(self.scene_fname_gather))
        log.info(f"Finish initializing dataset with {len(self.scene_fname_gather)} scenes in total.")
        
        self.preprocess = None

    def _sample_frames(self, video_path, start_min, start_max, sampling_rate=1, desired_fps=10):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        n_frames = len(vr)
        # logging.debug(f"n_frames: {n_frames}")
        start_frame = np.random.randint(start_min, start_max + 1)
        try:
            video_fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS
            video_fps = 30
        if desired_fps is not None:
            sampling_rate = int(video_fps / desired_fps)
        # logging.debug(f"Start frame: {start_frame}, Sampling rate: {sampling_rate}, Sequence length: {self.sequence_length}")
        frame_ids = list(range(start_frame, n_frames, sampling_rate))
        frame_ids = frame_ids[:self.sequence_length]
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = frames.astype(np.uint8)
        
        return frames, frame_ids, int(video_fps/sampling_rate)
    
    def _get_physprop_as_spatial_data(self, metadata, first_seg_frame):
        seg_ids = metadata['segmentation_id']
        # Seg ids can be something like [5, 7, 1, 2]
        # We need to map them to [2, 3, 0, 1]
        object_ids = np.zeros_like(seg_ids)
        for i, seg_id in enumerate(seg_ids):
            object_ids[i] = seg_ids.index(seg_id)
        
        mass = metadata['mass']
        bounciness = metadata['restitution']
        static_friction = metadata['friction']
        
        physprop_frame = np.full((first_seg_frame.shape[0], first_seg_frame.shape[1], 3), fill_value=-1, dtype=np.float32)
        for k, (obj_id, seg_id) in enumerate(zip(object_ids, seg_ids)):
            obj_mask = first_seg_frame == seg_id
            if obj_id == 0:
                continue
            physprop_frame[obj_mask, 0] = np.log10(mass[obj_id] + 1)
            physprop_frame[obj_mask, 1] = bounciness[obj_id]
            physprop_frame[obj_mask, 2] = static_friction[obj_id]
        return physprop_frame

    def _get_seg_frame_as_obj_index(self, seg_frame, seg_colors_dict):

        # seg_colors_dict: {object_id: [R, G, B], ...}
        # seg_frame: H x W x 3 (uint8)
        # Output: H x W (int), where each pixel is the object id (or 0 for background)

        # Build a color to object_id mapping for fast lookup
        color_to_objid = {}
        for obj_id, color in seg_colors_dict.items():
            # Convert color to tuple for hashable key
            color_tuple = tuple(color)
            color_to_objid[color_tuple] = int(obj_id)

        # Flatten seg_frame to (H*W, 3)
        h, w, c = seg_frame.shape
        seg_flat = seg_frame.reshape(-1, 3)
        obj_index_flat = np.zeros((h * w,), dtype=np.int32)

        # For each unique color in the frame, assign object id
        unique_colors, inverse_indices = np.unique(seg_flat, axis=0, return_inverse=True)
        for idx, color in enumerate(unique_colors):
            color_tuple = tuple(color.tolist())
            obj_id = color_to_objid.get(color_tuple, 0)  # 0 for background/unmatched
            obj_index_flat[inverse_indices == idx] = obj_id

        obj_index = obj_index_flat.reshape(h, w)
        return obj_index

    def __getitem__(self, index):
        max_retries = 3
        for _ in range(max_retries):
            try:

                data = dict()
                
                scene_fname = self.scene_fname_gather[index]
                scene_key = scene_fname
                if scene_key not in self.vlm_question_config:
                    scene_key = os.path.basename(scene_fname)
                question_config = self.vlm_question_config.get(scene_key, {})
                video_path = os.path.join(scene_fname, "rgba.mp4")
                seg_video_path = os.path.join(scene_fname, "segmentation.mp4")
                metadata_path = os.path.join(scene_fname, "metadata.json")
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                start_min = 0
                start_max = 0

                frames, frame_ids, fps = self._sample_frames(video_path, start_min, start_max, desired_fps=self.desired_fps)
                
                if len(frame_ids) < self.sequence_length:
                    # repeat the last frame till the sequence length
                    frames = np.concatenate([frames, frames[-1:].repeat(self.sequence_length - len(frame_ids), axis=0)])
                    frame_ids = frame_ids + list(range(len(frame_ids), self.sequence_length))

                if frames is None:  # Invalid video or too short
                    index = np.random.randint(len(self.scene_fname_gather))
                    continue
                video = torch.from_numpy(frames)
                video = video.permute(3, 0, 1, 2)  # Rearrange from [T, H, W, C] to [C, T, H, W]
                
                aspect_ratio = detect_aspect_ratio((video.shape[2], video.shape[3]))  # expects (W, H)
                self.video_size = VIDEO_RES_SIZE_INFO[self.resolution][aspect_ratio]
                    
                if os.path.exists(seg_video_path):
                    seg_frames, seg_frame_ids, seg_fps = self._sample_frames(seg_video_path, start_min, start_max, desired_fps=self.desired_fps)
                    first_seg_frame = seg_frames[0]
                    # Resize first_seg_frame to the video size
                    first_seg_frame = cv2.resize(first_seg_frame, (self.video_size[1], self.video_size[0]))
                    first_seg_frame = get_kubric_seg_frame_as_obj_index(first_seg_frame, metadata["segmentation_color_map"])
                else:
                    log.critical(f"No seg_video_path found for {scene_fname}")
                    index = np.random.randint(len(self.scene_fname_gather))
                    continue
                
                physprop_frame = None
                if self.conditioning_type == "image":
                    physprop_frame = self._get_physprop_as_spatial_data(metadata["object_data"], first_seg_frame)
                    physprop_frame = torch.from_numpy(physprop_frame).permute(2, 0, 1)
                    data["physprop"] = physprop_frame
                    data["first_seg_frame"] = first_seg_frame
                    
                # log.info(f"Before preprocess: video shape: {video.shape}")
                if self.preprocess is None:
                    self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(self.video_size)])
                    video = self.preprocess(video)
                    video = torch.clamp(video * 255.0, 0, 255).to(torch.uint8)
                else:
                    video = self.preprocess(video)
                    video = torch.clamp(video * 255.0, 0, 255).to(torch.uint8)
                # logging.debug(f"After preprocess: {video.shape}")

                data["video"] = video

                props_of_interest = metadata.get("props_of_interest", [])
                override_props = question_config.get("props")
                if override_props is not None:
                    props_of_interest = override_props
                if isinstance(props_of_interest, str):
                    props_of_interest = [props_of_interest]
                data["vlm_props"] = props_of_interest

                if "additional_questions" in question_config:
                    data["vlm_additional_questions"] = question_config["additional_questions"]

                if "direction" in question_config:
                    data["vlm_direction_annotation"] = question_config["direction"]

                # Load T5 embeddings
                if self.prompt_path is not None:
                    n_tokens = self.t5_text_embedding.shape[0]
                    if n_tokens < 512:
                        t5_text_embedding = torch.cat([self.t5_text_embedding, torch.zeros(512 - n_tokens, 1024)], dim=0)
                    t5_text_mask = torch.zeros(512, dtype=torch.int64)
                    t5_text_mask[:n_tokens] = 1
                    data["t5_text_embeddings"] = t5_text_embedding
                    data["t5_text_mask"] = t5_text_mask
                else:
                    data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
                    data["t5_text_mask"] = torch.zeros(512, dtype=torch.int64)
                    
                # Add metadata
                data["fps"] = fps
                data["frame_start"] = frame_ids[0]
                data["frame_end"] = frame_ids[-1] + 1
                data["num_frames"] = self.sequence_length
                data["image_size"] = torch.tensor([video.shape[2], video.shape[3], video.shape[2], video.shape[3]])
                data["padding_mask"] = torch.zeros(1, video.shape[2], video.shape[3])

                return data

            except Exception as e:
                warnings.warn(f"Invalid data encountered: {scene_fname}. Skipped "
                    f"(by randomly sampling another sample in the same dataset)."
                )
                warnings.warn("FULL TRACEBACK:")
                warnings.warn(traceback.format_exc())
                if _ == max_retries - 1:
                    raise RuntimeError(f"Failed to load data after {max_retries} attempts")
                index = np.random.randint(len(self.scene_fname_gather))
        return None

    def __len__(self):
        return len(self.scene_fname_gather)

    def __str__(self):
        return f"{len(self.scene_fname_gather)} samples from {self.dataset_json_path}"


class KubricDataset_v2(Dataset):
    def __init__(
        self,
        dataset_json_path,
        num_frames,
        resolution,
        conditioning_type="image",
        desired_fps=10,
        prompt_path=None,
        is_train=True,
        input_type="video",
        physprop_inputs="all",
        base_path=None,
        blob_type="circle",
        no_background_condition=False,
        t5_text_embeddings_dir=None,
    ):
        super(KubricDataset_v2, self).__init__()
        self.dataset_json_path = dataset_json_path
        self.sequence_length = num_frames
        self.is_train = is_train
        self.resolution = str(resolution)
        self.desired_fps = desired_fps
        self.conditioning_type = conditioning_type
        self.base_path = base_path
        self.blob_type = blob_type

        # self.prompt_path = prompt_path
        # self.t5_text_embedding = torch.load(self.prompt_path)
        self.t5_text_embeddings_dir = t5_text_embeddings_dir
        
        self.input_type = input_type
        self.physprop_inputs = physprop_inputs
        self.no_background_condition = no_background_condition

        assert conditioning_type in ["image", "fg_bg_vector", "image_blob"], "Invalid conditioning type"
        if conditioning_type == "image_blob":
            assert blob_type in ["circle", "convex_hull", "ellipse"], "Invalid blob type"
        assert (
            str(resolution) in VIDEO_RES_SIZE_INFO.keys()
        ), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."

        with open(self.dataset_json_path, "r") as f:
            valid_data_folders = json.load(f)
        
        # Valid data folders is a list of strings, each string is a folder name stats.json file
        self.scene_fname_gather = []
        for folder_stats_json_path in valid_data_folders:
            if self.base_path is not None:
                # Replace everything before 'kubric_generated' with self.base_path
                parts = folder_stats_json_path.split(os.sep)
                if "kubric_generated" in parts:
                    idx = parts.index("kubric_generated")
                    # Join self.base_path with the rest of the path after 'kubric_generated'
                    folder_stats_json_path = os.path.join(self.base_path, *parts[idx:])
                else:
                    log.warning(f"'kubric_generated' not found in scene_fname: {folder_stats_json_path}")

            with open(folder_stats_json_path, "r") as f:
                folder_stats = json.load(f)
            
            if is_train:
                self.scene_fname_gather.extend(folder_stats["scene_metadata"].keys())
            else:
                # Add only 10 scenes, but only add if the number of objects is greater than 10
                self.scene_fname_gather.extend(folder_stats["scene_metadata"].keys())
                # for key in folder_stats["scene_metadata"].keys():
                #     if folder_stats["scene_metadata"][key]["num_objects"] > 10:
                #         self.scene_fname_gather.append(key)
                #     if len(self.scene_fname_gather) >= 10:
                #         break
        
        self.scene_fname_gather = list(set(self.scene_fname_gather))
        log.info(f"Finish initializing dataset with {len(self.scene_fname_gather)} scenes in total.")
        
        # Load negative T5 text embeddings
        if self.base_path is not None and os.path.exists(os.path.join(self.base_path, "kubric_generated/t5_xxl/common_neg_prompt-v1.pt")):
            self.neg_t5_text_embeddings = torch.load(os.path.join(self.base_path, "kubric_generated/t5_xxl/common_neg_prompt-v1.pt"))
        elif os.path.exists("/net/acadia1a/data/sriram/vidgen/datasets/kubric_generated/t5_xxl/common_neg_prompt-v1.pt"):
            self.neg_t5_text_embeddings = torch.load("/net/acadia1a/data/sriram/vidgen/datasets/kubric_generated/t5_xxl/common_neg_prompt-v1.pt")
        else:
            raise FileNotFoundError("Negative T5 text embeddings not found")

        self.preprocess = None

    def _sample_frames(self, video_path, start_min, start_max, sampling_rate=1, desired_fps=10):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        n_frames = len(vr)
        # logging.debug(f"n_frames: {n_frames}")
        start_frame = np.random.randint(start_min, start_max + 1)
        try:
            video_fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS
            video_fps = 30
        if desired_fps is not None:
            sampling_rate = int(video_fps / desired_fps)
        # logging.debug(f"Start frame: {start_frame}, Sampling rate: {sampling_rate}, Sequence length: {self.sequence_length}")
        frame_ids = list(range(start_frame, n_frames, sampling_rate))
        frame_ids = frame_ids[:self.sequence_length]
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = frames.astype(np.uint8)
        
        return frames, frame_ids, int(video_fps/sampling_rate)

    def _get_depth_video(self, depth_path):
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

    def __getitem__(self, index):
        max_retries = 3
        for _ in range(max_retries):
            try:

                data = dict()
                
                scene_fname = self.scene_fname_gather[index]
                if self.base_path is not None:
                    # Replace everything before 'kubric_generated' with self.base_path
                    parts = scene_fname.split(os.sep)
                    if "kubric_generated" in parts:
                        idx = parts.index("kubric_generated")
                        # Join self.base_path with the rest of the path after 'kubric_generated'
                        scene_fname = os.path.join(self.base_path, *parts[idx:])
                    else:
                        log.warning(f"'kubric_generated' not found in scene_fname: {scene_fname}")
                
                video_path = os.path.join(scene_fname, "rgba.mp4")
                seg_video_path = os.path.join(scene_fname, "segmentation.mp4")
                metadata_path = os.path.join(scene_fname, "metadata.json")
                depth_path = os.path.join(scene_fname, "depth.mp4")
                depth_path_npz = os.path.join(scene_fname, "depth.npz")
                experiment_dir = os.path.dirname(os.path.dirname(scene_fname))
                fg_bg_id_path = os.path.join(experiment_dir, "fg_bg_id.json")
                props_of_interest_path = os.path.join(experiment_dir, "props_of_interest.json")
                props_of_interest = None
                if os.path.exists(props_of_interest_path):
                    with open(props_of_interest_path, "r") as f:
                        props_of_interest = json.load(f)

                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                start_min = 0
                start_max = 0

                if self.input_type == "video":
                    frames, frame_ids, fps = self._sample_frames(video_path, start_min, start_max, desired_fps=self.desired_fps)
                elif self.input_type == "depth":
                    # frames = self._get_depth_video(depth_path)
                    # frame_ids = list(range(0, frames.shape[0], 1))
                    frames, frame_ids, fps = self._sample_frames(depth_path, start_min, start_max, desired_fps=self.desired_fps)
                    fps = self.desired_fps
                
                if len(frame_ids) < self.sequence_length:
                    # repeat the last frame till the sequence length
                    frames = np.concatenate([frames, frames[-1:].repeat(self.sequence_length - len(frame_ids), axis=0)])
                    frame_ids = frame_ids + list(range(len(frame_ids), self.sequence_length))

                if frames is None:  # Invalid video or too short
                    index = np.random.randint(len(self.scene_fname_gather))
                    continue
                video = torch.from_numpy(frames)
                video = video.permute(3, 0, 1, 2)  # Rearrange from [T, H, W, C] to [C, T, H, W]
                
                aspect_ratio = detect_aspect_ratio((video.shape[2], video.shape[3]))  # expects (W, H)
                self.video_size = VIDEO_RES_SIZE_INFO[self.resolution][aspect_ratio]
                    
                if os.path.exists(seg_video_path):
                    seg_frames, seg_frame_ids, seg_fps = self._sample_frames(seg_video_path, start_min, start_max, desired_fps=self.desired_fps)
                    first_seg_frame = seg_frames[0]
                    # Resize first_seg_frame to the video size
                    first_seg_frame = cv2.resize(first_seg_frame, (self.video_size[1], self.video_size[0]))
                    first_seg_frame = get_kubric_seg_frame_as_obj_index(first_seg_frame, metadata["segmentation_color_map"])
                else:
                    log.critical(f"No seg_video_path found for {scene_fname}")
                    index = np.random.randint(len(self.scene_fname_gather))
                    continue

                if os.path.exists(depth_path):
                    # depth_frames = self._get_depth_video(depth_path)
                    depth_frames, depth_frame_ids, depth_fps = self._sample_frames(depth_path, start_min, start_max, desired_fps=self.desired_fps)
                    first_depth_frame = depth_frames[depth_frame_ids[0]]
                    first_depth_frame = cv2.resize(first_depth_frame, (self.video_size[1], self.video_size[0]))
                    data["first_depth_frame"] = np.transpose(first_depth_frame, (2, 0, 1))
                elif os.path.exists(depth_path_npz):
                    depth_frames = self._get_depth_video(depth_path_npz)
                    first_depth_frame = depth_frames[frame_ids[0]]
                    first_depth_frame = cv2.resize(first_depth_frame, (self.video_size[1], self.video_size[0]))
                    data["first_depth_frame"] = np.transpose(first_depth_frame, (2, 0, 1))
                else:
                    log.critical(f"No depth_path found for {scene_fname}")
                    index = np.random.randint(len(self.scene_fname_gather))
                    continue
                
                physprop_frame = None
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
                    if "object_name" in metadata["object_data"]:
                        object_name_idx = metadata["object_data"]["object_name"].index(force_object_name)
                        move_object_seg_id = metadata["object_data"]["segmentation_id"][object_name_idx]
                    else:
                        move_object_seg_id = metadata["object_data"]["segmentation_id"][-1] # Use the last object as the moving object
                    force_metadata["move_object_seg_id"] = move_object_seg_id
                    metadata_update.update(force_metadata)
                
                if "applied_velocities_image" in metadata:
                    velocity_information = metadata["applied_velocities_image"][0]
                    velocity_metadata = {}
                    velocity_metadata["dir_start_image_coordinates"] = velocity_information["image_coordinates"]
                    if "velocity_end_image_coordinates" in velocity_information:
                        velocity_metadata["dir_end_image_coordinates"] = velocity_information["velocity_end_image_coordinates"]    
                        velocity_sim_info = metadata["applied_velocities_simulator"][0]
                        velocity_object_name = velocity_sim_info["object_name"]
                        if "object_name" in metadata["object_data"]:
                            object_name_idx = metadata["object_data"]["object_name"].index(velocity_object_name)
                            move_object_seg_id = metadata["object_data"]["segmentation_id"][object_name_idx]
                        else:
                            move_object_seg_id = metadata["object_data"]["segmentation_id"][-1] # Use the last object as the moving object
                        velocity_metadata["move_object_seg_id"] = move_object_seg_id
                        metadata_update.update(velocity_metadata)

                if self.conditioning_type == "image":
                    physprop_frame = get_physprop_as_spatial_data(metadata["object_data"], first_seg_frame, physprop_type=self.physprop_inputs, physprops_range=None)
                    physprop_frame = torch.from_numpy(physprop_frame).permute(2, 0, 1)
                    data["physprop"] = physprop_frame
                    data["first_seg_frame"] = first_seg_frame
                elif self.conditioning_type == "fg_bg_vector":
                    physprop_vector = get_physprop_as_fg_bg_vector(metadata["object_data"], physprop_type=self.physprop_inputs, physprops_range=None)
                    data["physprop"] = torch.from_numpy(physprop_vector).to(torch.float32)
                elif self.conditioning_type == "image_blob":
                    if os.path.exists(fg_bg_id_path):
                        with open(fg_bg_id_path, "r") as f:
                            fg_bg_json = json.load(f)
                        fg_seg_start_id = fg_bg_json.get("fg_seg_start_id", None)
                        fg_seg_ids = fg_bg_json.get("fg_seg_id", None)
                        if fg_seg_start_id is not None:
                            fg_seg_ids = metadata["object_data"]["segmentation_id"][fg_seg_start_id:]
                        if "bg_seg_id" in fg_bg_json:
                            bg_seg_id = fg_bg_json["bg_seg_id"]
                    else:
                        fg_seg_ids = None
                        bg_seg_id = None
                    sim_metadata = metadata["object_data"]
                    sim_metadata.update(metadata_update)
                    physprop_blob, physprop_text_labels = get_physprop_as_image_blob(sim_metadata, first_seg_frame, physprop_type=self.physprop_inputs, physprops_range=None, fg_seg_ids=fg_seg_ids, bg_seg_id=bg_seg_id, props_of_interest=props_of_interest, blob_type=self.blob_type, return_physprop_text_labels=True, no_background_condition=self.no_background_condition)
                    data["physprop"] = torch.from_numpy(physprop_blob).permute(2, 0, 1)
                    data["first_seg_frame"] = first_seg_frame

                    # Generate answers to the questions based on the physprop frame
                    props = parse_metadata_for_props(sim_metadata, fg_seg_ids, bg_seg_id)
                    avg_props = parse_props_for_questionnaire(props)
                    data["avg_props"] = avg_props
                    question_type_path = os.path.join(experiment_dir, "question_type.json")
                    chosen_question_type = "None"
                    if os.path.exists(question_type_path):
                        with open(question_type_path, "r") as f:
                            question_types = json.load(f)
                            chosen_question_type = random.choice(question_types)
                    data["chosen_question_type"] = chosen_question_type
                    # data["physprop_text_labels"] = physprop_text_labels
                    data["props_of_interest"] = ",".join(props_of_interest) if props_of_interest else ""


                    start_coords = metadata_update.get("dir_start_image_coordinates")
                    end_coords = metadata_update.get("dir_end_image_coordinates")
                    # Use empty string as sentinel instead of None (PyTorch can't collate None)
                    direction_annotation = ""
                    direction_text = None
                    if start_coords is not None and end_coords is not None:
                        width = max(1, self.video_size[1])
                        height = max(1, self.video_size[0])
                        direction_annotation = {
                            "answer": "Yes",
                            "overlay": {
                                "type": "arrow",
                                "start": (start_coords[0] / width, start_coords[1] / height),
                                "end": (end_coords[0] / width, end_coords[1] / height),
                                "normalized": True,
                                "color": (255, 0, 0),
                                # "frame_indices": [0],
                            },
                        }
                        direction_text = get_object_movement_direction_text(start_coords, end_coords)
                    data["vlm_direction_annotation"] = direction_annotation

                # log.info(f"Before preprocess: video shape: {video.shape}")
                if self.preprocess is None:
                    self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(self.video_size)])
                    video = self.preprocess(video)
                    video = torch.clamp(video * 255.0, 0, 255).to(torch.uint8)
                else:
                    video = self.preprocess(video)
                    video = torch.clamp(video * 255.0, 0, 255).to(torch.uint8)
                # logging.debug(f"After preprocess: {video.shape}")

                data["video"] = video

                experiment_dir = os.path.dirname(os.path.dirname(scene_fname))
                if self.t5_text_embeddings_dir is not None:
                    experiment_folder_name = os.path.basename(experiment_dir)
                    folder_date_name = os.path.basename(os.path.dirname(scene_fname))
                    scene_filename = os.path.basename(scene_fname)
                    common_caption_path = os.path.join(self.t5_text_embeddings_dir, experiment_folder_name, folder_date_name, f"{scene_filename}.pt")
                    if not os.path.exists(common_caption_path):
                        common_caption_path = os.path.join(experiment_dir, "common_caption_cosmos.pt")
                    # log.info(f"Loading T5 text embeddings from {common_caption_path}")
                else:
                    common_caption_path = os.path.join(experiment_dir, "common_caption_cosmos.pt")    
                    if direction_text is not None:
                        if os.path.exists(os.path.join(experiment_dir, f"common_caption_cosmos_{direction_text}.pt")):
                            common_caption_path = os.path.join(experiment_dir, f"common_caption_cosmos_{direction_text}.pt")

                    if not os.path.exists(common_caption_path):
                        raise FileNotFoundError(f"common_caption_cosmos.pt not found at {common_caption_path}")
                t5_text_embedding = torch.load(common_caption_path)
                n_tokens = t5_text_embedding.shape[0]
                if n_tokens < 512:
                    t5_text_embedding = torch.cat([t5_text_embedding, torch.zeros(512 - n_tokens, 1024)], dim=0)
                t5_text_mask = torch.zeros(512, dtype=torch.int64)
                t5_text_mask[:n_tokens] = 1
                data["t5_text_embeddings"] = t5_text_embedding.to(torch.bfloat16)
                data["t5_text_mask"] = t5_text_mask

                neg_t5_text_embeddings = self.neg_t5_text_embeddings.to(dtype=torch.bfloat16).contiguous()
                n_tokens = neg_t5_text_embeddings.shape[0]
                if n_tokens < 512:
                    neg_t5_text_embeddings = torch.cat([neg_t5_text_embeddings, torch.zeros(512 - n_tokens, 1024)], dim=0)
                data["neg_t5_text_embeddings"] = neg_t5_text_embeddings
                
                # Load T5 embeddings
                # if self.prompt_path is not None:
                #     n_tokens = self.t5_text_embedding.shape[0]
                #     if n_tokens < 512:
                #         t5_text_embedding = torch.cat([self.t5_text_embedding, torch.zeros(512 - n_tokens, 1024)], dim=0)
                #     t5_text_mask = torch.zeros(512, dtype=torch.int64)
                #     t5_text_mask[:n_tokens] = 1
                #     data["t5_text_embeddings"] = t5_text_embedding
                #     data["t5_text_mask"] = t5_text_mask
                # else:
                #     data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
                #     data["t5_text_mask"] = torch.zeros(512, dtype=torch.int64)
                    
                # Add metadata
                data["fps"] = fps
                data["frame_start"] = frame_ids[0]
                data["frame_end"] = frame_ids[-1] + 1
                data["num_frames"] = self.sequence_length
                data["image_size"] = torch.tensor([video.shape[2], video.shape[3], video.shape[2], video.shape[3]])
                data["padding_mask"] = torch.zeros(1, video.shape[2], video.shape[3])
                data["conditioning_type"] = self.conditioning_type
                data["base_path"] = self.base_path

                return data

            except Exception as e:
                log.error(f"Error in __getitem__: {e}")
                warnings.warn(f"Invalid data encountered: {scene_fname}. Skipped "
                    f"(by randomly sampling another sample in the same dataset)."
                )
                warnings.warn("FULL TRACEBACK:")
                warnings.warn(traceback.format_exc())
                if _ == max_retries - 1:
                    raise RuntimeError(f"Failed to load data after {max_retries} attempts")
                index = np.random.randint(len(self.scene_fname_gather))
        return None

    def __len__(self):
        return len(self.scene_fname_gather)

    def __str__(self):
        return f"{len(self.scene_fname_gather)} samples from {self.dataset_json_path}"


class KubricDatasetBaseline(Dataset):
    def __init__(
        self,
        dataset_json_path,
        num_frames,
        resolution,
        desired_fps=10,
        is_train=True,
        input_type="video",
        base_path=None,
        t5_text_embeddings_dir=None,
    ):
        super(KubricDatasetBaseline, self).__init__()
        self.dataset_json_path = dataset_json_path
        self.sequence_length = num_frames
        self.is_train = is_train
        self.resolution = str(resolution)
        self.desired_fps = desired_fps
        self.base_path = base_path

        # self.prompt_path = prompt_path
        # self.t5_text_embedding = torch.load(self.prompt_path)
        
        self.input_type = input_type
        self.t5_text_embeddings_dir = t5_text_embeddings_dir
            
        assert (
            str(resolution) in VIDEO_RES_SIZE_INFO.keys()
        ), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."

        with open(self.dataset_json_path, "r") as f:
            valid_data_folders = json.load(f)
        
        # Valid data folders is a list of strings, each string is a folder name stats.json file
        self.scene_fname_gather = []
        for folder_stats_json_path in valid_data_folders:
            if self.base_path is not None:
                # Replace everything before 'kubric_generated' with self.base_path
                parts = folder_stats_json_path.split(os.sep)
                if "kubric_generated" in parts:
                    idx = parts.index("kubric_generated")
                    # Join self.base_path with the rest of the path after 'kubric_generated'
                    folder_stats_json_path = os.path.join(self.base_path, *parts[idx:])
                else:
                    log.warning(f"'kubric_generated' not found in scene_fname: {folder_stats_json_path}")

            with open(folder_stats_json_path, "r") as f:
                folder_stats = json.load(f)
            
            if is_train:
                self.scene_fname_gather.extend(folder_stats["scene_metadata"].keys())
            else:
                # Add only 10 scenes, but only add if the number of objects is greater than 10
                self.scene_fname_gather.extend(folder_stats["scene_metadata"].keys())
                # for key in folder_stats["scene_metadata"].keys():
                #     if folder_stats["scene_metadata"][key]["num_objects"] > 10:
                #         self.scene_fname_gather.append(key)
                #     if len(self.scene_fname_gather) >= 10:
                #         break
        
        self.scene_fname_gather = list(set(self.scene_fname_gather))
        log.info(f"Finish initializing dataset with {len(self.scene_fname_gather)} scenes in total.")
        
        # Load negative T5 text embeddings
        if self.base_path is not None and os.path.exists(os.path.join(self.base_path, "kubric_generated/t5_xxl/common_neg_prompt-v1.pt")):
            self.neg_t5_text_embeddings = torch.load(os.path.join(self.base_path, "kubric_generated/t5_xxl/common_neg_prompt-v1.pt"))
        elif os.path.exists("/net/acadia1a/data/sriram/vidgen/datasets/kubric_generated/t5_xxl/common_neg_prompt-v1.pt"):
            self.neg_t5_text_embeddings = torch.load("/net/acadia1a/data/sriram/vidgen/datasets/kubric_generated/t5_xxl/common_neg_prompt-v1.pt")
        else:
            raise FileNotFoundError("Negative T5 text embeddings not found")

        self.preprocess = None

    def _sample_frames(self, video_path, start_min, start_max, sampling_rate=1, desired_fps=10):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        n_frames = len(vr)
        # logging.debug(f"n_frames: {n_frames}")
        start_frame = np.random.randint(start_min, start_max + 1)
        try:
            video_fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS
            video_fps = 30
        if desired_fps is not None:
            sampling_rate = int(video_fps / desired_fps)
        # logging.debug(f"Start frame: {start_frame}, Sampling rate: {sampling_rate}, Sequence length: {self.sequence_length}")
        frame_ids = list(range(start_frame, n_frames, sampling_rate))
        frame_ids = frame_ids[:self.sequence_length]
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = frames.astype(np.uint8)
        
        return frames, frame_ids, int(video_fps/sampling_rate)

    def _get_depth_video(self, depth_path):
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

    def __getitem__(self, index):
        max_retries = 3
        for _ in range(max_retries):
            try:

                data = dict()
                
                scene_fname = self.scene_fname_gather[index]
                if self.base_path is not None:
                    # Replace everything before 'kubric_generated' with self.base_path
                    parts = scene_fname.split(os.sep)
                    if "kubric_generated" in parts:
                        idx = parts.index("kubric_generated")
                        # Join self.base_path with the rest of the path after 'kubric_generated'
                        scene_fname = os.path.join(self.base_path, *parts[idx:])
                    else:
                        log.warning(f"'kubric_generated' not found in scene_fname: {scene_fname}")
                
                video_path = os.path.join(scene_fname, "rgba.mp4")
                metadata_path = os.path.join(scene_fname, "metadata.json")
                depth_path = os.path.join(scene_fname, "depth.mp4")
                depth_path_npz = os.path.join(scene_fname, "depth.npz")
                experiment_dir = os.path.dirname(os.path.dirname(scene_fname))

                if not os.path.exists(metadata_path):
                    log.critical(f"No metadata_path found for {scene_fname}")
                    index = np.random.randint(len(self.scene_fname_gather))
                    continue
                
                start_min = 0
                start_max = 0

                if self.input_type == "video":
                    frames, frame_ids, fps = self._sample_frames(video_path, start_min, start_max, desired_fps=self.desired_fps)
                elif self.input_type == "depth":
                    # frames = self._get_depth_video(depth_path)
                    # frame_ids = list(range(0, frames.shape[0], 1))
                    frames, frame_ids, fps = self._sample_frames(depth_path, start_min, start_max, desired_fps=self.desired_fps)
                    fps = self.desired_fps
                
                if len(frame_ids) < self.sequence_length:
                    # repeat the last frame till the sequence length
                    frames = np.concatenate([frames, frames[-1:].repeat(self.sequence_length - len(frame_ids), axis=0)])
                    frame_ids = frame_ids + list(range(len(frame_ids), self.sequence_length))

                if frames is None:  # Invalid video or too short
                    index = np.random.randint(len(self.scene_fname_gather))
                    continue
                video = torch.from_numpy(frames)
                video = video.permute(3, 0, 1, 2)  # Rearrange from [T, H, W, C] to [C, T, H, W]
                
                aspect_ratio = detect_aspect_ratio((video.shape[2], video.shape[3]))  # expects (W, H)
                self.video_size = VIDEO_RES_SIZE_INFO[self.resolution][aspect_ratio]

                if os.path.exists(depth_path):
                    # depth_frames = self._get_depth_video(depth_path)
                    depth_frames, depth_frame_ids, depth_fps = self._sample_frames(depth_path, start_min, start_max, desired_fps=self.desired_fps)
                    first_depth_frame = depth_frames[depth_frame_ids[0]]
                    first_depth_frame = cv2.resize(first_depth_frame, (self.video_size[1], self.video_size[0]))
                    data["first_depth_frame"] = np.transpose(first_depth_frame, (2, 0, 1))
                elif os.path.exists(depth_path_npz):
                    depth_frames = self._get_depth_video(depth_path_npz)
                    first_depth_frame = depth_frames[frame_ids[0]]
                    first_depth_frame = cv2.resize(first_depth_frame, (self.video_size[1], self.video_size[0]))
                    data["first_depth_frame"] = np.transpose(first_depth_frame, (2, 0, 1))
                else:
                    log.critical(f"No depth_path found for {scene_fname}")
                    index = np.random.randint(len(self.scene_fname_gather))
                    continue
                
                # log.info(f"Before preprocess: video shape: {video.shape}")
                if self.preprocess is None:
                    self.preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(self.video_size)])
                    video = self.preprocess(video)
                    video = torch.clamp(video * 255.0, 0, 255).to(torch.uint8)
                else:
                    video = self.preprocess(video)
                    video = torch.clamp(video * 255.0, 0, 255).to(torch.uint8)
                # logging.debug(f"After preprocess: {video.shape}")

                data["video"] = video

                experiment_folder_name = os.path.basename(experiment_dir)
                folder_date_name = os.path.basename(os.path.dirname(scene_fname))
                scene_filename = os.path.basename(scene_fname)
                if self.t5_text_embeddings_dir is not None:
                    common_caption_path = os.path.join(self.t5_text_embeddings_dir, experiment_folder_name, folder_date_name, f"{scene_filename}.pt")
                    if not os.path.exists(common_caption_path):
                        common_caption_path = os.path.join(experiment_dir, "common_caption_cosmos.pt")
                    # log.info(f"Loading T5 text embeddings from {common_caption_path}")
                else:
                    common_caption_path = os.path.join(experiment_dir, "common_caption_cosmos.pt")

                if not os.path.exists(common_caption_path):
                    raise FileNotFoundError(f"common_caption_cosmos.pt not found at {common_caption_path}")
                t5_text_embedding = torch.load(common_caption_path)
                n_tokens = t5_text_embedding.shape[0]
                if n_tokens < 512:
                    t5_text_embedding = torch.cat([t5_text_embedding, torch.zeros(512 - n_tokens, 1024)], dim=0)
                t5_text_mask = torch.zeros(512, dtype=torch.int64)
                t5_text_mask[:n_tokens] = 1
                data["t5_text_embeddings"] = t5_text_embedding.to(torch.bfloat16)
                data["t5_text_mask"] = t5_text_mask

                neg_t5_text_embeddings = self.neg_t5_text_embeddings.to(dtype=torch.bfloat16).contiguous()
                n_tokens = neg_t5_text_embeddings.shape[0]
                if n_tokens < 512:
                    neg_t5_text_embeddings = torch.cat([neg_t5_text_embeddings, torch.zeros(512 - n_tokens, 1024)], dim=0)
                data["neg_t5_text_embeddings"] = neg_t5_text_embeddings

                # Add metadata
                data["fps"] = fps
                data["frame_start"] = frame_ids[0]
                data["frame_end"] = frame_ids[-1] + 1
                data["num_frames"] = self.sequence_length
                data["image_size"] = torch.tensor([video.shape[2], video.shape[3], video.shape[2], video.shape[3]])
                data["padding_mask"] = torch.zeros(1, video.shape[2], video.shape[3])
                data["base_path"] = self.base_path

                return data

            except Exception as e:
                log.error(f"Error in __getitem__: {e}")
                warnings.warn(f"Invalid data encountered: {scene_fname}. Skipped "
                    f"(by randomly sampling another sample in the same dataset)."
                )
                warnings.warn("FULL TRACEBACK:")
                warnings.warn(traceback.format_exc())
                if _ == max_retries - 1:
                    raise RuntimeError(f"Failed to load data after {max_retries} attempts")
                index = np.random.randint(len(self.scene_fname_gather))
        return None

    def __len__(self):
        return len(self.scene_fname_gather)

    def __str__(self):
        return f"{len(self.scene_fname_gather)} samples from {self.dataset_json_path}"
