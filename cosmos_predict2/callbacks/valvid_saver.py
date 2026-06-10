"""Callback for saving generated validation videos.

This callback expects each model ``validation_step`` to return an ``output_batch``
containing a list of generated videos under the key ``"generated_videos"``.
Each item in the list is a 5-D ``torch.Tensor`` with shape ``(1, C, T, H, W)``
where pixel values are in the range ``[-1, 1]``. The tensor is converted to
uint8 frames and saved as an mp4 file using :pyfunc:`save_video`.
"""

from __future__ import annotations

import os
from typing import List

import torch

from imaginaire.model import ImaginaireModel
from imaginaire.utils import distributed, log
from imaginaire.utils.callback import Callback

from cosmos_predict2.auxiliary.guardrail.common.io_utils import save_video
from cosmos_predict2.data.kubric_data.kubric_utils import save_physprop_as_image, save_physprop_as_text, save_physprop_as_image_blob

import uuid
import json
import torch.distributed as dist

class ValidationVideoSaver(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.name = self.__class__.__name__

    @distributed.rank0_only
    def on_training_step_end(self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0) -> None:
        """Save debug videos and VLM question details during training.

        This method saves VLM-related debug information including:
        - Decoded videos from latent space
        - Ground truth videos
        - Overlay videos (if available)
        - Question details and answers (as JSON)
        """
        # Check if VLM debug data is present
        if "vlm_debug_data" not in output_batch:
            return

        vlm_debug_data = output_batch["vlm_debug_data"]
        if not vlm_debug_data:
            return

        # Determine save directory
        if hasattr(self, "config") and hasattr(self.config, "job"):
            save_root = os.path.join(
                "checkpoints",
                self.config.job.project,
                self.config.job.group,
                self.config.job.name,
                "debug_videos",
            )
        else:
            save_root = os.path.join(os.getcwd(), "debug_videos")

        os.makedirs(save_root, exist_ok=True)

        # Get rank to distinguish different DP ranks
        rank = 0
        if dist.is_initialized():
            rank = dist.get_rank()

        # Process each sample's debug data
        for sample_data in vlm_debug_data:
            sample_index = sample_data.get("sample_index", 0)
            append_fname = sample_data.get("append_fname", "")
            rank_suffix = f"_rank{rank}"

            # Save decoded video
            if "decoded_video" in sample_data:
                self._save_debug_video(
                    sample_data["decoded_video"],
                    save_root,
                    iteration,
                    prefix="vlm_decoded",
                    sample_index=sample_index,
                    append_fname=append_fname,
                    rank_suffix=rank_suffix,
                )

            # Save ground truth video
            if "gt_video" in sample_data:
                self._save_debug_video(
                    sample_data["gt_video"],
                    save_root,
                    iteration,
                    prefix="vlm_gt",
                    sample_index=sample_index,
                    append_fname=append_fname,
                    rank_suffix=rank_suffix,
                )

            # Save overlay video
            if "overlay_video" in sample_data:
                self._save_debug_video(
                    sample_data["overlay_video"],
                    save_root,
                    iteration,
                    prefix="vlm_overlay",
                    sample_index=sample_index,
                    append_fname=append_fname,
                    rank_suffix=rank_suffix,
                )

        # Save question details once per iteration from the global list (not per-sample)
        # This avoids duplication since question_details are shared across all samples
        if "vlm_question_details" in output_batch:
            question_details = output_batch["vlm_question_details"]
            if question_details:
                # Use the first sample's index and append_fname if available
                first_sample = vlm_debug_data[0] if vlm_debug_data else {}
                sample_index = first_sample.get("sample_index", 0)
                append_fname = first_sample.get("append_fname", "")
                rank_suffix = f"_rank{rank}"

                self._save_question_details(
                    question_details,
                    save_root,
                    iteration,
                    sample_index=sample_index,
                    append_fname=append_fname,
                    rank_suffix=rank_suffix,
                )

    def _save_debug_video(
        self,
        video_tensor: torch.Tensor,
        save_root: str,
        iteration: int,
        prefix: str,
        sample_index: int = 0,
        append_fname: str = "",
        rank_suffix: str = "",
    ) -> None:
        """Save a debug video tensor to MP4."""
        try:
            # Handle different tensor shapes
            if video_tensor.dim() == 4:
                video_tensor = video_tensor.unsqueeze(0)

            frame_tensor = video_tensor[0].detach().cpu()

            # Normalize to [0, 255] uint8
            if frame_tensor.dtype.is_floating_point:
                frame_tensor = frame_tensor.clamp(-1.0, 1.0)
                frame_tensor = ((frame_tensor + 1.0) * 127.5).round()
            else:
                frame_tensor = frame_tensor.to(torch.float32)
                if frame_tensor.max() <= 1.0:
                    frame_tensor = frame_tensor * 255.0

            frame_uint8 = frame_tensor.clamp(0.0, 255.0).to(torch.uint8)
            video_np = frame_uint8.permute(1, 2, 3, 0).numpy()

            # Save video
            mp4_path = os.path.join(
                save_root,
                f"{prefix}_b{sample_index}_{iteration:06d}{append_fname}{rank_suffix}.mp4"
            )

            try:
                import imageio.v3 as imageio
                imageio.imwrite(mp4_path, video_np, fps=24, codec="libx264")
            except Exception as debug_err:
                log.warning(f"Failed to write debug video ({prefix}): {debug_err}")
        except Exception as debug_err:
            log.warning(f"Failed to write debug video ({prefix}): {debug_err}")

    def _save_question_details(
        self,
        details: List[dict],
        save_root: str,
        iteration: int,
        sample_index: int = 0,
        append_fname: str = "",
        rank_suffix: str = "",
    ) -> None:
        """Save VLM question details to JSON and extract overlay images."""
        if not details:
            return

        try:
            # Process details to extract and save overlay images
            processed_details = []
            for i, detail in enumerate(details):
                processed_detail = detail.copy()

                # Check if this detail has an overlay image
                overlay_image = detail.get("overlay_image")
                if overlay_image is not None and isinstance(overlay_image, torch.Tensor):
                    # Save the overlay image
                    overlay_filename = f"vlm_questions_overlay_b{sample_index}_q{i}_{iteration:06d}{append_fname}{rank_suffix}.png"
                    overlay_path = os.path.join(save_root, overlay_filename)

                    try:
                        self._save_overlay_image(overlay_image, overlay_path)
                        # Replace the tensor with the filename for JSON serialization
                        processed_detail["overlay_image"] = overlay_filename
                    except Exception as img_err:
                        log.warning(f"Failed to save overlay image for question {i}: {img_err}")
                        processed_detail["overlay_image"] = None

                processed_details.append(processed_detail)

            # Save the processed details to JSON
            json_path = os.path.join(
                save_root,
                f"vlm_questions_b{sample_index}_{iteration:06d}{append_fname}{rank_suffix}.json"
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(processed_details, f, indent=2)

        except Exception as debug_err:
            log.warning(f"Failed to write VLM debug QA file: {debug_err}")

    def _save_overlay_image(self, overlay_tensor: torch.Tensor, save_path: str) -> None:
        """Save an overlay image tensor to a PNG file.

        Args:
            overlay_tensor: [C, H, W] tensor in range [-1, 1]
            save_path: Path where to save the image
        """
        if overlay_tensor.dim() != 3:
            raise ValueError(f"Expected 3D tensor [C, H, W], got shape {overlay_tensor.shape}")

        # Convert to CPU and detach
        overlay_cpu = overlay_tensor.detach().cpu()

        # Clamp to valid range and convert to [0, 1]
        overlay_cpu = overlay_cpu.clamp(-1.0, 1.0)
        overlay_cpu = (overlay_cpu + 1.0) / 2.0

        # Convert to [0, 255] uint8
        overlay_uint8 = (overlay_cpu * 255.0).round().clamp(0, 255).to(torch.uint8)

        # Convert to numpy and transpose to [H, W, C]
        overlay_np = overlay_uint8.permute(1, 2, 0).numpy()

        # Save using imageio
        try:
            import imageio.v3 as imageio
            imageio.imwrite(save_path, overlay_np)
        except ImportError:
            # Fallback to PIL if imageio is not available
            from PIL import Image
            if overlay_np.shape[2] == 3:  # RGB
                pil_image = Image.fromarray(overlay_np, 'RGB')
            elif overlay_np.shape[2] == 1:  # Grayscale
                pil_image = Image.fromarray(overlay_np.squeeze(2), 'L')
            else:
                raise ValueError(f"Unsupported number of channels: {overlay_np.shape[2]}")
            pil_image.save(save_path)
        

    @distributed.rank0_only  # Only run on the main process.
    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        """Save generated videos produced during validation.

        Args:
            model: The ``ImaginaireModel`` instance (unused).
            data_batch: The input batch used by the model. ``fps`` can be
                optionally specified here.
            output_batch: Contains a list of generated videos under key
                ``"generated_videos"``.
            loss: Validation loss tensor (unused).
            iteration: Current global iteration, used for naming the files.
        """

        # ------------------------------------------------------------------
        # Fetch generated videos.
        # ------------------------------------------------------------------
        if "generated_videos" not in output_batch:
            # Nothing to save.
            return

        generated_videos: List[torch.Tensor] = output_batch["generated_videos"]
        if len(generated_videos) == 0:
            return

        # ------------------------------------------------------------------
        # Determine save directory.
        # ------------------------------------------------------------------
        # Default to checkpoints/<project>/<group>/<name>/val_outputs
        if hasattr(self, "config") and hasattr(self.config, "job"):
            save_root = os.path.join(
                "checkpoints",
                self.config.job.project,
                self.config.job.group,
                self.config.job.name,
                "val_outputs",
            )
        else:
            # Fall-back to current working directory if config is unavailable.
            save_root = os.path.join(os.getcwd(), "val_outputs")

        os.makedirs(save_root, exist_ok=True)

        if "physprop" in output_batch:
            if self.config.dataloader_val.dataset.conditioning_type == "image" or self.config.dataloader_val.dataset.conditioning_type == "image_blob":
                save_physprop_as_image(output_batch["physprop"], save_path=os.path.join(save_root, "physprop.png"), physprop_type=self.config.dataloader_val.dataset.physprop_inputs)
            elif self.config.dataloader_val.dataset.conditioning_type == "fg_bg_vector":
                # Save physprop as text file
                save_physprop_as_text(output_batch["physprop"], save_path=os.path.join(save_root, "physprop.txt"))

        # Frames-per-second: use batch value if present, else default.
        fps_in = data_batch.get("fps", 16)
        if torch.is_tensor(fps_in):
            fps_in = int(fps_in.item())
        fps = int(fps_in) if isinstance(fps_in, (int, float)) else 16

        # ------------------------------------------------------------------
        # Iterate through videos and save them.
        # ------------------------------------------------------------------
        for vid_idx, vid_tensor in enumerate(generated_videos):
            try:
                # Expected shape (1, C, T, H, W) or (C, T, H, W)
                if vid_tensor.dim() == 5 and vid_tensor.shape[0] == 1:
                    vid_tensor = vid_tensor.squeeze(0)

                if vid_tensor.dim() != 4:
                    log.warning(
                        f"[ValidationVideoSaver] Unexpected video tensor shape: {vid_tensor.shape}. Skipping."
                    )
                    continue

                # Convert to (T, H, W, C) uint8 in [0, 255].
                vid_np = vid_tensor.permute(1, 2, 3, 0).clamp(-1, 1)
                vid_np = ((vid_np + 1) / 2 * 255).to(torch.uint8).cpu().numpy()

                save_path = os.path.join(save_root, f"iter_{iteration:07d}_sample_{vid_idx}_{str(uuid.uuid4())[:4]}.mp4")
                save_video(save_path, vid_np, fps=fps)
                log.info(f"[ValidationVideoSaver] Saved generated video to {save_path}")
            except Exception as exc:  # pragma: no cover
                log.error(f"[ValidationVideoSaver] Failed to save video {vid_idx}: {exc}")