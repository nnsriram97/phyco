# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Script to automate Physics-IQ benchmark generation with Cosmos Predict2.
#
# The benchmark expects each scenario to be generated from its switch-frame
# image (and optional description prompt) and saved under `.model_name/`.
# This helper loads the dataset metadata, runs the text-only baseline model,
# optionally chains multiple generations to reach 5-second outputs, and writes
# the merged results using the required naming scheme.

import argparse
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from tqdm import tqdm

from imaginaire.utils import log
from imaginaire.utils.io import save_image_or_video

from examples.physprop_base_lora import (
    _DEFAULT_NEGATIVE_PROMPT,
    cleanup_distributed,
    setup_pipeline,
)


@dataclass
class BenchmarkCase:
    scenario: str
    description: str
    category: str
    generated_name: str
    switch_frame: Path

    @property
    def uid(self) -> str:
        return self.scenario.split("_", 1)[0]


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    """Mirror the relevant arguments from physprop_base_lora."""

    parser.add_argument("--model_size", choices=["2B", "14B"], default="2B")
    parser.add_argument("--resolution", choices=["480", "720"], default="480", type=str)
    parser.add_argument("--fps", choices=[10, 16], default=10, type=int)
    parser.add_argument("--state_t", type=int, default=None)
    parser.add_argument("--output_fps", type=int, default=24, help="FPS for the final saved video.")
    parser.add_argument("--dit_path", type=str, default="")

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
    )
    parser.add_argument("--init_lora_weights", action="store_true", default=True)

    parser.add_argument("--negative_prompt", type=str, default=_DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--num_conditional_frames", type=int, default=1, choices=[1, 5])
    parser.add_argument("--guidance", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--disable_guardrail", action="store_true", default=True)
    parser.add_argument("--offload_guardrail", action="store_true", default=True)
    parser.add_argument("--disable_prompt_refiner", action="store_true", default=True)
    parser.add_argument("--offload_prompt_refiner", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--append_fname", type=str, default="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Physics-IQ benchmark submissions using the PhysProp baseline."
    )

    # Benchmark-specific arguments.
    parser.add_argument(
        "--benchmark_root",
        type=Path,
        required=True,
        help="Root of the Physics-IQ benchmark checkout (containing 'physics-IQ-benchmark').",
    )
    parser.add_argument(
        "--switch_frames_dir",
        type=Path,
        default=None,
        help="Optional override for the directory containing switch-frame JPGs.",
    )
    parser.add_argument(
        "--descriptions_csv",
        type=Path,
        default=None,
        help="Optional override for the descriptions CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where `.model_name` lives (e.g., /path/to/.cosmos_predict2_text).",
    )
    parser.add_argument(
        "--filename_suffix",
        type=str,
        default="cosmos-predict2",
        help="Suffix appended to each video filename when `--use_generated_names` is not set.",
    )
    parser.add_argument(
        "--use_generated_names",
        action="store_true",
        help="Use the `generated_video_name` column from the CSV verbatim.",
    )
    parser.add_argument(
        "--scenario_ids",
        nargs="+",
        default=None,
        help="Scenario ID filters (supports comma lists and ranges, e.g., 0001 4-12 30,32).",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=1,
        help="Total number of equal partitions to slice the benchmark into (for distributed runs).",
    )
    parser.add_argument(
        "--split_index",
        type=int,
        default=0,
        help="Zero-based index of the split to process when --num_splits > 1.",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Limit the number of benchmark cases to process.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation when the target mp4 already exists.",
    )
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        default="",
        help="Optional text prepended to each description prompt.",
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="",
        help="Optional text appended to each description prompt.",
    )
    parser.add_argument(
        "--prompt_jsonl",
        type=Path,
        default=None,
        help="Optional JSONL file produced by generate_physiq_prompts_vllm.py to override scenario descriptions.",
    )
    parser.add_argument(
        "--target_seconds",
        type=float,
        default=5.0,
        help="Minimum clip duration in seconds to reach after stitching segments.",
    )
    parser.add_argument(
        "--min_total_frames",
        type=int,
        default=None,
        help="Override target_seconds by requesting an explicit total frame count.",
    )
    parser.add_argument(
        "--max_passes",
        type=int,
        default=1,
        help="Maximum number of chained generations per scenario.",
    )
    parser.add_argument(
        "--keep_followup_first_frame",
        action="store_true",
        help="Keep the first frame of chained generations instead of dropping duplicates.",
    )
    parser.add_argument(
        "--duplicate_last_frame",
        action="store_true",
        help="Pad outputs to the required duration by repeating the last frame when necessary.",
    )
    parser.add_argument(
        "--reseed_each_pass",
        action="store_true",
        help="Use a different same random seed for every chained generation (default is to not reseed).",
        default=False,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List the benchmark cases that would run without invoking the model.",
    )

    _add_common_model_args(parser)
    return parser


def resolve_benchmark_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Return (switch_frames_dir, descriptions_csv)."""

    switch_frames_dir = (
        args.switch_frames_dir
        if args.switch_frames_dir is not None
        else args.benchmark_root / "physics-IQ-benchmark" / "switch-frames"
    )
    descriptions_csv = (
        args.descriptions_csv
        if args.descriptions_csv is not None
        else args.benchmark_root / "descriptions" / "descriptions.csv"
    )

    if not switch_frames_dir.exists():
        raise FileNotFoundError(f"Switch frames directory not found: {switch_frames_dir}")
    if not descriptions_csv.exists():
        raise FileNotFoundError(f"Descriptions CSV not found: {descriptions_csv}")

    return switch_frames_dir, descriptions_csv


def load_cases(
    descriptions_csv: Path,
    switch_frames_dir: Path,
    include_ids: Optional[set[str]],
) -> List[BenchmarkCase]:
    cases: List[BenchmarkCase] = []
    with descriptions_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = row["scenario"]
            uid = scenario.split("_", 1)[0]
            if include_ids is not None and uid not in include_ids:
                continue

            switch_candidates = sorted(switch_frames_dir.glob(f"{uid}_switch-frames_anyFPS*.jpg"))
            if not switch_candidates:
                log.warning(f"No switch-frame found for scenario {scenario}, skipping")
                continue

            case = BenchmarkCase(
                scenario=scenario,
                description=row.get("description", ""),
                category=row.get("category", ""),
                generated_name=row.get("generated_video_name", f"{uid}_prediction.mp4"),
                switch_frame=switch_candidates[0],
            )
            cases.append(case)

    cases.sort(key=lambda c: c.uid)
    return cases


def slice_cases_for_split(cases: List[BenchmarkCase], num_splits: int, split_index: int) -> List[BenchmarkCase]:
    if num_splits <= 0:
        raise ValueError("--num_splits must be >= 1")
    if split_index < 0 or split_index >= num_splits:
        raise ValueError("--split_index must be in [0, num_splits)")
    if num_splits == 1 or not cases:
        return cases
    chunk = math.ceil(len(cases) / num_splits)
    start = chunk * split_index
    end = min(start + chunk, len(cases))
    return cases[start:end]


def parse_scenario_filters(raw_tokens: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not raw_tokens:
        return None
    ids: set[str] = set()
    for token in raw_tokens:
        for piece in token.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if "-" in piece:
                start_str, end_str = piece.split("-", 1)
                try:
                    start = int(start_str, 10)
                    end = int(end_str, 10)
                except ValueError as err:
                    raise ValueError(f"Invalid scenario range: {piece}") from err
                if start > end:
                    start, end = end, start
                for val in range(start, end + 1):
                    ids.add(f"{val:04d}")
            else:
                ids.add(piece.zfill(4))
    return ids


def load_prompt_overrides(jsonl_path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not jsonl_path.exists():
        log.warning(f"Prompt override file not found: {jsonl_path}")
        return mapping
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning(f"Skipping malformed JSON line in {jsonl_path}")
                continue
            scenario = record.get("scenario")
            prompt = record.get("vlm_prompt") or record.get("prompt") or record.get("description")
            if scenario and prompt:
                mapping[scenario] = prompt.strip()
    log.info(f"Loaded {len(mapping)} prompt overrides from {jsonl_path}")
    return mapping


def determine_output_fps(args: argparse.Namespace, pipe) -> int:
    if args.output_fps:
        return args.output_fps
    return 10 if pipe.config.state_t == 16 else 16


def concat_clips(clips: List[torch.Tensor], drop_followup_first: bool) -> torch.Tensor:
    processed = []
    for idx, clip in enumerate(clips):
        clip = clip if clip.ndim == 4 else clip[0]
        if idx > 0 and drop_followup_first and clip.shape[1] > 1:
            processed.append(clip[:, 1:, :, :])
        else:
            processed.append(clip)
    return torch.cat(processed, dim=1)


def count_total_frames(clips: List[torch.Tensor], drop_followup_first: bool) -> int:
    total = 0
    for idx, clip in enumerate(clips):
        frames = clip.shape[2] if clip.ndim == 5 else clip.shape[1]
        if idx > 0 and drop_followup_first and frames > 1:
            frames -= 1
        total += frames
    return total


def tensorize_video(video: torch.Tensor) -> torch.Tensor:
    if video is None:
        raise RuntimeError("Pipeline returned None.")
    tensor = video.detach()
    if tensor.ndim == 5:
        tensor = tensor[0]
    return tensor.to(dtype=torch.float32, device="cpu")


def save_last_frame_as_image(clip: torch.Tensor, tmp_dir: Path, prefix: str, pass_idx: int) -> Path:
    frame = clip[:, -1:, :, :] if clip.ndim == 4 else clip[0, :, -1:, :, :]
    tmp_path = tmp_dir / f"{prefix}_pass{pass_idx:02d}.jpg"
    save_image_or_video(frame, str(tmp_path))
    return tmp_path


def pad_with_last_frame(tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
    current = tensor.shape[1]
    if current >= target_frames:
        return tensor
    needed = target_frames - current
    last_frame = tensor[:, -1:, :, :]
    padding = last_frame.repeat(1, needed, 1, 1)
    return torch.cat([tensor, padding], dim=1)


def build_prompt(prefix: str, description: str, suffix: str) -> str:
    prompt_parts = [prefix.strip(), description.strip(), suffix.strip()]
    return " ".join(filter(None, prompt_parts))


def required_frames(args: argparse.Namespace, fps: int) -> int:
    if args.min_total_frames:
        return args.min_total_frames
    if args.target_seconds <= 0:
        return fps  # fallback to a single second if misconfigured
    return math.ceil(args.target_seconds * fps)

def clip_to_target_frames(tensor: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Truncate video tensor to exactly target_frames."""
    current = tensor.shape[1]
    if current <= target_frames:
        return tensor
    return tensor[:, :target_frames, :, :]

def run_case(
    pipe,
    args: argparse.Namespace,
    case: BenchmarkCase,
    target_frames: int,
    output_path: Path,
) -> bool:
    prompt = build_prompt(args.prompt_prefix, case.description, args.prompt_suffix)
    clips: List[torch.Tensor] = []
    temp_dir = Path(tempfile.mkdtemp(prefix=f"physiq_{case.uid}_"))
    temp_files: List[Path] = []
    input_path = str(case.switch_frame)
    success = False

    try:
        for pass_idx in range(args.max_passes):
            seed = args.seed + pass_idx if args.reseed_each_pass else args.seed
            video = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                input_path=input_path,
                num_conditional_frames=args.num_conditional_frames,
                guidance=args.guidance,
                seed=seed,
            )
            clip = tensorize_video(video)
            clips.append(clip)

            total_frames = count_total_frames(clips, not args.keep_followup_first_frame)
            log.info(f"{case.uid}: pass {pass_idx + 1} produced {clip.shape[1]} frames (total {total_frames})")
            if total_frames >= target_frames:
                success = True
                break
            if pass_idx == args.max_passes - 1:
                log.warning(f"{case.uid}: reached max passes ({args.max_passes}) without hitting {target_frames} frames")
                break

            next_input = save_last_frame_as_image(clip, temp_dir, case.uid, pass_idx)
            temp_files.append(next_input)
            input_path = str(next_input)

        if clips:
            merged = concat_clips(clips, drop_followup_first=not args.keep_followup_first_frame)
            effective_frames = merged.shape[1]
            if effective_frames < target_frames:
                if args.duplicate_last_frame:
                    merged = pad_with_last_frame(merged, target_frames)
                    log.info(
                        f"{case.uid}: duplicated last frame {target_frames - effective_frames} times to reach {target_frames} frames"
                    )
                    success = True
                else:
                    log.warning(
                        f"{case.uid}: only {effective_frames} frames generated (< {target_frames}); consider --duplicate_last_frame"
                    )
            elif effective_frames > target_frames:
                merged = clip_to_target_frames(merged, target_frames)
                log.info(f"{case.uid}: truncated from {effective_frames} to {target_frames} frames")
                success = True
            save_image_or_video(merged, str(output_path), fps=determine_output_fps(args, pipe))
            if success:
                log.success(f"{case.uid}: saved merged video -> {output_path}")
            else:
                log.warning(f"{case.uid}: saved {count_total_frames(clips, not args.keep_followup_first_frame)} frames (< {target_frames}) -> {output_path}")
        else:
            log.error(f"{case.uid}: pipeline produced no frames")
    finally:
        for tmp in temp_files:
            try:
                os.remove(tmp)
            except FileNotFoundError:
                pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass
    return success


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    switch_frames_dir, descriptions_csv = resolve_benchmark_paths(args)
    include_ids = parse_scenario_filters(args.scenario_ids)
    cases = load_cases(descriptions_csv, switch_frames_dir, include_ids)
    if args.prompt_jsonl:
        overrides = load_prompt_overrides(args.prompt_jsonl)
        applied = 0
        if overrides:
            for case in cases:
                if case.scenario in overrides:
                    case.description = overrides[case.scenario]
                    applied += 1
            log.info(f"Applied prompt overrides for {applied} / {len(cases)} cases.")
    total_cases = len(cases)
    cases = slice_cases_for_split(cases, args.num_splits, args.split_index)

    if not cases:
        log.error(
            f"No cases assigned to split {args.split_index} (num_splits={args.num_splits}, total cases={total_cases})."
        )
        return

    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    if not cases:
        log.error("No benchmark cases matched the provided filters.")
        return
    if args.num_splits > 1:
        log.info(
            f"Split {args.split_index + 1}/{args.num_splits}: processing {len(cases)} of {total_cases} total cases."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dry_run:
        for case in cases:
            log.info(f"[DRY RUN] {case.uid}: {case.generated_name} (switch: {case.switch_frame})")
        return

    pipe = setup_pipeline(args)
    fps = determine_output_fps(args, pipe)
    target_frames = required_frames(args, fps)

    completed = 0
    skipped = 0
    failures = 0

    if args.append_fname and os.path.sep in args.append_fname:
        log.warning("--append_fname contains path separators; they will become part of the saved filename.")

    try:
        for case in tqdm(cases, desc="Physics-IQ cases"):
            output_name = (
                case.generated_name
                if args.use_generated_names
                else f"{case.uid}_{args.filename_suffix}.mp4"
            )
            filename = f"{output_name}{args.append_fname}"
            output_path = args.output_dir / filename
            if args.skip_existing and output_path.exists():
                skipped += 1
                log.info(f"{case.uid}: skipping existing file {output_path}")
                continue

            try:
                if run_case(pipe, args, case, target_frames, output_path):
                    completed += 1
                else:
                    failures += 1
            except Exception as exc:
                failures += 1
                log.exception(f"{case.uid}: generation failed with error {exc}")
    finally:
        cleanup_distributed()

    log.info(
        f"Finished Physics-IQ export -> completed: {completed}, skipped: {skipped}, failures: {failures}, target fps: {fps}, target frames: {target_frames}"
    )


if __name__ == "__main__":
    main()
