# PhyCo public-release: file-deletion proposal

> **Status:** draft for review. Nothing has been deleted yet.
> Tick the checkboxes you agree with, leave a note next to anything you want preserved, and we'll execute the trim from this approved list.

## 1. Why this proposal exists

You're preparing this fork of `cosmos-predict2` for the CVPR 2026 PhyCo public release and want to ship **inference only** — no training, no VLM verification/reward training, no scratch artifacts.

The canonical inference workflow we must preserve is:

```
sbatch scripts/launch/run_physprop_kubric_mpgu.sh \
  --batch_input_json scripts/batch_jsons/benchmark/physiq_controlnet_v1.json \
  --pipeline_config controlnet_multi_24fps_120frames \
  --conditioning_type image_blob --blob_type circle \
  --controlnet_branch_names ctl1,ctl2,ctl3 --active_controlnets ctl1,ctl2,ctl3 \
  --use_lora --dynamic_controlnet  ...
```

Entry chain: `run_physprop_kubric_mpgu.sh` → `examples/physprop_kubric_video2world.py` → star-imports `cosmos_predict2/configs/physprop_conditioned/config_physprop_conditioned.py` → uses `pipelines/physprop_v2w.py` + `models/physprop_v2w_dit.py::PhyspropControlNetDiTMultiple`.

## 2. Verified import edges (so the deletions below are safe)

- `config_physprop_conditioned.py` imports only: `conditioner.py`, `defaults/conditioner.py`, `base/config_video2world.py`, `base/defaults/ema.py`, `models/physprop_v2w_dit.py`, `models/text2image_dit.py`, `tokenizers/tokenizer.py`, `imaginaire.{config, lazy_config}`. **No VLM, no PhysPred, no training experiments.**
- `pipelines/physprop_v2w.py` and `models/physprop_v2w_dit.py` contain zero runtime imports of VLM/physpred (one passing docstring mention only).
- `cosmos_predict2/configs/base/config.py::make_config()` *does* register VLM/physpred training configs unconditionally, but it is only ever called from `scripts/train.py` — which is being deleted with the rest of training.

So: deleting all training+VLM code below does **not** break the inference command.

---

## 3. Files to KEEP (load-bearing for the canonical command)

Listed for completeness so we don't accidentally touch them.

- **Entrypoints:** `scripts/launch/run_physprop_kubric_mpgu.sh`, `examples/physprop_kubric_video2world.py`, `examples/setup_utils.py`
- **Pipeline & model:** `cosmos_predict2/pipelines/{physprop_v2w.py, video2world.py, base.py}`, `cosmos_predict2/models/{physprop_v2w_dit.py, physprop_v2w_model.py, video2world_dit.py, video2world_model.py, text2image_dit.py, utils.py}`
- **Configs (inference-only):** `cosmos_predict2/configs/physprop_conditioned/{config_physprop_conditioned.py, defaults/conditioner.py}`, `cosmos_predict2/configs/base/{config_video2world.py, defaults/ema.py}`
- **Inference data utils:** `cosmos_predict2/data/dataset_utils.py`, `cosmos_predict2/data/kubric_data/kubric_utils.py`
- **Core runtime** (entire dirs): `cosmos_predict2/{auxiliary/{text_encoder.py, cosmos_reason1.py, guardrail/}, conditioner.py, schedulers/, module/, tokenizers/, networks/, utils/, vram_management/, callbacks/, datasets/}`, all of `imaginaire/`
- **Repo plumbing:** `pyproject.toml`, `uv.lock`, `requirements-conda.txt`, `requirements-docker.txt`, `Dockerfile`, `cosmos-predict2.yaml`, `LICENSE`, `ATTRIBUTIONS.md`, `CONTRIBUTING.md`, `README.md`, `.gitignore`, `.python-version`, `ruff.toml`, `bin/`, `justfile`
- **Inference batch JSON:** `scripts/batch_jsons/benchmark/physiq_controlnet_v1.json` (the file your canonical command points at)
- **Setup:** `setup_scripts/{download_checkpoint.sh, setup_env.sh}`

---

## 4. Definite deletions — TRAINING

### 4A. Training entrypoints & launchers
- [x] `scripts/train.py`
- [x] `scripts/train_accel.py`
- [x] `scripts/launch/train_v2w_action_mgpu.sh`
- [x] `scripts/launch/train_v2w_base_mgpu.sh`
- [x] `scripts/launch/train_v2w_ftkubric_mgpu_cloud.sh`
- [x] `scripts/launch/train_v2w_kubric_baseline_mgpu_cloud.sh`
- [x] `scripts/launch/train_v2w_kubric_mgpu.sh`
- [x] `scripts/launch/train_v2w_kubric_mgpu_cloud.sh`
- [x] `scripts/launch/train_v2w_physionpp_mgpu.sh`
- [x] `scripts/launch/train_v2w_physpred_mgpu.sh`
- [x] `scripts/launch/train_v2w_physpred_mse_mgpu.sh`
- [x] `scripts/launch/launch_wisa_sbatch.sh`
- [x] `scripts/launch/launch_wisa_t5_parallel.sh`
- [x] `scripts/launch/generate_simulation_prompts.sh`
- [x] `scripts/launch/setup_kubric_data.sh`
- [x] `scripts/launch/startup_script.sh`
- [x] `setup_scripts/setup_dataset.sh`
- [x] `setup_scripts/update_dataset.sh`
- [x] `setup_scripts/upload_checkpoint.sh`
- [ ] `cosmos_predict2/configs/base/config.py` — training-only registry; delete with `train.py`

### 4B. Training experiment configs
- [x] `cosmos_predict2/configs/physprop_conditioned/experiment/exp.py`
- [x] `cosmos_predict2/configs/physprop_conditioned/experiment/physpred_exp.py` (also VLM)
- [x] `cosmos_predict2/configs/physprop_conditioned/experiment/vlm_exp.py` (also VLM)
- [ ] `cosmos_predict2/configs/base/experiment/` — entire dir (kubric_base_retrain, groot, wisa_base_train, agibot_head_center_fisheye_color, cosmos_nemo_assets, utils)
- [ ] `cosmos_predict2/configs/action_conditioned/experiment/` — entire dir
- [x] `cosmos_predict2/configs/physprop_conditioned/defaults/data.py` — training data registry
- [ ] `cosmos_predict2/configs/base/defaults/{callbacks.py, checkpoint.py, data.py, model.py, optimizer.py, scheduler.py}` — training-only Hydra defaults; **keep `ema.py` only** (used by inference pipeline configs)

### 4C. Training datasets
- [ ] `cosmos_predict2/data/datasets.py`
- [ ] `cosmos_predict2/data/dataset_video.py`
- [ ] `cosmos_predict2/data/webdataloader.py`
- [x] `cosmos_predict2/data/physionpp/` — dir
- [x] `cosmos_predict2/data/wisa_data/` — dir
- [ ] `cosmos_predict2/data/json_data/` — dir
- [x] `cosmos_predict2/data/action_conditioned/` — dir
- [x] `cosmos_predict2/data/kubric_data/kubric_dataset.py` (keep `kubric_utils.py`)
- [ ] `datasets/agibot_head_center_fisheye_color.jsonl` (and the `datasets/` dir if empty after)
- [x] `cosmos_predict2/configs/action_conditioned/` — entire dir (action-conditioned training)
- [ ] `cosmos_predict2/checkpointer.py` — verify usage; delete if only training-side

### 4D. Training-only data prep & annotation
- [ ] `data_annotation/` — entire dir (physiq + wisa annotation web apps)
- [x] `scripts/data_creator/` — entire dir (caption + filename generation)
- [x] `scripts/prepare_agibot_fisheye_data.py`
- [x] `scripts/prompting_to_simulate/` — entire dir
- [x] `scripts/launch/run_kubric_baseline_embeddings.sh`
- [x] `scripts/launch/run_kubric_vllm_example.sh`

### 4E. Dataset-specific T5 embedding scripts (training-only)

**Decision needed:** ship a single generic T5 embedding script so users can encode their own prompts at inference time. Recommendation: keep `scripts/get_t5_embeddings.py`, delete the rest. **Answer: Yes, Keep this.**

- [x] `scripts/get_t5_embeddings_from_cosmos_nemo_assets.py`
- [x] `scripts/get_t5_embeddings_from_groot_dataset.py`
- [x] `scripts/get_t5_embeddings_physionpp.py`
- [x] `scripts/get_t5_embeddings_wisa.py`
- [x] `scripts/get_t5_embeddings_wisa-v2.py`
- [x] `scripts/get_t5_embeddings_wisa-v3.py`
- [ ] `scripts/get_t5_embeddings_prompt.py`
- [ ] `scripts/get_t5_embeddings_prompt_direction.py`
- [ ] `scripts/get_t5_embeddings_json.py`
- [ ] **Keep:** `scripts/get_t5_embeddings.py` (generic — needed at release for users to encode prompts) **Answer: Yes, Keep this.**

### 4F. Post-training documentation
- [ ] `documentations/post-training_video2world.md`
- [ ] `documentations/post-training_video2world_action.md`
- [ ] `documentations/post-training_video2world_agibot_fisheye.md`
- [ ] `documentations/post-training_video2world_cosmos_nemo_assets.md`
- [ ] `documentations/post-training_video2world_gr00t.md`
- [ ] `documentations/post-training_video2world_lora.md`
- [x] `documentations/train_physprop_predictor.md`

---

## 5. Definite deletions — VLM

### 5A. VLM/PhysPred model & training infrastructure
- [x] `cosmos_predict2/models/physprop_v2w_vlm_model.py`
- [x] `cosmos_predict2/models/latent_to_vision_adapter.py`
- [x] `cosmos_predict2/models/physprop_physpred.py` (PhysicalPropertyPredictor)
- [x] `cosmos_predict2/models/physprop_physpred_model.py`
- [x] `cosmos_predict2/pipelines/physprop_physpred_v2w.py`
- [x] `cosmos_predict2/auxiliary/vlm_physprop_verifier.py`
- [x] `cosmos_predict2/configs/physprop_conditioned/defaults/vlm_model.py`
- [x] `cosmos_predict2/configs/physprop_conditioned/defaults/physpred_model.py`

### 5B. VLM evaluation infrastructure
- [x] `vlm_questionnaire/` — entire dir (Q&A banks)
- [x] `scripts/quant_test/` — entire dir (VLM-based eval)
- [x] `scripts/vlm_question_test/` — entire dir (VLM probes)
- [x] `scripts/launch/test_vlm_with_debug_video.sh`
- [x] `scripts/test_vlm_correctness.py`
- [x] `scripts/fix_vlm_config.py`
- [ ] `setup_scripts/download_vlm_checkpoint.sh`
- [ ] `setup_scripts/upload_vlm_checkpoint.sh`
- [x] `examples/physpred_video2world.py` (PhysicalPropertyPredictor inference)
- [x] `examples/physprop_base_lora.py` — verify whether VLM-coupled before deleting

### 5C. VLM documentation
- [x] `documentations/train_vlm_physprop.md`
- [x] `documentations/VLM_IMPLEMENTATION_SUMMARY.md`
- [x] `documentations/VLM_MEMORY_EFFICIENT_TRAINING.md`
- [x] `documentations/VLM_MEMORY_FIX_SUMMARY.md`
- [x] `documentations/VLM_QUICKSTART.md`
- [x] `documentations/VLM_SHAPE_ERROR_DEBUG.md`
- [x] `documentations/README_VLM_CORRECTNESS_TEST.md`
- [x] `documentations/LORA_CHANGES_SUMMARY.md` (dev-scratch — distinct from `LORA_INFERENCE.md` which stays)
- [x] `documentations/LORA_SETUP_COMPLETE.md` (dev-scratch)

---

## 6. Definite deletions — personal / scratch

- [x] `commands.md` — personal command scratchpad
- [x] `quant_eval.md` — personal eval scratchpad
- [x] `stdout.txt` — 618 KB log file
- [x] `download_code_s3.sh`
- [x] `upload_code_s3.sh`
- [x] `documentations/README_kubric_baseline_embeddings.md` (training-only data prep doc)
- [x] `scripts/misc/decode_rle.py` — verify
- [x] `scripts/misc/fix_ownership_from_json.py` — dev tool
- [x] `scripts/misc/merge_lora_physprop_v2w.py` — verify (LoRA merge for training-output checkpoints)
- [x] `scripts/misc/merge_lora_simple.sh` — verify
- [x] `scripts/test_environment.py` — keep if useful for users debugging install; delete if it's personal sanity check (verify)

---

## 7. Decisions you need to make

These items are not strictly tied to "training" or "VLM" but are also not load-bearing for the canonical command. Pick one disposition per group.

### 7A. Other PhyCo / upstream inference variants

These are alternative entry points / launchers in `examples/` and `scripts/launch/`. They keep working without modification but inflate the surface area.

- [ ] `examples/video2world.py` — vanilla NVIDIA video2world (KEEP recommended)
- [x] `examples/text2image.py` — vanilla NVIDIA text2image (DELETE recommended for a focused PhyCo release)
- [x] `examples/text2world.py` — vanilla NVIDIA text2world (DELETE recommended)
- [ ] `examples/video2world_lora.py` — base LoRA inference (KEEP)
- [x] `examples/video2world_lvg.py` — long-video generation (DELETE recommended)
- [x] `examples/video2world_bestofn.py` — best-of-n sampling (DELETE recommended)
- [x] `examples/video2world_gr00t.py` — GR00T post-trained model inference (DELETE recommended; not paper-relevant)
- [x] `examples/action_video2world.py` — action-conditioned inference (DELETE recommended; not paper-relevant)
- [ ] `examples/eval_model_kubric_dataset.py` — Kubric eval harness (KEEP — paper eval)
- [ ] `examples/run_physiq_benchmark.py` — Physics-IQ benchmark generation (KEEP — paper benchmark)
- [x] `examples/physprop_video2world.py` — alternative physprop inference (DELETE if redundant with `physprop_kubric_video2world.py`; verify first)
- [ ] `scripts/launch/run_v2w_mgpu.sh` (KEEP if `examples/video2world.py` stays)
- [ ] `scripts/launch/run_v2w_lora_mgpu.sh` (KEEP if base LoRA stays)
- [x] `scripts/launch/run_lora_inference_example.sh` / `example_lora_inference.sh` (KEEP one)
- [ ] `scripts/launch/run_physprop_v2w_mgpu.sh` (DELETE if `examples/physprop_video2world.py` is removed)
- [ ] `scripts/launch/run_physprop_base_lora.sh` (DELETE if `examples/physprop_base_lora.py` is removed)
- [ ] `scripts/launch/run_eval_kubric_dataset.sh` (KEEP — used by `eval_model_kubric_dataset.py`)
- [ ] `scripts/launch/run_physiq_benchmark_sbatch.sh` (KEEP — paper benchmark)
- [x] `scripts/launch/{set_paths.sh, setup_env_script.sh, create_docker_path_links.sh}` — env helpers (KEEP one canonical, drop the others)

### 7B. Exploratory PhyCo DiT architectures (Phase-2 trim — defer)

You said "for now let's think about the files that are not useful in the release." These are the candidates for a *later* trim once the obvious deletions land. They aren't used by the canonical command but are imported by `config_physprop_conditioned.py`, so deleting them requires paired removal of their config blocks.

Default: **KEEP for now**, revisit later.
**Answer: Yes, Let's revisit this later.**

- [ ] `PhyspropConditionedMinimalV1LVGDiT` — base concat conditioning
- [ ] `PhyspropConditionedCrossAttnDiT` — cross-attention variant
- [ ] `PhyspropConditionedZeroConvDiT` — zero-conv variant
- [ ] `PhyspropConditionedTokenizerDiT` — frozen-tokenizer variant
- [ ] `PhyspropZeroConvDepthConditionedDiT` — depth conditioning
- [ ] `PhyspropControlNetDiT` — single-branch ControlNet v1
- [ ] `PhyspropControlNetDiTV2` — single-branch ControlNet v2 (image-blob)
- [x] `PhyspropControlNetDiTVector` — vector ControlNet
- [ ] `PhyspropControlNetDiTMultiple` — **canonical, must keep** **Answer: Yes, Keep this.**

### 7C. Personal batch JSONs

`scripts/batch_jsons/` contains 50+ JSONs. Most are personal test scenes (curling, ball drops, jenga, plexels, etc.). Recommendation: **delete all except `scripts/batch_jsons/benchmark/`** (Physics-IQ + paper benchmark inputs).
**Answer: Let's keep all the batch JSONs.**

- [ ] Delete: `scripts/batch_jsons/*.json` at the top level
- [ ] Keep: `scripts/batch_jsons/benchmark/*.json` (review individually if too many — VLM-coupled `object_drop_vlm-v*.json` should also go)

### 7D. Upstream NVIDIA inference docs

These are not training docs — they are inference user guides written by NVIDIA. Default: **KEEP** (useful to users).

- [x] `documentations/setup.md`
- [x] `documentations/performance.md`
- [x] `documentations/inference_text2image.md`
- [x] `documentations/inference_text2world.md`
- [x] `documentations/inference_video2world.md`
- [x] `documentations/LORA_INFERENCE.md`
- [x] `documentations/LORA_QUICK_REFERENCE.md`
- [x] `documentations/controlnet_multi_branch.md`

---

## 8. Risks & gotchas to keep in mind during execution

1. **`config.py::make_config()` & `import_all_modules_from_package`** — if we delete experiment files but keep `config.py`, the dynamic imports break. Resolution: delete `config.py` entirely as part of the training removal. It is only imported by `scripts/train.py`/`train_accel.py`, both of which are also being deleted.
2. **Star import in `physprop_kubric_video2world.py`** — any DiT class deletion under §7B requires paired removal of its `_NET_2B_*` and `_PIPELINE_2B_*` blocks in `config_physprop_conditioned.py`.
3. **`imaginaire/trainer.py`** — referenced by `scripts/train.py`. Should still keep `imaginaire/` intact (some modules are used by inference); just don't run training. Verify before pruning anything inside `imaginaire/`.
4. **`vram_management/`, `callbacks/`** — mixed training+inference utilities. Don't prune individual files; keep the dirs intact.
5. **`cosmos_predict2/configs/base/defaults/ema.py`** — must stay (used by inference pipeline configs). The other files in that dir are training-only and can go.

## 9. Verification (after the trim is executed)

**Answer: This pc doesn't have the environment set up to run the verification. You can try but won't work mostly I guess.**

```bash
# Imports still resolve
python -c "from cosmos_predict2.configs.physprop_conditioned.config_physprop_conditioned import *"
python -c "from cosmos_predict2.pipelines.physprop_v2w import PhyspropConditionedVideo2WorldPipeline"

# Launch script still loads
bash scripts/launch/run_physprop_kubric_mpgu.sh --help

# Lint catches dangling imports
ruff check cosmos_predict2 examples scripts

# End-to-end smoke test (1 split)
sbatch --array 0 ... scripts/launch/run_physprop_kubric_mpgu.sh \
  --batch_input_json scripts/batch_jsons/benchmark/physiq_controlnet_v1.json \
  --num_splits 1 --total_seconds 5.0  ...
```

## 10. Approximate impact

| Bucket | Files | Notes |
|---|---|---|
| Training code (4A–4D) | ~60 files + 6 dirs | Including `data_annotation/`, `scripts/data_creator/`, training launchers, training experiments, training data |
| T5 scripts (4E) | 9 of 10 | Keep one generic |
| Post-training docs (4F) | 7 files | |
| VLM code (5A–5C) | ~25 files + 2 dirs | Including `vlm_questionnaire/`, `scripts/quant_test/`, `scripts/vlm_question_test/` |
| Personal/scratch (§6) | ~10 files | Including 618 KB `stdout.txt` |
| Phase-2 (§7B) | TBD | Deferred |

Net repo footprint reduction: roughly half the files, including the largest non-data binary (`stdout.txt`).


## User response

Btw, I don't want to delete the base training codebase for the base model that was already there in the original cosmos-predict2 repo https://github.com/nvidia-cosmos/cosmos-predict2 . I want to delete any additional training code that was added by me other useless training related stuff like Groot, agibot, etc.