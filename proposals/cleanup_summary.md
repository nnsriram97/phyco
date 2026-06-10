# PhyCo public-release cleanup — what was actually removed

> **Status:** executed. This document records the trim that was performed against `release_cleanup.md`.
> Stats: **140 files deleted, ~52,800 LOC removed, 6 files edited.** Everything is recoverable from git history.

## Anchor

The single inference path preserved end-to-end is the canonical Physics-IQ benchmark generation:

```
scripts/launch/run_physprop_kubric_mpgu.sh
  → examples/physprop_kubric_video2world.py
  → cosmos_predict2/configs/physprop_conditioned/config_physprop_conditioned.py (star-imported)
  → cosmos_predict2/pipelines/physprop_v2w.py (PhyspropConditionedVideo2WorldPipeline)
  → cosmos_predict2/models/physprop_v2w_dit.py::PhyspropControlNetDiTMultiple
```

with `--pipeline_config controlnet_multi_24fps_120frames`, `--conditioning_type image_blob`, `--use_lora`, multi-branch ControlNet.

Upstream NVIDIA training infrastructure (`scripts/train.py`, `imaginaire/`, `cosmos_predict2/configs/base/{config.py, defaults/, experiment/cosmos_nemo_assets.py}`, action-conditioned post-training) is **kept intact** per the user's overriding instruction.

---

## 1. Deletions

### 1A. PhyCo training (added during PhyCo work)

| Path | Why removed |
|---|---|
| `scripts/launch/train_v2w_action_mgpu.sh` | PhyCo training launcher |
| `scripts/launch/train_v2w_base_mgpu.sh` | PhyCo training launcher |
| `scripts/launch/train_v2w_ftkubric_mgpu_cloud.sh` | PhyCo training launcher |
| `scripts/launch/train_v2w_kubric_baseline_mgpu_cloud.sh` | PhyCo training launcher |
| `scripts/launch/train_v2w_kubric_mgpu.sh` | PhyCo training launcher |
| `scripts/launch/train_v2w_kubric_mgpu_cloud.sh` | PhyCo training launcher |
| `scripts/launch/train_v2w_physionpp_mgpu.sh` | PhyCo training launcher |
| `scripts/launch/train_v2w_physpred_mgpu.sh` | PhyCo training launcher |
| `scripts/launch/train_v2w_physpred_mse_mgpu.sh` | PhyCo training launcher |
| `scripts/launch/launch_wisa_sbatch.sh` | WISA data prep |
| `scripts/launch/launch_wisa_t5_parallel.sh` | WISA T5 sharding |
| `scripts/launch/generate_simulation_prompts.sh` | LLM prompt-gen helper |
| `scripts/launch/setup_kubric_data.sh` | Kubric data prep |
| `scripts/launch/startup_script.sh` | Personal env setup |
| `scripts/launch/run_kubric_baseline_embeddings.sh` | Kubric embedding generation |
| `scripts/launch/run_kubric_vllm_example.sh` | vLLM caption example |
| `setup_scripts/setup_dataset.sh` | Dataset bootstrap |
| `setup_scripts/update_dataset.sh` | Dataset sync |
| `setup_scripts/upload_checkpoint.sh` | Checkpoint upload |
| `cosmos_predict2/configs/physprop_conditioned/experiment/exp.py` | PhyCo training experiments (16 variants) |
| `cosmos_predict2/configs/physprop_conditioned/experiment/physpred_exp.py` | PhysPred training experiments |
| `cosmos_predict2/configs/physprop_conditioned/experiment/vlm_exp.py` | VLM training experiments |
| `cosmos_predict2/configs/physprop_conditioned/defaults/data.py` | PhyCo training data registry |
| `cosmos_predict2/data/physionpp/physionpp_dataset.py` | Physion++ training dataset |
| `cosmos_predict2/data/wisa_data/wisa_dataset.py` | WISA training dataset |
| `scripts/data_creator/` (entire dir, 18 files) | Caption + filename + stats generation |
| `scripts/prompting_to_simulate/` (entire dir) | LLM-prompted simulation generation |
| `scripts/prepare_agibot_fisheye_data.py` | AgiBot data prep |

### 1B. Useless upstream experiments (per user comment: GR00T, agibot, etc.)

These were upstream NVIDIA cosmos-predict2 post-training case studies but were marked "useless" for the PhyCo release.

| Path | Why removed |
|---|---|
| `cosmos_predict2/configs/base/experiment/groot.py` | GR00T post-training |
| `cosmos_predict2/configs/base/experiment/agibot_head_center_fisheye_color.py` | AgiBot post-training |
| `cosmos_predict2/configs/base/experiment/wisa_base_train.py` | WISA post-training |
| `cosmos_predict2/configs/base/experiment/kubric_base_retrain.py` | PhyCo Kubric retrain |
| `datasets/agibot_head_center_fisheye_color.jsonl` | AgiBot data manifest |

Surviving upstream experiment: `cosmos_nemo_assets.py` (small example dataset).

### 1C. VLM rollout + PhysicalPropertyPredictor stack

| Path | Why removed |
|---|---|
| `cosmos_predict2/models/physprop_v2w_vlm_model.py` | VLM-rollout training model |
| `cosmos_predict2/models/latent_to_vision_adapter.py` | Latent→pseudo-video adapter for VLM |
| `cosmos_predict2/models/physprop_physpred.py` | PhysicalPropertyPredictor |
| `cosmos_predict2/models/physprop_physpred_model.py` | Predictor training wrapper |
| `cosmos_predict2/pipelines/physprop_physpred_v2w.py` | Predictor inference pipeline |
| `cosmos_predict2/auxiliary/vlm_physprop_verifier.py` | Qwen2.5-VL reward model |
| `cosmos_predict2/configs/physprop_conditioned/defaults/vlm_model.py` | VLM training config registry |
| `cosmos_predict2/configs/physprop_conditioned/defaults/physpred_model.py` | Predictor training config registry |
| `examples/physpred_video2world.py` | Predictor inference entry point |
| `examples/physprop_base_lora.py` | VLM-coupled LoRA inference |

### 1D. VLM evaluation infrastructure

| Path | Why removed |
|---|---|
| `vlm_questionnaire/` (14 JSON banks) | VLM Q&A question banks |
| `scripts/quant_test/` (7 files) | VLM-based quantitative evaluation |
| `scripts/vlm_question_test/` (6 files) | Per-VLM probe scripts (Qwen2.5/2-VL, Gemma3, LLaVA) |
| `scripts/launch/test_vlm_with_debug_video.sh` | VLM debug launcher |
| `scripts/test_vlm_correctness.py` | VLM sanity check |
| `scripts/fix_vlm_config.py` | VLM checkpoint config patcher |

### 1E. Removed PhyCo DiT variant

The user marked one Phase-2 architecture for immediate removal:

| Path/Symbol | Why removed |
|---|---|
| `PhyspropControlNetDiTVector` class in `models/physprop_v2w_dit.py` | Vector-input ControlNet variant; not used by canonical command |
| `PHYS_PROP_CONTROLNET_PREDICT2_VIDEO2WORLD_NET_2B_VECTOR` config | Paired removal |
| `PHYS_PROP_CONTROLNET_PREDICT2_VIDEO2WORLD_PIPELINE_2B_VECTOR` config | Paired removal |
| `PHYS_PROP_CONTROLNET_PREDICT2_V2W_2B_FSDP_CONFIG_VECTOR` + its `cs.store(...)` registration | Paired removal in `defaults/model.py` |
| `vector_controlnet_12fps_37frames` branch in `examples/physprop_kubric_video2world.py` | Paired removal |

**Other exploratory DiT variants are kept for now** (Phase-2 trim deferred, per user "Let's revisit this later"): `PhyspropConditionedMinimalV1LVGDiT`, `PhyspropConditionedCrossAttnDiT`, `PhyspropConditionedZeroConvDiT`, `PhyspropConditionedTokenizerDiT`, `PhyspropZeroConvDepthConditionedDiT`, `PhyspropControlNetDiT`, `PhyspropControlNetDiTV2`.

### 1F. Alternative inference variants

| Path | Why removed |
|---|---|
| `examples/text2image.py` | Vanilla NVIDIA text2image |
| `examples/text2world.py` | Vanilla NVIDIA text2world |
| `examples/video2world_lvg.py` | Long-video generation variant |
| `examples/video2world_bestofn.py` | Best-of-N sampling |
| `examples/video2world_gr00t.py` | GR00T post-trained inference |
| `examples/action_video2world.py` | Action-conditioned inference |
| `examples/physprop_video2world.py` | Alternative physprop inference (redundant with `physprop_kubric_video2world.py`) |
| `scripts/launch/run_physprop_v2w_mgpu.sh` | Orphaned (its example was deleted) |
| `scripts/launch/run_physprop_base_lora.sh` | Orphaned |
| `scripts/launch/run_physpred_v2w_mgpu.sh` | Orphaned |
| `scripts/launch/run_lora_inference_example.sh` | Duplicate of `example_lora_inference.sh` |
| `scripts/launch/example_lora_inference.sh` | Same as above |
| `scripts/launch/set_paths.sh`, `setup_env_script.sh`, `create_docker_path_links.sh` | Personal env helpers (replaced by `setup_scripts/setup_env.sh`) |

**Surviving inference launchers** (kept):
- `scripts/launch/run_physprop_kubric_mpgu.sh` — canonical command
- `scripts/launch/run_v2w_mgpu.sh` — vanilla NVIDIA video2world
- `scripts/launch/run_v2w_lora_mgpu.sh` — base LoRA inference
- `scripts/launch/run_eval_kubric_dataset.sh` — Kubric eval
- `scripts/launch/run_physiq_benchmark_sbatch.sh` — Physics-IQ benchmark

### 1G. Documentation

| Path | Why removed |
|---|---|
| `documentations/train_physprop_predictor.md` | PhysicalPropertyPredictor training |
| `documentations/train_vlm_physprop.md` | VLM-rollout training |
| `documentations/VLM_IMPLEMENTATION_SUMMARY.md` | VLM development log |
| `documentations/VLM_MEMORY_EFFICIENT_TRAINING.md` | VLM dev log |
| `documentations/VLM_MEMORY_FIX_SUMMARY.md` | VLM dev log |
| `documentations/VLM_QUICKSTART.md` | VLM quickstart |
| `documentations/VLM_SHAPE_ERROR_DEBUG.md` | VLM debug notes |
| `documentations/README_VLM_CORRECTNESS_TEST.md` | VLM correctness test docs |
| `documentations/LORA_CHANGES_SUMMARY.md` | LoRA dev notes |
| `documentations/LORA_SETUP_COMPLETE.md` | LoRA dev notes |
| `documentations/README_kubric_baseline_embeddings.md` | Training data prep |
| `documentations/setup.md` | Upstream NVIDIA setup guide |
| `documentations/performance.md` | Upstream NVIDIA perf guide |
| `documentations/inference_text2image.md` | Upstream NVIDIA inference docs |
| `documentations/inference_text2world.md` | Upstream NVIDIA inference docs |
| `documentations/inference_video2world.md` | Upstream NVIDIA inference docs |
| `documentations/LORA_INFERENCE.md` | Upstream NVIDIA LoRA inference docs |
| `documentations/LORA_QUICK_REFERENCE.md` | Upstream NVIDIA LoRA quick ref |
| `documentations/controlnet_multi_branch.md` | Upstream NVIDIA multi-branch ControlNet docs |

**Surviving docs** (`documentations/` after cleanup):
- `post-training_video2world.md`
- `post-training_video2world_action.md`
- `post-training_video2world_agibot_fisheye.md`
- `post-training_video2world_cosmos_nemo_assets.md`
- `post-training_video2world_gr00t.md`
- `post-training_video2world_lora.md`

### 1H. Dataset-specific T5 embedding scripts

Kept the generic `scripts/get_t5_embeddings.py` so users can encode their own prompts. Removed dataset-specific variants:

- `scripts/get_t5_embeddings_from_cosmos_nemo_assets.py`
- `scripts/get_t5_embeddings_from_groot_dataset.py`
- `scripts/get_t5_embeddings_physionpp.py`
- `scripts/get_t5_embeddings_wisa.py`, `_wisa-v2.py`, `_wisa-v3.py`

### 1I. Personal / scratch artifacts

| Path | Why removed |
|---|---|
| `commands.md` | Personal command scratchpad |
| `quant_eval.md` | Personal eval scratchpad |
| `stdout.txt` (618 KB) | Stale log file |
| `download_code_s3.sh`, `upload_code_s3.sh` | Personal S3 sync helpers |
| `scripts/misc/decode_rle.py` | Dev utility |
| `scripts/misc/fix_ownership_from_json.py` | Dev utility |
| `scripts/misc/merge_lora_physprop_v2w.py` | LoRA-merge dev tool |
| `scripts/misc/merge_lora_simple.sh` | LoRA-merge dev tool |
| `scripts/test_environment.py` | Personal sanity check |

---

## 2. Edits to keep imports clean

| File | Edit |
|---|---|
| `cosmos_predict2/configs/base/config.py` | Removed imports of deleted `physpred_model`, `vlm_model`, `register_training_and_val_data_physionpp`; removed their `register_*()` calls; removed the `import_all_modules_from_package(...physprop_conditioned.experiment)` call |
| `cosmos_predict2/configs/physprop_conditioned/defaults/model.py` | Removed `PIPELINE_2B_VECTOR` import, the `FSDP_CONFIG_VECTOR` dict, and its `cs.store(...)` registration |
| `cosmos_predict2/configs/physprop_conditioned/config_physprop_conditioned.py` | Removed `PhyspropControlNetDiTVector` import, `_NET_2B_VECTOR` config, `_PIPELINE_2B_VECTOR` config |
| `cosmos_predict2/models/physprop_v2w_dit.py` | Removed the `PhyspropControlNetDiTVector` class body and a stale docstring reference to `physprop_physpred.py` |
| `examples/physprop_kubric_video2world.py` | Removed the `vector_controlnet_12fps_37frames` `--pipeline_config` branch |
| `CLAUDE.md` | Rewritten to describe the post-cleanup repo |

---

## 3. Reconciliations applied (worth flagging)

These are places where the literal checkboxes in `release_cleanup.md` conflicted with the user's trailing comment ("don't delete the base training codebase... only PhyCo additions and useless stuff like GR00T/agibot"). The trailing comment took precedence:

1. **`scripts/train.py`, `scripts/train_accel.py`** were ticked but **kept** — they're upstream training entrypoints.
2. **`cosmos_predict2/configs/base/config.py`** was suggested for deletion in §4A but **kept** — it's the upstream training entrypoint that users may need.
3. **`cosmos_predict2/configs/base/defaults/{callbacks,checkpoint,data,model,optimizer,scheduler}.py`** — **kept** as upstream training defaults.
4. **`cosmos_predict2/data/{datasets.py, dataset_video.py, webdataloader.py, json_data/, action_conditioned/}`** — **kept** (upstream dataset infra).
5. **`cosmos_predict2/configs/action_conditioned/`** — **kept** (upstream action-conditioned post-training).
6. **`cosmos_predict2/data/kubric_data/kubric_dataset.py`** was ticked for deletion, but `examples/eval_model_kubric_dataset.py` (which was kept) imports `KubricDataset_v2` from it. **Restored** so eval works. Delete both together if you want eval gone.
7. **`data_annotation/`** was ticked-OFF and **kept**. Same for `cosmos_predict2/checkpointer.py`, `setup_scripts/{download,upload}_vlm_checkpoint.sh`, and `scripts/get_t5_embeddings_{prompt,prompt_direction,json}.py`.

The user should review §3 items 6 and 7 if they want a tighter trim.

---

## 4. Verification done before commit

```bash
# Syntax (AST) check on every edited file — all pass
python3 -c "import ast; ast.parse(open('<file>').read())"

# Bash syntax on the canonical launcher — passes
bash -n scripts/launch/run_physprop_kubric_mpgu.sh

# No remaining references to any deleted symbol/module across the repo
grep -rn 'physprop_physpred|physprop_v2w_vlm|latent_to_vision_adapter|vlm_physprop_verifier|PhyspropControlNetDiTVector|PIPELINE_2B_VECTOR|NET_2B_VECTOR|FSDP_CONFIG_VECTOR|phys_prop_vector_controlnet|configs.physprop_conditioned.defaults.(data|physpred_model|vlm_model)' \
  cosmos_predict2/ examples/ scripts/ imaginaire/
# (returns empty)
```

End-to-end inference was **not** run (no GPU env on this machine). Run the canonical command on a single split (`--num_splits 1 --total_seconds 5.0`) on a GPU machine to confirm.

---

## 5. What's left in the repo after cleanup

```
cosmos-predict2/
├── README.md, LICENSE, ATTRIBUTIONS.md, CONTRIBUTING.md, CLAUDE.md
├── pyproject.toml, uv.lock, requirements-conda.txt, requirements-docker.txt
├── Dockerfile, cosmos-predict2.yaml, justfile, ruff.toml, .python-version
├── bin/, .gitignore, .pre-commit-config*.yaml, .link-check.json
├── proposals/ (this directory)
├── data_annotation/  ← KEPT
├── datasets/ (README.md only after agibot jsonl removal)
├── documentations/ (6 post-training docs only)
├── checkpoints/ (model checkpoints, gitignored)
├── examples/
│   ├── physprop_kubric_video2world.py  ← canonical PhyCo inference
│   ├── eval_model_kubric_dataset.py
│   ├── run_physiq_benchmark.py
│   ├── video2world.py, video2world_lora.py
│   └── setup_utils.py
├── scripts/
│   ├── train.py, train_accel.py  ← upstream training kept
│   ├── download_checkpoints.py, get_t5_embeddings.py, get_t5_embeddings_{prompt,prompt_direction,json}.py
│   ├── prepare_batch_input_json.py
│   ├── analysis/ (Physics-IQ category scoring)
│   ├── batch_jsons/ (all kept per user)
│   └── launch/ (5 inference launchers + 1 sbatch wrapper)
├── setup_scripts/
│   ├── download_checkpoint.sh, setup_env.sh
│   └── download_vlm_checkpoint.sh, upload_vlm_checkpoint.sh  ← KEPT per user
├── imaginaire/  ← upstream framework (intact)
└── cosmos_predict2/
    ├── conditioner.py, checkpointer.py
    ├── auxiliary/ (text_encoder, cosmos_reason1, guardrail/)
    ├── callbacks/, datasets/, schedulers/, module/, networks/
    ├── tokenizers/, utils/, vram_management/, functional/
    ├── data/
    │   ├── dataset_utils.py, datasets.py, dataset_video.py, webdataloader.py
    │   ├── kubric_data/ (kubric_dataset.py + kubric_utils.py)
    │   ├── json_data/ (kept)
    │   └── action_conditioned/ (kept)
    ├── configs/
    │   ├── base/ (config.py, config_video2world.py, defaults/, experiment/{cosmos_nemo_assets,utils}.py)
    │   ├── action_conditioned/ (kept)
    │   └── physprop_conditioned/
    │       ├── config_physprop_conditioned.py  ← all DiT/pipeline blocks except VECTOR
    │       └── defaults/ (model.py, conditioner.py)
    ├── models/
    │   ├── physprop_v2w_dit.py  ← all DiT classes except Vector
    │   ├── physprop_v2w_model.py
    │   ├── video2world_{dit,model}.py, text2image_dit.py, utils.py
    │   └── action_video2world_{dit,model}.py, latent_to_vision_adapter.py REMOVED
    └── pipelines/
        ├── physprop_v2w.py  ← canonical PhyCo pipeline
        ├── video2world.py, base.py
        └── action_video2world.py, text2image.py
```
