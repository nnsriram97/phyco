<h1 align="center">PhyCo: Learning Controllable Physical Priors for Generative Motion</h1>

<p align="center">
  <a href="https://phyco-video.github.io/"><b>Project Page</b></a> ·
  <a href="https://arxiv.org/abs/2604.28169"><b>Paper (CVPR 2026)</b></a> ·
  <a href="https://huggingface.co/nnsriram97/phyco"><b>Checkpoint</b></a> ·
  <a href="https://huggingface.co/datasets/nnsriram97/phyco_kubric"><b>Dataset</b></a> ·
  <a href="https://github.com/nnsriram97/phyco-sim"><b>Simulation Code</b></a>
</p>

> **TL;DR.** PhyCo learns controllable physical priors — friction, restitution, deformation, and force — from simple block-sliding and ball-bouncing simulations, enabling physically grounded and continuously controllable video generation **without any simulator at inference**.

<p align="center">
  <img src="assets/duck_deform.gif" width="100%" alt="PhyCo rubber duck deformation demo">
</p>

This repository contains the **inference and evaluation** code for PhyCo, built on top of NVIDIA's [Cosmos-Predict2](https://github.com/nvidia-cosmos/cosmos-predict2) Video2World model. The simulation data-generation pipeline lives in [**PhyCo-Sim**](https://github.com/nnsriram97/phyco-sim), and the training data on [HuggingFace](https://huggingface.co/datasets/nnsriram97/phyco_kubric).

> ⚠️ Early public release focused on **inference + evaluation**. More tooling and docs will follow.

---

## Method

PhyCo conditions a **frozen** Cosmos-Predict2 Video2World diffusion transformer (DiT) with a **multi-branch ControlNet** that injects spatial physical-property maps. Each branch controls one family of properties:

| Branch | Physical property |
| --- | --- |
| `controlnet_1` | friction / restitution (bounciness) |
| `controlnet_2` | deformability |
| `controlnet_3` | applied force / motion direction |

We release a single ready-to-run checkpoint, **`phyco.pt`**, that bundles the base DiT and all three branches — pass it via `--dit_path`; no separate ControlNet files are needed. At runtime `--dynamic_controlnet` activates the relevant branch per scenario.

---

## Installation

**Requirements:** Linux x86-64, Python 3.10, an NVIDIA Ampere (or newer) GPU, CUDA 12.6. We use [`uv`](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # install uv (skip if present)

uv venv --python 3.10 --allow-existing                   # Python 3.10 venv
source .venv/bin/activate
uv sync --extra cu126 --active --inexact                 # deps (CUDA 12.6)

# pin pip tooling + use the headless OpenCV wheel (avoids libGL issues on servers)
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
.venv/bin/python -m pip install --no-cache-dir opencv-python-headless==4.11.0.86
```

Always run from the repo root with `PYTHONPATH=$(pwd)` (the launch scripts set this for you).

---

## Download checkpoints

```bash
pip install -U "huggingface_hub[cli]"

# 1. Base Cosmos-Predict2 weights (T5-11B text encoder + 2B Video2World DiT + tokenizer)
PYTHONPATH=$(pwd) python scripts/download_checkpoints.py \
  --model_sizes "2B" --model_types "video2world" --resolution "480" --fps "16"

# 2. PhyCo checkpoint -> checkpoints/phyco/phyco.pt
hf download nnsriram97/phyco phyco.pt --local-dir checkpoints/phyco
```

> The bundled `assets/common_neg_prompt-v1.pt` is the default negative-prompt embedding, so you don't need to download or pass anything for it. (If it's ever missing, the pipeline re-encodes the same prompt from `assets/common_neg_prompt-v1.txt`, which loads the T5-XXL encoder.)

---

## Quick start: controllable physics examples

These reproduce the qualitative, controllable-physics results from the paper — a good way to try PhyCo. The example sets for **friction**, **restitution**, **deformation**, and **force** ship their input frames and masks in `assets/qualitative/`, so they **run out of the box** (no external dataset needed). We sample **4 seeds (42, 0, 10, 31)**.

```bash
DIT=checkpoints/phyco/phyco.pt

# flags shared by every PhyCo run (negative-prompt embedding is bundled + used by default)
COMMON="--dit_path $DIT \
  --controlnet_branch_names controlnet_1,controlnet_2,controlnet_3 \
  --active_controlnets controlnet_1,controlnet_2,controlnet_3 \
  --controlnet_branch_scales 1.0,1.0,1.0 --dynamic_controlnet \
  --pipeline_config controlnet_multi_24fps_57frames \
  --conditioning_type image_blob --blob_type circle \
  --resolution 480 --fps 24 \
  --physprop_type friction_bounciness_deformable_force_move_dir \
  --num_splits 1 --split_index 0"

for name in friction_all-v3 restitution_all-v1 deformation_all-v1 force_all-v1; do
  PYTHONPATH=$(pwd) bash scripts/launch/run_physprop_kubric_mpgu.sh \
    --batch_input_json scripts/batch_jsons/benchmark/${name}.json \
    --post_dit_subfolder ${name} \
    --seeds 42,0,10,31 \
    $COMMON
done
```

Videos are written to `output/phyco/<name>/`, with one clip per seed (filenames prefixed `seed<N>_`). `--dynamic_controlnet` picks the branch per scenario (friction/restitution → `controlnet_1`, deformability → `controlnet_2`, force → `controlnet_3`).

> **Bring your own scene.** To run PhyCo on your own image you just need a segmentation mask for the object(s) to control. Draw one in under a minute with `scripts/misc/draw_circular_masks.py` (no SAM required) and wire it into a batch JSON — see **[Trying PhyCo on your own scenes](documentations/custom_scenes.md)**.

---

## Physics-IQ benchmark

To reproduce the paper's [Physics-IQ](https://github.com/google-deepmind/physics-IQ-benchmark) numbers, generate over the full benchmark (single seed) and score.

**Extra assets.** Physics-IQ uses real input frames and per-scenario segmentation masks:

```bash
# segmentation masks -> assets/physics_iq_segmentation_pkl/ (referenced by the JSON)
hf download nnsriram97/phyco --include "physics_iq_segmentation_pkl/*" --local-dir assets
```

The masks are pre-pointed to `assets/physics_iq_segmentation_pkl/` in `physiq_controlnet_v1.json`. The `input_video` frames come from the [Physics-IQ benchmark](https://github.com/google-deepmind/physics-IQ-benchmark) switch-frames — edit those paths in the JSON to point at your local checkout.

**Generate** (reusing `$COMMON` from the Quick-start block above):

```bash
PYTHONPATH=$(pwd) bash scripts/launch/run_physprop_kubric_mpgu.sh \
  --batch_input_json scripts/batch_jsons/benchmark/physiq_controlnet_v1.json \
  --post_dit_subfolder physics-IQ \
  --seed 42 --total_seconds 5.0 \
  $COMMON

# Multi-GPU / Slurm: fan out the 198 scenarios across GPUs with an array job
# sbatch --array 0-19 --gpus 1 -c 8 scripts/launch/run_physprop_kubric_mpgu.sh \
#   --num_splits 20 ...   # (drop --num_splits 1 --split_index 0 from $COMMON)
```

Videos land in `output/phyco/physics-IQ/`. `--pipeline_config controlnet_multi_24fps_57frames` (≈2.4 s) is the default; `controlnet_multi_24fps_120frames` gives ≈5 s clips.

**Score** (from your Physics-IQ benchmark checkout):

```bash
# 1. per-scenario MSE / IoU vs. real future frames
python code/run_physics_iq.py \
  --input_folders /path/to/phyco/output/phyco/physics-IQ \
  --output_folder ./output/phyco_physics-IQ \
  --descriptions_file descriptions/descriptions.csv

# 2. aggregate per physical-reasoning category (Solid Mechanics, Fluid Dynamics, ...)
PYTHONPATH=$(pwd) python scripts/analysis/compute_physiq_category_scores.py \
  --results_csv ./output/phyco_physics-IQ/results/physics-IQ.csv \
  --descriptions_csv /path/to/physics-IQ-benchmark/descriptions/descriptions.csv \
  --output_csv ./output/phyco_physics-IQ/results/category_scores.csv
```

> To rebuild a consolidated `phyco.pt` from a separate base DiT + per-branch runs, see `scripts/misc/consolidate_phyco_checkpoint.py`.

---

## Acknowledgements

Built on NVIDIA's [**Cosmos-Predict2**](https://github.com/nvidia-cosmos/cosmos-predict2) ([Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0); source files retain their original headers). Physical-property supervision comes from [PhyCo-Sim](https://github.com/nnsriram97/phyco-sim), which extends [Kubric](https://github.com/google-research/kubric). Evaluation uses the [Physics-IQ benchmark](https://github.com/google-deepmind/physics-IQ-benchmark).

## License

PhyCo is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license — see [LICENSE](LICENSE). Components inherited from Cosmos-Predict2 remain under their original Apache-2.0 / NVIDIA Open Model License terms.

## Citation

```bibtex
@InProceedings{Narayanan_2026_CVPR,
    author    = {Narayanan, Sriram and Jiang, Ziyu and Narasimhan, Srinivasa and Chandraker, Manmohan},
    title     = {PhyCo: Learning Controllable Physical Priors for Generative Motion},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {41892-41902}
}
```
