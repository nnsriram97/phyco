#!/bin/bash

# Cosmos Predict2 PhysProp Model Evaluation Script
# This script evaluates physics-conditioned video generation models on Kubric dataset

# Set default values
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=$(pwd)
CUDA_DEVICE=0
NUM_GPUS=1
SAVE_DIR="output/"
CONDITIONING_TYPE="image_blob"
PHYSPROP_INPUTS="friction_bounciness_deformable_force_move_dir"
PIPELINE_CONFIG="controlnet_24fps_73frames"
NUM_FRAMES=37
DESIRED_FPS=24
RESOLUTION="480"
BLOB_TYPE="circle"
GUIDANCE=7
SEED=42
BATCH_SIZE=1
NUM_WORKERS=4
BASE_PATH="/sriram-misc/datasets/"
DATASET_JSON_PATH="/sriram-misc/datasets/kubric_generated/use_folder_configs/force_move_dir-v2.json"
DIT_PATH="checkpoints/cosmos_predict2/physprop_vdm_kubric/physprop_controlnet_vdm_kubric_rigid_480p_16fps_data_v2_train_2025-10-17_06-43-30/checkpoints/model/iter_000009000.pt"
APPEND_FNAME=""

# ControlNet defaults
CONTROLNET_BRANCH_NAMES=""
ACTIVE_CONTROLNETS=""
CONTROLNET_SCALES=""
CONTROLNET_CKPT_PATHS=""
CONTROLNET_CHANNELS_PER=""
CONTROLNET_CHANNEL_GROUPS=""

# LoRA default values
USE_LORA=0
LORA_RANK=16
LORA_ALPHA=16
LORA_TARGET_MODULES="controlnet_blocks.0.self_attn.q_proj,controlnet_blocks.0.self_attn.k_proj,controlnet_blocks.0.self_attn.v_proj,controlnet_blocks.0.self_attn.output_proj,controlnet_blocks.0.mlp.layer1,controlnet_blocks.0.mlp.layer2,controlnet_blocks.0.cross_attn.q_proj,controlnet_blocks.0.cross_attn.k_proj,controlnet_blocks.0.cross_attn.v_proj,controlnet_blocks.0.cross_attn.output_proj"
INIT_LORA_WEIGHTS=1

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dataset_json_path PATH  Path to Kubric dataset JSON file (required)"
    echo "  --dit_path PATH           Path to model checkpoint (required)"
    echo "  --save_dir DIR            Output directory (default: output/eval_results/)"
    echo "  --num_frames N            Number of frames in video (default: 37)"
    echo "  --resolution RES          Video resolution (default: 256)"
    echo "  --desired_fps N           Desired FPS (default: 12)"
    echo "  --physprop_inputs TYPE    Physical property inputs (default: friction_and_bounciness)"
    echo "  --conditioning_type TYPE  Conditioning type (default: image_blob)"
    echo "  --pipeline_config CONFIG  Pipeline configuration (default: image_blob_controlnet_12fps_37frames)"
    echo "  --base_path PATH          Base path for dataset (to replace paths in JSON)"
    echo "  --blob_type TYPE          Blob type: circle, convex_hull, ellipse (default: circle)"
    echo "  --guidance N              Guidance value (default: 7)"
    echo "  --seed N                  Random seed (default: 42)"
    echo "  --batch_size N            Batch size for inference (default: 1)"
    echo "  --num_workers N           Number of data loader workers (default: 4)"
    echo "  --max_samples N           Maximum samples to evaluate (default: all)"
    echo "  --num_gpus N              Number of GPUs for context parallel (default: 1)"
    echo "  --cuda_device N           CUDA device ID (default: 0)"
    echo "  --disable_guardrail       Disable guardrail checks"
    echo "  --disable_prompt_refiner  Disable prompt refiner"
    echo "  --negative_prompt STR     Custom negative prompt"
    echo "  --regenerate_videos       Regenerate videos for the dataset"
    echo ""
    echo "ControlNet Multi-branch Options:"
    echo "  --controlnet_branch_names LIST    Comma-separated or JSON list of branch names"
    echo "  --active_controlnets LIST         Branches to enable during inference"
    echo "  --controlnet_branch_scales LIST   Conditioning scales per branch"
    echo "  --controlnet_branch_ckpts MAP     Branch checkpoint mapping (JSON, key=value, or file path)"
    echo "  --controlnet_channels_per_controlnet N  Override channels routed to each branch"
    echo "  --controlnet_channel_groups JSON  Explicit channel groups per branch (JSON list of lists)"
    echo "LoRA Options:"
    echo "  --use_lora                Enable LoRA inference mode"
    echo "  --lora_rank N             LoRA rank (default: 16, must match training)"
    echo "  --lora_alpha N            LoRA alpha (default: 16, must match training)"
    echo "  --lora_target_modules STR Comma-separated LoRA target modules (must match training)"
    echo "  --no_init_lora_weights    Disable LoRA weight initialization"
    echo ""
    echo "  --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Standard model evaluation"
    echo "  $0 --dataset_json_path /path/to/dataset.json --dit_path /path/to/checkpoint.pt"
    echo ""
    echo "  # LoRA model evaluation"
    echo "  $0 --dataset_json_path /path/to/dataset.json --dit_path /path/to/lora.pt --use_lora"
    echo ""
    echo "  # Evaluate first 100 samples only"
    echo "  $0 --dataset_json_path /path/to/dataset.json --dit_path /path/to/checkpoint.pt --max_samples 100"
    echo ""
    echo "  # Custom physical properties"
    echo "  $0 --dataset_json_path /path/to/dataset.json --dit_path /path/to/checkpoint.pt --physprop_inputs all"
    echo ""
    echo "Note: This script evaluates trained models on Kubric dataset using the dataloader."
    echo "      For LoRA models, ensure lora_rank, lora_alpha, and lora_target_modules match training config."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset_json_path)
      DATASET_JSON_PATH="$2"
      shift 2
      ;;
    --dit_path)
      DIT_PATH="$2"
      shift 2
      ;;
    --save_dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    --num_frames)
      NUM_FRAMES="$2"
      shift 2
      ;;
    --resolution)
      RESOLUTION="$2"
      shift 2
      ;;
    --desired_fps)
      DESIRED_FPS="$2"
      shift 2
      ;;
    --physprop_inputs)
      PHYSPROP_INPUTS="$2"
      shift 2
      ;;
    --conditioning_type)
      CONDITIONING_TYPE="$2"
      shift 2
      ;;
    --pipeline_config)
      PIPELINE_CONFIG="$2"
      shift 2
      ;;
    --base_path)
      BASE_PATH="$2"
      shift 2
      ;;
    --blob_type)
      BLOB_TYPE="$2"
      shift 2
      ;;
    --guidance)
      GUIDANCE="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --num_gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --cuda_device)
      CUDA_DEVICE="$2"
      shift 2
      ;;
    --disable_guardrail)
      DISABLE_GUARDRAIL="--disable_guardrail"
      shift 1
      ;;
    --disable_prompt_refiner)
      DISABLE_PROMPT_REFINER="--disable_prompt_refiner"
      shift 1
      ;;
    --negative_prompt)
      NEGATIVE_PROMPT="$2"
      shift 2
      ;;
    --controlnet_branch_names)
      CONTROLNET_BRANCH_NAMES="$2"
      shift 2
      ;;
    --active_controlnets)
      ACTIVE_CONTROLNETS="$2"
      shift 2
      ;;
    --controlnet_branch_scales)
      CONTROLNET_SCALES="$2"
      shift 2
      ;;
    --controlnet_branch_ckpts)
      CONTROLNET_CKPT_PATHS="$2"
      shift 2
      ;;
    --controlnet_channels_per_controlnet)
      CONTROLNET_CHANNELS_PER="$2"
      shift 2
      ;;
    --controlnet_channel_groups)
      CONTROLNET_CHANNEL_GROUPS="$2"
      shift 2
      ;;
    --use_lora)
      USE_LORA=1
      shift 1
      ;;
    --lora_rank)
      LORA_RANK="$2"
      shift 2
      ;;
    --lora_alpha)
      LORA_ALPHA="$2"
      shift 2
      ;;
    --lora_target_modules)
      LORA_TARGET_MODULES="$2"
      shift 2
      ;;
    --no_init_lora_weights)
      INIT_LORA_WEIGHTS=0
      shift 1
      ;;
    --regenerate_videos)
      REGENERATE_VIDEOS=1
      shift 1
      ;;
    --append_fname)
      APPEND_FNAME="$2"
      shift 2
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$DATASET_JSON_PATH" ]; then
    echo "Error: --dataset_json_path is required"
    echo ""
    show_help
    exit 1
fi

if [ -z "$DIT_PATH" ]; then
    echo "Error: --dit_path is required"
    echo ""
    show_help
    exit 1
fi

# Validate that files exist
if [ ! -f "$DATASET_JSON_PATH" ]; then
    echo "Error: Dataset JSON file does not exist: $DATASET_JSON_PATH"
    exit 1
fi

if [ ! -f "$DIT_PATH" ]; then
    echo "Error: Model checkpoint does not exist: $DIT_PATH"
    exit 1
fi

# Set environment variables
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "=========================================================="
echo "Cosmos Predict2 PhysProp Model Evaluation"
echo "=========================================================="
echo "Dataset JSON: $DATASET_JSON_PATH"
echo "Model checkpoint: $DIT_PATH"
echo "Save directory: $SAVE_DIR"
echo "Number of frames: $NUM_FRAMES"
echo "Resolution: $RESOLUTION"
echo "Desired FPS: $DESIRED_FPS"
echo "Physical property inputs: $PHYSPROP_INPUTS"
echo "Conditioning type: $CONDITIONING_TYPE"
echo "Pipeline config: $PIPELINE_CONFIG"
echo "Blob type: $BLOB_TYPE"
echo "Guidance: $GUIDANCE"
echo "Seed: $SEED"
echo "Batch size: $BATCH_SIZE"
echo "Num workers: $NUM_WORKERS"
if [ -n "$APPEND_FNAME" ]; then
    echo "Append filename: $APPEND_FNAME"
fi
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples: $MAX_SAMPLES"
else
    echo "Max samples: All samples in dataset"
fi
if [ -n "$BASE_PATH" ]; then
    echo "Base path: $BASE_PATH"
fi
if [ -n "$REGENERATE_VIDEOS" ]; then
    echo "Regenerate videos: $REGENERATE_VIDEOS"
fi
echo "Number of GPUs: $NUM_GPUS"
echo "CUDA device: $CUDA_DEVICE"
if [ -n "$DISABLE_GUARDRAIL" ]; then
    echo "Guardrail: Disabled"
fi
if [ -n "$DISABLE_PROMPT_REFINER" ]; then
    echo "Prompt refiner: Disabled"
fi
echo ""
echo "LoRA Configuration:"
if [ "$USE_LORA" -eq 1 ]; then
    echo "  LoRA mode: ENABLED"
    echo "  LoRA rank: $LORA_RANK"
    echo "  LoRA alpha: $LORA_ALPHA"
    echo "  Init LoRA weights: $([[ $INIT_LORA_WEIGHTS -eq 1 ]] && echo "Yes" || echo "No")"
    echo "  Target modules: ${LORA_TARGET_MODULES:0:80}..."
else
    echo "  LoRA mode: DISABLED (standard inference)"
fi
if [ -n "$CONTROLNET_BRANCH_NAMES" ]; then
    echo "  ControlNet branches: $CONTROLNET_BRANCH_NAMES"
fi
if [ -n "$ACTIVE_CONTROLNETS" ]; then
    echo "  Active ControlNets: $ACTIVE_CONTROLNETS"
fi
if [ -n "$CONTROLNET_SCALES" ]; then
    echo "  ControlNet scales: $CONTROLNET_SCALES"
fi
if [ -n "$CONTROLNET_CKPT_PATHS" ]; then
    echo "  ControlNet branch checkpoints: $CONTROLNET_CKPT_PATHS"
fi
echo "=========================================================="

# Count items in dataset JSON
if command -v jq >/dev/null 2>&1; then
    if [[ "$DATASET_JSON_PATH" == *.json ]]; then
        ITEM_COUNT=$(jq '. | length' "$DATASET_JSON_PATH" 2>/dev/null || echo "unknown")
        echo "Dataset contains: $ITEM_COUNT scenes"
    fi
fi

# Build the python command
CMD="python examples/eval_model_kubric_dataset.py \
    --model_size 2B \
    --dataset_json_path \"$DATASET_JSON_PATH\" \
    --dit_path \"$DIT_PATH\" \
    --save_dir \"$SAVE_DIR\" \
    --num_frames $NUM_FRAMES \
    --resolution $RESOLUTION \
    --desired_fps $DESIRED_FPS \
    --physprop_inputs $PHYSPROP_INPUTS \
    --conditioning_type $CONDITIONING_TYPE \
    --pipeline_config $PIPELINE_CONFIG \
    --blob_type $BLOB_TYPE \
    --guidance $GUIDANCE \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --disable_guardrail \
    --disable_prompt_refiner \
    --num_gpus $NUM_GPUS \
    --append_fname $APPEND_FNAME"

# Add optional arguments
if [ -n "$BASE_PATH" ]; then
    CMD="$CMD --base_path \"$BASE_PATH\""
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ -n "$NEGATIVE_PROMPT" ]; then
    CMD="$CMD --negative_prompt \"$NEGATIVE_PROMPT\""
fi

if [ -n "$CONTROLNET_BRANCH_NAMES" ]; then
    CMD="$CMD --controlnet_branch_names '$CONTROLNET_BRANCH_NAMES'"
fi

if [ -n "$ACTIVE_CONTROLNETS" ]; then
    CMD="$CMD --active_controlnets '$ACTIVE_CONTROLNETS'"
fi

if [ -n "$CONTROLNET_SCALES" ]; then
    CMD="$CMD --controlnet_branch_scales '$CONTROLNET_SCALES'"
fi

if [ -n "$CONTROLNET_CKPT_PATHS" ]; then
    CMD="$CMD --controlnet_branch_ckpts '$CONTROLNET_CKPT_PATHS'"
fi

if [ -n "$CONTROLNET_CHANNELS_PER" ]; then
    CMD="$CMD --controlnet_channels_per_controlnet $CONTROLNET_CHANNELS_PER"
fi

if [ -n "$CONTROLNET_CHANNEL_GROUPS" ]; then
    CMD="$CMD --controlnet_channel_groups '$CONTROLNET_CHANNEL_GROUPS'"
fi


# Add LoRA arguments if enabled
if [ "$USE_LORA" -eq 1 ]; then
    CMD="$CMD --use_lora"
    CMD="$CMD --lora_rank $LORA_RANK"
    CMD="$CMD --lora_alpha $LORA_ALPHA"
    CMD="$CMD --lora_target_modules \"$LORA_TARGET_MODULES\""
    if [ "$INIT_LORA_WEIGHTS" -eq 1 ]; then
        CMD="$CMD --init_lora_weights"
    fi
fi

if [ -n "$REGENERATE_VIDEOS" ]; then
    CMD="$CMD --regenerate_videos"
fi

echo ""
echo "Running command:"
echo "$CMD"
echo "=========================================================="

# Execute the command
eval $CMD

EXIT_CODE=$?

echo "=========================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Model evaluation completed successfully!"
    echo "Results saved to: $SAVE_DIR"

    # Show output directory contents if it exists
    if [ -d "$SAVE_DIR" ]; then
        echo ""
        echo "Generated videos:"
        find "$SAVE_DIR" -name "*.mp4" -type f | head -10
        VIDEO_COUNT=$(find "$SAVE_DIR" -name "*.mp4" -type f | wc -l)
        echo ""
        echo "Total videos generated: $VIDEO_COUNT"
    fi
else
    echo "❌ Model evaluation failed with exit code: $EXIT_CODE"
fi
echo "=========================================================="

exit $EXIT_CODE
