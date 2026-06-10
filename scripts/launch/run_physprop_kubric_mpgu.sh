#!/bin/bash

# Cosmos Predict2 PhysProp Video2World Launch Script
# This script runs physics-conditioned video generation using batch input JSON files

# Set default values
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=$(pwd)
CUDA_DEVICE=0
NUM_GPUS=1
SAVE_DIR="output/"
CONDITIONING_TYPE="fg_bg_vector"
PREPEND_FNAME=""
NUM_SPLITS=1
SPLIT_INDEX=""
TOTAL_SECONDS=""
SEEDS=""
# ControlNet defaults
CONTROLNET_BRANCH_NAMES=""
ACTIVE_CONTROLNETS=""
CONTROLNET_SCALES=""
CONTROLNET_CKPT_PATHS=""
CONTROLNET_CHANNELS_PER=""
CONTROLNET_CHANNEL_GROUPS=""
MASTER_PORT=29000

# LoRA default values
USE_LORA=0
LORA_RANK=16
LORA_ALPHA=16
# LORA_TARGET_MODULES="controlnet_blocks.0.self_attn.q_proj,controlnet_blocks.0.self_attn.k_proj,controlnet_blocks.0.self_attn.v_proj,controlnet_blocks.0.self_attn.output_proj,controlnet_blocks.0.mlp.layer1,controlnet_blocks.0.mlp.layer2,controlnet_blocks.0.cross_attn.q_proj,controlnet_blocks.0.cross_attn.k_proj,controlnet_blocks.0.cross_attn.v_proj,controlnet_blocks.0.cross_attn.output_proj,controlnet_blocks.1.self_attn.q_proj,controlnet_blocks.1.self_attn.k_proj,controlnet_blocks.1.self_attn.v_proj,controlnet_blocks.1.self_attn.output_proj,controlnet_blocks.1.mlp.layer1,controlnet_blocks.1.mlp.layer2,controlnet_blocks.1.cross_attn.q_proj,controlnet_blocks.1.cross_attn.k_proj,controlnet_blocks.1.cross_attn.v_proj,controlnet_blocks.1.cross_attn.output_proj,controlnet_blocks.2.self_attn.q_proj,controlnet_blocks.2.self_attn.k_proj,controlnet_blocks.2.self_attn.v_proj,controlnet_blocks.2.self_attn.output_proj,controlnet_blocks.2.mlp.layer1,controlnet_blocks.2.mlp.layer2,controlnet_blocks.2.cross_attn.q_proj,controlnet_blocks.2.cross_attn.k_proj,controlnet_blocks.2.cross_attn.v_proj,controlnet_blocks.2.cross_attn.output_proj,controlnet_blocks.3.self_attn.q_proj,controlnet_blocks.3.self_attn.k_proj,controlnet_blocks.3.self_attn.v_proj,controlnet_blocks.3.self_attn.output_proj,controlnet_blocks.3.mlp.layer1,controlnet_blocks.3.mlp.layer2,controlnet_blocks.3.cross_attn.q_proj,controlnet_blocks.3.cross_attn.k_proj,controlnet_blocks.3.cross_attn.v_proj,controlnet_blocks.3.cross_attn.output_proj,controlnet_blocks.4.self_attn.q_proj,controlnet_blocks.4.self_attn.k_proj,controlnet_blocks.4.self_attn.v_proj,controlnet_blocks.4.self_attn.output_proj,controlnet_blocks.4.mlp.layer1,controlnet_blocks.4.mlp.layer2,controlnet_blocks.4.cross_attn.q_proj,controlnet_blocks.4.cross_attn.k_proj,controlnet_blocks.4.cross_attn.v_proj,controlnet_blocks.4.cross_attn.output_proj"
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
INIT_LORA_WEIGHTS=1

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --batch_input_json PATH   Path to batch input JSON file (required)"
    echo "  --dit_path PATH           Path to model checkpoint"
    echo "  --resolution RES          Video resolution (default: 256)"
    echo "  --physprop_type TYPE      Physical property type (default: all)"
    echo "  --pipeline_config CONFIG  Pipeline configuration"
    echo "  --fps N                   Frames per second (default: 10)"
    echo "  --t5_embeddings_path PATH Global T5 embeddings path"
    echo "  --num_gpus N              Number of GPUs (default: 1)"
    echo "  --save_dir DIR            Output directory (default: output/)"
    echo "  --conditioning_type TYPE  Conditioning type (default: fg_bg_vector)"
    echo "  --blob_type TYPE          Blob type (default: circle)"
    echo "  --seed N                  Seed for random number generator (default: 42)"
    echo "  --seeds LIST              Comma-separated or JSON list of seeds"
    echo "  --run_ids LIST            Comma-separated list of run ids to run in the batch"
    echo "                            (e.g. 0,1,2,3) or range interval (e.g. 0-3) or a combination of both (e.g. 0,1,3-5)"
    echo "  --num_splits N            Split batch JSON into N equal parts for distributed runs (default: 1)"
    echo "  --split_index I           Zero-based index of the split to process (default: 0 or \$SLURM_ARRAY_TASK_ID)"
    echo ""
    echo "ControlNet Multi-branch Options:"
    echo "  --controlnet_branch_names LIST    Comma-separated or JSON list of branch names"
    echo "  --active_controlnets LIST         Branches to enable during inference"
    echo "  --controlnet_branch_scales LIST   Conditioning scales per branch"
    echo "  --controlnet_branch_ckpts MAP     Branch checkpoint mapping (JSON, key=value, or file path)"
    echo "  --controlnet_channels_per_controlnet N  Override channels routed to each branch"
    echo "  --controlnet_channel_groups JSON  Explicit channel groups per branch (JSON list of lists)"
    echo ""
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
    echo "  # Standard inference"
    echo "  $0 --batch_input_json scripts/batch_jsons/benchmark/physiq_controlnet_v1.json --dit_path /path/to/checkpoint.pt"
    echo ""
    echo "  # LoRA inference"
    echo "  $0 --batch_input_json /path/to/test.json --dit_path /path/to/lora_checkpoint.pt --use_lora"
    echo ""
    echo "  # LoRA with custom settings"
    echo "  $0 --batch_input_json /path/to/test.json --dit_path /path/to/lora.pt --use_lora --lora_rank 32 --lora_alpha 32"
    echo ""
    echo "Note: The script requires batch_input_json for operation and supports both"
    echo "      Kubric-generated data and custom image inputs with RLE segmentation."
    echo "      For LoRA models, ensure lora_rank, lora_alpha, and lora_target_modules match training config."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --batch_input_json)
      BATCH_INPUT_JSON="$2"
      shift 2
      ;;
    --dit_path)
      DIT_PATH="$2"
      shift 2
      ;;
    --resolution)
      RESOLUTION="$2"
      shift 2
      ;;
    --physprop_type)
      PHYSPROP_TYPE="$2"
      shift 2
      ;;
    --pipeline_config)
      PIPELINE_CONFIG="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --t5_embeddings_path)
      T5_EMBEDDINGS_PATH="$2"
      shift 2
      ;;
    --num_gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --save_dir)
      SAVE_DIR="$2"
      shift 2
      ;;
    --prepend_fname)
      PREPEND_FNAME="$2"
      shift 2
      ;;
    --conditioning_type)
      CONDITIONING_TYPE="$2"
      shift 2
      ;;
    --blob_type)
      BLOB_TYPE="$2"
      shift 2
      ;;
    --neg_t5_embedding_path)
      NEG_T5_EMBEDDING_PATH="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --run_ids)
      RUN_IDS="$2"
      shift 2
      ;;
    --total_seconds)
      TOTAL_SECONDS="$2"
      shift 2
      ;;
    --num_splits)
      NUM_SPLITS="$2"
      shift 2
      ;;
    --split_index)
      SPLIT_INDEX="$2"
      shift 2
      ;;
    --cuda_device)
      CUDA_DEVICE="$2"
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
    --no_background_condition)
      NO_BACKGROUND_CONDITION=true
      shift 1
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
    --post_dit_subfolder)
      POST_DIT_SUBFOLDER="$2"
      shift 2
      ;;
    --master_port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --dynamic_controlnet)
      DYNAMIC_CONTROLNET=true
      shift 1
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
if [ -z "$BATCH_INPUT_JSON" ]; then
    echo "Error: --batch_input_json is required"
    echo ""
    show_help
    exit 1
fi

if [ -z "$SPLIT_INDEX" ]; then
    if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
        SPLIT_INDEX="$SLURM_ARRAY_TASK_ID"
    else
        SPLIT_INDEX=0
    fi
fi

if [ -z "$NUM_SPLITS" ] || [ "$NUM_SPLITS" -lt 1 ]; then
    echo "Error: --num_splits must be >= 1 (got $NUM_SPLITS)"
    exit 1
fi

if [ "$SPLIT_INDEX" -lt 0 ] || [ "$SPLIT_INDEX" -ge "$NUM_SPLITS" ]; then
    echo "Error: --split_index ($SPLIT_INDEX) must be in [0, $((NUM_SPLITS - 1))]"
    exit 1
fi

# Validate that the batch input JSON file exists
if [ ! -f "$BATCH_INPUT_JSON" ]; then
    echo "Error: Batch input JSON file does not exist: $BATCH_INPUT_JSON"
    exit 1
fi

# Set environment variables
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "=================================================="
echo "Cosmos Predict2 PhysProp Video2World Generation"
echo "=================================================="
echo "Batch input JSON: $BATCH_INPUT_JSON"
echo "Model checkpoint: $DIT_PATH"
echo "Resolution: $RESOLUTION"
echo "Physical property type: $PHYSPROP_TYPE"
echo "Pipeline config: $PIPELINE_CONFIG"
echo "FPS: $FPS"
echo "Number of GPUs: $NUM_GPUS"
echo "Save directory: $SAVE_DIR"
echo "Conditioning type: $CONDITIONING_TYPE"
if [ -n "$T5_EMBEDDINGS_PATH" ]; then
    echo "T5 embeddings path: $T5_EMBEDDINGS_PATH"
fi
if [ -n "$NEG_T5_EMBEDDING_PATH" ]; then
    echo "Negative T5 embeddings path: $NEG_T5_EMBEDDING_PATH"
fi
if [ -n "$BLOB_TYPE" ]; then
    echo "Blob type: $BLOB_TYPE"
fi
if [ -n "$SEED" ]; then
    echo "Seed: $SEED"
fi
if [ -n "$SEEDS" ]; then
    echo "Multi-seed list: $SEEDS"
fi
if [ -n "$RUN_IDS" ]; then
    echo "Run IDs: $RUN_IDS"
fi
echo "Batch split: index $SPLIT_INDEX of $NUM_SPLITS"
if [ -n "$TOTAL_SECONDS" ]; then
    echo "Target duration: ${TOTAL_SECONDS}s"
fi
if [ -n "$CONTROLNET_BRANCH_NAMES" ]; then
    echo "ControlNet branches: $CONTROLNET_BRANCH_NAMES"
fi
if [ -n "$ACTIVE_CONTROLNETS" ]; then
    echo "Active ControlNets: $ACTIVE_CONTROLNETS"
fi
if [ -n "$CONTROLNET_SCALES" ]; then
    echo "ControlNet scales: $CONTROLNET_SCALES"
fi
if [ -n "$CONTROLNET_CKPT_PATHS" ]; then
    echo "ControlNet branch checkpoints: $CONTROLNET_CKPT_PATHS"
fi
if [ -n "$NO_BACKGROUND_CONDITION" ]; then
    echo "No background condition: $NO_BACKGROUND_CONDITION"
fi
if [ -n "$POST_DIT_SUBFOLDER" ]; then
    echo "Post DiT subfolder: $POST_DIT_SUBFOLDER"
fi
if [ -n "$DYNAMIC_CONTROLNET" ]; then
    echo "Dynamic ControlNet: $DYNAMIC_CONTROLNET"
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
echo "=================================================="

# Count items in batch JSON
if command -v jq >/dev/null 2>&1; then
    ITEM_COUNT=$(jq '. | length' "$BATCH_INPUT_JSON" 2>/dev/null || echo "unknown")
    echo "Batch contains: $ITEM_COUNT items"
else
    echo "Batch file loaded (install jq for item count)"
fi

# Build the torchrun command
PORT_VALUE=$((MASTER_PORT + SPLIT_INDEX))
CMD="torchrun --nproc_per_node=${NUM_GPUS} --master_port ${PORT_VALUE} examples/physprop_kubric_video2world.py \
    --model_size 2B \
    --disable_guardrail \
    --disable_prompt_refiner \
    --num_gpus ${NUM_GPUS} \
    --batch_input_json \"$BATCH_INPUT_JSON\" \
    --resolution $RESOLUTION \
    --physprop_type $PHYSPROP_TYPE \
    --fps $FPS \
    --save_dir \"$SAVE_DIR\" \
    --prepend_fname \"$PREPEND_FNAME\" \
    --conditioning_type $CONDITIONING_TYPE \
    --blob_type $BLOB_TYPE"

# Add optional arguments
if [ -n "$DIT_PATH" ]; then
    CMD="$CMD --dit_path \"$DIT_PATH\""
fi

if [ -n "$PIPELINE_CONFIG" ]; then
    CMD="$CMD --pipeline_config $PIPELINE_CONFIG"
fi

if [ -n "$T5_EMBEDDINGS_PATH" ]; then
    CMD="$CMD --t5_embeddings_path \"$T5_EMBEDDINGS_PATH\""
fi

if [ -n "$NEG_T5_EMBEDDING_PATH" ]; then
    CMD="$CMD --neg_t5_embedding_path \"$NEG_T5_EMBEDDING_PATH\""
fi

if [ -n "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi
if [ -n "$SEEDS" ]; then
    CMD="$CMD --seeds '$SEEDS'"
fi

if [ -n "$RUN_IDS" ]; then
    CMD="$CMD --run_ids '$RUN_IDS'"
fi

CMD="$CMD --num_splits $NUM_SPLITS --split_index $SPLIT_INDEX"

if [ -n "$TOTAL_SECONDS" ]; then
    CMD="$CMD --total_seconds $TOTAL_SECONDS"
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

if [ -n "$NO_BACKGROUND_CONDITION" ]; then
    CMD="$CMD --no_background_condition"
fi

if [ -n "$POST_DIT_SUBFOLDER" ]; then
    CMD="$CMD --post_dit_subfolder \"$POST_DIT_SUBFOLDER\""
fi

if [ -n "$DYNAMIC_CONTROLNET" ]; then
    CMD="$CMD --dynamic_controlnet"
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

echo "Running command:"
echo "$CMD"
echo "=================================================="

# Execute the command
eval $CMD

EXIT_CODE=$?

echo "=================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Video generation completed successfully!"
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
    echo "❌ Video generation failed with exit code: $EXIT_CODE"
fi
echo "=================================================="

exit $EXIT_CODE
