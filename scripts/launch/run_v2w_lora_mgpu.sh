#!/bin/bash

# Cosmos Predict2 Video2World Launch Script
# This script runs video2world generation with context parallelism

# Set default values
export CUDA_HOME=$CONDA_PREFIX
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
RESOLUTION=480
FPS=10
SAVE_DIR="output/base_lora_model/"
MODEL_SIZE="2B"
DISABLE_GUARDRAIL=true
DISABLE_PROMPT_REFINER=true

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input_path PATH         Path to input image/video (required)"
    echo "  --prompt TEXT             Text prompt for generation (required)"
    echo "  --dit_path PATH           Path to model checkpoint"
    echo "  --resolution RES          Video resolution (default: 480)"
    echo "  --fps N                   Frames per second (default: 10)"
    echo "  --num_gpus N              Number of GPUs (default: 1)"
    echo "  --save_dir DIR            Output directory (default: output/base_model/test_ramp_realistic-v1/)"
    echo "  --model_size SIZE         Model size (default: 2B)"
    echo "  --batch_input_json PATH   Path to batch input JSON file"
    echo "  --save_filename NAME      Custom save filename"
    echo "  --disable_guardrail       Disable guardrail (flag)"
    echo "  --disable_prompt_refiner  Disable prompt refiner (flag)"
    echo "  --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --input_path /path/to/image.jpg --prompt \"A cat walking\""
    echo "  $0 --input_path /path/to/video.mp4 --prompt \"A dog running\" --resolution 512 --fps 12"
    echo "  $0 --batch_input_json /path/to/config.json --prompt \"Physics simulation\""
}


while [[ $# -gt 0 ]]; do
  case $1 in
    --input_path)
      INPUT_PATH="$2"
      shift 2
      ;;
    --prompt)
      PROMPT="$2"
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
    --fps)
      FPS="$2"
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
    --model_size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    --batch_input_json)
      BATCH_INPUT_JSON="$2"
      shift 2
      ;;
    --save_filename)
      SAVE_FILENAME="$2"
      shift 2
      ;;
    --disable_guardrail)
      DISABLE_GUARDRAIL=true
      shift
      ;;
    --disable_prompt_refiner)
      DISABLE_PROMPT_REFINER=true
      shift
      ;;
    --prepend_fname)
      PREPEND_FNAME="$2"
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



echo "=================================================="
echo "Cosmos Predict2 Video2World Generation"
echo "=================================================="
echo "Input path: $INPUT_PATH"
echo "Prompt: $PROMPT"
echo "Model checkpoint: $DIT_PATH"
echo "Resolution: $RESOLUTION"
echo "FPS: $FPS"
echo "Number of GPUs: $NUM_GPUS"
echo "Save directory: $SAVE_DIR"
echo "Model size: $MODEL_SIZE"
if [ -n "$BATCH_INPUT_JSON" ]; then
    echo "Batch input JSON: $BATCH_INPUT_JSON"
fi
if [ -n "$SAVE_FILENAME" ]; then
    echo "Save filename: $SAVE_FILENAME"
fi
echo "=================================================="

# Count items in batch JSON if provided
if [ -n "$BATCH_INPUT_JSON" ] && command -v jq >/dev/null 2>&1; then
    ITEM_COUNT=$(jq '. | length' "$BATCH_INPUT_JSON" 2>/dev/null || echo "unknown")
    echo "Batch contains: $ITEM_COUNT items"
fi

# Build the torchrun command
CMD="torchrun --nproc_per_node=${NUM_GPUS} examples/video2world_lora.py \
    --model_size $MODEL_SIZE \
    --num_gpus ${NUM_GPUS} \
    --resolution $RESOLUTION \
    --fps $FPS \
    --save_dir \"$SAVE_DIR\" \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_target_modules \"q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2\""

# Add input path or batch input JSON
if [ -n "$INPUT_PATH" ]; then
    CMD="$CMD --input_path \"$INPUT_PATH\""
fi

if [ -n "$PROMPT" ]; then
    CMD="$CMD --prompt \"$PROMPT\""
fi

if [ -n "$BATCH_INPUT_JSON" ]; then
    CMD="$CMD --batch_input_json \"$BATCH_INPUT_JSON\""
fi

# Add optional arguments
if [ -n "$DIT_PATH" ]; then
    CMD="$CMD --dit_path \"$DIT_PATH\""
fi

if [ -n "$SAVE_FILENAME" ]; then
    CMD="$CMD --save_filename \"$SAVE_FILENAME\""
fi

# Add flags
if [ "$DISABLE_GUARDRAIL" = true ]; then
    CMD="$CMD --disable_guardrail"
fi

if [ "$DISABLE_PROMPT_REFINER" = true ]; then
    CMD="$CMD --disable_prompt_refiner"
fi

if [ -n "$PREPEND_FNAME" ]; then
    CMD="$CMD --prepend_fname \"$PREPEND_FNAME\""
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
else
    echo "❌ Video generation failed with exit code: $EXIT_CODE"
fi
echo "=================================================="

exit $EXIT_CODE