#!/bin/bash
export HF_HOME="./storage/cache"
export HUGGINGFACE_HUB_CACHE="./storage/cache"
export TRANSFORMERS_CACHE="./storage/cache"
export TORCH_HOME="./storage/cache"
export LD_LIBRARY_PATH=./storage/conda/vp/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PYTHONPATH=./ViP-LLaVA-JJ:$PYTHONPATH

MODEL_PATH="mucai/vip-llava-7b"

python  ./storage/LLaVA/llava/eval/model_vqa_seed.py \
    --model-path "$MODEL_PATH" \
    --question-file ./storage/SEED-Bench-H/SEED-Bench-v2-single-image.jsonl \
    --image-folder ./storage/SEED-Bench-H \
    --answers-file ./playground/data/eval/seed_bench/$MODEL_PATH.jsonl \




# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
# IFS=',' read -ra GPULIST <<< "$gpu_list"

# CHUNKS=${#GPULIST[@]}

# CKPT="llava-v1.5-7b"

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
#         --model-path liuhaotian/llava-v1.5-7b \
#         --question-file ./storage/SEED-Bench-H/SEED-Bench_v2_level1_2_3.json \
#         --image-folder ./storage/SEED-Bench-H/SEED-Bench-2-image \
#         --answers-file ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
#         --num-chunks $CHUNKS \
#         --chunk-idx $IDX \
#         --temperature 0 \
#         --conv-mode vicuna_v1 &
# done

# wait

# output_file=./playground/data/eval/seed_bench/answers/$CKPT/merge.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ./playground/data/eval/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
# done

# Evaluate
# python scripts/convert_seed_for_submission.py \
#     --annotation-file ./playground/data/eval/seed_bench/SEED-Bench.json \
#     --result-file $output_file \
#     --result-upload-file ./playground/data/eval/seed_bench/answers_upload/llava-v1.5-13b.jsonl

