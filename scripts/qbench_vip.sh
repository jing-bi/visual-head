#!/bin/bash

export HF_HOME="./storage/cache"
export HUGGINGFACE_HUB_CACHE="./storage/cache"
export TRANSFORMERS_CACHE="./storage/cache"
export TORCH_HOME="./storage/cache"
export LD_LIBRARY_PATH=./storage/conda/vp/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PYTHONPATH=./ViP-LLaVA-JJ:$PYTHONPATH


if [ "$1" = "dev" ]; then
    echo "Evaluating in 'dev' split."
elif [ "$1" = "test" ]; then
    echo "Evaluating in 'test' split."
else
    echo "Unknown split, please choose between 'dev' and 'test'."
    exit 1
fi

MODEL_PATH="mucai/vip-llava-7b"


python ./llava/eval/vip_vqa_qbench.py \
    --model-path "$MODEL_PATH" \
    --image-folder ./storage/Q-bench/images \
    --questions-file ./storage/Q-bench/llvisionqa_$1.json \
    --answers-file ./playground/data/eval/qbench/$MODEL_PATH.jsonl \
    --lang en
