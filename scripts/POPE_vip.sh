#!/bin/bash
export HF_HOME="./storage/cache"
export HUGGINGFACE_HUB_CACHE="./storage/cache"
export TRANSFORMERS_CACHE="./storage/cache"
export TORCH_HOME="./storage/cache"
export LD_LIBRARY_PATH=./storage/conda/vp/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PYTHONPATH=./ViP-LLaVA-JJ:$PYTHONPATH

#MODEL_PATH="mucai/vip-llava-7b"
MODEL_PATH="liuhaotian/llava-v1.5-7b"
python -m llava.eval.model_vqa_loader \
    --model-path "$MODEL_PATH" \
    --question-file ./storage/POPE/coco_pope_adversarial.jsonl\
    --image-folder ./storage/POPE/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$MODEL_PATH.jsonl \


# python llava/eval/eval_pope.py \
#     --annotation-dir ./playground/data/eval/pope/coco \
#     --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
#     --result-file ./playground/data/eval/pope/answers/llava-v1.5-13b.jsonl
