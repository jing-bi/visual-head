#  mamba activate ./storage/conda/vp

export HF_HOME="./storage/cache"
export HUGGINGFACE_HUB_CACHE="./storage/cache"
export TRANSFORMERS_CACHE="./storage/cache"
export TORCH_HOME="./storage/cache"
export LD_LIBRARY_PATH=./storage/conda/vp/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PYTHONPATH=./ViP-LLaVA-JJ:$PYTHONPATH



MODEL_PATH="mucai/vip-llava-7b"
SPLIT="mmbench_dev_en_20231003"

python ./llava/eval/vip_vqa_mmbench.py \
    --model-path "$MODEL_PATH" \
    --question-file ./storage/MMBench/mmbench_dev_en_20231003.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/$MODEL_PATH.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1


