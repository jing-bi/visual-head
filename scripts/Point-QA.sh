export HF_HOME="../storage/cache"
export HUGGINGFACE_HUB_CACHE="../storage/cache"
export TRANSFORMERS_CACHE="../storage/cache"
export TORCH_HOME="../storage/cache"
export LD_LIBRARY_PATH=../storage/conda/vp/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export PYTHONPATH=../ViP-LLaVA-JJ:$PYTHONPATH

MODEL_PATH="mucai/llava-1.5-llama-3-8b"

qs_level=("general" "obj" "super")
# visual_prompt_style=("visual" "plain")
visual_prompt_style=("plain")
for level in "${qs_level[@]}"; do
    for visual_prompt in "${visual_prompt_style[@]}"; do
        python ../llava/Point-QA.py \
            --model-path "$MODEL_PATH" \
            --question-file ../storage/point-QA/point_QA-$level.jsonl\
            --image-folder ../storage/point-QA \
            --answers-file ../playground/data/eval/point-QA/$MODEL_PATH-embedding_location-point-QA-$visual_prompt-$level.jsonl \
            --temperature 0 \
            --conv-mode conv_llava_llama_3 \
            --sname $visual_prompt-$level-ABCD-JJ

    done
done