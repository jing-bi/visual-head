cd ./ViP-LLaVA

export HF_HOME="./storage/cache"
export HUGGINGFACE_HUB_CACHE="./storage/cache"
export TRANSFORMERS_CACHE="./storage/cache"
export TORCH_HOME="./storage/cache"

model_name=mucai/vip-llava-7b
eval_dataset=pointQA_twice_test
python llava/eval/model_vqa_loader_vip_att.py  \
      --model-path  $model_name  \
      --question-file ./storage/cache/datasets--mucai--ViP-LLaVA-Instruct/snapshots/c885fc27808a7e91382619ff26ab438e7a1cfd3b/$eval_dataset.json \
      --image-folder  ./storage/pointingqa/Datasets/LookTwiceQA \
      --alpha 128 \
      --answers-file ./attres/vip-llava-7b_1.json

# general question Accuracy: 0.6342146734372264
# general question Accuracy: 0.5846611801786027
# object question Accuracy: 0.672

