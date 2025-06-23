import argparse
import torch
from tqdm import tqdm
import json
import os
import torch
from llava.config import Strategy
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.model.forward import llava_modify_inf
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path
from llava.eval.att_utils import get_heads, get_terminator
from llava.eval.att_data import create_data_loader, QBENCH


def eval_model(args):
    s = Strategy(args.sname)
    s.highlight = ""
    # dtype = torch.bfloat16 if "1.6" in args.model_path else torch.float16
    dtype = torch.bfloat16
    assert getattr(s, "capture", None) is not None, "Please specify the capture in the strategy."
    s.model = args.model_path.split("/")[-1]
    if getattr(s, "stage", None) is not None:
        s.heads = get_heads(s.model, s.stage, s.head_num)

    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if "llama-3" in model_name.lower():
        args.conv_mode = "llava_llama_3"
    elif "phi-3" in model_name.lower():
        args.conv_mode = "llava_phi_3"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_name=model_name, model_base=None, torch_dtype=dtype
    )
    model.config.tokenizer_padding_side = tokenizer.padding_side = "left"

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    questions = json.load(open(args.question_file))

    data_loader = create_data_loader(
        QBENCH, questions, args.image_folder, tokenizer, image_processor, model.config, batch_size=args.batch_size, args=args
    )
    terminators = get_terminator(model_name, tokenizer)
    total, correct = 0, 0
    tbar = tqdm(data_loader)
    anyres = getattr(model.config, "image_aspect_ratio", None)
    llava_modify_inf(model)
    meta = []
    for i, (batch_mask, input_ids, image_tensors, gt_list, idx_list, attention_mask, image_sizes) in enumerate(tbar):
        for gt, idx in zip(gt_list, idx_list):
            gt["id"] = idx
        meta.extend(gt_list)
    with open("meta.json", "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="vip-llava-v1.5")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./playground/data/qbench/images_llvisionqa")
    parser.add_argument("--question-file", type=str, default="./playground/data/qbench/llvisionqa_dev.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--sname", type=str, default=None)

    args = parser.parse_args()
    model_path = args.model_path.lower()
    if "13b" in model_path or "34b" in model_path:
        if "1.6" in model_path:
            args.batch_size = 1
        else:
            args.batch_size = 6
    elif "7b" in model_path or "8b" in model_path:
        if "1.6" in model_path:
            args.batch_size = 2
        else:
            args.batch_size = 16
    # args.batch_size = 10 if "13b" in args.model_path else 24

    eval_model(args)
