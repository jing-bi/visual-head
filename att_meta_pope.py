import argparse
import torch
import os
import json
from tqdm import tqdm
import re
from llava.config import Strategy
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path


from llava.eval.att_utils import get_chunk, get_terminator
from llava.eval.att_data import POPE, create_data_loader


def eval_model(args):
    # Model
    s = Strategy(args.sname)
    s.highlight = ""
    # s.capture = False
    dtype = torch.bfloat16 if "1.6" in args.model_path else torch.float16

    s.model = args.model_path.split("/")[-1]
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if "llama-3" in model_name.lower():
        args.conv_mode = "llava_llama_3"
    elif "phi-3" in model_name.lower():
        args.conv_mode = "llava_phi_3"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, torch_dtype=dtype)
    model.config.tokenizer_padding_side = tokenizer.padding_side = "left"

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.")

    data_loader = create_data_loader(
        POPE, questions, args.image_folder, tokenizer, image_processor, model.config, batch_size=args.batch_size, args=args, num_workers=16
    )
    terminators = get_terminator(model_name, tokenizer)
    total, correct = 0, 0
    tbar = tqdm(data_loader)
    anyres = getattr(model.config, "image_aspect_ratio", None)
    meta = []
    for i, (batch_mask, input_ids, image_tensors, gt_list, idx_list, attention_mask, image_sizes) in enumerate(tbar):
        meta.extend(gt_list)
    with open("pope.json", "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--sname", type=str)
    args = parser.parse_args()
    if "13b" in args.model_path or "34b" in args.model_path.lower():
        if "1.6" in args.model_path:
            args.batch_size = 1
        else:
            args.batch_size = 10
    elif "7b" in args.model_path or "8b" in args.model_path.lower():
        if "1.6" in args.model_path:
            args.batch_size = 3
        else:
            args.batch_size = 24
    eval_model(args)
