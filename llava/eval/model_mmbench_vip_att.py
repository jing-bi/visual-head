import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import json
from tqdm import tqdm
from llava.config import Strategy

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from llava.visual_prompt_organizer import vip_processor
import random

from llava.model.forward import llava_modify_inf
from llava.model.guidance import PassLogitsProcessor, ProbCFGLogitsProcessor
from mmbench import MMBenchDataset, custom_collate_fn

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    """Pad a sequence to the desired max length."""
    if len(sequence) >= max_length:
        return sequence
    return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])

def create_data_loader(image_folder, tokenizer, image_processor, model_config, batch_size, num_workers=8, args=None):
    def collate_fn(batch):
        batch_mask, batch_input_ids, batch_images, gt, idx = zip(*batch)
        # print(gt)
        # print(idx)
        batch_image_tensors = process_images(
            batch_images, image_processor, model_config, image_aspect_ratio=getattr(args, "image_aspect_ratio", None)
        )
        max_len = max([len(seq) for seq in batch_input_ids])
        batch_input_ids = torch.stack([pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in batch_input_ids])
        padded_batch_mask = torch.stack([pad_sequence_to_max_length(seq.squeeze(), max_len + 575) for seq in batch_mask])

        return padded_batch_mask, batch_input_ids, batch_image_tensors, gt, idx

    dataset = MMBenchDataset(args, tokenizer, image_processor, model_config)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn, prefetch_factor=4
    )
    return data_loader


def eval_model(args):
    # Model
    device = torch.device("cuda:0")
    disable_torch_init()
    s = Strategy(name = args.sname)
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if "llama-3" in model_name.lower():
        args.conv_mode = "llava_llama_3"
    elif "phi-3" in model_name.lower():
        args.conv_mode = "llava_phi_3"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )
    model.eval().to(device)
    data_loader = create_data_loader(
        args.image_folder, tokenizer, image_processor, model.config, batch_size=args.batch_size, args=args
    )
    correct = 0
    total = 0
    print(args.answers_file)
    answers_file = args.answers_file
    # os.makedirs(answers_file, exist_ok=True)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    print(answers_file)
    ans_file = open(answers_file, "w")

    if (
        "plain" in model_name
        and "finetune" not in model_name.lower()
        and "mmtag" not in args.conv_mode
    ):
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )

    if "llama-3" in model_name.lower():
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
    elif "phi-3" in model_name.lower():
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|end|>"),
        ]
    else:
        terminators = [
            tokenizer.eos_token_id,
        ]
    tbar = tqdm(data_loader)
    llava_modify_inf(model)
    i = 0
    # tbar = tqdm(enumerate(zip(data_loader, questions)), total=len(questions))
    for i, (batch_mask, input_ids, image_tensors, gt, idx) in enumerate(tbar):
        input_ids = input_ids.to(device=device, non_blocking=True)
        image_tensors = image_tensors.to(dtype=torch.float16, device=device, non_blocking=True)
        batch_mask = batch_mask.to(device=device, non_blocking=True)
        s.batch_idx = i
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensors,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=32,
                eos_token_id=terminators,
                use_cache=True,
                masked_token_map=batch_mask,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        total += 1
        total += 1
        print(output,"--",output[0])
        
        output = output.strip()
        print(output)
        print(gt)
        data = {"question_id": idx, "output": output, "gt": gt, "model_id": model_name, "is_correct": 0}

        if output.lower() == gt[0].lower():
            data["is_correct"] = 1
            correct += 1

        ans_file.write(json.dumps(data) + "\n")

        
        tbar.set_description(f"Acc: {round(correct / total, 3)}")
        tbar.update(1)
    ans_file.write(json.dumps({"total": total, "correct": correct, "accuracy": str(round(correct / total, 3))}))
    ans_file.close()


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
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--image_aspect_ratio", type=str, default=None)
    parser.add_argument("--visual_prompt_style", type=str, default=None)
    parser.add_argument("--attn", type=float, default=10.0)
    parser.add_argument("--perturb_weight", type=float, default=0.01)
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sname", type=str, default='visual-general')
    args = parser.parse_args()
    model_base = args.model_path.split("/")[-1]
    args.answers_file = args.answers_file.replace(
        ".jsonl", f"-{model_base}-{args.perturb_weight}-{args.num_beams}.jsonl"
    )
    args.attention_weight = args.attn
    eval_model(args)
