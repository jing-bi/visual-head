import argparse
import json
import math
import os
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from llava.config import Strategy
from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.model.forward import llava_modify_inf
from llava.model.guidance import PassLogitsProcessor, ProbCFGLogitsProcessor
from llava.model.highlight import bbox_highlight, txt_highlight
from llava.utils import disable_torch_init
from llava.visual_prompt_organizer import vip_processor
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def get_heads(model, stage="all", k=0):
    file = Path(f"/home/bij4/vp/insight/heads-100/{model}.csv")
    data = pd.read_csv(file)
    if stage != "all":
        data = data[data["Stage"] == stage]
    data = data.sort_values(by="Count", ascending=False)
    if k > 0:
        data = data.head(k)
    heads = []
    for idx, row in data.iterrows():
        heads.append((row["Layer"], row["Head"]))
    return heads


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, args):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.data_args = args

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        attempts = 0
        MAX_ATTEMPTS = 10
        while True:
            try:
                image, conversation = vip_processor(
                    line, image, image_size_anchor=self.image_processor.crop_size["height"], data_args=self.data_args
                )
                break
            except Exception as e:
                attempts += 1
                if attempts > MAX_ATTEMPTS:
                    print(f"Fail in ViP image processing...{e}")
                    return self.__getitem__(random.randint(0, len(self.questions) - 1))

        qs = conversation[0]["value"]
        gt = conversation[1]["value"]

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        # ----------------- Highlighting -----------------
        if Strategy().highlight == "imagetoken":
            txt_mask = txt_highlight(self.tokenizer, prompt)
            image_token_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]

            token_map = txt_mask[:image_token_start] + [1] * 576 + txt_mask[image_token_start + 1 :]
            token_map = torch.tensor(token_map)
        else:
            token_map = torch.tensor([1] * len(input_ids))
        gt_all = {}
        number = line["answer"]
        gt_all["number"] = number
        gt_all["option"] = gt
        gt_all["id"] = line["id"]
        image_token_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].item()
        # last_input_ids_idx = len(input_ids) - 1
        # user_prompt_len = last_input_ids_idx - image_token_start
        user_prompt_len = len(input_ids) - image_token_start - 1
        gt_all["user_prompt_len"] = user_prompt_len
        gt_all["image_token_start"] = image_token_start
        return token_map, input_ids, image, gt_all, line["id"]

    def __len__(self):
        return len(self.questions)


def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    """Pad a sequence to the desired max length."""
    if len(sequence) >= max_length:
        return sequence
    return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])


def padding(tokenizer, input_ids, batch_first, padding_value):
    if tokenizer.padding_side == "left":
        input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch.flip(input_ids, [1])
    return input_ids


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size, num_workers=8, args=None):
    def collate_fn(batch):
        batch_mask, batch_input_ids, batch_images, gts, idxs = zip(*batch)
        image_sizes = [img.size for img in batch_images]
        batch_image_tensors = process_images(batch_images, image_processor, model_config)
        max_len = max([len(seq) for seq in batch_input_ids])
        batch_input_ids = padding(tokenizer, batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_batch_mask = torch.stack(
            [pad_sequence_to_max_length(seq.squeeze(), max_len + 575, tokenizer.pad_token_id) for seq in batch_mask]
        )
        attention_mask = torch.where(batch_input_ids != tokenizer.pad_token_id, 1, 0)
        return padded_batch_mask, batch_input_ids, batch_image_tensors, gts, idxs, attention_mask, image_sizes

    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, args)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn, prefetch_factor=4
    )
    return data_loader


def eval_model(args):
    # Model
    s = Strategy(args.sname)
    s.capture = False
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
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name=model_name, model_base=None)
    model.config.tokenizer_padding_side = tokenizer.padding_side = "left"
    questions = json.load(open(os.path.expanduser(args.question_file)))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.")

    data_loader = create_data_loader(
        questions, args.image_folder, tokenizer, image_processor, model.config, batch_size=args.batch_size, args=args
    )
    correct = 0
    total = 0

    if "llama-3" in model_name.lower():
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    elif "phi-3" in model_name.lower():
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")]
    else:
        terminators = [tokenizer.eos_token_id]
    tbar = tqdm(data_loader)
    llava_modify_inf(model)
    anyres = getattr(model.config, "image_aspect_ratio", None)
    correct = 0
    total = 0
    for i, (batch_mask, input_ids, image_tensors, gt_list, idx_list, attention_mask, image_sizes) in enumerate(tbar):
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        if not isinstance(image_tensors, list):

            image_tensors = image_tensors.to(dtype=torch.float16, device="cuda", non_blocking=True)
        else:
            image_tensors = [i.to(dtype=torch.float16, device="cuda", non_blocking=True) for i in image_tensors]
        batch_mask = batch_mask.to(device="cuda", non_blocking=True)
        attention_mask = attention_mask.to(device="cuda", non_blocking=True)
        s.batch_idx = i
        image_args = {"images": image_tensors}
        if anyres is not None:
            image_args["image_sizes"] = image_sizes
        s.input_len = 0
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                # do_sample=True if args.temperature > 0 else False,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=32,
                eos_token_id=terminators,
                use_cache=True,
                masked_token_map=batch_mask,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=attention_mask,
                **image_args,
            )
        s.input_len = 0

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for sid, (idx, output, gt) in enumerate(zip(idx_list, outputs, gt_list)):
            output = output.replace(" <|end|>", "").replace(" <|eot_id|>", "").strip()

            data = {"question_id": idx, "text": output, "model_id": model_name, "batch_idx": i, "sample_idx": sid, "is_correct": 0}
            data.update(gt)
            if output.lower() == gt["option"][0].lower():
                data["is_correct"] = 1
                correct += 1

            ans_file.write(json.dumps(data) + "\n")

        total += len(idx_list)
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
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--sname", type=str, default=None)

    args = parser.parse_args()
    if "13b" in args.model_path or "34b" in args.model_path:
        if "1.6" in args.model_path:
            args.batch_size = 1
        else:
            args.batch_size = 10
    elif "7b" in args.model_path or "8b" in args.model_path:
        if "1.6" in args.model_path:
            args.batch_size = 3
        else:
            args.batch_size = 24
    # args.batch_size = 10 if "13b" in args.model_path else 24

    eval_model(args)
