import argparse
from collections import defaultdict
import torch
import os
import json
from tqdm import tqdm
import shortuuid

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


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(
        self,
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model_config,
        args,
        image_aspect_ratio=None,
    ):
        self.questions = questions  # from question_file
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.data_args = args
        self.image_aspect_ratio = getattr(args, "image_aspect_ratio", None)

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        attempts = 0
        MAX_ATTEMPTS = 100
        while True:
            try:
                image, conversation = vip_processor(
                    line,
                    image,
                    image_size_anchor=self.image_processor.crop_size["height"],
                    data_args=self.data_args,
                )
                break
            except:
                print("Fail in ViP image processing...")
                attempts += 1
                if attempts > MAX_ATTEMPTS:
                    print("Fail in all ViP image processing...")
                    return self.__getitem__(random.randint(0, len(self.questions) - 1))
        # print('==========================================')
        qs = conversation[0]["value"]
        gt = conversation[1]["value"]
        # print('qs:', qs)
        # print('------------------------------------------')
        # print('gt:', gt)
        # print('------------------------------------------')
        conv = conv_templates[args.conv_mode].copy()
        # print('conv:', conv)
        # print('------------------------------------------')
        conv.append_message(conv.roles[0], qs)
        # print('conv_roles_0:', conv)
        # print('------------------------------------------')
        conv.append_message(conv.roles[1], None)
        # print('conv_roles_1:', conv)
        # print('------------------------------------------')
        prompt = conv.get_prompt()

        # print("prompt:", prompt)
        # print("==========================================")
        image_tensor = process_images(
            [image],
            self.image_processor,
            self.model_config,
            image_aspect_ratio=self.image_aspect_ratio,
        )[0]
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        print(torch.where(input_ids == IMAGE_TOKEN_INDEX)[0])
        return input_ids, image_tensor, gt

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(
    questions,
    image_folder,
    tokenizer,
    image_processor,
    model_config,
    batch_size=1,
    num_workers=4,
    args=None,
):
    assert batch_size == 1, "batch_size must be 1"
    if num_workers == 0:
        print("Warning: num_workers is 0, this may cause issues with the DataLoader")
    dataset = CustomDataset(
        questions, image_folder, tokenizer, image_processor, model_config, args
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if "llama-3" in model_name.lower():
        args.conv_mode = "llava_llama_3"
    elif "phi-3" in model_name.lower():
        args.conv_mode = "llava_phi_3"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    questions = json.load(open(os.path.expanduser(args.question_file)))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
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

    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        args=args,
    )
    correct = 0
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

    i = 0
    attention_score = defaultdict(float)
    wrong_score = defaultdict(float)
    for i, ((input_ids, image_tensor, gt), line) in tqdm(
        enumerate(zip(data_loader, questions)), total=len(questions)
    ):
        idx = line["id"]
        i = i + 1

        # input_ids = input_ids.to(device="cuda", non_blocking=True)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=terminators,
                output_attentions=True,
                return_dict_in_generate=True,
                output_hidden_states=True,
                # use_cache=True,
                # use_cache=False,
            )
        output_ids = outputs.sequences
        input_token_len = input_ids.shape[1]
        answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        answer_idx = (
            tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].index(
                answer
            )
            - 2
        )
        attention = outputs.attentions[answer_idx]
        range_start = 35
        range_end = 35 + 576
        k = 100
        if answer.lower() == gt[0].lower():
            correct += 1
            # for layer in range(32):
            #     for head in range(32):
            #         head_att = attention[layer][0, head, 0]
            #         topk_indices = torch.topk(head_att, k).indices
            #         indices_in_range = [
            #             idx for idx in topk_indices if range_start <= idx <= range_end
            #         ]
            #         ratio = len(indices_in_range) / k
            #         attention_score[f"{layer}_{head}"] += ratio
            for layer in range(32):
                head_att = attention[layer][
                    0, :, 0
                ]  # Get all heads for the first token across all heads
                attention_score_in_range = torch.sum(
                    head_att[:, range_start:range_end], dim=-1
                ).tolist()
                for head in range(32):
                    attention_score[f"{layer}_{head}"] += round(
                        float(attention_score_in_range[head]), 2
                    )
                # topk_indices = torch.topk(head_att, k).indices  # Shape: (num_heads, k)

                # in_range_mask = (topk_indices >= range_start) & (
                #     topk_indices <= range_end
                # )
                # ratios = in_range_mask.sum(dim=1).float() / k  # Shape: (num_heads,)

                # for head, ratio in enumerate(ratios):
                #     attention_score[f"{layer}_{head}"] += round(float(ratio.item()), 2)

            with open("attention_score3.json", "w") as f:
                json.dump(attention_score, f)
        else:
            for layer in range(32):
                head_att = attention[layer][
                    0, :, 0
                ]  # Get all heads for the first token across all heads
                attention_score_in_range = torch.sum(
                    head_att[:, range_start:range_end], dim=-1
                ).tolist()
                for head in range(32):
                    wrong_score[f"{layer}_{head}"] += round(
                        float(attention_score_in_range[head]), 2
                    )
            with open("wrong_score3.json", "w") as f:
                json.dump(wrong_score, f)
        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "text": answer,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )

        print(f"Accuracy: {correct / (i + 1)}")

        torch.cuda.empty_cache()

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

    args = parser.parse_args()
    eval_model(args)
