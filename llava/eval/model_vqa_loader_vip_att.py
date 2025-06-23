import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

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

from llava.visual_prompt_organizer import vip_processor, vip_processor2
import random

from llava.model.highlight import bbox_highlight, txt_highlight
from llava.model.forward import llava_modify_inf
from llava.model.guidance import PassLogitsProcessor, ProbCFGLogitsProcessor


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
                image, conversation, vp = vip_processor2(
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
        qs = conversation[0]["value"]
        gt = conversation[1]["value"]
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        image_tensor = process_images(
            [image],
            self.image_processor,
            self.model_config,
            image_aspect_ratio=self.image_aspect_ratio,
        )[0]
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        vp_grid = bbox_highlight(image, vp["bboxes"])
        txt_mask = txt_highlight(self.tokenizer, prompt)
        image_token_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]

        masked_token_map = (
            txt_mask[:image_token_start]
            + vp_grid
            # + [1] * len(txt_mask[image_token_start + 1 :])
            + txt_mask[image_token_start + 1 :]
        )
        # print('txt_mask',txt_mask)
        # print('txt_mask[:image_token_start]',txt_mask[:image_token_start])
        # print('vp_grid',vp_grid)
        # print('txt_mask[image_token_start + 1 :]',txt_mask[image_token_start + 1 :])
        # print('masked_token_map',masked_token_map)
        masked_token_map = torch.LongTensor(masked_token_map)

        return (
            masked_token_map,
            input_ids,
            image_tensor,
            gt,
        )

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
    num_workers=8,
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
    correct = {}
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
    tbar = tqdm(enumerate(zip(data_loader, questions)), total=len(questions))
    for i, ((token_highlight, input_ids, image_tensor, gt), line) in tbar:
        idx = line["id"]
        i = i + 1
        input_ids = input_ids.repeat(3, 1)
        image_tensor = image_tensor.repeat(3, 1, 1, 1).to(torch.float16)
        llava_modify_inf(model)
        token_highlight = token_highlight.squeeze()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=32,
                eos_token_id=terminators,
                use_cache=True,
                attention_weight=args.attention_weight,
                masked_token_map=token_highlight,
                perturb_weight=args.perturb_weight,
                logits_processor=[
                    # ProbCFGLogitsProcessor(guidance_scale=args.cfg, use_log=False)
                    PassLogitsProcessor(guidance_scale=args.cfg, use_log=False)
                ],
            )

        input_token_len = input_ids.shape[1]
        # outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        outputs = [i.strip() for i in outputs]
        # outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    # "text": outputs,
                    "pred": outputs,
                    "gt": gt[0],
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
        for j, output in enumerate(outputs):

            if output.lower() == gt[0].lower():
                correct[j] = correct.get(j, 0) + 1

        print(f"Accuracy: {[correct / (i + 1) for correct in correct.values()]}")
        # tbar.set_description(f"Accuracy: {correct / (i + 1)}")
        tbar.update(1)
        model.reset_model()
        torch.cuda.empty_cache()
        ans_file.flush()
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
    args = parser.parse_args()
    model_base = args.model_path.split("/")[-1]
    args.answers_file = args.answers_file.replace(
        ".jsonl", f"-{model_base}-{args.perturb_weight}-{args.num_beams}.jsonl"
    )
    args.attention_weight = args.attn
    eval_model(args)
