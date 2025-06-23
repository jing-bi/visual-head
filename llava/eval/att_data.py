# Custom dataset class
import math
from pathlib import Path
import random

from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from llava.config import Strategy
from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.guidance import PassLogitsProcessor, ProbCFGLogitsProcessor
from llava.model.highlight import bbox_highlight, txt_highlight
from llava.visual_prompt_organizer import vip_processor
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def create_data_loader(
    data_class, questions, image_folder, tokenizer, image_processor, model_config, batch_size, num_workers=18, args=None
):
    def collate_fn(batch):
        batch_mask, batch_input_ids, batch_images, gts, idxs = zip(*batch)
        image_sizes = [img.size for img in batch_images]
        batch_image_tensors = process_images(batch_images, image_processor, model_config)

        batch_input_ids = padding(tokenizer, batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        padded_batch_mask = padding(tokenizer, batch_mask, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = torch.where(batch_input_ids != tokenizer.pad_token_id, 1, 0)
        [gts[i].update({"padding_len": padding}) for i, padding in enumerate((attention_mask == 0).sum(dim=1).tolist())]
        return padded_batch_mask, batch_input_ids, batch_image_tensors, gts, idxs, attention_mask, image_sizes

    dataset = data_class(questions, image_folder, tokenizer, image_processor, model_config, args)
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn, prefetch_factor=num_workers * 2
    )


def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
    """Pad a sequence to the desired max length."""
    if len(sequence) >= max_length:
        return sequence
    return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])


def padding(tokenizer, input_ids, batch_first, padding_value):
    assert tokenizer.padding_side == "left", "Padding side must be left"

    input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    input_ids = torch.flip(input_ids, [1])
    return input_ids


def highlight_mask(tokenizer, input_ids, image_token_index, highlight):
    if highlight == "imagetoken":
        txt_mask = txt_highlight(tokenizer, input_ids)
        image_token_start = torch.where(input_ids == image_token_index)[0]
        token_map = txt_mask[:image_token_start] + [1] * 576 + txt_mask[image_token_start + 1 :]
        token_map = torch.tensor(token_map)
    else:
        token_map = torch.tensor([1] * (len(input_ids) + 575))
    return token_map


class Point(Dataset):
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
        image = Image.open(Path(self.image_folder) / image_file).convert("RGB")
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

        conv = conv_templates[self.data_args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        # ----------------- Highlighting -----------------
        image_token_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].item()
        token_map = highlight_mask(self.tokenizer, input_ids, IMAGE_TOKEN_INDEX, Strategy().highlight)
        gt_all = {"number": line.get("answer"), "option": gt, "id": line.get("id")}
        user_prompt_len = len(input_ids) - image_token_start - 1
        gt_all["user_prompt_len"] = user_prompt_len
        gt_all["image_size"] = image.size
        gt_all["image_token_start"] = image_token_start
        return token_map, input_ids, image, gt_all, line["id"]

    def __len__(self):
        return len(self.questions)


class POPE(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, args):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.args = args
        self.s = Strategy()

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(Path(self.image_folder) / image_file).convert("RGB")
        if getattr(self.s, "white", False):
            image = Image.new("RGB", image.size, color="white")
        # image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        token_map = highlight_mask(self.tokenizer, input_ids, IMAGE_TOKEN_INDEX, Strategy().highlight)
        gt_all = {**line}
        image_token_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].item()
        user_prompt_len = len(input_ids) - image_token_start - 1
        gt_all.update(
            {
                "id": gt_all["question_id"],
                "user_prompt_len": user_prompt_len,
                "image_token_start": image_token_start,
                "image_size": getattr(image, "size", None),  # Extract image size if available
            }
        )
        return token_map, input_ids, image, gt_all, index

    def __len__(self):
        return len(self.questions)


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == "nan":
        return True
    if type(value) is str and value.lower() == "none":
        return True
    return False


def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


class MMBENCH(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, args):
        self.inputs = []
        self.lengths = []
        all_options = ["A", "B", "C", "D"]
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        for idx, row in tqdm(questions.iterrows(), total=len(questions), desc="Preprocessing Dataset"):
            options = get_options(row, all_options)
            cur_option_char = all_options[: len(options)]

            num_rounds = len(options) if args.all_rounds else 1

            for round_idx in range(num_rounds):
                qs = row["question"]
                gt = row["answer"]
                image = row["image"]
                for option_char, option in zip(cur_option_char, options):
                    qs = qs + "\n" + option_char + ". " + option
                if model_config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

                if args.single_pred_prompt:
                    if args.lang == "cn":
                        qs = qs + "\n" + "请直接回答选项字母。"
                    else:
                        qs = qs + "\n" + "Answer with the option's letter from the given choices directly."
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                options = options[1:] + options[:1]
                cur_option_char = cur_option_char[1:] + cur_option_char[:1]
                self.inputs.append({"prompt": prompt, "image": image, "ground_truth": gt, "index": row["index"], "round_idx": round_idx})
                self.lengths.append(len(options))

    def __getitem__(self, index):
        data = self.inputs[index]
        image = load_image_from_base64(data["image"])
        prompt = data["prompt"]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        token_map = highlight_mask(self.tokenizer, input_ids, IMAGE_TOKEN_INDEX, Strategy().highlight)
        gt_all = data
        image_token_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].item()
        user_prompt_len = len(input_ids) - image_token_start - 1
        gt_all.update(
            {
                "id": gt_all["index"],
                "user_prompt_len": user_prompt_len,
                "image_token_start": image_token_start,
                "image_size": image.size,
            }
        )
        return token_map, input_ids, image, gt_all, index

    def __len__(self):
        return len(self.lengths)


# Custom dataset class
class SEED(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, args):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.args = args

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image_file"]
        qs = line["question"]
        choice_a = line["choice_a"]
        choice_b = line["choice_b"]
        choice_c = line["choice_c"]
        choice_d = line["choice_d"]
        qs = (
            qs
            + "\n"
            + "A."
            + choice_a
            + "\n"
            + "B."
            + choice_b
            + "\n"
            + "C."
            + choice_c
            + "\n"
            + "D."
            + choice_d
            + "\n"
            + "Answer with the option's letter from the given choices directly."
        )
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(Path(self.image_folder) / image_file).convert("RGB")
        # image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        token_map = highlight_mask(self.tokenizer, input_ids, IMAGE_TOKEN_INDEX, Strategy().highlight)
        gt_all = {**line}
        image_token_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].item()
        user_prompt_len = len(input_ids) - image_token_start - 1
        gt_all.update(
            {
                "id": gt_all["question_id"],
                "user_prompt_len": user_prompt_len,
                "image_token_start": image_token_start,
                "image_size": getattr(image, "size", None),  # Extract image size if available
            }
        )
        return token_map, input_ids, image, gt_all, index

    def __len__(self):
        return len(self.questions)


class QBENCH(Dataset):

    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, args):
        self.questions = questions
        self.args = args
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_templates = conv_templates
        self.image_folder = image_folder
        self.tokenizer_image_token = tokenizer_image_token
        self.mm_use_im_start_end = getattr(args, "mm_use_im_start_end", False)
        self.args = args

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):

        line = self.questions[idx]
        filename = line["img_path"]

        # 1. Language Handling
        if self.args.lang == "en":
            message = line["question"] + "\nChoose between one of the options as follows:\n"
        elif self.args.lang == "zh":
            message = line["question"] + "\在下列选项中选择一个:\n"
        else:
            raise NotImplementedError(
                "Q-Bench does not support languages other than English (en) and Chinese (zh) yet. "
                "Contact us (https://github.com/VQAssessment/Q-Bench/) to convert Q-Bench into more languages."
            )

        # 2. Question and Options Formatting
        for choice, ans in zip(["A.", "B.", "C.", "D."], line["candidates"]):
            message += f"{choice} {ans}\n"
            if ans == line["correct_ans"]:
                gt = choice.strip(".")

        qs = message

        # 3. Prompt Construction with Image Tokens
        if self.mm_use_im_start_end:
            qs = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{qs}"
        else:
            qs = f"{DEFAULT_IMAGE_TOKEN}\n{qs}"

        # 4. Determine Conversation Mode
        conv_mode = self.args.conv_mode

        if self.args.conv_mode is not None and conv_mode != self.args.conv_mode:
            print(
                f"[WARNING] the auto inferred conversation mode is {conv_mode}, "
                f"while `--conv-mode` is {self.args.conv_mode}, using {self.args.conv_mode}"
            )
            conv_mode = self.args.conv_mode
        else:
            self.args.conv_mode = conv_mode

        # 5. Build Conversation Prompt
        conv = self.conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 6. Image Loading and Processing

        image = Image.open(Path(self.image_folder) / filename).convert("RGB")

        # 7. Tokenization
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        # ----------------- Highlighting -----------------
        image_token_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].item()
        token_map = highlight_mask(self.tokenizer, input_ids, IMAGE_TOKEN_INDEX, Strategy().highlight)
        gt_all = {**line}
        user_prompt_len = len(input_ids) - image_token_start - 1
        gt_all["user_prompt_len"] = user_prompt_len
        gt_all["image_size"] = image.size
        gt_all["image_token_start"] = image_token_start
        gt_all["gt"] = gt
        return token_map, input_ids, image, gt_all, idx
