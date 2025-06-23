import os
import json
import torch
import pandas as pd
import math
import base64
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.config import Strategy
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.highlight import bbox_highlight, txt_highlight
from llava.model.forward import llava_modify_inf
from llava.model.guidance import PassLogitsProcessor, ProbCFGLogitsProcessor

class MMBenchDataset(Dataset):
    def __init__(self, args, tokenizer, image_processor, model_config, transform=None):
        self.question_file = args.question_file
        self.image_folder = args.image_folder
        self.conv_mode = args.conv_mode
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.all_options = ['A', 'B', 'C', 'D']
        
        # Read the TSV file
        self.data = pd.read_csv(os.path.expanduser(args.question_file), sep='\t')
        
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        all_options = ['A', 'B', 'C', 'D']
        question = item["question"]
        options = self._get_options(item, all_options)
        options_dict = {opt: item[opt] for opt in self.all_options}
        options = " \n ".join([f"{key}: {value}" for key, value in options_dict.items() if not self._is_none(value)])
        #cur_option_char = all_options[:len(options)]
        for option_char, option in zip(all_options[:len(options)], options):
            question = question + '\n' + option_char + '. ' + option
        qs_idx = item['index']
        # qs = item["question"] + '\n' + options + '\n' + "Answer with the option's letter from the given choices directly."
        qs = item["question"] + '\n' + options
        gt = item["answer"]

        #image_index = item["index"]
        #image_path = os.path.join(self.image_folder, f"{image_index}.png")
        image = self._load_image_from_base64(item['image'])
        #image = Image.open(image_path).convert("RGB")
        # if self.transform:
        #     image = self.transform(image)

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        
        if Strategy().highlight == "imagetoken":
            txt_mask = txt_highlight(self.tokenizer, prompt)
            image_token_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]
            token_map = txt_mask[:image_token_start] + [1] * 576 + txt_mask[image_token_start + 1 :]
            token_map = torch.tensor(token_map)
        else:
            token_map = torch.tensor([1] * len(input_ids))
        
        return token_map, input_ids, image, gt, qs_idx

    def _is_none(self, value):
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        if isinstance(value, str) and value.lower() in {'nan', 'none'}:
            return True
        return False


    def _load_image_from_base64(self, image):
        return Image.open(BytesIO(base64.b64decode(image)))
    
    def _get_options(self, row, options):
        parsed_options = []
        for option in options:
            option_value = row[option]
            if self._is_none(option_value):
                break
            parsed_options.append(option_value)
        return parsed_options
    
def custom_collate_fn(batch):
    return batch[0]  
