import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.config import Strategy
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.model.forward import llava_modify_inf
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

def make_images_white(image_tensor, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]):

    if image_tensor.dim() == 3:

        for c in range(image_tensor.shape[0]):
            image_tensor[c, :, :] = (1 - image_mean[c]) / image_std[c]
    elif image_tensor.dim() == 4:

        for c in range(image_tensor.shape[1]):
            image_tensor[:, c, :, :] = (1 - image_mean[c]) / image_std[c]
    else:
        raise ValueError("dim should be 3 or 4")

    return image_tensor

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def locate_embadding(input_ids, inputs_embeds):
    # [INST] <image>
    # Is there a bus in the image?
    # Answer with "Yes" or "No"? [/INST]

    input_ids = input_ids.squeeze(0)  
    inputs_embeds = inputs_embeds.squeeze(0)  
    image_position = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0].item()
    image_token_start = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0].item()
    input_ids_last_idx = input_ids.shape[0] - 1
    inputs_embeds_last_idx = inputs_embeds.shape[0] - 1

    # image     image####
    image_input_start = image_position
    after_image_len = input_ids_last_idx - image_input_start
    image_input_end = inputs_embeds_last_idx - after_image_len 
    # print("image_input_end+1",image_input_end+1)
    # user prompt
    text_input_start = image_input_end + 3
    text_input_end = inputs_embeds_last_idx - 4

    # system input 1
    system_input_1_start = 0
    system_input_1_end = image_position - 1

    # system input 2
    system_input_2_start = inputs_embeds_last_idx - 3
    system_input_2_end = inputs_embeds_last_idx

    # print("text_input_start",text_input_start)
    # print("text_input_end",text_input_end)
    return [system_input_1_start, system_input_1_end], [image_input_start, image_input_end], [text_input_start, text_input_end], [system_input_2_start, system_input_2_end]

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["question"]
        choice_a = line.get("choice_a")
        choice_b = line.get("choice_b")
        choice_c = line.get("choice_c")
        choice_d = line.get("choice_d")

        qs = qs + "\n"
 
        if any([choice_a, choice_b, choice_c, choice_d]):
            
            if choice_a:
                qs += f"A. {choice_a}\n"
            if choice_b:
                qs += f"B. {choice_b}\n"
            if choice_c:
                qs += f"C. {choice_c}\n"
            if choice_d:
                qs += f"D. {choice_d}\n"
            qs += "Answer with the option's letter from the given choices directly."
        else:
            
            qs += "Answer with \"Yes\" or \"No\"."

        # qs = qs + "\n" + "A. 1" + "\n" + "B. 2" + "\n" + "C. 3" + "\n" + "D. 4" + '\n' + "Answer with the option's letter from the given choices directly."
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print("prompt",prompt)
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        if args.white_image == True:
            print('white-img')
            image_tensor = make_images_white(image_tensor)
        print("IMAGE_TOKEN_INDEX",IMAGE_TOKEN_INDEX)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        print("input_ids",input_ids)
        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def eval_model(args):
    s = Strategy(args.sname)
    s.capture = True
    s.model = args.model_path.split("/")[-1]
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if "llama-3" in model_name.lower():
        args.conv_mode = "llava_llama_3"
    elif "phi-3" in model_name.lower():
        args.conv_mode = "llava_phi_3"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)
    if "llama-3" in model_name.lower():
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    elif "phi-3" in model_name.lower():
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")]
    else:
        terminators = [
            tokenizer.eos_token_id,
        ]
    llava_modify_inf(model)
    total = 0
    correct = 0
    i = 0
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["question"]
        gt = line["gt"]
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        # print(input_ids)
        s.batch_idx = i
        i=i+1
        s.input_len = 0
        with torch.inference_mode():
            output_ids, inputs_embeds = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=terminators,
                use_cache=True,
            )
        # print("input_ids",input_ids)
        # print("inputs_embeds",inputs_embeds)
        print("inputs_embeds",len(inputs_embeds))
        sp_1_token, img_token, text_token, sp_2_token = locate_embadding(input_ids, inputs_embeds)
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print("len(outputs)",len(outputs))
        print(outputs)
        outputs = outputs.strip()
        total += 1
        first_word = re.sub(r"[^a-zA-Z]", "", outputs.split()[0])
        # print(first_word)
        is_correct = 0
        if first_word.lower() == gt.lower():
            is_correct = 1
            correct += 1
        print(f"Accuracy: {correct / total}")
        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": idx,
                    "outputs": outputs,
                    "is_correct": is_correct,
                    "sp_1_token": sp_1_token,
                    "img_token": img_token,
                    "text_token": text_token,
                    "sp_2_token": sp_2_token
                }
            )
            + "\n"
        )
        # ans_file.flush()
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
    parser.add_argument("--white-image", action='store_true', help="Use a white image if set")
    parser.add_argument("--sname", type=str, default="mmbench-no-mask")

    args = parser.parse_args()

    eval_model(args)
