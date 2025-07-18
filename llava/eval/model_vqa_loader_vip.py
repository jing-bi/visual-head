import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import re
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import math

from llava.visual_prompt_organizer import vip_processor
import random

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


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, args, image_aspect_ratio = None):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.data_args = args
        self.image_aspect_ratio = getattr(args, "image_aspect_ratio", None) 

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        attempts = 0
        MAX_ATTEMPTS = 100
        while True:
            try:
                image, conversation = vip_processor(line, image, image_size_anchor = self.image_processor.crop_size['height'], data_args = self.data_args)
                break
            except:
                print('Fail in ViP image processing...')
                attempts += 1
                if attempts > MAX_ATTEMPTS:
                    print('Fail in all ViP image processing...')
                    return self.__getitem__(random.randint(0, len(self.questions)-1))
        
        qs = conversation[0]['value']
        gt = conversation[1]['value']
        # print(gt)
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()


        image_tensor = process_images([image], self.image_processor, self.model_config, image_aspect_ratio = self.image_aspect_ratio)[0]
        if args.white_image == True:
            print('white-img')
            image_tensor = make_images_white(image_tensor)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        return input_ids, image_tensor, gt

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4, args=None):
    assert batch_size == 1, "batch_size must be 1"
    if num_workers == 0:
        print("Warning: num_workers is 0, this may cause issues with the DataLoader")
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, args)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if "llama-3" in  model_name.lower():
        args.conv_mode = "llava_llama_3"
    elif 'phi-3' in  model_name.lower(): 
        args.conv_mode = "llava_phi_3"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = json.load(open(os.path.expanduser(args.question_file)))
    
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config, args= args)
    correct = 0
    total = 0
    if "llama-3" in  model_name.lower():
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    elif 'phi-3' in  model_name.lower(): 
        terminators = [tokenizer.eos_token_id,  tokenizer.convert_tokens_to_ids("<|end|>")]
    else:
        terminators = [tokenizer.eos_token_id,]
    for i, ((input_ids, image_tensor, gt), line) in tqdm(enumerate(zip(data_loader, questions)), total=len(questions)):
        idx = line["id"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=terminators,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        total += 1
        # print(outputs)
        # print(type(outputs))
        # print(gt)
        # print(type(gt))
        #first_word = re.sub(r"[^a-zA-Z]", "", outputs.split()[0])
        ans_id = shortuuid.uuid()
        is_correct = 0
        if outputs.lower() == gt[0].lower():
            is_correct = 1
            correct += 1
        print(f"Accuracy: {correct / total}")
        ans_file.write(json.dumps({"question_id": idx,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "is_correct": is_correct
                                   #"metadata": {}
                                   }) + "\n")
        # if outputs.lower() == gt[0].lower():
        #     correct += 1
        #print(f"Accuracy: {correct / (i + 1)}")
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
    parser.add_argument("--qtype", type=str, default=None)
    parser.add_argument("--highlight", type=str, default=None)
    parser.add_argument("--white-image", action='store_true', help="Use a white image if set")
    args = parser.parse_args()
    eval_model(args)
