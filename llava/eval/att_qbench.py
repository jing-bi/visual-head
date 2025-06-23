import argparse
import torch
from tqdm import tqdm
import json
import os
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO

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

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    if "llama-3" in  model_name.lower():
        args.conv_mode = "llava_llama_3"
    elif 'phi-3' in  model_name.lower(): 
        args.conv_mode = "llava_phi_3"
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, True)
    
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    
    with open(args.questions_file) as f:
        llvqa_data = json.load(f)  
    
    if "llama-3" in  model_name.lower():
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    elif 'phi-3' in  model_name.lower(): 
        terminators = [tokenizer.eos_token_id,  tokenizer.convert_tokens_to_ids("<|end|>")]
    else:
        terminators = [tokenizer.eos_token_id,]
    correct = 0
    total = 0
    for i, llddata in enumerate(tqdm(llvqa_data)):
        filename = llddata["img_path"]
        if args.lang == "en":
            message = llddata["question"] + "\nChoose between one of the options as follows:\n"
        elif args.lang == "zh":
            message = llddata["question"] + "\在下列选项中选择一个:\n"
        else:
            raise NotImplementedError("Q-Bench does not support languages other than English (en) and Chinese (zh) yet. Contact us (https://github.com/VQAssessment/Q-Bench/) to convert  Q-Bench into more languages.")
        qs_type = llddata["type"]
        qs_concern = llddata["concern"]
        for choice, ans in zip(["A.", "B.", "C.", "D."], llddata["candidates"]):
            message += f"{choice} {ans}\n"
            if ans == llddata["correct_ans"]:
                gt = choice.strip(".") 
        qs = message
        
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        print(prompt)
        image = load_image(os.path.join(args.image_folder, filename))
        # image = load_image(args.image_folder + filename)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        # if args.white_image == True:
        #     print('white-img')
        #     image_tensor = make_images_white(image_tensor)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        

        with torch.inference_mode():
            output_ids, inputs_embeds = model.generate(
                input_ids,
                images=image_tensor,
                num_beams=1,
                do_sample=False,
                temperature=0,
                max_new_tokens=1024,
                use_cache=True,
                eos_token_id=terminators,
                stopping_criteria=[stopping_criteria])
        
        input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # print("outputs:",outputs)
        outputs = outputs.strip()
        # if outputs.endswith(stop_str):
        #     outputs = outputs[:-len(stop_str)]
        total += 1
        print("outputs:",outputs)
        if outputs.lower() == gt[0].lower():
            correct += 1
        print(f"Acc: {round(correct / total, 3)}")
        llddata["response"] = outputs
        ans_file.write(json.dumps({"type": qs_type,
                                   "qs_concern": qs_concern,
                                   "outputs": outputs,
                                   "gt": gt })+ "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.6-mistral-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="./playground/data/eval/qbench/images_llvisionqa")
    parser.add_argument("--questions-file", type=str, default="./playground/data/eval/qbench/llvisionqa_dev.json")
    parser.add_argument("--answers-file", type=str, default="./playground/data/eval/qbench/answer-tests.jsonl")
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct")
    parser.add_argument("--lang", type=str, default="en")
    args = parser.parse_args()

    eval_model(args)
