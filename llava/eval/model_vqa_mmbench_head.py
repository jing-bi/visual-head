import argparse
import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid
from llava.config import Strategy
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
from llava.model.forward import llava_modify_inf
from PIL import Image
import math


all_options = ['A', 'B', 'C', 'D']

def get_heads(model, stage="all", k=0):
    mask = torch.zeros((32, 32), dtype=int)
    mask[31, 22] = 1
    return mask
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

# def mask_specific_head(module, input, output, head_idx):
#     output[:, head_idx, :, :] = 0  # 将指定的 head 输出置为零
#     return output

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


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
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


def eval_model(args):
    # Model
    disable_torch_init()
    s = Strategy(args.sname)
    s.capture = False
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if "llama-3" in model_name.lower():
        args.conv_mode = "llava_llama_3"
    elif "phi-3" in model_name.lower():
        args.conv_mode = "llava_phi_3"
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    #s.heads = [[11,17]]    0.971   visual-obj_topk_values_region
    #s.heads = [[14,19]]    0.995   visual-obj_topk_values_region
    #s.heads = [[22,31]]    0.998   visual-obj_topk_region_density
    #s.heads = [[0,7]]      0.998   plain-obj_att_image
    #s.heads = [[0,24]]     0.998   plain-obj_att_image
    #s.heads = [[14,24]]    0.982   plain-obj_att_region_topk_value     
    #s.heads = [[14,24],[11,17]]    0.964
    #s.heads = [[10,6]]     0.981   plain-obj_att_region_topk_ratio
    s.heads = [[14,24],[10,6],[11,17],[14,19]]  


    questions = pd.read_table(os.path.expanduser(args.question_file))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')
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
    for index, row in tqdm(questions.iterrows(), total=len(questions)):
        options = get_options(row, all_options)
        cur_option_char = all_options[:len(options)]

        if args.all_rounds:
            num_rounds = len(options)
        else:
            num_rounds = 1

        for round_idx in range(num_rounds):
            idx = row['index']
            question = row['question']
            hint = row['hint']
            gt = row['answer']
            image = load_image_from_base64(row['image'])

            # if not is_none(hint):
            #     question = hint + '\n' + question
            for option_char, option in zip(all_options[:len(options)], options):
                question = question + '\n' + option_char + '. ' + option
            qs = cur_prompt = question
            # print(qs)
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

            if args.single_pred_prompt:
                if args.lang == 'cn':
                    qs = qs + '\n' + "请直接回答选项字母。"
                else:
                    qs = qs + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

            image_tensor = process_images([image], image_processor, model.config)[0]# torch.Size([3, 336, 336])
            # print(image_tensor.shape)
            if args.white_image == True:
                print('white-img')
                image_tensor = make_images_white(image_tensor)
                # print(image_tensor)
            # print(image_tensor.shape)
            s.input_len = 0
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            total += 1
            is_correct = 0
            if outputs.lower() == gt[0].lower():
                is_correct = 1
                correct += 1
            print(f"Acc: {round(correct / total, 3)}")
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "round_id": round_idx,
                                    #"prompt": cur_prompt,
                                    "text": outputs,
                                    #"options": options,
                                    #"option_char": cur_option_char,
                                    #"answer_id": ans_id,
                                    "model_id": model_name,
                                    "is_correct": is_correct
                                    }) + "\n")
            ans_file.flush()

            # rotate options
            options = options[1:] + options[:1]
            cur_option_char = cur_option_char[1:] + cur_option_char[:1]
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
    parser.add_argument("--all-rounds", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--white-image", action='store_true', help="Use a white image if set")
    parser.add_argument("--sname", type=str, default="maskout-set-in-code")
    args = parser.parse_args()

    eval_model(args)
