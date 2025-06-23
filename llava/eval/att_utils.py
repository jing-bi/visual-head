import hashlib
import math
from pathlib import Path
import random

import pandas as pd


def get_range(model):
    if "7b" in model:
        layer_num = 32
        heads_num = 32
    elif "13b" in model:
        layer_num = 40
        heads_num = 40
    elif "8b" in model:
        layer_num = 24
        heads_num = 32
    elif "phi" in model:
        layer_num = 24
        heads_num = 32
    else:
        raise ValueError(f"Model {model} not supported")
    early = 10 if "7b" in model else 13 if "13b" in model else 8
    mid = 20 if "7b" in model else 27 if "13b" in model else 16
    late = 32 if "7b" in model else 40 if "13b" in model else 24
    return layer_num, heads_num, early, mid, late


def string_to_number(model):
    return int(hashlib.sha256(model.encode()).hexdigest(), 16) % 10**8


def sample_heads(model, stage="all", k=0, top_heads=[]):
    layer_num, heads_num, early, mid, late = get_range(model)

    if stage == "early":
        heads = [(layer, head) for layer in range(early) for head in range(heads_num)]
    elif stage == "mid":
        heads = [(layer, head) for layer in range(early, mid) for head in range(heads_num)]
    elif stage == "late":
        heads = [(layer, head) for layer in range(mid, late) for head in range(heads_num)]
    elif stage == "all":
        heads = [(layer, head) for layer in range(layer_num) for head in range(heads_num)]
    else:
        raise ValueError(f"Stage {stage} not supported")
    others = [(layer, head) for layer, head in heads if (layer, head) not in top_heads]
    seed = string_to_number(model)
    random.seed(seed)
    res = random.sample(others, k)
    return sorted(res)


def get_heads(model, split, stage, type, num, method):
    # split = "point"
    # method = "sum-sub"
    file = Path(f"/home/bij4/vp/insight/heads-{method}/{model}|{split}.csv")
    data = pd.read_csv(file)
    if stage != "all":
        data = data[data["Stage"] == stage]

    if type == "top":
        data = data.sort_values(by=data.columns[-1], ascending=True).head(num)
        return [(row["Layer"], row["Head"]) for idx, row in data.iterrows()]
    if type == "bottom":
        data = data.sort_values(by=data.columns[-1], ascending=True).head(num)
        return [(row["Layer"], row["Head"]) for idx, row in data.iterrows()]
    if type == "others":
        top_heads = [(row["Layer"], row["Head"]) for idx, row in data.iterrows()]
        return sample_heads(model, stage, num, top_heads)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_terminator(model_name, tokenizer):
    if "llama-3" in model_name.lower():
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    elif "phi-3" in model_name.lower():
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")]
    else:
        terminators = [tokenizer.eos_token_id]
    return terminators


if __name__ == "__main__":
    model = "llava-1.5-llama-3-8b"
    stage = "others"
    k = 10
    heads = get_heads(model, "all", k, stage)
    print(heads)
