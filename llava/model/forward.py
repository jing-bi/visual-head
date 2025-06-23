from functools import partial
import math
import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as llama_apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import repeat_kv as llama_repeat_kv
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb as mistral_apply_rotary_pos_emb
from transformers.models.mistral.modeling_mistral import repeat_kv as mistral_repeat_kv

from llava.model.language_model.modeling_phi3 import apply_rotary_pos_emb as phi_apply_rotary_pos_emb
from llava.model.language_model.modeling_phi3 import repeat_kv as phi_repeat_kv
from llava.model.language_model.modeling_phi3 import Phi3Attention
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.language_model.llava_phi3 import LlavaPhi3ForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralForCausalLM
import torch.nn.functional as F
import types
from pathlib import Path
from llava.config import Strategy


import torch
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
def mistralforward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    masks=None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = mistral_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = mistral_repeat_kv(key_states, self.num_key_value_groups)
    value_states = mistral_repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is" f" {attn_weights.size()}"
        )
    attn_weights = modify(masks, attn_weights, attention_mask, self.layer_idx)
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")

        attn_weights = attn_weights + attention_mask
    attn_weights = maskout(attn_weights, self.layer_idx)
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    analyize(attn_weights, self.layer_idx)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_attn_bforward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    masks=None,
):

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = llama_repeat_kv(key_states, self.num_key_value_groups)
    value_states = llama_repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is" f" {attn_weights.size()}"
        )
    attn_weights = modify(masks, attn_weights, attention_mask, self.layer_idx)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")
        attn_weights = attn_weights + attention_mask

    attn_weights = maskout(attn_weights, self.layer_idx)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    analyize(attn_weights, self.layer_idx)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def phi_attn_bforward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    masks=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

    bsz, q_len, _ = hidden_states.size()

    qkv = self.qkv_proj(hidden_states)
    query_pos = self.num_heads * self.head_dim
    query_states = qkv[..., :query_pos]
    key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
    value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

    query_states, key_states = phi_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = phi_repeat_kv(key_states, self.num_key_value_groups)
    value_states = phi_repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is" f" {attn_weights.size()}"
        )
    attn_weights = modify(masks, attn_weights, attention_mask, self.layer_idx)
    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}")
        attn_weights = attn_weights + attention_mask
    attn_weights = maskout(attn_weights, self.layer_idx)
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
    analyize(attn_weights, self.layer_idx)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is" f" {attn_output.size()}")

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def inject_modification(self, highlight_mask):
    if "phi" not in self.model.name_or_path:
        if "mistral" in self.model.name_or_path:
            module_class = MistralAttention
            forward_func = mistralforward
        else:
            module_class = LlamaAttention
            forward_func = llama_attn_bforward
    else:
        module_class = Phi3Attention
        forward_func = phi_attn_bforward

    for module in self.model.modules():
        if isinstance(module, module_class):
            module.forward = types.MethodType(partial(forward_func, masks=highlight_mask), module)


def prepare_llava_hl_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    **kwargs,
):
    if past_key_values:
        input_ids = input_ids[:, -1:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "images": kwargs.get("images", None),
            "masked_token_map": kwargs.get("masked_token_map", None),
        }
    )
    return model_inputs


def llava_hl_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    images: Optional[torch.FloatTensor] = None,
    return_dict: Optional[bool] = None,
    masked_token_map: Optional[torch.LongTensor] = None,
):
    if inputs_embeds is None:
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, past_key_values, labels, images
        )

    self.inject_modification(masked_token_map)
    super_class = (
        LlavaPhi3ForCausalLM
        if "phi3" in self.__class__.__name__.lower()
        else LlavaMistralForCausalLM if "mistral" in self.__class__.__name__.lower() else LlavaLlamaForCausalLM
    )
    return super(super_class, self).forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )


def llava_modify_inf(model):
    model.forward = types.MethodType(llava_hl_forward, model)
    model.prepare_inputs_for_generation = types.MethodType(
        prepare_llava_hl_inputs_for_generation, model
    )
    model.inject_modification = types.MethodType(inject_modification, model)


def visualize(tmap):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.matshow(tmap[0].cpu().numpy(), cmap="viridis")
    fig.colorbar(cax)

    plt.title("Token Map Visualization")
    plt.xlabel("Token Index")
    plt.ylabel("Token Index")

    output_path = Path("/home/bij4/vp") / "token_map_visualization.png"
    plt.savefig(output_path)
    plt.close()


debug = False


def visualize_output(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if debug:
            visualize(result[0:1, :, :])
        return result

    return wrapper


@visualize_output
def att_mask_image_center(mask, causal):
    # b, seq_len, seq_len
    if causal.size(-2) == 1:
        return mask.unsqueeze(1).unsqueeze(2)
    mask = mask.unsqueeze(2)
    mask = mask * mask.transpose(1, 2)
    mask = mask.unsqueeze(1)
    mask[causal != 0] = 0
    return mask


@visualize_output
def att_mask_image_cross(mask):
    row_vector = mask.unsqueeze(2)
    column_vector = mask.unsqueeze(1)
    return row_vector + column_vector > 0


# Set the desired downsampled size, e.g., 64x64 or 128x128


def visualize_attention_matrix(attention_matrix, layer_idx, batch_idx):
    import matplotlib.pyplot as plt

    downsampled_size = (128, 128)

    bzs, num_heads, seq_len, _ = attention_matrix.size()
    for idx in range(bzs):
        image_idx = batch_idx * bzs + idx
        fig, axes = plt.subplots(4, 8, figsize=(20, 10))
        for head_idx in range(num_heads):
            downsampled_attention = F.interpolate(
                attention_matrix[idx, head_idx].unsqueeze(0).unsqueeze(0), size=downsampled_size, mode="bilinear", align_corners=False
            ).squeeze()

            ax = axes[head_idx // 8, head_idx % 8]  # Access subplot in 4x8 grid
            ax.imshow(downsampled_attention.cpu(), cmap="viridis", aspect="auto")
            ax.set_title(f"Head {head_idx + 1}")
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Add spacing for title
        plt.suptitle(f"Layer {layer_idx + 1}, Image {image_idx}")
        output_path = Path("/home/bij4/vp") / f"{image_idx}-{layer_idx}-{head_idx}.png"
        plt.savefig(output_path)
        plt.close(fig)


def plot_single_attention_head(attention_matrix, layer_idx=0, head_idx=0):
    downsampled_size = None
    head_attention = attention_matrix[layer_idx, head_idx]
    if downsampled_size:
        head_attention = (
            F.interpolate(
                head_attention.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                size=downsampled_size,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
        )  # Remove extra dimensions and move to CPU
    else:
        head_attention = head_attention.cpu()  # Ensure it's on CPU for Plotly

    # Convert the tensor to a numpy array for Plotly
    head_attention_np = head_attention.numpy()

    # Create the heatmap using Plotly
    fig = px.imshow(
        head_attention_np,
        color_continuous_scale="viridis",
        labels=dict(color="Attention Score"),
        title=f"Attention Head {head_idx + 1} of Layer {layer_idx + 1}",
    )

    # Customize layout for clarity
    fig.update_layout(width=600, height=600, xaxis_title="Tokens", yaxis_title="Tokens")

    fig.show()


import numpy as np
from scipy.ndimage import zoom
from plotly.subplots import make_subplots


def plot_attention_layer_grid(tensor, target_size=128):
    """
    Creates a large grid visualization of all attention layers using plotly.

    Args:
        tensor: torch.Tensor or numpy array of shape (n_layers, n_heads, height, width)
        target_size: int, size to resize each attention map to (default 128)
    """
    n_layers, n_heads = tensor.shape[:2]

    # Create subplots grid
    fig = make_subplots(
        rows=n_layers,
        cols=n_heads,
        subplot_titles=[f"Layer {l+1}, Head {h+1}" for l in range(n_layers) for h in range(n_heads)],
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )

    for layer in range(n_layers):
        for head in range(n_heads):
            # Resize attention map
            attn_map = (
                F.interpolate(
                    torch.tensor(tensor[layer, head], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                    size=(target_size, target_size),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze()
                .numpy()
            )

            # Add heatmap
            fig.add_trace(go.Heatmap(z=attn_map, colorscale="viridis", showscale=False), row=layer + 1, col=head + 1)

    # Update layout
    fig.update_layout(height=200 * n_layers, width=200 * n_heads, title="Complete Attention Visualization Grid", showlegend=False)

    return fig


def plot_headmap(map):
    fig = px.imshow(map, color_continuous_scale="viridis")
    fig.show()


def analyize(weight, layer_idx):
    s = Strategy()
    if s.input_len == 0:
        s.input_len = weight.size(-1)
    if weight.size(-1) - s.input_len == 1 and s.capture:
        folder = Path("../attention") / s.model / f"{s.name}"
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(weight.cpu(), folder / f"{s.batch_idx}-{layer_idx}.pt")


def maskout(attention, layer_idx):
    s = Strategy()
    if "maskout" not in s.name:
        return attention
    heads_layer = [h[1] for h in s.heads if h[0] == layer_idx]
    if len(heads_layer) == 0:
        return attention
    attention[:, heads_layer, :, :] = 0
    return attention


def modify(mask, attention, causal, layer_idx):
    s = Strategy()
    if s.highlight in ["visual", "plain"]:
        return attention
    if attention.size(-2) == 1:
        if s.step == "once":
            return attention
        else:
            mask = F.pad(mask, (0, attention.size(-1) - mask.size(-1)), "constant", 0)
    if s.map == "center":
        mask = att_mask_image_center(mask, causal)
        mask = torch.where(mask != 0, s.value, 1.0)
        attention.mul_(mask)
    return attention
