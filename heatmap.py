from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.subplots as sp
import plotly.graph_objects as go
import pandas as pd
import torch


folder = Path("insight/scores")
from plotly.subplots import make_subplots


models = [
    "llava-1.5-llama-3-8b",
    "llava-1.5-phi-3-mini-3.8B",
    "llava-v1.6-vicuna-7b",
    "llava-v1.6-vicuna-13b",
    "vip-llava-13b",
    "llava-v1.5-7b",
    "llava-v1.5-13b",
    "vip-llava-7b",
    "llava-v1.6-mistral-7b",
]
# splits = ['plain-general', 'visual-general', 'plain-obj', 'visual-obj','plain-super', 'visual-super']
splits = ["pope", "mm", "visual-general"]
nums = 1

fig = make_subplots(
    rows=len(models) * nums,
    cols=len(splits),
    vertical_spacing=0.05,  # Adjust vertical spacing (0 is no gap, 1 is full gap)
    horizontal_spacing=0.01,
)
from tqdm import tqdm
for i, model in enumerate(tqdm(models)):
    for j, split in enumerate(splits):
        path = folder / model/split / "imatt_sum.pt"
        data = torch.mean(torch.load(path), dim=0).to(torch.float16).to("cpu").numpy()
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        fig.add_trace(
            go.Heatmap(
                z=normalized_data,
                colorscale="portland",
                showlegend=False,  # Hide the legend
                showscale=True,  # Hide the colorbar
            ),
            row=i * nums + 1,
            col=j + 1,
        )

        # Path for second heatmap

        fig.update_xaxes(
            # range=[0, 1],
            tickvals=[0, 32],
            showgrid=False,  # Remove grid lines
            automargin=True,
            zeroline=False,  # Remove axis line at zero
            title_text=f"{model.replace('llava-','')}  {split} Image Attention",  # Title for the x-axis
            row=i * nums + 1,
            col=j + 1,
            title_standoff=0,
            tickfont=dict(size=12),  # Smaller font for tick values
            titlefont=dict(size=16),
        )
        path = folder / model / split / "reatt_sum.pt"
        data = torch.sum(torch.load(path), dim=0).to(torch.float16).to("cpu").numpy()
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        # Second Heatmap: split-region attention
        fig.add_trace(
            go.Heatmap(
                z=torch.sum(torch.load(path), dim=0).to(torch.float16).to("cpu").numpy(),
                colorscale="portland",
                showlegend=False,  # Hide the legend
                showscale=True,  # Hide the colorbar
            ),
            row=i * nums + 2,
            col=j + 1,
        )
        fig.update_xaxes(
            # range=[0, 1],
            tickvals=[0, 32],
            showgrid=False,  # Remove grid lines
            automargin=True,
            zeroline=False,  # Remove axis line at zero
            title_text=f"{split} Region Attention",  # Title for the x-axis
            row=i * 2 + 2,
            col=j + 1,
            title_standoff=0,
            tickfont=dict(size=12),  # Smaller font for tick values
            titlefont=dict(size=16),
        )

fig.update_layout(
    height=200*nums * len(models),  # Adjust height based on number of rows
    width=400*len(splits),  # Adjust width based on your preference
    showlegend=True,
    margin=dict(l=0, r=0, t=0, b=0),  # No margin
)

fig.write_image(f"{models[0]}_attention.png")
