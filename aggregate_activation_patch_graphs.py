import os


os.environ["HF_TOKEN"] = "<HF TOKEN HERE>"

import numpy as np
import pickle as pkl

from transformers import AutoTokenizer
from pathlib import Path
import argparse

# take model id from command line argument
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B", help="Model ID from HuggingFace")
args = parser.parse_args()
MODEL_NAME = args.model_id


FOLDER = Path(MODEL_NAME.replace("/", "_"))


with open(FOLDER / "token_template_wise_letter_patch_mean.pkl", "rb") as f:
    token_template_wise_letter_patch_mean = pkl.load(f)

with open(FOLDER / "token_template_wise_word_patch_mean.pkl", "rb") as f:
    token_template_wise_word_patch_mean = pkl.load(f)
    
with open(FOLDER / "token_template_wise_both_patch_mean.pkl", "rb") as f:
    token_template_wise_both_patch_counts = pkl.load(f)

with open(FOLDER / "attention_head_word_patch_mean.pkl", "rb") as f:
    attention_head_word_patch_info = pkl.load(f)
    attention_head_word_patch_mean = attention_head_word_patch_info["mean"]
    attention_head_word_patch_count = attention_head_word_patch_info["count"]
    
with open(FOLDER / "attention_head_letter_patch_mean.pkl", "rb") as f:
    attention_head_letter_patch_info = pkl.load(f)
    attention_head_letter_patch_mean = attention_head_letter_patch_info["mean"]
    attention_head_letter_patch_count = attention_head_letter_patch_info["count"]

with open(FOLDER / "attention_head_both_patch_mean.pkl", "rb") as f:
    attention_head_both_patch_info = pkl.load(f)
    attention_head_both_patch_mean = attention_head_both_patch_info["mean"]
    attention_head_both_patch_count = attention_head_both_patch_info["count"]


templates = [
        f"The number of <letter>'s in <count_subject> is <count>",
        f"Counting the letter <letter> in <count_subject> gives <count>",
        f"The number of <letter>'s found in <count_subject> is <count>",
        f"The <letter> count for <count_subject> equals <count>",
        f"The total number of <letter>'s in <count_subject> is <count>"
    ]

template_offsets = [
    {"pre-word": slice(None, 7), "word": slice(7, -2), "post-word": slice(-2,None), "pre-letter": slice(None, 4), "letter": slice(4, 5), "post-letter": slice(5,None)},
    {"pre-word": slice(None, 7), "word": slice(7, -2), "post-word": slice(-2,None), "pre-letter": slice(None, 5), "letter": slice(5, 6), "post-letter": slice(6,None)},
    {"pre-word": slice(None, 8), "word": slice(8, -2), "post-word": slice(-2,None), "pre-letter": slice(None, 4), "letter": slice(4, 5), "post-letter": slice(5,None)},
    {"pre-word": slice(None, 5), "word": slice(5, -2), "post-word": slice(-2,None), "pre-letter": slice(None, 2), "letter": slice(2, 3), "post-letter": slice(3,None)},
    {"pre-word": slice(None, 8), "word": slice(8, -2), "post-word": slice(-2,None), "pre-letter": slice(None, 5), "letter": slice(5, 6), "post-letter": slice(6,None)}
]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

weighted_sum_letter = None
count_letter = 0

weighted_sum_word = None
count_word = 0

weighted_sum_both = None
count_both = 0

def signed_magnitude_mean_last(x):
    weights = np.abs(x)
    numerator = np.sum(weights * x, axis=-1)
    denominator = np.sum(weights, axis=-1)
    out = np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0
    )
    return out

for (token_count, template_index) in token_template_wise_letter_patch_mean.keys():
    offsets = template_offsets[template_index]
    info = token_template_wise_letter_patch_mean[(token_count, template_index)]
    pre_letter_mean = signed_magnitude_mean_last(info["mean"][:,:, offsets["pre-letter"]])
    letter_mean = signed_magnitude_mean_last(info["mean"][:,:, offsets["letter"]])
    post_letter_mean = signed_magnitude_mean_last(info["mean"][:,:, offsets["post-letter"]])

    stacked = np.stack([pre_letter_mean, letter_mean, post_letter_mean], axis=-1)
    count_letter += info["count"]
    if weighted_sum_letter is None:
        weighted_sum_letter = info["count"] * stacked
    else:
        weighted_sum_letter += info["count"] * stacked


for (token_count, template_index) in token_template_wise_word_patch_mean.keys():
    offsets = template_offsets[template_index]
    info = token_template_wise_word_patch_mean[(token_count, template_index)]
    pre_letter_mean = signed_magnitude_mean_last(info["mean"][:,:, offsets["pre-word"]])
    letter_mean = signed_magnitude_mean_last(info["mean"][:,:, offsets["word"]])
    post_letter_mean = signed_magnitude_mean_last(info["mean"][:,:, offsets["post-word"]])

    stacked = np.stack([pre_letter_mean, letter_mean, post_letter_mean], axis=-1)
    count_word += info["count"]
    if weighted_sum_word is None:
        weighted_sum_word = info["count"] * stacked
    else:
        weighted_sum_word += info["count"] * stacked
        
for (token_count, template_index) in token_template_wise_both_patch_counts.keys():
    offsets = template_offsets[template_index]
    info = token_template_wise_both_patch_counts[(token_count, template_index)]
    pre_letter_mean = signed_magnitude_mean_last(info["mean"][:,:, offsets["pre-letter"]])
    letter_mean = signed_magnitude_mean_last(info["mean"][:,:, offsets["letter"]])
    letter_to_word = slice(offsets["letter"].stop, offsets["word"].start)
    letter_to_word_mean = signed_magnitude_mean_last(info["mean"][:,:, letter_to_word])  # for debugging
    word_mean = signed_magnitude_mean_last(info["mean"][:,:, offsets["word"]])
    post_word_mean = signed_magnitude_mean_last(info["mean"][:,:, offsets["post-word"]])

    stacked = np.stack([pre_letter_mean, letter_mean, letter_to_word_mean, word_mean, post_word_mean], axis=-1)
    
    
    count_both += info["count"]
    if weighted_sum_both is None:
        weighted_sum_both = info["count"] * stacked
    else:
        weighted_sum_both += info["count"] * stacked


overall_mean_letter = weighted_sum_letter / count_letter
overall_mean_word = weighted_sum_word / count_word
overall_mean_both = weighted_sum_both / count_both

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


def plot_activation_patch(
    patch: np.ndarray,  # shape [3, layers, parts]
    titles: list[str],  # 3 titles for each part
    part_labels: list[str],  # x axis labels
    save_file: str = None,
    y_step: int = 2
):
    assert patch.shape[0] == 3, "patch should have shape [3, layers, parts]"
    layers = patch.shape[1]
    parts = patch.shape[2]
    cmap = sns.diverging_palette(20, 150, as_cmap=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i in range(3):
        ax = axes[i]
        sns.heatmap(
            patch[i],
            cmap=cmap,
            center=0,
            cbar_kws={'label': 'Normalized Logit Diff'} if i == 2 else None,
            xticklabels=part_labels,
            yticklabels=np.arange(1, layers + 1),
            linewidths=0.3,
            linecolor="lightgray",
            ax=ax
        )
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("Layers")
        else:
            ax.set_ylabel("")
        ax.set_title(titles[i])
        ax.tick_params(axis='x')
        # Show only every `y_step` layer tick
        for j, label in enumerate(ax.get_yticklabels()):
            if j % y_step != 0:
                label.set_visible(False)
        # Add missing outer border (bottom + right)
        ax.add_patch(Rectangle((0, 0), parts, layers,
                               fill=False, edgecolor="lightgray", lw=0.5, clip_on=False))
    plt.tight_layout(pad=0.3)
    if save_file:
        plt.savefig(save_file, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_attention_head_patch(patch: np.ndarray, titles: list[str], save_file: str = None, y_step: int = 2):
    assert patch.shape[0] == 3, "patch should have shape [3, layers, heads]"
    layers = patch.shape[1]
    heads = patch.shape[2]
    cmap = sns.diverging_palette(20, 150, as_cmap=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i in range(3):
        ax = axes[i]
        sns.heatmap(
            patch[i],
            cmap=cmap,
            center=0,
            cbar_kws={'label': 'Normalized Logit Diff'} if i == 2 else None,
            xticklabels=np.arange(1, heads + 1),
            yticklabels=np.arange(1, layers + 1),
            linewidths=0.3,
            linecolor="lightgray",
            ax=ax
        )
        ax.set_xlabel("Attention Heads")
        if i == 0:
            ax.set_ylabel("Layers")
        else:
            ax.set_ylabel("")
        ax.set_title(titles[i])
        ax.tick_params(axis='x')
        # Show only every `y_step` layer tick
        for j, label in enumerate(ax.get_yticklabels()):
            if j % y_step != 0:
                label.set_visible(False)
        # Add missing outer border (bottom + right)
        ax.add_patch(Rectangle((0, 0), heads, layers,
                               fill=False, edgecolor="lightgray", lw=0.5, clip_on=False))
    plt.tight_layout(pad=0.3)
    if save_file:
        plt.savefig(save_file, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def plot_activation_patch(
    patch: np.ndarray,  # shape [3, layers, parts]
    titles: list[str],  # 3 titles for each part
    part_labels: list[str],  # x axis labels
    save_file: str = None,
    y_step: int = 2
):
    assert patch.shape[0] == 3, "patch should have shape [3, layers, parts]"
    layers = patch.shape[1]
    parts = patch.shape[2]
    cmap = sns.diverging_palette(20, 150, as_cmap=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i in range(3):
        ax = axes[i]
        sns.heatmap(
            patch[i],
            cmap=cmap,
            center=0,
            cbar_kws={'label': 'Normalized Logit Diff'} if i == 2 else None,
            xticklabels=part_labels,
            yticklabels=np.arange(1, layers + 1),
            linewidths=0.3,
            linecolor="lightgray",
            ax=ax
        )
        ax.set_xlabel("")
        if i == 0:
            ax.set_ylabel("Layers")
        else:
            ax.set_ylabel("")
        ax.set_title(titles[i])
        ax.tick_params(axis='x')
        # Show only every `y_step` layer tick
        for j, label in enumerate(ax.get_yticklabels()):
            if j % y_step != 0:
                label.set_visible(False)
        # Add missing outer border (bottom + right)
        ax.add_patch(Rectangle((0, 0), parts, layers,
                               fill=False, edgecolor="lightgray", lw=0.5, clip_on=False))
    plt.tight_layout(pad=0.3)
    if save_file:
        plt.savefig(save_file, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

plot_activation_patch(
    overall_mean_letter,
    titles=["residual stream", "attention heads", "MLP"],
    part_labels=["pre-letter", "letter", "post-letter"],
    save_file=str(FOLDER / "activation_patch_letter.pdf"),
)
plot_activation_patch(
    overall_mean_word,
    titles=["residual stream", "attention heads", "MLP"],
    part_labels=["pre-word", "word", "post-word"],
    save_file=str(FOLDER / "activation_patch_word.pdf"),
)

plot_activation_patch(
    overall_mean_both,
    titles=["residual stream", "attention heads", "MLP"],
    part_labels=["pre-letter", "letter", "letter->word", "word", "post-word"],
    save_file=str(FOLDER / "activation_patch_both.pdf"),
)

plot_attention_head_patch(
    np.stack([attention_head_letter_patch_mean, attention_head_word_patch_mean, attention_head_both_patch_mean], axis=0),
    titles=["Patching Letters", "Patching Words", "Patching Both Letters and Words"],
    save_file=str(FOLDER / "attention_head_patch.pdf"),
)

print("Count of letter patches:", attention_head_letter_patch_count)
print("Count of word patches:", attention_head_word_patch_count)
print("Count of both patches:", attention_head_both_patch_count)
