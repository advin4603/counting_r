import os


os.environ["HF_TOKEN"] = "HF_TOKEN_HERE"

import transformer_lens
import torch
import pickle
import argparse
from pathlib import Path

# take model id from command line argument
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model ID from HuggingFace")
args = parser.parse_args()
MODEL_NAME = args.model_id


FOLDER = Path(MODEL_NAME.replace("/", "_"))
FOLDER.mkdir(exist_ok=True)

torch.set_grad_enabled(False)

# %%
model_id = MODEL_NAME
model = transformer_lens.HookedTransformer.from_pretrained(model_name=model_id, device="cuda")

# %%
import random
from nltk.corpus import wordnet
from transformer_lens import HookedTransformer
random.seed(42)

def get_words(max_len=10):
    all_words = list(wordnet.all_synsets())
    words = [lemma.name() for synset in all_words for lemma in synset.lemmas()
             if "_" not in lemma.name() and len(lemma.name()) < max_len]
    return list(set(words))

def sample_word(words: list[str], letter: str, min_count=1, max_count=9, exclude: None | list[str]=None, model: HookedTransformer=None, token_count: int=None):
    candidates = [w for w in words if min_count <=
                  w.lower().count(letter.lower()) <= max_count]
    while True:
        choice = random.choice(candidates)
        if (exclude is None or choice != exclude) and (model is None or len(model.to_tokens(choice, prepend_bos=False)[0]) == token_count):
            return choice
        
def sample_word(words: list[str], exclude: None | list[str]=None, model: HookedTransformer=None, token_count: int=None):
    candidates = [w for w in words]
    while True:
        choice = random.choice(candidates)
        if (exclude is None or choice not in exclude) and (model is None or len(model.to_tokens(choice, prepend_bos=False)[0]) == token_count):
            return choice

def make_prompt(template, word, count, letter):
    return template.replace("<count_subject>", word).replace("<count>", "").replace("<letter>", letter)

def get_templates():
    user_assistant_turns = [
    [
        {"role": "user", "content": "How many <letter>'s are in <count_subject>?"},
        {"role": "assistant", "content": "The number is <count>"}
    ],
    [
        {"role": "user", "content": "Can you count the letter <letter> in <count_subject>?"},
        {"role": "assistant", "content": "The count is <count>"}
    ],
    [
        {"role": "user", "content": "Tell me how many times <letter> appears in <count_subject>."},
        {"role": "assistant", "content": "The number is <count>"}
    ],
    [
        {"role": "user", "content": "What is the count of <letter> in <count_subject>?"},
        {"role": "assistant", "content": "The count is <count>"}
    ],
    [
        {"role": "user", "content": "Please calculate the total number of <letter>'s in <count_subject>."},
        {"role": "assistant", "content": "The total number is <count>"}
    ],
    [
        {"role": "user", "content": "Could you find out how many <letter>'s are in <count_subject>?"},
        {"role": "assistant", "content": "The number is <count>"}
    ],
    [
        {"role": "user", "content": "I want to know the frequency of the letter <letter> in <count_subject>."},
        {"role": "assistant", "content": "The frequency is <count>"}
    ],
    [
        {"role": "user", "content": "Count all the <letter>'s present in <count_subject>."},
        {"role": "assistant", "content": "The total number is <count>"}
    ],
    [
        {"role": "user", "content": "How often does <letter> appear in <count_subject>?"},
        {"role": "assistant", "content": "The number of times is <count>"}
    ]
]
    
    return [model.tokenizer.apply_chat_template(template, add_generation_prompt=False, continue_final_message=True, tokenize=False) for template in user_assistant_turns]

def sample_letter(word):
    letters = list(set([c.lower() for c in word if c.isalpha()]))
    return random.choice(letters) if letters else None

def make_prompt(template, word, count, letter):
    return template.replace("<count_subject>", word).replace("<count>", "").replace("<letter>", letter)

# %%
from jaxtyping import Float
import numpy as np
from transformer_lens import ActivationCache
from torch import Tensor

def get_incomplete_tokens(incomplete_example: str, example_answer: str, model: HookedTransformer) -> tuple[torch.Tensor, torch.Tensor]:
    incomplete_tokens = model.to_tokens(incomplete_example + example_answer, prepend_bos=False)[:, :-1]
    return incomplete_tokens

def get_incomplete_str_tokens(incomplete_example: str, example_answer: str, model: HookedTransformer) -> list[str]:
    incomplete_tokens = model.to_str_tokens(incomplete_example + example_answer, prepend_bos=False)[:-1]
    return incomplete_tokens

def get_logits(incomplete_example: str, answer_set: list[str], model: HookedTransformer) -> tuple[Float[np.ndarray, "answer layer"], list[str]]:
    incomplete_tokens = get_incomplete_tokens(incomplete_example, answer_set[0], model)
    answer_tokens = model.to_tokens([incomplete_example + answer for answer in answer_set], prepend_bos=False)[:, -1]
    
    logits = model(incomplete_tokens)[0,-1, answer_tokens]
    return logits

def get_cache(incomplete_example: str, answer_set: list[str], model: HookedTransformer) -> ActivationCache:
    incomplete_tokens = get_incomplete_tokens(incomplete_example, answer_set[0], model)
    _, cache = model.run_with_cache(incomplete_tokens)
    return cache

def get_logit_diff_from_logits(logits: Float[Tensor, "batch seq_len n_vocab"], answer_set: list[str], answer: str, incomplete_example: str, model: HookedTransformer):
    answer_index = answer_set.index(answer)
    answer_tokens = model.to_tokens([incomplete_example + answer for answer in answer_set], prepend_bos=False)[:, -1]
    answer_token = answer_tokens[answer_index]
    other_tokens = torch.cat([answer_tokens[:answer_index], answer_tokens[answer_index+1:]])
    answer_logit = logits[0, -1, answer_token]
    other_logits = logits[0, -1, other_tokens]
    other_logits_mean = torch.mean(other_logits)
    return answer_logit - other_logits_mean

def get_normalized_logit_diff_from_logits(logits: Float[Tensor, "batch seq_len n_vocab"], answer_set: list[str], answer: str, min_logit_diff: float, max_logit_diff: float, model: HookedTransformer, incomplete_example: str):
    logit_diff = get_logit_diff_from_logits(logits, answer_set, answer, incomplete_example, model)
    return (logit_diff - min_logit_diff) / (max_logit_diff - min_logit_diff)

def get_logit_diff(incomplete_example: str, answer_set: list[str], model: HookedTransformer, answer: str):
    logits = get_logits(incomplete_example, answer_set, model)
    answer_index = answer_set.index(answer)
    answer_logit = logits[answer_index]
    other_logits = torch.cat([logits[:answer_index], logits[answer_index+1:]])
    other_logits_mean = torch.mean(other_logits)
    return (answer_logit - other_logits_mean).cpu().item()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_activation_patch(
    patch: Float[np.ndarray, "layers tokens"],
    title: str,
    tokens: list[str],
    save_file: str = None,
    y_step: int = 2
):
    plt.figure(figsize=(8, 5))
    cmap = sns.diverging_palette(20, 150, as_cmap=True)
    ax = sns.heatmap(
        patch,
        cmap=cmap,
        center=0,
        cbar_kws={'label': 'Normalized Logit Diff'},
        xticklabels=tokens,
        yticklabels=np.arange(1, patch.shape[0] + 1),
        linewidths=0.3,        # thinner gridlines
        linecolor="lightgray"  # subtle gray
    )

    ax.set_xlabel("")  
    ax.set_ylabel("Layers")
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')

    # Show only every `y_step` layer tick
    for i, label in enumerate(ax.get_yticklabels()):
        if i % y_step != 0:
            label.set_visible(False)

    # Add missing outer border (bottom + right)
    ax.add_patch(Rectangle((0, 0), patch.shape[1], patch.shape[0],
                           fill=False, edgecolor="lightgray", lw=0.5, clip_on=False))

    plt.tight_layout(pad=0.3)
    if save_file:
        plt.savefig(save_file, bbox_inches="tight")
    else:
        plt.show()
    
    plt.close()



def plot_all_patches(token_template_wise_mean: dict[tuple[int, int], dict], corruption_type: str):
    for (token_count, template), data in token_template_wise_mean.items():
        mean_patch = data["mean"]
        representative_example = data["representative_example"]
        if mean_patch is not None and representative_example is not None:
            for index, patch_type in enumerate(["Residual Stream", "Post MLP", "Post Attention"]):
                if index >= mean_patch.shape[0]:
                    break
                mean_patch_type = mean_patch[index]
                title = f"{patch_type} Activation Patching {corruption_type.title()} for template {template} and token count {token_count}"
                save_file = str(FOLDER / f"{corruption_type}_activation_patch_{patch_type.replace(' ', '_').lower()}_template_{template}_count_{token_count}.pdf")
                plot_activation_patch(mean_patch_type, title, representative_example, save_file=save_file)

# %%
from tqdm import tqdm
from collections import defaultdict

LOGIT_DIFF_CUTOFF = 1.5
NEGATIVE_SAMPLE_BATCH_SIZE = 15

LOGIT_DIFF_WORD_CORRUPTION_CUTOFF = 0.5
LOGIT_DIFF_LETTER_CORRUPTION_CUTOFF = 0.5


token_template_wise_word_patch_mean = defaultdict(lambda: {"count": 0, "mean": None, "representative_example": None})

token_template_wise_letter_patch_mean = defaultdict(lambda: {"count": 0, "mean": None, "representative_example": None})

token_template_wise_both_patch_mean = defaultdict(lambda: {"count": 0, "mean": None, "representative_example": None})

attention_head_word_patch_mean = None
attention_head_letter_patch_mean = None
attention_head_both_patch_mean = None

attention_head_word_patch_count = 0
attention_head_letter_patch_count = 0
attention_head_both_patch_count = 0

pbar = tqdm(total=0, position=0, leave=True, desc="Activation Patching")
done = set()
while True:
    template = random.choice(get_templates())
    template_index = get_templates().index(template)
    word = sample_word(get_words())
    letter = sample_letter(word)
    if letter is None:
        continue
    count = str(word.lower().count(letter.lower()))
    answers = ["1", "2", "3"]
    
    if count not in answers:
        continue
    
    prompt  = make_prompt(template, word, count, letter)
    if prompt in done:
        continue
    done.add(prompt)
    
    logit_diff = get_logit_diff(prompt, answers, model, count)
    if logit_diff < LOGIT_DIFF_CUTOFF:
        continue
    token_count = model.to_tokens(prompt, prepend_bos=False).shape[1]
    negative_done = set()
    
    
    pbar_inner = tqdm(total=0, position=0, leave=True, desc=f"Finding negatives for {word}, {template_index}")
    
    while True:
        candidates = []
        for _ in range(NEGATIVE_SAMPLE_BATCH_SIZE):
            negative_word = sample_word(get_words(), [word])
            negative_word_prompt = make_prompt(template, negative_word, count, letter)
            if model.to_tokens(negative_word_prompt, prepend_bos=False).shape[1] != token_count:
                continue
            if negative_word_prompt in negative_done:
                continue
            negative_done.add(negative_word_prompt)
            negative_logit_diff = get_logit_diff(negative_word_prompt, answers, model, count)
            candidates.append((negative_word, negative_logit_diff))
        if candidates:
            best_candidate_word, best_negative_logit_diff = min(candidates, key=lambda x: x[1])
            if logit_diff - best_negative_logit_diff >= LOGIT_DIFF_WORD_CORRUPTION_CUTOFF:
                negative_word = best_candidate_word
                negative_word_logit_diff = best_negative_logit_diff
                break
            candidates.clear()
            
        pbar_inner.update(1)
    pbar_inner.close()
    
    normal_cache = get_cache(prompt, answers, model)
    
    negative_word_tokens = get_incomplete_tokens(make_prompt(template, negative_word, count, letter), answers[0], model)
    
    original_tokens = get_incomplete_str_tokens(prompt, answers[0], model)
    
    corrupted_word_every_block_patch: Float[Tensor, "three layers tokens"] = transformer_lens.patching.get_act_patch_block_every(model, negative_word_tokens, normal_cache, lambda l: get_normalized_logit_diff_from_logits(l, answers, count, negative_word_logit_diff, logit_diff, model, prompt))

    token_template_wise_word_patch_mean[token_count, template_index]["count"] += 1
    if token_template_wise_word_patch_mean[token_count, template_index]["mean"] is None:
        token_template_wise_word_patch_mean[token_count, template_index]["mean"] = corrupted_word_every_block_patch.cpu().numpy()
    else:
        delta = corrupted_word_every_block_patch.cpu().numpy() - token_template_wise_word_patch_mean[token_count, template_index]["mean"]
        token_template_wise_word_patch_mean[token_count, template_index]["mean"] = token_template_wise_word_patch_mean[token_count, template_index]["mean"] + delta / token_template_wise_word_patch_mean[token_count, template_index]["count"]

    if token_template_wise_word_patch_mean[token_count, template_index]["representative_example"] is None:
        token_template_wise_word_patch_mean[token_count, template_index]["representative_example"] = original_tokens
        
    attention_head_word_patch_count += 1
    corrupted_word_every_head_patch = transformer_lens.patching.get_act_patch_attn_head_out_all_pos(model, negative_word_tokens, normal_cache, lambda l: get_normalized_logit_diff_from_logits(l, answers, count, negative_word_logit_diff, logit_diff, model, prompt))
    if attention_head_word_patch_mean is None:
        attention_head_word_patch_mean = corrupted_word_every_head_patch.cpu().numpy()
    else:
        delta = corrupted_word_every_head_patch.cpu().numpy() - attention_head_word_patch_mean
        attention_head_word_patch_mean = attention_head_word_patch_mean + delta / (attention_head_word_patch_count + 1)    
    
    plot_all_patches(token_template_wise_word_patch_mean, "word")
    pbar.update(0.5)


    with open(FOLDER / "token_template_wise_word_patch_mean.pkl", "wb") as f:
        pickle.dump(dict(token_template_wise_word_patch_mean), f)
    
    with open(FOLDER / "attention_head_word_patch_mean.pkl", "wb") as f:
        pickle.dump({"mean": attention_head_word_patch_mean, "count": attention_head_word_patch_count}, f)
    
    negative_candidates = []
    for candidate_negative_letter in "abcdefghijklmnopqrstuvwxyz":
        if candidate_negative_letter == letter:
            continue
        negative_letter_prompt = make_prompt(template, word, count, candidate_negative_letter)
        negative_logit_diff = get_logit_diff(negative_letter_prompt, answers, model, count)
        negative_candidates.append((candidate_negative_letter, negative_logit_diff))
    negative_letter, negative_letter_logit_diff = min(negative_candidates, key=lambda x: x[1])
    
    if logit_diff - negative_letter_logit_diff < LOGIT_DIFF_LETTER_CORRUPTION_CUTOFF:
        continue
    
    
    negative_letter_tokens = get_incomplete_tokens(make_prompt(template, word, count, negative_letter), answers[0], model)
    
    corrupted_letter_every_block_patch: Float[Tensor, "three layers tokens"] = transformer_lens.patching.get_act_patch_block_every(model, negative_letter_tokens, normal_cache, lambda l: get_normalized_logit_diff_from_logits(l, answers, count, negative_letter_logit_diff, logit_diff, model, prompt))
    
    token_template_wise_letter_patch_mean[token_count, template_index]["count"] += 1
    if token_template_wise_letter_patch_mean[token_count, template_index]["mean"] is None:
        token_template_wise_letter_patch_mean[token_count, template_index]["mean"] = corrupted_letter_every_block_patch.cpu().numpy()
    else:
        delta = corrupted_letter_every_block_patch.cpu().numpy() - token_template_wise_letter_patch_mean[token_count, template_index]["mean"]
        token_template_wise_letter_patch_mean[token_count, template_index]["mean"] = token_template_wise_letter_patch_mean[token_count, template_index]["mean"] + delta / token_template_wise_letter_patch_mean[token_count, template_index]["count"]
        
    if token_template_wise_letter_patch_mean[token_count, template_index]["representative_example"] is None:
        token_template_wise_letter_patch_mean[token_count, template_index]["representative_example"] = original_tokens
    
    attention_head_letter_patch_count += 1
    corrupted_letter_every_head_patch = transformer_lens.patching.get_act_patch_attn_head_out_all_pos(model, negative_letter_tokens, normal_cache, lambda l: get_normalized_logit_diff_from_logits(l, answers, count, negative_letter_logit_diff, logit_diff, model, prompt))
    if attention_head_letter_patch_mean is None:
        attention_head_letter_patch_mean = corrupted_letter_every_head_patch.cpu().numpy()
    else:
        delta = corrupted_letter_every_head_patch.cpu().numpy() - attention_head_letter_patch_mean
        attention_head_letter_patch_mean = attention_head_letter_patch_mean + delta / (attention_head_letter_patch_count + 1)    
        
    plot_all_patches(token_template_wise_letter_patch_mean, "letter")
    
    pbar.update(0.5)
    
    
    with open(FOLDER / "token_template_wise_letter_patch_mean.pkl", "wb") as f:
        pickle.dump(dict(token_template_wise_letter_patch_mean), f)
    
    with open(FOLDER / "attention_head_letter_patch_mean.pkl", "wb") as f:
        pickle.dump({"mean": attention_head_letter_patch_mean, "count": attention_head_letter_patch_count}, f)

    pbar_inner = tqdm(total=0, position=0, leave=True, desc=f"Finding double negatives for {word}, {template_index}")

    while True:
        candidates = []
        for _ in range(NEGATIVE_SAMPLE_BATCH_SIZE):
            negative_word = sample_word(get_words(), [word])
            for negative_letter in "abcdefghijklmnopqrstuvwxyz":    
                negative_word_prompt = make_prompt(template, negative_word, count, negative_letter)
                if model.to_tokens(negative_word_prompt, prepend_bos=False).shape[1] != token_count:
                    continue
                if negative_word_prompt in negative_done:
                    continue
                negative_done.add(negative_word_prompt)
                negative_logit_diff = get_logit_diff(negative_word_prompt, answers, model, count)
                candidates.append((negative_word, negative_letter, negative_logit_diff))
        if candidates:
            best_candidate_word, best_negative_letter, best_negative_logit_diff = min(candidates, key=lambda x: x[2])
            if logit_diff - best_negative_logit_diff >= LOGIT_DIFF_WORD_CORRUPTION_CUTOFF:
                negative_word = best_candidate_word
                negative_letter = best_negative_letter
                negative_word_logit_diff = best_negative_logit_diff
                break
            candidates.clear()
            
        pbar_inner.update(1)
    pbar_inner.close()


    negative_both_tokens = get_incomplete_tokens(make_prompt(template, negative_word, count, negative_letter), answers[0], model)
    corrupted_both_every_block_patch: Float[Tensor, "three layers tokens"] = transformer_lens.patching.get_act_patch_block_every(model, negative_both_tokens, normal_cache, lambda l: get_normalized_logit_diff_from_logits(l, answers, count, negative_word_logit_diff, logit_diff, model, prompt))

    token_template_wise_both_patch_mean[token_count, template_index]["count"] += 1
    if token_template_wise_both_patch_mean[token_count, template_index]["mean"] is None:
        token_template_wise_both_patch_mean[token_count, template_index]["mean"] = corrupted_both_every_block_patch.cpu().numpy()
    else:
        delta = corrupted_both_every_block_patch.cpu().numpy() - token_template_wise_both_patch_mean[token_count, template_index]["mean"]
        token_template_wise_both_patch_mean[token_count, template_index]["mean"] = token_template_wise_both_patch_mean[token_count, template_index]["mean"] + delta / token_template_wise_both_patch_mean[token_count, template_index]["count"]

    if token_template_wise_both_patch_mean[token_count, template_index]["representative_example"] is None:
        token_template_wise_both_patch_mean[token_count, template_index]["representative_example"] = original_tokens
    
    attention_head_both_patch_count += 1
    corrupted_both_every_head_patch = transformer_lens.patching.get_act_patch_attn_head_out_all_pos(model, negative_both_tokens, normal_cache, lambda l: get_normalized_logit_diff_from_logits(l, answers, count, negative_word_logit_diff, logit_diff, model, prompt))
    if attention_head_both_patch_mean is None:
        attention_head_both_patch_mean = corrupted_both_every_head_patch.cpu().numpy()
    else:
        delta = corrupted_both_every_head_patch.cpu().numpy() - attention_head_both_patch_mean
        attention_head_both_patch_mean = attention_head_both_patch_mean + delta / (attention_head_both_patch_count + 1)    
    
    plot_all_patches(token_template_wise_both_patch_mean, "both")
    
    
    with open(FOLDER / "token_template_wise_both_patch_mean.pkl", "wb") as f:
        pickle.dump(dict(token_template_wise_both_patch_mean), f)
    
    with open(FOLDER / "attention_head_both_patch_mean.pkl", "wb") as f:
        pickle.dump({"mean": attention_head_both_patch_mean, "count": attention_head_both_patch_count}, f)
    
    pbar.update(0.5)
