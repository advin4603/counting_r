import transformer_lens
import torch

from collections import defaultdict
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


import nltk

import random
from nltk.corpus import wordnet
import argparse
from pathlib import Path

# take model id from command line argument
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model ID from HuggingFace")
args = parser.parse_args()
MODEL_NAME = args.model_id

FOLDER = Path(MODEL_NAME.replace("/", "_"))
FOLDER.mkdir(exist_ok=True)
random.seed(42)


nltk.download('wordnet')

torch.set_grad_enabled(False)

# model_id = "Qwen/Qwen2.5-3B"
model_id = MODEL_NAME
model = transformer_lens.HookedTransformer.from_pretrained(model_name=model_id, device="cuda")

from transformer_lens import HookedTransformer, ActivationCache
from fancy_einsum import einsum
from jaxtyping import Float
import numpy as np

def get_logits(incomplete_example: str, answer_set: list[str], model: HookedTransformer) -> tuple[Float[np.ndarray, "answer layer"], list[str]]:
    incomplete_tokens = model.to_tokens(incomplete_example + answer_set[0], prepend_bos=False)[:, :-1]
    answer_tokens = model.to_tokens([incomplete_example + answer for answer in answer_set], prepend_bos=False)[:, -1]
    directions = model.tokens_to_residual_directions(answer_tokens)
    
    _, cache = model.run_with_cache(incomplete_tokens)
    accumulated_resid, labels = cache.accumulated_resid(layer=-1, incl_mid=True, pos_slice=-1, apply_ln=True, return_labels=True)
    
    return einsum("layers d_model, batch d_model -> batch layers", accumulated_resid[:, 0], directions).cpu().numpy(), labels

def normalize_logits(logits: Float[np.ndarray, "answer layer"]) -> Float[np.ndarray, "answer layer"]:
    min_logit = logits.min()
    max_logit = logits.max()
    return (logits - min_logit) / (max_logit - min_logit)


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



def get_words(max_len=10):
    all_words = list(wordnet.all_synsets())
    words = [lemma.name() for synset in all_words for lemma in synset.lemmas()
             if "_" not in lemma.name() and len(lemma.name()) < max_len]
    return list(set(words))

def sample_word(words: list[str], exclude: None | list[str]=None, model: HookedTransformer=None, token_count: int=None):
    candidates = [w for w in words]
    while True:
        choice = random.choice(candidates)
        if (exclude is None or choice not in exclude) and (model is None or len(model.to_tokens(choice, prepend_bos=False)[0]) == token_count):
            return choice
        
def sample_letter(word):
    letters = list(set([c.lower() for c in word if c.isalpha()]))
    return random.choice(letters) if letters else None

def make_prompt(template, word, count, letter):
    return template.replace("<count_subject>", word).replace("<count>", "").replace("<letter>", letter)



answer_logits = {"mean": None, "sq_mean": None, "count": 0}
answer_template_logits = defaultdict(lambda : {"mean": None, "sq_mean": None, "count": 0})

def plot_logit_lens(answer_logits: dict[str, dict[str, float]], answer_template_logits: dict[str, dict[int, dict[str, float]]], labels: list[str], answer_set: list[str]):
    
    mean_logits = answer_logits["mean"]
    sq_mean_logits = answer_logits["sq_mean"]
    if mean_logits is None or sq_mean_logits is None:
        return
    var_batch = sq_mean_logits - mean_logits**2
    var_batch = np.maximum(var_batch, 0.0)
    std_logits = np.sqrt(var_batch)
    
    plt.figure(figsize=(12, 6))
    # Plot a line with error boundary for this answer using sns
    sns.lineplot(x=labels, y=mean_logits, marker='o')
    plt.fill_between(labels, mean_logits - std_logits, mean_logits + std_logits, alpha=0.3, label='±1 Std Dev')

    plt.xlabel('Layer')
    plt.ylabel('Logit Diff $\Delta$')
    plt.title(f'Logit Diff Evolution through Layers')
    plt.xticks(
        ticks=np.arange(len(labels))[::2],   # every 2nd position
        labels=[labels[i] for i in range(0, len(labels), 2)],
        rotation=90
    )
    plt.legend()
    plt.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
    plt.tight_layout()
    plt.savefig(str(FOLDER / 'logit_lens.pdf'))
    plt.close()
    
    for template, logit_dict in answer_template_logits.items():
        mean_logits = logit_dict["mean"]
        sq_mean_logits = logit_dict["sq_mean"]
        if mean_logits is None or sq_mean_logits is None:
            continue
        var_batch = sq_mean_logits - mean_logits**2
        var_batch = np.maximum(var_batch, 0.0)
        std_logits = np.sqrt(var_batch)
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=labels, y=mean_logits, marker='o')
        plt.fill_between(labels, mean_logits - std_logits, mean_logits + std_logits, alpha=0.3, label='±1 Std Dev')

        plt.xlabel('Layer')
        plt.ylabel('Logit Diff $\Delta$')
        plt.title(f'Logit Diff Evolution through Layer for Template {template}')
        plt.xticks(
            ticks=np.arange(len(labels))[::2],   # every 2nd position
            labels=[labels[i] for i in range(0, len(labels), 2)],
            rotation=90
        )
        plt.legend()
        plt.grid(True, which='both', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
        plt.tight_layout()
        plt.savefig(str(FOLDER / f'logit_lens_template_{template}.pdf'))
        plt.close()

# add tqdm progress bar
pbar = tqdm(total=0, position=0, leave=True)
done = 0
while True:
    template = random.choice(get_templates())
    template_index = get_templates().index(template)
    word = sample_word(get_words()).lower()
    letter = sample_letter(word)
    if letter is None:
        continue
    count = str(word.lower().count(letter.lower()))
    answers = ["1", "2", "3"]
    
    if count not in answers:
        continue
    
    prompt = make_prompt(template, word, count, letter)
    logits, labels = get_logits(prompt, answers, model)
    logit_diff = 1.5 * logits[int(count) - 1] - logits.sum(axis=0) / 2
    answer_logits["count"] += 1
    if answer_logits["mean"] is None:
        answer_logits["mean"] = logit_diff
    else:
        delta = logit_diff - answer_logits["mean"]
        answer_logits["mean"] += delta / answer_logits["count"]
    
    if answer_logits["sq_mean"] is None:
        answer_logits["sq_mean"] = (logit_diff ** 2)
    else:
        delta = (logit_diff ** 2) - answer_logits["sq_mean"]
        answer_logits["sq_mean"] += delta / answer_logits["count"]
    
    answer_template_logits[template_index]["count"] += 1
    
    if answer_template_logits[template_index]["mean"] is None:
        answer_template_logits[template_index]["mean"] = logit_diff
    else:
        delta = logit_diff - answer_template_logits[template_index]["mean"]
        answer_template_logits[template_index]["mean"] += delta / answer_template_logits[template_index]["count"]

    if answer_template_logits[template_index]["sq_mean"] is None:
        answer_template_logits[template_index]["sq_mean"] = (logit_diff ** 2)
    else:
        delta = (logit_diff ** 2) - answer_template_logits[template_index]["sq_mean"]
        answer_template_logits[template_index]["sq_mean"] += delta / answer_template_logits[template_index]["count"]

    plot_logit_lens(answer_logits, answer_template_logits, labels, answers)
    pbar.update(1)
    
    done += 1
    
    with open(FOLDER / "logit_lens_count.txt", "w") as f:
        print(done, file=f)



