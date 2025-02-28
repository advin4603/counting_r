"""
Saves hidden states for train_probe_individual.py to train a linear probe.
"""

import json
import random
import tqdm
import numpy as np
import torch
import h5py
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.corpus import wordnet
import nltk
from collections import defaultdict

# Download WordNet if not already present
nltk.download('wordnet')

# The specific letter to count
target_letter = "e"  # You can change this to any letter

# Load tokenizer and model
model_id = "meta-llama/Llama-3.2-3B"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.eval()

# Dynamically retrieve hidden state size and number of layers
hidden_state_dim = model.config.hidden_size
# Assume the model returns all layers (embeddings + transformer layers)
num_layers = model.config.num_hidden_layers + 1

# Get words from WordNet with length restriction (less than 10 characters)
all_words = list(wordnet.all_synsets())
words = [lemma.name() for synset in all_words for lemma in synset.lemmas()
         if "_" not in lemma.name() and len(lemma.name()) < 10]

# Pre-process words to create buckets by letter count (1-9)
word_buckets = defaultdict(list)
for word in words:
    count = min(word.lower().count(target_letter.lower()), 9)
    if count > 0:  # Only include words that have at least one occurrence
        word_buckets[count].append(word)

# Check if we have words for all counts 1-9
for count in range(1, 10):
    if count not in word_buckets or len(word_buckets[count]) == 0:
        print(
            f"Warning: No words found with exactly {count} occurrences of letter '{target_letter}' that are less than 10 characters")
    else:
        print(
            f"Found {len(word_buckets[count])} words with {count} occurrences of letter '{target_letter}'")

templates = [
    f"The number of {target_letter}'s in <count_subject> is <count>",
    f"Counting the letter {target_letter} in <count_subject> gives <count>",
    f"The number of {target_letter}'s found in <count_subject> is <count>",
    f"The {target_letter} count for <count_subject> equals <count>",
    f"The total number of {target_letter}'s in <count_subject> is <count>"
]

# File paths with letter in name
metadata_file = f"letter_{target_letter}_short_count_metadata.jsonl"
hidden_states_file = f"letter_{target_letter}_short_count_hidden_states.h5"

# Initialize metadata file
with open(metadata_file, "w") as f:
    pass  # Create an empty file

# Initialize HDF5 file for hidden states with shape (0, num_layers, hidden_state_dim)
with h5py.File(hidden_states_file, "w") as hf:
    hf.create_dataset("count_subject_hidden_states", shape=(0, num_layers, hidden_state_dim),
                      maxshape=(None, num_layers, hidden_state_dim), dtype=np.float32)

examples_set = set()
# Target total examples (e.g., 5000)
count_examples = {i: 0 for i in range(1, 10)}
delete_counts = []

# Remove counts with too few unique words (less than 300)
for count, words in list(word_buckets.items()):
    if len(words) < 300:
        delete_counts.append(count)
        print(
            f"Note: Only {len(words)} unique words available for count {count}, which is less than target 300")

for count in delete_counts:
    if count in word_buckets:
        del word_buckets[count]

total = 7_000
pbar = tqdm.tqdm(total=total)

while len(examples_set) < total:
    count = random.randint(1, 9)
    if word_buckets.get(count, []):
        word = random.choice(word_buckets[count])
        template = random.choice(templates)
        incomplete_example = template.removesuffix(
            "<count>").replace("<count_subject>", word)
        complete_example = template.replace(
            "<count_subject>", word).replace("<count>", str(count))
        if complete_example not in examples_set:
            # Tokenize input
            inputs = tokenizer(incomplete_example, return_tensors="pt")
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            # Find start and end indices for the target word
            start_idx = None
            for i, token in enumerate(tokens):
                cleaned_token = token.strip().replace("▁", "")
                if cleaned_token and (word.startswith(cleaned_token) or cleaned_token in word):
                    start_idx = i
                    break
            if start_idx is None:
                continue
            word_tokens = tokenizer(word, add_special_tokens=False)[
                "input_ids"]
            end_idx = start_idx + len(word_tokens) - 1

            # Get model predictions with all hidden states
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                logits = outputs.logits
                hidden_states = outputs.hidden_states

            # For each layer, sum the token representations over the token span
            layer_representations = []
            for layer_state in hidden_states:
                # shape: [token_count, hidden_state_dim]
                token_reps = layer_state[0, start_idx:end_idx+1]
                # shape: [hidden_state_dim]
                aggregated = torch.sum(token_reps, dim=0)
                layer_representations.append(aggregated)
            # Stack to get a tensor of shape (num_layers, hidden_state_dim)
            hidden_concat = torch.stack(layer_representations)

            # Save metadata
            metadata = {
                "word": word,
                "word_length": len(word),
                "target_letter": target_letter,
                "count": count,
                "incomplete_example": incomplete_example,
                "complete_example": complete_example,
                "tokenized_input": tokens,
                "indices": {"start": start_idx, "end": end_idx},
                "model_prediction": tokenizer.decode(logits[0, -1].argmax(dim=-1).item()),
            }
            with open(metadata_file, "a") as f:
                f.write(json.dumps(metadata) + "\n")

            # Save hidden states incrementally
            with h5py.File(hidden_states_file, "a") as hf:
                count_hs_dataset = hf["count_subject_hidden_states"]
                new_shape = (
                    count_hs_dataset.shape[0] + 1, num_layers, hidden_state_dim)
                count_hs_dataset.resize(new_shape)
                count_hs_dataset[-1:] = hidden_concat.cpu().numpy().reshape(1,
                                                                            num_layers, hidden_state_dim)

            word_buckets[count].remove(word)
            examples_set.add(complete_example)
            count_examples[count] += 1
            pbar.update(1)
            pbar.set_description(f"Counts: {dict(count_examples)}")
pbar.close()

print(
    f"\nGenerated examples for letter '{target_letter}' counts with words less than 10 characters:")
for count, num in sorted(count_examples.items()):
    print(f"Count {count}: {num} examples")
print(f"Total: {sum(count_examples.values())} examples")
print("\nDataset complete!")
