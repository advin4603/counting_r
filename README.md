# From Early Encoding to Late Suppression: Interpreting LLMs on Character Counting Tasks

This repo contains accompanying code for the AACL-2026 Findings paper: From Early Encoding to Late Suppression: Interpreting LLMs on
Character Counting Tasks.

## Abstract
Large language models (LLMs) still fail at counting the characters in a word, even while they excel on far harder benchmarks. Using character counting (e.g., "How many `p's are in apple?") as a controlled task, we find a consistent pattern across Llama, Qwen, and Gemma in both base and instruction-tuned variants. The models compute the correct answer internally but fall back on degenerate output strategies, such as always predicting "1" or guessing uniformly among "1", "2", and "3". Probing classifiers, activation patching, and logit lens analysis show that character-level information is encoded in early and mid layers, then actively attenuated by a small set of later components, mainly the penultimate and final layer MLP and sometimes the final layer attention heads. The failure is not missing information or insufficient scale but structured interference in the computation graph, which explains why these errors persist under scaling and instruction tuning.

## Usage
- Ensure python >= 3.10
- Use uv to manage dependencies from pyproject.toml
