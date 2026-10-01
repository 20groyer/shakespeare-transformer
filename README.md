# Shakespeare GPT: A Transformer Language Model Built From Scratch

A **character-level, decoder-only Transformer** implemented in PyTorch and trained from scratch on a consumer laptop (MacBook Air) to generate Shakespeare-style text.

> Final project for **CS 307: Artificial Intelligence** by **Gideon Royer**, May 2026.

| | |
|---|---|
| **Language / Framework** | Python 3.12, PyTorch |
| **Model** | Decoder-only Transformer (same family as GPT-2) |
| **Final model size** | ~3.26M parameters |
| **Training data** | Tiny Shakespeare (~1 MB, ~40,000 lines) |
| **Hardware** | Consumer laptop (Apple Silicon GPU via MPS; no cloud compute) |
| **Best result** | Train loss 4.27 → **1.24**, validation loss 4.28 → **1.53** over 5,000 steps |

---

## Highlights

- Implemented the core pieces of a modern LLM by hand: **tokenization, multi-head causal self-attention, feed-forward blocks, residual connections, layer normalization, dropout, a training loop, and autoregressive text generation**.
- Ran a **controlled series of experiments** (training length, embedding size) and tracked train/validation loss to see what actually moved the needle.
- Adapted the training configuration to **hardware constraints**, scaling the model down from ~10.8M to a size a laptop could train, and added Apple Silicon (MPS) device support.
- Read and cross-referenced the original paper, *Attention Is All You Need* (Vaswani et al., 2017), alongside the tutorial to understand the *why* behind each component, not just the code.

## Sample output

After 5,000 training steps, the model has learned the *shape* of a Shakespeare play: speaker names in capitals, line breaks, verse-like rhythm, and archaic vocabulary. It does this purely from character patterns, with no built-in notion of words or grammar:

```
HASTINGS:
Tranger sorrowly earn no malace and have all of two
that praspect repung; and much chiet, becarousen
Rave of swiness the langually with you: Tears,
```

Many of the words are invented, which is expected for a small character-level model. The structure, however, is clearly Shakespearean.

## Results

Each run changed one thing at a time. All runs use `n_head = 4`, `n_layer = 4`, `block_size = 256`, `dropout = 0.2`, and `learning_rate = 3e-4`.

| Run | Change | `batch_size` | `max_iters` | `n_embd` | Params | Final train loss | Final val loss |
|---|---|---|---|---|---|---|---|
| 0 | Tutorial defaults (`n_layer = 6`, `n_head = 6`) | 64 | 5000 | 384 | 10.79M | not completed (too slow on a laptop) | not completed |
| 1 | Scaled down to fit laptop | 32 | 3000 | 64 | 0.22M | 1.678 | 1.840 |
| 2 | More training steps | 32 | 5000 | 64 | 0.22M | 1.569 | 1.749 |
| 3 | Larger embeddings | 32 | 5000 | 128 | 0.84M | 1.385 | 1.612 |
| 4 | Larger embeddings again | 32 | 5000 | **256** | **3.26M** | **1.239** | **1.530** |

**What the numbers say**

- **Model capacity mattered more than training time.** Going from 3,000 to 5,000 steps improved validation loss by ~0.09. Going from 64 to 256 embedding dimensions improved it by a further ~0.22.
- **Scaling worked, with diminishing returns.** Each doubling of `n_embd` helped, but the gains shrank.
- **There is a visible train/validation gap** (1.24 vs 1.53 in the best run). With only ~1 MB of text, the model is starting to memorize rather than generalize, which points to more data as the next lever (see Future Work).

## How it works

1. **Tokenization:** every unique character in the text becomes an integer ID (a vocabulary of 65 characters). Text becomes numbers the model can process.
2. **The Transformer:** token and position embeddings feed into a stack of blocks. Each block contains multi-head causal self-attention (every character can look at the characters before it, never ahead), followed by a feed-forward network, with residual connections and layer norm around each.
3. **Training loop:** sample a random batch, predict the next character at every position, compare against the truth with cross-entropy loss, and update weights with backpropagation (AdamW).
4. **Generation:** start from a single seed character, sample the next character from the predicted probability distribution, append it, and repeat.

The model is **decoder-only**, the same family as the GPT models.

## Getting started

```bash
# 1. Clone the repo
git clone https://github.com/20groyer/shakespeare-transformer.git
cd shakespeare-transformer

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install PyTorch
pip install torch

# 4. Train and generate
python gpt.py
```

`gpt.py` automatically uses an NVIDIA GPU (CUDA), Apple Silicon GPU (MPS), or CPU, in that order. It prints the parameter count and train/validation loss every 500 steps, then generates 500 characters of text at the end.

To change the experiment, edit the hyperparameters at the top of `gpt.py`. The `# was ...` comments show the original tutorial values.

> **Heads up:** training on CPU alone is slow. If you just want to experiment quickly, lower `max_iters` or `n_embd`.

## Repository contents

| File | Description |
|---|---|
| `gpt.py` | Model, training loop, and text generation (single file) |
| `input.txt` | Training corpus: Tiny Shakespeare |
| `README.md` | This file |

## What I learned

- **Transformers and attention.** Multi-head self-attention, feed-forward layers, residual connections, layer normalization, and dropout, the building blocks of modern deep learning, and how they fit together.
- **Tokenization.** What tokens actually are, why this project uses characters, and why production models like GPT use subword tokens instead.
- **Encoder vs. decoder.** The architectural difference, what each looks like in code, and what it means in practice. GPT-style models are decoder-only, like the one built here.
- **Baselines matter.** A simple bigram model served as a baseline to appreciate how much the Transformer improves on naive approaches.
- **Matrices and optimization.** How batched matrix operations make training efficient, plus hands-on experience with PyTorch and with managing a Python training environment (including debugging a broken virtual environment along the way).
- **Working within constraints.** Making real tradeoffs between model size, training time, and the hardware available.

## Limitations

- **Small, single-source dataset:** one ~1 MB corpus of Shakespeare.
- **Consumer hardware:** a laptop capped both model size and training time.
- **Scale:** production language models are orders of magnitude larger in parameters, data, and compute.
- **Character-level modeling:** the model has no word-level understanding, so it often produces plausible-looking but invented words.

## Future work

- **More and varied training data**, beyond Shakespeare.
- **A larger model.** Layers and embedding dimension had the biggest effect on validation loss, so continuing to scale them is the obvious next step.
- **Dedicated compute.** A GPU or cloud platform (Google Colab Pro, AWS, Lambda Labs) would make much larger experiments feasible.
- **A Swahili language model.** This project made me curious about generating text in my own language, **Swahili**, and training a model to write a poem or something similar. Swahili has a rich poetic tradition, and its word structure (words built by adding prefixes and suffixes to a root) is an interesting test of character-level versus subword modeling. The first steps would be finding a clean, properly licensed Swahili corpus and comparing character-level against subword tokenization.

## Conclusions

- **The Transformer is approachable.** The principles behind today's large language models can be built and understood in a few hundred lines of Python.
- **Effective AI at scale is a resource problem as much as an algorithmic one.** The biggest improvements in this project came from giving the model more capacity.
- **Hands-on implementation turns surface-level familiarity into real understanding.** Building the model took me from using AI concepts to understanding how they work.

## Acknowledgments

- Andrej Karpathy's [*Let's build GPT*](https://github.com/karpathy/ng-video-lecture) tutorial and code (MIT License), the foundation of this project.
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017).
- Tiny Shakespeare dataset, from Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).

## About the author

**Gideon Royer**: CS 307: Artificial Intelligence final project. Feel free to reach out via [GitHub](https://github.com/20groyer) or [LinkedIn](https://www.linkedin.com/in/gideon-royer).
