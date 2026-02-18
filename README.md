# MicroGPT Java

[Andrej Karpathy](https://karpathy.ai/) is a GOAT, and we all know that. He posted a banger on X few days back - A pure,
dependency-free Python implementation of GPT.

![Karpathy's Tweet](/docs/karpathy_tweet.png)

Here's his implementation: [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). It is a
minimal transformer language model built entirely from scratch with **no external ML dependencies**.

I found it very interesting and tried to implement on my own and this project is the result. It is a Java implementation
of Karpathy's MicroGPT with few additions.

## What it does?

The project trains a character-level language model on a dataset of names and generates new, plausible-sounding names.
More importantly, I tried to engineer it as a **step-by-step educational journey**, starting from the simplest
statistical baseline and progressively building up to a full GPT-style transformer.

---

## The Progression

Each step introduces one key idea on top of the previous:

| Step | Model                            | Key Concept                                                    |
|------|----------------------------------|----------------------------------------------------------------|
| 1    | Statistical Bigram               | Count character pairs, Laplace smoothing, probability sampling |
| 2    | Neural Bigram (manual gradients) | Logits, softmax, cross-entropy, SGD                            |
| 3    | Neural Bigram (autograd)         | Computational graph, chain rule, automatic differentiation     |
| 4    | MLP Language Model               | Embeddings, context window, hidden layer                       |
| 5    | GPT (single-head attention)      | Self-attention, causal mask, residual connections, RMSNorm     |
| 6    | GPT (multi-head attention)       | Multi-head attention, Adam optimizer, LR decay                 |

---

## Sample Output

After 1000 training steps on ~32k names:

```
Step 1000 / 1000 | Loss: 2.37 | LR: 0.000010

--- Samples ---
Sample  1: phara
Sample  2: laaran
Sample  3: elaned
Sample  4: maret
Sample  5: aiara
Sample  6: malenan
Sample  7: odanini
Sample  8: tely
Sample  9: keba
Sample 10: ena
```

---

## Architecture

![MicroGPT Architecture](/docs/microgpt_architecture.png)

**Hyperparameters** (matching Karpathy's gist):

- Vocabulary size: 27 (a–z + BOS)
- Embedding dimension: 16
- Context window (block size): 16
- Attention heads: 4
- Transformer layers: 1
- Total parameters: ~4,256

---

## Getting Started

**Prerequisites**: Java 17+, Maven

```bash
git clone https://github.com/your-username/microgpt-java
cd microgpt-java
mvn compile
mvn exec:java -Dexec.mainClass="com.anirudhology.microgpt.Runner"
```

`names.txt` is downloaded automatically on first run from Karpathy's makemore dataset.

---

## Key Design Decisions

- **No external ML dependencies** — everything including autograd, embeddings, attention, and Adam is implemented from
  scratch using only Java standard library
- **No bias in Linear layers** — matches modern transformer practice (LLaMA, Mistral); RMSNorm handles the shift
- **Pre-Norm style** — RMSNorm applied before each sub-layer, more stable than the original Post-Norm transformer
- **Learnable gamma in RMSNorm** — starts at 1.0 (identity), only diverges if the optimizer finds it useful
- **`Attention` interface** — allows `TransformerBlock` to use either `CausalSelfAttention` or
  `MultiHeadCausalSelfAttention` transparently, making the single → multi-head progression visible in code

---

## Learn More

See [CONCEPTS.md](docs/CONCEPTS.md) for a detailed walkthrough of every concept in the project including tokenization,
autograd, attention, transformers, Adam, and all the design decisions, explained in logical order from basic to advanced.

---

## Acknowledgements

Inspired by [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). The
original Python implementation achieves the same thing in ~200 lines. This Java port expands it into a structured,
educational codebase while preserving the spirit of the original.
