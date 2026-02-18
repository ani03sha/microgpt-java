# MicroGPT Java

A Java implementation inspired by Andrej
Karpathy's [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a minimal transformer
language model built entirely from scratch with **no external ML dependencies**.

The project trains a character-level language model on a dataset of names and generates new, plausible-sounding names.
More importantly, it is structured as a **step-by-step educational journey** — starting from the simplest statistical
baseline and progressively building up to a full GPT-style transformer.

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

```
tokens
  └─ token embedding + positional embedding
       └─ RMSNorm
            └─ N × TransformerBlock
                    ├─ RMSNorm → Attention → residual
                    └─ RMSNorm → MLP (expand→ReLU→contract) → residual
                         └─ RMSNorm
                              └─ Linear (output head)
                                   └─ logits [vocabularySize]
```

**Hyperparameters** (matching Karpathy's gist):

- Vocabulary size: 27 (a–z + BOS)
- Embedding dimension: 16
- Context window (block size): 16
- Attention heads: 4
- Transformer layers: 1
- Total parameters: ~4,256

---

## Project Structure

```
src/main/java/com/anirudhology/microgpt/
│
├── Runner.java                          Entry point — runs all 6 steps
│
├── autograd/
│   └── Value.java                       Scalar autograd engine (the foundation)
│
├── tokenizer/
│   └── CharacterTokenizer.java          Character-level tokenizer with BOS token
│
├── data/
│   ├── TextCorpus.java                  Downloads and reads names.txt
│   ├── NGramDatasetBuilder.java         Sliding-window (context, target) pairs
│   └── TrainingExample.java             Record: int[] context + int target
│
├── nn/                                  Neural network building blocks
│   ├── Linear.java                      Fully-connected layer (no bias)
│   ├── Embedding.java                   Token embedding lookup table
│   ├── PositionalEmbedding.java         Positional embedding lookup table
│   ├── RMSNormalization.java            RMSNorm with learnable gamma
│   ├── Attention.java                   Interface for attention mechanisms
│   ├── CausalSelfAttention.java         Single-head causal self-attention
│   ├── MultiHeadCausalSelfAttention.java  Multi-head causal self-attention
│   └── TransformerBlock.java            Pre-Norm block: attention + MLP + residuals
│
├── model/                               Full language models
│   ├── BaselineBigramModel.java         Step 1
│   ├── NeuralBigramModel.java           Step 2
│   ├── NeuralBigramAutogradModel.java   Step 3
│   ├── MLPLanguageModel.java            Step 4
│   └── GPTLanguageModel.java            Steps 5–6
│
└── optimizer/
    └── AdamOptimizer.java               Adam with bias correction
```

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
  `MultiHeadCausalSelfAttention` transparently, making the single→multi-head progression visible in code

---

## Learn More

See [CONCEPTS.md](CONCEPTS.md) for a detailed walkthrough of every concept in the project — tokenization, autograd,
attention, transformers, Adam, and all the design decisions — explained in logical order from basic to advanced.

---

## Acknowledgements

Inspired by [Andrej Karpathy's microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95). The
original Python implementation achieves the same thing in ~200 lines. This Java port expands it into a structured,
educational codebase while preserving the spirit of the original.
