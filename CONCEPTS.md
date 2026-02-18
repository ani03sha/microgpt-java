# MicroGPT Java — Concepts & Progression Guide

A Java implementation of Andrej Karpathy's [MicroGPT](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) — a minimal transformer language model built from scratch with no external ML dependencies.

This document walks through every concept in the project in the order they appear, from the simplest statistical baseline to the full transformer. Each section explains the *what*, the *why*, and the *how*.

---

## Table of Contents

1. [The Task: Language Modeling](#1-the-task-language-modeling)
2. [Tokenization](#2-tokenization)
3. [Data Loading & Dataset Building](#3-data-loading--dataset-building)
4. [Step 1 — Statistical Bigram Model](#4-step-1--statistical-bigram-model)
5. [Step 2 — Neural Bigram (Manual Gradients)](#5-step-2--neural-bigram-manual-gradients)
6. [Step 3 — Automatic Differentiation (Autograd)](#6-step-3--automatic-differentiation-autograd)
7. [Step 4 — MLP Language Model](#7-step-4--mlp-language-model)
8. [Step 5 — GPT Transformer (Single-Head)](#8-step-5--gpt-transformer-single-head)
9. [Step 6 — GPT Transformer (Multi-Head)](#9-step-6--gpt-transformer-multi-head)
10. [Optimization: Adam](#10-optimization-adam)
11. [Design Decisions](#11-design-decisions)
12. [Package Structure](#12-package-structure)

---

## 1. The Task: Language Modeling

The goal is simple: **given a sequence of characters, predict the next character**.

We train on a list of names (e.g. `emma`, `olivia`, `noah`). After training, the model generates new names that look plausible — it has learned the statistical patterns of how characters follow each other in names.

**Why names?** They are short, structured enough to show real learning, but simple enough to train from scratch in minutes.

**Autoregressive generation**: at inference time, we feed the model a context, sample the next character, append it to the context, and repeat — generating one character at a time. This is called *autoregressive* because each output becomes part of the next input.

**Loss metric — Negative Log-Likelihood (NLL)**:

```
loss = -log(P(correct next character))
```

- Perfect prediction (P = 1.0) → loss = 0
- Bad prediction (P = 0.1) → loss = 2.30
- Random guess over 27 chars → loss ≈ 3.30

Lower loss = better model. This is our compass throughout.

---

## 2. Tokenization

**File**: `tokenizer/CharacterTokenizer.java`

We use *character-level tokenization* — every unique character in the dataset becomes a token with an integer ID.

```
a=0, b=1, c=2, ..., z=25, <BOS>=26
```

**BOS (Beginning of Sequence)** is a special token with the highest ID (`vocabSize - 1 = 26`). It serves two purposes:
1. As the *starting signal* when generating — the model begins with BOS and predicts the first character
2. As the *end signal* — the model learns to emit BOS when the name is complete

**`withBOSOnBothSides("emma")`** → `[26, 4, 12, 12, 0, 26]`

This surrounds each name with BOS so the model learns both how names start and how they end.

---

## 3. Data Loading & Dataset Building

**Files**: `data/TextCorpus.java`, `data/NGramDatasetBuilder.java`, `data/TrainingExample.java`

`TextCorpus` downloads `names.txt` from Karpathy's makemore repo if not present, then reads, trims, and shuffles all names.

`NGramDatasetBuilder` turns the documents into a flat list of `(context, target)` pairs using a *sliding window*:

```
Name: "emma" with BOS on both sides → [26, 4, 12, 12, 0, 26]

context=[26, 26, 26], target=4   (predict 'e' from BOS padding)
context=[26, 26,  4], target=12  (predict 'm' from "..e")
context=[26,  4, 12], target=12  (predict 'm' from ".em")
context=[ 4, 12, 12], target=0   (predict 'a' from "emm")
context=[12, 12,  0], target=26  (predict BOS=end from "mma")
```

`TrainingExample` is a simple record:

```java
public record TrainingExample(int[] context, int target) {}
```

---

## 4. Step 1 — Statistical Bigram Model

**File**: `model/BaselineBigramModel.java`

The simplest possible language model: **count how often each character follows each other character**, then normalize.

### Counting
```
counts[i][j] = how many times character j follows character i
```

After seeing all names, we get a 27×27 table of co-occurrence counts.

### Laplace Smoothing
Without smoothing, unseen character pairs would have probability 0, causing `log(0) = -∞`. We add a small constant `alpha` to every count:

```
P(j | i) = (counts[i][j] + alpha) / sum_k(counts[i][k] + alpha)
```

This ensures every transition has at least some probability, acting as a *prior* that all transitions are possible.

### Sampling
To generate a character, we sample from a probability distribution using the **CDF (Cumulative Distribution Function) trick**:

```java
double r = random.nextDouble();  // uniform in [0, 1)
double cdf = 0.0;
for (int i = 0; i < probs.length; i++) {
    cdf += probs[i];
    if (r <= cdf) return i;      // found the sampled character
}
```

This correctly samples each character proportional to its probability.

### Baseline NLL
This model achieves ~2.45 NLL. Any neural model that can't beat this isn't learning anything useful.

---

## 5. Step 2 — Neural Bigram (Manual Gradients)

**File**: `model/NeuralBigramModel.java`

Same prediction task, but instead of a count table we use a **learned weight matrix** and gradient descent.

### Logits
The weight matrix `W[27][27]` holds raw unnormalized scores called *logits*. The value `W[i][j]` represents "how likely is character j to follow character i" — but as a real number (positive or negative), not a probability.

### Softmax
We convert logits to a valid probability distribution:

```
softmax(z)_j = exp(z_j) / sum_k exp(z_k)
```

Properties:
- All outputs are in (0, 1)
- Outputs sum to 1
- Higher logit → higher probability

**Numerical stability**: subtract the maximum logit before exponentiating. `exp(z - max)` gives identical probabilities but avoids overflow.

### Cross-Entropy Loss
```
loss = -log(P(correct_char))
```

Combined with softmax, this is called *cross-entropy loss*. It penalizes the model heavily when it assigns low probability to the correct answer.

### Manual Gradient
For softmax + cross-entropy, the gradient has a closed-form:

```
dL/dlogit_j = p_j - 1(j == target)
```

- For the correct character: gradient = `p - 1` (negative → increase this logit)
- For all other characters: gradient = `p` (positive → decrease these logits)

### SGD Update
```
weight -= learning_rate * gradient
```

We subtract because we want to go in the direction that *decreases* the loss (gradient descent).

### Temperature
At inference time, we divide logits by a temperature `T` before softmax:
- `T < 1` (e.g. 0.5): sharper distribution, more confident, less random
- `T = 1`: unmodified
- `T > 1`: flatter distribution, more random/creative

---

## 6. Step 3 — Automatic Differentiation (Autograd)

**File**: `autograd/Value.java`

Computing gradients by hand works for simple models, but becomes impractical as models grow. We need a system that computes gradients automatically.

### The Computational Graph
Every operation creates a **computation graph** — a directed acyclic graph where:
- **Nodes** are scalar values (`Value` objects)
- **Edges** connect each result to its inputs

Every `Value` stores:
- `data` — the forward-pass result
- `gradient` — accumulated gradient from the backward pass
- `children` — the `Value` objects this was computed from
- `backwardFn` — a lambda that computes how to propagate gradient to children

### Chain Rule
Backpropagation is just the chain rule applied systematically:

```
dL/dx = dL/d(output) * d(output)/dx
```

Where `dL/d(output)` is the gradient flowing in from above, and `d(output)/dx` is the *local gradient* — how this operation affects its inputs.

### Operations and their Local Gradients

| Operation | Forward | Local gradient for x | Local gradient for y |
|-----------|---------|---------------------|---------------------|
| `x + y` | `x.data + y.data` | 1 | 1 |
| `x * y` | `x.data * y.data` | `y.data` | `x.data` |
| `x^n` | `x.data^n` | `n * x.data^(n-1)` | — |
| `exp(x)` | `e^x.data` | `e^x.data` | — |
| `log(x)` | `ln(x.data)` | `1 / x.data` | — |
| `relu(x)` | `max(0, x.data)` | `1 if x.data > 0 else 0` | — |

### Backward Pass
`backward()` runs the chain rule through the entire graph:

```java
public void backward() {
    // 1. Build topological order (children before parents)
    List<Value> topo = topologicalSort();

    // 2. Seed the gradient at the loss node
    this.gradient = 1.0;

    // 3. Walk backwards, applying each node's backwardFn
    for (Value v : reversed(topo)) {
        v.backwardFn.run();
    }
}
```

**Topological sort** ensures we always process a node *after* all nodes that depend on it have already propagated their gradient contributions.

### Gradient Accumulation
Gradients are *accumulated* (`+=`) not overwritten. This handles cases where the same `Value` is used in multiple operations. Before each training step, all gradients must be zeroed.

---

## 7. Step 4 — MLP Language Model

**Files**: `model/MLPLanguageModel.java`, `nn/Embedding.java`, `nn/PositionalEmbedding.java`, `nn/Linear.java`

The bigram only looks at one previous character. We want to look at `N` previous characters (context window). This requires moving from a lookup table to a proper neural network.

### Token Embeddings
**File**: `nn/Embedding.java`

An embedding table maps each token ID to a dense vector of floats:

```
Embedding[vocabularySize][embeddingDimension]
```

Instead of a one-hot vector (sparse, 27 dimensions), each token gets a 10-dimensional dense vector that the model learns to place in a meaningful space. Semantically similar characters end up near each other.

### Positional Embeddings
**File**: `nn/PositionalEmbedding.java`

A second lookup table maps each *position* in the context to its own learned vector:

```
PositionalEmbedding[blockSize][embeddingDimension]
```

This tells the model *where* in the context each character appears, since position matters ("a" at position 0 vs position 2 carry different information).

### Linear Layer (No Bias)
**File**: `nn/Linear.java`

A fully-connected layer: `output = input @ W`

```java
for (int j = 0; j < outputDimension; j++) {
    Value sum = new Value(0.0);
    for (int i = 0; i < inputDimension; i++) {
        sum = sum.add(input[i].multiply(weights[i][j]));
    }
    output[j] = sum;
}
```

Weights are initialized with **Xavier/Glorot initialization**:

```
scale = sqrt(2 / (inputDimension + outputDimension))
weight ~ Gaussian(0, scale)
```

This keeps the variance of activations roughly constant through layers, preventing vanishing or exploding gradients.

**No bias**: Modern transformers omit bias terms. Normalization layers (RMSNorm) already handle the shift, and biases add parameters without much benefit.

### MLP Architecture

```
context [blockSize] token IDs
    ↓ (token embedding lookup)
tokenEmbeddings [blockSize × embDim]
    ↓ (+ positional embeddings, element-wise)
combinedEmbeddings [blockSize × embDim]
    ↓ (flatten)
x [blockSize * embDim]
    ↓ (Linear → tanh)
hidden [hiddenDimension]
    ↓ (Linear)
logits [vocabularySize]
    ↓ (softmax → -log → loss)
```

### Tanh Activation
The hidden layer uses tanh (hyperbolic tangent):
- Output range: (-1, 1)
- Smooth, differentiable, zero-centred
- Squashes large values, preventing the hidden layer from growing unboundedly

---

## 8. Step 5 — GPT Transformer (Single-Head)

**Files**: `nn/CausalSelfAttention.java`, `nn/TransformerBlock.java`, `model/GPTLanguageModel.java`

The MLP mixes all context positions together by flattening. Attention is different — it lets each position *selectively focus* on other positions.

### RMSNorm
**File**: `nn/RMSNormalization.java`

Root Mean Square Normalization stabilizes training by normalizing activations:

```
RMS(x) = sqrt(mean(x²) + ε)
output[i] = x[i] / RMS(x) * gamma[i]
```

- `ε = 1e-5` prevents division by zero
- `gamma` is a learnable scale, initialized to 1.0 (identity at the start)
- No mean subtraction (unlike LayerNorm) — simpler, used in LLaMA, Mistral

Applied *before* each sub-layer (Pre-Norm style), which is more stable than Post-Norm.

### Causal Self-Attention
**File**: `nn/CausalSelfAttention.java`

Attention lets each position query all previous positions and ask: *which positions are most relevant to me right now?*

**Step 1: Project to Q, K, V**

Each input vector is linearly projected to three vectors:
- **Query (Q)**: "what am I looking for?"
- **Key (K)**: "what do I contain?"
- **Value (V)**: "what will I contribute if attended to?"

```
Q = input @ W_q    [seqLen × headDim]
K = input @ W_k    [seqLen × headDim]
V = input @ W_v    [seqLen × headDim]
```

**Step 2: Compute attention scores**

```
scores[i][j] = dot(Q[i], K[j]) / sqrt(headDim)
```

`scores[i][j]` = how much position `i` should attend to position `j`.

The `sqrt(headDim)` scaling prevents the dot products from growing too large (which would push softmax into a near-zero-gradient region).

**Step 3: Causal mask**

Language modeling requires that position `i` can only see positions `≤ i` (not the future). We set future positions to `-∞` before softmax:

```java
for (int j = i + 1; j < seqLen; j++) {
    scores[i][j] = new Value(Double.NEGATIVE_INFINITY);
}
```

`exp(-∞) = 0` → these positions get zero attention weight.

**Step 4: Softmax + weighted sum**

```
attnWeights = softmax(scores)          [seqLen × seqLen]
output[i] = sum_j(attnWeights[i][j] * V[j])   [seqLen × headDim]
```

Each output is a weighted average of all (visible) value vectors, where the weights come from how relevant each position's key was to the current query.

**Step 5: Output projection**

```
output = attended @ W_o    [seqLen × embDim]
```

Projects back to the original embedding dimension.

### Transformer Block (Pre-Norm with Residuals)
**File**: `nn/TransformerBlock.java`

A full transformer block has two sub-layers, each with a *residual connection*:

```
// Attention sub-layer
x = x + Attention(RMSNorm(x))

// MLP sub-layer
x = x + MLP(RMSNorm(x))
```

**Residual connections** (skip connections) allow gradients to flow directly through the network without passing through every layer. This solves the vanishing gradient problem and makes very deep networks trainable.

**MLP inside the block** (position-wise feedforward):
```
hidden = ReLU(x @ W_fc1)     embDim → 4*embDim  (expand)
output = hidden @ W_fc2       4*embDim → embDim  (contract)
```

The 4× expansion gives the network capacity to learn complex per-position transformations. ReLU activation introduces non-linearity.

### GPT Language Model
**File**: `model/GPTLanguageModel.java`

```
tokens [blockSize]
    ↓ token embedding + positional embedding
x [seqLen × embDim]
    ↓ RMSNorm
x [seqLen × embDim]
    ↓ N × TransformerBlock
x [seqLen × embDim]
    ↓ RMSNorm (final)
x [seqLen × embDim]
    ↓ Linear (output head): x[last_position]
logits [vocabularySize]
```

Only the **last position's** output is used to predict the next token — it has attended to all previous positions and aggregates the full context.

---

## 9. Step 6 — GPT Transformer (Multi-Head)

**Files**: `nn/MultiHeadCausalSelfAttention.java`, `nn/Attention.java`

A single attention head looks at all positions through one "lens". **Multi-head attention** runs several independent heads in parallel, each attending to a different subspace of the embedding.

### The Attention Interface
**File**: `nn/Attention.java`

Both attention types implement a common interface:

```java
public interface Attention {
    Value[][] forward(Value[][] input);
    List<Value> parameters();
}
```

`TransformerBlock` holds an `Attention` reference, and picks the implementation based on a `useMultiHead` flag — allowing the two approaches to be compared directly.

### How Multi-Head Works

```
input [seqLen × embDim]
    ↓ W_q, W_k, W_v projections
Q, K, V [seqLen × embDim]
    ↓ split into numHeads slices of headDim = embDim / numHeads
For each head h:
    Q_h, K_h, V_h [seqLen × headDim]
    → single-head attention → headOut_h [seqLen × headDim]
    ↓ concatenate all heads
concatenated [seqLen × embDim]
    ↓ W_o output projection
output [seqLen × embDim]
```

With `embDim=16` and `numHeads=4`, each head has `headDim=4`. Head 0 attends over dims 0–3, head 1 over dims 4–7, etc.

### Why Multi-Head?

Each head can specialize:
- One head might track syntactic patterns
- Another might track positional proximity
- Another might track semantic similarity

Crucially, the **parameter count is identical** to single-head attention — multi-head just uses the same parameters more efficiently by attending to multiple subspaces simultaneously.

---

## 10. Optimization: Adam

**File**: `optimizer/AdamOptimizer.java`

Plain SGD (`param -= lr * grad`) has one learning rate for all parameters. **Adam (Adaptive Moment Estimation)** uses per-parameter adaptive learning rates.

### Update Rule

```
m = β1 * m + (1 - β1) * grad        # first moment (momentum)
v = β2 * v + (1 - β2) * grad²       # second moment (variance)

m̂ = m / (1 - β1^t)                  # bias-corrected first moment
v̂ = v / (1 - β2^t)                  # bias-corrected second moment

param -= lr * m̂ / (sqrt(v̂) + ε)
```

**First moment** (m): exponentially weighted average of gradients. Provides momentum — parameters that consistently receive gradients in the same direction update faster.

**Second moment** (v): exponentially weighted average of squared gradients. Normalizes the update by the recent gradient magnitude — parameters with large, noisy gradients get smaller updates.

**Bias correction**: early in training, `m` and `v` are initialized to 0 and are biased toward zero. Dividing by `(1 - β^t)` corrects for this.

**Hyperparameters used**: `β1=0.85`, `β2=0.99`, `ε=1e-8`

### Linear Learning Rate Decay

```java
lr = initialLR * (1 - step / numSteps)
```

Starts at `0.01` and linearly decays to nearly `0` by the final step. Early training takes large steps to explore; later training takes small, precise steps to converge.

### Training Loop Pattern

```java
optimizer.zeroGradient(model.parameters());   // clear old gradients
double loss = model.trainStep(example);        // forward + backward
optimizer.step(model.parameters(), lr);        // Adam update
```

Gradients must be zeroed before each backward pass because `Value.backward()` *accumulates* gradients (`+=`).

---

## 11. Design Decisions

### No bias in Linear layers
Modern transformers (LLaMA, Mistral) remove bias terms. Since RMSNorm already handles scale and shift, biases add parameters without meaningful benefit. Removing them reduces the parameter count and matches current practice.

### Learnable gamma in RMSNorm
Although Karpathy's minimal version omits gamma, we keep it. It starts at 1.0 (identity) and only changes if the optimizer finds a better value — in the worst case it's a no-op, in the best case it adds useful expressiveness.

### Pre-Norm vs Post-Norm
We apply RMSNorm *before* each sub-layer (Pre-Norm). The original transformer paper used Post-Norm (after), but Pre-Norm is more stable and is the convention in modern models.

### Character-level tokenization
Simple, no dependencies, vocabulary size is tiny (27 tokens). Sub-optimal for real language (BPE or WordPiece would give much better compression) but perfect for learning the fundamentals.

### Xavier/Glorot initialization
Weights are initialized as Gaussian with `scale = sqrt(2 / (in + out))`. This keeps the variance of activations approximately constant through layers, preventing gradients from vanishing or exploding at initialization.

### BOS on both sides
Surrounding each name with `<BOS>` on both ends teaches the model two things simultaneously: how names begin (from BOS → first char) and how names end (last char → BOS).

### Per-token vs per-document training
Karpathy's original averages loss over all positions in a document and does one backward pass per document. Our implementation does one backward pass per token pair (randomly sampled). Per-document would give more stable gradients; per-token is simpler and still converges.

---

## 12. Package Structure

```
com.anirudhology.microgpt
│
├── Runner.java                     Entry point, runs all 6 steps in sequence
│
├── autograd/
│   ├── Value.java                  Scalar autograd node (the engine of everything)
│   └── ValueDemo.java              Demonstrations of autograd operations
│
├── tokenizer/
│   └── CharacterTokenizer.java     Character-level tokenizer with BOS token
│
├── data/
│   ├── TextCorpus.java             Downloads and reads names.txt
│   ├── NGramDatasetBuilder.java    Builds (context, target) pairs via sliding window
│   └── TrainingExample.java        Record: int[] context + int target
│
├── nn/                             Neural network building blocks
│   ├── Linear.java                 Fully-connected layer (no bias)
│   ├── Embedding.java              Token embedding lookup table
│   ├── PositionalEmbedding.java    Positional embedding lookup table
│   ├── RMSNormalization.java       RMSNorm with learnable gamma
│   ├── Attention.java              Interface for attention mechanisms
│   ├── CausalSelfAttention.java    Single-head causal self-attention
│   ├── MultiHeadCausalSelfAttention.java  Multi-head causal self-attention
│   └── TransformerBlock.java       Pre-Norm block: attention + MLP + residuals
│
├── model/                          Full language models
│   ├── BaselineBigramModel.java    Step 1: statistical bigram
│   ├── NeuralBigramModel.java      Step 2: neural bigram, manual gradients
│   ├── NeuralBigramAutogradModel.java  Step 3: neural bigram, autograd
│   ├── MLPLanguageModel.java       Step 4: MLP with context window
│   └── GPTLanguageModel.java       Steps 5–6: full transformer (flag for single/multi-head)
│
└── optimizer/
    └── AdamOptimizer.java          Adam with bias correction + LR decay support
```

### The 6-Step Progression

| Step | Model | Key Concept Introduced | Approx. NLL |
|------|-------|----------------------|-------------|
| 1 | Statistical Bigram | Counting, Laplace smoothing, sampling | ~2.45 |
| 2 | Neural Bigram (manual) | Logits, softmax, SGD, manual gradients | ~2.35 |
| 3 | Neural Bigram (autograd) | Computational graph, chain rule, `Value` | ~2.35 |
| 4 | MLP | Embeddings, context window, hidden layer, tanh | ~2.1 |
| 5 | GPT (single-head) | Attention, causal mask, residuals, RMSNorm | ~2.4 |
| 6 | GPT (multi-head) | Multi-head attention, `Attention` interface, Adam | ~2.3 |

Steps 5–6 have higher NLL than the MLP at 1000 training steps because the transformer has more parameters and needs more steps to converge — but it has far greater capacity for longer sequences.
