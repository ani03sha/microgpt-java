# MicroGPT Java — Concepts & Progression Guide

This document walks through every concept in the project in the order they appear, from the simplest statistical
baseline to the full transformer. Each section explains the *what*, the *why*, and the *how*.

---

## Table of Contents

1. [Problem Statement: Language Modeling](#1-problem-statement-language-modeling)
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

---

## 1. Problem Statement: Language Modeling

The goal of this project is simple: **given a sequence of characters, predict the next character**.

We train on a list of names (e.g. `emma`, `olivia`, `noah`). After training, the model generates new names that look
plausible. Thus, the model learns statistical patterns of how characters follow each other in names.

**Why names?** They are short, structured enough to show real learning, but simple enough to train from scratch in
minutes.

**Autoregressive generation**: at inference time, we feed the model a context, sample the next character, append it to
the context, and repeat, thus, generating one character at a time. This is called *autoregressive* because each output
becomes part of the next input.

**Loss metric: Negative Log-Likelihood (NLL)**:

$$\mathcal{L} = -\log P(\text{correct next character})$$

- Perfect prediction $(P = 1.0)$ → loss $= 0$
- Bad prediction $(P = 0.1)$ → loss $= 2.30$
- Random guess over 27 chars → loss $\approx 3.30$

Lower loss means better model. This is our compass throughout.

---

## 2. Tokenization

**File**: `tokenizer/CharacterTokenizer.java`

We use *character-level tokenization* where every unique character in the dataset becomes a token with an integer ID.

```
a=0, b=1, c=2, ..., z=25, <BOS>=26
```

**BOS (Beginning of Sequence)** is a special token with the highest ID (`vocabSize - 1 = 26`). It serves two purposes:

1. As the *starting signal* when generating: the model begins with BOS and predicts the first character
2. As the *end signal*: the model learns to emit BOS when the name is complete

**`withBOSOnBothSides("emma")`** → `[26, 4, 12, 12, 0, 26]`

This surrounds each name with BOS so the model learns both how names start and how they end.

---

## 3. Data Loading & Dataset Building

**Files**: `data/TextCorpus.java`, `data/NGramDatasetBuilder.java`, `data/TrainingExample.java`

`TextCorpus` downloads `names.txt` from Karpathy's makemore repo if not present, then reads, trims, and shuffles all
names.

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
public record TrainingExample(int[] context, int target) {
}
```

---

## 4. Step 1 — Statistical Bigram Model

**File**: `model/BaselineBigramModel.java`

This class represents the simplest possible language model where we count how often each character follows each other
character, then normalize.

### Counting

```
counts[i][j] = how many times character j follows character i
```

After seeing all names, we get a 27×27 table of co-occurrence counts.

### Laplace Smoothing

Without smoothing, unseen character pairs would have probability 0, causing $\log(0) = -\infty$. We add a small constant
$\alpha$ to every count:

$$P(j \mid i) = \frac{\text{counts}[i][j] + \alpha}{\sum_k \left(\text{counts}[i][k] + \alpha\right)}$$

This ensures every transition has at least some probability, acting as a *prior* that all transitions are possible.

### Sampling

To generate a character, we sample from a probability distribution using the **CDF (Cumulative Distribution Function)
trick**:

```java
private int sampleFromDistribution(double[] probabilities) {
    // Random number in [0, 1)
    double r = this.random.nextDouble();
    // Cumulative distribution function
    double cdf = 0.0;
    for (int i = 0; i < probabilities.length; i++) {
        cdf += probabilities[i];
        if (r <= cdf) {
            return i;
        }
    }
    // Fallback for numerical precision
    return probabilities.length - 1;
}
```

This correctly samples each character proportional to its probability.

### Baseline NLL

This model achieves ~2.45 NLL. Any neural model that can't beat this isn't learning anything useful.

---

## 5. Step 2 — Neural Bigram (Manual Gradients)

**File**: `model/NeuralBigramModel.java`

Here also we have prediction task, but instead of a count table we use a **learned weight matrix** and gradient descent.

### Logits

The weight matrix $W \in \mathbb{R}^{27 \times 27}$ holds raw unnormalized scores called *logits*. The value $W[i][j]$
represents "how likely is character $j$ to follow character $i$", but as a real number (positive or negative), not a
probability.

### Softmax

We convert logits to a valid probability distribution:

$$\text{softmax}(\mathbf{z})_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$$

Properties:

- All outputs are in $(0, 1)$
- Outputs sum to $1$
- Higher logit → higher probability

**Numerical stability**: subtract the maximum logit before exponentiating. $e^{z - \max(z)}$ gives identical
probabilities but avoids overflow.

### Cross-Entropy Loss

$$\mathcal{L} = -\log P{\_true}$$

Combined with softmax, this is called *cross-entropy loss*. It penalizes the model heavily when it assigns low
probability to the correct answer.

### Manual Gradient

For softmax + cross-entropy, the gradient has a closed-form:

$$\frac{\partial \mathcal{L}}{\partial \text{logit}_j} = p_j - \mathbf{1}[j = \text{target}]$$

- For the correct character: gradient $= p - 1$ (negative → increase this logit)
- For all other characters: gradient $= p$ (positive → decrease these logits)

### SGD Update

$$\theta \leftarrow \theta - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta}$$

We subtract because we want to go in the direction that *decreases* the loss (gradient descent).

### Temperature

At inference time, we divide logits by a temperature $T$ before softmax:

- $T < 1$ (e.g. 0.5): sharper distribution, more confident, less random
- $T = 1$: unmodified
- $T > 1$: flatter distribution, more random/creative

---

## 6. Step 3 — Automatic Differentiation (Autograd)

**File**: `autograd/Value.java`

Computing gradients by hand works for simple models, but becomes impractical as models grow. We need a system that
computes gradients automatically.

### The Computational Graph

Every operation creates a **computation graph**. It is a directed acyclic graph where:

- **Nodes** are scalar values (`Value` objects)
- **Edges** connect each result to its inputs

Every `Value` stores:

- `data` — the forward-pass result
- `gradient` — accumulated gradient from the backward pass
- `children` — the `Value` objects this was computed from
- `backwardFn` — a lambda that computes how to propagate gradient to children

### Chain Rule

Backpropagation is just the chain rule applied systematically:

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial \text{output}} \cdot \frac{\partial \text{output}}{\partial x}$$

Where $\frac{\partial \mathcal{L}}{\partial \text{output}}$ is the gradient flowing in from above, and
$\frac{\partial \text{output}}{\partial x}$ is the *local gradient*, how this operation affects its inputs.

### Operations and their Local Gradients

| Operation        | Forward      | Local gradient for $x$ | Local gradient for $y$ |
|------------------|--------------|------------------------|------------------------|
| $x + y$          | $x + y$      | $1$                    | $1$                    |
| $x \cdot y$      | $x \cdot y$  | $y$                    | $x$                    |
| $x^n$            | $x^n$        | $n \cdot x^{n-1}$      | —                      |
| $\exp(x)$        | $e^x$        | $e^x$                  | —                      |
| $\log(x)$        | $\ln(x)$     | $\dfrac{1}{x}$         | —                      |
| $\text{relu}(x)$ | $\max(0, x)$ | $\mathbf{1}[x > 0]$    | —                      |

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

**Topological sort** ensures we always process a node *after* all nodes that depend on it have already propagated their
gradient contributions.

### Gradient Accumulation

Gradients are *accumulated* ($+=$) not overwritten. This handles cases where the same `Value` is used in multiple
operations. Before each training step, all gradients must be zeroed.

---

## 7. Step 4 — MLP Language Model

**Files**: `model/MLPLanguageModel.java`, `nn/Embedding.java`, `nn/PositionalEmbedding.java`, `nn/Linear.java`

The bigram only looks at one previous character. We want to look at $N$ previous characters (context window). This
requires moving from a lookup table to a proper neural network.

### Token Embeddings

**File**: `nn/Embedding.java`

An embedding table maps each token ID to a dense vector of floats:

```
Embedding[vocabularySize][embeddingDimension]
```

Instead of a one-hot vector (sparse, 27 dimensions), each token gets a 10-dimensional dense vector that the model learns
to place in a meaningful space. Semantically similar characters end up near each other.

### Positional Embeddings

**File**: `nn/PositionalEmbedding.java`

A second lookup table maps each *position* in the context to its own learned vector:

```
PositionalEmbedding[blockSize][embeddingDimension]
```

This tells the model *where* in the context each character appears, since position matters ("a" at position 0 vs
position 2 carry different information).

### Linear Layer (No Bias)

**File**: `nn/Linear.java`

A fully-connected layer: $\mathbf{y} = \mathbf{x} W$

```java
public Value[] forward(Value[] input) {
    Value[] output = new Value[this.outputDimension];

    // For each output neuron
    for (int j = 0; j < this.outputDimension; j++) {
        // Start with bias
        Value sum = new Value(0.0);

        // Add weighted inputs: sum = b + x₀*w₀ⱼ + x₁*w₁ⱼ + ...
        for (int i = 0; i < this.inputDimension; i++) {
            sum = sum.add(input[i].multiply(this.weights[i][j]));
        }
        output[j] = sum;
    }
    return output;
}
```

Weights are initialized with **Xavier/Glorot initialization**:

$$\text{scale} = \sqrt{\frac{2}{d_{\text{in}} + d_{\text{out}}}}$$

$$w \sim \mathcal{N}(0,\ \text{scale}^2)$$

This keeps the variance of activations roughly constant through layers, preventing vanishing or exploding gradients.

**No bias**: Modern transformers omit bias terms. Normalization layers (RMSNorm) already handle the shift, and biases
add parameters without much benefit.

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

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- Output range: $(-1,\ 1)$
- Smooth, differentiable, zero-centred
- Squashes large values, preventing the hidden layer from growing unboundedly

---

## 8. Step 5 — GPT Transformer (Single-Head)

**Files**: `nn/CausalSelfAttention.java`, `nn/TransformerBlock.java`, `model/GPTLanguageModel.java`

The MLP mixes all context positions together by flattening. Attention is different because it lets each position *selectively
focus* on other positions.

### RMSNorm

**File**: `nn/RMSNormalization.java`

Root Mean Square Normalization stabilizes training by normalizing activations:

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \varepsilon}$$

$$\text{output}_i = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot \gamma_i$$

- $\varepsilon = 10^{-5}$ prevents division by zero
- $\gamma$ is a learnable scale, initialized to $1.0$ (identity at the start)
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

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V \qquad \in \mathbb{R}^{L \times d_h}$$

**Step 2: Compute attention scores**

$$\text{scores}[i][j] = \frac{Q_i \cdot K_j}{\sqrt{d_h}}$$

$\text{scores}[i][j]$ = how much position $i$ should attend to position $j$.

The $\sqrt{d_h}$ scaling prevents the dot products from growing too large (which would push softmax into a
near-zero-gradient region).

**Step 3: Causal mask**

Language modeling requires that position $i$ can only see positions $\leq i$ (not the future). We set future positions
to $-\infty$ before softmax:

```java
private void applyCausalMask(Value[][] scores) {
    final int sequenceLength = scores.length;
    for (int i = 0; i < sequenceLength; i++) { // Query position
        for (int j = i + 1; j < sequenceLength; j++) { // Future key position
            // Replace with -infinity
            // exp(-∞) = 0 after softmax → zero attention to future!
            scores[i][j] = new Value(Double.NEGATIVE_INFINITY);
        }
    }
}
```

$e^{-\infty} = 0$ → these positions get zero attention weight.

**Step 4: Softmax + weighted sum**

$$\mathbf{A} = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right), \qquad \text{output} = \mathbf{A} V$$

Each output is a weighted average of all (visible) value vectors, where the weights come from how relevant each
position's key was to the current query.

**Step 5: Output projection**

$$\text{output} = \text{attended} \cdot W_O \qquad \in \mathbb{R}^{L \times d_{\text{model}}}$$

Projects back to the original embedding dimension.

### Transformer Block (Pre-Norm with Residuals)

**File**: `nn/TransformerBlock.java`

A full transformer block has two sub-layers, each with a *residual connection*:

$$x \leftarrow x + \text{Attention}(\text{RMSNorm}(x))$$

$$x \leftarrow x + \text{MLP}(\text{RMSNorm}(x))$$

**Residual connections** (skip connections) allow gradients to flow directly through the network without passing through
every layer. This solves the vanishing gradient problem and makes very deep networks trainable.

**MLP inside the block** (position-wise feedforward):

$$h = \text{ReLU}(x W_1), \quad \text{output} = h W_2$$

where $W_1 \in \mathbb{R}^{d \times 4d}$ (expand) and $W_2 \in \mathbb{R}^{4d \times d}$ (contract).

The 4× expansion gives the network capacity to learn complex per-position transformations.

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

Only the **last position's** output is used to predict the next token — it has attended to all previous positions and
aggregates the full context.

---

## 9. Step 6 — GPT Transformer (Multi-Head)

**Files**: `nn/MultiHeadCausalSelfAttention.java`, `nn/Attention.java`

A single attention head looks at all positions through one "lens". **Multi-head attention** runs several independent
heads in parallel, each attending to a different subspace of the embedding.

### The Attention Interface

**File**: `nn/Attention.java`

Both attention types implement a common interface:

```java
public interface Attention {
    Value[][] forward(Value[][] input);

    List<Value> parameters();
}
```

`TransformerBlock` holds an `Attention` reference, and picks the implementation based on a `useMultiHead` flag —
allowing the two approaches to be compared directly.

### How Multi-Head Works

$$\text{head}_h = \text{Attention}(Q_h,\ K_h,\ V_h), \quad h = 1, \ldots, H$$

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \cdot W_O$$

where $Q_h, K_h, V_h$ are slices of the full $Q, K, V$ projections of width $d_h = d_{\text{model}} / H$.

```
input [seqLen × embDim]
    ↓ W_q, W_k, W_v projections
Q, K, V [seqLen × embDim]
    ↓ split into H slices of headDim = embDim / H
For each head h:
    Q_h, K_h, V_h [seqLen × headDim]
    → single-head attention → headOut_h [seqLen × headDim]
    ↓ concatenate all heads
concatenated [seqLen × embDim]
    ↓ W_o output projection
output [seqLen × embDim]
```

With $d_{\text{model}} = 16$ and $H = 4$, each head has $d_h = 4$. Head 0 attends over dims 0–3, head 1 over dims 4–7,
etc.

### Why Multi-Head?

Each head can specialize:

- One head might track syntactic patterns
- Another might track positional proximity
- Another might track semantic similarity

Crucially, the **parameter count is identical** to single-head attention — multi-head just uses the same parameters more
efficiently by attending to multiple subspaces simultaneously.

---

## 10. Optimization: Adam

**File**: `optimizer/AdamOptimizer.java`

Plain SGD has one learning rate for all parameters. **Adam (Adaptive Moment Estimation)** uses per-parameter adaptive
learning rates.

### Update Rule

$$m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \qquad \text{(first moment — momentum)}$$

$$v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \qquad \text{(second moment — variance)}$$

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \qquad \text{(bias correction)}$$

$$\theta_t \leftarrow \theta_{t-1} - \frac{\eta\, \hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

**First moment** ($m$): exponentially weighted average of gradients. Provides momentum: parameters that consistently
receive gradients in the same direction update faster.

**Second moment** ($v$): exponentially weighted average of squared gradients. Normalizes the update by the recent
gradient magnitude: parameters with large, noisy gradients get smaller updates.

**Bias correction**: early in training, $m$ and $v$ are initialized to $0$ and are biased toward zero. Dividing by
$(1 - \beta^t)$ corrects for this.

**Hyperparameters used**: $\beta_1 = 0.85$, $\beta_2 = 0.99$, $\varepsilon = 10^{-8}$

### Linear Learning Rate Decay

$$\eta_t = \eta_0 \cdot \left(1 - \frac{t}{T}\right)$$

Starts at $\eta_0 = 0.01$ and linearly decays to nearly $0$ by step $T$. Early training takes large steps to explore;
later training takes small, precise steps to converge.

### Training Loop Pattern

```java
private void run() {
    for (int step = 0; step < numberOfSteps; step++) {
        double learningRate = initialLearningRate * (1.0 - (double) step / numberOfSteps);

        optimizer.zeroGradient(model.parameters());
        double loss = model.trainStep(examples.get(step % examples.size()));
        optimizer.step(model.parameters(), learningRate);

        if ((step + 1) % 100 == 0) {
            System.out.printf("Step %4d / %4d | Loss: %.4f | LR: %.6f%n",
                    step + 1, numberOfSteps, loss, learningRate);
        }
    }
}
```

Gradients must be zeroed before each backward pass because `Value.backward()` *accumulates* gradients ($+=$).

---

## 11. Design Decisions

### No bias in Linear layers

Modern transformers (LLaMA, Mistral) remove bias terms. Since RMSNorm already handles scale and shift, biases add
parameters without meaningful benefit. Removing them reduces the parameter count and matches current practice.

### Learnable gamma in RMSNorm

Although Karpathy's minimal version omits $\gamma$, we keep it. It starts at $1.0$ (identity) and only changes if the
optimizer finds a better value. In the worst case it's a no-op, in the best case it adds useful expressiveness.

### Pre-Norm vs Post-Norm

We apply RMSNorm *before* each sub-layer (Pre-Norm). The original transformer paper used Post-Norm (after), but Pre-Norm
is more stable and is the convention in modern models.

### Character-level tokenization

Simple, no dependencies, vocabulary size is tiny (27 tokens). Sub-optimal for real language (BPE or WordPiece would give
much better compression) but perfect for learning the fundamentals.

### Xavier/Glorot initialization

Weights are initialized as $w \sim \mathcal{N}\!\left(0,\ \frac{2}{d_{\text{in}} + d_{\text{out}}}\right)$. This keeps
the variance of activations approximately constant through layers, preventing gradients from vanishing or exploding at
initialization.

### BOS on both sides

Surrounding each name with `<BOS>` on both ends teaches the model two things simultaneously: how names begin (from BOS →
first char) and how names end (last char → BOS).

### Per-token vs per-document training

Karpathy's original averages loss over all positions in a document and does one backward pass per document. Our
implementation does one backward pass per token pair (randomly sampled). Per-document would give more stable gradients;
per-token is simpler and still converges.

---

### The 6-Step Progression

| Step | Model                    | Key Concept Introduced                            | Approx. NLL |
|------|--------------------------|---------------------------------------------------|-------------|
| 1    | Statistical Bigram       | Counting, Laplace smoothing, sampling             | ~2.45       |
| 2    | Neural Bigram (manual)   | Logits, softmax, SGD, manual gradients            | ~2.35       |
| 3    | Neural Bigram (autograd) | Computational graph, chain rule, `Value`          | ~2.35       |
| 4    | MLP                      | Embeddings, context window, hidden layer, tanh    | ~2.1        |
| 5    | GPT (single-head)        | Attention, causal mask, residuals, RMSNorm        | ~2.4        |
| 6    | GPT (multi-head)         | Multi-head attention, `Attention` interface, Adam | ~2.3        |

Steps 5–6 have higher NLL than the MLP at 1000 training steps because the transformer has more parameters and needs more
steps to converge — but it has far greater capacity for longer sequences.
