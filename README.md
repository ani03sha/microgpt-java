# MicroGPT in Java

Java implementation of [Karpathy's microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95).

## Concepts

### 1. Logits

Logits are raw, unnormalized scores that come directly from a neural network before being converted to probabilities.

- They can be any real number: -ve, 0, or +ve.
- These are not probabilities as they don't sum to 1 and not bounded to `[0, 1]`.
- Larger logit mean higher probability after conversion.
- Their relative values matter, i.e., if $logit[A] > logit[B] → P(A) > P(B)$

**Why Logit?**

The term comes from "log-odds" or "logistic unit". Let's understand the math behind it:

Starting from probability $p$:

- Odds: $odds = p / (1 - p)$. If $p = 0.8 → odds = 0.8/0.2 = 4$ (4-to-1 in favor).
- Log odds (logit): $logit = log(p / (1 - p)$. If $p = 0.8 → logit(4) = 1.39$.
- Inverse (sigmoid): $p = 1 / (1 + e^(-logit))$. If $logit = 1.39 → p ≈ 0.8$.

In classification context, logits are the values before applying sigmoid (binary) or softmax (multi-class).
