package com.anirudhology.microgpt.nn;

import com.anirudhology.microgpt.autograd.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Multi-head causal self attention
 * <p>
 * Splits Q, K, V into nHead independent heads, each attending
 * to a different subspace.
 * Outputs concatenated back to embeddingDimension.
 */
public class MultiHeadCausalSelfAttention {

    private final int embeddingDimension;
    private final int numHeads;
    private final int headDimension;  // embeddingDimension / numHeads

    // Projects to full embeddingDimension (then split into heads)
    private final Linear queryProjection;
    private final Linear keyProjection;
    private final Linear valueProjection;

    // Output projection: embeddingDimension → embeddingDimension
    private final Linear outputProjection;

    public MultiHeadCausalSelfAttention(int embeddingDimension, int numHeads, Random random) {
        if (embeddingDimension % numHeads != 0) {
            throw new IllegalArgumentException(String.format("embeddingDimension %d must be divisible by numHeads %d", embeddingDimension, numHeads));
        }

        this.embeddingDimension = embeddingDimension;
        this.numHeads = numHeads;
        this.headDimension = embeddingDimension / numHeads;

        // All projections: embeddingDimension -> embeddingDimension (not embeddingDimension -> headDimension!)
        this.queryProjection = new Linear(embeddingDimension, embeddingDimension, random);
        this.keyProjection = new Linear(embeddingDimension, embeddingDimension, random);
        this.valueProjection = new Linear(embeddingDimension, embeddingDimension, random);
        this.outputProjection = new Linear(embeddingDimension, embeddingDimension, random);
    }

    /**
     * Multi-head attention forward pass
     */
    public Value[][] forward(Value[][] input) {
        final int sequenceLength = input.length;

        // Step 1: Project to Q, K, V
        final Value[][] Q = project(input, this.queryProjection);
        final Value[][] K = project(input, this.keyProjection);
        final Value[][] V = project(input, this.valueProjection);

        // Step 2: Run attention for each head independently
        // Each head uses a slice of Q, K, V of size headDimension
        final Value[][] concatenated = new Value[sequenceLength][this.embeddingDimension];
        for (int head = 0; head < numHeads; head++) {
            int start = head * this.headDimension; // Slice start index
            // Extract this head's slice from Q, K, V
            final Value[][] Q_h = slice(Q, start, this.headDimension);
            final Value[][] K_h = slice(K, start, this.headDimension);
            final Value[][] V_h = slice(V, start, this.headDimension);

            // Compute attention for this head (with causal mask)
            Value[][] headOutput = singleHeadAttention(Q_h, K_h, V_h);

            // Place this head's output into the right slice of concatenated
            for (int position = 0; position < sequenceLength; position++) {
                if (this.headDimension >= 0) {
                    System.arraycopy(headOutput[position], 0, concatenated[position], start, this.headDimension);
                }
            }
        }

        // Step 3: Output projection
        final Value[][] output = new Value[sequenceLength][];
        for (int position = 0; position < sequenceLength; position++) {
            output[position] = outputProjection.forward(concatenated[position]);
        }
        return output;
    }

    /**
     * Single head attention: Q, K, V -> attended output
     */
    private Value[][] singleHeadAttention(Value[][] Q, Value[][] K, Value[][] V) {
        final int sequenceLength = Q.length;
        final double scale = 1.0 / Math.sqrt(this.headDimension);

        // Compute attention scores
        final Value[][] scores = new Value[sequenceLength][sequenceLength];
        for (int i = 0; i < sequenceLength; i++) {
            for (int j = 0; j < sequenceLength; j++) {
                Value dotProduct = new Value(0.0);
                for (int dimension = 0; dimension < this.headDimension; dimension++) {
                    dotProduct = dotProduct.add(Q[i][dimension].multiply(K[j][dimension]));
                }
                scores[i][j] = dotProduct.multiply(scale);
            }
        }
        // Apply causal mask (future positions -> -infinity
        for (int i = 0; i < sequenceLength; i++) {
            for (int j = i + 1; j < sequenceLength; j++) {
                scores[i][j] = new Value(Double.NEGATIVE_INFINITY);
            }
        }

        // Softmax each row -> attention weights
        final Value[][] attentionWeights = new Value[sequenceLength][];
        for (int i = 0; i < sequenceLength; i++) {
            attentionWeights[i] = softmax(scores[i]);
        }

        // Weighted sum of V
        Value[][] output = new Value[sequenceLength][this.headDimension];
        for (int i = 0; i < sequenceLength; i++) {
            for (int dimension = 0; dimension < this.headDimension; dimension++) {
                output[i][dimension] = new Value(0.0);
            }
            for (int j = 0; j < sequenceLength; j++) {
                for (int dimension = 0; dimension < this.headDimension; dimension++) {
                    output[i][dimension] = output[i][dimension].add(attentionWeights[i][j].multiply(V[j][dimension]));
                }
            }
        }
        return output;
    }

    public List<Value> parameters() {
        final List<Value> parameters = new ArrayList<>();
        parameters.addAll(this.queryProjection.parameters());
        parameters.addAll(this.keyProjection.parameters());
        parameters.addAll(this.valueProjection.parameters());
        parameters.addAll(this.outputProjection.parameters());
        return parameters;

    }

    /**
     * Apply linear projection to each position in sequence
     */
    private Value[][] project(Value[][] input, Linear layer) {
        final Value[][] result = new Value[input.length][];
        for (int position = 0; position < input.length; position++) {
            result[position] = layer.forward(input[position]);
        }
        return result;
    }

    /**
     * Extract a slice [start, start + size] along the embedding dimension
     */
    private Value[][] slice(Value[][] input, int start, int size) {
        final Value[][] result = new Value[input.length][size];
        for (int position = 0; position < input.length; position++) {
            System.arraycopy(input[position], start, result[position], 0, size);
        }
        return result;
    }

    private Value[] softmax(Value[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (Value logit : logits) {
            if (logit.getData() != Double.NEGATIVE_INFINITY) {
                max = Math.max(max, logit.getData());
            }
        }

        final Value[] exponents = new Value[logits.length];
        Value sum = new Value(0.0);
        for (int i = 0; i < logits.length; i++) {
            if (logits[i].getData() != Double.NEGATIVE_INFINITY) {
                exponents[i] = logits[i].subtract(max).exp();
                sum = sum.add(exponents[i]);
            }
        }

        final Value[] probabilities = new Value[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = logits[i].getData() == Double.NEGATIVE_INFINITY ? new Value(0.0) : exponents[i].divide(sum);
        }
        return probabilities;
    }
}
