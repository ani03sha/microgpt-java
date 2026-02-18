package com.anirudhology.microgpt.nn;

import com.anirudhology.microgpt.autograd.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Causal self attention layer.
 * <p>
 * "Causal" means each position can only attend to PREVIOUS positions
 * (not future ones).
 * <p>
 * This is essential for language modeling - we can't peek at the answer
 * when training!
 */
public class CausalSelfAttention implements Attention {

    // Q, K, V vector size
    private final int headDimension;

    // Q, K, V projection layers (each: embeddingDimension → headDimension)
    private final Linear queryProjection;
    private final Linear keyProjection;
    private final Linear valueProjection;

    // Output projection (headDimension → embeddingDimension)
    // Brings the attended output back to original embedding size
    private final Linear outputProjection;

    public CausalSelfAttention(int embeddingDimension, int headDimension, Random random) {
        // Input/output embedding size
        this.headDimension = headDimension;

        // Initialize all four projection layers
        this.queryProjection = new Linear(embeddingDimension, headDimension, random);
        this.keyProjection = new Linear(embeddingDimension, headDimension, random);
        this.valueProjection = new Linear(embeddingDimension, headDimension, random);
        this.outputProjection = new Linear(headDimension, embeddingDimension, random);
    }

    /**
     * Get all trainable parameters
     */
    public List<Value> parameters() {
        List<Value> parameters = new ArrayList<>();
        parameters.addAll(this.queryProjection.parameters());
        parameters.addAll(this.keyProjection.parameters());
        parameters.addAll(this.valueProjection.parameters());
        parameters.addAll(this.outputProjection.parameters());
        return parameters;
    }

    /**
     * Full attention forward pass
     * <p>
     * Input [sequenceLength x embeddingDimension] (token + positional embeddings)
     * Output [sequenceLength x embeddingDimension] (context aware representations)
     * <p>
     * Steps:
     * 1. Project to Q, K, V
     * 2. Compute attention scores
     * 3. Apply causal mask
     * 4. Softmax + weighted sum
     * 5. Project output back to embeddingDim
     *
     */
    public Value[][] forward(Value[][] input) {
        // Step 1: Project to Q, K, V
        Value[][] Q = projectToQ(input);   // [sequenceLength × headDimension]
        Value[][] K = projectToK(input);   // [sequenceLength × headDimension]
        Value[][] V = projectToV(input);   // [sequenceLength × headDimension]

        // Step 2: Compute attention scores
        Value[][] scores = computeAttentionScores(Q, K);  // [sequenceLength × sequenceLength]

        // Step 3: Apply causal mask (prevent attending to future)
        applyCausalMask(scores);

        // Step 4a: Softmax → attention weights
        Value[][] attnWeights = softmaxRows(scores); // [sequenceLength × sequenceLength]

        // Step 4b: Weighted sum of Values
        Value[][] attended = weightedSum(attnWeights, V);  // [sequenceLength × headDimension]

        // Step 5: Project output back to embeddingDim
        Value[][] output = new Value[input.length][];
        for (int pos = 0; pos < input.length; pos++) {
            output[pos] = this.outputProjection.forward(attended[pos]);  // headDimension → embeddingDimension
        }
        return output;
    }

    private Value[][] project(Value[][] x, Linear layer) {
        final int sequenceLength = x.length;
        final Value[][] projected = new Value[sequenceLength][];
        for (int position = 0; position < sequenceLength; position++) {
            projected[position] = layer.forward(x[position]);
        }
        return projected;
    }

    /**
     * Step 1: Project each token embedding into Q, K, V spaces.
     * <p>
     * Input:  sequence of embeddings [seqLen × embeddingDim]
     * Output: Q, K, V arrays each of shape [seqLen × headDim]
     */
    private Value[][] projectToQ(Value[][] x) {
        return project(x, this.queryProjection);
    }

    private Value[][] projectToK(Value[][] x) {
        return project(x, this.keyProjection);
    }

    private Value[][] projectToV(Value[][] x) {
        return project(x, this.valueProjection);
    }

    /**
     * Step 2: Compute raw attention scores between all Q, K pairs.
     * <p>
     * For each pair (i, j): score = Q[i] * K [j] / sqrt(headDimension).
     * scores[i][j] = how much position i attends to position j
     */
    private Value[][] computeAttentionScores(Value[][] Q, Value[][] K) {
        final int sequenceLength = Q.length;
        double scale = 1.0 / Math.sqrt(this.headDimension); // Scaling factor

        final Value[][] scores = new Value[sequenceLength][sequenceLength];

        for (int i = 0; i < sequenceLength; i++) { // Query position
            for (int j = 0; j < sequenceLength; j++) { // Key position
                // Dot product: Q[i] · K[j]
                Value dotProduct = new Value(0.0);
                for (int dimension = 0; dimension < this.headDimension; dimension++) {
                    dotProduct = dotProduct.add(Q[i][dimension].multiply(K[j][dimension]));
                }
                // Scale by 1/sqrt(headDimension)
                scores[i][j] = dotProduct.multiply(scale);
            }
        }
        return scores;
    }

    /**
     * Step 3: Apply causal mask to attention scores.
     * <p>
     * Prevents position i from attending to position j
     * when j > i.
     * <p>
     * Sets future positions to -infinity so softmax
     * gives them 0 weight
     */
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

    /**
     * Step 4a: Apply softmax row-wise to attention scores.
     * <p>
     * Each row becomes a probability distribution over positions.
     */
    private Value[][] softmaxRows(Value[][] scores) {
        final int sequenceLength = scores.length;
        final Value[][] attentionWeights = new Value[sequenceLength][];
        for (int i = 0; i < sequenceLength; i++) {
            attentionWeights[i] = softmax(scores[i]);
        }
        return attentionWeights;
    }

    /**
     * Step 4b: Compute weighted sum of values using attention weights.
     * <p>
     * For each query position i: output[i] = sum over j of (attentionWeights[i][j] * V[j]
     * Output shape: [sequenceLength X sequenceLength]
     */
    private Value[][] weightedSum(Value[][] attentionWeights, Value[][] V) {
        final int sequenceLength = attentionWeights.length;
        Value[][] output = new Value[sequenceLength][this.headDimension];
        for (int i = 0; i < sequenceLength; i++) {
            // Initialize output to 0
            for (int dimension = 0; dimension < this.headDimension; dimension++) {
                output[i][dimension] = new Value(0.0);
            }
            // Key/Value position
            for (int j = 0; j < sequenceLength; j++) {
                for (int dimension = 0; dimension < this.headDimension; dimension++) {
                    // Add weighted value
                    output[i][dimension] = output[i][dimension].add(attentionWeights[i][j].multiply(V[j][dimension]));
                }
            }
        }
        return output;
    }

    /**
     * Standard softmax for a 1D array for values
     */
    private Value[] softmax(Value[] logits) {
        // find max for numerical stability
        double max = Double.NEGATIVE_INFINITY;
        for (Value logit : logits) {
            max = Math.max(max, logit.getData());
        }
        final Value[] exponents = new Value[logits.length];
        Value sum = new Value(0.0);

        for (int i = 0; i < logits.length; i++) {
            // Use 0.0 for -infinity positions (masked out)
            if (logits[i].getData() == Double.NEGATIVE_INFINITY) {
                exponents[i] = new Value(0.0);
            } else {
                exponents[i] = logits[i].subtract(max).exp();
                sum = sum.add(exponents[i]);
            }
        }
        final Value[] probabilities = new Value[logits.length];
        for (int i = 0; i < logits.length; i++) {
            if (logits[i].getData() == Double.NEGATIVE_INFINITY) {
                probabilities[i] = new Value(0.0);
            } else {
                probabilities[i] = exponents[i].divide(sum);
            }
        }
        return probabilities;
    }
}
