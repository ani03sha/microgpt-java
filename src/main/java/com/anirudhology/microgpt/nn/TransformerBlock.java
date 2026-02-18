package com.anirudhology.microgpt.nn;

import com.anirudhology.microgpt.autograd.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * A single transformer block.
 * <p>
 * Architecture (Pre-Norm style)
 * <p>
 * x = x + Attention(RMSNorm(x) -> Attention with residual
 * <p>
 * Stack N of these to build a full transformer
 */
public class TransformerBlock {

    // Attention sub-layer
    private final RMSNormalization attentionNormalization;
    private final MultiHeadCausalSelfAttention attention;

    // MLP sub-layer
    private final RMSNormalization mlpNormalization;
    private final Linear mlpFc1; // Expand: embeddingDimension -> 4 * embeddingDimension
    private final Linear mlpFc2; // Contract: 4 * embeddingDimension -> embeddingDimension;

    /*public TransformerBlock(int embeddingDimension, int headDimension, Random random) {

        // Attention sub-layer
        this.attentionNormalization = new RMSNormalization(embeddingDimension);
        this.attention = new CausalSelfAttention(embeddingDimension, headDimension, random);

        // MLP sub-layer
        int mlpHiddenDimension = 4 * embeddingDimension;
        this.mlpNormalization = new RMSNormalization(embeddingDimension);
        this.mlpFc1 = new Linear(embeddingDimension, mlpHiddenDimension, random);
        this.mlpFc2 = new Linear(mlpHiddenDimension, embeddingDimension, random);
    }*/

    public TransformerBlock(int embeddingDimension, int numHeads, Random random) {
        this.attentionNormalization = new RMSNormalization(embeddingDimension);
        this.attention = new MultiHeadCausalSelfAttention(embeddingDimension, numHeads, random);
        int mlpHiddenDimension = 4 * embeddingDimension;
        this.mlpNormalization = new RMSNormalization(embeddingDimension);
        this.mlpFc1 = new Linear(embeddingDimension, mlpHiddenDimension, random);
        this.mlpFc2 = new Linear(mlpHiddenDimension, embeddingDimension, random);
    }

    /**
     * Forward pass through one transformer block
     *
     * @param input [sequenceLength * embeddingDimension]
     * @return [sequenceLength * embeddingDimension] (same shape!)
     */
    public Value[][] forward(Value[][] input) {
        // --- Attention sub-layer ---
        // Step 1: Normalize input
        Value[][] normed = this.attentionNormalization.forward(input);

        // Step 2: Self-attention
        final Value[][] attentionOutput = this.attention.forward(normed);

        // Step 3: Residual connection: x = x + attention(norm(x))
        input = add(input, attentionOutput);

        // --- MLP sub-layer ---
        // Step 4: Normalize
        normed = this.mlpNormalization.forward(input);

        // Step 5: MLP (expand -> relu -> contract)
        final Value[][] mlpOutput = mlpForward(normed);

        // Step 6: Residual contraction: x = x + mlp(norm(x))
        input = add(input, mlpOutput);
        return input;
    }

    /**
     * Get all learnable parameters
     */
    public List<Value> parameters() {
        final List<Value> params = new ArrayList<>();
        params.addAll(this.attentionNormalization.parameters());
        params.addAll(this.attention.parameters());
        params.addAll(this.mlpNormalization.parameters());
        params.addAll(this.mlpFc1.parameters());
        params.addAll(this.mlpFc2.parameters());
        return params;
    }

    /**
     * Feedforward MLP inside the transformer block.
     * <p>
     * embeddingDimension -> (4 * embeddingDimension) -> relu -> embeddingDimension
     * <p>
     * Applied independently to each position (no mixing between positions here,
     * that's attention's job!)
     */
    private Value[][] mlpForward(Value[][] input) {
        final int sequenceLength = input.length;
        final Value[][] output = new Value[sequenceLength][];

        for (int position = 0; position < sequenceLength; position++) {
            // Expand: embeddingDimension -> 4 * embeddingDimension
            final Value[] hidden = this.mlpFc1.forward(input[position]);
            // ReLU activation
            for (int i = 0; i < hidden.length; i++) {
                hidden[i] = hidden[i].relu();
            }
            // Contract: 4 * embeddingDimension -> embeddingDimension
            output[position] = this.mlpFc2.forward(hidden);
        }
        return output;
    }

    /**
     * Element-wise addition of two [sequenceLength * dimension] arrays
     */
    private Value[][] add(Value[][] a, Value[][] b) {
        final int sequenceLength = a.length;
        final int dimension = a[0].length;
        final Value[][] result = new Value[sequenceLength][dimension];

        for (int position = 0; position < sequenceLength; position++) {
            for (int d = 0; d < dimension; d++) {
                result[position][d] = a[position][d].add(b[position][d]);
            }
        }
        return result;
    }
}
