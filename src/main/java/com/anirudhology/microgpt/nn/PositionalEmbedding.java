package com.anirudhology.microgpt.nn;

import com.anirudhology.microgpt.autograd.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Positional embeddings encode position information.
 * <p>
 * Unlike token embeddings (what the character is), positional embeddings
 * encode where the character appears in the sequence.
 * <p>
 * We add them to token embeddings: combined: token_emb + pos_emb
 */
public class PositionalEmbedding {

    // Maximum sequence length
    private final int maxPositions;
    private final int embeddingDimension;
    private final Value[][] weights;

    public PositionalEmbedding(int maxPositions, int embeddingDimension, Random random) {
        this.maxPositions = maxPositions;
        this.embeddingDimension = embeddingDimension;
        this.weights = new Value[maxPositions][embeddingDimension];

        // Initialize with small random values
        final double scale = 1.0 / Math.sqrt(embeddingDimension);
        for (int position = 0; position < maxPositions; position++) {
            for (int dimension = 0; dimension < embeddingDimension; dimension++) {
                final double value = random.nextGaussian() * scale;
                this.weights[position][dimension] = new Value(value, "pos_embedding_" + position + "_" + dimension);
            }
        }
    }

    /**
     * Get positional embedding for a specific position
     *
     * @param position Position index (0 to maxPositions - 1)
     * @return Positional embedding vector
     */
    public Value[] forward(int position) {
        if (position < 0 || position >= this.maxPositions) {
            throw new IllegalArgumentException(String.format("Position %d out of range [0, %d)", position, this.maxPositions));

        }
        return this.weights[position];
    }

    /**
     * Get positional embeddings for a sequence of positions.
     *
     * @param length length of sequence
     * @return array of positional embeddings [pos0, pos1, pos2,...]
     */
    public Value[][] forwardAll(int length) {
        if (length < 0 || length > this.maxPositions) {
            throw new IllegalArgumentException(String.format("Sequence length %d out of range [0, %d)", length, this.maxPositions));
        }

        final Value[][] result = new Value[length][];
        System.arraycopy(this.weights, 0, result, 0, length);
        return result;
    }

    /**
     * Get all trainable parameters
     *
     * @return list of parameters
     */
    public List<Value> parameters() {
        final List<Value> parameters = new ArrayList<>();
        for (int position = 0; position < this.maxPositions; position++) {
            for (int dimension = 0; dimension < this.embeddingDimension; dimension++) {
                parameters.add(this.weights[position][dimension]);
            }
        }
        return parameters;
    }
}
