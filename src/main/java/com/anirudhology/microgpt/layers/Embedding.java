package com.anirudhology.microgpt.layers;

import com.anirudhology.microgpt.autograd.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Embedding {

    private final int vocabularySize;
    private final int embeddingDimension;
    private final Value[][] weights;

    public Embedding(int vocabularySize, int embeddingDimension, Random random) {
        this.vocabularySize = vocabularySize;
        this.embeddingDimension = embeddingDimension;
        this.weights = new Value[vocabularySize][embeddingDimension];

        // Initialize with small random values
        // Scale: ~1/sqrt(embeddingDimension) for better gradient flow
        double scale = 1.0 / Math.sqrt(embeddingDimension);
        for (int i = 0; i < vocabularySize; i++) {
            for (int j = 0; j < embeddingDimension; j++) {
                double value = random.nextGaussian() * scale;
                this.weights[i][j] = new Value(value, "emb_" + i + "_" + j);
            }
        }
    }

    /**
     * Look up embedding for a single token.
     *
     * @param tokenId The token ID
     * @return Embedding vector (length =
     * embeddingDim)
     */
    public Value[] forward(int tokenId) {
        return this.weights[tokenId];
    }

    /**
     * Look up and concatenate embeddings for multiple tokens.
     * <p>
     * Example: context = ['e', 'm', 'm'] with
     * embeddingDim=6
     * Returns: [emb_e || emb_m || emb_m] (length =
     * 3 × 6 = 18)
     *
     * @param tokenIds Array of token IDs (context)
     * @return Flattened concatenated embeddings
     */
    public Value[] forward(int[] tokenIds) {
        final Value[] result = new Value[tokenIds.length * this.embeddingDimension];

        for (int i = 0; i < tokenIds.length; i++) {
            Value[] embedding = this.weights[tokenIds[i]];
            System.arraycopy(embedding, 0, result, i * this.embeddingDimension, this.embeddingDimension);
        }
        return result;
    }

    public List<Value> parameters() {
        final List<Value> params = new ArrayList<>();
        for (int i = 0; i < this.vocabularySize; i++) {
            for (int j = 0; j < this.embeddingDimension; j++) {
                params.add(this.weights[i][j]);
            }
        }
        return params;
    }

    public int getEmbeddingDimension() {
        return this.embeddingDimension;
    }
}
