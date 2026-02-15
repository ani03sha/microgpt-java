package com.anirudhology.microgpt.training;

import com.anirudhology.microgpt.tokenizer.CharacterTokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * BigramModel is a statistical language model that predicts the next character
 * based solely on the current character. It is the simplest baseline model which
 * is purely frequency based with no neural network involved.
 * <p>
 * Properties:
 * 1. No learning or optimization, just counting and normalizing
 * 2. Memoryless, only current character matters (Markov property)
 * 3. Fast: no gradient descent, single pass through data
 * 4. Interpretable: we can inspect probability matrix directly
 * 5. Cannot capture long-range dependencies
 * <p>
 * This baseline model serves as a benchmark and neural models should beat this.
 */
public class BaselineBigramModel {

    // Total number of unique characters + BOS
    private final int vocabularySize;

    // The id of the BOS (boundary) token
    private final int bosId;

    // Tracks how many times each character pair appears
    private final int[][] counts;

    // Normalized version of counts where each row is
    // a probability distribution. Each row sums to 1.0
    // probability[i][j] => probability of seeing character j
    // given we just saw character i
    private final double[][] probabilities;

    private final Random random;

    public BaselineBigramModel(int vocabularySize, int bosId, long seed) {
        this.vocabularySize = vocabularySize;
        this.bosId = bosId;
        this.counts = new int[vocabularySize][vocabularySize];
        this.probabilities = new double[vocabularySize][vocabularySize];
        this.random = new Random(seed);
    }

    /**
     * This is the training step where we process all the corpus
     *
     * @param documents input documents
     * @param tokenizer tokenizer to tokenize the input documents
     * @param alpha     very small values to avoid zero probabilities
     */
    public void fit(List<String> documents, CharacterTokenizer tokenizer, double alpha) {
        // Count occurrences of each adjacent character pair.
        for (String document : documents) {
            final List<Integer> sequence = tokenizer.withBOSOnBothSides(document);
            for (int i = 0; i < sequence.size() - 1; i++) {
                int current = sequence.get(i);
                int next = sequence.get(i + 1);
                this.counts[current][next]++;
            }
        }

        // Normalize to probabilities with Laplace smoothing (alpha-smoothing).
        // Laplace smoothing is necessary to avoid zero probabilities for a
        // character pair which was never seen in the corpus.
        //
        // This also acts as a prior belief that all transitions are possible.
        for (int current = 0; current < this.vocabularySize; current++) {
            double rowSum = 0.0;
            // Calculate sum with smoothing
            for (int next = 0; next < this.vocabularySize; next++) {
                rowSum += this.counts[current][next] + alpha;
            }
            // Normalize each cell
            for (int next = 0; next < this.vocabularySize; next++) {
                this.probabilities[current][next] = (this.counts[current][next] + alpha) / rowSum;
            }
        }
    }

    /**
     * Measures how well the model predicts the data.
     * <p>
     * - For each character transition, lookup the predicted probability
     * - Take negative lgo: - log(P(next|current))
     * - Average across all transitions
     * <p>
     * The aim is to lower this value. Lower the value, better the model
     *
     * @param documents all documents in the corpus
     * @param tokenizer to tokenize the string into characters
     * @return negative log likelihood
     */
    public double averageNegativeLogLikelihood(List<String> documents, CharacterTokenizer tokenizer) {
        double total = 0.0;
        long nPairs = 0;
        for (String document : documents) {
            final List<Integer> sequence = tokenizer.withBOSOnBothSides(document);
            for (int i = 0; i < sequence.size() - 1; i++) {
                // Get character pair
                int current = sequence.get(i);
                int next = sequence.get(i + 1);
                double probability = this.probabilities[current][next];
                // Negative log likelihood
                total -= Math.log(probability);
                nPairs++;
            }
        }
        return total / Math.max(1, nPairs);
    }

    /**
     * This is the generation process. For example, generating a name
     * This is autoregressive which means each prediction becomes input
     * for the next prediction
     *
     * @param tokenizer tokenizes the text
     * @param maxLength max length of predicted text
     * @return generated text
     */
    public String sample(CharacterTokenizer tokenizer, int maxLength) {
        final List<Integer> out = new ArrayList<>();

        // Start with the BOS token
        int current = this.bosId;

        // Iterate until the maxLength reached
        for (int step = 0; step < maxLength; step++) {
            int next = sampleFromDistribution(this.probabilities[current]);
            // Stop if we generate BOS (end marker)
            if (next == this.bosId) {
                break;
            }
            out.add(next);
            // Current character becomes context for next prediction
            current = next;
        }
        return tokenizer.decode(out);
    }

    /**
     * This method ensures each character is sampled proportional
     * to its probability.
     *
     * @param probabilities w.r.t current character
     * @return sampled id
     */
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
}
