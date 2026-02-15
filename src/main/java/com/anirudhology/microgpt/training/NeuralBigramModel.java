package com.anirudhology.microgpt.training;

import com.anirudhology.microgpt.tokenizer.CharacterTokenizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * This is the neural network version of the bigram model.
 * Instead of counting frequencies, it learns a weight matrix
 * through gradient descent.
 * <p>
 * Thus, same prediction task (next character given current)
 * but using optimization instead of statistics.
 */
public class NeuralBigramModel {

    // Number of unique characters + BOS
    private final int vocabularySize;

    // BOS token id
    private final int bosId;

    // Learnable parameters (logits) - raw unnormalized scores
    // (can be any real number).
    // The value weights[i][j] is a score for character j following
    // character i
    private final double[][] weights;

    // For initialization and sampling
    private final Random random;

    public NeuralBigramModel(int vocabularySize, int bosId, long seed) {
        this.vocabularySize = vocabularySize;
        this.bosId = bosId;
        this.weights = new double[vocabularySize][vocabularySize];
        this.random = new Random(seed);

        /* Small random initialization.
         * - Sample from normal distribution: N(0, 0.01^2)
         * - Mean = 0, Standard Deviation: 0.01
         * - Small because starts near uniform distribution
         * - Random to break symmetry, if all weights are identical,
         *   gradients would be identical
         */
        for (int i = 0; i < vocabularySize; i++) {
            for (int j = 0; j < vocabularySize; j++) {
                this.weights[i][j] = this.random.nextGaussian() * 0.01;
            }
        }
    }

    /**
     * The training method.
     *
     * @param documents    the input corpus
     * @param tokenizer    to tokenize the corpus
     * @param learningRate rate at which learning is happening
     * @param shuffleSeed  random seed for shuffling
     * @return NLL on training dataset
     */
    public double trainEpoch(
            List<String> documents,
            CharacterTokenizer tokenizer,
            double learningRate,
            long shuffleSeed
    ) {
        /* Why shuffling?
         * - Prevents model from learning data order
         * - Each epoch sees examples in different order
         * - Reduces overfitting to sequence patterns
         * - SGD works better with randomized data.
         * */
        final List<String> shuffledDocuments = new ArrayList<>(documents);
        Collections.shuffle(shuffledDocuments, new Random(shuffleSeed));

        double totalLoss = 0.0;
        long totalPairs = 0;

        // Process each document just like baseline bigram
        for (String document : shuffledDocuments) {
            List<Integer> sequence = tokenizer.withBOSOnBothSides(document);

            for (int i = 0; i < sequence.size() - 1; i++) {
                int current = sequence.get(i);
                int next = sequence.get(i + 1);

                // Forward pass - compute probabilities using softmax
                // which converts logits into probabilities
                double[] probabilities = softmax(this.weights[current]);

                // Negative log converts probabilities to loss.
                // Perfect prediction (P=1.0) → Loss = 0
                // Bad prediction (P-0.1) → Loss = 2.303
                // Terrible prediction P(0.01) → Loss = 4.605
                // Epsilon (1e-12) prevents log(0) = -∞ for numerical safety
                totalLoss -= Math.log(probabilities[next] + 1e-12);
                totalPairs++;

                // Gradient for softmax + cross-entropy
                // dL/dlogit_j = p_j - 1 (j == next)
                // Positive gradient → weight is too high → decrease it
                // Negative gradient → weight is too low → increase it (via subtraction of negative)
                // Magnitude shows how much correction needed
                for (int j = 0; j < this.vocabularySize; j++) {
                    double gradient = probabilities[j] - (j == next ? 1.0 : 0.0);
                    /*
                     * Stochastic Gradient Descent (SGD)
                     * - Stochastic: Update after each example (not batch)
                     * - Gradient: Direction of steepest increase in loss
                     * - Descent: Go opposite direction (subtract) to decrease loss
                     */
                    this.weights[current][j] -= learningRate * gradient;
                }
            }
        }
        return totalLoss / Math.max(1, totalPairs);
    }

    public double averageNegativeLogLikelihood(List<String> docs, CharacterTokenizer tokenizer) {
        double totalLoss = 0.0;
        long totalPairs = 0;

        for (String doc : docs) {
            List<Integer> sequence = tokenizer.withBOSOnBothSides(doc);
            for (int t = 0; t < sequence.size() - 1; t++) {
                int current = sequence.get(t);
                int next = sequence.get(t + 1);
                double[] probabilities = softmax(this.weights[current]);
                totalLoss -= Math.log(probabilities[next] + 1e-12);
                totalPairs++;
            }
        }
        return totalLoss / Math.max(1, totalPairs);
    }

    /**
     * This is the heart of text generation, and it generated new text using trained model.
     * This is how we go from a trained NN to actual creative output (like generating name,
     * words, sentences). This makes our model a "generative" model.
     * <p>
     * It generated text one character at a time in the autoregressive manner:
     * 1. Start with BOS (beginning marker)
     * 2. Predict next character based on current character
     * 3. Sample from probability distribution
     * 4. Use sampled character as input for next prediction
     * 5. Repeat until BOS (end marker) or max length
     *
     * @return generated word
     */
    public String sample(CharacterTokenizer tokenizer, int maxLength, double temperature) {
        // Stores generated character ids
        List<Integer> out = new ArrayList<>();
        // Tracks the current context and starts with BOS token
        int current = this.bosId;

        // The variable maxLength is a safety limit to prevent infinite generation
        for (int step = 0; step < maxLength; step++) {
            // Get the row from the weight matrix for the current character
            double[] logits = new double[this.vocabularySize];
            for (int j = 0; j < this.vocabularySize; j++) {
                // Apply temperature scaling
                logits[j] = this.weights[current][j] / temperature;
            }

            // Softmax converts logits → valid probability distribution
            double[] probabilities = softmax(logits);
            // Sample next character by stochastic sampling (don't just pick
            // the max but sample probabilistically)
            int next = sampleFromDistribution(probabilities);

            // Check for end of sequence
            if (next == this.bosId) {
                break;
            }
            out.add(next);
            // Generated character becomes the context for next prediction.
            // This is the autoregressive magic!
            current = next;
        }
        return tokenizer.decode(out);
    }

    /**
     * Gives valid probability distribution summing to 1.0
     *
     * @param logits for the current character
     * @return converted probabilities from logits
     */
    private double[] softmax(double[] logits) {
        // 1. Find max for numerical stability
        double max = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            max = Math.max(max, logit);
        }

        // Step 2: Compute exp(logit - max) and sum
        double sum = 0.0;
        double[] exponents = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            // Stability trick to prevent numerical issues
            exponents[i] = Math.exp(logits[i] - max);
            sum += exponents[i];
        }

        // Step 3: Normalize
        double[] probabilities = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = exponents[i] / sum;
        }
        return probabilities;
    }

    /**
     * This method ensures each character is sampled proportional
     * to its probability.
     *
     * @param probabilities w.r.t current character
     * @return sampled id
     */
    private int sampleFromDistribution(double[] probabilities) {
        double r = this.random.nextDouble();
        double cdf = 0.0;
        for (int i = 0; i < probabilities.length; i++) {
            cdf += probabilities[i];
            if (r <= cdf) {
                return i;
            }
        }
        return probabilities.length - 1;
    }
}
