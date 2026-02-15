package com.anirudhology.microgpt.training;

import com.anirudhology.microgpt.autograd.Value;
import com.anirudhology.microgpt.tokenizer.CharacterTokenizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * It does the same task as NeuralBigramModel but uses automatic differentiation
 * instead of manually computed gradients.
 */
public class NeuralBigramAutogradModel {

    // Size of the vocabulary + BOS
    private final int vocabularySize;

    // BOS token id
    private final int bosId;

    // Learnable parameters (logits) - raw unnormalized scores (Value objects)
    // The value weights[i][j] is a score for character j following character i.
    // These are smart numbers that track gradients.
    private final Value[][] weights;

    // Flat list of all learnable parameters for easy iteration during updates
    private final List<Value> params;

    // For initialization and sampling
    private final Random random;

    public NeuralBigramAutogradModel(int vocabularySize, int bosId, long seed) {
        this.vocabularySize = vocabularySize;
        this.bosId = bosId;
        this.random = new Random(seed);
        this.weights = new Value[vocabularySize][vocabularySize];
        this.params = new ArrayList<>(vocabularySize * vocabularySize);

        // Creating value objects instead of primitives with data and label
        for (int i = 0; i < vocabularySize; i++) {
            for (int j = 0; j < vocabularySize; j++) {
                // The label "w_i_j" means "weight" from character i to character j.
                Value p = new Value(this.random.nextGaussian() * 0.01, "w_" + i + "_" + j);
                this.weights[i][j] = p;
                this.params.add(p);
            }
        }
    }

    /**
     * The training method
     *
     * @param documents    input corpus
     * @param tokenizer    tokenizes input data to character tokens
     * @param learningRate rate at which learning is done
     * @param shuffleSeed  seed to shuffle the sorted documents
     * @return NLL on training dataset
     */
    public double train(List<String> documents, CharacterTokenizer tokenizer, double learningRate, long shuffleSeed) {
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
        for (String doc : shuffledDocuments) {
            List<Integer> sequences = tokenizer.withBOSOnBothSides(doc);
            for (int j = 0; j < sequences.size() - 1; j++) {
                int current = sequences.get(j);
                int next = sequences.get(j + 1);

                // Forward pass - compute probabilities using softmax
                // which converts logits into probabilities
                Value[] probabilities = softmaxValueRow(current);

                // Negative log converts probabilities to loss.
                // Perfect prediction (P=1.0) → Loss = 0
                // Bad prediction (P-0.1) → Loss = 2.303
                // Terrible prediction P(0.01) → Loss = 4.605
                Value loss = probabilities[next].log().neg(); // -log(p_true)

                totalLoss += loss.getData();
                totalPairs++;

                // Gradients accumulate (use += in backwardFn), so must reset
                // before each backward pass.
                zeroParamGrads();

                /* Due to this line, no manual gradient formulas are needed.
                *
                * 1. Builds topological order of all nodes
                * 2. Sets loss.grad = 1.0 (∂(loss)/∂(loss) = 1)
                * 3. Calls each node's backwardFn in reverse order
                * 4. Propagates gradients all the way back to weights
                */
                loss.backward();

                // Update weights (SGD)
                for (Value p : this.params) {
                    p.setData(p.getData() - learningRate * p.getGradient());
                }
            }
        }
        return totalLoss / Math.max(1, totalPairs);
    }

    public double averageNegativeLogLikelihood(List<String> docs, CharacterTokenizer tokenizer) {
        double total = 0.0;
        long n = 0;

        for (String doc : docs) {
            List<Integer> sequences = tokenizer.withBOSOnBothSides(doc);
            for (int j = 0; j < sequences.size() - 1; j++) {
                int current = sequences.get(j);
                int next = sequences.get(j + 1);
                double[] probabilities = softmaxDoubleRow(current);
                total -= Math.log(probabilities[next] + 1e-12);
                n++;
            }
        }
        return total / Math.max(1, n);
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
     *  @return generated word
     */
    public String sample(CharacterTokenizer tokenizer, int maxLength, double temperature) {
        // Stores generated character ids
        List<Integer> out = new ArrayList<>();
        // Tracks the current context and starts with BOS token
        int current = this.bosId;

        // The variable maxLength is a safety limit to prevent infinite generation
        for (int step = 0; step < maxLength; step++) {
            // We are using doubles here because we are not training, just sampling.
            // This makes it faster as no computation graph overhead is there, and simpler.
            // During inference, autograd is not needed.
            double[] probabilities = softmaxDoubleRow(current, temperature);
            int next = sampleFromDistribution(probabilities);

            if (next == this.bosId) {
                break;
            }
            out.add(next);
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

    private double[] softmaxDoubleRow(int current, double temperature) {
        double[] logits = new double[this.vocabularySize];
        for (int i = 0; i < this.vocabularySize; i++) {
            logits[i] = this.weights[current][i].getData() / temperature;
        }

        // Find max for numerical stability
        double max = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            max = Math.max(max, logit);
        }

        double sum = 0.0;
        double[] exponents = new double[this.vocabularySize];
        for (int i = 0; i < this.vocabularySize; i++) {
            exponents[i] = Math.exp(logits[i] - max);
            sum += exponents[i];
        }

        double[] probabilities = new double[this.vocabularySize];
        for (int i = 0; i < this.vocabularySize; i++) {
            probabilities[i] = exponents[i] / sum;
        }
        return probabilities;
    }

    private double[] softmaxDoubleRow(int current) {
        return softmaxDoubleRow(current, 1.0);
    }

    private void zeroParamGrads() {
        for (Value p : this.params) {
            p.setGradient(0.0);
        }
    }

    private Value[] softmaxValueRow(int current) {
        Value[] logits = this.weights[current];

        // Find max for numerical stability. Note this is still a double.
        // Why? Max operation isn't differentiable in the traditional sense
        // (gradient is 0 or 1), and it's just for numerical stability.
        // We subtract it from the data, not the computational graph
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (Value logit : logits) {
            maxLogit = Math.max(maxLogit, logit.getData());
        }

        // Compute exp(logit - max) and sum
        Value[] exponents = new Value[this.vocabularySize];
        Value sumExpectations = new Value(0.0);

        for (int i = 0; i < this.vocabularySize; i++) {
            exponents[i] = logits[i].subtract(maxLogit).exp(); // exp(logit - max)
            sumExpectations = sumExpectations.add(exponents[i]);
        }

        // Normalize probabilities
        Value[] probabilities = new Value[this.vocabularySize];
        for (int i = 0; i < this.vocabularySize; i++) {
            probabilities[i] = exponents[i].divide(sumExpectations);
        }
        return probabilities;
    }
}
