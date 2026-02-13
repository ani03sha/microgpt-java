package com.anirudhology.microgpt.training;

import com.anirudhology.microgpt.tokenizer.CharTokenizer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class NeuralBigramModel {

    private static final Logger log = LoggerFactory.getLogger(NeuralBigramModel.class);
    private final int vocabularySize;
    private final int bosId;
    private final double[][] weights;
    private final Random random;

    public NeuralBigramModel(int vocabularySize, int bosId, long seed) {
        this.vocabularySize = vocabularySize;
        this.bosId = bosId;
        this.weights = new double[vocabularySize][vocabularySize];
        this.random = new Random(seed);

        // Small random init
        for (int i = 0; i < vocabularySize; i++) {
            for (int j = 0; j < vocabularySize; j++) {
                weights[i][j] = random.nextGaussian() * 0.01;
            }
        }
    }

    public double trainEpoch(List<String> docs, CharTokenizer tokenizer, double learningRate, long shuffleSeed) {
        final List<String> shuffled = new ArrayList<>(docs);
        Collections.shuffle(shuffled, new Random(shuffleSeed));

        double totalLoss = 0.0;
        long totalPairs = 0;

        for (String doc : shuffled) {
            List<Integer> sequence = tokenizer.withBOSOnBothSides(doc);

            for (int t = 0; t < sequence.size() - 1; t++) {
                int current = sequence.get(t);
                int next = sequence.get(t + 1);

                double[] probabilities = softmax(this.weights[current]);

                // NLL loss: -log(p_true)
                totalLoss += -Math.log(probabilities[next] + 1e-12);
                totalPairs++;

                // Gradient for softmax + cross-entropy
                // dL/dlogit_j = p_j - 1 (j == next)
                for (int j = 0; j < this.vocabularySize; j++) {
                    double gradient = probabilities[j] - (j == next ? 1.0 : 0.0);
                    this.weights[current][j] -= learningRate * gradient; // SGD update
                }
            }
        }
        return totalLoss / Math.max(1, totalPairs);
    }

    public double averageNegativeLogLikelihood(List<String> docs, CharTokenizer tokenizer) {
        double totalLoss = 0.0;
        long totalPairs = 0;

        for (String doc : docs) {
            List<Integer> sequence = tokenizer.withBOSOnBothSides(doc);
            for (int t = 0; t < sequence.size() - 1; t++) {
                int current = sequence.get(t);
                int next = sequence.get(t + 1);
                double[] probabilities = softmax(this.weights[current]);
                totalLoss += -Math.log(probabilities[next] + 1e-12);
                totalPairs++;
            }
        }
        return totalLoss / Math.max(1, totalPairs);
    }

    public String sample(CharTokenizer tokenizer, int maxLength, double temperature) {
        List<Integer> out = new ArrayList<>();
        int current = this.bosId;

        for (int step = 0; step < maxLength; step++) {
            double[] logits = new double[this.vocabularySize];
            for (int j = 0; j < this.vocabularySize; j++) {
                logits[j] = this.weights[current][j] / temperature;
            }

            double[] probabilities = softmax(logits);
            int next = sampleFromDistribution(probabilities);

            if (next == this.bosId) {
                break;
            }
            out.add(next);
            current = next;
        }
        return tokenizer.decode(out);
    }

    private double[] softmax(double[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            max = Math.max(max, logit);
        }

        double sum = 0.0;
        double[] expectations = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            expectations[i] = Math.exp(logits[i] - max); // Stability trick
            sum += expectations[i];
        }

        double[] probabilities = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = expectations[i] / sum;
        }
        return probabilities;
    }

    private int sampleFromDistribution(double[] probabilities) {
        double r = random.nextDouble();
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
