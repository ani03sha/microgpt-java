package com.anirudhology.microgpt.training;

import com.anirudhology.microgpt.tokenizer.CharTokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class BigramModel {

    private final int vocabularySize;
    private final int bosId;
    private final int[][] counts;
    private final double[][] probabilities;
    private final Random random;

    public BigramModel(int vocabularySize, int bosId, long seed) {
        this.vocabularySize = vocabularySize;
        this.bosId = bosId;
        this.counts = new int[vocabularySize][vocabularySize];
        this.probabilities = new double[vocabularySize][vocabularySize];
        this.random = new Random(seed);
    }

    public void fit(List<String> docs, CharTokenizer tokenizer, double alpha) {
        // Count transitions
        for (String doc : docs) {
            final List<Integer> sequence = tokenizer.withBOSOnBothSides(doc);
            for (int i = 0; i < sequence.size() - 1; i++) {
                int current = sequence.get(i);
                int next = sequence.get(i + 1);
                this.counts[current][next]++;
            }
        }

        // Normalize to probabilities with Laplace smoothing
        for (int current = 0; current < this.vocabularySize; current++) {
            double rowSum = 0.0;
            for (int next = 0; next < this.vocabularySize; next++) {
                rowSum += this.counts[current][next] + alpha;
            }
            for (int next = 0; next < this.vocabularySize; next++) {
                this.probabilities[current][next] = (this.counts[current][next] + alpha) / rowSum;
            }
        }
    }

    public double averageNegativeLogLikelihood(List<String> docs, CharTokenizer tokenizer) {
        double total = 0.0;
        long nPairs = 0;
        for (String doc : docs) {
            final List<Integer> sequence = tokenizer.withBOSOnBothSides(doc);
            for (int i = 0; i < sequence.size() - 1; i++) {
                int current = sequence.get(i);
                int next = sequence.get(i + 1);
                double probability = this.probabilities[current][next];
                total += -Math.log(probability);
                nPairs++;
            }
        }
        return total / Math.max(1, nPairs);
    }

    public String sample(CharTokenizer tokenizer, int maxLength) {
        final List<Integer> out = new ArrayList<>();
        int current = this.bosId;

        for (int step = 0; step < maxLength; step++) {
            int next = sampleFromDistribution(this.probabilities[current]);
            if (next == this.bosId) {
                break;
            }
            out.add(next);
            current = next;
        }
        return tokenizer.decode(out);
    }

    private int sampleFromDistribution(double[] distribution) {
        double r = random.nextDouble();
        double cdf = 0.0;
        for (int i = 0; i < distribution.length; i++) {
            cdf += distribution[i];
            if (r <= cdf) {
                return i;
            }
        }
        return distribution.length - 1; // Numeric fallback
    }
}
