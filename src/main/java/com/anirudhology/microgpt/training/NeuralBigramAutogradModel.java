package com.anirudhology.microgpt.training;

import com.anirudhology.microgpt.autograd.Value;
import com.anirudhology.microgpt.tokenizer.CharTokenizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class NeuralBigramAutogradModel {

    private final int vocabularySize;
    private final int bosId;
    private final Value[][] weights; // [currentToken][nextToken] logits
    private final List<Value> params;
    private final Random random;

    public NeuralBigramAutogradModel(int vocabularySize, int bosId, long seed) {
        this.vocabularySize = vocabularySize;
        this.bosId = bosId;
        this.random = new Random(seed);
        this.weights = new Value[vocabularySize][vocabularySize];
        this.params = new ArrayList<>(vocabularySize * vocabularySize);

        for (int i = 0; i < vocabularySize; i++) {
            for (int j = 0; j < vocabularySize; j++) {
                Value p = new Value(this.random.nextGaussian() * 0.01, "w_" + i + "_" + j);
                this.weights[i][j] = p;
                this.params.add(p);
            }
        }
    }

    public double train(List<String> docs, CharTokenizer tokenizer, double learningRate, long shuffleSeed) {
        List<String> shuffled = new ArrayList<>(docs);
        Collections.shuffle(shuffled, new Random(shuffleSeed));

        double totalLoss = 0.0;
        long totalPairs = 0;

        for (String doc : shuffled) {
            List<Integer> sequences = tokenizer.withBOSOnBothSides(doc);

            for (int j = 0; j < sequences.size() - 1; j++) {
                int current = sequences.get(j);
                int next = sequences.get(j + 1);

                Value[] probabilities = softmaxValueRow(current);
                Value loss = probabilities[next].log().neg(); // -log(p_true)

                totalLoss += loss.getData();
                totalPairs++;

                zeroParamGrads();
                loss.backward();

                for (Value p : this.params) {
                    p.setData(p.getData() - learningRate * p.getGrad());
                }
            }
        }
        return totalLoss / Math.max(1, totalPairs);
    }

    public double averageNegativeLogLikelihood(List<String> docs, CharTokenizer tokenizer) {
        double total = 0.0;
        long n = 0;

        for (String doc : docs) {
            List<Integer> sequences = tokenizer.withBOSOnBothSides(doc);
            for (int j = 0; j < sequences.size() - 1; j++) {
                int current = sequences.get(j);
                int next = sequences.get(j + 1);
                double[] probabilities = softmaxDoubleRow(current);
                total += -Math.log(probabilities[next] + 1e-12);
                n++;
            }
        }
        return total / Math.max(1, n);
    }

    public String sample(CharTokenizer tokenizer, int maxLength, double temperature) {
        List<Integer> out = new ArrayList<>(maxLength);
        int current = this.bosId;

        for (int step = 0; step < maxLength; step++) {
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

        double max = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            max = Math.max(max, logit);
        }

        double sum = 0.0;
        double[] expectations = new double[this.vocabularySize];
        for (int i = 0; i < this.vocabularySize; i++) {
            expectations[i] = Math.exp(logits[i] - max);
            sum += expectations[i];
        }

        double[] probabilities = new double[this.vocabularySize];
        for (int i = 0; i < this.vocabularySize; i++) {
            probabilities[i] = expectations[i] / sum;
        }
        return probabilities;
    }

    private double[] softmaxDoubleRow(int current) {
        return softmaxDoubleRow(current, 1.0);
    }

    private void zeroParamGrads() {
        for (Value p : this.params) {
            p.setGrad(0.0);
        }
    }

    private Value[] softmaxValueRow(int current) {
        Value[] logits = this.weights[current];

        double maxLogit = Double.NEGATIVE_INFINITY;
        for (Value logit : logits) {
            maxLogit = Math.max(maxLogit, logit.getData());
        }

        Value[] expectations = new Value[this.vocabularySize];
        Value sumExpectations = new Value(0.0);

        for (int i = 0; i < this.vocabularySize; i++) {
            expectations[i] = logits[i].subtract(maxLogit).expectation(); // exp(logit - max)
            sumExpectations = sumExpectations.add(expectations[i]);
        }

        Value[] probabilities = new Value[this.vocabularySize];
        for (int i = 0; i < this.vocabularySize; i++) {
            probabilities[i] = expectations[i].divide(sumExpectations);
        }
        return probabilities;
    }
}
