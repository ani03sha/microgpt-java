package com.anirudhology.microgpt.model;

import com.anirudhology.microgpt.autograd.Value;
import com.anirudhology.microgpt.data.TrainingExample;
import com.anirudhology.microgpt.nn.Embedding;
import com.anirudhology.microgpt.nn.Linear;
import com.anirudhology.microgpt.nn.PositionalEmbedding;
import com.anirudhology.microgpt.tokenizer.CharacterTokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Simple MLP language model.
 * <p>
 * Architecture:
 * Input (context) → Embedding → Concatenate →
 * Hidden Layer (tanh) → Output → Softmax
 */
public class MLPLanguageModel {

    private final int vocabularySize;
    private final int blockSize;

    private final Embedding embedding;
    private final PositionalEmbedding positionalEmbedding;
    private final Linear hiddenLayer;
    private final Linear outputLayer;
    private final List<Value> allParameters;

    private final Random random;

    public MLPLanguageModel(
            int vocabularySize,
            int blockSize,
            int embeddingDimension,
            int hiddenDimension,
            long seed
    ) {
        this.vocabularySize = vocabularySize;
        this.blockSize = blockSize;
        this.random = new Random(seed);

        // Build layers
        this.embedding = new Embedding(vocabularySize, embeddingDimension, this.random);
        this.positionalEmbedding = new PositionalEmbedding(blockSize, embeddingDimension, this.random);

        int inputDimension = blockSize * embeddingDimension;
        // Flattened embeddings
        this.hiddenLayer = new Linear(inputDimension, hiddenDimension, this.random);
        this.outputLayer = new Linear(hiddenDimension, vocabularySize, this.random);

        // Collect all parameters
        this.allParameters = new ArrayList<>();
        this.allParameters.addAll(this.embedding.parameters());
        this.allParameters.addAll(this.positionalEmbedding.parameters());
        this.allParameters.addAll(this.hiddenLayer.parameters());
        this.allParameters.addAll(this.outputLayer.parameters());

        System.out.printf("MLP Model initialized: %d parameters%n", this.allParameters.size());
    }

    /**
     * Forward pass: compute logits for next token
     * <p>
     * It is only based on tokens.
     *
     * @param context array of token ids
     * @return logits for each token in vocabulary
     */
    public Value[] forward(int[] context) {
        // 1. Embedding lookup and concatenation
        final Value[] x = this.embedding.forward(context);

        // 2. Hidden layer with tanh activation
        Value[] h = this.hiddenLayer.forward(x);
        h = tanh(h);
        // 3. Output layer
        return outputLayer.forward(h);
    }

    /**
     * Forward pass: compute logits for next token.
     * <p>
     * It is based on tokens and positions
     *
     * @param context array of token ids
     * @return logits for each token in vocabulary
     */
    public Value[] positionalForward(int[] context) {
        final int sequenceLength = context.length;

        // 1. Get token embeddings for each position
        final Value[][] tokenEmbeddings = new Value[sequenceLength][];
        for (int position = 0; position < sequenceLength; position++) {
            tokenEmbeddings[position] = this.embedding.forward(context[position]);
        }

        // 2. Get positional embeddings for each position
        final Value[][] positionalEmbeddings = this.positionalEmbedding.forwardAll(sequenceLength);

        // 3. Combine token and positional embeddings element wise
        final Value[][] combinedEmbeddings = new Value[sequenceLength][];
        for (int position = 0; position < sequenceLength; position++) {
            combinedEmbeddings[position] = new Value[tokenEmbeddings[position].length];
            for (int dimension = 0; dimension < tokenEmbeddings[position].length; dimension++) {
                // Add token embeddings and positional embeddings
                combinedEmbeddings[position][dimension] = tokenEmbeddings[position][dimension].add(positionalEmbeddings[position][dimension]);
            }
        }

        // 4. Flatten combined embeddings into single vector
        final Value[] result = new Value[sequenceLength * tokenEmbeddings[0].length];
        for (int position = 0; position < sequenceLength; position++) {
            System.arraycopy(combinedEmbeddings[position], 0, result, position * tokenEmbeddings[0].length, tokenEmbeddings[0].length);
        }

        // 5. Hidden layer with tanh activation
        Value[] h = this.hiddenLayer.forward(result);
        h = tanh(h);

        // 6. Output layer
        return this.outputLayer.forward(h);
    }

    /**
     * Compute loss for a training example
     *
     * @param example instance of TrainingExample
     * @return Cross-entropy loss
     */
    public Value computeLoss(TrainingExample example, boolean shouldUsePositionalEncoding) {
        final Value[] logits = shouldUsePositionalEncoding
                ? positionalForward(example.context())
                : forward(example.context());
        final Value[] probabilities = softmax(logits);
        // Negative log likelihood
        return probabilities[example.target()].log().neg();
    }

    /**
     * Train on a single example (one SGD step)
     *
     * @param example      instance of TrainingExample
     * @param learningRate Learning rate
     * @return Loss Value
     */
    public double trainStep(TrainingExample example, double learningRate, boolean shouldUsePositionalEncoding) {
        // Forward pass
        final Value loss = computeLoss(example, shouldUsePositionalEncoding);
        double lossValue = loss.getData();

        // Zero gradients
        zeroGradient();

        // Backward pass
        loss.backward();

        // DEBUG: Monitor gradients occasionally
        if (Math.random() < 0.001) {  // 0.1% of the time
            double totalGrad = 0.0;
            double maxGrad = 0.0;
            double totalData = 0.0;

            for (Value p : this.allParameters) {
                double absGrad =
                        Math.abs(p.getGradient());
                totalGrad += absGrad;
                maxGrad = Math.max(maxGrad, absGrad);
                totalData += Math.abs(p.getData());
            }

            System.out.printf("[DEBUG] loss=%.4f, avg|grad|=%.6f, max|grad|=%.6f, avg|param|=%.6f%n",
                    lossValue,
                    totalGrad / allParameters.size(),
                    maxGrad,
                    totalData / allParameters.size());
        }


        // Update parameters: θ = θ - lr * ∇θ
        for (Value parameter : this.allParameters) {
            double newValue = parameter.getData() - learningRate * parameter.getGradient();
            parameter.setData(newValue);
        }
        return lossValue;
    }

    /**
     * Generate text autoregressively
     *
     * @param tokenizer   Tokenizer for decoding
     * @param maxLength   Maximum length to generate
     * @param temperature Sampling temperature
     * @return Generated text
     */
    public String generate(CharacterTokenizer tokenizer, int maxLength, double temperature) {
        final List<Integer> generated = new ArrayList<>();
        final int bosId = tokenizer.getBOSId();

        // Initialize context with BOS
        final int[] context = new int[this.blockSize];
        Arrays.fill(context, bosId);

        for (int step = 0; step < maxLength; step++) {
            // Get logits (extract data for sampling)
            final Value[] logits = positionalForward(context);
            double[] logitData = new double[this.vocabularySize];
            for (int i = 0; i < this.vocabularySize; i++) {
                logitData[i] = logits[i].getData() / temperature;
            }

            // Sample next token
            final double[] probabilities = softmaxDouble(logitData);
            int nextToken = sample(probabilities);

            // Stop if BOS generated
            if (nextToken == bosId) {
                break;
            }
            generated.add(nextToken);

            // Shift context and append new token
            shiftLeft(context, nextToken);
        }

        return tokenizer.decode(generated);
    }

    private void zeroGradient() {
        for (Value parameter : this.allParameters) {
            parameter.setGradient(0.0);
        }
    }

    private Value[] softmax(Value[] logits) {
        // Find max for numerical stability
        double max = Double.NEGATIVE_INFINITY;
        for (Value logit : logits) {
            max = Math.max(max, logit.getData());
        }

        // Compute exp(logit - max)
        final Value[] exponents = new Value[logits.length];
        Value sum = new Value(0.0);
        for (int i = 0; i < logits.length; i++) {
            exponents[i] = logits[i].subtract(max).exp();
            sum = sum.add(exponents[i]);
        }

        // Normalize
        final Value[] probabilities = new Value[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = exponents[i].divide(sum);
        }
        return probabilities;
    }

    private double[] softmaxDouble(double[] logits) {
        double max = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            max = Math.max(max, logit);
        }

        double sum = 0.0;
        final double[] exponents = new double[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exponents[i] = Math.exp(logits[i] - max);
            sum += exponents[i];
        }

        final double[] probabilities = new double[exponents.length];
        for (int i = 0; i < exponents.length; i++) {
            probabilities[i] = exponents[i] / sum;
        }
        return probabilities;
    }

    private Value[] tanh(Value[] x) {
        final Value[] result = new Value[x.length];
        for (int i = 0; i < x.length; i++) {
            result[i] = x[i].tanh();
        }
        return result;
    }

    private void shiftLeft(int[] context, int newToken) {
        System.arraycopy(context, 1, context, 0, context.length - 1);
        context[context.length - 1] = newToken;
    }

    private int sample(double[] probabilities) {
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
