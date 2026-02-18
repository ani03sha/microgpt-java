package com.anirudhology.microgpt.nn;

import com.anirudhology.microgpt.autograd.Value;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Linear (fully-connected) layer
 */
public class Linear {

    private final int inputDimension;
    private final int outputDimension;
    private final Value[][] weights;

    public Linear(int inputDimension, int outputDimension, Random random) {
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.weights = new Value[inputDimension][outputDimension];

        // Xavier/Glorot initialization for better gradient flow
        double scale = Math.sqrt(2.0 / (inputDimension + outputDimension));

        for (int i = 0; i < inputDimension; i++) {
            for (int j = 0; j < outputDimension; j++) {
                this.weights[i][j] = new Value(random.nextGaussian() * scale);
            }
        }
    }

    /**
     * Forward pass: y = x @ W + b
     *
     * @param input Input vector (length: inputDimension
     * @return Output vector (length: outputDimension
     */
    public Value[] forward(Value[] input) {
        if (input.length != this.inputDimension) {
            throw new IllegalArgumentException(String.format("Expected input size %d, got %d", this.inputDimension, input.length));
        }

        Value[] output = new Value[this.outputDimension];

        // For each output neuron
        for (int j = 0; j < this.outputDimension; j++) {
            // Start with bias
            Value sum = new Value(0.0);

            // Add weighted inputs: sum = b + x₀*w₀ⱼ + x₁*w₁ⱼ + ...
            for (int i = 0; i < this.inputDimension; i++) {
                sum = sum.add(input[i].multiply(this.weights[i][j]));
            }
            output[j] = sum;
        }
        return output;
    }

    /**
     * Get all parameters for optimization
     */
    public List<Value> parameters() {
        final List<Value> params = new ArrayList<>();

        // Add weights
        for (int i = 0; i < this.inputDimension; i++) {
            for (int j = 0; j < this.outputDimension; j++) {
                params.add(this.weights[i][j]);
            }
        }
        return params;
    }
}
