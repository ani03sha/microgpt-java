package com.anirudhology.microgpt.layers;

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
    private final Value[] bias;

    public Linear(int inputDimension, int outputDimension) {
        this.inputDimension = inputDimension;
        this.outputDimension = outputDimension;
        this.weights = new Value[inputDimension][outputDimension];
        this.bias = new Value[outputDimension];

        final Random random = new Random();

        // Xavier/Glorot initialization for better gradient flow
        double scale = Math.sqrt(2.0 / (inputDimension + outputDimension));

        for (int i = 0; i < inputDimension; i++) {
            for (int j = 0; j < outputDimension; j++) {
                this.weights[i][j] = new Value(random.nextGaussian() * scale);
            }
        }

        for (int j = 0; j < outputDimension; j++) {
            this.bias[j] = new Value(0.0); // Initialize bias to zero
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
            Value sum = this.bias[j];

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

        // Add biases
        for (int j = 0; j < this.outputDimension; j++) {
            params.add(this.bias[j]);
        }
        return params;
    }
}
