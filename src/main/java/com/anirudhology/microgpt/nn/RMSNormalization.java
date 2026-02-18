package com.anirudhology.microgpt.nn;

import com.anirudhology.microgpt.autograd.Value;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Root Mean Square Normalization (RMSNorm)
 * <p>
 * This is simpler than Layer Normalization - only normalizes by RMS,
 * no mean subtraction.
 * <p>
 * Formula: output = (x / RMS(x)) * gamma
 * <p>
 * Where RMS(x) = sqrt(mean(x^2)) + epsilon
 * <p>
 * This is used in LLaMa, Mistral, etc.
 */
public class RMSNormalization {

    // Small constant to prevent division by zero
    private static final double EPSILON = 1e-5;
    // Learnable scale parameter (one per dimension)
    // Initialized to 1.0 so initially it is a no-op
    private final Value[] gamma;

    public RMSNormalization(int dimension) {
        this.gamma = new Value[dimension];

        // Initialize gamma to 1.0 - identity transform initially
        for (int i = 0; i < dimension; i++) {
            this.gamma[i] = new Value(1.0, "gamma_" + i);
        }
    }

    /**
     * Normalize a single vector.
     *
     * @param inputs input vector
     * @return normalized vector
     */
    public Value[] forward(Value[] inputs) {
        // Step 1: Compute RMS: sqrt(mean(x^2) + epsilon)
        double sumSquares = 0.0;
        for (Value input : inputs) {
            sumSquares += input.getData() * input.getData();
        }
        double rms = Math.sqrt(sumSquares / inputs.length + EPSILON);

        // Step 2: Normalize and scale
        final Value[] output = new Value[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            // input[i] / rms * gamma[i]
            output[i] = inputs[i].multiply(1.0 / rms).multiply(this.gamma[i]);
        }
        return output;
    }

    /**
     * Normalize a sequence of vectors (apply to each position)
     */
    public Value[][] forward(Value[][] inputs) {
        final Value[][] output = new Value[inputs.length][];
        for (int i = 0; i < inputs.length; i++) {
            output[i] = forward(inputs[i]);
        }
        return output;
    }

    /**
     * Get all trainable parameters (just gamma)
     */
    public List<Value> parameters() {
        return new ArrayList<>(Arrays.asList(this.gamma));
    }
}
