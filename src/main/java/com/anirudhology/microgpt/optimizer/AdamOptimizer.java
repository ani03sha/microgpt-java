package com.anirudhology.microgpt.optimizer;

import com.anirudhology.microgpt.autograd.Value;

import java.util.List;

/**
 * Adam optimizer (Adaptive Moment Estimation).
 * <p>
 * Combines momentum + adaptive learning rates for faster convergence.
 * <p>
 * Update rule:
 * m = beta1 * m + (1 - beta1) * grad (first moment)
 * v = beta2 * v + (1 - beta2) * grad² (second moment)
 * <p>
 * m_hat = m / (1 - beta1^t) (bias correction)
 * v_hat = v / (1 - beta2^t) (bias correction)
 * param = param - lr * m_hat / (√v_hat + ε)
 */
public class AdamOptimizer {

    private final double beta1;
    private final double beta2;
    private final double epsilon;

    // Fist and second moment estimates, one per parameter
    private final double[] m;
    private final double[] v;

    // Track hoe many steps taken (for bias correction)
    private int step;

    public AdamOptimizer(List<Value> parameters) {
        this(parameters, 0.85, 0.99, 1e-8);
    }

    public AdamOptimizer(List<Value> parameters, double beta1, double beta2, double epsilon) {
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.step = 0;

        // Initialize moments to zero for all parameters
        this.m = new double[parameters.size()];
        this.v = new double[parameters.size()];
    }

    /**
     * Perform one Adam update step.
     * <p>
     * Call this after loss.backward() has computed gradients
     */
    public void step(List<Value> parameters, double learningRate) {
        this.step++;

        // Bias correction terms
        double biasCorrection1 = 1.0 - Math.pow(this.beta1, this.step); // 1 - β1^t
        double biasCorrection2 = 1.0 - Math.pow(this.beta2, this.step); // 1 - β2^t

        for (int i = 0; i < parameters.size(); i++) {
            Value parameter = parameters.get(i);
            double gradient = parameter.getGradient();

            // Update first moment (momentum)
            this.m[i] = this.beta1 * this.m[i] + (1.0 - this.beta1) * gradient;
            // Update second moment (adaptive learning rate)
            this.v[i] = this.beta2 * this.v[i] + (1.0 - this.beta2) * gradient * gradient;

            // Bias-corrected moments
            double mHat = this.m[i] / biasCorrection1;
            double vHat = this.v[i] / biasCorrection2;

            // Update parameter
            parameter.setData(parameter.getData() - learningRate * mHat / (Math.sqrt(vHat) + this.epsilon));
        }
    }

    /**
     * Zero all parameter gradients.
     * Call this before loss.backward()
     */
    public void zeroGradient(List<Value> parameters) {
        for (Value parameter : parameters) {
            parameter.setGradient(0.0);
        }
    }
}
