package com.anirudhology.microgpt.autograd;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * This is a computational graph node that:
 * 1. Stores a number (forward pass)
 * 2. Tracks how it was created (parent nodes + operation)
 * 3. Automatically computes gradients (backward pass)
 * 4. Enables gradient-based optimization without manual derivative calculations
 * <p>
 * In simple words, it represents a "smart number" that remembers its history and
 * can tell you how changing it affects the final output.
 */
public class Value {

    // The actual number (forward value) this node represents.
    private double data;

    // The derivative of the final output w.r.t this value. Gradient: ∂(output)/∂(this).
    private double gradient;

    // Parent nodes in computation graph (which values were used to create this value)
    private final List<Value> previous;

    // How to propagate gradient backward which means a function that knows how to push
    // gradients from this node to its parents
    private Runnable backwardFn;

    // For debugging: "+", "*", "relu", etc.
    private final String operation;

    // For debugging: "x", "w1", etc.
    private final String label;

    /**
     * Every operation creates a new node that remembers its parents:
     * <p>
     * Value a = new Value(2.0, "a");
     * Value b = new Value(3.0, "b");
     * Value c = a.multiply(b);  // c = a * b = 6.0
     * Value d = c.add(1.0);     // d = c + 1 = 7.0
     * Value e = d.relu();       // e = relu(d) = 7.0
     * <p>
     * Computational Graph:
     * a(2.0)    b(3.0)
     * \        /
     * \      /
     * \    /
     * c = a*b (6.0)
     * |
     * d = c+1 (7.0)
     * |
     * e = relu(d) (7.0)
     * <p>
     * Each node knows:
     * - Its value (forward pass already computed)
     * - How it was created (parents + operation)
     * - How to backpropagate gradients (backward function)
     */

    public Value(double data) {
        this(data, List.of(), "", "");
    }

    public Value(double data, String label) {
        this(data, List.of(), "", label);
    }

    private Value(double data, List<Value> previous, String operation, String label) {
        this.data = data;
        this.gradient = 0.0;
        this.previous = previous;
        this.operation = operation;
        this.label = label;
        this.backwardFn = () -> {
        };
    }

    public double getData() {
        return this.data;
    }

    public void setData(double data) {
        this.data = data;
    }

    public double getGradient() {
        return this.gradient;
    }

    public void setGradient(double gradient) {
        this.gradient = gradient;
    }

    public Value add(Value other) {
        Value out = new Value(this.data + other.data, List.of(this, other), "+", "");
        out.backwardFn = () -> {
            this.gradient += out.gradient; // ∂(out)/∂(this) = 1
            other.gradient += out.gradient;  // ∂(out)/∂(other) = 1
        };
        return out;
    }

    public Value add(double c) {
        return add(new Value(c));
    }

    public Value multiply(Value other) {
        Value out = new Value(this.data * other.data, List.of(this, other), "*", "");
        out.backwardFn = () -> {
            this.gradient += other.data * out.gradient; // ∂(out)/∂(this) = other
            other.gradient += this.data * out.gradient; // ∂(out)/∂(other) = this
        };
        return out;
    }

    public Value multiply(double c) {
        return multiply(new Value(c));
    }

    public Value pow(double exponent) {
        double outData = Math.pow(this.data, exponent);
        Value out = new Value(outData, List.of(this), "pow", "");
        // ∂(out)/∂(this) = n × this^(n-1)
        out.backwardFn = () -> this.gradient += exponent * Math.pow(this.data, exponent - 1.0) * out.gradient;
        return out;
    }

    public Value exp() {
        double outData = Math.exp(this.data);
        Value out = new Value(outData, List.of(this), "exp", "");
        // ∂(out)/∂(this) = e^this = out
        // The derivative of e^x is itself!
        out.backwardFn = () -> this.gradient += outData * out.gradient;
        return out;
    }

    public Value log() {
        if (this.data <= 0.0) {
            throw new IllegalArgumentException("Data must be greater than zero. Got " + this.data);
        }
        Value out = new Value(Math.log(this.data), List.of(this), "log", "");
        // ∂(out)/∂(this) = 1/this
        out.backwardFn = () -> this.gradient += (1.0 / this.data) * out.gradient;
        return out;
    }

    public Value relu() {
        double outData = Math.max(0.0, this.data);
        Value out = new Value(outData, List.of(this), "relu", "");
        // ∂(out)/∂(this) = 1 if this > 0, else 0
        out.backwardFn = () -> this.gradient += (this.data > 0.0 ? 1.0 : 0.0) * out.gradient;
        return out;
    }

    public Value neg() {
        return this.multiply(-1.0);
    }

    public Value subtract(double c) {
        return this.add(-c);
    }

    public Value divide(Value other) {
        return this.multiply(other.pow(-1.0));
    }

    /**
     * Squashes values to [-1, 1], non-linear (enables complex learning patterns),
     * and smooth gradients (better than ReLU for small networks)
     */
    public Value tanh() {
        double outData = Math.tanh(this.data);
        Value out = new Value(outData, List.of(this), "tanh", "");
        out.backwardFn = () -> {
            // Derivative of tanh(x) = 1 - tanh²(x)
            this.gradient += (1.0 - outData * outData) * out.gradient;
        };
        return out;
    }

    /**
     * Implements multivariate chain rule
     * <p>
     * 1. Builds topological order
     * 2. Initializes output gradient
     * 3. Calculates reverse topological order
     */
    public void backward() {
        List<Value> topology = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        // Must compute parent gradients before child gradients in reverse
        buildTopology(this, visited, topology);

        // ∂L/∂L = 1 (output w.r.t.itself)
        this.gradient = 1.0;
        // Go from output → inputs, calling each node's backward function.
        for (int i = topology.size() - 1; i >= 0; i--) {
            topology.get(i).backwardFn.run();
        }
    }

    /**
     * Orders nodes so that parents come before children
     */
    private void buildTopology(Value value, Set<Value> visited, List<Value> topology) {
        if (visited.contains(value)) {
            return;
        }
        visited.add(value);
        for (Value v : value.previous) {
            buildTopology(v, visited, topology);
        }
        topology.add(value);
    }

    @Override
    public String toString() {
        return "Value(data=" + data + ", grad=" + gradient + ", operation=" + operation + ", label=" + label + ")";
    }
}
