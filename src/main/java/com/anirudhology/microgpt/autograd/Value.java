package com.anirudhology.microgpt.autograd;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class Value {

    // Stores forward value of this node (the actual number)
    private double data;
    // Stores derivative of final output w.r.t this node (dL/d(this)
    private double grad;

    // Parents in computation graph (inputs used to make this node)
    private final List<Value> previous;
    // Local backpropagation rule for this node.
    // When run it pushes gradient from this node to its parents.
    private Runnable backwardFn;

    // To debug metadata
    private final String operation;
    private final String label;

    // Note: Every operation creates a new node that remembers parents and how to backprop.

    public Value(double data) {
        this(data, List.of(), "", "");
    }

    public Value(double data, String label) {
        this(data, List.of(), "", label);
    }

    private Value(double data, List<Value> previous, String operation, String label) {
        this.data = data;
        this.grad = 0.0;
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

    public double getGrad() {
        return this.grad;
    }

    public void setGrad(double grad) {
        this.grad = grad;
    }

    public Value add(Value other) {
        Value out = new Value(this.data + other.data, List.of(this, other), "+", "");
        out.backwardFn = () -> {
            this.grad += 1.0 * out.grad;
            other.grad += 1.0 * out.grad;
        };
        return out;
    }

    public Value add(double c) {
        return add(new Value(c));
    }

    public Value multiply(Value other) {
        Value out = new Value(this.data * other.data, List.of(this, other), "*", "");
        out.backwardFn = () -> {
            this.grad += other.data * out.grad;
            other.grad += this.data * out.grad;
        };
        return out;
    }

    public Value multiply(double c) {
        return multiply(new Value(c));
    }

    public Value pow(double exponent) {
        double outData = Math.pow(this.data, exponent);
        Value out = new Value(outData, List.of(this), "pow", "");
        out.backwardFn = () -> {
            this.grad += exponent * Math.pow(this.data, exponent - 1.0) * out.grad;
        };
        return out;
    }

    public Value expectation() {
        double outData = Math.exp(this.data);
        Value out = new Value(outData, List.of(this), "exp", "");
        out.backwardFn = () -> {
            this.grad += outData * out.grad;
        };
        return out;
    }

    public Value log() {
        if (this.data <= 0.0) {
            throw new IllegalArgumentException("Data must be greater than zero. Got " + this.data);
        }
        Value out = new Value(Math.log(this.data), List.of(this), "log", "");
        out.backwardFn = () -> {
            this.grad += (1.0 / this.data) * out.grad;
        };
        return out;
    }

    public Value relu() {
        double outData = Math.max(0.0, this.data);
        Value out = new Value(outData, List.of(this), "relu", "");
        out.backwardFn = () -> {
            this.grad += (this.data > 0.0 ? 1.0 : 0.0) * out.grad;
        };
        return out;
    }

    public Value neg() {
        return this.multiply(-1.0);
    }

    public Value subtract(Value other) {
        return this.add(other.neg());
    }

    public Value subtract(double c) {
        return this.add(-c);
    }

    public Value divide(Value other) {
        return this.multiply(other.pow(-1.0));
    }

    public Value divide(double c) {
        return this.multiply(1.0 / c);
    }

    public void backward() {
        List<Value> topology = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        buildTopology(this, visited, topology);

        this.grad = 1.0;
        for (int i = topology.size() - 1; i >= 0; i--) {
            topology.get(i).backwardFn.run();
        }
    }

    public void zeroGradGraph() {
        List<Value> topology = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        buildTopology(this, visited, topology);
        for (Value v : topology) {
            v.grad = 0.0;
        }
    }

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
        return "Value(data=" + data + ", grad=" + grad + ", operation=" + operation + ", label=" + label + ")";
    }
}
