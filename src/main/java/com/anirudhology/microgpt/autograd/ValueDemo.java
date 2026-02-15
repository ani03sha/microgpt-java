package com.anirudhology.microgpt.autograd;

public class ValueDemo {

    static void main() {
        testPolynomial();
        testRelu();
        testNumericalGradient();
    }

    private static void testPolynomial() {
        Value x = new Value(3.0, "x");
        Value y = x.multiply(x).add(x.multiply(2.0)).add(1.0); // x^2 + 2x + 1
        y.backward();

        System.out.println("poly y=" + y.getData() + " (expected 16.0)");
        System.out.println("poly dy/dx=" + x.getGradient() + " (expected 8.0)");
    }

    private static void testRelu() {
        Value x = new Value(-2.0, "x");
        Value y = x.relu();
        y.backward();

        System.out.println("relu y=" + y.getData() + " (expected 0.0)");
        System.out.println("relu dy/dx=" + x.getGradient() + " (expected 0.0)");
    }

    private static void testNumericalGradient() {
        double x0 = 1.2;
        Value x = new Value(x0, "x");
        Value y = x.multiply(x).add(x.exp()); // f(x)=x^2 + e^x
        y.backward();

        double analytic = x.getGradient();

        double h = 1e-6;
        double fPlus = f(x0 + h);
        double fMinus = f(x0 - h);
        double numeric = (fPlus - fMinus) / (2.0 * h);

        System.out.println("analytic grad = " + analytic);
        System.out.println("numeric  grad = " + numeric);
        System.out.println("abs diff      = " + Math.abs(analytic - numeric));
    }

    private static double f(double x) {
        return x * x + Math.exp(x);
    }
}
