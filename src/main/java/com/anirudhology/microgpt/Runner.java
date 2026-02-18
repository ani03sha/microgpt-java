package com.anirudhology.microgpt;

import com.anirudhology.microgpt.data.NGramDatasetBuilder;
import com.anirudhology.microgpt.data.TextCorpus;
import com.anirudhology.microgpt.data.TrainingExample;
import com.anirudhology.microgpt.model.BaselineBigramModel;
import com.anirudhology.microgpt.model.GPTLanguageModel;
import com.anirudhology.microgpt.model.MLPLanguageModel;
import com.anirudhology.microgpt.model.NeuralBigramAutogradModel;
import com.anirudhology.microgpt.model.NeuralBigramModel;
import com.anirudhology.microgpt.optimizer.AdamOptimizer;
import com.anirudhology.microgpt.tokenizer.CharacterTokenizer;

import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Runner {

    static void main() {
        TextCorpus textCorpus = new TextCorpus();
        final List<String> docs = textCorpus.readCorpus("input.txt");

        CharacterTokenizer tokenizer = new CharacterTokenizer();
        tokenizer.buildVocabulary(docs);

        int split = (int) (docs.size() * 0.9);
        final List<String> trainDocs = docs.subList(0, split);
        final List<String> validationDocs = docs.subList(split, docs.size());

        /*runBaselineBigram(tokenizer, docs);
        runNeuralBigram(tokenizer, trainDocs, validationDocs);
        runNeuralAutogradBigram(tokenizer, trainDocs, validationDocs);
        runMLPLanguageModel(tokenizer, docs);*/
        runGPTLanguageModel(tokenizer, docs);
    }

    private static void runNeuralAutogradBigram(CharacterTokenizer tokenizer, List<String> trainDocs, List<String> validationDocs) {
        NeuralBigramAutogradModel neuralBigramAutogradModel = new NeuralBigramAutogradModel(
                tokenizer.getVocabularySize(),
                tokenizer.getBOSId(),
                42L
        );

        double learningRate = 0.5;
        long epochs = 20;

        for (int e = 0; e < epochs; e++) {
            double trainingNll = neuralBigramAutogradModel.train(trainDocs, tokenizer, learningRate, 1000L + e);
            double validationNll = neuralBigramAutogradModel.averageNegativeLogLikelihood(validationDocs, tokenizer);
            System.out.printf("Epoch %2d | TrainingNLL %.4f | ValidationNLL %.4f | Learning Rate %.4f%n",
                    e, trainingNll, validationNll, learningRate);
            learningRate *= 0.98;
        }

        System.out.println("\n--- samples ---");
        for (int i = 0; i < 20; i++) {
            System.out.printf("sample %2d: %s%n", i + 1, neuralBigramAutogradModel.sample(tokenizer, 16, 0.9));
        }
    }

    private static void runNeuralBigram(CharacterTokenizer tokenizer, List<String> trainDocs, List<String> validationDocs) {
        NeuralBigramModel neuralBigramModel = new NeuralBigramModel(tokenizer.getVocabularySize(), tokenizer.getBOSId(), 42L);

        double learningRate = 0.5;
        int epochs = 30;

        for (int e = 1; e <= epochs; e++) {
            double trainingNll = neuralBigramModel.trainEpoch(trainDocs, tokenizer, learningRate, 1000L + e);
            double validationNLL = neuralBigramModel.averageNegativeLogLikelihood(validationDocs, tokenizer);
            System.out.printf("Epoch %2d | TrainingNLL %.4f | ValidationNLL %.4f | Learning Rate %.4f%n", e, trainingNll, validationNLL, learningRate);
            learningRate *= 0.98;
        }

        System.out.println("\n--- Samples ---");
        for (int i = 0; i < 20; i++) {
            System.out.printf("sample %2d: %s%n", i + 1, neuralBigramModel.sample(tokenizer, 16, 0.9));
        }
    }

    private static void runBaselineBigram(CharacterTokenizer tokenizer, List<String> docs) {
        BaselineBigramModel model = new BaselineBigramModel(tokenizer.getVocabularySize(), tokenizer.getBOSId(), 42L);
        model.fit(docs, tokenizer, 1.0); // alpha = 0.1 - Laplace smoothing

        double nll = model.averageNegativeLogLikelihood(docs, tokenizer);
        System.out.printf("Bigram avg NLL: %.4f%n", nll);

        System.out.println("\n--- Bigram samples ---");
        for (int i = 0; i < 20; i++) {
            System.out.printf("sample %2d: %s%n", i + 1, model.sample(tokenizer, 16));
        }
    }

    private static void runMLPLanguageModel(CharacterTokenizer tokenizer, List<String> documents) {
        // Build dataset
        final int blockSize = 3; // Context size
        final List<TrainingExample> examples = NGramDatasetBuilder.build(documents, tokenizer, blockSize, true);

        // Create model
        final MLPLanguageModel model = new MLPLanguageModel(tokenizer.getVocabularySize(), blockSize, 10, 100, 42L);

        // Training Loop
        int epochs = 10;
        double learningRate = 0.01;

        boolean shouldUsePositionalEncoding = true;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;

            // Shuffle examples in each epoch
            Collections.shuffle(examples, new Random(42L + epoch));

            for (int i = 0; i < examples.size(); i++) {
                double loss = model.trainStep(examples.get(i), learningRate, shouldUsePositionalEncoding);
                totalLoss += loss;

                // Print progress every 1000 steps
                if ((i + 1) % 1000 == 0) {
                    double avgLoss = totalLoss / (i + 1);
                    System.out.printf("Epoch %d, Step %d/%d, Avg Loss: %.4f%n",
                            epoch + 1, i + 1, examples.size(), avgLoss);
                }
            }

            double averageLoss = totalLoss / examples.size();
            System.out.printf("Epoch %d complete: Average Loss = %.4f%n", epoch + 1, averageLoss);

            // Decay learning rate
            learningRate *= 0.9;

            // Generate samples
            System.out.println("\n--- Samples ---");
            for (int i = 0; i < 5; i++) {
                final String sample = model.generate(tokenizer, 20, 1.0);
                System.out.printf("Sample %d: %s%n", i + 1, sample);
                System.out.println();
            }
        }
    }

    private static void runGPTLanguageModel(CharacterTokenizer tokenizer, List<String> documents) {
        // Hyperparameters
        final int blockSize = 16;
        final int embeddingDimension = 16;
        final int headDimension = 16;
        final int numberOfLayers = 1;
        int numHeads = 4;

        // Build dataset
        final List<TrainingExample> examples = NGramDatasetBuilder.build(documents, tokenizer, blockSize, true);
        System.out.printf("Total training examples: %d%n", examples.size());

        // Create Model
        final GPTLanguageModel model = new GPTLanguageModel(
                tokenizer.getVocabularySize(),
                blockSize,
                embeddingDimension,
                /*headDimension,*/
                numHeads,
                numberOfLayers,
                42L
        );

        // Create Adam Optimizer
        final AdamOptimizer optimizer = new AdamOptimizer(model.parameters());

        int numberOfSteps = 1000;
        double initialLearningRate = 0.01;
        double runningLoss = 0.0;
        double smoothing = 0.95; // Exponential moving average

        for (int step = 0; step < numberOfSteps; step++) {
            // Linear LR decay: starts at 0.01, reaches 0 at final step
            double learningRate = initialLearningRate * (1.0 - (double) step / numberOfSteps);

            // Zero gradients -> forward -> backward -> Adam step
            optimizer.zeroGradient(model.parameters());
            double loss = model.trainStepOptimized(examples.get(step % examples.size()));
            optimizer.step(model.parameters(), learningRate);

            // Smooth the loss
            runningLoss = smoothing * runningLoss + (1 - smoothing) * loss;

            if ((step + 1) % 100 == 0) {
                System.out.printf("Step %4d / %4d | Loss: %.4f | LR: %.6f%n", step + 1, numberOfSteps, loss, learningRate);
            }
        }

        System.out.println("\n--- Samples ---");
        for (int i = 0; i < 10; i++) {
            System.out.printf("Sample %d: %s%n", i + 1, model.generate(tokenizer, 20, 0.5));
        }

        /*// Training
        int epochs = 10;
        double learningRate = 0.01;

        for (int epoch = 0; epoch < epochs; epoch++) {
            long start = System.currentTimeMillis();
            double totalLoss = 0.0;

            Collections.shuffle(examples, new Random(42L + epoch));

            for (int i = 0; i < examples.size(); i++) {
                totalLoss += model.trainStep(examples.get(i), learningRate);

                if ((i + 1) % 1000 == 0) {
                    System.out.printf("Epoch %d, Step %d/%d, AvgLoss: %.4f%n",
                            epoch + 1, i + 1,
                            examples.size(),
                            totalLoss / (i + 1));

                }
            }

            long elapsed = (System.currentTimeMillis() - start) / 1000;
            System.out.printf("Epoch %d complete: Loss=%.4f (%ds)%n",
                    epoch + 1,
                    totalLoss / examples.size(),
                    elapsed
            );

            learningRate *= 0.9;

            System.out.println("\n--- Samples ---");
            for (int i = 0; i < 5; i++) {
                System.out.printf("Sample %d: %s%n", i + 1, model.generate(tokenizer, 20, 1.0));
            }
            System.out.println();
        }*/
    }
}
