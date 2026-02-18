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

/**
 * Runs each model in sequence, from simplest to most powerful.
 * <p>
 * Step 1: Statistical Bigram       - pure counting, no neural network
 * Step 2: Neural Bigram            - same task, learned weights, manual gradients
 * Step 3: Neural Bigram (Autograd) - same model, automatic differentiation via Value
 * Step 4: MLP Language Model       - wider context window, embeddings, hidden layer
 * Step 5: GPT Transformer          - multi-head attention, transformer blocks, Adam
 */
public class Runner {

    static void main() {
        TextCorpus textCorpus = new TextCorpus();
        final List<String> docs = textCorpus.readCorpus("input.txt");

        CharacterTokenizer tokenizer = new CharacterTokenizer();
        tokenizer.buildVocabulary(docs);

        int split = (int) (docs.size() * 0.9);
        final List<String> trainDocs = docs.subList(0, split);
        final List<String> validationDocs = docs.subList(split, docs.size());

        printStep(1, "Statistical Bigram",
                "Count character pair frequencies. No learning, no gradients. Pure statistics.");
        runBaselineBigram(tokenizer, docs);

        printStep(2, "Neural Bigram (manual gradients)",
                "Same bigram task but replace the count table with a learned weight matrix.\n" +
                "  Gradients computed by hand: dL/dlogit = p - 1_correct.");
        runNeuralBigram(tokenizer, trainDocs, validationDocs);

        printStep(3, "Neural Bigram (autograd)",
                "Same model, but gradients are now computed automatically by the Value\n" +
                "  computation graph. No hand-derived gradient formulas needed.");
        runNeuralAutogradBigram(tokenizer, trainDocs, validationDocs);

        printStep(4, "MLP Language Model",
                "Wider context window (3 chars). Token + positional embeddings flattened\n" +
                "  into a hidden layer with tanh activation.");
        runMLPLanguageModel(tokenizer, docs);

        printStep(5, "GPT Transformer (single-head attention)",
                "CausalSelfAttention: one attention head over the full embedding dimension.\n" +
                "  Simpler, fewer parameters, but limited in what patterns it can capture.");
        runGPTLanguageModel(tokenizer, docs, false);

        printStep(6, "GPT Transformer (multi-head attention)",
                "MultiHeadCausalSelfAttention: splits embedding into " + 4 + " heads, each\n" +
                "  attending to a different subspace. More expressive, same parameter count.");
        runGPTLanguageModel(tokenizer, docs, true);
    }

    // -------------------------------------------------------------------------
    // Step 1: Statistical Bigram
    // -------------------------------------------------------------------------

    private static void runBaselineBigram(CharacterTokenizer tokenizer, List<String> docs) {
        BaselineBigramModel model = new BaselineBigramModel(tokenizer.getVocabularySize(), tokenizer.getBOSId(), 42L);
        model.fit(docs, tokenizer, 1.0);

        double nll = model.averageNegativeLogLikelihood(docs, tokenizer);
        System.out.printf("Avg NLL: %.4f%n", nll);

        System.out.println("\n--- Samples ---");
        for (int i = 0; i < 10; i++) {
            System.out.printf("Sample %2d: %s%n", i + 1, model.sample(tokenizer, 16));
        }
    }

    // -------------------------------------------------------------------------
    // Step 2: Neural Bigram with manual gradients
    // -------------------------------------------------------------------------

    private static void runNeuralBigram(CharacterTokenizer tokenizer, List<String> trainDocs, List<String> validationDocs) {
        NeuralBigramModel model = new NeuralBigramModel(tokenizer.getVocabularySize(), tokenizer.getBOSId(), 42L);

        double learningRate = 0.5;
        int epochs = 30;

        for (int e = 1; e <= epochs; e++) {
            double trainNll = model.trainEpoch(trainDocs, tokenizer, learningRate, 1000L + e);
            double valNll = model.averageNegativeLogLikelihood(validationDocs, tokenizer);
            System.out.printf("Epoch %2d | Train NLL: %.4f | Val NLL: %.4f | LR: %.4f%n",
                    e, trainNll, valNll, learningRate);
            learningRate *= 0.98;
        }

        System.out.println("\n--- Samples ---");
        for (int i = 0; i < 10; i++) {
            System.out.printf("Sample %2d: %s%n", i + 1, model.sample(tokenizer, 16, 0.9));
        }
    }

    // -------------------------------------------------------------------------
    // Step 3: Neural Bigram with autograd (Value class)
    // -------------------------------------------------------------------------

    private static void runNeuralAutogradBigram(CharacterTokenizer tokenizer, List<String> trainDocs, List<String> validationDocs) {
        NeuralBigramAutogradModel model = new NeuralBigramAutogradModel(
                tokenizer.getVocabularySize(),
                tokenizer.getBOSId(),
                42L
        );

        double learningRate = 0.5;
        int epochs = 20;

        for (int e = 0; e < epochs; e++) {
            double trainNll = model.train(trainDocs, tokenizer, learningRate, 1000L + e);
            double valNll = model.averageNegativeLogLikelihood(validationDocs, tokenizer);
            System.out.printf("Epoch %2d | Train NLL: %.4f | Val NLL: %.4f | LR: %.4f%n",
                    e, trainNll, valNll, learningRate);
            learningRate *= 0.98;
        }

        System.out.println("\n--- Samples ---");
        for (int i = 0; i < 10; i++) {
            System.out.printf("Sample %2d: %s%n", i + 1, model.sample(tokenizer, 16, 0.9));
        }
    }

    // -------------------------------------------------------------------------
    // Step 4: MLP Language Model
    // -------------------------------------------------------------------------

    private static void runMLPLanguageModel(CharacterTokenizer tokenizer, List<String> documents) {
        final int blockSize = 3;
        final List<TrainingExample> examples = NGramDatasetBuilder.build(documents, tokenizer, blockSize, true);

        final MLPLanguageModel model = new MLPLanguageModel(
                tokenizer.getVocabularySize(), blockSize, 10, 100, 42L);

        int epochs = 10;
        double learningRate = 0.01;

        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0.0;
            Collections.shuffle(examples, new Random(42L + epoch));

            for (int i = 0; i < examples.size(); i++) {
                totalLoss += model.trainStep(examples.get(i), learningRate, true);
                if ((i + 1) % 1000 == 0) {
                    System.out.printf("Epoch %d, Step %d/%d, Avg Loss: %.4f%n",
                            epoch + 1, i + 1, examples.size(), totalLoss / (i + 1));
                }
            }

            System.out.printf("Epoch %d complete: Avg Loss = %.4f%n",
                    epoch + 1, totalLoss / examples.size());
            learningRate *= 0.9;
        }

        System.out.println("\n--- Samples ---");
        for (int i = 0; i < 10; i++) {
            System.out.printf("Sample %2d: %s%n", i + 1, model.generate(tokenizer, 20, 1.0));
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: GPT Transformer
    // -------------------------------------------------------------------------

    private static void runGPTLanguageModel(CharacterTokenizer tokenizer, List<String> documents, boolean useMultiHead) {
        final int blockSize = 16;
        final int embeddingDimension = 16;
        final int numHeads = 4;
        final int numberOfLayers = 1;

        final List<TrainingExample> examples = NGramDatasetBuilder.build(documents, tokenizer, blockSize, true);
        System.out.printf("Total training examples: %d%n", examples.size());

        final GPTLanguageModel model = new GPTLanguageModel(
                tokenizer.getVocabularySize(),
                blockSize,
                embeddingDimension,
                numHeads,
                numberOfLayers,
                useMultiHead,
                42L
        );

        final AdamOptimizer optimizer = new AdamOptimizer(model.parameters());

        int numberOfSteps = 1000;
        double initialLearningRate = 0.01;

        for (int step = 0; step < numberOfSteps; step++) {
            double learningRate = initialLearningRate * (1.0 - (double) step / numberOfSteps);

            optimizer.zeroGradient(model.parameters());
            double loss = model.trainStep(examples.get(step % examples.size()));
            optimizer.step(model.parameters(), learningRate);

            if ((step + 1) % 100 == 0) {
                System.out.printf("Step %4d / %4d | Loss: %.4f | LR: %.6f%n",
                        step + 1, numberOfSteps, loss, learningRate);
            }
        }

        System.out.println("\n--- Samples ---");
        for (int i = 0; i < 10; i++) {
            System.out.printf("Sample %2d: %s%n", i + 1, model.generate(tokenizer, 20, 0.5));
        }
    }

    // -------------------------------------------------------------------------
    // Utility
    // -------------------------------------------------------------------------

    private static void printStep(int step, String title, String description) {
        System.out.println("\n" + "=".repeat(70));
        System.out.printf("  Step %d: %s%n", step, title);
        System.out.println("=".repeat(70));
        System.out.println("  " + description);
        System.out.println("-".repeat(70));
    }
}
