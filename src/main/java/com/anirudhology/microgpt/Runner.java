package com.anirudhology.microgpt;

import com.anirudhology.microgpt.tokenizer.CharacterTokenizer;
import com.anirudhology.microgpt.tokenizer.TextCorpus;
import com.anirudhology.microgpt.training.BaselineBigramModel;
import com.anirudhology.microgpt.training.NeuralBigramAutogradModel;
import com.anirudhology.microgpt.training.NeuralBigramModel;

import java.util.List;

public class Runner {

    static void main() {
        TextCorpus textCorpus = new TextCorpus();
        final List<String> docs = textCorpus.readCorpus("input.txt");

        CharacterTokenizer tokenizer = new CharacterTokenizer();
        tokenizer.buildVocabulary(docs);

        int split = (int) (docs.size() * 0.9);
        final List<String> trainDocs = docs.subList(0, split);
        final List<String> validationDocs = docs.subList(split, docs.size());

        runBaselineBigram(tokenizer, docs);
        runNeuralBigram(tokenizer, trainDocs, validationDocs);
        runNeuralAutogradBigram(tokenizer, trainDocs, validationDocs);
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
}
