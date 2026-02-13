package com.anirudhology.microgpt;

import com.anirudhology.microgpt.tokenizer.CharTokenizer;
import com.anirudhology.microgpt.tokenizer.TextCorpus;
import com.anirudhology.microgpt.training.BigramModel;
import com.anirudhology.microgpt.training.NeuralBigramModel;

import java.util.List;

public class Runner {

    static void main() {
        TextCorpus textCorpus = new TextCorpus();
        final List<String> docs = textCorpus.readCorpus("input.txt");

        CharTokenizer tokenizer = new CharTokenizer();
        tokenizer.buildVocabulary(docs);

        // ------------ Baseline Bigram Model ------------

        BigramModel model = new BigramModel(tokenizer.getVocabularySize(), tokenizer.getBOSId(), 42L);
        model.fit(docs, tokenizer, 1.0); // alpha = 0.1 - Laplace smoothing

        double nll = model.averageNegativeLogLikelihood(docs, tokenizer);
        System.out.printf("Bigram avg NLL: %.4f%n", nll);

        System.out.println("\n--- Bigram samples ---");
        for (int i = 0; i < 20; i++) {
            System.out.printf("sample %2d: %s%n", i + 1, model.sample(tokenizer, 16));
        }

        // ------------- Neural Bigram Model --------------
        int split = (int) (docs.size() * 0.9);
        final List<String> trainDocs = docs.subList(0, split);
        final List<String> validationDocs = docs.subList(split, docs.size());

        NeuralBigramModel neuralBigramModel = new NeuralBigramModel(tokenizer.getVocabularySize(), tokenizer.getBOSId(), 42L);

        double learningRate = 0.5;
        int epochs = 30;

        for (int e = 1; e <= epochs; e++) {
            double trainingNll = neuralBigramModel.trainEpoch(trainDocs, tokenizer, learningRate, 1000L + e);
            double validationNLL = neuralBigramModel.averageNegativeLogLikelihood(validationDocs, tokenizer);
            System.out.printf("Epoch %2d | TrainingNLL %.4f | ValidationNLL %.4f | lr %.4f%n", e, trainingNll, validationNLL, learningRate);
            learningRate *= 0.98;
        }

        System.out.println("\n--- Samples ---");
        for (int i = 0; i < 20; i++) {
            System.out.printf("sample %2d: %s%n", i + 1, neuralBigramModel.sample(tokenizer, 16, 0.9));
        }
    }
}
